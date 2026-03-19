"""
Bead inference — NPU → PyTorch CPU → cosine-similarity fallback.

Algorithm
---------
1. If models_dir/bundle_id/model.dxnn exists, run it on the DX-M1 NPU via
   dx_engine to get per-class scores for every known SKU.
2. If model.dxnn is absent but model.pth exists, run the MobileNetV2
   classifier on CPU via PyTorch.
3. For beads not covered by the trained model (added after the last
   training run), fall back to ResNet-50 cosine similarity.
4. Merge scores, sort descending, apply threshold, return top-K.

Both the NPU engine and the PyTorch model are cached in memory and
reloaded automatically when their files change on disk.  trainer.py calls
evict_model_cache() after each completed training run so changes are
picked up immediately.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path
from typing import Any

from embedder import cosine_similarity, embed_image_bytes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caches (shared lock for both NPU and PyTorch entries)
# ---------------------------------------------------------------------------

_classifier_cache: dict[str, dict[str, Any]] = {}  # cache_key → entry
_cache_lock = threading.Lock()


def evict_model_cache(model_dir_str: str) -> None:
    """
    Remove cached classifier entries for a bundle directory.  Called by
    trainer.py after a completed training run so the next inference
    reloads from disk (covers both NPU .dxnn and PyTorch .pth entries).
    """
    base = Path(model_dir_str)
    keys_to_remove = [str(base / "model.dxnn"), str(base / "model.pth")]
    with _cache_lock:
        removed = [k for k in keys_to_remove if _classifier_cache.pop(k, None)]
    if removed:
        logger.info("Evicted classifier cache for '%s' (%s)", model_dir_str, removed)


# ---------------------------------------------------------------------------
# NPU classifier (DX-M1 via dx_engine)
# ---------------------------------------------------------------------------

def _load_npu_classifier(
    models_dir: Path,
    bundle_id: str = "current",
) -> dict[str, Any] | None:
    """
    Load a compiled .dxnn model for DX-M1 NPU inference.

    Returns a cache entry dict, or None if no .dxnn model exists or
    dx_engine is unavailable.  Uses mtime-based cache invalidation.
    """
    dxnn_path = models_dir / bundle_id / "model.dxnn"
    meta_path = models_dir / bundle_id / "metadata.json"

    if not dxnn_path.exists() or not meta_path.exists():
        return None

    cache_key = str(dxnn_path)
    mtime     = dxnn_path.stat().st_mtime

    with _cache_lock:
        entry = _classifier_cache.get(cache_key)
        if entry is not None and entry.get("mtime") == mtime:
            return entry

    try:
        from dx_engine import InferenceEngine  # noqa: PLC0415
    except ImportError:
        logger.warning("dx_engine not importable — NPU inference unavailable")
        return None

    try:
        meta   = json.loads(meta_path.read_text())
        labels = meta.get("labels") or meta.get("classes") or []
        if not labels:
            logger.warning("NPU model metadata has no class labels at '%s'", meta_path)
            return None

        engine = InferenceEngine(str(dxnn_path))
        input_info = engine.get_input_tensors_info()[0]  # {name, shape, dtype}

        new_entry: dict[str, Any] = {
            "engine":     engine,
            "labels":     labels,
            "input_info": input_info,
            "mtime":      mtime,
            "kind":       "npu",
        }
        with _cache_lock:
            _classifier_cache[cache_key] = new_entry

        logger.info(
            "Loaded NPU classifier: %d classes, bundle '%s', input=%s %s",
            len(labels), bundle_id, input_info["shape"], input_info["dtype"],
        )
        return new_entry

    except Exception:
        logger.exception("Failed to load NPU classifier from '%s'", dxnn_path)
        return None


def _run_npu_classifier(
    entry: dict[str, Any],
    image_bytes: bytes,
) -> dict[str, float]:
    """
    Run DX-M1 NPU forward pass.  Returns {class_name: probability}.

    Handles both NHWC uint8 (quantized) and NCHW/NHWC float32 layouts,
    reading the expected format from the engine's input tensor info.
    """
    import numpy as np        # noqa: PLC0415
    from PIL import Image     # noqa: PLC0415

    engine     = entry["engine"]
    labels     = entry["labels"]
    input_info = entry["input_info"]
    shape      = input_info["shape"]   # e.g. [1, 224, 224, 3] or [1, 3, 224, 224]
    dtype      = input_info["dtype"]   # e.g. np.uint8 or np.float32

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Determine spatial dims from shape (NHWC or NCHW)
    if len(shape) == 4 and shape[3] == 3:        # NHWC
        h, w = shape[1], shape[2]
        nhwc = True
    else:                                          # NCHW
        h, w = shape[2], shape[3]
        nhwc = False

    img = img.resize((w, h))
    arr = np.array(img, dtype=np.uint8)  # [H, W, 3] uint8

    if dtype == np.uint8:
        # Quantized model: raw pixel values, NHWC
        inp = arr[np.newaxis].astype(np.uint8)          # [1, H, W, 3]
    else:
        # Float model: apply ImageNet normalisation
        arr_f = arr.astype(np.float32) / 255.0
        mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr_f = (arr_f - mean) / std
        if nhwc:
            inp = np.ascontiguousarray(arr_f[np.newaxis])   # [1, H, W, 3]
        else:
            inp = np.ascontiguousarray(arr_f.transpose(2, 0, 1)[np.newaxis])  # [1,3,H,W]
        inp = inp.astype(np.float32)

    outputs = engine.run(inp)
    logits  = outputs[0].flatten()                      # [num_classes]

    # Softmax
    logits  = logits - logits.max()
    exp_l   = np.exp(logits)
    probs   = exp_l / exp_l.sum()

    return {cls: round(float(p), 4) for cls, p in zip(labels, probs)}


# ---------------------------------------------------------------------------
# PyTorch CPU classifier
# ---------------------------------------------------------------------------

def _load_classifier(
    models_dir: Path,
    bundle_id: str = "current",
) -> dict[str, Any] | None:
    """
    Load a trained MobileNetV2 classifier from  models_dir/bundle_id.

    Returns a cache entry dict, or None if no trained model exists.
    The loaded model is cached; cache is invalidated by mtime comparison.
    """
    model_path = models_dir / bundle_id / "model.pth"
    meta_path  = models_dir / bundle_id / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        return None

    cache_key = str(model_path)
    mtime     = model_path.stat().st_mtime

    with _cache_lock:
        entry = _classifier_cache.get(cache_key)
        if entry is not None and entry.get("mtime") == mtime:
            return entry

    # (Re-)load from disk
    import torch                           # noqa: PLC0415
    import torch.nn as nn                  # noqa: PLC0415
    from torchvision import (              # noqa: PLC0415
        models as tv_models, transforms,
    )

    try:
        meta   = json.loads(meta_path.read_text())
        labels = meta.get("labels") or meta.get("classes") or []
        if not labels:
            logger.warning(
                "Trained model metadata has no class labels at '%s'", meta_path,
            )
            return None

        img_size = int(meta.get("image_size", 224))
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = tv_models.mobilenet_v2(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(labels))
        net.load_state_dict(
            torch.load(str(model_path), map_location=device, weights_only=True),
        )
        net.to(device).eval()

        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        new_entry: dict[str, Any] = {
            "net":    net,
            "labels": labels,
            "device": device,
            "tf":     tf,
            "mtime":  mtime,
        }
        with _cache_lock:
            _classifier_cache[cache_key] = new_entry

        logger.info(
            "Loaded trained classifier: %d classes, bundle '%s', device=%s",
            len(labels), bundle_id, device,
        )
        return new_entry

    except Exception:
        logger.exception("Failed to load trained classifier from '%s'", model_path)
        return None


def _run_classifier(
    entry: dict[str, Any],
    image_bytes: bytes,
) -> dict[str, float]:
    """Run the MobileNetV2 forward pass.  Returns {class_name: probability}."""
    import torch             # noqa: PLC0415
    from PIL import Image    # noqa: PLC0415

    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = entry["tf"](img).unsqueeze(0).to(entry["device"])
    with torch.no_grad():
        probs = torch.softmax(entry["net"](tensor), dim=1)[0].cpu().tolist()
    return {cls: round(p, 4) for cls, p in zip(entry["labels"], probs)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    """Average a list of embedding vectors component-wise (and renormalise)."""
    import numpy as np   # noqa: PLC0415
    arr  = np.array(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean /= norm
    return mean.tolist()


# ---------------------------------------------------------------------------
# Public inference entry point
# ---------------------------------------------------------------------------

async def run_inference(
    image_bytes: bytes,
    bead_records: list[dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.65,
    models_dir: Path | None = None,
    bundle_id: str = "current",
) -> list[dict[str, Any]]:
    """
    Parameters
    ----------
    image_bytes:   Raw bytes of the query image.
    bead_records:  List of dicts with keys: bead_id, name, embeddings.
    top_k:         Maximum number of results to return.
    threshold:     Minimum confidence score to include a result.
    models_dir:    Root models directory; enables trained-classifier path.
    bundle_id:     Model bundle sub-directory to load (default: "current").

    Returns
    -------
    List of dicts: bead_id, name, confidence (float 0-1).
    """
    if not bead_records:
        logger.warning("No bead records in the data lake; returning empty results")
        return []

    # ── Try DX-M1 NPU classifier (model.dxnn) ───────────────────────────────
    classifier_scores: dict[str, float] = {}
    if models_dir is not None:
        try:
            npu_entry = _load_npu_classifier(models_dir, bundle_id)
            if npu_entry is not None:
                classifier_scores = _run_npu_classifier(npu_entry, image_bytes)
                logger.info(
                    "NPU classifier scored %d classes (bundle '%s')",
                    len(classifier_scores), bundle_id,
                )
        except Exception:
            logger.exception(
                "NPU classifier inference failed; trying PyTorch fallback",
            )
            classifier_scores = {}

    # ── Fall back to PyTorch CPU classifier (model.pth) ─────────────────────
    if not classifier_scores and models_dir is not None:
        try:
            entry = _load_classifier(models_dir, bundle_id)
            if entry is not None:
                classifier_scores = _run_classifier(entry, image_bytes)
                logger.info(
                    "PyTorch classifier scored %d classes (bundle '%s')",
                    len(classifier_scores), bundle_id,
                )
        except Exception:
            logger.exception(
                "PyTorch classifier inference failed; falling back to cosine similarity",
            )
            classifier_scores = {}

    # ── Cosine-similarity fallback for beads not in the trained model ────────
    uncovered = [
        b for b in bead_records if b["bead_id"] not in classifier_scores
    ]

    cosine_scores: dict[str, float] = {}
    if uncovered:
        query_vec = embed_image_bytes(image_bytes)
        for bead in uncovered:
            embeddings: list[list[float]] = bead.get("embeddings") or []
            if not embeddings:
                continue
            rep = _mean_embedding(embeddings)
            cosine_scores[bead["bead_id"]] = cosine_similarity(query_vec, rep)

    # ── Merge scores, sort, apply threshold ──────────────────────────────────
    scored: list[tuple[float, dict[str, Any]]] = []
    for bead in bead_records:
        bid = bead["bead_id"]
        if bid in classifier_scores:
            scored.append((classifier_scores[bid], bead))
        elif bid in cosine_scores:
            scored.append((cosine_scores[bid], bead))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, bead in scored[:top_k]:
        if score < threshold:
            break
        results.append({
            "bead_id":    bead["bead_id"],
            "name":       bead.get("name", ""),
            "confidence": round(score, 4),
        })

    logger.info(
        "Inference: %d match(es) above %.2f "
        "(classifier: %d classes, cosine fallback: %d beads)",
        len(results), threshold,
        len(classifier_scores), len(uncovered),
    )
    return results
