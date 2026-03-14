"""
Bead inference — trained classifier with cosine-similarity fallback.

Algorithm
---------
1. If a trained MobileNetV2 model exists in models_dir/bundle_id/model.pth,
   run it to get a per-class softmax probability for every known SKU.
2. For beads that are *not* covered by the trained classifier (added after
   the last training run), fall back to ResNet-50 cosine similarity.
3. Merge scores, sort descending, apply threshold, return top-K.

The trained model is cached in memory and reloaded automatically whenever
the model.pth file is replaced (mtime check).  trainer.py calls
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
# Trained-classifier cache
# ---------------------------------------------------------------------------

_classifier_cache: dict[str, dict[str, Any]] = {}  # model_path_str → entry
_cache_lock = threading.Lock()


def evict_model_cache(model_dir_str: str) -> None:
    """
    Remove a cached classifier entry.  Called by trainer.py after a
    completed training run so the next inference reloads from disk.

    model_dir_str is the string representation of  models_dir / bundle_id.
    """
    cache_key = str(Path(model_dir_str) / "model.pth")
    with _cache_lock:
        removed = _classifier_cache.pop(cache_key, None)
    if removed is not None:
        logger.info("Evicted classifier cache for '%s'", model_dir_str)


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

    # ── Try trained MobileNetV2 classifier ───────────────────────────────────
    classifier_scores: dict[str, float] = {}
    if models_dir is not None:
        try:
            entry = _load_classifier(models_dir, bundle_id)
            if entry is not None:
                classifier_scores = _run_classifier(entry, image_bytes)
                logger.info(
                    "Classifier scored %d classes (bundle '%s')",
                    len(classifier_scores), bundle_id,
                )
        except Exception:
            logger.exception(
                "Classifier inference failed; falling back to cosine similarity",
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
