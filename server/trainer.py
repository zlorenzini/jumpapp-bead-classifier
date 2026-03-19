"""
Background trainer for the bead classifier.

Workflow
--------
1. Open a fresh MongoDB connection (isolated from the main event loop) and
   pull every bead record that has stored images.
2. Export images to a temporary on-disk ImageFolder directory structure
   (one sub-directory per SKU / bead_id).
3. Fine-tune a MobileNetV2 classifier on the exported images using PyTorch.
4. Save model.pth + metadata.json to  models_dir / bundle_id.
5. Evict the cached classifier in inference.py so the next request reloads
   from the freshly written checkpoint.

Job lifecycle:  queued → running → done | error | stopped
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def list_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return list(_jobs.values())


def start_training(
    *,
    mongodb_uri: str,
    mongodb_db: str,
    collection_name: str,
    models_dir: Path,
    bundle_id: str = "current",
    epochs: int = 10,
    img_size: int = 224,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> dict[str, Any]:
    """Kick off an async training job.  Returns the job dict immediately."""
    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "id":         job_id,
        "status":     "queued",
        "bundle_id":  bundle_id,
        "epochs":     epochs,
        "logs":       [],
        "progress":   None,
        "result":     None,
        "error":      None,
        "stop":       False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _jobs_lock:
        _jobs[job_id] = job

    config = dict(
        mongodb_uri=mongodb_uri,
        mongodb_db=mongodb_db,
        collection_name=collection_name,
        models_dir=str(models_dir),
        bundle_id=bundle_id,
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size,
        lr=lr,
    )
    threading.Thread(target=_run, args=(job_id, config), daemon=True).start()
    return job


# ---------------------------------------------------------------------------
# Async DB helper — runs in a fresh event loop inside the training thread
# ---------------------------------------------------------------------------

async def _fetch_bead_images(
    mongodb_uri: str,
    mongodb_db: str,
    collection_name: str,
) -> list[dict[str, Any]]:
    """Open a dedicated motor client and pull every bead with raw image data."""
    from motor.motor_asyncio import AsyncIOMotorClient  # noqa: PLC0415
    client = AsyncIOMotorClient(mongodb_uri, serverSelectionTimeoutMS=15_000)
    try:
        col = client[mongodb_db][collection_name]
        cursor = col.find(
            {},
            {
                "_id":              0,
                "sku":              1,
                "bead_id":          1,
                "name":             1,
                # v2 — angles array
                "angles.image_b64": 1,
                "angles.mime_type": 1,
                # v1 — parallel images array
                "images":           1,
            },
        )
        return await cursor.to_list(length=None)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run(job_id: str, config: dict[str, Any]) -> None:  # noqa: PLR0912, PLR0915
    import torch                              # noqa: PLC0415
    import torch.nn as nn                    # noqa: PLC0415
    import torch.optim as optim              # noqa: PLC0415
    from torchvision import (                # noqa: PLC0415
        datasets, models as tv_models, transforms,
    )
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415
    from PIL import Image                    # noqa: PLC0415

    job = _jobs[job_id]
    job["status"] = "running"

    def log(msg: str) -> None:
        job["logs"].append(msg)
        print(f"[trainer:{job_id[:8]}] {msg}", flush=True)

    tmp_dir: Path | None = None
    try:
        # ── Pull images from MongoDB ─────────────────────────────────────
        log("Loading bead records from MongoDB…")
        docs: list[dict[str, Any]] = asyncio.run(
            _fetch_bead_images(
                config["mongodb_uri"],
                config["mongodb_db"],
                config["collection_name"],
            )
        )

        if not docs:
            raise ValueError(
                "No bead records in the database. "
                "Use POST /add-bead to ingest images first."
            )

        # ── Export to ImageFolder on disk ────────────────────────────────
        tmp_dir = Path(tempfile.mkdtemp(prefix="bead_train_"))
        train_root = tmp_dir / "images"
        total_images = 0

        for doc in docs:
            sku = doc.get("sku") or doc.get("bead_id") or ""
            if not sku:
                continue

            class_dir = train_root / sku
            class_dir.mkdir(parents=True, exist_ok=True)

            # v2 path — angles[].image_b64
            angles = doc.get("angles") or []
            pairs: list[tuple[str, str]] = [
                (a["image_b64"], a.get("mime_type", "image/jpeg"))
                for a in angles
                if a.get("image_b64")
            ]
            # v1 path — images[].data
            if not pairs:
                pairs = [
                    (img["data"], img.get("mime_type", "image/jpeg"))
                    for img in (doc.get("images") or [])
                    if img.get("data")
                ]

            for i, (img_b64, _mime) in enumerate(pairs):
                try:
                    raw = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img.save(class_dir / f"img_{i}.jpg", "JPEG", quality=95)
                    total_images += 1
                except Exception as exc:
                    log(f"  Warning: could not decode image {i} for '{sku}': {exc}")

        # Remove class directories that ended up empty (all exports failed)
        for d in list(train_root.iterdir()):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

        classes_with_images = [
            d.name for d in train_root.iterdir() if d.is_dir()
        ]
        log(
            f"Exported {total_images} images across "
            f"{len(classes_with_images)} classes"
        )

        if total_images == 0:
            raise ValueError(
                "No images could be exported from MongoDB. "
                "Re-ingest beads via POST /add-bead and try again."
            )
        if len(classes_with_images) < 2:
            raise ValueError(
                f"Need at least 2 bead classes to train a classifier "
                f"(found {len(classes_with_images)}). "
                "Add more beads via POST /add-bead."
            )

        # ── Build datasets ───────────────────────────────────────────────
        img_size   = int(config["img_size"])
        epochs     = int(config["epochs"])
        batch_size = int(config["batch_size"])
        lr         = float(config["lr"])

        tf_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05,
            ),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        full_ds_train = datasets.ImageFolder(str(train_root), transform=tf_train)
        full_ds_val   = datasets.ImageFolder(str(train_root), transform=tf_val)
        labels: list[str] = full_ds_train.classes

        idx   = list(range(len(full_ds_train)))
        random.shuffle(idx)
        n_val = max(1, int(0.2 * len(idx)))
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        import os  # noqa: PLC0415
        num_workers = min(4, os.cpu_count() or 1)
        pin = torch.cuda.is_available()

        train_ld = DataLoader(
            Subset(full_ds_train, train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
        )
        val_ld = DataLoader(
            Subset(full_ds_val, val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        )
        log(
            f"Train: {len(train_idx)} images | "
            f"Val: {len(val_idx)} images | "
            f"Classes: {len(labels)}"
        )

        # ── Model setup ──────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")

        net = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(labels))
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # ── Training loop ────────────────────────────────────────────────
        best_acc: float = 0.0
        best_state: dict | None = None

        for epoch in range(1, epochs + 1):
            if job.get("stop"):
                log("Training stopped by request.")
                job["status"] = "stopped"
                return

            # Train
            net.train()
            train_loss, n_batches = 0.0, 0
            for imgs, lbls in train_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(net(imgs), lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

            avg_loss = train_loss / max(1, n_batches)

            # Validate
            net.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_ld:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = net(imgs).argmax(dim=1)
                    correct += (preds == lbls).sum().item()
                    total   += lbls.size(0)

            acc = correct / total if total > 0 else 0.0
            if acc > best_acc:
                best_acc   = acc
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

            scheduler.step()

            msg = (
                f"Epoch {epoch}/{epochs}  "
                f"loss={avg_loss:.4f}  val_acc={acc:.4f}"
            )
            log(msg)
            job["progress"] = {
                "epoch":        epoch,
                "epochs":       epochs,
                "train_loss":   round(avg_loss, 4),
                "val_accuracy": round(acc, 4),
            }

        # ── Save model ───────────────────────────────────────────────────
        if best_state:
            net.load_state_dict(best_state)

        bundle_id = config["bundle_id"]
        model_dir = Path(config["models_dir"]) / bundle_id
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(net.state_dict(), str(model_dir / "model.pth"))

        metadata: dict[str, Any] = {
            "labels":       labels,
            "classes":      labels,
            "num_classes":  len(labels),
            "image_size":   img_size,
            "architecture": "mobilenet_v2",
            "trained_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epochs":       epochs,
            "accuracy":     round(best_acc, 4),
            "device":       str(device),
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log(f"Saved model → {model_dir}/model.pth  (val_acc={best_acc:.4f})")

        # ── Export to ONNX ───────────────────────────────────────────────
        onnx_path = model_dir / "model.onnx"
        try:
            net.cpu().eval()
            dummy = torch.zeros(1, 3, img_size, img_size)
            torch.onnx.export(
                net, dummy, str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
            )
            log(f"Exported ONNX → {onnx_path}")
        except Exception as exc:
            log(f"ONNX export failed (NPU path unavailable): {exc}")

        # ── Compile to .dxnn for DX-M1 NPU ──────────────────────────────
        if onnx_path.exists():
            import subprocess as _sp  # noqa: PLC0415
            dxnn_path = model_dir / "model.dxnn"
            dx_com = (
                shutil.which("dx-com")
                or shutil.which("dxc")
            )
            if dx_com:
                try:
                    result_cp = _sp.run(
                        [dx_com, str(onnx_path), "-o", str(dxnn_path)],
                        capture_output=True, text=True, timeout=300,
                    )
                    if result_cp.returncode == 0:
                        log(f"Compiled NPU model → {dxnn_path}")
                    else:
                        log(
                            f"dx-com compilation failed (exit {result_cp.returncode}); "
                            f"NPU inference unavailable until resolved.\n"
                            f"{result_cp.stderr.strip()}"
                        )
                except Exception as exc:
                    log(f"dx-com error: {exc}")
            else:
                log(
                    "dx-com not found in PATH — ONNX model is ready for NPU compilation.\n"
                    f"To compile manually:  dx-com {onnx_path} -o {dxnn_path}\n"
                    "Once model.dxnn is present the NPU will be used automatically."
                )

        # Evict cached classifier so the next inference reloads from disk
        import inference as infer_module  # noqa: PLC0415
        infer_module.evict_model_cache(str(model_dir))

        result: dict[str, Any] = {
            "bundle_id": bundle_id,
            "model_dir": str(model_dir),
            "classes":   labels,
            "accuracy":  round(best_acc, 4),
            "epochs":    epochs,
            "images":    total_images,
        }
        job["status"] = "done"
        job["result"] = result
        log(f"Training complete — val_accuracy={best_acc:.4f}")

    except Exception as exc:
        import traceback  # noqa: PLC0415
        job["status"] = "error"
        job["error"]  = str(exc)
        job.setdefault("logs", []).append(f"Error: {exc}")
        print(
            f"[trainer:{job_id[:8]}] ERROR\n{traceback.format_exc()}",
            flush=True,
        )
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
