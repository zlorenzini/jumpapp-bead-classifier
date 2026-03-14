"""
GET /bundles

Reports the model backbone and current state of the embedding-based
classifier.  Mirrors the shape of JumpSmartsRuntime /bundles without
pretending to have gradient-trained model bundles.

This is the "identity card" for the AI system — safe to call frequently,
no heavy compute, suitable for UI dashboards and diagnostics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

# jumpapp.json sits two directories above this file (server/routes/ → root)
_JUMPAPP_JSON = Path(__file__).resolve().parent.parent.parent / "jumpapp.json"


def _jumpapp_meta() -> dict:
    """Read name + version from jumpapp.json (cached after first call)."""
    try:
        data = json.loads(_JUMPAPP_JSON.read_text())
        return {"name": data.get("name", ""), "version": data.get("version", "")}
    except Exception:
        return {"name": "jumpapp-bead-classifier", "version": "unknown"}


def _torchvision_version() -> str:
    """Return a human-readable torchvision version string, or a fallback."""
    try:
        import torchvision
        return f"torchvision-{torchvision.__version__}"
    except Exception:
        return "torchvision-unknown"


@router.get("")
async def get_bundles():
    """Return backbone descriptor and live embedding-store statistics."""
    import db                        # noqa: PLC0415
    from config import get_settings  # noqa: PLC0415

    cfg = get_settings()

    try:
        stats = await db.get_bundle_stats(cfg.mongodb_collection)
    except Exception as exc:
        logger.exception("Failed to query bundle stats from MongoDB")
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "reason": "database_unavailable"},
        )

    return {
        "status": "ok",
        "bundles": [
            {
                "id":            "backbone",
                "type":          "embedder",
                "model":         "resnet50",
                "embedding_dim": 2048,
                "version":       _torchvision_version(),
                "trainable":     False,
            },
            {
                "id":           "bead-embeddings",
                "type":         "embedding-store",
                "classes":      stats["classes"],
                "embeddings":   stats["embeddings"],
                "last_updated": stats["last_updated"],
                "storage":      "mongodb",
            },
        ],
        "jumpapp": _jumpapp_meta(),
    }
