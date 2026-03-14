"""
POST /train               — start a MobileNetV2 training job
GET  /train               — list all training jobs
GET  /train/{job_id}      — job status, progress, and logs
POST /train/{job_id}/stop — request graceful stop

Training exports all bead images from MongoDB to a temporary ImageFolder,
fine-tunes a MobileNetV2 classifier, and saves model.pth + metadata.json
to the models directory.  Inference automatically uses the trained model
once it exists, with cosine-similarity fallback for beads added after the
last training run.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class TrainRequest(BaseModel):
    bundle_id:  str   = Field("current", description="Model bundle name (sub-directory under models/)")
    epochs:     int   = Field(10,    ge=1, le=200, description="Number of training epochs")
    img_size:   int   = Field(224,   ge=32, le=640, description="Input image size (square)")
    batch_size: int   = Field(16,    ge=1, le=256,  description="Mini-batch size")
    lr:         float = Field(1e-4,  gt=0,          description="Initial learning rate")
    reason:     Optional[str] = None  # "manual" | "scheduled" | "ui-request"


@router.post("")
async def start_train(req: TrainRequest = TrainRequest()):
    """
    Kick off a background training job.

    Returns immediately with a job id.  Poll GET /train/{id} for status.
    """
    import state    # noqa: PLC0415
    import trainer  # noqa: PLC0415
    from config import get_settings  # noqa: PLC0415

    cfg = get_settings()
    job = trainer.start_training(
        mongodb_uri=cfg.mongodb_uri,
        mongodb_db=cfg.mongodb_db,
        collection_name=cfg.mongodb_collection,
        models_dir=state.models_dir,
        bundle_id=req.bundle_id,
        epochs=req.epochs,
        img_size=req.img_size,
        batch_size=req.batch_size,
        lr=req.lr,
    )
    return {
        "id":         job["id"],
        "status":     job["status"],
        "bundle_id":  job["bundle_id"],
        "epochs":     job["epochs"],
        "created_at": job["created_at"],
    }


@router.get("")
async def list_jobs():
    """List all training jobs (most recent first)."""
    import trainer  # noqa: PLC0415
    jobs = sorted(trainer.list_jobs(), key=lambda j: j["created_at"], reverse=True)
    return [
        {
            "id":         j["id"],
            "status":     j["status"],
            "bundle_id":  j["bundle_id"],
            "epochs":     j["epochs"],
            "progress":   j["progress"],
            "created_at": j["created_at"],
        }
        for j in jobs
    ]


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Return full status, progress, logs, and result for a training job."""
    import trainer  # noqa: PLC0415
    job = trainer.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training job '{job_id}' not found.",
        )
    return {
        "id":         job["id"],
        "status":     job["status"],
        "bundle_id":  job["bundle_id"],
        "epochs":     job["epochs"],
        "progress":   job["progress"],
        "result":     job["result"],
        "error":      job["error"],
        "logs":       job["logs"],
        "created_at": job["created_at"],
    }


@router.post("/{job_id}/stop")
async def stop_job(job_id: str):
    """Request a graceful stop for a running training job."""
    import trainer  # noqa: PLC0415
    job = trainer.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training job '{job_id}' not found.",
        )
    if job["status"] not in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is already '{job['status']}' — cannot stop.",
        )
    job["stop"] = True
    return {"id": job_id, "status": "stop_requested"}
