"""
POST /add-bead
POST /add-bead/upload

Ingest one bead angle (a single image + metadata) into the embedding store.

JSON mode  — Content-Type: application/json
-----------
{
    "name":       "Green Silicone 10mm Sphere",
    "sku":        "GS-10-SPH",
    "image_b64":  "<base64>",          # mutually exclusive with image_path
    "image_path": "/tmp/capture.jpg",  # file already on disk (e.g. from /capture)
    "mime_type":  "image/jpeg",        # optional, default image/jpeg
    "source":     "Temu",              # vendor / origin
    "cost":       3.49                 # price paid for this bead batch
}

Multipart mode  — POST /add-bead/upload
--------------
  name        text
  sku         text
  source      text   (default: "Unknown")
  cost        float  (default: 0.0)
  mime_type   text   (optional)
  image       file   mutually exclusive with image_path
  image_path  text   absolute path on disk (mutually exclusive with image)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Deferred singletons (avoid import-time side effects)
# ---------------------------------------------------------------------------

def _deps():
    import db                        # noqa: PLC0415
    from config import get_settings  # noqa: PLC0415
    return db, get_settings()


def _embed(raw: bytes) -> list[float]:
    from embedder import embed_image_bytes  # noqa: PLC0415
    return embed_image_bytes(raw)


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------

class AddBeadRequest(BaseModel):
    name:       str        = Field(..., description="Human-readable bead name")
    sku:        str        = Field(..., description="Unique business key, e.g. GS-10-SPH")
    image_b64:  str | None = Field(None, description="Base64-encoded image")
    image_path: str | None = Field(None, description="Absolute path to an image on disk")
    mime_type:  str        = Field("image/jpeg", description="MIME type of the image")
    source:     str        = Field("Unknown",    description="Vendor/origin, e.g. Temu, Amazon")
    cost:       float      = Field(0.0, ge=0.0,  description="Price paid for this batch")


# ---------------------------------------------------------------------------
# Shared ingestion logic
# ---------------------------------------------------------------------------

async def _ingest(
    name: str,
    sku: str,
    raw_bytes: bytes,
    image_b64: str,
    mime_type: str,
    source: str,
    cost: float,
) -> dict:
    db, cfg = _deps()

    try:
        embedding = _embed(raw_bytes)
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    try:
        bead = await db.add_bead_angle(
            collection_name=cfg.mongodb_collection,
            name=name,
            sku=sku,
            image_b64=image_b64,
            mime_type=mime_type,
            embedding=embedding,
            source=source,
            cost=cost,
        )
    except Exception as exc:
        logger.exception("DB write failed")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return {"status": "ok", "bead": bead}


# ---------------------------------------------------------------------------
# POST /add-bead  (JSON)
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
async def add_bead_json(req: AddBeadRequest):
    """Add one bead angle via JSON body."""

    if req.image_b64 and req.image_path:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "reason": "Ambiguous image source",
                "details": "Provide either image_b64 or image_path, not both",
            },
        )
    if not req.image_b64 and not req.image_path:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "reason": "Missing image",
                "details": "Provide image_b64 (base64 string) or image_path (absolute file path)",
            },
        )

    if req.image_b64:
        try:
            raw_bytes = base64.b64decode(req.image_b64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "reason": "Invalid base64",
                    "details": "image_b64 could not be decoded",
                },
            )
        image_b64 = req.image_b64
    else:
        p = Path(req.image_path)
        if not p.exists():
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "reason": "File not found",
                    "details": f"image_path does not exist: {req.image_path}",
                },
            )
        raw_bytes = p.read_bytes()
        image_b64 = base64.b64encode(raw_bytes).decode()

    return await _ingest(
        name=req.name,
        sku=req.sku,
        raw_bytes=raw_bytes,
        image_b64=image_b64,
        mime_type=req.mime_type,
        source=req.source,
        cost=req.cost,
    )


# ---------------------------------------------------------------------------
# POST /add-bead/upload  (multipart)
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def add_bead_upload(
    name:       str            = Form(...),
    sku:        str            = Form(...),
    source:     str            = Form("Unknown"),
    cost:       float          = Form(0.0),
    mime_type:  str            = Form("image/jpeg"),
    image_path: Optional[str]  = Form(None),
    image:      Optional[UploadFile] = File(None),
):
    """Add one bead angle via multipart/form-data."""

    if image and image_path:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "reason": "Ambiguous image source",
                "details": "Provide either an uploaded file or image_path, not both",
            },
        )

    if image:
        raw_bytes = await image.read()
        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "reason": "Empty file",
                    "details": "Uploaded file contains no data",
                },
            )
        image_b64 = base64.b64encode(raw_bytes).decode()
        resolved_mime = image.content_type or mime_type
    elif image_path:
        p = Path(image_path)
        if not p.exists():
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "reason": "File not found",
                    "details": f"image_path does not exist: {image_path}",
                },
            )
        raw_bytes = p.read_bytes()
        image_b64 = base64.b64encode(raw_bytes).decode()
        resolved_mime = mime_type
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "reason": "Missing image",
                "details": "Provide an uploaded image file or an image_path field",
            },
        )

    return await _ingest(
        name=name,
        sku=sku,
        raw_bytes=raw_bytes,
        image_b64=image_b64,
        mime_type=resolved_mime,
        source=source,
        cost=cost,
    )
