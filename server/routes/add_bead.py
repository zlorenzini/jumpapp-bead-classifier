"""
POST /add-bead
POST /add-bead/upload

Ingest one bead angle (a single image + metadata) into the embedding store.

JSON mode  — Content-Type: application/json
-----------
{
    "name":              "Green Silicone 10mm Hexagon",
    "sku":               "GS-10-HEX",
    "image_b64":         "<base64>",          # mutually exclusive with image_path
    "image_path":        "/tmp/capture.jpg",  # file already on disk (e.g. from /capture)
    "mime_type":         "image/jpeg",        # optional, default image/jpeg
    "source":            "Temu",              # vendor / origin
    "cost":              3.49,                # price paid for this bead batch
    # Ontology (all optional)
    "material_category": "silicone",          # glass | silicone | acrylic | metal | ceramic | wood | gemstone | organic | resin | rhinestone | fabric | other
    "material":          "silicone",          # specific sub-type, e.g. "czech glass", "sterling silver"
    "color":             "dusty rose",        # primary colour name
    "color_family":      "pink",              # red | orange | yellow | green | blue | purple | pink | brown | white | black | gray | gold | silver | multicolor
    "shape":             "hexagon",           # round | oval | tube | flat | bicone | hexagon | rondelle | faceted | drop | cube | heart | star | letter | other
    "size_mm":           10.0,                # nominal diameter in millimetres
    "finish":            "matte",             # glossy | matte | transparent | frosted | metallic | iridescent | pearlized | crackle | painted | etched | glow | uv-reactive | other
    "hole_type":         "center-drilled",   # center-drilled | large-hole | top-drilled | side-drilled
    "focal_bead":        false,               # true if this is a focal bead (character, animal, or saying)
    "focal_subject":     "Stitch",            # short name of the character or saying
    "is_3d":             true,               # true = 3-D figurine; false = flat 2-D bead ~6-8mm thick
    "focal_description": "Blue alien wearing a Santa hat, seated, painted details",
    "tags":              ["silicone", "hexagon"]
}

Multipart mode  — POST /add-bead/upload
--------------
  name               text             (required)
  sku                text             (required)
  source             text             (default: "Unknown")
  cost               float            (default: 0.0)
  mime_type          text             (optional)
  image              file             mutually exclusive with image_path
  image_path         text             absolute path on disk (mutually exclusive with image)
  material_category  text             (optional)
  material           text             (optional)
  color              text             (optional)
  color_family       text             (optional)
  shape              text             (optional)
  size_mm            float            (optional)
  finish             text             (optional)
  hole_type          text             (optional)
  focal_bead         bool             (optional)
  focal_subject      text             (optional — character/saying name)
  is_3d              bool             (optional)
  focal_description  text             (optional — detailed description)
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
    name:              str              = Field(..., description="Human-readable bead name")
    sku:               str              = Field(..., description="Unique business key, e.g. GS-10-SPH")
    image_b64:         str | None       = Field(None, description="Base64-encoded image")
    image_path:        str | None       = Field(None, description="Absolute path to an image on disk")
    mime_type:         str              = Field("image/jpeg", description="MIME type of the image")
    source:            str              = Field("Unknown",    description="Vendor/origin, e.g. Temu, Amazon")
    cost:              float            = Field(0.0, ge=0.0,  description="Price paid for this batch")
    # Ontology fields (all optional)
    material_category: str | None       = Field(None, description="glass | silicone | acrylic | metal | ceramic | wood | gemstone | organic | resin | rhinestone | fabric | other")
    material:          str | None       = Field(None, description="Specific sub-type, e.g. 'czech glass', 'sterling silver', 'rose quartz'")
    color:             str | None       = Field(None, description="Primary colour name, e.g. 'dusty rose', 'cobalt blue'")
    color_family:      str | None       = Field(None, description="Normalised colour group: red | orange | yellow | green | blue | purple | pink | brown | white | black | gray | gold | silver | multicolor")
    shape:             str | None       = Field(None, description="Bead shape: round | oval | tube | flat | bicone | hexagon | rondelle | faceted | drop | cube | heart | star | letter | other")
    size_mm:           float | None     = Field(None, ge=0.0, description="Nominal diameter in millimetres")
    finish:            str | None       = Field(None, description="Surface finish: glossy | matte | transparent | frosted | metallic | iridescent | pearlized | crackle | painted | etched | glow | uv-reactive | other")
    hole_type:         str | None       = Field(None, description="Hole construction: center-drilled | large-hole | top-drilled | side-drilled")
    focal_bead:        bool | None      = Field(None, description="True if this is a focal bead — the central decorative character, animal, or saying bead")
    focal_subject:     str | None       = Field(None, description="Short name of the character or saying depicted, e.g. 'Stitch', 'Highland Cow', 'Mama Bear'")
    is_3d:             bool | None      = Field(None, description="True = 3-D sculpted figurine of the character. False (2-D) = flat bead ~6-8 mm thick with image on both faces.")
    focal_description: str | None       = Field(None, description="Detailed description beyond subject: pose, colours, style, holiday variant, etc.")
    tags:              list[str] | None = Field(None, description="Arbitrary extra tags for filtering")


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
    *,
    material_category: str | None = None,
    material: str | None = None,
    color: str | None = None,
    color_family: str | None = None,
    shape: str | None = None,
    size_mm: float | None = None,
    finish: str | None = None,
    hole_type: str | None = None,
    focal_bead: bool | None = None,
    focal_subject: str | None = None,
    is_3d: bool | None = None,
    focal_description: str | None = None,
    tags: list[str] | None = None,
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
            material_category=material_category,
            material=material,
            color=color,
            color_family=color_family,
            shape=shape,
            size_mm=size_mm,
            finish=finish,
            hole_type=hole_type,
            focal_bead=focal_bead,
            focal_subject=focal_subject,
            is_3d=is_3d,
            focal_description=focal_description,
            tags=tags,
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
        material_category=req.material_category,
        material=req.material,
        color=req.color,
        color_family=req.color_family,
        shape=req.shape,
        size_mm=req.size_mm,
        finish=req.finish,
        hole_type=req.hole_type,
        focal_bead=req.focal_bead,
        focal_subject=req.focal_subject,
        is_3d=req.is_3d,
        focal_description=req.focal_description,
        tags=req.tags,
    )


# ---------------------------------------------------------------------------
# POST /add-bead/upload  (multipart)
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def add_bead_upload(
    name:              str                  = Form(...),
    sku:               str                  = Form(...),
    source:            str                  = Form("Unknown"),
    cost:              float                = Form(0.0),
    mime_type:         str                  = Form("image/jpeg"),
    image_path:        Optional[str]        = Form(None),
    image:             Optional[UploadFile] = File(None),
    material_category: Optional[str]        = Form(None),
    material:          Optional[str]        = Form(None),
    color:             Optional[str]        = Form(None),
    color_family:      Optional[str]        = Form(None),
    shape:             Optional[str]        = Form(None),
    size_mm:           Optional[float]      = Form(None),
    finish:            Optional[str]        = Form(None),
    hole_type:         Optional[str]        = Form(None),
    focal_bead:        Optional[bool]       = Form(None),
    focal_subject:     Optional[str]        = Form(None),
    is_3d:             Optional[bool]       = Form(None),
    focal_description: Optional[str]        = Form(None),
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
        material_category=material_category,
        material=material,
        color=color,
        color_family=color_family,
        shape=shape,
        size_mm=size_mm,
        finish=finish,
        hole_type=hole_type,
        focal_bead=focal_bead,
        focal_subject=focal_subject,
        is_3d=is_3d,
        focal_description=focal_description,
    )
