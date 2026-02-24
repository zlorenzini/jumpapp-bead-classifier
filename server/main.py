"""
Bead Classifier JumpApp – FastAPI server.

Endpoints
---------
GET  /status               Health check + readiness flags
POST /infer                Identify beads in a product image
POST /train                Add a new bead reference to the data lake
GET  /beads                List all beads in the data lake
GET  /beads/{bead_id}      Fetch a single bead record
DELETE /beads/{bead_id}    Remove a bead from the data lake
POST /shopify/tag          Push detected bead tags to a Shopify product

Start
-----
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Storage root (location-independent – works wherever the USB drive is mounted)
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
JUMPAPP_ROOT  = Path(os.environ.get("JUMPAPP_ROOT", _DEFAULT_ROOT))
MODELS_DIR    = JUMPAPP_ROOT / "models"
DATASET_DIR   = JUMPAPP_ROOT / "dataset"
ENDPOINTS_DIR = JUMPAPP_ROOT / "endpoints"

for _d in (MODELS_DIR, DATASET_DIR, ENDPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Endpoint modules stored on the USB drive can be imported directly.
if str(ENDPOINTS_DIR) not in sys.path:
    sys.path.insert(0, str(ENDPOINTS_DIR))

# Server-local modules (config, db, inference, embedder, shopify_client).
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from config import get_settings
import db
import inference as infer_module
import shopify_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: open / close MongoDB connection
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    cfg = get_settings()
    await db.connect(cfg.mongodb_uri, cfg.mongodb_db)
    yield
    await db.disconnect()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bead Classifier",
    version="1.0.0",
    description="Identify bead types in product images and sync tags to Shopify.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BeadMatch(BaseModel):
    bead_id: str
    name: str
    confidence: float = Field(ge=0.0, le=1.0)


class InferResponse(BaseModel):
    matches: list[BeadMatch]
    image_filename: str | None = None


class TrainRequest(BaseModel):
    bead_id: str = Field(..., description="Unique identifier, e.g. round-red-glass-001")
    name: str
    color: str = ""
    finish: str = ""
    shape: str = ""
    tags: list[str] = []
    focal_bead: bool = False
    is_3d: bool = Field(False, description="True = 3-D figurine; False = flat (~5–6 mm thick) with character on both faces")
    focal_description: str = ""
    image_b64: str = Field(..., description="Base64-encoded JPEG or PNG")
    mime_type: str = "image/jpeg"


class TrainResponse(BaseModel):
    bead_id: str
    message: str


class ShopifyTagRequest(BaseModel):
    product_id: str = Field(..., description="Shopify numeric product ID")
    bead_ids: list[str]
    replace: bool = Field(False, description="Replace existing bead: tags instead of merging")


class ShopifyTagResponse(BaseModel):
    product_id: str
    tags: str


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/status")
async def status_endpoint():
    """
    Returns server health and readiness flags.
    The UI polls this on startup to decide whether to enable the Run button.
    """
    cfg = get_settings()
    db_ok = db._db is not None

    bead_count = 0
    if db_ok:
        try:
            beads = await db.get_all_beads(cfg.mongodb_collection)
            bead_count = len(beads)
        except Exception:
            db_ok = False

    return {
        "status": "ok",
        "version": "1.0.0",
        "db_connected": db_ok,
        "bead_count": bead_count,
        "shopify_configured": bool(cfg.shopify_store and cfg.shopify_api_token),
        "storage": {
            "root":      str(JUMPAPP_ROOT),
            "models":    str(MODELS_DIR),
            "dataset":   str(DATASET_DIR),
            "endpoints": str(ENDPOINTS_DIR),
        },
    }


# ---------------------------------------------------------------------------
# POST /infer
# ---------------------------------------------------------------------------

@app.post("/infer", response_model=InferResponse)
async def infer(image: UploadFile = File(..., description="Product photo (JPEG / PNG)")):
    """
    Analyse a product image and return the best-matching bead types.

    Accepts multipart/form-data with a single ``image`` file field.
    """
    cfg = get_settings()

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    bead_records = await db.get_bead_embeddings(cfg.mongodb_collection)
    if not bead_records:
        raise HTTPException(
            status_code=422,
            detail="No beads in the data lake. Add reference images via POST /train first.",
        )

    try:
        results = await infer_module.run_inference(
            image_bytes=image_bytes,
            bead_records=bead_records,
            top_k=cfg.top_k,
            threshold=cfg.confidence_threshold,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    matches = [BeadMatch(**r) for r in results]
    return InferResponse(matches=matches, image_filename=image.filename)


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
async def train(req: TrainRequest):
    """
    Add a new bead reference image to the data lake.

    If a bead with the same ``bead_id`` already exists, the new image and its
    embedding are appended so the classifier becomes more robust.
    """
    cfg = get_settings()

    try:
        raw_bytes = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="image_b64 is not valid Base64")

    try:
        from embedder import embed_image_bytes
        embedding = embed_image_bytes(raw_bytes)
    except Exception as exc:
        logger.exception("Embedding failed during training")
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    try:
        await db.upsert_bead(
            collection_name=cfg.mongodb_collection,
            bead_id=req.bead_id,
            name=req.name,
            color=req.color,
            finish=req.finish,
            shape=req.shape,
            tags=req.tags,
            focal_bead=req.focal_bead,
            is_3d=req.is_3d,
            focal_description=req.focal_description,
            image_b64=req.image_b64,
            mime_type=req.mime_type,
            embedding=embedding,
        )
    except Exception as exc:
        logger.exception("DB upsert failed during training")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return TrainResponse(
        bead_id=req.bead_id,
        message=f"Bead '{req.bead_id}' saved successfully.",
    )


# ---------------------------------------------------------------------------
# GET /beads  &  GET /beads/{bead_id}  &  DELETE /beads/{bead_id}
# ---------------------------------------------------------------------------

@app.get("/beads", response_model=list[dict[str, Any]])
async def list_beads():
    """List all bead records (without raw image data)."""
    cfg = get_settings()
    return await db.get_all_beads(cfg.mongodb_collection)


@app.get("/beads/{bead_id}")
async def get_bead(bead_id: str):
    """Fetch a single bead record including image data."""
    cfg = get_settings()
    bead = await db.get_bead_by_id(cfg.mongodb_collection, bead_id)
    if bead is None:
        raise HTTPException(status_code=404, detail=f"Bead '{bead_id}' not found")
    return bead


@app.delete("/beads/{bead_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bead(bead_id: str):
    """Remove a bead from the data lake."""
    cfg = get_settings()
    deleted = await db.delete_bead(cfg.mongodb_collection, bead_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Bead '{bead_id}' not found")


# ---------------------------------------------------------------------------
# POST /shopify/tag
# ---------------------------------------------------------------------------

@app.post("/shopify/tag", response_model=ShopifyTagResponse)
async def shopify_tag(req: ShopifyTagRequest):
    """
    Update a Shopify product's tags with the detected bead types.

    Tags are prefixed with ``bead:`` so they can be filtered in Shopify.
    """
    cfg = get_settings()
    if not cfg.shopify_store or not cfg.shopify_api_token:
        raise HTTPException(
            status_code=503,
            detail=(
                "Shopify is not configured. "
                "Set SHOPIFY_STORE and SHOPIFY_API_TOKEN in your .env file."
            ),
        )

    try:
        updated = await shopify_client.update_product_tags(
            product_id=req.product_id,
            bead_ids=req.bead_ids,
            store=cfg.shopify_store,
            token=cfg.shopify_api_token,
            version=cfg.shopify_api_version,
            replace=req.replace,
        )
    except Exception as exc:
        logger.exception("Shopify tag update failed")
        raise HTTPException(status_code=502, detail=str(exc))

    return ShopifyTagResponse(
        product_id=str(updated["id"]),
        tags=updated.get("tags", ""),
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run("main:app", host=cfg.host, port=cfg.port, reload=True)

