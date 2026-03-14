"""
Bead Classifier JumpApp – FastAPI server.

Endpoints
---------
GET  /bundles              Backbone + embedding-store descriptor (identity card)
GET  /preview              Grab a camera frame (no inference) — for live preview
GET  /status               Health check + readiness flags
POST /add-bead             Ingest one bead angle (image + metadata) into the store
POST /add-bead/upload      Same via multipart/form-data
POST /infer                Identify beads in a product image
POST /capture              Capture a frame via JumpNet then run inference
POST /train                Start a MobileNetV2 training job (async)
GET  /train                List all training jobs
GET  /train/{id}           Job status, progress, and logs
POST /train/{id}/stop      Request graceful stop of a running job
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

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Storage root (location-independent – works wherever the USB drive is mounted)
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
JUMPAPP_ROOT  = Path(os.environ.get("JUMPAPP_ROOT", _DEFAULT_ROOT))
MODELS_DIR    = JUMPAPP_ROOT / "models"
DATASET_DIR   = JUMPAPP_ROOT / "dataset"
ENDPOINTS_DIR = JUMPAPP_ROOT / "endpoints"
_storage_source: str = "env" if os.environ.get("JUMPAPP_ROOT") else "default"

for _d in (MODELS_DIR, DATASET_DIR, ENDPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Endpoint modules stored on the USB drive can be imported directly.
if str(ENDPOINTS_DIR) not in sys.path:
    sys.path.insert(0, str(ENDPOINTS_DIR))

# Server-local modules (config, db, inference, embedder, shopify_client).
_SERVER_DIR = Path(__file__).resolve().parent
_UI_DIR     = _SERVER_DIR.parent / "ui"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from config import get_settings
import db
import inference as infer_module
import shopify_client
from routes.add_bead import router as add_bead_router
from routes.train    import router as train_router
from routes.bundles  import router as bundles_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage root resolution via JumpNet
# ---------------------------------------------------------------------------

async def _resolve_storage_root(jumpnet_url: str) -> Path | None:
    """
    Query JumpNet GET /status for announced JUMPAPP drive mounts.
    Returns the root Path for *this* app (matched by jumpapp.json ``name``),
    or None if JumpNet is unreachable or no matching drive is found.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{jumpnet_url}/status")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning("Could not reach JumpNet for storage resolution: %s", exc)
        return None

    jumpapps = data.get("storage", {}).get("jumpapps", [])
    for entry in jumpapps:
        if entry.get("name") == "jumpapp-bead-classifier":
            root = Path(entry["root"])
            logger.info("Storage root announced by JumpNet: %s", root)
            return root

    if jumpapps:
        logger.warning(
            "JumpNet found %d JUMPAPP drive(s) but none named 'jumpapp-bead-classifier': %s",
            len(jumpapps),
            [e.get("name") for e in jumpapps],
        )
    else:
        logger.warning("JumpNet reported no mounted JUMPAPP drives")

    return None

# ---------------------------------------------------------------------------
# Lifespan: open / close MongoDB connection
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global JUMPAPP_ROOT, MODELS_DIR, DATASET_DIR, ENDPOINTS_DIR, _storage_source

    cfg = get_settings()

    # ── Resolve storage root from JumpNet (USB drive location) ──────────────
    # JUMPAPP_ROOT env var always wins; otherwise ask JumpNet; then fall-back
    # to the default (directory containing this app's jumpapp.json).
    if not os.environ.get("JUMPAPP_ROOT"):
        announced = await _resolve_storage_root(cfg.jumpnet_url)
        if announced is not None:
            JUMPAPP_ROOT  = announced
            MODELS_DIR    = JUMPAPP_ROOT / "models"
            DATASET_DIR   = JUMPAPP_ROOT / "dataset"
            ENDPOINTS_DIR = JUMPAPP_ROOT / "endpoints"
            _storage_source = "jumpnet"

            for _d in (MODELS_DIR, DATASET_DIR, ENDPOINTS_DIR):
                _d.mkdir(parents=True, exist_ok=True)

            if str(ENDPOINTS_DIR) not in sys.path:
                sys.path.insert(0, str(ENDPOINTS_DIR))

    logger.info(
        "Storage root: %s  (models=%s  dataset=%s)",
        JUMPAPP_ROOT, MODELS_DIR, DATASET_DIR,
    )

    # ── Publish resolved paths to shared state ───────────────────────────────
    import state as _state  # noqa: PLC0415
    _state.models_dir = MODELS_DIR

    # ── MongoDB ──────────────────────────────────────────────────────────────
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

app.include_router(add_bead_router, prefix="/add-bead", tags=["add-bead"])
app.include_router(train_router,    prefix="/train",    tags=["train"])
app.include_router(bundles_router,  prefix="/bundles",  tags=["bundles"])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def ui_root() -> FileResponse:
    return FileResponse(_UI_DIR / "index.html")


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



class CaptureRequest(BaseModel):
    device: str | None = None
    resolution: str | None = None
    fps: int | None = None
    warmup: int | None = None
    input_format: str | None = None  # "mjpeg" (default) or "yuyv422" (raw, no on-chip JPEG)


class CaptureResponse(BaseModel):
    matches: list[BeadMatch]
    image_b64: str
    mime_type: str = "image/jpeg"
    device: str | None = None
    resolution: str | None = None
    latency_ms: int | None = None


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
            "source":    _storage_source,
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
            detail="No beads in the data lake. Add reference images via POST /add-bead first.",
        )

    try:
        results = await infer_module.run_inference(
            image_bytes=image_bytes,
            bead_records=bead_records,
            top_k=cfg.top_k,
            threshold=cfg.confidence_threshold,
            models_dir=MODELS_DIR,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    matches = [BeadMatch(**r) for r in results]
    return InferResponse(matches=matches, image_filename=image.filename)


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
# GET /preview  — grab a frame from JumpNet, no inference
# ---------------------------------------------------------------------------

class PreviewResponse(BaseModel):
    image_b64:  str
    mime_type:  str = "image/jpeg"
    latency_ms: float | None = None
    device:     str | None = None
    resolution: str | None = None


@app.get("/preview", response_model=PreviewResponse)
async def preview():
    """
    Capture a single camera frame via JumpNet and return the raw image.
    No inference is performed — intended for live viewfinder polling.
    """
    cfg = get_settings()

    payload = {
        "imageOnly":    True,
        "device":       cfg.capture_device,
        "resolution":   cfg.capture_resolution,
        "warmup":       cfg.capture_warmup,
        "inputFormat":  cfg.capture_input_format,
    }
    if cfg.capture_fps is not None:
        payload["fps"] = cfg.capture_fps

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{cfg.jumpnet_url}/capture", json=payload)
            r.raise_for_status()
            frame = r.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach JumpNet at {cfg.jumpnet_url}. Is it running?",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"JumpNet capture error: {exc.response.text}",
        )

    image_b64 = frame.get("image_b64", "")
    if not image_b64:
        raise HTTPException(status_code=502, detail="JumpNet returned no image data")

    return PreviewResponse(
        image_b64=image_b64,
        mime_type=frame.get("mime_type", "image/jpeg"),
        latency_ms=frame.get("latency_ms"),
        device=frame.get("device"),
        resolution=frame.get("resolution"),
    )


# POST /capture  — grab a frame from JumpNet, run embedding inference
# ---------------------------------------------------------------------------

@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest = CaptureRequest()):
    """
    Capture a frame from the local camera via JumpNet, embed it with
    ResNet-50, and return the best-matching beads from the data lake.

    JumpNet must be running (default: http://localhost:4080) and ffmpeg
    must be available on the JumpNet host machine.
    """
    cfg = get_settings()

    payload = {
        "imageOnly":   True,
        "device":      req.device      or cfg.capture_device,
        "resolution":  req.resolution  or cfg.capture_resolution,
        "warmup":      req.warmup      or cfg.capture_warmup,
        "inputFormat": req.input_format or cfg.capture_input_format,
    }
    resolved_fps = req.fps or cfg.capture_fps
    if resolved_fps is not None:
        payload["fps"] = resolved_fps

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{cfg.jumpnet_url}/capture", json=payload)
            r.raise_for_status()
            frame = r.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach JumpNet at {cfg.jumpnet_url}. Is it running?",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"JumpNet capture error: {exc.response.text}",
        )

    image_b64  = frame.get("image_b64", "")
    mime_type  = frame.get("mime_type", "image/jpeg")
    latency_ms = frame.get("latency_ms")

    if not image_b64:
        raise HTTPException(status_code=502, detail="JumpNet returned no image data")

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=502, detail="JumpNet returned invalid base64 image")

    bead_records = await db.get_bead_embeddings(cfg.mongodb_collection)
    if not bead_records:
        raise HTTPException(
            status_code=422,
            detail="No beads in the data lake. Add reference images via POST /add-bead first.",
        )

    try:
        results = await infer_module.run_inference(
            image_bytes=image_bytes,
            bead_records=bead_records,
            top_k=cfg.top_k,
            threshold=cfg.confidence_threshold,
            models_dir=MODELS_DIR,
        )
    except Exception as exc:
        logger.exception("Inference failed during /capture")
        raise HTTPException(status_code=500, detail=str(exc))

    matches = [BeadMatch(**r) for r in results]
    return CaptureResponse(
        matches=matches,
        image_b64=image_b64,
        mime_type=mime_type,
        device=frame.get("device"),
        resolution=frame.get("resolution"),
        latency_ms=latency_ms,
    )


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

