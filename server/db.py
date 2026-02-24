"""
MongoDB data-lake client.

Bead document schema
--------------------
{
    "_id":       ObjectId,
    "bead_id":   str,          # human-readable unique ID, e.g. "round-red-glass-001"
    "name":      str,
    "color":     str,
    "finish":    str,
    "shape":     str,
    "tags":      [str],        # arbitrary extra tags
    "focal_bead": bool,        # True if this is a focal bead
    "is_3d":      bool,        # True = 3-D figurine; False = flat (~5-6 mm thick)
    "focal_description": str,  # optional description (rare but not unique)
    "images": [               # one or more reference images
        {
            "data":      str,  # Base64-encoded JPEG/PNG
            "mime_type": str,  # "image/jpeg" | "image/png"
        }
    ],
    "embeddings": [[float]],   # list of embedding vectors (one per image)
    "created_at": datetime,
    "updated_at": datetime,
}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect(uri: str, db_name: str) -> None:
    """Open the MongoDB connection and ensure indexes exist."""
    global _client, _db
    _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5_000)
    _db = _client[db_name]
    await _ensure_indexes()
    logger.info("Connected to MongoDB at %s (db=%s)", uri, db_name)


async def disconnect() -> None:
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("Disconnected from MongoDB")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialised – call connect() first")
    return _db


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

async def _ensure_indexes() -> None:
    col = get_db()["beads"]
    await col.create_indexes([
        IndexModel([("bead_id", ASCENDING)], unique=True),
        IndexModel([("name", ASCENDING)]),
        IndexModel([("color", ASCENDING)]),
        IndexModel([("shape", ASCENDING)]),
    ])


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

async def upsert_bead(
    collection_name: str,
    bead_id: str,
    name: str,
    color: str,
    finish: str,
    shape: str,
    tags: list[str],
    focal_bead: bool,
    is_3d: bool,
    focal_description: str,
    image_b64: str,
    mime_type: str,
    embedding: list[float],
) -> dict[str, Any]:
    """
    Insert a new bead record or append a reference image + embedding to an
    existing one.  Returns the full updated document.
    """
    col = get_db()[collection_name]
    now = datetime.now(timezone.utc)

    existing = await col.find_one({"bead_id": bead_id})
    if existing is None:
        doc: dict[str, Any] = {
            "bead_id": bead_id,
            "name": name,
            "color": color,
            "finish": finish,
            "shape": shape,
            "tags": tags,
            "focal_bead": focal_bead,
            "is_3d": is_3d,
            "focal_description": focal_description,
            "images": [{"data": image_b64, "mime_type": mime_type}],
            "embeddings": [embedding],
            "created_at": now,
            "updated_at": now,
        }
        await col.insert_one(doc)
    else:
        await col.update_one(
            {"bead_id": bead_id},
            {
                "$push": {
                    "images": {"data": image_b64, "mime_type": mime_type},
                    "embeddings": embedding,
                },
                "$set": {
                    "name": name,
                    "color": color,
                    "finish": finish,
                    "shape": shape,
                    "tags": tags,
                    "focal_bead": focal_bead,
                    "is_3d": is_3d,
                    "focal_description": focal_description,
                    "updated_at": now,
                },
            },
        )

    return await col.find_one({"bead_id": bead_id}, {"_id": 0})


async def get_all_beads(collection_name: str) -> list[dict[str, Any]]:
    """Return all bead documents without the raw image data (for listing)."""
    col = get_db()[collection_name]
    cursor = col.find({}, {"_id": 0, "images": 0})
    return await cursor.to_list(length=None)


async def get_bead_embeddings(collection_name: str) -> list[dict[str, Any]]:
    """Return bead_id + embeddings for all beads.  Used during inference."""
    col = get_db()[collection_name]
    cursor = col.find({}, {"_id": 0, "bead_id": 1, "name": 1, "embeddings": 1})
    return await cursor.to_list(length=None)


async def get_bead_by_id(collection_name: str, bead_id: str) -> dict[str, Any] | None:
    col = get_db()[collection_name]
    return await col.find_one({"bead_id": bead_id}, {"_id": 0})


async def delete_bead(collection_name: str, bead_id: str) -> bool:
    col = get_db()[collection_name]
    result = await col.delete_one({"bead_id": bead_id})
    return result.deleted_count > 0
