"""
MongoDB data-lake client.

Bead document schema  (v2 — angles array)
------------------------------------------
{
    "_id":        ObjectId,
    "name":       str,            # human-readable display name
    "sku":        str,            # unique business key, e.g. "GS-10-SPH"
    # Ontology fields (all optional; omitted when not supplied)
    "material_category": str,    # glass | silicone | acrylic | metal | ceramic | wood | gemstone | organic | resin | rhinestone | fabric | other
    "material":   str,           # specific sub-type, e.g. "czech glass", "sterling silver"
    "color":      str,           # primary colour name, e.g. "dusty rose"
    "color_family": str,         # normalised group: red | blue | gold | multicolor | …
    "shape":      str,           # round | oval | tube | flat | bicone | hexagon | rondelle | …
    "size_mm":    float,         # nominal diameter in mm
    "finish":     str,           # glossy | matte | transparent | frosted | metallic | …
    "hole_type":  str,           # center-drilled | large-hole | top-drilled | side-drilled
    "focal_bead":        bool, # True if this is a focal bead (character, animal, or saying)
    "focal_subject":     str,  # short name of the character/saying, e.g. "Stitch", "Mama Bear"
    "is_3d":             bool, # True = 3-D sculpted figurine; False = flat 2-D bead ~6-8 mm thick
    "focal_description": str,  # detailed description: pose, colours, variant, etc.
    "tags":       [str],         # arbitrary extra tags
    "angles": [                   # one entry per captured angle / image
        {
            "image_id":  str,     # UUID for this angle
            "image_b64": str,     # Base64-encoded JPEG/PNG
            "mime_type": str,     # "image/jpeg" | "image/png"
            "embedding": [float], # 2048-dim ResNet-50 vector (L2-normalised)
            "source":    str,     # vendor / origin, e.g. "Temu", "Amazon"
            "cost":      float,   # price paid for this batch (business metadata)
            "timestamp": datetime,
        }
    ],
    "created_at": datetime,
    "updated_at": datetime,
}

Legacy documents (v1) stored parallel arrays ``images[]`` and ``embeddings[]``
instead of ``angles[]``.  ``get_bead_embeddings()`` handles both shapes.
"""

from __future__ import annotations

import logging
import uuid
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
        IndexModel([("sku", ASCENDING)], unique=True, sparse=True),
        IndexModel([("name", ASCENDING)]),
        IndexModel([("material_category", ASCENDING)]),
        IndexModel([("shape", ASCENDING)]),
        IndexModel([("color_family", ASCENDING)]),
        IndexModel([("focal_bead", ASCENDING)]),
        IndexModel([("focal_subject", ASCENDING)]),
        # Legacy v1 index kept for backward compat
        IndexModel([("bead_id", ASCENDING)], unique=True, sparse=True),
    ])


# ---------------------------------------------------------------------------
# v2  add_bead_angle  (new ingestion path)
# ---------------------------------------------------------------------------

async def add_bead_angle(
    collection_name: str,
    name: str,
    sku: str,
    image_b64: str,
    mime_type: str,
    embedding: list[float],
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
) -> dict[str, Any]:
    """
    Append one angle (image + embedding + business metadata) to a bead record.

    If no document with the given ``sku`` exists it is created.
    Ontology fields (material_category, shape, etc.) are stored at the bead
    level; only non-None values overwrite existing values on upsert.
    Returns a summary dict (no raw image data) representing the stored angle.
    """
    col = get_db()[collection_name]
    now = datetime.now(timezone.utc)
    image_id = str(uuid.uuid4())

    angle: dict[str, Any] = {
        "image_id":  image_id,
        "image_b64": image_b64,
        "mime_type": mime_type,
        "embedding": embedding,
        "source":    source,
        "cost":      cost,
        "timestamp": now,
    }

    # Collect only the non-None ontology fields so we never store null values.
    ontology: dict[str, Any] = {
        k: v for k, v in {
            "material_category": material_category,
            "material":          material,
            "color":             color,
            "color_family":      color_family,
            "shape":             shape,
            "size_mm":           size_mm,
            "finish":            finish,
            "hole_type":         hole_type,
            "focal_bead":        focal_bead,
            "focal_subject":     focal_subject,
            "is_3d":             is_3d,
            "focal_description": focal_description,
            "tags":              tags,
        }.items()
        if v is not None
    }

    existing = await col.find_one({"sku": sku})
    if existing is None:
        doc: dict[str, Any] = {
            "name":       name,
            "sku":        sku,
            "angles":     [angle],
            "created_at": now,
            "updated_at": now,
            **ontology,
        }
        result = await col.insert_one(doc)
        embedding_id = str(result.inserted_id) + ":0"
    else:
        angle_index = len(existing.get("angles") or [])
        await col.update_one(
            {"sku": sku},
            {
                "$push": {"angles": angle},
                "$set":  {"name": name, "updated_at": now, **ontology},
            },
        )
        embedding_id = str(existing["_id"]) + f":{angle_index}"

    return {
        "name":         name,
        "sku":          sku,
        "embedding_dim": len(embedding),
        "image_id":     image_id,
        "embedding_id": embedding_id,
        "timestamp":    now.isoformat(),
        "source":       source,
        "cost":         cost,
    }


# ---------------------------------------------------------------------------
# v1  upsert_bead  (legacy ingestion path — kept for /train backward compat)
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
    """Return all bead documents without raw image data (for listing)."""
    col = get_db()[collection_name]
    # Strip image payload from both v1 (images[].data) and v2 (angles[].image_b64)
    cursor = col.find({}, {"_id": 0, "images": 0, "angles.image_b64": 0})
    docs = await cursor.to_list(length=None)
    # Normalise: expose angle count as a top-level summary field
    for doc in docs:
        if "angles" in doc:
            doc["angle_count"] = len(doc["angles"])
        elif "embeddings" in doc:
            doc["angle_count"] = len(doc["embeddings"])
    return docs


async def get_bead_embeddings(collection_name: str) -> list[dict[str, Any]]:
    """
    Return bead identity + all embedding vectors for every bead.
    Used by the inference engine.

    Handles both:
      v1  — parallel ``embeddings: [[float]]`` array, keyed by ``bead_id``
      v2  — ``angles: [{embedding, ...}]`` array, keyed by ``sku``
    """
    col = get_db()[collection_name]
    cursor = col.find(
        {},
        {"_id": 0, "bead_id": 1, "sku": 1, "name": 1,
         "embeddings": 1, "angles.embedding": 1},
    )
    raw = await cursor.to_list(length=None)

    results = []
    for doc in raw:
        # v2 path
        if "angles" in doc:
            embeddings = [a["embedding"] for a in doc["angles"] if a.get("embedding")]
            if not embeddings:
                continue
            results.append({
                "bead_id":    doc.get("sku") or doc.get("bead_id") or "",
                "name":       doc.get("name", ""),
                "embeddings": embeddings,
            })
        # v1 path
        elif "embeddings" in doc:
            results.append({
                "bead_id":    doc.get("bead_id") or doc.get("sku") or "",
                "name":       doc.get("name", ""),
                "embeddings": doc["embeddings"],
            })

    return results


async def get_bead_by_id(collection_name: str, bead_id: str) -> dict[str, Any] | None:
    """Look up by sku (v2) or bead_id (v1)."""
    col = get_db()[collection_name]
    doc = await col.find_one(
        {"$or": [{"sku": bead_id}, {"bead_id": bead_id}]},
        {"_id": 0},
    )
    return doc


async def delete_bead(collection_name: str, bead_id: str) -> bool:
    col = get_db()[collection_name]
    result = await col.delete_one({"$or": [{"sku": bead_id}, {"bead_id": bead_id}]})
    return result.deleted_count > 0


# ---------------------------------------------------------------------------
# Bundle stats (used by GET /bundles)
# ---------------------------------------------------------------------------

async def get_bundle_stats(collection_name: str) -> dict[str, Any]:
    """
    Return aggregate counts across all bead documents:
      classes       — number of distinct bead records
      embeddings    — total stored embedding vectors across all beads
      last_updated  — ISO-8601 timestamp of the most recently modified bead,
                      or None if the collection is empty

    Handles both v2 (angles[]) and v1 (embeddings[]) document shapes.
    """
    col = get_db()[collection_name]

    pipeline = [
        {
            "$project": {
                "updated_at": 1,
                # v2: count angles array; v1: count embeddings array; fallback 0
                "n_embeddings": {
                    "$cond": {
                        "if":   {"$isArray": "$angles"},
                        "then": {"$size": "$angles"},
                        "else": {
                            "$cond": {
                                "if":   {"$isArray": "$embeddings"},
                                "then": {"$size": "$embeddings"},
                                "else": 0,
                            }
                        },
                    }
                },
            }
        },
        {
            "$group": {
                "_id":          None,
                "classes":      {"$sum": 1},
                "embeddings":   {"$sum": "$n_embeddings"},
                "last_updated": {"$max": "$updated_at"},
            }
        },
    ]

    results = await col.aggregate(pipeline).to_list(length=1)
    if not results:
        return {"classes": 0, "embeddings": 0, "last_updated": None}

    row = results[0]
    ts = row.get("last_updated")
    return {
        "classes":      row["classes"],
        "embeddings":   row["embeddings"],
        "last_updated": ts.isoformat() if isinstance(ts, datetime) else str(ts) if ts else None,
    }
