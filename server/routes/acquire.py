"""
POST /acquire

Search the internet for bead images matching a query string, extract
metadata from text context, download each image, and ingest it into the
bead data lake via the existing /add-bead pipeline.

This endpoint does NOT crawl websites or parse HTML.  It fetches only the
direct image URLs returned by the search API.

Request body
------------
{
    "query":       "10mm silicone hexagon dusty rose",
    "max_results": 5
}

Response
--------
{
    "query":      "10mm silicone hexagon dusty rose",
    "ingested": [
        { "sku": "10mm-silicone-hexagon-dusty-rose", "name": "...",
          "image_url": "https://…/img.jpg", "embedding_id": "…" }
    ],
    "skipped": [
        { "image_url": "https://…/img2.jpg", "reason": "HTTP 403" }
    ],
    "errors": []
}
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class AcquireRequest(BaseModel):
    query:       str = Field(..., min_length=1, max_length=500,
                             description="Free-text bead description, e.g. '10mm silicone hexagon dusty rose'")
    max_results: int = Field(5, ge=1, le=50,
                             description="Maximum number of search results to process")


@router.post("")
async def acquire(req: AcquireRequest):
    """
    Search for bead images online, extract metadata, and ingest them.

    Both the search client and the LLM metadata extractor are pluggable
    interfaces.  The default stubs return zero results; wire in real
    implementations via the acquisition module.
    """
    from acquisition import acquire_beads  # noqa: PLC0415

    try:
        summary = await acquire_beads(
            query=req.query,
            max_results=req.max_results,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query":    req.query,
        "ingested": [
            {
                "sku":          r.sku,
                "name":         r.name,
                "image_url":    r.image_url,
                "embedding_id": r.embedding_id,
            }
            for r in summary.ingested
        ],
        "skipped": [
            {"image_url": r.image_url, "reason": r.reason}
            for r in summary.skipped
        ],
        "errors": [
            {"image_url": r.image_url, "error": r.error}
            for r in summary.errors
        ],
    }
