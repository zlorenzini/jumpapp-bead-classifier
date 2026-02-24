"""
Shopify Admin REST API client.

Only the tag-update flow is implemented:
  - Fetch the current product by ID
  - Merge detected bead tags with existing tags
  - PATCH the product with the updated tag list

Requires:
  SHOPIFY_STORE      e.g. my-store.myshopify.com
  SHOPIFY_API_TOKEN  shpat_xxxxxxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TAG_PREFIX = "bead:"   # e.g. bead:round-red-glass-001


def _api_base(store: str, version: str) -> str:
    return f"https://{store}/admin/api/{version}"


def _headers(token: str) -> dict[str, str]:
    return {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }


async def get_product(
    product_id: str | int,
    store: str,
    token: str,
    version: str = "2024-01",
) -> dict[str, Any]:
    """Fetch a Shopify product by numeric ID."""
    url = f"{_api_base(store, version)}/products/{product_id}.json"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=_headers(token))
        resp.raise_for_status()
        return resp.json()["product"]


async def update_product_tags(
    product_id: str | int,
    bead_ids: list[str],
    store: str,
    token: str,
    version: str = "2024-01",
    replace: bool = False,
) -> dict[str, Any]:
    """
    Merge detected bead IDs into the product's existing Shopify tags.

    Parameters
    ----------
    product_id: Shopify numeric product ID.
    bead_ids:   List of bead_id strings to tag (will be prefixed with
                ``bead:``).
    replace:    If True, remove all existing ``bead:*`` tags before adding
                the new ones.  If False (default), merge.

    Returns the updated product dict.
    """
    if not store or not token:
        raise ValueError(
            "SHOPIFY_STORE and SHOPIFY_API_TOKEN must be set to push tags"
        )

    product = await get_product(product_id, store, token, version)
    existing_tags: list[str] = [
        t.strip() for t in product.get("tags", "").split(",") if t.strip()
    ]

    if replace:
        # Remove all previous bead: tags
        existing_tags = [t for t in existing_tags if not t.startswith(_TAG_PREFIX)]

    # Add new bead tags (de-duplicated)
    new_tags = {f"{_TAG_PREFIX}{bid}" for bid in bead_ids}
    merged = list(dict.fromkeys(existing_tags + sorted(new_tags)))

    url = f"{_api_base(store, version)}/products/{product_id}.json"
    payload = {"product": {"id": product_id, "tags": ", ".join(merged)}}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.put(url, json=payload, headers=_headers(token))
        resp.raise_for_status()
        updated = resp.json()["product"]

    logger.info(
        "Updated Shopify product %s tags → %s",
        product_id,
        updated.get("tags"),
    )
    return updated
