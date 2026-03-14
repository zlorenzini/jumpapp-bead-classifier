"""
server/acquisition.py  —  Internet-sourced bead acquisition pipeline.

This module fetches bead images from the internet using a search API and
normalises raw metadata with an LLM extractor.  Both components are defined
as abstract interfaces so they can be swapped for real implementations
(e.g. Google Custom Search, SerpAPI, OpenAI) without changing the pipeline.

IMPORTANT – what this module does NOT do:
  • It does NOT crawl or spider websites.
  • It does NOT parse HTML or follow links.
  • It fetches exactly the image bytes at each URL returned by the search API
    and nothing else.

Pipeline
--------
  query string
      │
      ▼
  SearchClient.search(query, max_results)
      │  returns list[SearchResult]
      ▼
  For each result:
    1. LLMExtractor.extract(title, alt_text, caption, surrounding_text)
           → BeadMetadata (normalised fields)
    2. fetch image bytes from result.image_url  (single HTTP GET, no parsing)
    3. ingest(image_bytes, metadata)  →  calls add_bead._ingest() internally
      │
      ▼
  AcquisitionSummary  (ingested / skipped / errors)
"""

from __future__ import annotations

import abc
import base64
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """One item returned by the search API."""
    image_url:       str
    alt_text:        str  = ""
    caption:         str  = ""
    page_title:      str  = ""
    surrounding_text: str = ""
    source_domain:   str  = ""


@dataclass
class BeadMetadata:
    """Normalised bead description produced by the LLM extractor."""
    name:              str
    sku:               str
    color:             str        = ""
    color_family:      str        = ""
    shape:             str        = ""
    material_category: str        = ""
    material:          str        = ""
    size_mm:           float | None = None
    finish:            str        = ""
    hole_type:         str        = ""
    focal_bead:        bool | None = None
    focal_subject:     str        = ""
    is_3d:             bool | None = None
    focal_description: str        = ""
    source:            str        = "web"
    cost:              float | None = None


@dataclass
class IngestRecord:
    """Record of one successfully ingested image."""
    sku:         str
    name:        str
    image_url:   str
    embedding_id: str = ""


@dataclass
class SkipRecord:
    """Record of a result that was skipped (duplicate SKU, bad image, etc.)."""
    image_url: str
    reason:    str


@dataclass
class ErrorRecord:
    """Record of a result that failed with an unexpected error."""
    image_url: str
    error:     str


@dataclass
class AcquisitionSummary:
    ingested: list[IngestRecord]   = field(default_factory=list)
    skipped:  list[SkipRecord]     = field(default_factory=list)
    errors:   list[ErrorRecord]    = field(default_factory=list)


# ---------------------------------------------------------------------------
# SearchClient interface  (stub provided)
# ---------------------------------------------------------------------------

class SearchClient(abc.ABC):
    """
    Abstract search client.  Implementations should call a search API
    (e.g. Google Custom Search, SerpAPI, Bing Image Search) and return
    structured results — NOT raw HTML.
    """

    @abc.abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """
        Execute a search and return at most ``max_results`` items.
        Each item must contain a direct image URL.
        """


class StubSearchClient(SearchClient):
    """
    Stub implementation that returns zero results.

    Replace with a real implementation backed by a search API:

        class GoogleImageSearchClient(SearchClient):
            def __init__(self, api_key: str, cx: str): ...
            async def search(self, query, max_results): ...

    Wire it in by passing an instance to acquire_beads():

        await acquire_beads(query, max_results, search_client=GoogleImageSearchClient(...))
    """

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        logger.warning(
            "StubSearchClient: no real search API configured — returning 0 results. "
            "Replace with a concrete SearchClient implementation."
        )
        return []


# ---------------------------------------------------------------------------
# LLMExtractor interface  (stub provided)
# ---------------------------------------------------------------------------

class LLMExtractor(abc.ABC):
    """
    Abstract metadata extractor.  Implementations should call an LLM
    (e.g. OpenAI, Anthropic) with the text context and return structured
    bead metadata.
    """

    @abc.abstractmethod
    async def extract(
        self,
        *,
        page_title:       str,
        alt_text:         str,
        caption:          str,
        surrounding_text: str,
        query:            str,
    ) -> dict[str, Any]:
        """
        Return a dict with any subset of these keys:
          name, sku, color, shape, material, size_mm, source, cost

        Missing keys are filled by normalize_metadata().
        """


class StubLLMExtractor(LLMExtractor):
    """
    Stub extractor that derives metadata heuristically from the available
    text fields — no LLM call.  Useful for testing the pipeline without
    API credentials.

    Replace with a real implementation:

        class OpenAIExtractor(LLMExtractor):
            def __init__(self, api_key: str, model: str = "gpt-4o-mini"): ...
            async def extract(self, *, page_title, alt_text, ...): ...
    """

    async def extract(
        self,
        *,
        page_title:       str,
        alt_text:         str,
        caption:          str,
        surrounding_text: str,
        query:            str,
    ) -> dict[str, Any]:
        # Prefer the most descriptive text field available
        text = caption or alt_text or page_title or surrounding_text or query

        return {
            "name":              text[:120].strip() or query,
            "color":             _first_match(text, _COLOR_WORDS),
            "color_family":      _infer_color_family(text),
            "shape":             _first_match(text, _SHAPE_WORDS),
            "material":          _first_match(text, _MATERIAL_WORDS),
            "material_category": _infer_material_category(text),
            "size_mm":           _extract_size_mm(text),
            "source":            "web",
            "cost":              None,
        }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
    "black", "brown", "grey", "gray", "gold", "silver", "rose", "coral",
    "teal", "turquoise", "lavender", "cream", "ivory", "beige", "dusty",
    "navy", "magenta", "cyan", "maroon", "olive", "lime", "cobalt",
    "scarlet", "crimson", "burgundy", "blush", "fuchsia", "peach",
    "salmon", "lemon", "mustard", "amber", "mint", "aqua", "indigo",
    "violet", "plum", "tan", "khaki",
]

# Maps colour words to their normalised family.
_COLOR_FAMILY_MAP: dict[str, str] = {
    "red": "red", "crimson": "red", "scarlet": "red", "maroon": "red",
    "burgundy": "red", "coral": "red",
    "pink": "pink", "blush": "pink", "rose": "pink", "fuchsia": "pink",
    "magenta": "pink", "dusty rose": "pink",
    "orange": "orange", "peach": "orange", "salmon": "orange",
    "yellow": "yellow", "lemon": "yellow", "mustard": "yellow", "amber": "yellow",
    "green": "green", "lime": "green", "olive": "green", "mint": "green",
    "teal": "green",
    "blue": "blue", "navy": "blue", "cobalt": "blue", "turquoise": "blue",
    "aqua": "blue", "cyan": "blue",
    "purple": "purple", "lavender": "purple", "violet": "purple",
    "plum": "purple", "indigo": "purple",
    "brown": "brown", "tan": "brown", "beige": "brown", "khaki": "brown",
    "cream": "white", "ivory": "white", "white": "white",
    "black": "black", "ebony": "black",
    "gray": "gray", "grey": "gray",
    "silver": "silver",
    "gold": "gold", "brass": "gold",
    "multicolor": "multicolor", "multi": "multicolor", "rainbow": "multicolor",
}

_SHAPE_WORDS = [
    "round", "sphere", "ball", "oval", "faceted", "bicone", "tube", "cylinder",
    "flat", "disc", "disk", "square", "cube", "hexagon", "octagon", "star",
    "heart", "teardrop", "drop", "nugget", "chip", "coin", "rondelle",
    "seed", "bugle", "barrel", "letter", "abacus", "spacer",
]

_MATERIAL_WORDS = [
    "glass", "crystal", "acrylic", "plastic", "silicone", "wood", "wooden",
    "metal", "brass", "copper", "silver", "gold", "gemstone", "stone",
    "ceramic", "clay", "resin", "pearl", "shell", "turquoise", "jade",
    "lava", "howlite", "jasper", "agate", "quartz", "obsidian", "garnet",
    "amethyst", "malachite", "lapis", "morganite", "moonstone", "zirconia",
    "rhinestone", "fabric", "lampwork", "millefiori", "czech",
]

# Maps material words to their high-level category.
_MATERIAL_CATEGORY_MAP: dict[str, str] = {
    # glass
    "glass": "glass", "crystal": "glass", "czech": "glass",
    "lampwork": "glass", "millefiori": "glass", "seed bead": "glass",
    "delica": "glass",
    # silicone
    "silicone": "silicone",
    # acrylic / plastic
    "acrylic": "acrylic", "plastic": "acrylic", "pony": "acrylic",
    "bubblegum": "acrylic", "ccb": "acrylic",
    # metal
    "metal": "metal", "brass": "metal", "copper": "metal",
    "silver": "metal", "gold": "metal", "stainless": "metal",
    "sterling": "metal", "zinc": "metal", "alloy": "metal",
    # ceramic / clay
    "ceramic": "ceramic", "clay": "ceramic", "polymer": "ceramic",
    "porcelain": "ceramic",
    # wood
    "wood": "wood", "wooden": "wood", "bamboo": "wood",
    # gemstone / stone
    "gemstone": "gemstone", "stone": "gemstone", "quartz": "gemstone",
    "agate": "gemstone", "jade": "gemstone", "turquoise": "gemstone",
    "jasper": "gemstone", "obsidian": "gemstone", "garnet": "gemstone",
    "amethyst": "gemstone", "malachite": "gemstone", "lapis": "gemstone",
    "howlite": "gemstone", "lava": "gemstone", "moonstone": "gemstone",
    "morganite": "gemstone", "tiger eye": "gemstone",
    # organic
    "shell": "organic", "bone": "organic", "horn": "organic",
    "amber": "organic", "pearl": "organic", "coral": "organic",
    # resin
    "resin": "resin",
    # rhinestone
    "rhinestone": "rhinestone", "zirconia": "rhinestone",
    # fabric
    "fabric": "fabric", "cloth": "fabric",
}

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*)?mm", re.IGNORECASE)


def _first_match(text: str, words: list[str]) -> str:
    tl = text.lower()
    for w in words:
        if w in tl:
            return w
    return ""


def _extract_size_mm(text: str) -> float | None:
    """Return the first millimetre measurement found in text, or None."""
    m = _SIZE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _infer_color_family(text: str) -> str:
    """Return the normalised colour family for the first colour word found."""
    tl = text.lower()
    # Check multi-word entries first (e.g. "dusty rose").
    for phrase, family in _COLOR_FAMILY_MAP.items():
        if phrase in tl:
            return family
    return ""


def _infer_material_category(text: str) -> str:
    """Return the material category for the first material keyword found."""
    tl = text.lower()
    # Check multi-word entries first (e.g. "tiger eye").
    for keyword, category in _MATERIAL_CATEGORY_MAP.items():
        if keyword in tl:
            return category
    return ""


def _slugify(text: str) -> str:
    """Convert arbitrary text to a lowercase hyphenated slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_]+", "-", text)[:60]


def normalize_metadata(raw: dict[str, Any], query: str, source_domain: str) -> BeadMetadata:
    """
    Merge raw extractor output with sensible defaults.

    ``name`` and ``sku`` are generated from the query if the extractor
    did not produce them.  Ontology fields (color_family, material_category)
    are inferred from other text if not explicitly returned by the extractor.
    """
    name              = (raw.get("name") or "").strip() or query
    sku_hint          = (raw.get("sku") or "").strip() or _slugify(name)
    color             = (raw.get("color") or "").strip()
    color_family      = (raw.get("color_family") or "").strip()
    shape             = (raw.get("shape") or "").strip()
    material          = (raw.get("material") or "").strip()
    material_category = (raw.get("material_category") or "").strip()
    finish            = (raw.get("finish") or "").strip()
    hole_type         = (raw.get("hole_type") or "").strip()
    focal_subject     = (raw.get("focal_subject") or "").strip()
    focal_description = (raw.get("focal_description") or "").strip()
    source            = (raw.get("source") or source_domain or "web").strip()

    # focal_bead / is_3d: only accept explicit booleans from extractor
    raw_focal = raw.get("focal_bead")
    focal_bead: bool | None = bool(raw_focal) if raw_focal is not None else None
    raw_3d = raw.get("is_3d")
    is_3d: bool | None = bool(raw_3d) if raw_3d is not None else None

    # size_mm: accept numeric or parse from string
    raw_size = raw.get("size_mm")
    size_mm: float | None = None
    if isinstance(raw_size, (int, float)):
        size_mm = float(raw_size)
    elif isinstance(raw_size, str) and raw_size.strip():
        m = _SIZE_RE.search(raw_size)
        if m:
            try:
                size_mm = float(m.group(1))
            except ValueError:
                pass

    # Infer color_family from color if not already set
    if not color_family and color:
        color_family = _COLOR_FAMILY_MAP.get(color.lower(), "")

    # Infer material_category from material if not already set
    if not material_category and material:
        material_category = _MATERIAL_CATEGORY_MAP.get(material.lower(), "")

    # Cost: only accept a numeric value
    raw_cost = raw.get("cost")
    try:
        cost = float(raw_cost) if raw_cost is not None else None
    except (TypeError, ValueError):
        cost = None

    return BeadMetadata(
        name=name,
        sku=sku_hint,
        color=color,
        color_family=color_family,
        shape=shape,
        material_category=material_category,
        material=material,
        size_mm=size_mm,
        finish=finish,
        hole_type=hole_type,
        focal_bead=focal_bead,
        focal_subject=focal_subject,
        is_3d=is_3d,
        focal_description=focal_description,
        source=source,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Image fetcher
# ---------------------------------------------------------------------------

# Only accept these MIME types — do not fetch or parse HTML
_ALLOWED_IMAGE_MIME = frozenset({
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
})

# Hard size limit: reject images larger than 10 MB
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def fetch_image(
    url: str,
    client: httpx.AsyncClient,
) -> tuple[bytes, str]:
    """
    Fetch exactly the image at ``url``.  No redirects to HTML pages are
    followed: if the final Content-Type is not an image MIME type the
    response is rejected.

    Returns (raw_bytes, mime_type).
    Raises ValueError for bad URLs, wrong content types, or oversized files.
    """
    # Reject non-http(s) schemes to prevent SSRF via file:// or similar
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"Rejected non-HTTP URL scheme: {url!r}")

    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in _ALLOWED_IMAGE_MIME:
        raise ValueError(
            f"URL did not return an image (Content-Type: {content_type!r}): {url}"
        )

    raw = response.content
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large ({len(raw) // 1024} KB > "
            f"{_MAX_IMAGE_BYTES // 1024} KB limit): {url}"
        )
    if len(raw) == 0:
        raise ValueError(f"Empty image response from {url}")

    # Normalise common aliases
    if content_type in ("image/jpg",):
        content_type = "image/jpeg"

    return raw, content_type


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def acquire_beads(
    query: str,
    max_results: int = 5,
    *,
    search_client: SearchClient | None = None,
    extractor: LLMExtractor | None = None,
) -> AcquisitionSummary:
    """
    Run the full acquisition pipeline for the given ``query``.

    Parameters
    ----------
    query:          Free-text bead description, e.g. "10mm silicone hexagon dusty rose".
    max_results:    Maximum number of search results to process.
    search_client:  SearchClient instance.  Defaults to StubSearchClient.
    extractor:      LLMExtractor instance.  Defaults to StubLLMExtractor.

    Returns
    -------
    AcquisitionSummary with ingested, skipped, and error records.
    """
    if search_client is None:
        search_client = StubSearchClient()
    if extractor is None:
        extractor = StubLLMExtractor()

    summary = AcquisitionSummary()

    # ── 1. Search ─────────────────────────────────────────────────────────────
    logger.info("Acquisition: searching for %r (max %d)", query, max_results)
    try:
        results = await search_client.search(query, max_results)
    except Exception as exc:
        logger.exception("Search API error")
        summary.errors.append(ErrorRecord(image_url="", error=f"Search failed: {exc}"))
        return summary

    logger.info("Acquisition: %d result(s) returned", len(results))

    # ── 2. Process each result ────────────────────────────────────────────────
    # Deferred imports to avoid import-time side effects on the embedder
    from routes.add_bead import _ingest  # noqa: PLC0415

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "JumpApp-BeadClassifier/1.0 (image-fetch; not a crawler)"},
        follow_redirects=True,
    ) as http:
        for result in results:
            url = result.image_url
            try:
                # ── 2a. Extract metadata from text context ─────────────────
                raw_meta = await extractor.extract(
                    page_title=result.page_title,
                    alt_text=result.alt_text,
                    caption=result.caption,
                    surrounding_text=result.surrounding_text,
                    query=query,
                )
                meta = normalize_metadata(raw_meta, query, result.source_domain)

                # ── 2b. Fetch image bytes only (no HTML parsing) ───────────
                raw_bytes, mime_type = await fetch_image(url, http)
                image_b64 = base64.b64encode(raw_bytes).decode("ascii")

                # ── 2c. Ingest via shared add-bead logic ───────────────────
                result_dict = await _ingest(
                    name=meta.name,
                    sku=meta.sku,
                    raw_bytes=raw_bytes,
                    image_b64=image_b64,
                    mime_type=mime_type,
                    source=meta.source,
                    cost=meta.cost if meta.cost is not None else 0.0,
                    material_category=meta.material_category or None,
                    material=meta.material or None,
                    color=meta.color or None,
                    color_family=meta.color_family or None,
                    shape=meta.shape or None,
                    size_mm=meta.size_mm,
                    finish=meta.finish or None,
                    hole_type=meta.hole_type or None,
                    focal_bead=meta.focal_bead,
                    focal_subject=meta.focal_subject or None,
                    is_3d=meta.is_3d,
                    focal_description=meta.focal_description or None,
                )

                bead      = result_dict.get("bead", {})
                embedding_id = bead.get("embedding_id", "")
                summary.ingested.append(
                    IngestRecord(
                        sku=meta.sku,
                        name=meta.name,
                        image_url=url,
                        embedding_id=embedding_id,
                    )
                )
                logger.info("Acquired: sku=%r  name=%r  url=%s", meta.sku, meta.name, url)

            except httpx.HTTPStatusError as exc:
                summary.skipped.append(
                    SkipRecord(image_url=url, reason=f"HTTP {exc.response.status_code}")
                )
            except ValueError as exc:
                summary.skipped.append(SkipRecord(image_url=url, reason=str(exc)))
            except Exception as exc:
                logger.exception("Unexpected error for URL %s", url)
                summary.errors.append(ErrorRecord(image_url=url, error=str(exc)))

    return summary
