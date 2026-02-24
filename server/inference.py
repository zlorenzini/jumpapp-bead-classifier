"""
Similarity-based bead inference.

Algorithm
---------
1. Embed the query image with the ResNet-50 backbone.
2. Load all stored bead embeddings from MongoDB.
3. For each bead, compute the *maximum* cosine similarity across all of
   its stored reference embeddings (a bead can have multiple references).
4. Sort by score descending, apply the confidence threshold, return top-K.
"""

from __future__ import annotations

import logging
from typing import Any

from embedder import cosine_similarity, embed_image_bytes

logger = logging.getLogger(__name__)


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    """Average a list of embedding vectors component-wise (and renormalise)."""
    import numpy as np
    arr = np.array(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean /= norm
    return mean.tolist()


async def run_inference(
    image_bytes: bytes,
    bead_records: list[dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.65,
) -> list[dict[str, Any]]:
    """
    Parameters
    ----------
    image_bytes:   Raw bytes of the query image.
    bead_records:  List of dicts with keys: bead_id, name, embeddings.
    top_k:         Maximum number of results to return.
    threshold:     Minimum cosine-similarity score to include a result.

    Returns
    -------
    List of dicts: bead_id, name, confidence (float 0-1).
    """
    if not bead_records:
        logger.warning("No bead records in the data lake; returning empty results")
        return []

    query_vec = embed_image_bytes(image_bytes)

    scored: list[tuple[float, dict[str, Any]]] = []
    for bead in bead_records:
        embeddings: list[list[float]] = bead.get("embeddings") or []
        if not embeddings:
            continue

        # Representative vector: average over all reference embeddings
        rep = _mean_embedding(embeddings)
        score = cosine_similarity(query_vec, rep)
        scored.append((score, bead))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, bead in scored[:top_k]:
        if score < threshold:
            break
        results.append({
            "bead_id":    bead["bead_id"],
            "name":       bead.get("name", ""),
            "confidence": round(score, 4),
        })

    logger.info("Inference returned %d match(es) above threshold %.2f",
                len(results), threshold)
    return results
