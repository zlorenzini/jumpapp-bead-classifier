"""
Image embedder using a pre-trained ResNet-50 backbone.

The classification head is removed so that the model outputs a 2048-dim
feature vector for each image.  Vectors are L2-normalised before storage and
comparison so that cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import base64
import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import so the server starts even if torch is unavailable.
_model = None
_transform = None


def _load_model():
    global _model, _transform
    if _model is not None:
        return

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    backbone = models.resnet50(weights=weights)
    # Remove fully-connected classification head → output is (batch, 2048)
    backbone.fc = torch.nn.Identity()
    backbone.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=weights.transforms().mean,
                    std=weights.transforms().std),
    ])

    _model = backbone
    _transform = transform
    logger.info("ResNet-50 embedder loaded on %s", device)


def embed_image_bytes(image_bytes: bytes) -> list[float]:
    """Return a normalised 2048-dim embedding for raw image bytes."""
    _load_model()
    import torch

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    device = next(_model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        vec = _model(tensor).squeeze(0).cpu().numpy()  # (2048,)

    # L2-normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


def embed_b64(image_b64: str) -> list[float]:
    """Decode a Base64 string and return its embedding."""
    raw = base64.b64decode(image_b64)
    return embed_image_bytes(raw)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    Because embeddings are L2-normalised this is equivalent to a dot product.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))
