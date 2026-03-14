"""
Centralised configuration loaded from environment variables / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # MongoDB
    # ------------------------------------------------------------------
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "bead_classifier"
    mongodb_collection: str = "beads"

    # ------------------------------------------------------------------
    # Shopify
    # ------------------------------------------------------------------
    shopify_store: str = ""          # e.g. my-store.myshopify.com
    shopify_api_token: str = ""      # shpat_xxxxxxxxxxxxxxxxxxxx
    shopify_api_version: str = "2024-01"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    confidence_threshold: float = 0.65
    top_k: int = 5

    # ------------------------------------------------------------------
    # JumpNet integration
    # ------------------------------------------------------------------
    jumpnet_url: str = "http://localhost:4080"
    capture_device: str = "/dev/video0"
    capture_resolution: str = "960x720"
    capture_fps: int | None = None  # None = auto (native max for format: 15 fps for yuyv422 @ 960x720)
    capture_warmup: int = 10
    capture_input_format: str = "yuyv422"  # "mjpeg" or "yuyv422" (raw, no on-chip JPEG)

    # ------------------------------------------------------------------
    # Web acquisition
    # ------------------------------------------------------------------
    # SerpAPI (https://serpapi.com/) — Google Images via a managed API
    serpapi_key: str = ""

    # Google Custom Search Engine (https://developers.google.com/custom-search/)
    google_api_key: str = ""   # API key restricted to Custom Search API
    google_cx:      str = ""   # Programmable Search Engine ID

    # OpenAI — LLM metadata extractor
    openai_api_key: str = ""
    openai_model:   str = "gpt-4o-mini"

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache()
def get_settings() -> Settings:
    return Settings()
