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
    # Server
    # ------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache()
def get_settings() -> Settings:
    return Settings()
