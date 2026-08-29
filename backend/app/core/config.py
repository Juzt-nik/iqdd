"""Centralized configuration, sourced from environment variables so the
same image can run in different environments (local, Docker, cloud)
without code changes, per assessment section 11."""
from __future__ import annotations

import os
from pathlib import Path


class Settings:
    # Database: defaults to a local SQLite file; set DATABASE_URL to a
    # Postgres DSN (e.g. postgresql://user:pass@host:5432/db) to switch.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./storage/iqdd.db")

    # Where uploaded images and heatmaps are persisted on disk.
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", "./storage/images"))

    # Where trained model artifacts (tree/, cnn/) live.
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "./data/models"))

    # CORS: comma-separated list of allowed origins for the frontend.
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    ALLOWED_CONTENT_TYPES: set[str] = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

    # If no trained models are found at startup, fail fast with a clear
    # error rather than serving broken analyses silently.
    REQUIRE_MODELS: bool = os.getenv("REQUIRE_MODELS", "true").lower() == "true"


settings = Settings()
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
