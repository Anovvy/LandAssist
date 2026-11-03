"""Health check and model version inventory endpoints."""

from typing import Any

from fastapi import APIRouter

from app.config import settings
from app.db import db

router = APIRouter(tags=["health & versions"])


@router.get("/healthz")
async def health_check() -> dict[str, Any]:
    """Liveness and readiness probe reporting service status and active DB mode."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "default_model_version": settings.default_model_version,
        "database_mode": "postgis" if db.is_connected else "demo_fallback",
    }


@router.get("/v1/versions")
async def get_versions() -> dict[str, Any]:
    """Return available model versions in PostGIS for instant rollback and A/B testing."""
    versions = await db.list_available_versions()
    return {
        "default_model_version": settings.default_model_version,
        "versions": versions,
    }
