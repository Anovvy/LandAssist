"""OAuth 2.0 dynamic configuration router for LandAssist Web Editor."""

from typing import Any
from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.get("/config")
async def get_auth_config() -> dict[str, Any]:
    """Provide public OAuth 2.0 client configuration to Web Editor.

    Reads from environment variables (e.g. Render Dashboard or .env).
    Does NOT expose any secrets (SPAs use PKCE public client flow).
    """
    return {
        "osm_client_id": settings.osm_client_id or "",
        "osm_scope": settings.osm_scope,
    }
