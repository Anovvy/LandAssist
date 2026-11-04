"""Health check, interactive Visual QA map, and model version inventory endpoints."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import settings
from app.db import db

router = APIRouter(tags=["health & versions"])
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
QA_HTML_PATH = REPO_ROOT / "outputs" / "tangerang_v1.0.0_qa.html"


@router.get("/healthz")
@router.get("/v1/health")
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


@router.get("/qa", response_class=HTMLResponse)
async def visual_qa_map() -> HTMLResponse:
    """Serve the interactive Kabupaten Tangerang Satellite Visual QA Map directly in the browser."""
    if QA_HTML_PATH.exists():
        return HTMLResponse(content=QA_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        content=(
            "<html><body style='font-family:sans-serif;padding:2rem;'>"
            "<h2>LandAssist Visual QA Map</h2>"
            "<p>Run <code>python -m ml.src.vectorize --html-preview outputs/tangerang_v1.0.0_qa.html</code> "
            "to generate the full satellite QA overlay, or query <a href='/v1/landuse.geojson'>/v1/landuse.geojson</a>.</p>"
            "</body></html>"
        )
    )
