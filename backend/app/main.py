"""FastAPI entrypoint for the LandAssist OSM Rapid API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import db
from app.routers import auth, geojson, health, predictions, tiles

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
if not FRONTEND_DIST.exists():
    # Support containerized environment where WORKDIR is /app
    _app_frontend = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if _app_frontend.exists():
        FRONTEND_DIST = _app_frontend


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of the asyncpg PostGIS connection pool."""
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production spatial backend serving Sentinel-2 + Google Satellite Embedding (GSE) "
        "land-use predictions (`landuse=industrial` & `landuse=residential`) in Kabupaten "
        "Tangerang directly to OpenStreetMap Rapid Editor and JOSM."
    ),
    lifespan=lifespan,
)

# Permissive CORS middleware so web-based OSM editors (https://rapideditor.org)
# can fetch custom GeoJSON and MVT layers directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(geojson.router)
app.include_router(tiles.router)
app.include_router(auth.router)
app.include_router(predictions.router)

# Mount custom Rapid Editor frontend directly under /editor
if FRONTEND_DIST.exists():
    app.mount("/editor", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="editor")
