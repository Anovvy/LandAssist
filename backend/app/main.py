"""FastAPI entrypoint for the LandAssist OSM Rapid API."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import db
from app.routers import geojson, health, tiles


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
