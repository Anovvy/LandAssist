"""Mapbox Vector Tile (MVT) endpoint powered by PostGIS ST_AsMVT."""

from fastapi import APIRouter, HTTPException, Path, Query, Response

from app.config import settings
from app.db import db

router = APIRouter(prefix="/v1", tags=["vector-tiles"])


@router.get("/tiles/{z}/{x}/{y}.pbf")
async def get_mvt_tile(
    z: int = Path(..., ge=0, le=22, description="Tile zoom level (0-22)"),
    x: int = Path(..., ge=0, description="Tile X coordinate"),
    y: int = Path(..., ge=0, description="Tile Y coordinate"),
    version: str = Query(
        default=settings.default_model_version,
        description="Model prediction version tag",
    ),
) -> Response:
    """Serve Mapbox Vector Tiles (.pbf) generated on-the-fly by PostGIS ST_AsMVT."""
    max_coord = (1 << z) - 1
    if x > max_coord or y > max_coord:
        raise HTTPException(
            status_code=400,
            detail=f"Tile coordinates ({x}, {y}) out of bounds for zoom level {z}.",
        )

    tile_bytes = await db.query_mvt_tile(z=z, x=x, y=y, model_version=version)
    return Response(
        content=tile_bytes,
        media_type="application/vnd.mapbox-vector-tile",
        headers={"Cache-Control": "public, max-age=300"},
    )
