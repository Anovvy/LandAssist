"""OSM-compliant GeoJSON FeatureCollection endpoint for Rapid Editor & JOSM."""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import db

router = APIRouter(prefix="/v1", tags=["geojson"])

# Default bounding box covering Kabupaten Tangerang, Banten (EPSG:4326)
KAB_TANGERANG_DEFAULT_BBOX = "106.33,-6.38,106.72,-6.01"


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse and validate a 'minLon,minLat,maxLon,maxLat' WGS84 bounding box."""
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox format. Expected 4 comma-separated floats: 'minLon,minLat,maxLon,maxLat'.",
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox coordinates. All 4 values must be valid floating-point numbers.",
        ) from exc

    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise HTTPException(
            status_code=400,
            detail="Longitude values must be within [-180, 180].",
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise HTTPException(
            status_code=400,
            detail="Latitude values must be within [-90, 90].",
        )
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox bounds: minLon must be < maxLon and minLat must be < maxLat.",
        )
    return min_lon, min_lat, max_lon, max_lat


@router.get("/landuse.geojson")
async def get_landuse_geojson(
    bbox: str = Query(
        default=KAB_TANGERANG_DEFAULT_BBOX,
        description="Bounding box in EPSG:4326 format: minLon,minLat,maxLon,maxLat",
    ),
    version: str = Query(
        default=settings.default_model_version,
        description="Model prediction version tag (supports instant rollback)",
    ),
    landuse: Literal["industrial", "residential"] | None = Query(
        default=None,
        description="Optional filter by OSM landuse class ('industrial' or 'residential')",
    ),
    min_confidence: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum prediction confidence threshold [0.0, 1.0]",
    ),
) -> JSONResponse:
    """Serve OSM-tagged land-use polygons as a GeoJSON FeatureCollection for Rapid Editor."""
    parsed_bbox = parse_bbox(bbox)
    features: list[dict[str, Any]] = await db.query_geojson_features(
        bbox=parsed_bbox,
        model_version=version,
        landuse=landuse,
        min_confidence=min_confidence,
        limit=settings.max_features_per_request,
    )

    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "region": "Kabupaten Tangerang, Indonesia",
            "crs": "EPSG:4326",
            "model_version": version,
            "feature_count": len(features),
        },
        "features": features,
    }

    return JSONResponse(
        content=feature_collection,
        media_type="application/geo+json",
        headers={"Cache-Control": "public, max-age=60"},
    )
