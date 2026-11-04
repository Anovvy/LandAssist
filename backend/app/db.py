"""Async PostGIS connection pool manager with Kabupaten Tangerang fallback dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_GEOJSON_PATH = REPO_ROOT / "outputs" / "tangerang_v1.0.0.geojson"

# Built-in realistic demo polygons for Kabupaten Tangerang (EPSG:4326)
# Enables zero-config CI testing, local development, and offline Rapid Editor demos
DEMO_FEATURES: list[dict[str, Any]] = [
    {
        "id": 101,
        "landuse_class": "industrial",
        "confidence": 0.94,
        "area_m2": 48250.0,
        "model_version": "v1.0.0",
        "aoi_source": "kab_tangerang_2025",
        "bbox": (106.4780, -6.2240, 106.4830, -6.2190),
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [106.4780, -6.2240],
                        [106.4830, -6.2240],
                        [106.4830, -6.2190],
                        [106.4780, -6.2190],
                        [106.4780, -6.2240],
                    ]
                ]
            ],
        },
    },
    {
        "id": 102,
        "landuse_class": "industrial",
        "confidence": 0.89,
        "area_m2": 36120.5,
        "model_version": "v1.0.0",
        "aoi_source": "kab_tangerang_2025",
        "bbox": (106.4910, -6.2160, 106.4955, -6.2115),
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [106.4910, -6.2160],
                        [106.4955, -6.2160],
                        [106.4955, -6.2115],
                        [106.4910, -6.2115],
                        [106.4910, -6.2160],
                    ]
                ]
            ],
        },
    },
    {
        "id": 103,
        "landuse_class": "residential",
        "confidence": 0.92,
        "area_m2": 29400.0,
        "model_version": "v1.0.0",
        "aoi_source": "kab_tangerang_2025",
        "bbox": (106.5260, -6.2530, 106.5310, -6.2485),
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [106.5260, -6.2530],
                        [106.5310, -6.2530],
                        [106.5310, -6.2485],
                        [106.5260, -6.2485],
                        [106.5260, -6.2530],
                    ]
                ]
            ],
        },
    },
    {
        "id": 104,
        "landuse_class": "residential",
        "confidence": 0.87,
        "area_m2": 21850.0,
        "model_version": "v1.0.0",
        "aoi_source": "kab_tangerang_2025",
        "bbox": (106.5380, -6.2420, 106.5425, -6.2380),
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [106.5380, -6.2420],
                        [106.5425, -6.2420],
                        [106.5425, -6.2380],
                        [106.5380, -6.2380],
                        [106.5380, -6.2420],
                    ]
                ]
            ],
        },
    },
]


def _compute_geometry_bbox(geom: dict[str, Any]) -> tuple[float, float, float, float]:
    """Compute `(min_lon, min_lat, max_lon, max_lat)` for a Polygon or MultiPolygon GeoJSON dict."""
    lons: list[float] = []
    lats: list[float] = []
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            for pt in ring:
                lons.append(float(pt[0]))
                lats.append(float(pt[1]))
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for pt in ring:
                    lons.append(float(pt[0]))
                    lats.append(float(pt[1]))
    if not lons or not lats:
        return (106.45, -6.25, 106.55, -6.18)
    return (min(lons), min(lats), max(lons), max(lats))


def _load_local_tangerang_features() -> list[dict[str, Any]]:
    """Load real vectorized Kabupaten Tangerang polygons from `outputs/tangerang_v1.0.0.geojson` if present."""
    if not LOCAL_GEOJSON_PATH.exists():
        return []
    try:
        raw = json.loads(LOCAL_GEOJSON_PATH.read_text(encoding="utf-8"))
        loaded: list[dict[str, Any]] = []
        for idx, feat in enumerate(raw.get("features", []), start=1000):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            if not geom:
                continue
            loaded.append(
                {
                    "id": idx,
                    "landuse_class": str(props.get("landuse", "residential")),
                    "confidence": float(props.get("la:confidence", 0.75)),
                    "area_m2": float(props.get("area_m2", 1600.0)),
                    "model_version": str(props.get("la:version", "v1.0.0")),
                    "aoi_source": "kab_tangerang_2025",
                    "bbox": _compute_geometry_bbox(geom),
                    "geometry": geom,
                }
            )
        return loaded
    except (OSError, ValueError, KeyError):
        return []


DEMO_FEATURES.extend(_load_local_tangerang_features())


class DatabaseManager:
    """Manages the asyncpg connection pool and executes spatial PostGIS queries."""

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    @property
    def is_connected(self) -> bool:
        return self.pool is not None

    async def connect(self) -> None:
        """Initialize the PostGIS connection pool if DATABASE_URL is configured."""
        if not settings.database_url:
            logger.info(
                "No DATABASE_URL configured; running in demo_fallback mode with Kabupaten Tangerang polygons."
            )
            return
        try:
            pool_kwargs: dict[str, Any] = {
                "min_size": 1,
                "max_size": 10,
                "command_timeout": 30.0,
            }
            # Supavisor transaction pooler on port 6543 requires statement_cache_size=0
            if ":6543" in settings.database_url or "pooler.supabase.com" in settings.database_url:
                pool_kwargs["statement_cache_size"] = 0

            self.pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                **pool_kwargs,
            )
            logger.info("Connected to PostGIS database pool.")
        except (OSError, asyncpg.PostgresError, ValueError) as exc:
            logger.warning(
                "Could not connect to PostGIS (%s); falling back to in-memory demo dataset.",
                exc,
            )
            self.pool = None

    async def disconnect(self) -> None:
        """Close the PostGIS connection pool cleanly."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def list_available_versions(self) -> list[dict[str, Any]]:
        """Return available model versions and polygon counts for rollback / A/B testing."""
        if self.pool is not None:
            query = """
                SELECT
                    model_version,
                    COUNT(*)::int AS feature_count,
                    ROUND(AVG(confidence)::numeric, 4)::float AS avg_confidence,
                    MAX(created_at)::text AS latest_created_at
                FROM landuse_predictions
                GROUP BY model_version
                ORDER BY latest_created_at DESC;
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(r) for r in rows]

        return [
            {
                "model_version": "v1.0.0",
                "feature_count": len(DEMO_FEATURES),
                "avg_confidence": round(
                    sum(f["confidence"] for f in DEMO_FEATURES) / len(DEMO_FEATURES), 4
                ),
                "latest_created_at": "2025-05-01T00:00:00Z",
            }
        ]

    async def query_geojson_features(
        self,
        bbox: tuple[float, float, float, float],
        model_version: str,
        landuse: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Query OSM-tagged GeoJSON Features within a WGS84 bounding box."""
        min_lon, min_lat, max_lon, max_lat = bbox

        if self.pool is not None:
            query = """
                SELECT
                    id,
                    landuse_class,
                    confidence,
                    area_m2,
                    model_version,
                    ST_AsGeoJSON(geom, 6) AS geom_json
                FROM landuse_predictions
                WHERE model_version = $5
                  AND confidence >= $6
                  AND ($7::varchar IS NULL OR landuse_class = $7)
                  AND ST_Intersects(
                      geom,
                      ST_MakeEnvelope($1, $2, $3, $4, 4326)
                  )
                ORDER BY id
                LIMIT $8;
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    query,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    model_version,
                    min_confidence,
                    landuse,
                    limit,
                )

            features: list[dict[str, Any]] = []
            for row in rows:
                features.append(
                    {
                        "type": "Feature",
                        "id": f"landassist_{row['id']}",
                        "properties": {
                            "landuse": row["landuse_class"],
                            "source": "LandAssist AI (Sentinel-2 + GSE)",
                            "la:version": row["model_version"],
                            "la:confidence": round(float(row["confidence"]), 4),
                            "area_m2": round(float(row["area_m2"]), 1),
                        },
                        "geometry": json.loads(row["geom_json"]),
                    }
                )
            return features

        # Fallback mode: filter built-in Kabupaten Tangerang polygons
        features = []
        for item in DEMO_FEATURES:
            if item["model_version"] != model_version:
                continue
            if item["confidence"] < min_confidence:
                continue
            if landuse and item["landuse_class"] != landuse:
                continue
            f_min_lon, f_min_lat, f_max_lon, f_max_lat = item["bbox"]
            intersects = not (
                f_max_lon < min_lon
                or f_min_lon > max_lon
                or f_max_lat < min_lat
                or f_min_lat > max_lat
            )
            if not intersects:
                continue

            features.append(
                {
                    "type": "Feature",
                    "id": f"landassist_{item['id']}",
                    "properties": {
                        "landuse": item["landuse_class"],
                        "source": "LandAssist AI (Sentinel-2 + GSE)",
                        "la:version": item["model_version"],
                        "la:confidence": round(float(item["confidence"]), 4),
                        "area_m2": round(float(item["area_m2"]), 1),
                    },
                    "geometry": item["geometry"],
                }
            )
            if len(features) >= limit:
                break

        return features

    async def query_mvt_tile(
        self,
        z: int,
        x: int,
        y: int,
        model_version: str,
    ) -> bytes:
        """Generate a Mapbox Vector Tile (.pbf) directly from PostGIS via ST_AsMVT."""
        if self.pool is not None:
            query = """
                WITH tile_bounds AS (
                    SELECT ST_TileEnvelope($1, $2, $3) AS env_3857,
                           ST_Transform(ST_TileEnvelope($1, $2, $3), 4326) AS env_4326
                ),
                mvt_geom AS (
                    SELECT
                        p.id,
                        p.landuse_class AS landuse,
                        'LandAssist AI (Sentinel-2 + GSE)' AS source,
                        p.model_version AS "la:version",
                        ROUND(p.confidence::numeric, 4) AS "la:confidence",
                        ROUND(p.area_m2::numeric, 1) AS area_m2,
                        ST_AsMVTGeom(
                            ST_Transform(p.geom, 3857),
                            tb.env_3857,
                            4096,
                            256,
                            true
                        ) AS geom
                    FROM landuse_predictions p
                    CROSS JOIN tile_bounds tb
                    WHERE p.model_version = $4
                      AND ST_Intersects(p.geom, tb.env_4326)
                )
                SELECT COALESCE(ST_AsMVT(mvt_geom.*, 'landuse', 4096, 'geom'), ''::bytea) AS mvt
                FROM mvt_geom
                WHERE geom IS NOT NULL;
            """
            async with self.pool.acquire() as conn:
                val = await conn.fetchval(query, z, x, y, model_version)
                return bytes(val) if val else b""

        return b""


db = DatabaseManager()
