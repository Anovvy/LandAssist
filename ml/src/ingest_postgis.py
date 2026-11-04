"""PostGIS Ingestion Script using asyncpg (Native FastAPI & Supabase Driver).

Loads vectorized LandAssist AI polygons into the `landuse_predictions` PostGIS
table tagged with `model_version`. Supports `--replace-version` to refresh only
a target version while preserving prior versions for instant rollback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv
from shapely.geometry import MultiPolygon, Polygon, shape

load_dotenv()

DDL_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS postgis;",
    """
    CREATE TABLE IF NOT EXISTS landuse_predictions (
        id SERIAL PRIMARY KEY,
        landuse_class VARCHAR(32) NOT NULL CHECK (landuse_class IN ('industrial', 'residential')),
        confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
        area_m2 DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (area_m2 >= 0.0),
        model_version VARCHAR(32) NOT NULL DEFAULT 'v1.0.0',
        aoi_source VARCHAR(64) NOT NULL DEFAULT 'kab_tangerang_2025',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        geom GEOMETRY(MultiPolygon, 4326) NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_landuse_predictions_geom
    ON landuse_predictions USING GIST (geom);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_landuse_predictions_version_class
    ON landuse_predictions (model_version, landuse_class);
    """,
]


def _promote_to_multipolygon(geom: Polygon | MultiPolygon) -> MultiPolygon:
    """Ensure geometry is a `MultiPolygon` for strict PostGIS schema compatibility."""
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom


async def ensure_postgis_schema(conn: asyncpg.Connection) -> None:
    """Create the `landuse_predictions` table and spatial GIST indexes if absent."""
    for stmt in DDL_STATEMENTS:
        await conn.execute(stmt)


async def ingest_features_to_postgis(
    features: list[dict[str, Any]],
    db_url: str,
    model_version: str = "v1.0.0",
    replace_version: bool = True,
    batch_size: int = 500,
) -> int:
    """Ingest GeoJSON features into Supabase PostGIS via asyncpg."""
    # Normalize postgresql+psycopg2 to standard postgresql
    if db_url.startswith("postgresql+psycopg2://"):
        db_url = db_url.replace("postgresql+psycopg2://", "postgresql://", 1)

    conn = await asyncpg.connect(db_url, statement_cache_size=0)
    try:
        await ensure_postgis_schema(conn)

        async with conn.transaction():
            if replace_version:
                deleted = await conn.execute(
                    "DELETE FROM landuse_predictions WHERE model_version = $1;",
                    model_version,
                )
                print(f"[PostGIS] Cleaned prior records: {deleted}")

            if not features:
                return 0

            records: list[tuple[Any, ...]] = []
            for feat in features:
                props = feat.get("properties", {})
                raw_geom = feat.get("geometry")
                if not raw_geom:
                    continue

                geom_obj = shape(raw_geom)
                if geom_obj.is_empty:
                    continue

                mpoly = _promote_to_multipolygon(geom_obj)
                raw_landuse = str(
                    props.get("landuse", props.get("landuse_class", "residential"))
                ).lower()
                if raw_landuse not in ("industrial", "residential"):
                    raw_landuse = "residential"

                confidence = float(
                    props.get("la:confidence", props.get("confidence", 0.85))
                )
                area_m2 = float(props.get("area_m2", 0.0))
                aoi_source = str(props.get("aoi_source", "kab_tangerang_2025"))

                records.append(
                    (
                        raw_landuse,
                        confidence,
                        area_m2,
                        model_version,
                        aoi_source,
                        mpoly.wkt,
                    )
                )

            # Insert in batches using ST_GeomFromText
            insert_sql = """
                INSERT INTO landuse_predictions
                    (landuse_class, confidence, area_m2, model_version, aoi_source, geom)
                VALUES
                    ($1, $2, $3, $4, $5, ST_GeomFromText($6, 4326));
            """
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                await conn.executemany(insert_sql, batch)

            return len(records)
    finally:
        await conn.close()


def ingest_geojson_to_postgis(
    geojson_path: str | Path,
    db_url: str,
    model_version: str = "v1.0.0",
    replace_version: bool = True,
    batch_size: int = 500,
) -> int:
    """Read a `.geojson` file and ingest into PostGIS using asyncpg."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    return asyncio.run(
        ingest_features_to_postgis(
            features=features,
            db_url=db_url,
            model_version=model_version,
            replace_version=replace_version,
            batch_size=batch_size,
        )
    )


def main() -> None:
    """CLI entrypoint for ingesting vectorized LandAssist predictions into PostGIS."""
    parser = argparse.ArgumentParser(
        description="Ingest LandAssist GeoJSON polygons into PostGIS with version tagging."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to vectorized .geojson file.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/postgres",
        ),
        help="PostGIS connection string (defaults to $DATABASE_URL).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0.0",
        help="Model version tag for rollback-safe partitioning.",
    )
    parser.add_argument(
        "--replace-version",
        action="store_true",
        default=True,
        help="Replace existing rows for this model_version while preserving older versions.",
    )
    parser.add_argument(
        "--no-replace-version",
        action="store_false",
        dest="replace_version",
        help="Append rows without deleting existing rows for this model_version.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of rows per batch INSERT.",
    )

    args = parser.parse_args()
    inserted = ingest_geojson_to_postgis(
        geojson_path=args.input,
        db_url=args.db_url,
        model_version=args.model_version,
        replace_version=args.replace_version,
        batch_size=args.batch_size,
    )
    print(
        f"[OK] Ingested {inserted} polygons into PostGIS landuse_predictions "
        f"(model_version={args.model_version}, replace_version={args.replace_version})."
    )


if __name__ == "__main__":
    main()
