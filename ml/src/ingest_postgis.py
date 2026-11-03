"""PostGIS Ingestion Script with Model Versioning and Zero-Downtime Rollback Support.

Loads vectorized LandAssist AI polygons into the `landuse_predictions` PostGIS
table tagged with `model_version`. Supports `--replace-version` to refresh only
a target version while preserving prior versions for instant rollback.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Union

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DDL_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS postgis;",
    """
    CREATE TABLE IF NOT EXISTS landuse_predictions (
        id BIGSERIAL PRIMARY KEY,
        landuse VARCHAR(32) NOT NULL,
        class_id SMALLINT NOT NULL,
        model_version VARCHAR(32) NOT NULL,
        confidence REAL NOT NULL,
        area_m2 DOUBLE PRECISION NOT NULL,
        source VARCHAR(128) NOT NULL DEFAULT 'LandAssist AI (Sentinel-2 + GSE)',
        geom GEOMETRY(MultiPolygon, 4326) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_landuse_predictions_geom
    ON landuse_predictions USING GIST (geom);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_landuse_predictions_version
    ON landuse_predictions (model_version, landuse);
    """,
]


def _promote_to_multipolygon(geom: Union[Polygon, MultiPolygon]) -> MultiPolygon:
    """Ensure geometry is a `MultiPolygon` for strict PostGIS schema compatibility."""
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom


def ensure_postgis_schema(engine: Engine) -> None:
    """Create the `landuse_predictions` table and spatial GIST indexes if absent."""
    with engine.begin() as conn:
        for stmt in DDL_STATEMENTS:
            conn.execute(text(stmt))


def ingest_gdf_to_postgis(
    gdf: gpd.GeoDataFrame,
    db_url: str,
    model_version: str = "v1.0.0",
    replace_version: bool = True,
    batch_size: int = 500,
) -> int:
    """Ingest a LandAssist `GeoDataFrame` into PostGIS `landuse_predictions`.

    Args:
        gdf: Vectorized polygons in `EPSG:4326`.
        db_url: SQLAlchemy database URL (`postgresql+psycopg2://...`).
        model_version: Version tag (`v1.0.0`) stored on every row.
        replace_version: If `True`, atomically deletes existing rows matching
            `model_version` before inserting the fresh predictions, leaving all
            other versions intact for instant rollback.
        batch_size: Number of polygons inserted per SQL batch.

    Returns:
        Number of inserted polygon rows.
    """
    engine = create_engine(db_url, pool_pre_ping=True)
    ensure_postgis_schema(engine)

    if gdf.empty:
        if replace_version:
            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM landuse_predictions WHERE model_version = :ver"),
                    {"ver": model_version},
                )
        return 0

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        mpoly = _promote_to_multipolygon(geom)
        records.append(
            {
                "landuse": str(row.get("landuse", "residential")),
                "class_id": int(row.get("class_id", 2)),
                "model_version": str(row.get("la:version", model_version)),
                "confidence": float(row.get("la:confidence", 0.5)),
                "area_m2": float(row.get("area_m2", 0.0)),
                "source": str(
                    row.get("source", "LandAssist AI (Sentinel-2 + GSE)")
                ),
                "wkt": mpoly.wkt,
            }
        )

    insert_sql = text(
        """
        INSERT INTO landuse_predictions
            (landuse, class_id, model_version, confidence, area_m2, source, geom)
        VALUES
            (:landuse, :class_id, :model_version, :confidence, :area_m2, :source,
             ST_GeomFromText(:wkt, 4326))
        """
    )

    with engine.begin() as conn:
        if replace_version:
            conn.execute(
                text("DELETE FROM landuse_predictions WHERE model_version = :ver"),
                {"ver": model_version},
            )
        for idx in range(0, len(records), batch_size):
            conn.execute(insert_sql, records[idx : idx + batch_size])

    return len(records)


def ingest_geojson_to_postgis(
    geojson_path: Union[str, Path],
    db_url: str,
    model_version: str = "v1.0.0",
    replace_version: bool = True,
    batch_size: int = 500,
) -> int:
    """Read a `.geojson` or `.gpkg` file and ingest into PostGIS."""
    gdf = gpd.read_file(Path(geojson_path))
    return ingest_gdf_to_postgis(
        gdf=gdf,
        db_url=db_url,
        model_version=model_version,
        replace_version=replace_version,
        batch_size=batch_size,
    )


def main() -> None:
    """CLI entrypoint for ingesting vectorized LandAssist predictions into PostGIS."""
    parser = argparse.ArgumentParser(
        description="Ingest LandAssist GeoJSON/GPKG polygons into PostGIS with version tagging."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to vectorized .geojson or .gpkg file.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg2://landassist:landassist@localhost:5432/landassist",
        ),
        help="SQLAlchemy PostGIS connection string (defaults to $DATABASE_URL).",
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
