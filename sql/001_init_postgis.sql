-- =============================================================================
-- LandAssist PostGIS Schema Initialization
-- Target: PostgreSQL 16 + PostGIS 3.4
-- Region: Kabupaten Tangerang, Banten, Indonesia (EPSG:4326)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

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

-- GIST spatial index for fast bounding-box (ST_Intersects / &&) and MVT tile queries
CREATE INDEX IF NOT EXISTS idx_landuse_predictions_geom
    ON landuse_predictions USING GIST (geom);

-- Composite B-tree index for instant model version filtering and rollback
CREATE INDEX IF NOT EXISTS idx_landuse_predictions_version_class
    ON landuse_predictions (model_version, landuse_class);

-- =============================================================================
-- Helper Function: Generate Mapbox Vector Tile (MVT) for z/x/y & model_version
-- Used by FastAPI GET /v1/tiles/{z}/{x}/{y}.pbf and pg_tileserv / Martin
-- =============================================================================
CREATE OR REPLACE FUNCTION get_landuse_mvt(
    z INTEGER,
    x INTEGER,
    y INTEGER,
    target_version VARCHAR DEFAULT 'v1.0.0'
)
RETURNS BYTEA
LANGUAGE plpgsql
STABLE
PARALLEL SAFE
AS $$
DECLARE
    mvt BYTEA;
BEGIN
    WITH tile_bounds AS (
        SELECT ST_TileEnvelope(z, x, y) AS env_3857,
               ST_Transform(ST_TileEnvelope(z, x, y), 4326) AS env_4326
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
        WHERE p.model_version = target_version
          AND ST_Intersects(p.geom, tb.env_4326)
    )
    SELECT ST_AsMVT(mvt_geom.*, 'landuse', 4096, 'geom')
    INTO mvt
    FROM mvt_geom
    WHERE geom IS NOT NULL;

    RETURN COALESCE(mvt, ''::bytea);
END;
$$;
