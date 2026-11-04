"""Unit tests for the LandAssist FastAPI backend."""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_healthz_returns_200_and_version() -> None:
    """Verify GET /healthz reports status ok and service version 0.1.0."""
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["default_model_version"] == "v1.0.0"
    assert payload["database_mode"] in ("postgis", "demo_fallback")


def test_get_versions_returns_available_model_versions() -> None:
    """Verify GET /v1/versions lists model versions for rollback and A/B testing."""
    response = client.get("/v1/versions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model_version"] == "v1.0.0"
    assert isinstance(payload["versions"], list)
    assert len(payload["versions"]) >= 1
    assert payload["versions"][0]["model_version"] == "v1.0.0"


def test_landuse_geojson_returns_valid_osm_feature_collection() -> None:
    """Verify GET /v1/landuse.geojson returns valid GeoJSON FeatureCollection with OSM tags."""
    response = client.get("/v1/landuse.geojson")
    assert response.status_code == 200
    assert "application/geo+json" in response.headers["content-type"]

    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["crs"] == "EPSG:4326"
    assert len(data["features"]) >= 2

    classes_seen = set()
    for feature in data["features"]:
        assert feature["type"] == "Feature"
        props = feature["properties"]
        assert props["landuse"] in ("industrial", "residential")
        assert props["source"] == "LandAssist AI (Sentinel-2 + GSE)"
        assert props["la:version"] == "v1.0.0"
        assert 0.0 <= props["la:confidence"] <= 1.0
        assert props["area_m2"] > 0

        geom = feature["geometry"]
        assert geom["type"] in ("Polygon", "MultiPolygon")
        classes_seen.add(props["landuse"])

    assert "industrial" in classes_seen
    assert "residential" in classes_seen


def test_landuse_geojson_bbox_and_class_filtering() -> None:
    """Verify bbox and landuse query parameters filter Kabupaten Tangerang features."""
    response = client.get(
        "/v1/landuse.geojson?bbox=106.45,-6.26,106.56,-6.20&landuse=industrial"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) >= 1
    for feature in data["features"]:
        assert feature["properties"]["landuse"] == "industrial"


def test_landuse_geojson_invalid_bbox_returns_400() -> None:
    """Verify malformed or inverted bbox parameters raise HTTP 400."""
    resp_malformed = client.get("/v1/landuse.geojson?bbox=invalid")
    assert resp_malformed.status_code == 400

    resp_inverted = client.get("/v1/landuse.geojson?bbox=106.70,-6.20,106.40,-6.30")
    assert resp_inverted.status_code == 400


def test_cors_headers_for_rapid_editor() -> None:
    """Verify CORS headers allow https://rapideditor.org to fetch GeoJSON layers."""
    response = client.get(
        "/v1/landuse.geojson",
        headers={"Origin": "https://rapideditor.org"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


def test_mvt_tile_endpoint() -> None:
    """Verify GET /v1/tiles/{z}/{x}/{y}.pbf returns Mapbox Vector Tile content-type."""
    response = client.get("/v1/tiles/14/13038/8476.pbf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
    assert "max-age=300" in response.headers.get("cache-control", "")
