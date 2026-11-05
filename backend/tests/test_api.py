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
    assert payload["default_model_version"] == "v1.1.0"
    assert payload["database_mode"] in ("postgis", "demo_fallback")


def test_get_versions_returns_available_model_versions() -> None:
    """Verify GET /v1/versions lists model versions for rollback and A/B testing."""
    response = client.get("/v1/versions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model_version"] == "v1.1.0"
    assert isinstance(payload["versions"], list)
    assert len(payload["versions"]) >= 1
    assert payload["versions"][0]["model_version"] in ("v1.1.0", "v1.0.0")


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
        assert props["la:version"] in ("v1.1.0", "v1.0.0")
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


def test_editor_endpoint_returns_200() -> None:
    """Verify GET /editor/ serves the LandAssist Web Editor HTML entrypoint."""
    response = client.get("/editor/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "LandAssist" in response.text


def test_auth_config_endpoint_returns_200() -> None:
    """Verify GET /v1/auth/config returns dynamic OAuth 2.0 configuration structure."""
    response = client.get("/v1/auth/config")
    assert response.status_code == 200
    payload = response.json()
    assert "osm_client_id" in payload
    assert "osm_scope" in payload
    assert "write_api" in payload["osm_scope"]



def test_prediction_lifecycle_stage_revert_reject() -> None:
    """Verify prediction lifecycle states (stage -> revert -> reject)."""
    target_id = 101
    # 1. Stage prediction
    resp_stage = client.post(f"/v1/predictions/{target_id}/stage?user=tester_user")
    assert resp_stage.status_code == 200
    assert resp_stage.json()["lifecycle"] == "staged"

    # Verify status in geojson
    resp_all = client.get("/v1/landuse.geojson?status=all")
    p1 = next((f for f in resp_all.json()["features"] if f["properties"].get("la:id") == target_id), None)
    if p1:
        assert p1["properties"]["la:status"] == "staged"

    # 2. Revert prediction back to unreviewed (Ctrl+Z)
    resp_revert = client.post(f"/v1/predictions/{target_id}/revert")
    assert resp_revert.status_code == 200
    assert resp_revert.json()["lifecycle"] == "unreviewed"

    # 3. Reject prediction
    resp_reject = client.post(f"/v1/predictions/{target_id}/reject?user=tester_user")
    assert resp_reject.status_code == 200
    assert resp_reject.json()["lifecycle"] == "rejected"

    # Reset back to unreviewed for clean state
    client.post(f"/v1/predictions/{target_id}/revert")


def test_prediction_batch_commit() -> None:
    """Verify batch commit marks predictions as accepted with osm_id."""
    payload = {
        "items": [{"id": 101, "osm_id": 999001}, {"id": 102, "osm_id": 999002}],
        "reviewed_by": "osm_validator_user",
    }
    resp = client.post("/v1/predictions/batch-commit", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["committed_count"] >= 1

    # Reset back to unreviewed
    client.post("/v1/predictions/101/revert")
    client.post("/v1/predictions/102/revert")
