# ruff: noqa: E402
"""Comprehensive Unit Tests for LandAssist ML & Geospatial Pipeline.

Tests:
1. 64-Channel SegFormer model instantiation & patch embedding adaptation
2. WeightedCE + Multiclass Foreground Dice Loss
3. 2D Hann Window blending & sliding-window grid generation
4. Chaikin corner-cutting & metric polygon smoothing
5. Probability map discretization, GDAL sieve, & OSM GeoJSON vectorization
6. MultiPolygon promotion & topological validity
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, Polygon

from ml.src.infer_tangerang import _generate_tile_Grid, build_2d_hann_window
from ml.src.ingest_postgis import _promote_to_multipolygon
from ml.src.model import (
    WeightedCEDiceLoss,
    build_segformer_64ch,
    resolve_checkpoint_path,
)
from ml.src.vectorize import (
    chaikin_smooth_ring,
    prob_map_to_osm_polygons,
    smooth_and_simplify_polygon,
)


def test_build_segformer_64ch_architecture() -> None:
    """Verify SegFormer MiT-B2 config adapts from 3 to 64 GSE channels."""
    model = build_segformer_64ch(num_classes=3, in_channels=64, pretrained=False)
    assert model.config.num_channels == 64
    assert model.config.num_labels == 3

    # Inspect first patch embedding conv projection
    segformer = model.segformer
    if hasattr(segformer, "stages"):
        proj = segformer.stages[0].patch_embeddings.proj
    else:
        proj = segformer.encoder.patch_embeddings[0].proj
    assert proj.in_channels == 64

    # Dummy forward pass (B=1, C=64, H=160, W=160)
    dummy_input = torch.randn(1, 64, 160, 160, dtype=torch.float32)
    with torch.no_grad():
        out = model(pixel_values=dummy_input)
    logits = out.logits
    assert logits.shape[1] == 3  # 3 output semantic classes


def test_weighted_ce_dice_loss() -> None:
    """Verify combined Weighted Cross-Entropy and Foreground Dice loss calculation."""
    criterion = WeightedCEDiceLoss(class_weights=(0.5, 3.5, 1.0))
    dummy_logits = torch.randn(2, 3, 40, 40, dtype=torch.float32, requires_grad=True)
    dummy_targets = torch.randint(0, 3, (2, 160, 160), dtype=torch.long)

    loss = criterion(dummy_logits, dummy_targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert dummy_logits.grad is not None


def test_build_2d_hann_window() -> None:
    """Verify 2D Hann window dimensions, boundaries, and central peak."""
    hann = build_2d_hann_window(tile_size=160, min_weight=0.05)
    assert hann.shape == (160, 160)
    assert hann.dtype == np.float32
    assert np.isclose(hann[80, 80], 1.0, atol=1e-2)
    assert hann[0, 0] >= 0.05
    assert hann.min() >= 0.05
    assert hann.max() <= 1.0


def test_generate_tile_grid() -> None:
    """Verify sliding window tile grid coordinates cover the full raster."""
    coords = _generate_tile_Grid(height=320, width=320, tile_size=160, overlap=80)
    assert len(coords) > 0
    assert (0, 0) in coords
    # Check that bottom-right tile is reached
    assert any(r + 160 == 320 and c + 160 == 320 for r, c in coords)


def test_chaikin_smooth_ring() -> None:
    """Verify Chaikin corner-cutting produces smooth closed rings."""
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    smoothed = chaikin_smooth_ring(square, iterations=2)
    assert len(smoothed) > len(square)
    # Check closed ring
    assert np.allclose(smoothed[0], smoothed[-1])


def test_smooth_and_simplify_polygon() -> None:
    """Verify metric polygon smoothing and Douglas-Peucker simplification."""
    square = Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)])
    smoothed = smooth_and_simplify_polygon(
        square,
        simplify_tol_m=2.0,
        min_hole_area_m2=500.0,
        chaikin_iterations=2,
    )
    assert smoothed.is_valid
    assert not smoothed.is_empty
    assert smoothed.area > 5000.0


def test_prob_map_to_osm_polygons() -> None:
    """Verify conversion of (3, H, W) probability map to OSM-tagged GeoDataFrame."""
    H, W = 160, 160
    # Background=0, Industrial=1 (center block), Residential=2 (surrounding)
    prob_mmap = np.zeros((3, H, W), dtype=np.float16)
    prob_mmap[0, :, :] = 0.8
    # Add industrial zone (40x40 pixels = 1,600 px >> 16 px sieve threshold)
    prob_mmap[1, 40:80, 40:80] = 0.85
    prob_mmap[0, 40:80, 40:80] = 0.10

    # Add residential zone (40x40 pixels)
    prob_mmap[2, 90:130, 90:130] = 0.90
    prob_mmap[0, 90:130, 90:130] = 0.05

    transform = from_bounds(106.60, -6.22, 106.62, -6.20, W, H)
    gdf = prob_map_to_osm_polygons(
        prob_mmap=prob_mmap,
        transform=transform,
        crs="EPSG:4326",
        model_version="v1.0.0",
        min_pixels=16,
    )

    assert not gdf.empty
    assert len(gdf) >= 2
    assert "landuse" in gdf.columns
    assert "la:version" in gdf.columns
    assert "la:confidence" in gdf.columns
    assert "area_m2" in gdf.columns
    assert set(gdf["landuse"].unique()).issubset({"industrial", "residential"})

    for geom in gdf.geometry:
        assert geom.is_valid
        assert not geom.is_empty


def test_promote_to_multipolygon() -> None:
    """Verify single Polygon is promoted to MultiPolygon and invalid geoms repaired."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    mp = _promote_to_multipolygon(poly)
    assert isinstance(mp, MultiPolygon)
    assert mp.is_valid

    # Self-intersecting bowtie polygon
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    mp_repaired = _promote_to_multipolygon(bowtie)
    assert isinstance(mp_repaired, MultiPolygon)
    assert mp_repaired.is_valid


def test_resolve_checkpoint_path() -> None:
    """Verify resolve_checkpoint_path finds v1.0.0 from manifest.json."""
    ckpt_path = resolve_checkpoint_path(model_version="v1.0.0")
    assert ckpt_path.exists()
    assert ckpt_path.name == "segformer-mitb2-gse64-v1.0.0.pt"
