"""Automated Regional 64-Band Inference & PostGIS Ingestion for Kabupaten Tangerang.

Orchestrates the complete regional pipeline per the `regional-inference-ingest` skill:
1. Queries Google Earth Engine API for the official administrative boundary of Kabupaten Tangerang.
2. Partitions the ~1,193 km² AOI into a spatial grid (e.g. 2x2 or 3x3) to comply with GEE's 48 MB limit.
3. Downloads 64-band Google Satellite Embeddings (GSE) for each grid cell and mosaics them.
4. Executes anti-OOM sliding-window inference with 2D Hann window blending into disk-backed float16 memmaps.
5. Runs GDAL sieve filtering, UTM Chaikin boundary smoothing (EPSG:32748), and AOI spatial clipping.
6. Generates an interactive satellite Visual QA map and ingests into Supabase PostGIS with zero downtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import shape
from shapely.ops import unary_union

from .download_gse import download_gse_64band_geotiff, init_earth_engine
from .infer_tangerang import load_versioned_checkpoint, run_sliding_window_inference
from .ingest_postgis import ingest_geojson_to_postgis
from .vectorize import (
    export_visual_qa_html,
    prob_map_to_osm_polygons,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AOI_PATH = REPO_ROOT / "data" / "aoi_kab_tangerang.geojson"
DEFAULT_CHECKPOINT = REPO_ROOT / "weight" / "segformer-mitb2-gse64-v1.0.0.pt"
KAB_TANGERANG_ADM2_CODE = 18028


def fetch_kab_tangerang_aoi_from_gee(output_path: Path) -> tuple[float, float, float, float]:
    """Fetch Kabupaten Tangerang administrative boundary from GEE and save to GeoJSON."""
    import ee  # type: ignore

    init_earth_engine()
    kab = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.eq("ADM2_CODE", KAB_TANGERANG_ADM2_CODE)
    ).first()

    geom = kab.geometry()
    geom_info = geom.getInfo()

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Kabupaten Tangerang",
                    "province": "Banten",
                    "country": "Indonesia",
                    "adm2_code": KAB_TANGERANG_ADM2_CODE,
                    "source": "FAO/GAUL/2015/level2",
                },
                "geometry": geom_info,
            }
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")
    print(f"[AOI] Saved Kabupaten Tangerang administrative boundary to {output_path}")

    bounds = shape(geom_info).bounds
    return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))


def generate_spatial_grid(
    bbox: tuple[float, float, float, float],
    rows: int = 2,
    cols: int = 2,
    buffer_deg: float = 0.002,
) -> list[dict[str, Any]]:
    """Partition a master bounding box into overlapping sub-grid bounding boxes.

    Args:
        bbox: Master (min_lon, min_lat, max_lon, max_lat).
        rows: Number of grid rows.
        cols: Number of grid columns.
        buffer_deg: Overlap buffer in degrees (~200m at equator) to avoid edge artifacts.

    Returns:
        List of dicts with grid coordinates and cell metadata.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / cols
    lat_step = (max_lat - min_lat) / rows

    grid_cells: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            cell_min_lon = max(min_lon, min_lon + c * lon_step - buffer_deg)
            cell_max_lon = min(max_lon, min_lon + (c + 1) * lon_step + buffer_deg)
            cell_min_lat = max(min_lat, min_lat + r * lat_step - buffer_deg)
            cell_max_lat = min(max_lat, min_lat + (r + 1) * lat_step + buffer_deg)

            grid_cells.append(
                {
                    "row": r,
                    "col": c,
                    "cell_id": f"r{r}_c{c}",
                    "bbox": (cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat),
                }
            )

    return grid_cells


def download_regional_gse_tiles(
    grid_cells: list[dict[str, Any]],
    tiles_dir: Path,
    merged_output: Path,
    target_year: int = 2025,
    force: bool = False,
) -> Path:
    """Download each grid cell's 64-band GSE raster and merge into a single regional GeoTIFF."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_files: list[Path] = []

    print(f"[REGIONAL DOWNLOAD] Downloading {len(grid_cells)} spatial GSE sub-tiles...")
    for cell in grid_cells:
        cell_id = cell["cell_id"]
        tile_path = tiles_dir / f"tile_{cell_id}.tif"
        tile_files.append(tile_path)
        print(f"\n--- Sub-Tile [{cell_id}] BBOX: {cell['bbox']} ---")
        download_gse_64band_geotiff(
            output_path=tile_path,
            bbox=cell["bbox"],
            target_year=target_year,
            force=force,
        )

    # Merge sub-tiles into master regional GeoTIFF
    print(f"\n[MOSAIC] Mosaicking {len(tile_files)} 64-band tiles into {merged_output}...")
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    sources = [rasterio.open(fp) for fp in tile_files]
    try:
        mosaic, out_trans = merge(sources)
        out_meta = sources[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw",
            }
        )
        with rasterio.open(merged_output, "w", **out_meta) as dst:
            dst.write(mosaic)
            for b_idx in range(1, 65):
                dst.set_band_description(b_idx, f"A{b_idx - 1:02d}")

        print(
            f"[MOSAIC COMPLETE] Saved master 64-band GeoTIFF to {merged_output} "
            f"(shape={mosaic.shape}, CRS={out_meta['crs']})."
        )
    finally:
        for s in sources:
            s.close()

    return merged_output


def clip_geojson_to_aoi(
    input_geojson_path: Path,
    aoi_path: Path,
    output_geojson_path: Path,
) -> int:
    """Spatially filter and clip polygons to remain strictly inside Kabupaten Tangerang."""
    with open(aoi_path, "r", encoding="utf-8") as f:
        aoi_data = json.load(f)

    aoi_geom = unary_union(
        [shape(feat["geometry"]) for feat in aoi_data["features"] if feat.get("geometry")]
    )

    with open(input_geojson_path, "r", encoding="utf-8") as f:
        poly_data = json.load(f)

    kept_features: list[dict[str, Any]] = []
    for feat in poly_data.get("features", []):
        raw_geom = feat.get("geometry")
        if not raw_geom:
            continue
        geom = shape(raw_geom)
        if geom.is_empty or not geom.is_valid:
            geom = geom.buffer(0)
        if aoi_geom.intersects(geom):
            # Compute intersection area percentage
            intersection = aoi_geom.intersection(geom)
            if not intersection.is_empty and intersection.area >= 0.5 * geom.area:
                kept_features.append(feat)

    result_data = {
        "type": "FeatureCollection",
        "metadata": {
            **poly_data.get("metadata", {}),
            "clipped_by": str(aoi_path.name),
            "feature_count": len(kept_features),
        },
        "features": kept_features,
    }

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(
        f"[CLIP] Kept {len(kept_features)} of {len(poly_data.get('features', []))} "
        f"polygons inside {aoi_path.name} -> {output_geojson_path}"
    )
    return len(kept_features)


def run_regional_pipeline(
    aoi_geojson: Path = DEFAULT_AOI_PATH,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    target_year: int = 2025,
    grid_rows: int = 2,
    grid_cols: int = 2,
    model_version: str = "v1.1.0",
    skip_download: bool = False,
    skip_infer: bool = False,
    skip_vectorize: bool = False,
    skip_ingest: bool = False,
    temp_dir: Path | None = None,
) -> None:
    """Execute end-to-end regional inference and ingestion for Kabupaten Tangerang."""
    work_temp = temp_dir or (REPO_ROOT / "tmp_mmap_regional")
    work_temp.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "data"
    tiles_dir = data_dir / "tangerang_tiles"
    regional_tif = data_dir / "kab_tangerang_gse64.tif"
    pred_mmap = work_temp / "pred_sum.mmap"

    outputs_dir = REPO_ROOT / "outputs"
    raw_geojson = outputs_dir / f"kab_tangerang_{model_version}_raw.geojson"
    clipped_geojson = outputs_dir / f"kab_tangerang_{model_version}.geojson"
    qa_html = outputs_dir / f"kab_tangerang_{model_version}_qa.html"

    # Step 1: AOI Discovery
    print("=" * 70)
    print("STEP 1: Earth Engine AOI Discovery for Kabupaten Tangerang")
    print("=" * 70)
    if not aoi_geojson.exists():
        bbox = fetch_kab_tangerang_aoi_from_gee(aoi_geojson)
    else:
        with open(aoi_geojson, "r", encoding="utf-8") as f:
            d = json.load(f)
        geom = shape(d["features"][0]["geometry"])
        bbox = (float(geom.bounds[0]), float(geom.bounds[1]), float(geom.bounds[2]), float(geom.bounds[3]))
        print(f"[AOI] Loaded existing AOI from {aoi_geojson}: BBOX={bbox}")

    # Step 2: Download & Mosaic GSE 64-band
    print("\n" + "=" * 70)
    print("STEP 2: 64-Band Google Satellite Embedding (GSE) Acquisition")
    print("=" * 70)
    if not skip_download and not regional_tif.exists():
        grid_cells = generate_spatial_grid(bbox, rows=grid_rows, cols=grid_cols)
        download_regional_gse_tiles(
            grid_cells=grid_cells,
            tiles_dir=tiles_dir,
            merged_output=regional_tif,
            target_year=target_year,
        )
    elif regional_tif.exists():
        print(f"[SKIP DOWNLOAD] Using existing regional GeoTIFF: {regional_tif}")
    else:
        print("[SKIP DOWNLOAD] --skip-download specified.")

    # Step 3: Anti-OOM Sliding-Window Inference
    print("\n" + "=" * 70)
    print("STEP 3: Sliding-Window Inference with 2D Hann Blending")
    print("=" * 70)
    if not skip_infer:
        if not regional_tif.exists():
            raise FileNotFoundError(f"Regional GeoTIFF not found: {regional_tif}")
        model, meta = load_versioned_checkpoint(checkpoint)
        run_sliding_window_inference(
            model=model,
            gse_tif_path=regional_tif,
            temp_dir=work_temp,
            tile_size=160,
            overlap=80,
            batch_size=8,
        )
    else:
        print("[SKIP INFER] Skipping model inference.")

    # Step 4: Sieve + UTM Chaikin Vectorization & AOI Clipping
    print("\n" + "=" * 70)
    print("STEP 4: Post-Processing, Vectorization & Administrative Boundary Clipping")
    print("=" * 70)
    if not skip_vectorize:
        if not pred_mmap.exists():
            raise FileNotFoundError(f"Prediction memmap not found: {pred_mmap}")
        with rasterio.open(regional_tif) as ref_ds:
            height, width = ref_ds.height, ref_ds.width

        prob_map = np.memmap(pred_mmap, dtype="float16", mode="r", shape=(3, height, width))
        prob_map_to_osm_polygons(
            prob_map=prob_map,
            ref_tif_path=regional_tif,
            output_geojson_path=raw_geojson,
            model_version=model_version,
            min_pixels=16,
            sigma=0.8,
            industrial_thresh=0.38,
            residential_thresh=0.45,
            utm_epsg="EPSG:32748",
            aoi_source="kab_tangerang_2025",
        )

        # Clip strictly to Kabupaten Tangerang administrative boundary
        clip_geojson_to_aoi(
            input_geojson_path=raw_geojson,
            aoi_path=aoi_geojson,
            output_geojson_path=clipped_geojson,
        )

        # Generate interactive HTML Visual QA Map
        export_visual_qa_html(
            geojson_path=clipped_geojson,
            out_html_path=qa_html,
        )
        print(f"[QA MAP] Generated interactive satellite QA map at {qa_html}")
    else:
        print("[SKIP VECTORIZE] Skipping vectorization.")

    # Step 5: Supabase PostGIS Zero-Downtime Ingestion
    print("\n" + "=" * 70)
    print("STEP 5: Versioned PostGIS Ingestion & Quality Gate Verification")
    print("=" * 70)
    if not skip_ingest:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            env_file = REPO_ROOT / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DATABASE_URL="):
                        db_url = line.split("=", 1)[1].strip().strip("'").strip('"')
                        break
        if not db_url:
            print("[WARN] DATABASE_URL not found; skipping PostGIS live ingestion.")
        else:
            ingested_count = ingest_geojson_to_postgis(
                geojson_path=clipped_geojson,
                db_url=db_url,
                model_version=model_version,
                replace_version=True,
                batch_size=500,
            )
            print(f"[POSTGIS INGESTED] Successfully stored {ingested_count} polygons in PostGIS (version {model_version}).")
    else:
        print("[SKIP INGEST] Skipping database ingestion.")

    print("\n" + "=" * 70)
    print("🎉 REGIONAL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"• Master GeoJSON : {clipped_geojson}")
    print(f"• Visual QA Map  : {qa_html}")
    print(f"• Model Version  : {model_version}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-End Regional 64-Band GSE Inference & PostGIS Ingestion for Kabupaten Tangerang."
    )
    parser.add_argument(
        "--aoi",
        type=Path,
        default=DEFAULT_AOI_PATH,
        help="Path to Kabupaten Tangerang AOI GeoJSON (auto-downloaded from GEE if missing).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to SegFormer model checkpoint weights.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Target year for GSE representation (default: 2025).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=2,
        help="Number of spatial grid rows (default: 2).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=2,
        help="Number of spatial grid columns (default: 2).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.1.0",
        help="Model prediction version tag for PostGIS versioning (default: v1.1.0).",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip GSE download.")
    parser.add_argument("--skip-infer", action="store_true", help="Skip model inference.")
    parser.add_argument("--skip-vectorize", action="store_true", help="Skip vectorization.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip PostGIS ingestion.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run AOI discovery and print grid plan without downloading heavy rasters.",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Executing AOI discovery and grid planning...")
        if not args.aoi.exists():
            bbox = fetch_kab_tangerang_aoi_from_gee(args.aoi)
        else:
            with open(args.aoi, "r", encoding="utf-8") as f:
                d = json.load(f)
            geom = shape(d["features"][0]["geometry"])
            bbox = (float(geom.bounds[0]), float(geom.bounds[1]), float(geom.bounds[2]), float(geom.bounds[3]))
        grid = generate_spatial_grid(bbox, rows=args.grid_rows, cols=args.grid_cols)
        print(f"[DRY RUN] Kabupaten Tangerang AOI: {args.aoi} (BBOX={bbox})")
        print(f"[DRY RUN] Partitioned into {len(grid)} spatial sub-tiles ({args.grid_rows}x{args.grid_cols}):")
        for cell in grid:
            print(f"  • Cell [{cell['cell_id']}]: BBOX={cell['bbox']}")
        print("[DRY RUN] Ready for execution!")
        return

    run_regional_pipeline(
        aoi_geojson=args.aoi,
        checkpoint=args.checkpoint,
        target_year=args.year,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        model_version=args.model_version,
        skip_download=args.skip_download,
        skip_infer=args.skip_infer,
        skip_vectorize=args.skip_vectorize,
        skip_ingest=args.skip_ingest,
    )


if __name__ == "__main__":
    main()
