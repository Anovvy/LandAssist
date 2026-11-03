"""Automated 64-Band Google Satellite Embedding (GSE) Downloader via Earth Engine API.

Inspired by GeoChip & LandAssist Data Prep: downloads `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
(64 bands, `A00`..`A63`, 10m resolution, `EPSG:4326`) for any target bounding box (`--bbox`)
or shapefile (`--aoi`). Automatically casts to `Float32` (`.toFloat()`) and performs adaptive
band-chunking (`<= 22 MB` per request) to bypass Google Earth Engine's 48 MB `getDownloadURL`
limit without requiring manual Google Drive exports.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import requests
from tqdm import tqdm


def init_earth_engine(
    project_id: Optional[str] = None,
    service_account_key: Optional[str] = None,
) -> None:
    """Authenticate and initialize the Google Earth Engine Python client."""
    import ee  # type: ignore

    proj = project_id or os.environ.get("EE_PROJECT_ID")
    sa_path = service_account_key or os.environ.get(
        "EE_SERVICE_ACCOUNT_KEY", "credentials.json"
    )

    for candidate in (sa_path, "service_account.json"):
        if candidate and os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                key_data = json.load(f)
            client_email = key_data.get("client_email")
            resolved_proj = key_data.get("project_id", proj)
            creds = ee.ServiceAccountCredentials(client_email, candidate)
            ee.Initialize(creds, project=resolved_proj)
            print(
                f"[GEE AUTH] Service Account authenticated ({client_email}) | Project: {resolved_proj}"
            )
            return

    if proj:
        ee.Initialize(project=proj)
        print(f"[GEE AUTH] OAuth credentials initialized | Project: {proj}")
    else:
        ee.Initialize()
        print("[GEE AUTH] Default OAuth credentials initialized.")


def parse_bbox_string(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse `'minLon,minLat,maxLon,maxLat'` into validated WGS84 floats."""
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "Expected --bbox in 'minLon,minLat,maxLon,maxLat' format (4 comma-separated floats)."
        )
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(
            f"Invalid bbox ({bbox_str}): minLon < maxLon and minLat < maxLat required."
        )
    return min_lon, min_lat, max_lon, max_lat


def _stream_download(url: str, target_path: Path, desc: str) -> None:
    """Stream an HTTP response to disk with a progress bar."""
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(target_path, "wb") as f, tqdm(
        desc=desc,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_gse_64band_geotiff(
    output_path: Union[str, Path],
    bbox: Tuple[float, float, float, float],
    target_year: int = 2025,
    polygon_coords: Optional[Sequence[Sequence[float]]] = None,
    force: bool = False,
) -> Path:
    """Download a 64-band `Float32` GSE GeoTIFF (`EPSG:4326`, 10m) via adaptive band chunking."""
    import ee  # type: ignore

    out_tif = Path(output_path)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists() and out_tif.stat().st_size > 0 and not force:
        print(f"[GSE] File already exists at {out_tif} (use --force to overwrite).")
        return out_tif

    min_lon, min_lat, max_lon, max_lat = bbox
    if polygon_coords is not None:
        aoi_geom = ee.Geometry.Polygon([list(polygon_coords)])
    else:
        aoi_geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterBounds(
        aoi_geom
    )
    year_col = dataset.filterDate(f"{target_year}-01-01", f"{target_year + 1}-01-01")

    if int(year_col.size().getInfo()) > 0:
        gse_img = year_col.first().clip(aoi_geom).toFloat()
        print(f"[GSE] Selected annual embedding for year {target_year}.")
    else:
        latest_img = dataset.sort("system:time_start", False).first()
        gse_img = latest_img.clip(aoi_geom).toFloat()
        print(
            f"[GSE] Year {target_year} not yet published; falling back to latest available annual GSE."
        )

    all_bands: List[str] = list(gse_img.bandNames().getInfo())
    if len(all_bands) != 64:
        raise RuntimeError(f"Expected 64 GSE bands, got {len(all_bands)}.")

    # Estimate raster pixel count to choose safe band chunk size (<= 22 MB per GEE request)
    mid_lat = (min_lat + max_lat) / 2.0
    width_px = max(1, int((abs(max_lon - min_lon) * 111320.0 * np.cos(np.radians(mid_lat))) / 10.0))
    height_px = max(1, int((abs(max_lat - min_lat) * 110574.0) / 10.0))
    bytes_per_band = max(1, width_px * height_px * 4)
    safe_target_bytes = 22 * 1024 * 1024
    chunk_size = max(1, min(8, int(safe_target_bytes // bytes_per_band)))

    chunk_ranges = [
        all_bands[i : i + chunk_size] for i in range(0, len(all_bands), chunk_size)
    ]
    temp_dir = out_tif.parent / f".tmp_gse_chunks_{datetime.datetime.now().strftime('%H%M%S')}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    region_coords: Any = aoi_geom.coordinates().getInfo()

    chunk_files: List[Path] = []
    try:
        for idx, band_subset in enumerate(chunk_ranges):
            chunk_file = temp_dir / f"chunk_{idx:03d}.tif"
            chunk_files.append(chunk_file)
            sub_img = gse_img.select(band_subset)
            params = {
                "scale": 10,
                "crs": "EPSG:4326",
                "region": region_coords,
                "format": "GEO_TIFF",
            }
            url = sub_img.getDownloadURL(params)
            _stream_download(
                url,
                chunk_file,
                desc=f"GSE Chunk {idx + 1}/{len(chunk_ranges)} ({band_subset[0]}..{band_subset[-1]})",
            )

        profile = None
        arrays: List[np.ndarray] = []
        for cf in chunk_files:
            with rasterio.open(cf) as src:
                if profile is None:
                    profile = src.profile.copy()
                arrays.append(src.read())

        full_data = np.concatenate(arrays, axis=0)
        assert profile is not None
        profile.update(
            count=len(all_bands),
            dtype="float32",
            driver="GTiff",
            compress="lzw",
            nodata=None,
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(full_data)
            for b_idx, b_name in enumerate(all_bands, start=1):
                dst.set_band_description(b_idx, b_name)

        print(
            f"[OK] Saved 64-band GSE GeoTIFF to {out_tif} "
            f"(shape={full_data.shape}, CRS=EPSG:4326)."
        )
        return out_tif
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    """CLI entrypoint for downloading a 64-band GSE GeoTIFF for a new region."""
    parser = argparse.ArgumentParser(
        description="Download 64-band Google Satellite Embeddings (GSE) GeoTIFF via Earth Engine."
    )
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Target WGS84 bounding box: 'minLon,minLat,maxLon,maxLat' (e.g. '106.45,-6.30,106.60,-6.15').",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output 64-band GeoTIFF path (e.g. 'data/tangerang_gse64.tif').",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Target annual GSE representation year (default: 2025).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud / Earth Engine Project ID (or set $EE_PROJECT_ID).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output GeoTIFF if it already exists.",
    )

    args = parser.parse_args()
    bbox_tuple = parse_bbox_string(args.bbox)
    init_earth_engine(project_id=args.project)
    download_gse_64band_geotiff(
        output_path=args.output,
        bbox=bbox_tuple,
        target_year=args.year,
        force=args.force,
    )


if __name__ == "__main__":
    main()
