"""Anti-Pixelated & Anti-Island Polygon Vectorization Pipeline.

Converts raw class probability maps into clean, cartographically smooth
OpenStreetMap-ready polygons by combining:
1. Chunked Gaussian probability smoothing (`sigma=0.8`)
2. Minority-aware probability thresholding (`industrial_thresh=0.38`)
3. GDAL multi-class `sieve` (`min_pixels=16` -> 1,600 m² Minimum Mapping Unit)
4. Metric projection to `EPSG:32748` (UTM Zone 48S Tangerang)
5. Chaikin corner-cutting smoothing + Douglas-Peucker topological simplification
6. Export to `EPSG:4326` with OSM tags (`landuse`, `la:version`, `la:confidence`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import shapes, sieve
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as shapely_transform

from .model import CLASS_MAPPING


def chaikin_smooth_ring(
    coords: Sequence[tuple[float, float]],
    iterations: int = 2,
) -> list[tuple[float, float]]:
    """Apply Chaikin's corner-cutting algorithm to a closed polygon ring.

    Replaces sharp 90-degree 10m raster pixel staircase corners with smooth
    quarter-point cuts (`0.75 * P_i + 0.25 * P_{i+1}` and `0.25 * P_i + 0.75 * P_{i+1}`).

    Args:
        coords: Sequence of `(x, y)` ring coordinates.
        iterations: Number of Chaikin refinement passes (default `2`).

    Returns:
        Closed list of `(x, y)` tuples representing the smoothed ring.
    """
    pts = np.asarray(coords, dtype=np.float64)[:, :2]
    if len(pts) < 4:
        return [tuple(map(float, p)) for p in pts]

    # Strip duplicate closing vertex during iterative cutting
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    if len(pts) < 3 or iterations <= 0:
        closed = np.vstack([pts, pts[0:1]])
        return [(float(p[0]), float(p[1])) for p in closed]

    for _ in range(iterations):
        next_pts = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * next_pts
        r = 0.25 * pts + 0.75 * next_pts
        interleaved = np.empty((len(pts) * 2, 2), dtype=np.float64)
        interleaved[0::2] = q
        interleaved[1::2] = r
        pts = interleaved

    closed = np.vstack([pts, pts[0:1]])
    return [(float(p[0]), float(p[1])) for p in closed]


def _smooth_single_polygon(
    poly: Polygon,
    simplify_tol_m: float = 4.0,
    min_hole_area_m2: float = 2500.0,
    chaikin_iterations: int = 2,
) -> Polygon:
    """Smooth exterior and valid interior rings of a single metric Polygon."""
    if poly.is_empty:
        return poly

    ext_coords = chaikin_smooth_ring(poly.exterior.coords, iterations=chaikin_iterations)
    kept_holes: list[list[tuple[float, float]]] = []
    for interior in poly.interiors:
        hole_poly = Polygon(interior)
        if hole_poly.area >= min_hole_area_m2:
            kept_holes.append(
                chaikin_smooth_ring(interior.coords, iterations=chaikin_iterations)
            )

    smoothed = Polygon(ext_coords, kept_holes).buffer(0)
    if simplify_tol_m > 0.0:
        smoothed = smoothed.simplify(simplify_tol_m, preserve_topology=True).buffer(0)
    return smoothed


def smooth_and_simplify_polygon(
    geom: Polygon | MultiPolygon,
    simplify_tol_m: float = 4.0,
    min_hole_area_m2: float = 2500.0,
    chaikin_iterations: int = 2,
) -> Polygon | MultiPolygon:
    """Smooth a metric Polygon/MultiPolygon via Chaikin corner-cutting and simplify.

    Args:
        geom: Input `Polygon` or `MultiPolygon` in metric CRS (e.g. `EPSG:32748`).
        simplify_tol_m: Douglas-Peucker simplification tolerance in meters (`4.0m`).
        min_hole_area_m2: Interior donut holes smaller than this area (`2,500 m²`)
            are filled/removed to eliminate interior speckle.
        chaikin_iterations: Number of Chaikin corner-cutting passes (`2`).

    Returns:
        Topologically valid, smoothed `Polygon` or `MultiPolygon`.
    """
    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return _smooth_single_polygon(
            geom,
            simplify_tol_m=simplify_tol_m,
            min_hole_area_m2=min_hole_area_m2,
            chaikin_iterations=chaikin_iterations,
        )

    if isinstance(geom, MultiPolygon):
        parts = [
            _smooth_single_polygon(
                part,
                simplify_tol_m=simplify_tol_m,
                min_hole_area_m2=min_hole_area_m2,
                chaikin_iterations=chaikin_iterations,
            )
            for part in geom.geoms
            if not part.is_empty
        ]
        valid_parts: list[Polygon] = []
        for p in parts:
            if isinstance(p, Polygon) and not p.is_empty:
                valid_parts.append(p)
            elif isinstance(p, MultiPolygon):
                valid_parts.extend([sub for sub in p.geoms if not sub.is_empty])
        if not valid_parts:
            return Polygon()
        if len(valid_parts) == 1:
            return valid_parts[0]
        return MultiPolygon(valid_parts).buffer(0)

    return geom


def _chunked_discretize_and_smooth(
    prob_mmap: np.ndarray,
    sigma: float = 0.8,
    industrial_thresh: float = 0.38,
    residential_thresh: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth probability map in row chunks and discretize into `{0, 1, 2}` mask."""
    _, height, width = prob_mmap.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    winning_prob = np.zeros((height, width), dtype=np.float32)

    halo = max(4, int(np.ceil(sigma * 4.0))) if sigma > 0 else 0
    chunk_h = max(64, height // 20)

    for r0 in range(0, height, chunk_h):
        r1 = min(height, r0 + chunk_h)
        pad_r0 = max(0, r0 - halo)
        pad_r1 = min(height, r1 + halo)

        sub_prob = np.asarray(prob_mmap[:, pad_r0:pad_r1, :], dtype=np.float32)
        if sigma > 0.0:
            sub_prob = gaussian_filter(sub_prob, sigma=(0.0, sigma, sigma))

        local_s = r0 - pad_r0
        local_e = local_s + (r1 - r0)
        chunk_prob = sub_prob[:, local_s:local_e, :]

        p_bg = chunk_prob[0]
        p_ind = chunk_prob[1]
        p_res = chunk_prob[2]

        chunk_mask = np.zeros((r1 - r0, width), dtype=np.uint8)
        chunk_conf = p_bg.copy()

        # Assign residential where residential probability exceeds threshold and background
        res_cond = (p_res >= residential_thresh) & (p_res >= p_bg)
        chunk_mask[res_cond] = 2
        chunk_conf[res_cond] = p_res[res_cond]

        # Minority boost for industrial (7.8% coverage): lower threshold (0.38)
        ind_cond = (p_ind >= industrial_thresh) & (
            p_ind >= p_res * (industrial_thresh / max(residential_thresh, 1e-6))
        )
        chunk_mask[ind_cond] = 1
        chunk_conf[ind_cond] = p_ind[ind_cond]

        mask[r0:r1, :] = chunk_mask
        winning_prob[r0:r1, :] = chunk_conf

    return mask, winning_prob


def prob_map_to_osm_polygons(
    prob_mmap: np.ndarray,
    transform: Affine,
    crs: Any,
    model_version: str = "v1.0.0",
    min_pixels: int = 16,
    sigma: float = 0.8,
    industrial_thresh: float = 0.38,
    residential_thresh: float = 0.45,
    utm_epsg: str = "EPSG:32748",
) -> gpd.GeoDataFrame:
    """Convert a `(3, H, W)` probability array into smooth OSM-tagged polygons.

    Pipeline stages:
    1. Chunked Gaussian probability filter (`sigma=0.8`)
    2. Threshold discretization with minority class sensitivity (`industrial_thresh=0.38`)
    3. GDAL `rasterio.features.sieve(mask, size=min_pixels, connectivity=8)` to
       remove salt-and-pepper islands `< 16` pixels (`1,600 m²` at 10m resolution)
    4. Polygon contour extraction via `rasterio.features.shapes`
    5. Reprojection to metric `EPSG:32748` (UTM Zone 48S Tangerang)
    6. Chaikin corner-cutting + Douglas-Peucker simplification (`4.0m`) + hole filtering
    7. Reprojection to `EPSG:4326` with OSM schema attributes.

    Args:
        prob_mmap: Array or memmap of shape `(3, H, W)` with class probabilities.
        transform: Affine transform of the source raster.
        crs: Coordinate reference system of the source raster.
        model_version: Model version tag written to `la:version`.
        min_pixels: Minimum contiguous pixel count for GDAL sieve (`16` = 1,600 m²).
        sigma: Standard deviation for spatial Gaussian filter on probabilities.
        industrial_thresh: Probability threshold for class `1` (`industrial`).
        residential_thresh: Probability threshold for class `2` (`residential`).
        utm_epsg: Metric UTM CRS for Tangerang (`"EPSG:32748"`).

    Returns:
        `geopandas.GeoDataFrame` in `EPSG:4326` ready for PostGIS and Rapid Editor.
    """
    if prob_mmap.ndim != 3 or prob_mmap.shape[0] < 3:
        raise ValueError(
            f"Expected prob_mmap of shape (3, H, W), got {prob_mmap.shape}"
        )

    raw_mask, _ = _chunked_discretize_and_smooth(
        prob_mmap=prob_mmap,
        sigma=sigma,
        industrial_thresh=industrial_thresh,
        residential_thresh=residential_thresh,
    )

    if min_pixels > 1:
        sieved_mask = sieve(raw_mask.astype(np.uint8), size=int(min_pixels), connectivity=8)
    else:
        sieved_mask = raw_mask.astype(np.uint8)

    src_crs = CRS.from_user_input(crs) if crs is not None else CRS.from_epsg(4326)
    metric_crs = CRS.from_user_input(utm_epsg)
    wgs84_crs = CRS.from_epsg(4326)

    to_utm = Transformer.from_crs(src_crs, metric_crs, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(metric_crs, wgs84_crs, always_xy=True).transform

    height, width = sieved_mask.shape
    min_area_m2 = float(min_pixels) * 100.0
    records: list[dict[str, Any]] = []

    foreground_mask = sieved_mask > 0
    for geom_dict, val in shapes(sieved_mask, mask=foreground_mask, transform=transform):
        class_id = int(val)
        if class_id not in (1, 2):
            continue

        raw_geom = shape(geom_dict)
        if raw_geom.is_empty:
            continue

        # Compute mean class probability within polygon bounding window
        minx, miny, maxx, maxy = raw_geom.bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=transform)
        r0 = max(0, int(np.floor(win.row_off)))
        c0 = max(0, int(np.floor(win.col_off)))
        r1 = min(height, int(np.ceil(win.row_off + win.height)))
        c1 = min(width, int(np.ceil(win.col_off + win.width)))

        if r1 > r0 and c1 > c0:
            sub_prob = np.asarray(prob_mmap[class_id, r0:r1, c0:c1], dtype=np.float32)
            sub_sieved = sieved_mask[r0:r1, c0:c1] == class_id
            if np.any(sub_sieved):
                mean_conf = float(np.mean(sub_prob[sub_sieved]))
            else:
                mean_conf = float(np.mean(sub_prob))
        else:
            mean_conf = 0.5

        geom_utm = shapely_transform(to_utm, raw_geom)
        smoothed_utm = smooth_and_simplify_polygon(
            geom_utm,
            simplify_tol_m=4.0,
            min_hole_area_m2=2500.0,
            chaikin_iterations=2,
        )
        if smoothed_utm.is_empty:
            continue

        area_m2 = float(smoothed_utm.area)
        if area_m2 < min_area_m2:
            continue

        geom_wgs84 = shapely_transform(to_wgs84, smoothed_utm)
        if not geom_wgs84.is_valid:
            geom_wgs84 = geom_wgs84.buffer(0)
        if geom_wgs84.is_empty:
            continue
        records.append(
            {
                "landuse": CLASS_MAPPING[class_id],
                "class_id": class_id,
                "la:version": model_version,
                "la:confidence": round(mean_conf, 4),
                "area_m2": round(area_m2, 2),
                "source": "LandAssist AI (Sentinel-2 + GSE)",
                "geometry": geom_wgs84,
            }
        )

    columns = [
        "landuse",
        "class_id",
        "la:version",
        "la:confidence",
        "area_m2",
        "source",
        "geometry",
    ]
    if not records:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs="EPSG:4326")


def export_visual_qa_html(
    gdf: gpd.GeoDataFrame,
    html_path: str | Path,
) -> Path:
    """Generate a standalone interactive Leaflet + Satellite QA map HTML before PostGIS ingestion."""
    out_html = Path(html_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    geojson_str = gdf.to_json() if not gdf.empty else '{"type": "FeatureCollection", "features": []}'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>LandAssist — Pre-PostGIS Interactive Visual QA</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    body, html {{ margin: 0; padding: 0; height: 100%; font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; }}
    #map {{ width: 100%; height: 100%; }}
    .qa-panel {{
      position: absolute; top: 14px; right: 14px; z-index: 1000;
      background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(8px);
      padding: 16px; border-radius: 12px; width: 290px;
      border: 1px solid rgba(148, 163, 184, 0.25); box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    .qa-panel h3 {{ margin: 0 0 8px 0; font-size: 15px; color: #38bdf8; }}
    .qa-row {{ margin: 10px 0; font-size: 13px; }}
    .badge-ind {{ display: inline-block; width: 12px; height: 12px; background: #f59e0b; border-radius: 3px; margin-right: 6px; vertical-align: middle; }}
    .badge-res {{ display: inline-block; width: 12px; height: 12px; background: #10b981; border-radius: 3px; margin-right: 6px; vertical-align: middle; }}
    input[type=range] {{ width: 100%; cursor: pointer; }}
    .stat-box {{ background: rgba(30, 41, 59, 0.8); padding: 8px 10px; border-radius: 8px; font-size: 12px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="qa-panel">
    <h3>🛰️ LandAssist Pre-Ingest QA</h3>
    <div class="stat-box" id="stats">Loading polygons...</div>
    <div class="qa-row">
      <label><input type="checkbox" id="chkInd" checked /> <span class="badge-ind"></span>Industrial (landuse=industrial)</label><br/>
      <label><input type="checkbox" id="chkRes" checked /> <span class="badge-res"></span>Residential (landuse=residential)</label>
    </div>
    <div class="qa-row">
      <label>Min Confidence: <strong id="confVal">0.35</strong></label>
      <input type="range" id="confSlider" min="0.0" max="0.95" step="0.05" value="0.35" />
    </div>
    <div class="qa-row">
      <label>Polygon Fill Opacity: <strong id="opacVal">0.40</strong></label>
      <input type="range" id="opacSlider" min="0.05" max="0.90" step="0.05" value="0.40" />
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const rawData = {geojson_str};
    const map = L.map('map').setView([-6.22, 106.52], 13);
    const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19, attribution: 'Esri World Imagery'
    }}).addTo(map);
    const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19, attribution: '© OpenStreetMap'
    }});
    L.control.layers({{"Satellite Imagery": sat, "OpenStreetMap": osm}}).addTo(map);

    let geoLayer = null;
    function renderLayer() {{
      if (geoLayer) map.removeLayer(geoLayer);
      const showInd = document.getElementById('chkInd').checked;
      const showRes = document.getElementById('chkRes').checked;
      const minConf = parseFloat(document.getElementById('confSlider').value);
      const fillOpac = parseFloat(document.getElementById('opacSlider').value);
      document.getElementById('confVal').textContent = minConf.toFixed(2);
      document.getElementById('opacVal').textContent = fillOpac.toFixed(2);

      let countInd = 0, countRes = 0, areaIndHa = 0, areaResHa = 0;
      geoLayer = L.geoJSON(rawData, {{
        filter: (f) => {{
          const p = f.properties || {{}};
          const c = p['la:confidence'] ?? 1.0;
          if (c < minConf) return false;
          if (p.landuse === 'industrial' && !showInd) return false;
          if (p.landuse === 'residential' && !showRes) return false;
          return true;
        }},
        style: (f) => {{
          const isInd = f.properties.landuse === 'industrial';
          return {{
            color: isInd ? '#f59e0b' : '#10b981',
            weight: 2,
            fillColor: isInd ? '#f59e0b' : '#10b981',
            fillOpacity: fillOpac
          }};
        }},
        onEachFeature: (f, layer) => {{
          const p = f.properties || {{}};
          if (p.landuse === 'industrial') {{ countInd++; areaIndHa += (p.area_m2 || 0) / 10000.0; }}
          if (p.landuse === 'residential') {{ countRes++; areaResHa += (p.area_m2 || 0) / 10000.0; }}
          layer.bindPopup(
            `<b>landuse=${{p.landuse}}</b><br/>` +
            `Confidence: <b>${{(p['la:confidence'] * 100).toFixed(1)}}%</b><br/>` +
            `Area: <b>${{(p.area_m2 || 0).toLocaleString()}} m²</b> (${{((p.area_m2 || 0)/10000).toFixed(2)}} ha)<br/>` +
            `Version: <code>${{p['la:version']}}</code>`
          );
        }}
      }}).addTo(map);

      document.getElementById('stats').innerHTML =
        `Visible Polygons: <b>${{countInd + countRes}}</b><br/>` +
        `• Industrial: <b>${{countInd}}</b> (${{areaIndHa.toFixed(1)}} ha)<br/>` +
        `• Residential: <b>${{countRes}}</b> (${{areaResHa.toFixed(1)}} ha)`;
    }}

    renderLayer();
    if (geoLayer && geoLayer.getLayers().length > 0) {{
      map.fitBounds(geoLayer.getBounds(), {{padding: [24, 24]}});
    }}
    ['chkInd', 'chkRes', 'confSlider', 'opacSlider'].forEach(id => {{
      document.getElementById(id).addEventListener('input', renderLayer);
    }});
  </script>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")
    return out_html


def main() -> None:
    """CLI entrypoint for converting probability memmaps into smoothed GeoJSON/GPKG."""
    parser = argparse.ArgumentParser(
        description="Vectorize LandAssist probability memmap into smoothed OSM polygons."
    )
    parser.add_argument(
        "--prob-mmap",
        type=str,
        required=True,
        help="Path to float16 probability memmap (pred_sum.mmap).",
    )
    parser.add_argument(
        "--ref-tif",
        type=str,
        required=True,
        help="Reference GeoTIFF for affine transform, CRS, and dimensions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output vector path (.geojson or .gpkg).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0.0",
        help="Model version string embedded in la:version.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=16,
        help="Minimum contiguous pixels for GDAL sieve (16 px = 1,600 m²).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma over probability channels.",
    )
    parser.add_argument(
        "--industrial-thresh",
        type=float,
        default=0.38,
        help="Probability threshold for minority industrial class.",
    )
    parser.add_argument(
        "--residential-thresh",
        type=float,
        default=0.45,
        help="Probability threshold for residential class.",
    )
    parser.add_argument(
        "--utm-epsg",
        type=str,
        default="EPSG:32748",
        help="Metric UTM CRS for Chaikin smoothing and area calculation (default EPSG:32748).",
    )
    parser.add_argument(
        "--html-preview",
        type=str,
        default=None,
        help="Optional path to generate a standalone interactive Leaflet satellite QA map (.html).",
    )

    args = parser.parse_args()
    with rasterio.open(args.ref_tif) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    prob_mmap = np.memmap(
        args.prob_mmap,
        dtype="float16",
        mode="r",
        shape=(3, height, width),
    )

    gdf = prob_map_to_osm_polygons(
        prob_mmap=prob_mmap,
        transform=transform,
        crs=crs,
        model_version=args.model_version,
        min_pixels=args.min_pixels,
        sigma=args.sigma,
        industrial_thresh=args.industrial_thresh,
        residential_thresh=args.residential_thresh,
        utm_epsg=args.utm_epsg,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    driver = "GPKG" if out_path.suffix.lower() == ".gpkg" else "GeoJSON"
    gdf.to_file(out_path, driver=driver)
    print(f"[OK] Exported {len(gdf)} smoothed polygons to {out_path} ({driver}).")

    if args.html_preview:
        preview_path = export_visual_qa_html(gdf, args.html_preview)
        print(f"[OK] Generated interactive visual QA map at {preview_path}")



if __name__ == "__main__":
    main()
