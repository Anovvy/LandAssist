#!/usr/bin/env python3
"""Generator for ml/notebooks/regional_inference_kab_tangerang_colab.ipynb."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = REPO_ROOT / "ml" / "notebooks" / "regional_inference_kab_tangerang_colab.ipynb"


def create_markdown_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
    }


def create_code_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines],
    }


def main():
    cells = []

    # Title & Introduction
    cells.append(
        create_markdown_cell([
            "# LandAssist: Full Regional 64-Band Inference (Kabupaten Tangerang)",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anovvy/LandAssist/blob/main/ml/notebooks/regional_inference_kab_tangerang_colab.ipynb)",
            "",
            "## Overview",
            "This notebook executes the **end-to-end regional inference and vectorization pipeline** for all of **Kabupaten Tangerang (~1,193 km²)** on a **Google Colab GPU T4 runtime**:",
            "1. **Earth Engine AOI Discovery**: Queries `FAO/GAUL/2015/level2` (`ADM2_CODE=18028`) for the official Kabupaten Tangerang administrative boundary.",
            "2. **GEE 48 MB Limit Mitigation**: Partitions the 1,193 km² bounding box into a **2x2 spatial grid (4 sub-tiles)** with 200m buffer overlap.",
            "3. **64-Band GSE Acquisition & Mosaic**: Downloads Google Satellite Embeddings via Earth Engine and mosaics them into a single regional GeoTIFF (`4,875 x 3,890` px @ 10m GSD).",
            "4. **GPU T4 Batched Sliding-Window Inference**: SegFormer MiT-B2 64-channel inference with 2D Hann window blending into disk memmap buffers (~1-2 minutes on GPU).",
            "5. **Metric Smoothing & Administrative Clipping**: GDAL Sieve (1,600 m²), Chaikin corner-cutting in UTM Zone 48S (`EPSG:32748`), and administrative boundary clipping.",
            "6. **Supabase PostGIS Ingestion**: Batch ingestion into `public.landuse_predictions` with `model_version='v1.1.0'`, instantly reflected on the live LandAssist Web Editor.",
        ])
    )

    # Cell 1: Environment & GPU Verification
    cells.append(
        create_code_cell([
            "# Cell 1: Verify GPU T4 & Install Dependencies",
            "import subprocess",
            "import sys",
            "import os",
            "import torch",
            "",
            "print(f'PyTorch Version: {torch.__version__}')",
            "print(f'CUDA Available: {torch.cuda.is_available()}')",
            "if torch.cuda.is_available():",
            "    print(f'Active GPU: {torch.cuda.get_device_name(0)}')",
            "    print(f'Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')",
            "else:",
            "    print('WARNING: Running on CPU! Enable GPU in Runtime -> Change runtime type -> T4 GPU.')",
            "",
            "# Install required geospatial & ML libraries in Colab",
            "if 'google.colab' in sys.modules:",
            "    subprocess.check_call([",
            "        sys.executable, '-m', 'pip', 'install', '-q',",
            "        'earthengine-api>=0.1.390', 'rasterio>=1.3.9', 'shapely>=2.0.2',",
            "        'geopandas>=0.14.3', 'pyproj>=3.6.1', 'transformers>=4.38.0',",
            "        'asyncpg>=0.29.0', 'scipy>=1.12.0'",
            "    ])",
            "",
            "# Clone or pull latest LandAssist repository",
            "REPO_DIR = '/content/LandAssist'",
            "if not os.path.exists(REPO_DIR):",
            "    !git clone https://github.com/Anovvy/LandAssist.git /content/LandAssist",
            "else:",
            "    !cd /content/LandAssist && git pull origin main",
            "",
            "if REPO_DIR not in sys.path:",
            "    sys.path.insert(0, REPO_DIR)",
            "",
            "print('Environment and LandAssist codebase successfully staged.')",
        ])
    )

    # Cell 2: Earth Engine Authentication & AOI Boundary Retrieval
    cells.append(
        create_code_cell([
            "# Cell 2: Earth Engine Authentication & AOI Boundary Retrieval",
            "#@title Google Earth Engine Authentication & Tangerang Boundary",
            "import ee",
            "import json",
            "import os",
            "from pathlib import Path",
            "from shapely.geometry import shape",
            "",
            "# Google Cloud Project ID registered for Earth Engine",
            "EE_PROJECT_ID = 'landassist-511504'  #@param {type:'string'}",
            "",
            "is_initialized = False",
            "",
            "# Option A: Check if Service Account credentials.json was uploaded to Colab",
            "sa_candidates = [",
            "    Path('/content/credentials.json'),",
            "    Path('/content/LandAssist/credentials.json'),",
            "    Path('/content/service_account.json'),",
            "]",
            "sa_key = next((p for p in sa_candidates if p.exists() and p.is_file()), None)",
            "if sa_key:",
            "    try:",
            "        key_data = json.loads(sa_key.read_text(encoding='utf-8'))",
            "        client_email = key_data.get('client_email')",
            "        resolved_proj = key_data.get('project_id', EE_PROJECT_ID)",
            "        creds = ee.ServiceAccountCredentials(client_email, str(sa_key))",
            "        ee.Initialize(credentials=creds, project=resolved_proj)",
            "        print(f'Google Earth Engine initialized via Service Account ({client_email}) | Project: {resolved_proj}')",
            "        is_initialized = True",
            "    except Exception as e:",
            "        print(f'Service account init note: {e}')",
            "",
            "# Option B: Interactive Colab OAuth Flow",
            "if not is_initialized:",
            "    try:",
            "        ee.Initialize(project=EE_PROJECT_ID)",
            "        print(f'Google Earth Engine successfully initialized with project: {EE_PROJECT_ID}')",
            "        is_initialized = True",
            "    except Exception as exc:",
            "        print(f'Cached connection not ready ({exc}). Launching interactive ee.Authenticate()...')",
            "        try:",
            "            ee.Authenticate()",
            "            ee.Initialize(project=EE_PROJECT_ID)",
            "            print(f'Google Earth Engine initialized successfully with project: {EE_PROJECT_ID}')",
            "            is_initialized = True",
            "        except Exception as e2:",
            "            print(f'\\n[!] Failed with project {EE_PROJECT_ID}: {e2}')",
            "            user_proj = input('Enter your Google Cloud Project ID for Earth Engine: ').strip()",
            "            if user_proj:",
            "                ee.Initialize(project=user_proj)",
            "                print(f'Google Earth Engine initialized with project: {user_proj}')",
            "                is_initialized = True",
            "            else:",
            "                print('No project specified. Tip: Upload your local credentials.json to /content/ and re-run.')",
            "                raise",
            "",
            "# Fetch official administrative boundary for Kabupaten Tangerang",
            "KAB_TANGERANG_ADM2_CODE = 18028",
            "print(f'Fetching official boundary for Kabupaten Tangerang (GAUL ADM2_CODE={KAB_TANGERANG_ADM2_CODE})...')",
            "kab_feature = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(",
            "    ee.Filter.eq('ADM2_CODE', KAB_TANGERANG_ADM2_CODE)",
            ").first()",
            "",
            "aoi_geom = kab_feature.geometry()",
            "aoi_info = aoi_geom.getInfo()",
            "",
            "AOI_PATH = Path('/content/aoi_kab_tangerang.geojson')",
            "AOI_PATH.write_text(json.dumps({",
            "    'type': 'FeatureCollection',",
            "    'features': [{",
            "        'type': 'Feature',",
            "        'properties': {'name': 'Kabupaten Tangerang', 'adm2_code': KAB_TANGERANG_ADM2_CODE},",
            "        'geometry': aoi_info",
            "    }]",
            "}, indent=2), encoding='utf-8')",
            "",
            "bounds = shape(aoi_info).bounds",
            "print(f'Kabupaten Tangerang AOI Bounds: {bounds}')",
            "print(f'Bounding Box (min_lon, min_lat, max_lon, max_lat): {bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}')",
            "print('AOI boundary ready.')",
        ])
    )

    # Cell 3: Spatial Grid Partitioning (2x2)
    cells.append(
        create_code_cell([
            "# Cell 3: 2x2 Spatial Grid Partitioning (Bypassing GEE 48MB Limit)",
            "def generate_spatial_grid(bbox, rows=2, cols=2, buffer_deg=0.002):",
            "    min_lon, min_lat, max_lon, max_lat = bbox",
            "    lon_step = (max_lon - min_lon) / cols",
            "    lat_step = (max_lat - min_lat) / rows",
            "    cells = []",
            "    for r in range(rows):",
            "        for c in range(cols):",
            "            c_min_lon = max(min_lon, min_lon + c * lon_step - buffer_deg)",
            "            c_max_lon = min(max_lon, min_lon + (c + 1) * lon_step + buffer_deg)",
            "            c_min_lat = max(min_lat, min_lat + r * lat_step - buffer_deg)",
            "            c_max_lat = min(max_lat, min_lat + (r + 1) * lat_step + buffer_deg)",
            "            cells.append({",
            "                'cell_id': f'r{r}_c{c}',",
            "                'row': r, 'col': c,",
            "                'bbox': (c_min_lon, c_min_lat, c_max_lon, c_max_lat)",
            "            })",
            "    return cells",
            "",
            "grid_cells = generate_spatial_grid(bounds, rows=2, cols=2, buffer_deg=0.002)",
            "print(f'Generated {len(grid_cells)} spatial sub-tiles for Kabupaten Tangerang:')",
            "for g in grid_cells:",
            "    print(f\"  Tile [{g['cell_id']}]: BBOX = {g['bbox']}\")",
        ])
    )

    # Cell 4: Download 64-Band GSE Sub-tiles & Mosaic
    cells.append(
        create_code_cell([
            "# Cell 4: Download 64-Band Google Satellite Embeddings & Mosaic GeoTIFF",
            "import rasterio",
            "from rasterio.merge import merge",
            "from ml.src.download_gse import download_gse_64band_geotiff",
            "",
            "TILES_DIR = Path('/content/tangerang_tiles')",
            "TILES_DIR.mkdir(parents=True, exist_ok=True)",
            "REGIONAL_TIF = Path('/content/kab_tangerang_gse64.tif')",
            "",
            "# Download each 64-band tile from GEE",
            "tile_paths = []",
            "for cell in grid_cells:",
            "    c_id = cell['cell_id']",
            "    tile_fp = TILES_DIR / f'tile_{c_id}.tif'",
            "    tile_paths.append(tile_fp)",
            "    print(f'\\nDownloading GSE 64-band sub-tile [{c_id}]...')",
            "    download_gse_64band_geotiff(output_path=tile_fp, bbox=cell['bbox'], target_year=2025)",
            "",
            "# Mosaic sub-tiles into master regional GeoTIFF",
            "print(f'\\nMerging {len(tile_paths)} sub-tiles into master GeoTIFF...')",
            "sources = [rasterio.open(p) for p in tile_paths]",
            "try:",
            "    mosaic, out_trans = merge(sources)",
            "    out_meta = sources[0].meta.copy()",
            "    out_meta.update({",
            "        'driver': 'GTiff',",
            "        'height': mosaic.shape[1],",
            "        'width': mosaic.shape[2],",
            "        'transform': out_trans,",
            "        'compress': 'lzw'",
            "    })",
            "    with rasterio.open(REGIONAL_TIF, 'w', **out_meta) as dst:",
            "        dst.write(mosaic)",
            "        for b in range(1, 65):",
            "            dst.set_band_description(b, f'A{b-1:02d}')",
            "    print(f'Master 64-band GeoTIFF saved to {REGIONAL_TIF}')",
            "    print(f'Shape: {mosaic.shape} (Channels: {mosaic.shape[0]}, Height: {mosaic.shape[1]}, Width: {mosaic.shape[2]})')",
            "finally:",
            "    for s in sources:",
            "        s.close()",
        ])
    )

    # Cell 5: Load SegFormer Model onto GPU T4
    cells.append(
        create_code_cell([
            "# Cell 5: Load SegFormer MiT-B2 Model onto GPU T4",
            "import urllib.request",
            "from ml.src.model import build_segformer_64ch",
            "",
            "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "print(f'Running model on: {DEVICE}')",
            "",
            "CHECKPOINT_PATH = Path('/content/LandAssist/weight/segformer-mitb2-gse64-v1.0.0.pt')",
            "",
            "# If weight file not present, check Google Drive or auto-download from GitHub Release",
            "if not CHECKPOINT_PATH.exists():",
            "    DRIVE_WEIGHT = Path('/content/drive/MyDrive/LandAssist/segformer-mitb2-gse64-v1.0.0.pt')",
            "    if DRIVE_WEIGHT.exists():",
            "        import shutil",
            "        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)",
            "        shutil.copy(DRIVE_WEIGHT, CHECKPOINT_PATH)",
            "        print(f'Copied checkpoint from Google Drive: {DRIVE_WEIGHT}')",
            "    else:",
            "        print('Downloading official v1.0.0 model weights from GitHub Releases...')",
            "        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)",
            "        WEIGHTS_URL = 'https://github.com/Anovvy/LandAssist/releases/download/weights-v1.0.0/segformer-mitb2-gse64-v1.0.0.pt'",
            "        try:",
            "            urllib.request.urlretrieve(WEIGHTS_URL, CHECKPOINT_PATH)",
            "            print(f'Downloaded {CHECKPOINT_PATH.name} ({CHECKPOINT_PATH.stat().st_size / (1024*1024):.2f} MB) successfully.')",
            "        except Exception as dl_err:",
            "            print(f'Auto-download failed: {dl_err}. Please upload segformer-mitb2-gse64-v1.0.0.pt manually.')",
            "            from google.colab import files",
            "            uploaded = files.upload()",
            "            for fn in uploaded.keys():",
            "                if fn.endswith('.pt'):",
            "                    CHECKPOINT_PATH = Path('/content') / fn",
            "                    break",
            "",
            "# Build model architecture & load state dict",
            "model = build_segformer_64ch(num_classes=3, in_channels=64, pretrained=False)",
            "ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')",
            "state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt",
            "model.load_state_dict(state_dict)",
            "model = model.to(DEVICE)",
            "model.eval()",
            "print(f'SegFormer MiT-B2 64-Channel successfully loaded and ready on {DEVICE}!')",
        ])
    )

    # Cell 6: Fast Batched GPU Inference with 2D Hann Memmap
    cells.append(
        create_code_cell([
            "# Cell 6: Batched GPU T4 Inference with 2D Hann Window Memmap Blending",
            "from ml.src.infer_tangerang import run_sliding_window_inference",
            "",
            "WORK_DIR = Path('/content/tmp_mmap_regional')",
            "WORK_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "print(f'Starting GPU T4 sliding-window inference on master raster: {REGIONAL_TIF}...')",
            "prob_memmap, out_meta = run_sliding_window_inference(",
            "    model=model,",
            "    geotiff_path=REGIONAL_TIF,",
            "    work_dir=WORK_DIR,",
            "    tile_size=160,",
            "    overlap=80,",
            "    device=DEVICE,",
            "    num_classes=3,",
            ")",
            "",
            "print(f'Inference finished! Probability memmap shape: {prob_memmap.shape}')",
        ])
    )

    # Cell 7: Vectorization, Chaikin Smoothing, & Administrative Clipping
    cells.append(
        create_code_cell([
            "# Cell 7: GDAL Sieve, UTM 48S Chaikin Smoothing & AOI Clipping",
            "from ml.src.vectorize import prob_map_to_osm_polygons",
            "from ml.src.regional_kab_tangerang import clip_geojson_to_aoi",
            "",
            "RAW_GEOJSON = Path('/content/kab_tangerang_v1.1.0_raw.geojson')",
            "FINAL_GEOJSON = Path('/content/kab_tangerang_v1.1.0.geojson')",
            "",
            "print('Vectorizing probability map to OSM-compliant polygons...')",
            "features_count = prob_map_to_osm_polygons(",
            "    prob_map=prob_memmap,",
            "    transform=out_meta['transform'],",
            "    crs=out_meta['crs'],",
            "    output_geojson_path=RAW_GEOJSON,",
            "    min_area_pixels=16,  # 1600 m2 sieve MMU",
            "    model_version='v1.1.0',",
            "    confidence_threshold=0.5,",
            ")",
            "print(f'Raw vectorization produced {features_count} polygons.')",
            "",
            "# Clip strictly to official Kabupaten Tangerang boundary",
            "print(f'Clipping polygons to official administrative boundary ({AOI_PATH})...')",
            "clipped_count = clip_geojson_to_aoi(",
            "    input_geojson_path=RAW_GEOJSON,",
            "    aoi_path=AOI_PATH,",
            "    output_geojson_path=FINAL_GEOJSON,",
            ")",
            "print(f'Final Kabupaten Tangerang dataset: {clipped_count} polygons saved to {FINAL_GEOJSON}!')",
        ])
    )

    # Cell 8: Satellite Visual QA Map Generation
    cells.append(
        create_code_cell([
            "# Cell 8: Generate Leaflet Satellite Visual QA Map",
            "from ml.src.vectorize import export_visual_qa_html",
            "",
            "QA_HTML_PATH = Path('/content/kab_tangerang_v1.1.0_qa.html')",
            "export_visual_qa_html(",
            "    geojson_path=FINAL_GEOJSON,",
            "    output_html_path=QA_HTML_PATH,",
            "    model_version='v1.1.0',",
            ")",
            "print(f'Interactive Satellite QA viewer saved to: {QA_HTML_PATH}')",
        ])
    )

    # Cell 9: Automated Ingestion to Supabase PostGIS
    cells.append(
        create_code_cell([
            "# Cell 9: Ingest v1.1.0 Polygons into Supabase PostGIS",
            "import asyncio",
            "from ml.src.ingest_postgis import ingest_geojson_to_postgis",
            "",
            "# Enter your Supabase DATABASE_URL or load from environment",
            "import getpass",
            "db_url = os.environ.get('DATABASE_URL')",
            "if not db_url:",
            "    db_url = getpass.getpass('Enter Supabase DATABASE_URL (Port 5432/6543): ')",
            "",
            "if db_url:",
            "    print(f'Ingesting {FINAL_GEOJSON} into Supabase PostGIS table landuse_predictions...')",
            "    inserted = asyncio.run(ingest_geojson_to_postgis(",
            "        geojson_path=FINAL_GEOJSON,",
            "        database_url=db_url,",
            "        model_version='v1.1.0',",
            "        truncate_version=True,",
            "    ))",
            "    print(f'SUCCESS! Ingested {inserted} polygons with model_version=v1.1.0 into Supabase!')",
            "    print('The live LandAssist Web Editor (https://landassist-api.onrender.com/editor/) now serves full-county predictions!')",
            "else:",
            "    print('DATABASE_URL skipped. You can manually download the GeoJSON below.')",
        ])
    )

    # Cell 10: Download Artifacts
    cells.append(
        create_code_cell([
            "# Cell 10: Download GeoJSON & QA Map Artifacts to Local Machine",
            "from google.colab import files",
            "",
            "print('Downloading final outputs...')",
            "if FINAL_GEOJSON.exists():",
            "    files.download(str(FINAL_GEOJSON))",
            "if QA_HTML_PATH.exists():",
            "    files.download(str(QA_HTML_PATH))",
        ])
    )

    notebook_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook_data, indent=2), encoding="utf-8")
    print(f"Generated Colab GPU T4 notebook at: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
