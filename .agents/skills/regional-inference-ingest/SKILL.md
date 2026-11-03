---
name: regional-inference-ingest
description: Operational workflow for downloading 64-band GSE via Earth Engine, running anti-OOM sliding-window inference, GDAL sieve + UTM Chaikin vectorization with interactive HTML visual QA, and zero-downtime versioned PostGIS ingestion.
---

# Skill: Regional Inference & Zero-Downtime PostGIS Ingestion (`regional-inference-ingest`)

Use this skill when executing or automating an end-to-end regional inference run (e.g., updating Kabupaten Tangerang with a new model version `v1.1.0`, or expanding coverage to Kota Tangerang, Tangerang Selatan, or Bekasi) and ingesting the resulting polygons into Supabase PostGIS.

---

## Step 1: Download or Validate the 64-Band GSE GeoTIFF (`ml/src/download_gse.py`)

If you do not yet have the 64-band Google Satellite Embedding (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) GeoTIFF for the target region, download it automatically using `ml.src.download_gse` (which casts to `Float32` and performs adaptive `<= 22 MB` band-chunking to bypass Google Earth Engine's 48 MB direct download limit):

```bash
python -m ml.src.download_gse \
  --bbox "106.45,-6.30,106.60,-6.15" \
  --year 2025 \
  --output data/region_gse64.tif
```

Before running inference, verify that the input raster satisfies LandAssist's geospatial and channel invariants:
1. Band count **MUST** be exactly `64` (Google Satellite Embeddings `A00`..`A63`).
2. Spatial resolution should be ~10m (`0.0000898°` in `EPSG:4326` or `10.0m` in `EPSG:32748`).
3. Verify sufficient disk space in the temporary directory (`--temp-dir`) for the `float16` `np.memmap` accumulators (`(num_classes, H, W) * 2 bytes` + `(H, W) * 2 bytes`).

```bash
python -c "import rasterio; ds = rasterio.open('data/region_gse64.tif'); assert ds.count == 64, f'Expected 64 bands, got {ds.count}'; print(ds.crs, ds.shape)"
```

---

## Step 2: Execute Anti-OOM Sliding-Window Inference (`ml/src/infer_tangerang.py`)

Run `ml/src/infer_tangerang.py` with `tile_size=160` and `overlap=80` (50% overlap blended with a 2D Hann window):

```bash
python -m ml.src.infer_tangerang \
  --checkpoint weights/segformer-mitb2-gse64-v1.0.0.pt \
  --input-tif data/region_gse64.tif \
  --temp-dir ./tmp_mmap \
  --tile-size 160 \
  --overlap 80 \
  --batch-size 8
```

- **Output:** Disk-backed `float16` probability memmap `./tmp_mmap/pred_sum.mmap` of shape `(3, H, W)`.

---

## Step 3: Run GDAL Sieve, UTM Chaikin Vectorization & Pre-Ingest Visual QA (`ml/src/vectorize.py`)

Convert the probability memmap into clean, OSM-compliant polygons and generate a standalone interactive HTML Visual QA map (`--html-preview`) to inspect predictions over satellite imagery **before** touching PostGIS:

```bash
python -m ml.src.vectorize \
  --prob-mmap ./tmp_mmap/pred_sum.mmap \
  --ref-tif data/region_gse64.tif \
  --output outputs/region_v1.0.0.geojson \
  --model-version v1.0.0 \
  --min-pixels 16 \
  --sigma 0.8 \
  --industrial-thresh 0.38 \
  --residential-thresh 0.45 \
  --utm-epsg EPSG:32748 \
  --html-preview outputs/region_v1.0.0_qa.html
```

- **Visual QA Gate:** Open `outputs/region_v1.0.0_qa.html` in any browser (or drag-and-drop `outputs/region_v1.0.0.geojson` into `https://rapideditor.org/edit` -> Custom Data) to inspect polygon boundaries, test confidence thresholds interactively, and verify `area_m2 >= 1600.0`.

---

## Step 4: Versioned PostGIS Ingestion (`ml/src/ingest_postgis.py`)

Once visual QA is approved, ingest the GeoJSON into `public.landuse_predictions` tagged with `model_version`:

```bash
# To append a new region to an existing model version without deleting other regions:
python -m ml.src.ingest_postgis \
  --input outputs/region_v1.0.0.geojson \
  --model-version v1.0.0 \
  --no-replace-version \
  --batch-size 500
```

> **Note:** Pass `--replace-version` (default) if you are refreshing the entire dataset for that `model_version`, or `--no-replace-version` if you are appending an additional kabupaten/kota to the same version.

---

## Step 5: Canary Verification in Rapid Editor & Rollback Procedure

1. **Canary Test:** Query the newly ingested version explicitly before changing the default version:
   ```bash
   curl "http://localhost:8000/v1/landuse.geojson?bbox=106.45,-6.30,106.60,-6.15&version=v1.0.0"
   ```
2. **Promote to Default:** Once verified in Rapid Editor (`rapideditor.org`), update `DEFAULT_MODEL_VERSION` in `.env` / deployment environment variables.
3. **Instant Rollback:** If mappers report false positives or boundary regressions, set `DEFAULT_MODEL_VERSION=v1.0.0` (or append `?version=v1.0.0` in Rapid Editor) for an immediate zero-downtime rollback.
