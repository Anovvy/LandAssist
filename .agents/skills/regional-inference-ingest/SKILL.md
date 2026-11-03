---
name: regional-inference-ingest
description: Operational workflow for running anti-OOM sliding-window inference, GDAL sieve + UTM Chaikin vectorization, and zero-downtime versioned PostGIS ingestion for a new region or model version.
---

# Skill: Regional Inference & Zero-Downtime PostGIS Ingestion (`regional-inference-ingest`)

Use this skill when executing or automating an end-to-end regional inference run (e.g., updating Kabupaten Tangerang with a new model version `v1.1.0`, or expanding coverage to Kota Tangerang, Tangerang Selatan, or West Java) and ingesting the resulting polygons into Supabase PostGIS.

---

## Step 1: Pre-Flight Validation of the 64-Band GSE GeoTIFF

Before running inference, verify that the input raster satisfies LandAssist's geospatial and channel invariants:
1. Band count **MUST** be exactly `64` (Google Satellite Embeddings).
2. Spatial resolution should be ~10m (`0.0000898°` in `EPSG:4326` or `10.0m` in `EPSG:32748`).
3. Verify sufficient disk space in the temporary directory (`--tmp-dir`) for the `float16` `np.memmap` accumulators (`(num_classes, H, W) * 2 bytes` + `(H, W) * 2 bytes`).

```bash
python -c "import rasterio; ds = rasterio.open('data/region_gse64.tif'); assert ds.count == 64, f'Expected 64 bands, got {ds.count}'; print(ds.crs, ds.shape)"
```

---

## Step 2: Execute Anti-OOM Sliding-Window Inference (`ml/src/infer_sliding.py`)

Run `ml/src/infer_sliding.py` with `patch_size=160` and `stride=80` (50% overlap blended with a 2D Hann window):

```bash
python -m ml.src.infer_sliding \
  --input data/region_gse64.tif \
  --output outputs/region_pred_v1.1.0.tif \
  --weights weights/segformer_mit_b2_v1.1.0.pt \
  --patch-size 160 \
  --stride 80 \
  --batch-size 8
```

- **Output:** A 2-band GeoTIFF (`outputs/region_pred_v1.1.0.tif`) where Band 1 is the uint8 class ID (`0=Background`, `1=Industrial`, `2=Residential`) and Band 2 is the uint8 confidence score (`0..255` scaled from `[0.0, 1.0]`).

---

## Step 3: Run GDAL Sieve & Metric UTM Chaikin Vectorization (`ml/src/vectorize.py`)

Convert the raster prediction into clean, OSM-compliant polygons:

```bash
python -m ml.src.vectorize \
  --input outputs/region_pred_v1.1.0.tif \
  --output outputs/region_v1.1.0.geojson \
  --model-version v1.1.0 \
  --sieve-pixels 16 \
  --min-area-sqm 1600.0 \
  --chaikin-iterations 2 \
  --simplify-tolerance-m 5.0
```

- **Quality Check:** Verify that all output features have `geometry.type in ("Polygon", "MultiPolygon")`, coordinates in `EPSG:4326` (`lon, lat`), and `area_sqm >= 1600.0`.

---

## Step 4: Versioned PostGIS Ingestion (`ml/src/ingest_postgis.py`)

Ingest the GeoJSON into `public.landuse_predictions` tagged with the new `model_version` (`v1.1.0`) while leaving `v1.0.0` untouched:

```bash
python -m ml.src.ingest_postgis \
  --input outputs/region_v1.1.0.geojson \
  --model-version v1.1.0 \
  --chunk-size 500
```

> **Note:** Only pass `--replace-version` if you are re-ingesting the exact same `model_version` and want to replace its previous rows.

---

## Step 5: Canary Verification in Rapid Editor & Rollback Procedure

1. **Canary Test:** Query the newly ingested version explicitly before changing the default version:
   ```bash
   curl "http://localhost:8000/v1/landuse.geojson?bbox=106.45,-6.30,106.60,-6.15&version=v1.1.0"
   ```
2. **Promote to Default:** Once verified in Rapid Editor (`rapideditor.org`), update `DEFAULT_MODEL_VERSION=v1.1.0` in `.env` / deployment environment variables.
3. **Instant Rollback:** If mappers report false positives or boundary regressions, set `DEFAULT_MODEL_VERSION=v1.0.0` (or append `?version=v1.0.0` in Rapid Editor) for an immediate zero-downtime rollback.
