---
name: add-osm-landuse-class
description: Step-by-step guide for adding a new OpenStreetMap landuse segmentation class (e.g., commercial, farmland, retail) across LandAssist's SegFormer ML pipeline, Chaikin vectorizer, PostGIS schema, and FastAPI endpoints.
---

# Skill: Add a New OpenStreetMap Landuse Class (`add-osm-landuse-class`)

Use this skill whenever the user asks to expand LandAssist beyond `industrial` and `residential` by adding a new OpenStreetMap `landuse=*` category (for example, `commercial`, `farmland`, `retail`, or `forest`).

## Prerequisites & Architectural Invariants

Before modifying any files, review `AGENTS.md`:
- Input tensors remain **64-band Google Satellite Embeddings (GSE)** (`in_channels=64`).
- Class `0` is always reserved for `background`.
- Every foreground class ID `k >= 1` maps deterministically to a valid OpenStreetMap `landuse=<tag>` string.

---

## Step 1: Update Model Class Definitions (`ml/src/model.py` & `ml/src/infer_sliding.py`)

1. Open `ml/src/model.py`:
   - Increment `NUM_CLASSES` (e.g., from `3` to `4`).
   - Update `ID2LABEL` and `LABEL2ID` dictionaries to include the new class index and OSM tag name (e.g., `3: "commercial"`).
   - Ensure `create_segformer_gse64(num_classes=NUM_CLASSES, ...)` passes the updated `num_labels`, `id2label`, and `label2id` to `SegformerConfig` / `SegformerForSemanticSegmentation`.
2. Open `ml/src/infer_sliding.py`:
   - Confirm `num_classes` defaults to `NUM_CLASSES` imported from `ml.src.model` so the disk-backed `np.memmap` probability accumulator allocates `(num_classes, H, W)` in `float16`.

---

## Step 2: Update Vectorizer Class Mapping (`ml/src/vectorize.py`)

1. Open `ml/src/vectorize.py`:
   - Update `CLASS_ID_TO_OSM_LANDUSE` to map the new integer class ID (e.g., `3: "commercial"`).
   - Verify that `sieve_and_vectorize()` loops over all foreground class IDs in `CLASS_ID_TO_OSM_LANDUSE`, applies `rasterio.features.sieve(..., size=sieve_pixels)` (`16` pixels = $1,600\text{ m}^2$), performs Chaikin smoothing in `EPSG:32748`, and writes `"landuse": class_name` in the output GeoJSON `properties`.

---

## Step 3: Update PostGIS Schema Constraint (`sql/001_init_postgis.sql`)

1. Open `sql/001_init_postgis.sql`:
   - Locate the `CHECK` constraint on `public.landuse_predictions.class_name`:
     ```sql
     class_name TEXT NOT NULL CHECK (class_name IN ('industrial', 'residential', 'commercial'))
     ```
   - Add a migration script (e.g., `sql/002_add_<class_name>_class.sql`) to alter the constraint safely in existing Supabase / PostGIS deployments without dropping existing prediction rows:
     ```sql
     ALTER TABLE public.landuse_predictions
       DROP CONSTRAINT IF EXISTS landuse_predictions_class_name_check;

     ALTER TABLE public.landuse_predictions
       ADD CONSTRAINT landuse_predictions_class_name_check
       CHECK (class_name IN ('industrial', 'residential', 'commercial'));
     ```

---

## Step 4: Update FastAPI Query Validation & Tests (`backend/app/routers/geojson.py` & `backend/tests/test_api.py`)

1. Open `backend/app/routers/geojson.py`:
   - Update the `class_name` query parameter documentation/validation (if restricted by an enum or regex) to allow filtering by `?class_name=<new_class>`.
   - Confirm the returned GeoJSON `Feature.properties.landuse` preserves the exact OSM tag (`"commercial"`, etc.).
2. Open `backend/tests/test_api.py`:
   - Add a test case verifying that `/v1/landuse.geojson` can return and filter features with the new `landuse` class.

---

## Step 5: Verification Checklist

Run the linter and unit test suite before reporting completion:

```bash
ruff check backend/ ml/src/ scripts/
pytest backend/tests/ -v
```
