# LandAssist — Workspace Rules (`AGENTS.md`)

This document defines the mandatory architectural, geospatial, machine learning, and security rules for **LandAssist**. All Antigravity IDE agents, CLI workflows, subagents, and automated SDK Pull Request reviewers (`scripts/agy_code_review.py`) **MUST** enforce these rules on every change.

---

## Rule 1: Geospatial & OpenStreetMap Standards (`EPSG:4326` & Valid Geometries)

1. **Coordinate Reference System (CRS):**
   - All geometries stored in PostGIS (`public.landuse_predictions`) and served via FastAPI (`/v1/landuse.geojson`) **MUST** use **`EPSG:4326` (WGS84)** in standard **`(longitude, latitude)`** coordinate order.
   - Metric geometric operations (area calculation in $\text{m}^2$, Chaikin corner-cutting, Douglas-Peucker simplification) **MUST** first project to **UTM Zone 48S (`EPSG:32748`)** before reprojecting back to `EPSG:4326`.
2. **Topological Validity:**
   - Every polygon written during vectorization (`ml/src/vectorize.py`) or ingested into PostGIS (`ml/src/ingest_postgis.py`) **MUST** be topologically valid (`geom.buffer(0)` in Shapely and `ST_MakeValid(geom)` in PostGIS) and stored as `MultiPolygon, 4326`.
3. **OpenStreetMap Tagging Compliance:**
   - Every GeoJSON `Feature` returned to Rapid Editor **MUST** include valid OSM tags inside `properties`:
     - `landuse`: strictly `"industrial"` or `"residential"` (or an explicitly registered OSM landuse class).
     - `confidence`: float rounded to `[0.0, 1.0]`.
     - `area_sqm`: metric area $\ge 1600.0\text{ m}^2$ (default minimum threshold).
     - `model_version`: semantic version string (e.g., `"v1.0.0"`).
     - `source`: `"LandAssist AI (Sentinel-2 GSE)"`.

---

## Rule 2: PostGIS Query Performance & Spatial Indexing

1. **Mandatory Spatial Index Usage:**
   - Every SQL query touching `public.landuse_predictions` in `backend/app/routers/geojson.py` or `backend/app/routers/tiles.py` **MUST** filter geometries using the `GIST` index via `ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))` or `ST_TileEnvelope(:z, :x, :y)`.
   - **NEVER** execute unbounded full-table sequential scans on `public.landuse_predictions` in API endpoints.
2. **Parameterized SQL & Bounding Box Guards:**
   - All SQL queries **MUST** use parameterized bindings (`sqlalchemy.text` with bound parameters) to prevent SQL injection.
   - Bounding boxes (`bbox`) **MUST** be validated (`min_lon < max_lon`, `min_lat < max_lat`, and maximum span $\le 0.5^\circ \times 0.5^\circ$) before querying PostGIS.

---

## Rule 3: ML & Anti-OOM Architecture (64-Channel GSE)

1. **64-Channel SegFormer Input Contract:**
   - Input rasters are **64-band Google Satellite Embeddings (GSE)** (`shape = (64, H, W)`).
   - `ml/src/model.py` **MUST** configure `SegformerForSemanticSegmentation` (`nvidia/mit-b2`) with `num_channels=64` (`in_channels=64`) and `num_labels=3` (`0: Background`, `1: Industrial`, `2: Residential`). Never hardcode 3-channel RGB assumptions into dataset loaders or inference pipelines.
2. **Anti-OOM Sliding-Window Inference:**
   - Regional inference (`ml/src/infer_sliding.py`) **MUST NEVER** load an entire regional 64-band GeoTIFF into RAM at once.
   - Always read patches via `rasterio.windows.Window(col_off, row_off, patch_size, patch_size)` (`patch_size=160`, `stride=80`) and accumulate weighted probabilities + weights on disk using `np.memmap(..., dtype=np.float16)` blended with a **2D Hann Window** (`_create_hann_window_2d`).
3. **Cartographic Post-Processing:**
   - Raw class masks **MUST** pass through `rasterio.features.sieve` (`size=16` pixels = $1,600\text{ m}^2$ at 10m resolution) prior to polygonization, followed by **Chaikin Corner-Cutting** (`ml/src/vectorize.py`) in `EPSG:32748`.

---

## Rule 4: Versioning, Security & Git Hygiene

1. **Multi-Version Coexistence & Zero-Downtime Rollback:**
   - Never overwrite or `TRUNCATE` production predictions from previous model versions during ingestion unless `--replace-version` is explicitly passed for that specific `model_version`.
   - API endpoints (`/v1/landuse.geojson`, `/v1/tiles/{z}/{x}/{y}.pbf`) **MUST** support filtering by `?version=<semver>` and fall back to `settings.default_model_version`.
2. **Atomic & Scoped Conventional Commits:**
   - All commits **MUST** follow Conventional Commits (`feat(api):`, `feat(ml):`, `feat(db):`, `ci(devops):`, `fix(vectorize):`, `docs(readme):`) and group logically related files into atomic units.
3. **Zero Secrets & Zero Binary Bloat in Git:**
   - **NEVER** commit `.env` files, database credentials, API keys, raw `.tif`/`.tiff`/`.npy`/`.npz`/`.mmap` datasets, or `.pt`/`.pth`/`.safetensors` model weights to Git. Store model weights on Hugging Face Hub (`HF_MODEL_REPO`).
