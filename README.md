<p align="center">
  <img src="docs/assets/landassist_logo.jpg" alt="LandAssist Logo" width="550"/>
</p>

<h1 align="center">LandAssist 🛰️🗺️</h1>

<p align="center">
  <strong>AI-Assisted Landuse Segmentation & Vector Delivery System for OpenStreetMap Rapid Editor</strong><br>
  <em>Production Release: v0.1.0 — Pilot Area: Kabupaten Tangerang, Banten, Indonesia</em>
</p>

<p align="center">
  <a href="https://github.com/Anovvy/LandAssist/actions/workflows/ci-review.yml"><img src="https://img.shields.io/github/actions/workflow/status/Anovvy/LandAssist/ci-review.yml?branch=main&label=CI%20%26%20Antigravity%20Review&logo=github" alt="CI Status"></a>
  <a href="https://github.com/Anovvy/LandAssist/actions/workflows/cd-deploy.yml"><img src="https://img.shields.io/github/actions/workflow/status/Anovvy/LandAssist/cd-deploy.yml?branch=main&label=CD%20Container%20(GHCR)&logo=docker" alt="CD Status"></a>
  <a href="https://github.com/Anovvy/LandAssist/pkgs/container/landassist%2Flandassist-api"><img src="https://img.shields.io/badge/Container-ghcr.io-blue?logo=docker" alt="GHCR"></a>
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python" alt="Python Version"></a>
  <a href="https://postgis.net/"><img src="https://img.shields.io/badge/PostGIS-17%20%2F%203.3.7-336791?logo=postgresql" alt="PostGIS Version"></a>
  <a href="https://huggingface.co/docs/transformers/model_doc/segformer"><img src="https://img.shields.io/badge/Model-SegFormer%20MIT--B2%20(64--ch%20GSE)-orange" alt="Model Architecture"></a>
  <a href="https://rapideditor.org"><img src="https://img.shields.io/badge/Integrated-OSM%20Rapid%20Editor-brightgreen?logo=openstreetmap" alt="OpenStreetMap Rapid"></a>
</p>

---

## 📖 Executive Overview

**LandAssist** bridges 10-meter satellite Earth Observation and community-driven OpenStreetMap (OSM) mapping. Utilizing **64-channel Google Satellite Embeddings (GSE)** synthesized from Sentinel-2 Level-2A time-series multispectral imagery, LandAssist classifies **Industrial (`landuse=industrial`)** and **Residential (`landuse=residential`)** parcels across **Kabupaten Tangerang, Indonesia** (~30 km × 30 km AOI).

Raw raster predictions are converted into cartographically clean, topology-validated GIS polygons and streamed directly into **LandAssist Web Editor** (built-in standalone OSM web editor at `/editor`) and external GIS clients for human-in-the-loop verification.

> [!IMPORTANT]
> **Human-in-the-Loop Philosophy:** LandAssist strictly rejects automated bulk imports into OpenStreetMap. Every AI-generated polygon is served as an assistive reference layer in **LandAssist Web Editor**, empowering human mappers to review, edit, and approve each boundary before committing changesets to the global OSM database.

---

## 🏛️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Remote Sensing & Embedding Ingestion"]
        GEE["Google Earth Engine API\n(Sentinel-2 L2A Time-Series)"] --> GSE["Google Satellite Embeddings (GSE)\n64-Band Mosaics @ 10m Ground Resolution"]
    end

    subgraph DeepLearning["2. ML Segmentation & Anti-OOM Inference"]
        GSE --> DS["Training Dataset:\n125 Chips (160x160 px, 50% Overlap)"]
        DS --> SegFormer["SegFormer MIT-B2 (in_channels=64, 3 classes)\nFocal + Dice Loss | mIoU: 74.2% | F1: 0.81"]
        SegFormer --> Infer["Sliding-Window Inference\n2D Hann Window Blending + float16 np.memmap"]
    end

    subgraph Vectorization["3. Cartographic Post-Processing"]
        Infer --> Sieve["GDAL Sieve Filtering (Area >= 1,600 m²)"]
        Sieve --> Chaikin["Chaikin Corner-Cutting Smoothing\n(UTM Zone 48S / EPSG:32748)"]
        Chaikin --> Reproject["Topological Validation & Reprojection\n(WGS84 / EPSG:4326)"]
    end

    subgraph DataLayer["4. Cloud Spatial Storage & Fast API Delivery"]
        Reproject --> DB[("Supabase PostGIS 17 + PostGIS 3.3.7\nTable: landuse_predictions (106 Polygons)\nGIST Spatial Index + Supavisor IPv4 Pooler")]
        DB --> API["FastAPI Spatial Engine\nDual-Mode: PostGIS Pooler + Local GeoJSON Fallback"]
    end

    subgraph Presentation["5. Mapping Client & QA Interfaces"]
        API --> GeoJSON["GET /v1/landuse.geojson\n(OSM-Tagged FeatureCollection)"]
        API --> MVT["GET /v1/tiles/{z}/{x}/{y}.pbf\n(Binary Mapbox Vector Tiles)"]
        API --> QA["GET /qa\n(Leaflet + Satellite Visual QA Map)"]
        API --> BuiltInEditor["GET /editor\n(Embedded OSM Rapid Editor)"]
        GeoJSON --> PublicRapid["https://rapideditor.org/edit\n(Custom Data URL)"]
    end
```

---

## 🚀 Key Technical Highlights

### 1. 64-Band Google Satellite Embeddings (GSE)
- Uses **64-channel deep embeddings** capturing seasonal phenology, surface texture, and multispectral reflectance rather than simple 3-band RGB imagery.
- Adapts `transformers.SegformerForSemanticSegmentation` (`nvidia/mit-b2`) with a customized stage-0 patch embedding projection layer accepting `in_channels=64` without disrupting pretrained weights.
- Trained on **125 annotated chips** across Kabupaten Tangerang using **Focal + Dice Loss** with $90^\circ$ rotational and diagonal-flip augmentations to combat class imbalance.
- Checkpoint registered in [`weight/manifest.json`](weight/manifest.json) with SHA-256 integrity verification (`704e6e66...`).

### 2. Anti-OOM Sliding-Window Inference Engine
- Processes regional satellite rasters through windowed reads (`rasterio.windows.Window`).
- Incorporates **2D Hann Window blending** on overlapping tiles (50% stride) to completely eliminate tile-boundary seams.
- Utilizes disk-backed `float16 np.memmap` accumulators (`tmp_mmap/`), keeping runtime RAM strictly under 2.5 GB even on massive rasters.

### 3. Cartographic Polygonization & Smoothing
- **GDAL Sieve**: Removes salt-and-pepper noise smaller than 16 pixels ($1,600\text{ m}^2$, matching typical residential plot clusters).
- **Chaikin Corner Cutting**: Executes metric boundary smoothing in **UTM Zone 48S (`EPSG:32748`)** followed by Douglas-Peucker simplification ($5.0\text{ m}$ tolerance).
- **EPSG:4326 Re-projection**: Validates all geometries (`ST_IsValid` / `shapely.make_valid`) for standards-compliant OpenStreetMap ingestion.

### 4. Enterprise Spatial API & Supabase PostGIS 17
- Connects to **Supabase PostGIS 17 + PostGIS 3.3.7** via `asyncpg` with **Supavisor Transaction Pooler support (`port 6543`, IPv4)** and `statement_cache_size=0`.
- Currently holds **106 validated polygons** for the Kabupaten Tangerang pilot area.
- Employs **Dual-Mode Graceful Degradation**: If `DATABASE_URL` is omitted or temporarily unreachable, the API seamlessly falls back to an in-memory cached GeoJSON dataset with zero downtime.

### 5. Automated CI/CD & Antigravity AI Code Reviewer
- **CI Pipeline ([`.github/workflows/ci-review.yml`](.github/workflows/ci-review.yml))**:
  - Runs **Ruff Linter** across all Python modules.
  - Executes **7 automated unit tests** covering health, GeoJSON spatial bbox queries, CORS headers, and MVT delivery.
  - Spawns an automated **AI Code Reviewer** powered by **Google Gemini 2.5 Flash (`google-genai`)** on every Pull Request and direct push to `main`.
- **CD Pipeline ([`.github/workflows/cd-deploy.yml`](.github/workflows/cd-deploy.yml))**:
  - Builds and tags Docker container images with automatic lowercase GHCR sanitization (`ghcr.io/anovvy/landassist/landassist-api`).
  - Supports zero-downtime deployment triggers via webhook.

---

## 🌐 Endpoints & Interactive Interfaces

| Route | Method | Description | Content-Type |
| :--- | :---: | :--- | :--- |
| **`/editor`** | `GET` | **Integrated OSM Rapid Editor** running directly in the browser | `text/html` |
| **`/qa`** | `GET` | **Interactive Visual QA Map** (Leaflet + Google Satellite overlay) | `text/html` |
| **`/docs`** | `GET` | **Interactive OpenAPI / Swagger Documentation** | `text/html` |
| **`/healthz`** | `GET` | Service readiness probe, version, and active DB mode (`postgis` / `demo_fallback`) | `application/json` |
| **`/v1/versions`** | `GET` | Inventory of available model versions for instant rollback | `application/json` |
| **`/v1/landuse.geojson`** | `GET` | OSM-tagged GeoJSON FeatureCollection with `bbox` and `class_name` filters | `application/geo+json` |
| **`/v1/tiles/{z}/{x}/{y}.pbf`** | `GET` | Binary Mapbox Vector Tiles generated dynamically via PostGIS `ST_TileEnvelope` | `application/vnd.mapbox-vector-tile` |

---

## 🗺️ LandAssist Web Editor & OpenStreetMap Integration

Mappers can inspect and incorporate LandAssist predictions using either the standalone built-in web editor or external GIS clients:

### Option A: Standalone LandAssist Web Editor (`/editor`) — Recommended
LandAssist provides its own **standalone AI-assisted OpenStreetMap web editor** based on the iD / Rapid v2 engine, deeply integrated with the FastAPI backend without external catalog constraints.

1. Start the FastAPI server locally:
   ```bash
   uvicorn app.main:app --app-dir backend --reload --port 8000
   ```
2. Navigate to **[http://localhost:8000/editor](http://localhost:8000/editor)**.
3. The editor automatically centers on the **Kabupaten Tangerang** AOI (`106.50, -6.22`, zoom 14.5) with satellite basemap and AI predictions active:
   - 🏭 **Industrial Parcels**: Rendered with **Amber Gold (`#F59E0B`)** high-visibility outlines.
   - 🏡 **Residential Parcels**: Rendered with **Emerald Green (`#10B981`)** outlines.
4. Click any AI polygon to open the **LandAssist AI Inspector**:
   - Displays real-time **Model Confidence** (e.g. `94.0%`), **Estimated Area** in Hectares & m², and data source metadata.
   - Click **`+ Add This Feature to OSM`**: The polygon **immediately transforms** into the native iD editor style, color, and preset (`landuse=residential` in native gold/amber area styling or `landuse=industrial` in native pink/purple area styling) with pre-populated tags (`source=LandAssist AI (Sentinel-2 GSE)`, `model_version=v1.0.0`).
   - Click **`- Ignore This Feature`**: Dismisses false positives from the current curation session.
   - Full Undo/Redo (`Ctrl+Z` / `Ctrl+Y`) and instant vertex editing.

### Option B: Public Rapid Editor ([rapideditor.org](https://rapideditor.org/edit))
1. Navigate to **[https://rapideditor.org/edit](https://rapideditor.org/edit)** and pan to **Kabupaten Tangerang** (e.g. Balaraja, Cikupa, Curug, or Tigaraksa around `106.50, -6.22`, zoom level 14+).
2. Press `F` (or open the **Map Data** pane on the right toolbar).
3. Under **Custom Data**, click **Add Custom Data URL** and paste:
   ```text
   http://localhost:8000/v1/landuse.geojson
   ```
   *(Or your production HTTPS tunnel domain, e.g. `https://<your-api-domain>/v1/landuse.geojson`)*
4. All AI-segmented industrial (`#F59E0B`) and residential (`#10B981`) boundaries will render instantly with standardized OSM tags.
5. Click **Add This Feature to OSM** to accept and convert any polygon into a live OpenStreetMap feature.

---

## 🛠️ Local Quickstart Guide

### 1. Prerequisites
- **Python 3.11+** or **3.12**
- **Git**
- *(Optional)* **Docker & Docker Compose** for containerized execution.

### 2. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Anovvy/LandAssist.git
cd LandAssist

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install backend & testing dependencies
pip install -r backend/requirements.txt
```

### 3. Environment Configuration
Copy the template and configure your credentials:
```bash
cp .env.example .env
```
Key configuration parameters in `.env`:
```ini
# Supabase PostGIS IPv4 Transaction Pooler (Port 6543)
DATABASE_URL=postgresql://postgres.[PROJECT_REF]:[ENCODED_PASSWORD]@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres

# Google AI Studio API Key for Antigravity PR Reviews
GEMINI_API_KEY=your_gemini_api_key_here

# Default Model Serving Version
DEFAULT_MODEL_VERSION=v1.0.0
PORT=8000
```

> [!TIP]
> If your database password contains special characters (`@`, `#`, `$`, `%`), encode it using the built-in utility:
> ```bash
> python scripts/test_db_connection.py --encode-password "YourP@ssw0rd!"
> ```

### 4. Verify Database Connection
Run the connection diagnostic utility to verify PostgreSQL version, PostGIS extension, and spatial indexes:
```bash
python scripts/test_db_connection.py
```

### 5. Launch the Server Locally
```bash
uvicorn app.main:app --app-dir backend --reload --port 8000
```
- Open Swagger Documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**
- Open Visual QA Map: **[http://localhost:8000/qa](http://localhost:8000/qa)**
- Open Rapid Editor: **[http://localhost:8000/editor](http://localhost:8000/editor)**

---

## 🧪 Testing & Code Quality

LandAssist maintains a strict 100% test pass rate and clean linting standards.

### Run Ruff Linter
```bash
python -m ruff check backend/ ml/src/ scripts/
# Output: All checks passed!
```

### Run FastAPI & PostGIS Pytest Suite (7 Tests)
```bash
python -m pytest backend/tests -v
```
**Test Results:**
```text
backend/tests/test_api.py::test_healthz_returns_200_and_version            PASSED [ 14%]
backend/tests/test_api.py::test_get_versions_returns_available_model_versions PASSED [ 28%]
backend/tests/test_api.py::test_landuse_geojson_returns_valid_osm_feature_collection PASSED [ 42%]
backend/tests/test_api.py::test_landuse_geojson_bbox_and_class_filtering   PASSED [ 57%]
backend/tests/test_api.py::test_landuse_geojson_invalid_bbox_returns_400  PASSED [ 71%]
backend/tests/test_api.py::test_cors_headers_for_rapid_editor              PASSED [ 85%]
backend/tests/test_api.py::test_mvt_tile_endpoint                          PASSED [100%]

============================== 7 passed in 1.25s ==============================
```

### Run Antigravity AI Code Review (Dry-Run Mode)
```bash
python scripts/agy_code_review.py --dry-run
# Output: [Antigravity Review] PR passed all LandAssist workspace rules.
```

---

## 🔄 Multi-Pillar Versioning & Zero-Downtime Rollback

LandAssist enforces end-to-end traceability across 5 pillars:

| Pillar | Governance Mechanism | Active Pilot Value |
| :--- | :--- | :--- |
| **1. Source Code** | Git Semantic Versioning Tags | `v0.1.0` |
| **2. Model Weights** | SHA-256 Digest in `weight/manifest.json` | `segformer-mitb2-gse64-v1.0.0.pt` |
| **3. Training Dataset** | Versioned Patch Registry | `tangerang-125chips-v1.0` |
| **4. Database Polygons** | `model_version` column in `public.landuse_predictions` | `v1.0.0` (106 polygons) |
| **5. API Delivery** | Query parameter `?version=` & URL prefix `/v1/` | `GET /v1/landuse.geojson?version=v1.0.0` |

### Zero-Downtime Rollback Protocol
When new predictions (e.g. `v1.1.0`) are ingested into PostGIS, previous `v1.0.0` polygons are never overwritten.
- **A/B Testing:** Mappers can compare versions simply by toggling `?version=v1.1.0` in Rapid Editor.
- **Instant Rollback:** If regressions occur, revert `DEFAULT_MODEL_VERSION=v1.0.0` in `.env` to switch all production traffic in **0.0 seconds** without re-running ML pipelines.

---

## 📁 Repository Structure

```text
LandAssist/
├── .agents/                        # Antigravity IDE Agent customizations
│   ├── hooks.json                  # Automated quality hooks
│   └── skills/                     # Domain-specific operational playbooks
│       ├── add-osm-landuse-class/  # Skill for expanding landuse classes
│       └── regional-inference-ingest/ # Skill for regional AOI scaling
├── .github/
│   └── workflows/
│       ├── ci-review.yml           # Ruff lint, Pytest (7 tests), Gemini AI review
│       └── cd-deploy.yml           # Docker Buildx multi-layer build & push to GHCR
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application, CORS, routers & /editor mount
│   │   ├── config.py               # Pydantic Settings management
│   │   ├── db.py                   # Asyncpg PostGIS pooler & GeoJSON fallback
│   │   └── routers/
│   │       ├── health.py           # /healthz, /v1/versions, /qa visual viewer
│   │       ├── geojson.py          # /v1/landuse.geojson spatial bbox delivery
│   │       └── tiles.py            # /v1/tiles/{z}/{x}/{y}.pbf Mapbox MVT
│   ├── tests/
│   │   └── test_api.py             # 7 FastAPI + PostGIS endpoint unit tests
│   ├── Dockerfile                  # Production container (Cloud Run PORT binding)
│   └── requirements.txt            # Minimal runtime dependencies
├── data/                           # Regional GeoTIFF rasters (Git-ignored)
├── docs/
│   └── assets/
│       └── landassist_logo.jpg     # Official LandAssist Hero Logo
├── frontend/                       # OpenStreetMap Rapid Editor static distribution
├── ml/
│   ├── notebooks/
│   │   └── finetune_segformer_64ch_colab.ipynb # Google Colab T4 GPU fine-tuning
│   ├── src/
│   │   ├── download_gse.py         # Automated Google Earth Engine 64-band downloader
│   │   ├── model.py                # 64-channel SegFormer MIT-B2 architecture
│   │   ├── infer_tangerang.py      # Anti-OOM 2D Hann sliding-window inference
│   │   ├── vectorize.py            # Sieve, UTM 48S Chaikin smoothing & QA generator
│   │   └── ingest_postgis.py       # High-performance PostGIS batch ingestion
│   └── requirements-ml.txt         # PyTorch, rasterio, geopandas dependencies
├── outputs/
│   ├── tangerang_v1.0.0.geojson    # 106 validated OSM polygons (Kabupaten Tangerang)
│   └── tangerang_v1.0.0_qa.html    # Standalone Leaflet Satellite Visual QA Map
├── scripts/
│   ├── agy_code_review.py          # Automated Gemini 2.5 Flash / Static PR Reviewer
│   ├── manage_weights.py           # GitHub Releases weight asset manager
│   └── test_db_connection.py       # Supabase PostGIS connection & pooler diagnostic
├── sql/
│   └── 001_init_postgis.sql        # PostGIS schema, GIST indexes & MVT function
├── weight/
│   ├── manifest.json               # SHA-256 weight integrity & metrics registry
│   ├── README.md                   # Weights documentation
│   └── RELEASE_NOTES_v1.0.0.md     # Checkpoint release notes
├── .env.example                    # Environment template with Supabase guide
├── .gitignore                      # Strict isolation for rasters, weights, and caches
├── AGENTS.md                       # Antigravity AI Agent workspace rules
├── docker-compose.yml              # Local containerized PostGIS + FastAPI stack
└── pyproject.toml                  # Ruff & Pytest tool configuration
```

---

## 👥 Contributing & Licensing

LandAssist is developed as an open geospatial AI research and mapping enablement platform.
- **Code License:** Apache 2.0
- **Data Attributions:**
  - Contains modified Copernicus Sentinel-2 data processed via Google Earth Engine.
  - Vector layers conform to OpenStreetMap Open Database License (ODbL).
- **Maintainer:** [Anovvy](https://github.com/Anovvy) & the LandAssist Engineering Team.
