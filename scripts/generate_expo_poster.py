#!/usr/bin/env python3
"""Generate a professional A3 Expo Showcase Poster for LandAssist.

Produces:
1. docs/assets/poster_a3.html (Self-contained, responsive, printable A3 HTML)
2. docs/assets/LandAssist_Expo_Poster_A3.pdf (Print-ready A3 vector PDF via Chrome Headless)
3. docs/assets/LandAssist_Expo_Poster_A3.png (High-resolution preview image)
"""

import base64
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"
POSTER_ASSETS = DOCS_ASSETS / "poster_assets"


def to_b64(path: Path) -> str:
    """Encode file to base64 data URI."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    else:
        mime = "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


def generate_poster_html() -> Path:
    """Generate self-contained HTML for the A3 poster."""
    qr_editor_b64 = to_b64(POSTER_ASSETS / "qr_landassist_editor.png")
    qr_github_b64 = to_b64(POSTER_ASSETS / "qr_landassist_github.png")
    ym_logo_b64 = to_b64(POSTER_ASSETS / "youthmappers_unj_logo.png")
    map_bg_b64 = to_b64(POSTER_ASSETS / "map_abstract_pattern.png")
    landassist_logo_b64 = to_b64(DOCS_ASSETS / "landassist_logo.jpg")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LandAssist — A3 Expo Showcase Poster</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@500;700&display=swap');

    @page {{
      size: 297mm 420mm;
      margin: 0;
    }}

    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    :root {{
      --primary-orange: #E85D25;
      --primary-orange-dark: #C2410C;
      --accent-green: #15803D;
      --accent-green-dark: #166534;
      --accent-teal: #0E7490;
      --accent-cyan: #0284C7;
      --navy-900: #0F172A;
      --navy-800: #1E293B;
      --navy-700: #334155;
      --slate-600: #475569;
      --slate-500: #64748B;
      --slate-400: #94A3B8;
      --slate-200: #E2E8F0;
      --slate-100: #F1F5F9;
      --bg-cream: #F8FAFC;
      --card-white: #FFFFFF;
    }}

    html, body {{
      width: 297mm;
      height: 420mm;
      margin: 0;
      padding: 0;
      overflow: hidden;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background-color: var(--bg-cream);
      color: var(--navy-900);
      line-height: 1.4;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    .poster-container {{
      width: 297mm;
      height: 420mm;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      background: #F8FAFC;
      overflow: hidden;
    }}

    /* HEADER SYSTEM */
    .header-system {{
      position: relative;
      background-color: #FFFFFF;
      border-bottom: 4px solid var(--primary-orange);
      flex-shrink: 0;
    }}

    .header-top-pattern {{
      height: 72mm;
      width: 100%;
      background-image: url('{map_bg_b64}');
      background-size: cover;
      background-position: center 30%;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16mm;
    }}

    .header-top-pattern::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.7) 70%, #FFFFFF 100%);
    }}

    .header-institution-tag {{
      position: absolute;
      top: 6mm;
      left: 16mm;
      z-index: 5;
      font-family: 'Montserrat', sans-serif;
      font-size: 8.5pt;
      font-weight: 800;
      color: var(--navy-900);
      text-transform: uppercase;
      letter-spacing: 1.2px;
      background: rgba(255,255,255,0.95);
      padding: 2.2mm 5.5mm;
      border-radius: 4mm;
      box-shadow: 0 3px 10px rgba(0,0,0,0.08);
      border: 1px solid rgba(0,0,0,0.06);
    }}

    .header-expo-tag {{
      position: absolute;
      top: 6mm;
      right: 16mm;
      z-index: 5;
      font-family: 'Montserrat', sans-serif;
      font-size: 8.5pt;
      font-weight: 800;
      color: var(--primary-orange);
      text-transform: uppercase;
      letter-spacing: 1.2px;
      background: rgba(255,255,255,0.95);
      padding: 2.2mm 5.5mm;
      border-radius: 4mm;
      box-shadow: 0 3px 10px rgba(0,0,0,0.08);
      border: 1px solid rgba(232,93,37,0.2);
    }}

    .logo-badge-container {{
      position: absolute;
      left: 50%;
      transform: translateX(-50%);
      top: 14mm;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 10mm;
    }}

    .circle-logo-card {{
      width: 40mm;
      height: 40mm;
      background: #FFFFFF;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 12px 28px rgba(0,0,0,0.18), 0 0 0 4px #FFFFFF, 0 0 0 8px var(--primary-orange);
      overflow: hidden;
    }}

    .circle-logo-card.secondary-logo {{
      box-shadow: 0 12px 28px rgba(0,0,0,0.18), 0 0 0 4px #FFFFFF, 0 0 0 8px var(--accent-teal);
    }}

    .circle-logo-card img {{
      width: 84%;
      height: 84%;
      object-fit: contain;
    }}

    /* HERO ORANGE BANNER */
    .hero-orange-block {{
      background: linear-gradient(135deg, #E85D25 0%, #D84F17 100%);
      padding: 10mm 16mm 8mm 16mm;
      position: relative;
      text-align: center;
      color: #FFFFFF;
    }}

    .hero-main-title {{
      font-family: 'Montserrat', sans-serif;
      font-size: 42pt;
      font-weight: 900;
      letter-spacing: 3px;
      text-transform: uppercase;
      line-height: 1;
      margin-bottom: 2.5mm;
      text-shadow: 0 3px 8px rgba(0,0,0,0.2);
    }}

    .hero-subtitle {{
      font-family: 'Inter', sans-serif;
      font-size: 12pt;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: #FFF2EB;
      margin-bottom: 3.5mm;
    }}

    .hero-pills-row {{
      display: flex;
      justify-content: center;
      gap: 3.5mm;
      flex-wrap: wrap;
    }}

    .hero-pill {{
      background: rgba(0, 0, 0, 0.25);
      padding: 1.5mm 4mm;
      border-radius: 3.5mm;
      font-size: 8pt;
      font-weight: 700;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: #FFFFFF;
      border: 1px solid rgba(255,255,255,0.3);
    }}

    /* GREEN BANNER STRIP */
    .banner-green-strip {{
      background-color: var(--accent-green);
      color: #FFFFFF;
      padding: 3.2mm 16mm;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: 'Montserrat', sans-serif;
      font-size: 9pt;
      font-weight: 800;
      letter-spacing: 1px;
      text-transform: uppercase;
    }}

    /* METRIC CARDS ROW */
    .metrics-ribbon {{
      padding: 6mm 16mm 5mm 16mm;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 5mm;
      flex-shrink: 0;
    }}

    .metric-card {{
      background: #FFFFFF;
      border-radius: 4.5mm;
      padding: 4mm 5mm;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
      border: 1.5px solid var(--slate-200);
      position: relative;
      border-top: 5px solid var(--primary-orange);
    }}

    .metric-card:nth-child(2) {{ border-top-color: var(--accent-teal); }}
    .metric-card:nth-child(3) {{ border-top-color: var(--accent-green); }}
    .metric-card:nth-child(4) {{ border-top-color: var(--navy-900); }}

    .metric-number {{
      font-family: 'Montserrat', sans-serif;
      font-size: 21pt;
      font-weight: 900;
      color: var(--navy-900);
      line-height: 1;
      margin-bottom: 1.5mm;
    }}

    .metric-label {{
      font-size: 8.5pt;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--slate-700);
      letter-spacing: 0.5px;
      margin-bottom: 1mm;
    }}

    .metric-subtext {{
      font-size: 7.2pt;
      font-weight: 500;
      color: var(--slate-500);
      line-height: 1.25;
    }}

    /* MAIN CONTENT 2-COLUMN GRID */
    .main-grid {{
      flex: 1;
      padding: 0 16mm 5mm 16mm;
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 6mm;
      overflow: hidden;
    }}

    .col-left, .col-right {{
      display: flex;
      flex-direction: column;
      gap: 5.5mm;
    }}

    /* SECTION CARD SYSTEM */
    .section-card {{
      background: #FFFFFF;
      border-radius: 4.5mm;
      padding: 5mm 5.5mm;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
      border: 1.5px solid var(--slate-200);
    }}

    .card-header {{
      display: flex;
      align-items: center;
      gap: 3.5mm;
      margin-bottom: 3.5mm;
      padding-bottom: 2.2mm;
      border-bottom: 1.5px solid var(--slate-100);
    }}

    .step-badge {{
      width: 7.5mm;
      height: 7.5mm;
      border-radius: 2.5mm;
      background: var(--primary-orange);
      color: #FFFFFF;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Montserrat', sans-serif;
      font-weight: 900;
      font-size: 9pt;
      flex-shrink: 0;
    }}

    .step-badge.teal {{ background: var(--accent-teal); }}
    .step-badge.green {{ background: var(--accent-green); }}
    .step-badge.navy {{ background: var(--navy-900); }}

    .card-title {{
      font-family: 'Montserrat', sans-serif;
      font-size: 11pt;
      font-weight: 800;
      color: var(--navy-900);
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }}

    .card-body-text {{
      font-size: 8.2pt;
      color: var(--navy-800);
      line-height: 1.48;
      margin-bottom: 2.8mm;
    }}

    /* PIPELINE GRID */
    .pipeline-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 3.5mm;
    }}

    .pipe-box {{
      background: var(--slate-100);
      border-radius: 3.5mm;
      padding: 3.5mm;
      border-left: 4px solid var(--primary-orange);
    }}

    .pipe-box.stage2 {{ border-left-color: var(--accent-teal); }}
    .pipe-box.stage3 {{ border-left-color: var(--accent-green); }}
    .pipe-box.stage4 {{ border-left-color: var(--navy-900); }}

    .pipe-tag {{
      font-family: 'Fira Code', monospace;
      font-size: 6.8pt;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--primary-orange-dark);
      margin-bottom: 1mm;
      display: inline-block;
    }}
    .pipe-box.stage2 .pipe-tag {{ color: var(--accent-teal); }}
    .pipe-box.stage3 .pipe-tag {{ color: var(--accent-green); }}
    .pipe-box.stage4 .pipe-tag {{ color: var(--navy-900); }}

    .pipe-title {{
      font-size: 8.5pt;
      font-weight: 800;
      color: var(--navy-900);
      margin-bottom: 1.2mm;
    }}

    .pipe-desc {{
      font-size: 7.2pt;
      color: var(--slate-600);
      line-height: 1.38;
    }}

    /* TAXONOMY */
    .taxonomy-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 3mm;
      margin-top: 1mm;
    }}

    .tax-card {{
      background: #FFFFFF;
      border: 1px solid var(--slate-200);
      border-radius: 3mm;
      padding: 3mm;
      text-align: center;
    }}

    .tax-color {{
      width: 100%;
      height: 3.5mm;
      border-radius: 1.5mm;
      margin-bottom: 1.8mm;
    }}

    .tax-title {{
      font-size: 8.2pt;
      font-weight: 800;
      color: var(--navy-900);
      margin-bottom: 0.8mm;
    }}

    .tax-tag {{
      font-family: 'Fira Code', monospace;
      font-size: 6.8pt;
      color: var(--primary-orange-dark);
      background: #FFF3EB;
      padding: 0.6mm 2mm;
      border-radius: 1.5mm;
      display: inline-block;
    }}

    /* BENCHMARK TABLE */
    .benchmark-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 7.6pt;
      margin-top: 2mm;
    }}

    .benchmark-table th {{
      background: var(--navy-900);
      color: #FFFFFF;
      text-align: left;
      padding: 2.2mm 3mm;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 7.2pt;
      letter-spacing: 0.5px;
    }}

    .benchmark-table th:first-child {{ border-radius: 2.5mm 0 0 0; }}
    .benchmark-table th:last-child {{ border-radius: 0 2.5mm 0 0; }}

    .benchmark-table td {{
      padding: 2.2mm 3mm;
      border-bottom: 1px solid var(--slate-200);
      color: var(--navy-800);
    }}

    .benchmark-table tr:last-child td {{
      border-bottom: none;
    }}

    .benchmark-table tr:nth-child(even) {{
      background: var(--slate-100);
    }}

    .badge-win {{
      color: var(--accent-green-dark);
      font-weight: 800;
    }}

    /* QR CODE INTERACTION CENTER (The Highlight for Expo Passersby) */
    .qr-showcase-card {{
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
      color: #FFFFFF;
      border-radius: 5mm;
      padding: 5mm 5.5mm;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
      border: 2px solid #334155;
    }}

    .qr-showcase-title {{
      font-family: 'Montserrat', sans-serif;
      font-size: 12pt;
      font-weight: 900;
      color: #FFFFFF;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 1.2mm;
      display: flex;
      align-items: center;
      gap: 3mm;
    }}

    .qr-showcase-subtitle {{
      font-size: 7.6pt;
      color: var(--slate-400);
      margin-bottom: 4mm;
      line-height: 1.35;
    }}

    .qr-pair-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4mm;
    }}

    .qr-box {{
      background: #FFFFFF;
      border-radius: 4mm;
      padding: 3.5mm 3mm;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      color: var(--navy-900);
      border: 2.5px solid var(--primary-orange);
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}

    .qr-box.github-box {{
      border-color: var(--accent-cyan);
    }}

    .qr-img-wrapper {{
      width: 38mm;
      height: 38mm;
      background: #FFFFFF;
      padding: 1.5mm;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 2mm;
    }}

    .qr-img-wrapper img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}

    .qr-box-title {{
      font-family: 'Montserrat', sans-serif;
      font-size: 9pt;
      font-weight: 800;
      color: var(--primary-orange-dark);
      text-transform: uppercase;
      line-height: 1.1;
      margin-bottom: 1mm;
    }}

    .qr-box.github-box .qr-box-title {{
      color: var(--accent-cyan);
    }}

    .qr-box-desc {{
      font-size: 6.8pt;
      color: var(--slate-600);
      line-height: 1.25;
      font-weight: 600;
      margin-bottom: 1.5mm;
    }}

    .qr-url-pill {{
      font-family: 'Fira Code', monospace;
      font-size: 5.6pt;
      font-weight: 700;
      background: var(--slate-100);
      padding: 1mm 2mm;
      border-radius: 1.8mm;
      color: var(--navy-900);
      white-space: nowrap;
      letter-spacing: -0.2px;
    }}

    /* FOOTER SYSTEM */
    .footer-system {{
      background: #FFFFFF;
      border-top: 2px solid var(--slate-200);
      padding: 4mm 16mm 4.5mm 16mm;
      display: flex;
      flex-direction: column;
      gap: 2.5mm;
      flex-shrink: 0;
    }}

    .footer-credits-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 7.6pt;
      color: var(--navy-800);
    }}

    .team-highlight {{
      font-weight: 700;
      color: var(--navy-900);
    }}

    .footer-partners-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 2mm;
      border-top: 1px solid var(--slate-200);
      font-size: 6.8pt;
      color: var(--slate-500);
      font-weight: 600;
    }}

    .bottom-accent-bar {{
      height: 3mm;
      width: 100%;
      background: linear-gradient(90deg, #E85D25 0%, #E85D25 33%, #15803D 33%, #15803D 66%, #0E7490 66%, #0E7490 100%);
      flex-shrink: 0;
    }}
  </style>
</head>
<body>
  <div class="poster-container">

    <!-- TOP HEADER -->
    <header class="header-system">
      <div class="header-top-pattern">
        <div class="header-institution-tag">Universitas Negeri Jakarta • YouthMappers Chapter</div>

        <!-- CENTER CIRCULAR LOGOS -->
        <div class="logo-badge-container">
          <div class="circle-logo-card" title="YouthMappers at UNJ">
            <img src="{ym_logo_b64}" alt="YouthMappers UNJ Logo">
          </div>
          <div class="circle-logo-card secondary-logo" title="LandAssist AI System">
            <img src="{landassist_logo_b64}" alt="LandAssist Logo">
          </div>
        </div>

        <div class="header-expo-tag">Geospatial AI Expo 2026</div>
      </div>

      <!-- ORANGE HERO BLOCK -->
      <div class="hero-orange-block">
        <h1 class="hero-main-title">LANDASSIST</h1>
        <p class="hero-subtitle">AI-Powered Earth Observation & Vector Delivery System for OpenStreetMap</p>
        <div class="hero-pills-row">
          <span class="hero-pill">Target AOI: Kabupaten Tangerang (1,193 km²)</span>
          <span class="hero-pill">Foundation Model: Google Satellite Embeddings 64-Band</span>
          <span class="hero-pill">Deep Learning: SegFormer MiT-B2</span>
          <span class="hero-pill">Cloud Vector Engine: PostGIS + FastAPI</span>
        </div>
      </div>

      <!-- GREEN BANNER STRIP -->
      <div class="banner-green-strip">
        <span>Making Maps Together with Multimodal Geospatial Machine Learning</span>
        <span>OpenStreetMap-Compliant Feature Delivery</span>
      </div>
    </header>

    <!-- KEY METRICS ROW -->
    <section class="metrics-ribbon">
      <div class="metric-card">
        <div class="metric-number">1,193 km²</div>
        <div class="metric-label">Regional Coverage</div>
        <div class="metric-subtext">Entire county boundary mapped via 2x2 grid tile partitioning</div>
      </div>
      <div class="metric-card">
        <div class="metric-number">64-Band</div>
        <div class="metric-label">Multimodal GSE</div>
        <div class="metric-subtext">Google Satellite Embeddings @ 10m GSD (Sentinel-2 annual composite)</div>
      </div>
      <div class="metric-card">
        <div class="metric-number">70.86%</div>
        <div class="metric-label">SegFormer mIoU</div>
        <div class="metric-subtext">Test evaluation on residential & industrial ground truth labels</div>
      </div>
      <div class="metric-card">
        <div class="metric-number">&lt; 200 ms</div>
        <div class="metric-label">Vector Delivery</div>
        <div class="metric-subtext">Sub-second spatial queries via PostGIS GIST index & streaming GeoJSON</div>
      </div>
    </section>

    <!-- MAIN 2-COLUMN GRID -->
    <main class="main-grid">

      <!-- LEFT COLUMN -->
      <div class="col-left">

        <!-- CARD 1 -->
        <div class="section-card">
          <div class="card-header">
            <div class="step-badge">01</div>
            <div class="card-title">Challenge & Autonomous Geospatial Vision</div>
          </div>
          <p class="card-body-text">
            Traditional mapping of rapidly urbanizing regions like <strong>Kabupaten Tangerang (~1,193 km²)</strong> requires hundreds of hours of manual remote tracing. Human digitizers face high cognitive fatigue, subjective boundary placement, and prolonged turnaround times.
          </p>
          <p class="card-body-text">
            <strong>LandAssist</strong> solves this with a foundation-model pipeline that converts 64-band multispectral embeddings into topologically regularized, vector-clean OpenStreetMap polygons served directly to mappers in real time.
          </p>
        </div>

        <!-- CARD 2 -->
        <div class="section-card">
          <div class="card-header">
            <div class="step-badge teal">02</div>
            <div class="card-title">End-to-End Deep Learning & Vector Pipeline</div>
          </div>
          <div class="pipeline-grid">
            <div class="pipe-box stage1">
              <span class="pipe-tag">Stage 1 • Acquisition</span>
              <div class="pipe-title">Earth Engine 64-Band GSE</div>
              <div class="pipe-desc">Partitions AOI into 4 sub-tiles (mitigating GEE 48 MB limit), acquiring annual Sentinel-2 latent embeddings at 10m ground resolution.</div>
            </div>
            <div class="pipe-box stage2">
              <span class="pipe-tag">Stage 2 • Segmentation</span>
              <div class="pipe-title">SegFormer MiT-B2 Inference</div>
              <div class="pipe-desc">Custom 64-input patch stem with 2D Hann window blending over disk memory-mapped buffers. Fast GPU T4 batched execution (~2 min).</div>
            </div>
            <div class="pipe-box stage3">
              <span class="pipe-tag">Stage 3 • Regularization</span>
              <div class="pipe-title">GDAL Sieve & Chaikin Curvature</div>
              <div class="pipe-desc">Eliminates noise (< 1,600 m² MMU) and applies Chaikin corner-cutting in UTM Zone 48S (EPSG:32748) to smooth pixel-step staircase artifacts.</div>
            </div>
            <div class="pipe-box stage4">
              <span class="pipe-tag">Stage 4 • Vector Ingestion</span>
              <div class="pipe-title">PostGIS & FastAPI Vector Tiles</div>
              <div class="pipe-desc">Streamed ingestion into Supabase PostGIS with GIST indexing; served over REST GeoJSON and Mapbox Vector Tiles (MVT) with CORS.</div>
            </div>
          </div>
        </div>

        <!-- CARD 3 -->
        <div class="section-card">
          <div class="card-header">
            <div class="step-badge green">03</div>
            <div class="card-title">OSM Landuse Taxonomy & Class Mapping</div>
          </div>
          <p class="card-body-text">
            LandAssist models strictly output OpenStreetMap-compliant attribute tags with lineage tracking (<code style="font-family:'Fira Code'; font-size:7.2pt; color:var(--primary-orange-dark);">source=LandAssist_v1.0.0;GSE</code>):
          </p>
          <div class="taxonomy-row">
            <div class="tax-card">
              <div class="tax-color" style="background:#94A3B8;"></div>
              <div class="tax-title">Background</div>
              <span class="tax-tag">class=0</span>
              <p style="font-size:6.6pt; color:var(--slate-500); margin-top:1.2mm;">Vegetation, water bodies, agriculture, open rural land.</p>
            </div>
            <div class="tax-card">
              <div class="tax-color" style="background:#EA580C;"></div>
              <div class="tax-title">Industrial</div>
              <span class="tax-tag">landuse=industrial</span>
              <p style="font-size:6.6pt; color:var(--slate-500); margin-top:1.2mm;">Warehouses, manufacturing plants, industrial estates.</p>
            </div>
            <div class="tax-card">
              <div class="tax-color" style="background:#16A34A;"></div>
              <div class="tax-title">Residential</div>
              <span class="tax-tag">landuse=residential</span>
              <p style="font-size:6.6pt; color:var(--slate-500); margin-top:1.2mm;">High & medium density settlements, villages, suburban blocks.</p>
            </div>
          </div>
        </div>

      </div>

      <!-- RIGHT COLUMN -->
      <div class="col-right">

        <!-- CARD 4 -->
        <div class="section-card">
          <div class="card-header">
            <div class="step-badge navy">04</div>
            <div class="card-title">Standalone Web Editor (Render Cloud 24/7)</div>
          </div>
          <p class="card-body-text">
            LandAssist hosts a standalone, customized Rapid/iD Web Editor deployed in the cloud. Mappers can visualize predicted polygons seamlessly overlaid on real-world high-resolution satellite basemaps.
          </p>
          <div style="background:var(--slate-100); border-radius:3mm; padding:3mm; border-left:3.5px solid var(--accent-cyan); margin-bottom:2.5mm;">
            <div style="font-size:7.6pt; font-weight:800; color:var(--navy-900); margin-bottom:1mm;">Key Editor Capabilities:</div>
            <ul style="font-size:7.1pt; color:var(--slate-700); padding-left:4mm; line-height:1.45;">
              <li>Automatic Bounding Box querying via spatial REST API</li>
              <li>Single-click conversion into editable OSM polygon features</li>
              <li>Preset OSM tagging with verified confidence scores</li>
              <li>Zero local environment or GIS software installation needed</li>
            </ul>
          </div>

          <!-- TABLE -->
          <table class="benchmark-table">
            <thead>
              <tr>
                <th>Workflow Metric</th>
                <th>Manual Digitizing</th>
                <th>LandAssist AI</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Time (1,193 km²)</strong></td>
                <td>~3 to 4 Weeks</td>
                <td class="badge-win">~12 Minutes (GPU T4)</td>
              </tr>
              <tr>
                <td><strong>Labor & Fatigue</strong></td>
                <td>High Mapper Strain</td>
                <td class="badge-win">Automated Cloud Batch</td>
              </tr>
              <tr>
                <td><strong>Boundary Geometry</strong></td>
                <td>Subjective Variances</td>
                <td class="badge-win">Chaikin Regularized</td>
              </tr>
              <tr>
                <td><strong>Vector Access</strong></td>
                <td>Manual Export/Upload</td>
                <td class="badge-win">Live PostGIS API</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- CARD 5: QR SCAN CENTER -->
        <div class="qr-showcase-card">
          <div class="qr-showcase-title">
            <span>Explore Live at the Expo</span>
          </div>
          <p class="qr-showcase-subtitle">Scan directly with your smartphone to test the live web platform and inspect the open-source engineering:</p>

          <div class="qr-pair-grid">

            <!-- QR 1: WEB EDITOR -->
            <div class="qr-box">
              <div class="qr-box-title">Live Web Editor</div>
              <div class="qr-img-wrapper">
                <img src="{qr_editor_b64}" alt="QR Code LandAssist Web Editor">
              </div>
              <div class="qr-box-desc">Test interactive AI polygon editing over Tangerang</div>
              <span class="qr-url-pill">landassist-api.onrender.com/editor/</span>
            </div>

            <!-- QR 2: GITHUB REPOSITORY -->
            <div class="qr-box github-box">
              <div class="qr-box-title">GitHub Repository</div>
              <div class="qr-img-wrapper">
                <img src="{qr_github_b64}" alt="QR Code GitHub Repository">
              </div>
              <div class="qr-box-desc">Inspect SegFormer weights, FastAPI API & ML code</div>
              <span class="qr-url-pill">github.com/Anovvy/LandAssist</span>
            </div>

          </div>
        </div>

      </div>

    </main>

    <!-- FOOTER SYSTEM -->
    <footer class="footer-system">
      <div class="footer-credits-row">
        <div>
          <span class="team-highlight">Project Lead & ML Architecture:</span> Bayu Nabil (Head of Project Management)
        </div>
        <div>
          <span class="team-highlight">Dataset Annotation & Labeling:</span> Yabes B. • Ahmad F. R. • FireLies • Naumam
        </div>
      </div>
      <div class="footer-partners-bar">
        <div>YouthMappers Universitas Negeri Jakarta Chapter • Open Geospatial Research Initiative</div>
        <div>Tech Stack: PyTorch • Google Earth Engine • Supabase PostGIS • FastAPI • Rapid/iD • Render Cloud</div>
        <div>Jakarta, Indonesia • November 2026</div>
      </div>
    </footer>

    <!-- BOTTOM DECORATIVE COLOR BAR -->
    <div class="bottom-accent-bar"></div>

  </div>
</body>
</html>
"""

    out_html = DOCS_ASSETS / "poster_a3.html"
    out_html.write_text(html_content, encoding="utf-8")
    print(f"Generated standalone A3 HTML poster at: {out_html}")
    return out_html


def render_pdf_and_png(html_path: Path):
    """Use Headless Chrome to compile print-ready A3 PDF and high-res PNG."""
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not Path(chrome_path).exists():
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    if not Path(chrome_path).exists():
        print("[WARN] Neither Chrome nor Edge executable found; skipping PDF/PNG rendering.")
        return

    out_pdf = DOCS_ASSETS / "LandAssist_Expo_Poster_A3.pdf"
    out_png = DOCS_ASSETS / "LandAssist_Expo_Poster_A3.png"

    file_url = html_path.as_uri()

    print(f"Rendering print-ready A3 PDF via {Path(chrome_path).name}...")
    pdf_cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_pdf}",
        file_url,
    ]
    res_pdf = subprocess.run(pdf_cmd, capture_output=True, text=True)
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        print(f"SUCCESS: Print-ready A3 PDF saved to: {out_pdf} ({out_pdf.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[ERROR] PDF generation failed: {res_pdf.stderr}")

    print(f"Rendering high-resolution A3 PNG preview via {Path(chrome_path).name}...")
    # Exact A3 ratio: 1123 x 1588 with scale factor 2 gives 2246 x 3176 px with 0 margin
    png_cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1123,1588",
        "--force-device-scale-factor=2",
        f"--screenshot={out_png}",
        file_url,
    ]
    res_png = subprocess.run(png_cmd, capture_output=True, text=True)
    if out_png.exists() and out_png.stat().st_size > 0:
        print(f"SUCCESS: High-resolution A3 PNG saved to: {out_png} ({out_png.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[ERROR] PNG generation failed: {res_png.stderr}")


def main():
    html_path = generate_poster_html()
    render_pdf_and_png(html_path)


if __name__ == "__main__":
    main()
