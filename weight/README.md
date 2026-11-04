# LandAssist Model Weights Registry (`weight/`)

Folder ini merupakan lokasi penyimpanan lokal untuk checkpoint bobot model **SegFormer 64-Channel (`nvidia/mit-b2`)** hasil fine-tuning pada Google Satellite Embeddings (GSE).

> **Penting (Git Hygiene):** File binary bobot model (`*.pt`, ~100–110 MB) **di-ignore oleh Git** (`.gitignore`) agar tidak membebani riwayat commit atau ditolak oleh batas 100 MB GitHub push. Sebagai gantinya:
> 1. File `.pt` di-upload sebagai asset pada **GitHub Releases** (`https://github.com/Anovvy/LandAssist/releases`).
> 2. Metadata versi, metrik evaluasi (`IoU`, `FG-mIoU`), URL unduhan, dan checksum **SHA-256** dicatat secara terstruktur di dalam [`weight/manifest.json`](manifest.json) yang di-track oleh Git.

---

## Standar Penamaan File (Naming Convention)

Setiap file checkpoint di dalam `weight/` wajib mengikuti format Semantic Versioning:

```text
weight/segformer-mitb2-gse64-<version>.pt
```

Contoh:
- `weight/segformer-mitb2-gse64-v1.0.0.pt` (Baseline 125 chips Tangerang)
- `weight/segformer-mitb2-gse64-v1.1.0.pt` (Iterasi penambahan sampel kawasan industri/permukiman)

---

## Alur Kerja Versioning & GitHub Release (`scripts/manage_weights.py`)

### 1. Taruh File `.pt` dari Google Drive ke Folder `weight/`
Salin file `.pt` hasil fine-tuning Colab dari Google Drive Anda ke:
```text
C:\Users\bayun\OneDrive\Documents\GitHub\LandAssist\weight\
```

### 2. Registrasi Metadata, Checksum SHA-256 & Buat Release Notes Otomatis
Jalankan perintah berikut untuk membaca metadata dari dalam file `.pt`, menghitung hash SHA-256, memperbarui `weight/manifest.json`, dan men-generate catatan rilis (`weight/RELEASE_NOTES_v1.0.0.md`):

```powershell
python scripts/manage_weights.py register --set-default
```
*(Atau spesifikasikan path/versi secara eksplisit: `python scripts/manage_weights.py register --file weight/nama_file.pt --version v1.0.0 --set-default`)*

### 3. Upload ke GitHub Releases
Anda dapat meng-upload file `.pt` ke GitHub Releases secara otomatis melalui script (menggunakan `GITHUB_TOKEN` atau GitHub CLI `gh`):

```powershell
$env:GITHUB_TOKEN="ghp_xxx_token_anda"
python scripts/manage_weights.py publish --version v1.0.0
```

Atau secara manual melalui Web UI GitHub:
1. Buka **`https://github.com/Anovvy/LandAssist/releases/new`**
2. Isi **Tag**: `weights-v1.0.0` dan **Release Title**: `LandAssist SegFormer 64-Ch Weights (v1.0.0)`
3. Copy-paste isi file `weight/RELEASE_NOTES_v1.0.0.md` ke kolom deskripsi.
4. Drag & drop file `weight/segformer-mitb2-gse64-v1.0.0.pt` ke bagian **Attach binaries**, lalu klik **Publish release**.

### 4. Download Bobot pada Mesin Lain / Server Deployment
Siapa pun (atau pipeline CI/CD) dapat mengunduh bobot versi tertentu beserta verifikasi integritas SHA-256 hanya dengan satu perintah:

```powershell
python scripts/manage_weights.py download --version v1.0.0
```
