#!/usr/bin/env python3
"""Model Weight Versioning & GitHub Release Manager for LandAssist.

Manages 64-channel SegFormer (`.pt`) checkpoints stored locally in `weight/`
and published as binary assets on GitHub Releases (`Anovvy/LandAssist`):
- `register`: Inspect a `.pt` file in `weight/`, normalize its filename and
  metadata structure, compute its SHA-256 checksum, update `weight/manifest.json`,
  and generate Markdown release notes (`weight/RELEASE_NOTES_<version>.md`).
- `publish`: Create or update a GitHub Release (`weights-<version>`) and upload
  the `.pt` asset via GitHub REST API (`$GITHUB_TOKEN`) or GitHub CLI (`gh`).
- `download`: Download a versioned `.pt` checkpoint from GitHub Releases into
  `weight/` and verify its SHA-256 integrity against `weight/manifest.json`.
- `list`: Display all registered model versions, metrics, and local status.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR = REPO_ROOT / "weight"
MANIFEST_PATH = WEIGHT_DIR / "manifest.json"
DEFAULT_REPO = "Anovvy/LandAssist"


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load `weight/manifest.json` or initialize a default structure."""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "repository": os.environ.get("GITHUB_RELEASE_REPO", DEFAULT_REPO),
        "default_version": os.environ.get("DEFAULT_MODEL_VERSION", "v1.0.0"),
        "models": {},
    }


def save_manifest(manifest: dict[str, Any], manifest_path: Path = MANIFEST_PATH) -> None:
    """Write `weight/manifest.json` with deterministic formatting."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class _TorchZipMetadataUnpickler(pickle.Unpickler):
    """Lightweight unpickler to read metadata from a PyTorch ZIP archive without importing torch."""

    def persistent_load(self, _pid: Any) -> None:
        return None

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("torch") or module.startswith("numpy"):
            return lambda *a, **kw: None
        return super().find_class(module, name)


def _read_pt_metadata_without_torch(pt_path: Path) -> dict[str, Any] | None:
    """Extract scalar/dict metadata directly from a `.pt` ZIP archive's `data.pkl`."""
    try:
        with zipfile.ZipFile(pt_path, "r") as zf:
            pkl_candidates = [n for n in zf.namelist() if n.endswith("data.pkl")]
            if not pkl_candidates:
                return None
            raw = zf.read(pkl_candidates[0])
            data = _TorchZipMetadataUnpickler(io.BytesIO(raw)).load()
            if isinstance(data, dict) and "model_state_dict" in data:
                return {k: v for k, v in data.items() if k != "model_state_dict"}
    except Exception:
        return None
    return None


def inspect_and_normalize_checkpoint(
    pt_path: Path,
    override_version: str | None = None,
) -> dict[str, Any]:
    """Read a PyTorch `.pt` checkpoint, ensure versioned metadata wrapper, and return metadata."""
    try:
        import torch
    except ImportError:
        zip_meta = _read_pt_metadata_without_torch(pt_path) or {}
        return {
            "model_version": override_version or str(zip_meta.get("model_version", "v1.0.0")),
            "backbone": str(zip_meta.get("backbone", "nvidia/mit-b2")),
            "in_channels": int(zip_meta.get("in_channels", 64)),
            "num_classes": int(zip_meta.get("num_classes", 3)),
            "dataset_version": str(zip_meta.get("dataset_version", "ds-v1.0-rf2-125chips")),
            "class_mapping": {
                str(k): str(v)
                for k, v in zip_meta.get(
                    "class_mapping", {0: "background", 1: "industrial", 2: "residential"}
                ).items()
            },
            "val_metrics": zip_meta.get("val_metrics", {}) or {},
            "test_metrics": zip_meta.get("test_metrics", {}) or {},
            "created_at": zip_meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    modified = False

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        version = override_version or str(ckpt.get("model_version", "v1.0.0"))
        if override_version and ckpt.get("model_version") != override_version:
            ckpt["model_version"] = override_version
            modified = True
    elif isinstance(ckpt, dict):
        # Raw state_dict (e.g. best_weights_tmp.pt) -> wrap into full versioned checkpoint
        version = override_version or "v1.0.0"
        ckpt = {
            "model_state_dict": ckpt,
            "model_version": version,
            "backbone": "nvidia/mit-b2",
            "dataset_version": "ds-v1.0-rf2-125chips",
            "in_channels": 64,
            "num_classes": 3,
            "class_mapping": {0: "background", 1: "industrial", 2: "residential"},
            "val_metrics": {},
            "test_metrics": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        modified = True
    else:
        raise ValueError(f"Unsupported checkpoint format in {pt_path}")

    if modified:
        torch.save(ckpt, pt_path)

    return {
        "model_version": str(ckpt.get("model_version", "v1.0.0")),
        "backbone": str(ckpt.get("backbone", "nvidia/mit-b2")),
        "in_channels": int(ckpt.get("in_channels", 64)),
        "num_classes": int(ckpt.get("num_classes", 3)),
        "dataset_version": str(ckpt.get("dataset_version", "ds-v1.0-rf2-125chips")),
        "class_mapping": {
            str(k): str(v)
            for k, v in ckpt.get(
                "class_mapping", {0: "background", 1: "industrial", 2: "residential"}
            ).items()
        },
        "val_metrics": ckpt.get("val_metrics", {}) or {},
        "test_metrics": ckpt.get("test_metrics", {}) or {},
        "created_at": ckpt.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }


def generate_release_notes(entry: dict[str, Any], repo: str) -> str:
    """Create a rich Markdown release note summary for GitHub Releases."""
    ver = entry["model_version"]
    tag = entry["release_tag"]
    fname = entry["filename"]
    val_m = entry.get("val_metrics") or {}
    test_m = entry.get("test_metrics") or {}

    lines = [
        f"## 🛰️ LandAssist 64-Channel SegFormer Weights (`{ver}`)",
        "",
        f"Fine-tuned **`{entry['backbone']}`** semantic segmentation checkpoint accepting **64-band Google Satellite Embeddings (GSE)** (`in_channels={entry['in_channels']}`) for Kabupaten Tangerang (`Industrial` & `Residential` mapping).",
        "",
        "### 📦 Artifact & Lineage Metadata",
        f"- **Model Version:** `{ver}`",
        f"- **Release Tag:** `{tag}`",
        f"- **Asset Filename:** `{fname}`",
        f"- **File Size:** `{entry['size_mb']} MB`",
        f"- **SHA-256 Checksum:** `{entry['sha256']}`",
        f"- **Backbone:** `{entry['backbone']}` (`{entry['in_channels']}` channels -> `{entry['num_classes']}` classes)",
        f"- **Dataset Lineage:** `{entry['dataset_version']}`",
        "",
    ]

    if val_m or test_m:
        lines.extend(
            [
                "### 📊 Evaluation Metrics (IoU)",
                "| Split | Industrial IoU | Residential IoU | Background IoU | Foreground mIoU (`FG-mIoU`) | Overall mIoU |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        if val_m:
            lines.append(
                f"| **Validation** | `{val_m.get('iou_industrial', '-')}` | "
                f"`{val_m.get('iou_residential', '-')}` | "
                f"`{val_m.get('iou_background', '-')}` | "
                f"**`{val_m.get('fg_mIoU', '-')}`** | "
                f"`{val_m.get('mIoU', '-')}` |"
            )
        if test_m:
            lines.append(
                f"| **Test Set** | `{test_m.get('iou_industrial', '-')}` | "
                f"`{test_m.get('iou_residential', '-')}` | "
                f"`{test_m.get('iou_background', '-')}` | "
                f"**`{test_m.get('fg_mIoU', '-')}`** | "
                f"`{test_m.get('mIoU', '-')}` |"
            )
        lines.append("")

    lines.extend(
        [
            "### 🚀 Quick Download & Inference Usage",
            "```powershell",
            f"# 1. Download and verify SHA-256 checksum into weight/{fname}",
            f"python scripts/manage_weights.py download --version {ver}",
            "",
            "# 2. Run Anti-OOM Hann Sliding-Window Inference on a 64-band GSE GeoTIFF",
            f"python -m ml.src.infer_tangerang --model-version {ver} --input-tif data/tangerang_gse_64band.tif --temp-dir ./tmp_mmap",
            "```",
            "",
            f"**Direct Download URL:** `{entry['download_url']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def cmd_register(args: argparse.Namespace) -> int:
    """Register a `.pt` file inside `weight/` into `weight/manifest.json`."""
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if args.file:
        source_path = Path(args.file).resolve()
    else:
        pt_files = sorted(
            WEIGHT_DIR.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not pt_files:
            print(
                f"[ERROR] Tidak ditemukan file .pt di dalam {WEIGHT_DIR}.\n"
                f"Silakan taruh file weight hasil fine-tuning Drive Anda ke folder `weight/` terlebih dahulu.",
                file=sys.stderr,
            )
            return 1
        source_path = pt_files[0]

    if not source_path.exists():
        print(f"[ERROR] File tidak ditemukan: {source_path}", file=sys.stderr)
        return 1

    print(f"[1/4] Memeriksa metadata checkpoint: {source_path.name} ...")
    meta = inspect_and_normalize_checkpoint(source_path, override_version=args.version)
    version = meta["model_version"]
    if not version.startswith("v"):
        version = f"v{version}"
        meta["model_version"] = version

    canonical_filename = f"segformer-mitb2-gse64-{version}.pt"
    target_path = WEIGHT_DIR / canonical_filename

    if source_path.resolve() != target_path.resolve():
        print(f"[2/4] Menyalin/menstandarkan nama file ke: weight/{canonical_filename} ...")
        shutil.copy2(source_path, target_path)
    else:
        print(f"[2/4] Nama file sudah sesuai standar: weight/{canonical_filename}")

    print("[3/4] Menghitung checksum SHA-256 & ukuran file ...")
    sha256_hex = compute_sha256(target_path)
    size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)

    manifest = load_manifest()
    repo = args.repo or manifest.get("repository", DEFAULT_REPO)
    manifest["repository"] = repo
    release_tag = f"weights-{version}"
    download_url = (
        f"https://github.com/{repo}/releases/download/{release_tag}/{canonical_filename}"
    )

    entry = {
        "model_version": version,
        "filename": canonical_filename,
        "release_tag": release_tag,
        "download_url": download_url,
        "sha256": sha256_hex,
        "size_mb": size_mb,
        "backbone": meta["backbone"],
        "in_channels": meta["in_channels"],
        "num_classes": meta["num_classes"],
        "dataset_version": meta["dataset_version"],
        "class_mapping": meta["class_mapping"],
        "val_metrics": meta["val_metrics"],
        "test_metrics": meta["test_metrics"],
        "created_at": meta["created_at"],
    }

    manifest.setdefault("models", {})[version] = entry
    if args.set_default or not manifest.get("default_version"):
        manifest["default_version"] = version

    save_manifest(manifest)

    notes_path = WEIGHT_DIR / f"RELEASE_NOTES_{version}.md"
    notes_path.write_text(generate_release_notes(entry, repo=repo), encoding="utf-8")

    print(f"[4/4] Berhasil memperbarui weight/manifest.json untuk versi {version}!")
    print(f"      - File Lokal     : {target_path}")
    print(f"      - Ukuran         : {size_mb} MB")
    print(f"      - SHA-256        : {sha256_hex}")
    print(f"      - Release Tag    : {release_tag}")
    print(f"      - Release Notes  : {notes_path}")
    return 0


def _publish_via_github_api(
    repo: str,
    tag: str,
    title: str,
    notes: str,
    asset_path: Path,
    token: str,
) -> str:
    """Create or fetch a GitHub Release and upload the binary `.pt` asset via REST API."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "LandAssist-Weight-Manager",
    }

    # Check if release tag already exists
    release_obj: dict[str, Any] | None = None
    get_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(get_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            release_obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    if release_obj is None:
        create_url = f"https://api.github.com/repos/{repo}/releases"
        payload = json.dumps(
            {
                "tag_name": tag,
                "target_commitish": "main",
                "name": title,
                "body": notes,
                "draft": False,
                "prerelease": False,
            }
        ).encode("utf-8")
        create_req = urllib.request.Request(
            create_url,
            data=payload,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(create_req, timeout=30) as resp:
            release_obj = json.loads(resp.read().decode("utf-8"))

    # Delete existing asset with same name if present
    for asset in release_obj.get("assets", []):
        if asset.get("name") == asset_path.name:
            del_url = f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
            del_req = urllib.request.Request(del_url, headers=headers, method="DELETE")
            with urllib.request.urlopen(del_req, timeout=30):
                pass

    upload_url_template = release_obj["upload_url"].split("{")[0]
    upload_url = f"{upload_url_template}?name={asset_path.name}"
    file_bytes = asset_path.read_bytes()
    upload_req = urllib.request.Request(
        upload_url,
        data=file_bytes,
        headers={
            **headers,
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(file_bytes)),
        },
        method="POST",
    )
    with urllib.request.urlopen(upload_req, timeout=600) as resp:
        asset_resp = json.loads(resp.read().decode("utf-8"))
    return str(asset_resp.get("browser_download_url", release_obj.get("html_url")))


def cmd_publish(args: argparse.Namespace) -> int:
    """Publish a registered `.pt` weight file to GitHub Releases."""
    manifest = load_manifest()
    version = args.version or manifest.get("default_version", "v1.0.0")
    if not version.startswith("v"):
        version = f"v{version}"

    models = manifest.get("models", {})
    if version not in models or not models[version].get("sha256"):
        print(f"[INFO] Versi {version} belum teregistrasi penuh, menjalankan `register` otomatis ...")
        reg_args = argparse.Namespace(
            file=args.file,
            version=version,
            repo=args.repo,
            set_default=True,
        )
        rc = cmd_register(reg_args)
        if rc != 0:
            return rc
        manifest = load_manifest()

    entry = manifest["models"][version]
    repo = args.repo or manifest.get("repository", DEFAULT_REPO)
    asset_path = WEIGHT_DIR / entry["filename"]
    if not asset_path.exists():
        print(f"[ERROR] File bobot lokal tidak ditemukan: {asset_path}", file=sys.stderr)
        return 1

    tag = entry["release_tag"]
    title = f"LandAssist SegFormer 64-Ch Weights ({version})"
    notes = generate_release_notes(entry, repo=repo)
    notes_path = WEIGHT_DIR / f"RELEASE_NOTES_{version}.md"
    notes_path.write_text(notes, encoding="utf-8")

    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        print(f"[PUBLISH] Mengunggah {asset_path.name} ({entry['size_mb']} MB) ke GitHub Release `{tag}` ({repo}) via REST API ...")
        try:
            url = _publish_via_github_api(
                repo=repo,
                tag=tag,
                title=title,
                notes=notes,
                asset_path=asset_path,
                token=token,
            )
            print(f"[OK] Berhasil di-publish ke GitHub Releases!\n     Download URL: {url}")
            return 0
        except Exception as exc:
            print(f"[ERROR] Gagal mengunggah via GitHub REST API: {exc}", file=sys.stderr)
            return 1

    gh_bin = shutil.which("gh")
    if gh_bin:
        print(f"[PUBLISH] Mengunggah {asset_path.name} ke GitHub Release `{tag}` menggunakan GitHub CLI (`gh`) ...")
        cmd = [
            gh_bin,
            "release",
            "create",
            tag,
            str(asset_path),
            "--repo",
            repo,
            "--title",
            title,
            "--notes-file",
            str(notes_path),
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if res.returncode == 0:
            print(f"[OK] Berhasil di-publish ke GitHub Releases: {entry['download_url']}")
            return 0

    print(
        "\n[INFO] Metadata & Release Notes sudah siap, namun `$env:GITHUB_TOKEN` belum di-set dan `gh` CLI tidak terdeteksi.\n"
        "Pilih salah satu cara berikut untuk menyelesaikan upload ke GitHub Release:\n\n"
        "CARA A — Otomatis via Terminal (menggunakan Personal Access Token GitHub):\n"
        "  $env:GITHUB_TOKEN=\"ghp_token_github_anda\"\n"
        f"  python scripts/manage_weights.py publish --version {version}\n\n"
        "CARA B — Upload via Browser (10 Detik di GitHub Web UI):\n"
        f"  1. Buka: https://github.com/{repo}/releases/new\n"
        f"  2. Isi Tag: `{tag}` dan Release Title: `{title}`\n"
        f"  3. Copy-paste isi file `{notes_path}` ke kolom deskripsi.\n"
        f"  4. Drag & drop file `weight/{entry['filename']}` lalu klik 'Publish release'.\n"
    )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download a versioned `.pt` checkpoint from GitHub Releases and verify its SHA-256."""
    manifest = load_manifest()
    version = args.version or manifest.get("default_version", "v1.0.0")
    if not version.startswith("v"):
        version = f"v{version}"

    models = manifest.get("models", {})
    if version not in models:
        print(
            f"[ERROR] Versi {version} tidak ditemukan di dalam weight/manifest.json. "
            f"Versi tersedia: {list(models.keys())}",
            file=sys.stderr,
        )
        return 1

    entry = models[version]
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = WEIGHT_DIR / entry["filename"]
    expected_sha256 = entry.get("sha256")

    if dest_path.exists() and not args.force:
        if expected_sha256:
            actual_sha = compute_sha256(dest_path)
            if actual_sha == expected_sha256:
                print(f"[OK] File weight/{entry['filename']} sudah ada dan lolos verifikasi SHA-256.")
                return 0
            print("[WARN] Checksum file lokal berbeda dari manifest, mengunduh ulang ...")
        else:
            print(f"[OK] File weight/{entry['filename']} sudah tersedia secara lokal.")
            return 0

    url = entry["download_url"]
    print(f"[DOWNLOAD] Mengunduh {entry['filename']} dari {url} ...")
    headers = {"User-Agent": "LandAssist-Weight-Manager"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    tmp_path = dest_path.with_suffix(".pt.part")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, tmp_path.open("wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        tmp_path.replace(dest_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[ERROR] Gagal mengunduh dari GitHub Releases ({url}): {exc}", file=sys.stderr)
        return 1

    if expected_sha256:
        actual_sha = compute_sha256(dest_path)
        if actual_sha != expected_sha256:
            print(
                f"[ERROR] Checksum SHA-256 tidak cocok!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual  : {actual_sha}",
                file=sys.stderr,
            )
            return 1
        print(f"[OK] Unduhan selesai & terverifikasi SHA-256 ({actual_sha[:12]}...): {dest_path}")
    else:
        print(f"[OK] Unduhan selesai: {dest_path}")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    """List all registered model versions in `weight/manifest.json`."""
    manifest = load_manifest()
    default_ver = manifest.get("default_version", "v1.0.0")
    models = manifest.get("models", {})

    print(f"Repository      : {manifest.get('repository', DEFAULT_REPO)}")
    print(f"Default Version : {default_ver}\n")
    print(f"{'VERSION':<12} {'LOCAL FILE':<12} {'SIZE (MB)':<11} {'SHA-256 (PREFIX)':<18} {'RELEASE TAG':<16}")
    print("-" * 74)
    for ver, item in models.items():
        marker = "*" if ver == default_ver else " "
        local_path = WEIGHT_DIR / item["filename"]
        local_status = "READY" if local_path.exists() else "MISSING"
        size_str = f"{item['size_mb']}" if item.get("size_mb") is not None else "-"
        sha_str = item["sha256"][:16] if item.get("sha256") else "(unregistered)"
        print(
            f"{marker}{ver:<11} {local_status:<12} {size_str:<11} {sha_str:<18} {item.get('release_tag', '-'):<16}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage LandAssist 64-channel SegFormer weights and GitHub Releases."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser(
        "register",
        help="Inspect a .pt file in weight/, compute SHA-256, and update weight/manifest.json.",
    )
    p_reg.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the .pt file (defaults to newest .pt in weight/).",
    )
    p_reg.add_argument(
        "--version",
        type=str,
        default=None,
        help="Explicit semantic version (e.g. v1.0.0). Overrides checkpoint metadata if set.",
    )
    p_reg.add_argument("--repo", type=str, default=None, help="GitHub repository (owner/repo).")
    p_reg.add_argument(
        "--set-default",
        action="store_true",
        help="Set this version as default_version in weight/manifest.json.",
    )

    p_pub = sub.add_parser(
        "publish",
        help="Upload the registered .pt weight in weight/ to GitHub Releases.",
    )
    p_pub.add_argument("--version", type=str, default=None, help="Model version to publish (e.g. v1.0.0).")
    p_pub.add_argument("--file", type=str, default=None, help="Optional path to .pt file.")
    p_pub.add_argument("--repo", type=str, default=None, help="GitHub repository (owner/repo).")
    p_pub.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token.")

    p_dl = sub.add_parser(
        "download",
        help="Download a versioned .pt weight from GitHub Releases into weight/ and verify SHA-256.",
    )
    p_dl.add_argument("--version", type=str, default=None, help="Model version to download (default: default_version).")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if local file exists.")

    sub.add_parser("list", help="List all model versions registered in weight/manifest.json.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "register":
        return cmd_register(args)
    if args.command == "publish":
        return cmd_publish(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "list":
        return cmd_list(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
