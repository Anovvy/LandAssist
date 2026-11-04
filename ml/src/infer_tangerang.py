"""Anti-OOM Sliding-Window Inference on 64-Band GSE GeoTIFF Rasters.

Streams 64-band Google Satellite Embedding (GSE) tiles directly from disk via
`rasterio.windows.Window`, blends overlapping predictions with a 2D Hann
(raised-cosine) window to eliminate tile boundary seams, and accumulates
probabilities in disk-backed `numpy.memmap` arrays (`float16`) to prevent
RAM exhaustion on regional Tangerang rasters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model import load_versioned_checkpoint, resolve_checkpoint_path


def build_2d_hann_window(
    tile_size: int = 160,
    min_weight: float = 0.05,
) -> np.ndarray:
    """Create a 2D Hann (raised-cosine) blending window for seamless tile stitching.

    The center of the tile receives weight `1.0` while edges taper smoothly
    toward `min_weight` (`> 0` so border pixels on raster edges remain valid).

    Args:
        tile_size: Spatial height/width of each square inference window.
        min_weight: Floor weight at tile boundaries (default `0.05`).

    Returns:
        `(tile_size, tile_size)` float32 weight matrix in `[min_weight, 1.0]`.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not (0.0 < min_weight <= 1.0):
        raise ValueError(f"min_weight must be in (0, 1], got {min_weight}")

    if tile_size == 1:
        return np.ones((1, 1), dtype=np.float32)

    hann_1d = np.hanning(tile_size).astype(np.float32)
    hann_2d = np.outer(hann_1d, hann_1d)
    max_val = float(hann_2d.max())
    if max_val > 0.0:
        hann_2d = hann_2d / max_val

    window = min_weight + (1.0 - min_weight) * hann_2d
    return window.astype(np.float32)


def _generate_tile_Grid(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    """Compute `(row_off, col_off)` sliding window top-left coordinates covering `(height, width)`."""
    stride = max(1, tile_size - overlap)

    if height <= tile_size:
        row_offsets = [0]
    else:
        row_offsets = list(range(0, height - tile_size + 1, stride))
        if row_offsets[-1] != height - tile_size:
            row_offsets.append(height - tile_size)

    if width <= tile_size:
        col_offsets = [0]
    else:
        col_offsets = list(range(0, width - tile_size + 1, stride))
        if col_offsets[-1] != width - tile_size:
            col_offsets.append(width - tile_size)

    return [(r, c) for r in row_offsets for c in col_offsets]


def run_sliding_window_inference(
    model: nn.Module,
    gse_tif_path: Union[str, Path],
    temp_dir: Union[str, Path],
    tile_size: int = 160,
    overlap: int = 80,
    batch_size: int = 8,
    device: Union[str, torch.device] = "cuda",
    num_classes: int = 3,
) -> Tuple[np.memmap, Dict[str, Any]]:
    """Run memory-bounded sliding-window inference over a 64-band GSE GeoTIFF.

    Reads windows strictly via `rasterio.windows.Window` and writes weighted
    probabilities into disk-backed `float16` memmaps (`pred_sum.mmap` and
    `weight_sum.mmap`) before normalizing row-chunks in-place.

    Args:
        model: Trained 64-channel SegFormer model.
        gse_tif_path: Path to the 64-band GSE GeoTIFF raster.
        temp_dir: Directory to store `pred_sum.mmap` and `weight_sum.mmap`.
        tile_size: Sliding window spatial dimension (default `160`).
        overlap: Overlap between adjacent windows in pixels (default `80`, 50%).
        batch_size: Number of windows evaluated per GPU/CPU forward pass.
        device: Torch device (`"cuda"` or `"cpu"`).
        num_classes: Number of output segmentation classes (default `3`).

    Returns:
        Tuple of `(prob_mmap, raster_meta)` where `prob_mmap` has shape `(3, H, W)`
        and `dtype=float16`.
    """
    tif_path = Path(gse_tif_path)
    out_temp_dir = Path(temp_dir)
    out_temp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    hann_window = build_2d_hann_window(tile_size=tile_size, min_weight=0.05)

    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        in_channels = src.count

        pred_sum_path = out_temp_dir / "pred_sum.mmap"
        weight_sum_path = out_temp_dir / "weight_sum.mmap"

        pred_sum = np.memmap(
            pred_sum_path,
            dtype="float16",
            mode="w+",
            shape=(num_classes, height, width),
        )
        weight_sum = np.memmap(
            weight_sum_path,
            dtype="float16",
            mode="w+",
            shape=(height, width),
        )

        coords = _generate_tile_Grid(
            height=height,
            width=width,
            tile_size=tile_size,
            overlap=overlap,
        )

        use_amp = torch_device.type == "cuda"
        with torch.inference_mode():
            for batch_start in tqdm(
                range(0, len(coords), batch_size),
                desc="Sliding-window GSE inference",
                leave=False,
            ):
                batch_coords = coords[batch_start : batch_start + batch_size]
                batch_chips: List[np.ndarray] = []

                for r_off, c_off in batch_coords:
                    win = Window(
                        col_off=c_off,
                        row_off=r_off,
                        width=tile_size,
                        height=tile_size,
                    )
                    chip = src.read(
                        window=win,
                        boundless=True,
                        fill_value=0.0,
                    ).astype(np.float32)
                    chip = np.nan_to_num(chip, nan=0.0, posinf=0.0, neginf=0.0)
                    batch_chips.append(chip)

                batch_np = np.stack(batch_chips, axis=0)  # (B, C, tile_size, tile_size)
                batch_tensor = torch.from_numpy(batch_np).to(torch_device)

                with torch.amp.autocast(device_type=torch_device.type, enabled=use_amp):
                    outputs = model(pixel_values=batch_tensor)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    if logits.shape[-2:] != (tile_size, tile_size):
                        logits = F.interpolate(
                            logits,
                            size=(tile_size, tile_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

                for idx, (r_off, c_off) in enumerate(batch_coords):
                    valid_h = min(tile_size, height - r_off)
                    valid_w = min(tile_size, width - c_off)
                    w_patch = hann_window[:valid_h, :valid_w]
                    weighted_prob = probs[idx, :, :valid_h, :valid_w] * w_patch[None, :, :]

                    cur_pred = pred_sum[:, r_off : r_off + valid_h, c_off : c_off + valid_w].astype(
                        np.float32
                    )
                    pred_sum[:, r_off : r_off + valid_h, c_off : c_off + valid_w] = (
                        cur_pred + weighted_prob
                    ).astype(np.float16)

                    cur_weight = weight_sum[r_off : r_off + valid_h, c_off : c_off + valid_w].astype(
                        np.float32
                    )
                    weight_sum[r_off : r_off + valid_h, c_off : c_off + valid_w] = (
                        cur_weight + w_patch
                    ).astype(np.float16)

        pred_sum.flush()
        weight_sum.flush()

        # Normalize row-by-row in chunks to avoid full-raster float32 allocation in RAM
        chunk_h = max(1, height // 50)
        for r0 in range(0, height, chunk_h):
            r1 = min(height, r0 + chunk_h)
            w_chunk = np.maximum(weight_sum[r0:r1, :].astype(np.float32), 1e-6)
            p_chunk = pred_sum[:, r0:r1, :].astype(np.float32) / w_chunk[None, :, :]
            pred_sum[:, r0:r1, :] = np.clip(p_chunk, 0.0, 1.0).astype(np.float16)

        pred_sum.flush()

    raster_meta: Dict[str, Any] = {
        "height": height,
        "width": width,
        "count": in_channels,
        "num_classes": num_classes,
        "transform": transform,
        "crs": crs,
        "pred_sum_path": str(pred_sum_path),
        "weight_sum_path": str(weight_sum_path),
    }
    return pred_sum, raster_meta


def main() -> None:
    """CLI entrypoint for running anti-OOM sliding-window inference on a GSE GeoTIFF."""
    parser = argparse.ArgumentParser(
        description="Run Anti-OOM 64-band SegFormer sliding-window inference on Tangerang GSE GeoTIFF."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to versioned SegFormer checkpoint (.pt). If omitted, resolves from weight/manifest.json.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Semantic model version in weight/manifest.json (e.g. v1.0.0) when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--input-tif",
        type=str,
        required=True,
        help="Path to 64-band GSE GeoTIFF.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="./tmp_mmap",
        help="Directory for disk-backed float16 memmap probability buffers.",
    )
    parser.add_argument("--tile-size", type=int, default=160, help="Sliding window size.")
    parser.add_argument("--overlap", type=int, default=80, help="Sliding window overlap.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward pass.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda/cpu).")

    args = parser.parse_args()
    ckpt_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        model_version=args.model_version,
    )
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model, meta = load_versioned_checkpoint(ckpt_path, device=args.device)
    _, raster_meta = run_sliding_window_inference(
        model=model,
        gse_tif_path=args.input_tif,
        temp_dir=args.temp_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        f"[OK] Inference complete (model={meta.get('model_version', 'v1.0.0')}). "
        f"Probability memmap saved at: {raster_meta['pred_sum_path']}"
    )


if __name__ == "__main__":
    main()
