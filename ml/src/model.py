"""SegFormer 64-Channel Architecture, Loss Functions, and Versioned Checkpointing.

Adapts HuggingFace SegFormer (MiT-B2 backbone) to accept 64-channel Google
Satellite Embedding (GSE) inputs and addresses severe class imbalance (7.8%
industrial coverage in Tangerang) via combined Weighted Cross-Entropy +
Multiclass Foreground Dice Loss.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

CLASS_MAPPING: Dict[int, str] = {
    0: "background",
    1: "industrial",
    2: "residential",
}

ID2LABEL: Dict[int, str] = dict(CLASS_MAPPING)
LABEL2ID: Dict[str, int] = {v: k for k, v in CLASS_MAPPING.items()}


def _build_mit_b2_config(num_classes: int = 3) -> SegformerConfig:
    """Create a standalone SegformerConfig matching nvidia/mit-b2 architecture."""
    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=768,
        num_labels=num_classes,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def _get_first_patch_embedding(model: SegformerForSemanticSegmentation) -> nn.Module:
    """Resolve the Stage-0 patch embedding module across transformers v4 and v5+."""
    segformer = model.segformer
    if hasattr(segformer, "stages") and len(segformer.stages) > 0:
        return segformer.stages[0].patch_embeddings
    if hasattr(segformer, "encoder"):
        encoder = segformer.encoder
        if hasattr(encoder, "patch_embeddings") and len(encoder.patch_embeddings) > 0:
            return encoder.patch_embeddings[0]
        if hasattr(encoder, "patch_embed") and len(encoder.patch_embed) > 0:
            return encoder.patch_embed[0]
    raise AttributeError(
        f"Cannot locate first patch embedding on {type(segformer).__name__}"
    )


def build_segformer_64ch(
    model_name: str = "nvidia/mit-b2",
    num_classes: int = 3,
    in_channels: int = 64,
    pretrained: bool = True,
) -> SegformerForSemanticSegmentation:
    """Build a SegFormer semantic segmentation model adapted for 64-band GSE inputs.

    Adapts the first patch embedding layer (`stages[0].patch_embeddings.proj` in
    transformers v5+ or `encoder.patch_embeddings[0].proj` in v4) from 3 RGB
    channels to `in_channels` (default 64) by repeating the mean RGB convolutional
    kernel weights across all input channels and scaling by `3.0 / in_channels`
    to preserve activation variance in early layers.

    Args:
        model_name: HuggingFace model identifier (e.g., `"nvidia/mit-b2"`).
        num_classes: Number of target semantic classes (`0: background`,
            `1: industrial`, `2: residential`).
        in_channels: Number of input spectral/embedding bands (default 64).
        pretrained: Whether to initialize with ImageNet pretrained weights. Falls
            back to `SegformerConfig` initialization if offline or `pretrained=False`.

    Returns:
        Configured `SegformerForSemanticSegmentation` accepting `(B, in_channels, H, W)`.
    """
    model: Optional[SegformerForSemanticSegmentation] = None
    if pretrained:
        try:
            cfg = SegformerConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=cfg,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            model = None

    if model is None:
        config = _build_mit_b2_config(num_classes=num_classes)
        model = SegformerForSemanticSegmentation(config)

    patch_embed_0 = _get_first_patch_embedding(model)
    old_proj: nn.Conv2d = patch_embed_0.proj
    new_proj = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        dilation=old_proj.dilation,
        groups=old_proj.groups,
        bias=(old_proj.bias is not None),
    )

    with torch.no_grad():
        # Average RGB weights -> (out_channels, 1, kH, kW) and scale by 3 / in_channels
        mean_rgb_weight = old_proj.weight.mean(dim=1, keepdim=True)
        scale_factor = float(old_proj.in_channels) / float(in_channels)
        repeated_weight = mean_rgb_weight.repeat(1, in_channels, 1, 1) * scale_factor
        new_proj.weight.copy_(repeated_weight)
        if old_proj.bias is not None and new_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    patch_embed_0.proj = new_proj
    model.config.num_channels = in_channels
    return model


class WeightedCEDiceLoss(nn.Module):
    """Combined Weighted Cross-Entropy and Multiclass Foreground Dice Loss.

    Designed to counter extreme class imbalance where `industrial` occupies only
    7.8% of labeled pixels. Upsamples SegFormer's `(B, C, H/4, W/4)` logits to
    full `(B, C, H, W)` resolution prior to computing losses.
    """

    def __init__(
        self,
        class_weights: Sequence[float] = (0.5, 3.5, 1.0),
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_weights",
            torch.tensor(list(class_weights), dtype=torch.float32),
        )
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute `0.5 * WeightedCE + 0.5 * ForegroundDice`.

        Args:
            logits: Raw model logits of shape `(B, C, h, w)` or `(B, C, H, W)`.
            targets: Ground-truth integer mask of shape `(B, H, W)`.

        Returns:
            Scalar differentiable loss tensor.
        """
        target_h, target_w = targets.shape[-2], targets.shape[-1]
        if logits.shape[-2:] != (target_h, target_w):
            logits = F.interpolate(
                logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.cross_entropy(
            logits,
            targets.long(),
            weight=weights,
            ignore_index=self.ignore_index,
        )

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(
            valid_mask, targets, torch.zeros_like(targets)
        ).long()
        targets_one_hot = (
            F.one_hot(safe_targets, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .to(dtype=logits.dtype)
        )

        valid_mask_4d = valid_mask.unsqueeze(1).to(dtype=logits.dtype)
        probs = probs * valid_mask_4d
        targets_one_hot = targets_one_hot * valid_mask_4d

        # Foreground classes (1: industrial, 2: residential)
        fg_probs = probs[:, 1:, :, :] if num_classes > 1 else probs
        fg_targets = targets_one_hot[:, 1:, :, :] if num_classes > 1 else targets_one_hot

        dims = (0, 2, 3)
        intersection = torch.sum(fg_probs * fg_targets, dim=dims)
        cardinality = torch.sum(fg_probs + fg_targets, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def save_versioned_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    model_version: str = "v1.0.0",
    backbone: str = "nvidia/mit-b2",
    dataset_version: str = "ds-v1.0-rf2-125chips",
    val_metrics: Optional[Dict[str, Any]] = None,
    in_channels: int = 64,
    num_classes: int = 3,
) -> Path:
    """Save model weights bundled with reproducibility and lineage metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_version": model_version,
        "backbone": backbone,
        "dataset_version": dataset_version,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "class_mapping": CLASS_MAPPING,
        "val_metrics": val_metrics or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, out_path)
    return out_path


def load_versioned_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    pretrained_fallback: bool = False,
) -> Tuple[SegformerForSemanticSegmentation, Dict[str, Any]]:
    """Load a versioned 64-channel SegFormer checkpoint and return `(model, metadata)`."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "nvidia/mit-b2")
    num_classes = int(ckpt.get("num_classes", 3))
    in_channels = int(ckpt.get("in_channels", 64))

    model = build_segformer_64ch(
        model_name=backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained_fallback,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, metadata
