# Copyright (c) OpenMMLab. Auxiliary segmentation head for SA-DINO.
"""Lightweight segmentation head on neck features — predicts tooth semantic
segmentation map.  Provides gradient signal to backbone/neck to learn
tooth-structure-aware features.  Discarded at inference time."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AuxSegHead(nn.Module):
    """Predict tooth segmentation from the largest-scale neck feature map.

    Architecture: 2x (3x3 conv + GN + ReLU) -> 4x bilinear upsample -> 1x1 conv
    Output: [B, num_classes, H/4, W/4] logits (1/4 of input resolution).
    """

    def __init__(self,
                 in_channels: int = 256,
                 mid_channels: int = 128,
                 num_classes: int = 34,
                 upsample_scale: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.upsample_scale = upsample_scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, feat: Tensor) -> Tensor:
        """
        Args:
            feat: [B, C, H, W] largest-scale neck feature (stride /8).
        Returns:
            [B, num_classes, H*s, W*s] logits.
        """
        x = self.convs(feat)
        if self.upsample_scale > 1:
            x = F.interpolate(
                x, scale_factor=self.upsample_scale,
                mode='bilinear', align_corners=False)
        return self.cls_conv(x)


def aux_seg_loss(logits: Tensor,
                 gt_sem_seg: Tensor,
                 num_classes: int,
                 ignore_index: int = 255) -> Tensor:
    """Cross-entropy loss between predicted seg logits and GT tooth mask.

    Args:
        logits: [B, num_classes, H', W'] from AuxSegHead.
        gt_sem_seg: [B, 1, H_full, W_full] integer tooth-ID mask.
        num_classes: number of classes (including background 0).
        ignore_index: pixels to ignore (e.g. padding).
    Returns:
        Scalar CE loss.
    """
    target = gt_sem_seg[:, 0].long()
    target = target.clamp(0, num_classes - 1)

    if (logits.shape[2] != target.shape[1] or
            logits.shape[3] != target.shape[2]):
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=logits.shape[2:],
            mode='nearest').squeeze(1).long()

    return F.cross_entropy(
        logits, target, ignore_index=ignore_index)
