# Copyright (c) OpenMMLab. Tooth embedding encoder for SA-DINO.
"""Encode tooth instance-ID mask [B,1,H,W] -> multi-scale prior features
using learnable per-tooth-ID embeddings instead of raw scalar values."""

from typing import Tuple

import torch
import torch.nn as nn


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    ng = min(32, num_channels)
    while num_channels % ng != 0 and ng > 1:
        ng -= 1
    return nn.GroupNorm(num_groups=ng, num_channels=num_channels)


class _ConvGNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = _make_group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ToothEmbeddingEncoder(nn.Module):
    """Encode [B,1,H,W] tooth-ID mask into ``num_levels`` prior feature maps.

    Unlike ToothPriorEncoder which treats the mask as a continuous scalar,
    this module uses nn.Embedding to give each tooth ID a distinct learnable
    representation, preserving the categorical nature of tooth positions.

    Output strides match the neck feature pyramid: /8, /16, /32, /64 for
    num_levels=4 (same as ToothPriorEncoder).
    """

    def __init__(self,
                 num_classes: int = 34,
                 embed_dim: int = 32,
                 prior_channels: int = 64,
                 num_levels: int = 4) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f'num_levels must be >= 1, got {num_levels}')
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.prior_channels = prior_channels
        self.num_levels = num_levels

        self.tooth_embed = nn.Embedding(num_classes, embed_dim)

        self.down = nn.Sequential(
            _ConvGNAct(embed_dim, 48, stride=2),
            _ConvGNAct(48, 48, stride=2),
            _ConvGNAct(48, 64, stride=2),
        )

        self.level_projs = nn.ModuleList()
        self.level_downs = nn.ModuleList()
        self.level_projs.append(nn.Conv2d(64, prior_channels, 1))
        for _ in range(num_levels - 1):
            self.level_downs.append(
                _ConvGNAct(prior_channels, prior_channels, stride=2))
            self.level_projs.append(
                nn.Conv2d(prior_channels, prior_channels, 1))

    def forward(self, tooth_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            tooth_mask: [B, 1, H, W] int or float tensor with tooth IDs.
        Returns:
            Tuple of [B, prior_channels, H_l, W_l] at each pyramid level.
        """
        if tooth_mask.dim() != 4 or tooth_mask.size(1) != 1:
            raise ValueError(
                f'tooth_mask must be [B,1,H,W], got {tuple(tooth_mask.shape)}')

        ids = tooth_mask[:, 0].long().clamp(0, self.num_classes - 1)
        emb = self.tooth_embed(ids)  # [B, H, W, embed_dim]
        emb = emb.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]

        x = self.down(emb)

        outs = []
        t = x
        for i in range(self.num_levels):
            t = self.level_projs[i](t)
            outs.append(t)
            if i < self.num_levels - 1:
                t = self.level_downs[i](t)
        return tuple(outs)
