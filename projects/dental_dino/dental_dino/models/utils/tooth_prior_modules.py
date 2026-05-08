# Copyright (c) OpenMMLab. Dental tooth-prior utilities.
"""Tooth semantic mask -> multi-scale prior + mask-guided feature modulation."""

from typing import Tuple

import torch
import torch.nn as nn
from mmengine.logging import MMLogger


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


class ToothPriorEncoder(nn.Module):
    """Encode [B,1,H,W] tooth labels to ``num_levels`` maps at strides /8 ... /(8*2^{L-1})."""

    def __init__(self,
                 in_channels: int = 1,
                 prior_channels: int = 64,
                 num_levels: int = 4) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f'num_levels must be >= 1, got {num_levels}')
        self.prior_channels = prior_channels
        self.num_levels = num_levels
        # Reach stride /8 w.r.t. input mask (three stride-2 blocks).
        self.down = nn.Sequential(
            _ConvGNAct(in_channels, 32, stride=2),
            _ConvGNAct(32, 32, stride=2),
            _ConvGNAct(32, 64, stride=2),
        )
        self.level_projs = nn.ModuleList()
        self.level_downs = nn.ModuleList()
        ch = 64
        self.level_projs.append(nn.Conv2d(ch, prior_channels, 1))
        for _ in range(num_levels - 1):
            self.level_downs.append(
                _ConvGNAct(prior_channels, prior_channels, stride=2))
            self.level_projs.append(nn.Conv2d(prior_channels, prior_channels, 1))

    def forward(self, tooth_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if tooth_mask.dim() != 4 or tooth_mask.size(1) != 1:
            raise ValueError(
                f'tooth_mask must be [B,1,H,W], got {tuple(tooth_mask.shape)}')
        x = self.down(tooth_mask)
        outs = []
        t = x
        for i in range(self.num_levels):
            t = self.level_projs[i](t)
            outs.append(t)
            if i < self.num_levels - 1:
                t = self.level_downs[i](t)
        return tuple(outs)


class MaskGuidedFeatureModulation(nn.Module):
    """feat_out = feat * (1 + sigmoid(Wg(p))) + Wb(p)."""

    def __init__(self,
                 feat_channels: int,
                 prior_channels: int,
                 num_levels: int,
                 gate_log_interval: int = 200) -> None:
        super().__init__()
        self.gate_log_interval = gate_log_interval
        self._step = 0
        self.gate_convs = nn.ModuleList([
            nn.Conv2d(prior_channels, feat_channels, 1) for _ in range(num_levels)
        ])
        self.bias_convs = nn.ModuleList([
            nn.Conv2d(prior_channels, feat_channels, 1) for _ in range(num_levels)
        ])

    def forward(self, mlvl_feats: Tuple[torch.Tensor, ...],
                priors: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        if len(mlvl_feats) != len(priors):
            raise RuntimeError(
                f'mlvl_feats ({len(mlvl_feats)}) vs priors ({len(priors)})')
        out = []
        logger = MMLogger.get_current_instance()
        for i, (feat, pr) in enumerate(zip(mlvl_feats, priors)):
            if feat.shape[0] != pr.shape[0]:
                raise RuntimeError('batch size mismatch feat vs prior')
            if feat.shape[2:] != pr.shape[2:]:
                raise RuntimeError(
                    f'spatial mismatch level {i}: feat {feat.shape} prior {pr.shape}'
                )
            gate = torch.sigmoid(self.gate_convs[i](pr))
            bias = self.bias_convs[i](pr)
            if self._step % self.gate_log_interval == 0:
                with torch.no_grad():
                    logger.info(
                        '[MaskGuidedFeatureModulation] lvl=%d feat=%s prior=%s '
                        'gate mean=%.6f min=%.6f max=%.6f', i,
                        tuple(feat.shape), tuple(pr.shape),
                        float(gate.mean()), float(gate.min()), float(gate.max()))
            out.append(feat * (1.0 + gate) + bias)
        self._step += 1
        return tuple(out)
