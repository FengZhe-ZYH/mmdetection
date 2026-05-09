# Copyright (c) OpenMMLab. Tooth-guided query reference generation (CPU).

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def invert_sigmoid_normalized(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse sigmoid for normalized [0, 1] coordinates (e.g. DINO box logits)."""
    return _inverse_sigmoid(x, eps=eps)


def build_tooth_reference_points(
        tooth_mask_hw: np.ndarray,
        queries_per_component: int,
        max_total_queries: int,
        *,
        min_area_px: int = 32,
) -> np.ndarray:
    """From tooth instance-id map (H,W), build normalized cxcywh references.

    Args:
        tooth_mask_hw: 2D int mask, 0 = background.
        queries_per_component: Points sampled per connected component (grid).
        max_total_queries: Cap total points; truncate in component visit order.
        min_area_px: Ignore tiny components.

    Returns:
        ndarray float32 [N, 4] in normalized cxcywh.

    Raises:
        RuntimeError: No foreground component found.
    """
    if tooth_mask_hw.ndim != 2:
        raise ValueError(f'expected 2D mask, got {tooth_mask_hw.shape}')
    h, w = tooth_mask_hw.shape[:2]
    fg = (tooth_mask_hw > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8)
    refs: List[np.ndarray] = []
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        xs = stats[lab, cv2.CC_STAT_LEFT]
        ys = stats[lab, cv2.CC_STAT_TOP]
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        k = max(1, int(queries_per_component))
        gx = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
        gy = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
        for ix in gx:
            for iy in gy:
                cx_px = xs + ix * bw
                cy_px = ys + iy * bh
                cx = cx_px / float(w)
                cy = cy_px / float(h)
                ww = bw / float(w)
                hh = bh / float(h)
                refs.append(np.array([cx, cy, ww, hh], dtype=np.float32))
                if len(refs) >= max_total_queries:
                    out = np.stack(refs, axis=0)
                    return out
    if not refs:
        raise RuntimeError(
            'tooth-guided queries: no tooth region found in mask (connected '
            'components empty or all below min_area_px).')
    return np.stack(refs, axis=0)


def batch_tooth_reference_points(
        tooth_masks: torch.Tensor,
        queries_per_component: int,
        max_slots: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    """Batch of normalized cxcywh boxes on device, padded to max_slots (repeat last).

    Args:
        tooth_masks: [B, 1, H, W] float or int on any device.
    """
    if tooth_masks.dim() != 4 or tooth_masks.size(1) != 1:
        raise ValueError(f'expected [B,1,H,W], got {tuple(tooth_masks.shape)}')
    tooth_masks = tooth_masks.detach().cpu()
    bsz = tooth_masks.shape[0]
    out = []
    for b in range(bsz):
        m = tooth_masks[b, 0].numpy().astype(np.int32)
        r = build_tooth_reference_points(
            m,
            queries_per_component,
            max_total_queries=max_slots,
        )
        n = r.shape[0]
        if n < max_slots:
            pad = np.repeat(r[-1:], max_slots - n, axis=0)
            r = np.concatenate([r, pad], axis=0)
        elif n > max_slots:
            r = r[:max_slots]
        out.append(torch.from_numpy(r))
    stacked = torch.stack(out, dim=0).to(device=device, dtype=dtype)
    assert stacked.shape == (bsz, max_slots, 4)
    return stacked
