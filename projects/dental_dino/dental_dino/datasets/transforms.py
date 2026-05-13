# Copyright (c) OpenMMLab. Dental dataset transforms (Plan.md experiment B).

from __future__ import annotations

import numpy as np
import mmcv
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConcatSemSegToImage(BaseTransform):
    """Append binary tooth/instance mask as 4th channel (uint8 0/255).

    Expects ``results['img']`` as HxWx3 BGR (uint8) and ``gt_seg_map`` as
    HxW aligned with ``img`` after the same geometric transforms.
    """

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' not in results or results['gt_seg_map'] is None:
            raise KeyError(
                'ConcatSemSegToImage: missing gt_seg_map; use '
                'LoadAnnotations(with_seg=True) and seg data_prefix.')
        img = results['img']
        seg = results['gt_seg_map']
        if img.shape[:2] != seg.shape[:2]:
            raise RuntimeError(
                f'ConcatSemSegToImage: img HW {img.shape[:2]} != seg '
                f'{seg.shape[:2]}')
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                'ConcatSemSegToImage: expected HxWx3 image, got '
                f'{img.shape}')
        mask = (seg > 0).astype(np.uint8) * 255
        results['img'] = np.ascontiguousarray(
            np.concatenate([img, mask[:, :, np.newaxis]], axis=-1))
        return results

@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    """Apply CLAHE to image.

    Works for:
    - HxW grayscale
    - HxWx3 pseudo-color image copied from grayscale
    - generic uint8 multi-channel images
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def transform(self, results: dict) -> dict:
        img = results['img']

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = mmcv.clahe(img, self.clip_limit, self.tile_grid_size)

        elif img.ndim == 3 and img.shape[2] == 3:
            # 对“灰度复制成3通道”的情况，直接只处理一个通道再复制回去
            ch = mmcv.clahe(img[:, :, 0], self.clip_limit, self.tile_grid_size)
            img = np.stack([ch, ch, ch], axis=-1)

        else:
            # 兜底：逐通道处理
            out = []
            for i in range(img.shape[2]):
                out.append(
                    mmcv.clahe(img[:, :, i], self.clip_limit, self.tile_grid_size)
                )
            img = np.stack(out, axis=-1)

        results['img'] = np.ascontiguousarray(img)
        return results