# Copyright (c) OpenMMLab. Dental dataset transforms (Plan.md experiment B).

from __future__ import annotations

import numpy as np
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
