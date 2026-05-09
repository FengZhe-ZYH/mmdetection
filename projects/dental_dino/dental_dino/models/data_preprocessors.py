# Copyright (c) OpenMMLab. Four-channel image preprocessor (RGB + binary mask).

from __future__ import annotations

import math
from numbers import Number
from typing import Sequence

import torch
import torch.nn.functional as F
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class DetDataPreprocessor4Ch(DetDataPreprocessor):
    """DetDataPreprocessor variant for 4-channel inputs (BGR+mask → RGB+mask).

    MMEngine's :class:`ImgDataPreprocessor` only allows ``mean``/``std`` of
    length 1 or 3; this class accepts length-4 statistics and applies
    BGR→RGB **only to the first three channels**, leaving the mask channel
    untouched (after normalization with the 4th mean/std).
    """

    def __init__(
        self,
        mean: Sequence[Number],
        std: Sequence[Number],
        bgr_to_rgb: bool = True,
        **kwargs,
    ) -> None:
        if len(mean) != 4 or len(std) != 4:
            raise ValueError(
                'DetDataPreprocessor4Ch: `mean` and `std` must have 4 elements '
                f'(RGB + mask), got len(mean)={len(mean)} len(std)={len(std)}')
        kwargs = dict(kwargs)
        kwargs['mean'] = None
        kwargs['std'] = None
        kwargs['bgr_to_rgb'] = False
        super().__init__(**kwargs)
        self._swap_bgr_to_rgb = bgr_to_rgb
        self.register_buffer(
            'mean4',
            torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
            persistent=False)
        self.register_buffer(
            'std4',
            torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
            persistent=False)

    def _per_tensor_preprocess(self, _batch_input: torch.Tensor) -> torch.Tensor:
        if self._swap_bgr_to_rgb:
            c = _batch_input.shape[0]
            if c == 4:
                bgr = _batch_input[:3]
                m = _batch_input[3:4]
                _batch_input = torch.cat([bgr[[2, 1, 0]], m], dim=0)
            elif c == 3:
                _batch_input = _batch_input[[2, 1, 0], ...]
            else:
                raise ValueError(
                    f'DetDataPreprocessor4Ch: expected 3 or 4 channels, got {c}')
        _batch_input = _batch_input.float()
        return (_batch_input - self.mean4) / self.std4

    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)
        data = self.cast_data(data)
        _batch_inputs = data['inputs']

        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = [
                self._per_tensor_preprocess(t) for t in _batch_inputs
            ]
            batch_inputs = stack_batch(
                batch_inputs, self.pad_size_divisor, self.pad_value)
        elif isinstance(_batch_inputs, torch.Tensor):
            if _batch_inputs.dim() != 4:
                raise TypeError(
                    f'Expected NCHW tensor, got shape {_batch_inputs.shape}')
            if self._swap_bgr_to_rgb:
                c = _batch_inputs.shape[1]
                if c == 4:
                    bgr = _batch_inputs[:, :3]
                    m = _batch_inputs[:, 3:4]
                    rgb = bgr[:, [2, 1, 0], ...]
                    _batch_inputs = torch.cat([rgb, m], dim=1)
                elif c == 3:
                    _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
                else:
                    raise ValueError(
                        f'DetDataPreprocessor4Ch: expected 3 or 4 channels, '
                        f'got {c}')
            _batch_inputs = _batch_inputs.float()
            _batch_inputs = (_batch_inputs - self.mean4) / self.std4
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(
                _batch_inputs, (0, pad_w, 0, pad_h), 'constant',
                self.pad_value)
        else:
            raise TypeError(
                'DetDataPreprocessor4Ch: inputs must be list of tensors or '
                f'batched tensor, got {type(_batch_inputs)}')

        data['inputs'] = batch_inputs
        data_samples = data.get('data_samples')

        if data_samples is not None:
            if isinstance(batch_inputs, torch.Tensor):
                batch_input_shape = tuple(batch_inputs.size()[-2:])
            else:
                batch_input_shape = tuple(batch_inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                from mmdet.models.utils.misc import samplelist_boxtype2tensor
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                batch_inputs, data_samples = batch_aug(batch_inputs,
                                                       data_samples)

        return {'inputs': batch_inputs, 'data_samples': data_samples}
