# Copyright (c) OpenMMLab. DINO with optional tooth-prior feature modulation.

from typing import List, Tuple, Union

import torch
from mmengine.logging import MMLogger
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from ..utils.tooth_prior_modules import (MaskGuidedFeatureModulation,
                                         ToothPriorEncoder)


@MODELS.register_module()
class DINOToothPrior(DINO):
    """DINO detector: optional tooth semantic prior modulates neck features (RT-DETRv3-style idea).

    Tooth semantics are expected in ``data_sample.gt_sem_seg`` (from COCO ``seg`` prefix +
    ``LoadAnnotations(with_seg=True)``), aligned spatially with the input image after padding.

    Args:
        use_tooth_prior_encoder (bool): Build and run :class:`ToothPriorEncoder`.
        use_mask_guided_modulation (bool): Apply mask-guided modulation on neck outputs.
        tooth_prior_channels (int): Prior embedding channels.
        gate_log_interval (int): Log gate statistics every N forwards.
    """

    def __init__(self,
                 *args,
                 use_tooth_prior_encoder: bool = True,
                 use_mask_guided_modulation: bool = True,
                 tooth_prior_channels: int = 64,
                 gate_log_interval: int = 200,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_tooth_prior_encoder = use_tooth_prior_encoder
        self.use_mask_guided_modulation = use_mask_guided_modulation
        self.tooth_prior_channels = tooth_prior_channels
        self.gate_log_interval = gate_log_interval

        if self.use_mask_guided_modulation and not self.use_tooth_prior_encoder:
            raise ValueError(
                'DINOToothPrior: use_mask_guided_modulation requires '
                'use_tooth_prior_encoder=True.')

        self.uses_tooth_prior = (
            self.use_tooth_prior_encoder and self.use_mask_guided_modulation)

        if self.uses_tooth_prior:
            # ChannelMapper / neck outputs `embed_dims` at each level.
            feat_ch = self.embed_dims
            self.tooth_prior_encoder = ToothPriorEncoder(
                in_channels=1,
                prior_channels=tooth_prior_channels,
                num_levels=self.num_feature_levels)
            self.mask_modulation = MaskGuidedFeatureModulation(
                feat_channels=feat_ch,
                prior_channels=tooth_prior_channels,
                num_levels=self.num_feature_levels,
                gate_log_interval=gate_log_interval)
        else:
            self.tooth_prior_encoder = None
            self.mask_modulation = None

        MMLogger.get_current_instance().info(
            '[DINOToothPrior] use_tooth_prior_encoder=%s '
            'use_mask_guided_modulation=%s num_feature_levels=%d',
            self.use_tooth_prior_encoder, self.use_mask_guided_modulation,
            self.num_feature_levels)

    def init_weights(self) -> None:
        super().init_weights()
        if not self.uses_tooth_prior:
            return

        def _init_prior_conv(module: torch.nn.Module) -> None:
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.tooth_prior_encoder.apply(_init_prior_conv)
        self.mask_modulation.apply(_init_prior_conv)

    def _tooth_tensor_from_samples(
            self, batch_data_samples: SampleList,
            ref_hw: Tuple[int, int]) -> torch.Tensor:
        """Stack tooth masks [B,1,H,W] float on current device; strict checks."""
        masks: List[Tensor] = []
        for s in batch_data_samples:
            if not hasattr(s, 'gt_sem_seg') or s.gt_sem_seg is None:
                raise KeyError(
                    'DINOToothPrior: missing gt_sem_seg (tooth semantic). '
                    'Enable dataset seg prefix and LoadAnnotations(with_seg=True).')
            m = s.gt_sem_seg.sem_seg
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.dim() != 3:
                raise ValueError(
                    f'gt_sem_seg.sem_seg must be [H,W] or [1,H,W], got {tuple(m.shape)}')
            if m.shape[-2:] != ref_hw:
                raise RuntimeError(
                    f'tooth mask spatial {tuple(m.shape[-2:])} != input {ref_hw}. '
                    'Enable DetDataPreprocessor.pad_seg and matching pipeline.')
            masks.append(m)
        batched = torch.stack(masks, dim=0).float()
        if batched.dim() != 4 or batched.size(1) != 1:
            raise RuntimeError(
                f'expected tooth tensor [B,1,H,W], got {tuple(batched.shape)}')
        return batched

    def _extract_feat_with_tooth(
            self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
    ) -> Tuple[Tensor, ...]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        if not self.uses_tooth_prior:
            return x
        ref_hw = (batch_inputs.shape[2], batch_inputs.shape[3])
        tooth = self._tooth_tensor_from_samples(batch_data_samples, ref_hw)
        priors = self.tooth_prior_encoder(tooth)
        if len(priors) != len(x):
            raise RuntimeError(
                f'prior levels {len(priors)} != neck levels {len(x)}')
        x = self.mask_modulation(x, priors)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self._extract_feat_with_tooth(batch_inputs,
                                                  batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self._extract_feat_with_tooth(batch_inputs,
                                                  batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self._extract_feat_with_tooth(batch_inputs,
                                                  batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        return self.bbox_head.forward(**head_inputs_dict)
