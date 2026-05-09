# Copyright (c) OpenMMLab. SA-DINO: Semantic-Aware DINO for dental detection.
"""DINO + deep tooth semantic prior integration:
- ToothEmbeddingEncoder: per-tooth-ID learnable embeddings
- Improved MaskGuidedFeatureModulation with negative-bias gate init
- SemanticCrossAttention on encoder output memory
- Auxiliary tooth segmentation head
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from ..utils.aux_seg_head import AuxSegHead, aux_seg_loss
from ..utils.semantic_cross_attention import (SemanticCrossAttention,
                                              flatten_multi_scale_priors)
from ..utils.tooth_embedding_encoder import ToothEmbeddingEncoder
from ..utils.tooth_prior_modules import MaskGuidedFeatureModulation


@MODELS.register_module()
class SADINO(DINO):
    """Semantic-Aware DINO: deep integration of tooth semantic mask into DINO.

    Compared to DINOToothPrior, this model:
    1. Uses per-tooth-ID embeddings instead of raw scalar mask values
    2. Applies semantic cross-attention on encoder output memory
    3. Uses improved gate modulation with negative-bias initialization
    4. Adds auxiliary segmentation loss for backbone/neck regularization
    """

    def __init__(self,
                 *args,
                 # tooth embedding encoder
                 num_tooth_classes: int = 34,
                 tooth_embed_dim: int = 32,
                 tooth_prior_channels: int = 64,
                 # mask-guided modulation
                 use_mask_modulation: bool = True,
                 modulation_strength: float = 1.0,
                 gate_init_bias: float = -2.0,
                 # semantic cross-attention on encoder output
                 use_semantic_cross_attn: bool = True,
                 num_semantic_cross_attn_layers: int = 3,
                 semantic_cross_attn_heads: int = 8,
                 semantic_cross_attn_dropout: float = 0.0,
                 # auxiliary segmentation
                 use_aux_seg: bool = True,
                 aux_seg_weight: float = 0.4,
                 aux_seg_mid_channels: int = 128,
                 aux_seg_upsample: int = 2,
                 # logging
                 gate_log_interval: int = 500,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_tooth_classes = num_tooth_classes
        self.use_mask_modulation = use_mask_modulation
        self.use_semantic_cross_attn = use_semantic_cross_attn
        self.use_aux_seg = use_aux_seg
        self.aux_seg_weight = float(aux_seg_weight)

        self.tooth_encoder = ToothEmbeddingEncoder(
            num_classes=num_tooth_classes,
            embed_dim=tooth_embed_dim,
            prior_channels=tooth_prior_channels,
            num_levels=self.num_feature_levels)

        self.mask_modulation = None
        if use_mask_modulation:
            self.mask_modulation = MaskGuidedFeatureModulation(
                feat_channels=self.embed_dims,
                prior_channels=tooth_prior_channels,
                num_levels=self.num_feature_levels,
                gate_log_interval=gate_log_interval,
                modulation_strength=modulation_strength,
                gate_init_bias=gate_init_bias)

        self.semantic_cross_attn_layers = None
        if use_semantic_cross_attn:
            self.semantic_cross_attn_layers = nn.ModuleList([
                SemanticCrossAttention(
                    embed_dims=self.embed_dims,
                    prior_channels=tooth_prior_channels,
                    num_heads=semantic_cross_attn_heads,
                    dropout=semantic_cross_attn_dropout)
                for _ in range(num_semantic_cross_attn_layers)
            ])

        self.aux_seg_head = None
        if use_aux_seg:
            self.aux_seg_head = AuxSegHead(
                in_channels=self.embed_dims,
                mid_channels=aux_seg_mid_channels,
                num_classes=num_tooth_classes,
                upsample_scale=aux_seg_upsample)

        self._cached_tooth_priors: Tuple[Tensor, ...] | None = None
        self._cached_neck_feats: Tuple[Tensor, ...] | None = None

        log = MMLogger.get_current_instance()
        log.info(
            '[SADINO] tooth_encoder=%s modulation=%s(strength=%s,gate_bias=%s) '
            'semantic_cross_attn=%s(layers=%d) aux_seg=%s(w=%s)',
            True, use_mask_modulation, modulation_strength, gate_init_bias,
            use_semantic_cross_attn, num_semantic_cross_attn_layers,
            use_aux_seg, aux_seg_weight)

    def init_weights(self) -> None:
        super().init_weights()

        def _init_conv(module: nn.Module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.tooth_encoder.apply(_init_conv)
        if self.mask_modulation is not None:
            for pp in self.mask_modulation.prior_projs:
                _init_conv(pp)
            for bc in self.mask_modulation.bias_convs:
                _init_conv(bc)
        if self.aux_seg_head is not None:
            self.aux_seg_head.apply(_init_conv)

    def _tooth_tensor_from_samples(
            self, batch_data_samples: SampleList,
            ref_hw: Tuple[int, int]) -> Tensor:
        masks: List[Tensor] = []
        for s in batch_data_samples:
            if not hasattr(s, 'gt_sem_seg') or s.gt_sem_seg is None:
                raise KeyError(
                    'SADINO: missing gt_sem_seg. Use seg data_prefix and '
                    'LoadAnnotations(with_seg=True).')
            m = s.gt_sem_seg.sem_seg
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.shape[-2:] != ref_hw:
                raise RuntimeError(
                    f'tooth mask {tuple(m.shape[-2:])} != input {ref_hw}. '
                    'Enable pad_seg in DetDataPreprocessor.')
            masks.append(m)
        return torch.stack(masks, dim=0).float()

    def _extract_feat_with_tooth(
            self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
    ) -> Tuple[Tensor, ...]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)

        ref_hw = (batch_inputs.shape[2], batch_inputs.shape[3])
        tooth = self._tooth_tensor_from_samples(batch_data_samples, ref_hw)
        priors = self.tooth_encoder(tooth)
        self._cached_tooth_priors = priors
        self._cached_neck_feats = x

        if self.use_mask_modulation and self.mask_modulation is not None:
            x = self.mask_modulation(x, priors)

        return x

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Override to inject semantic cross-attention after encoder."""
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        if (self.use_semantic_cross_attn and
                self.semantic_cross_attn_layers is not None and
                self._cached_tooth_priors is not None):
            prior_tokens = flatten_multi_scale_priors(
                self._cached_tooth_priors)
            for cross_attn in self.semantic_cross_attn_layers:
                memory = cross_attn(memory, prior_tokens)

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self._extract_feat_with_tooth(
            batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.use_aux_seg and self.aux_seg_head is not None:
            if self._cached_neck_feats is None:
                raise RuntimeError('neck feats not cached for aux seg')
            seg_logits = self.aux_seg_head(self._cached_neck_feats[0])
            ref_hw = (batch_inputs.shape[2], batch_inputs.shape[3])
            tooth_mask = self._tooth_tensor_from_samples(
                batch_data_samples, ref_hw)
            losses['loss_aux_seg'] = self.aux_seg_weight * aux_seg_loss(
                seg_logits, tooth_mask, self.num_tooth_classes)

        self._cached_tooth_priors = None
        self._cached_neck_feats = None
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self._extract_feat_with_tooth(
            batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        self._cached_tooth_priors = None
        self._cached_neck_feats = None
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self._extract_feat_with_tooth(
            batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        out = self.bbox_head.forward(**head_inputs_dict)
        self._cached_tooth_priors = None
        self._cached_neck_feats = None
        return out
