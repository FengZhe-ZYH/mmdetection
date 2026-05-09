# Copyright (c) OpenMMLab. SA-DINO: Semantic-Aware DINO for dental detection.
"""DINO + deep tooth semantic prior integration:
- ToothEmbeddingEncoder: per-tooth-ID learnable embeddings
- Improved MaskGuidedFeatureModulation with negative-bias gate init
- SemanticCrossAttention on encoder output memory
- Auxiliary tooth segmentation head
- Optional tooth-guided decoder query initialization
- Optional RoI-based refinement using high-res C2 features (v3a)
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from ..utils.aux_seg_head import AuxSegHead, aux_seg_loss
from ..utils.roi_refine_head import (ToothRoIRefineHead, apply_delta_to_boxes,
                                     cxcywh_to_xyxy, refine_loss,
                                     xyxy_to_cxcywh)
from ..utils.semantic_cross_attention import (SemanticCrossAttention,
                                              flatten_multi_scale_priors)
from ..utils.tooth_embedding_encoder import ToothEmbeddingEncoder
from ..utils.tooth_guided_query_init import ToothGuidedQueryInit
from ..utils.tooth_local_attention import ToothLocalAttention
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
                 # tooth-guided query initialization
                 use_tooth_guided_queries: bool = False,
                 max_tooth_queries: int = 32,
                 # auxiliary segmentation
                 use_aux_seg: bool = True,
                 aux_seg_weight: float = 0.4,
                 aux_seg_mid_channels: int = 128,
                 aux_seg_upsample: int = 2,
                 # RoI refinement (v3a)
                 use_roi_refine: bool = False,
                 roi_refine_c2_channels: int = 256,
                 roi_refine_feat_channels: int = 256,
                 roi_refine_h: int = 7,
                 roi_refine_w: int = 10,
                 roi_refine_num_dcn: int = 2,
                 roi_refine_tooth_expand: float = 0.3,
                 roi_refine_loss_weight: float = 1.0,
                 roi_refine_topk: int = 100,
                 roi_refine_iou_thr: float = 0.5,
                 # tooth-local dense attention (v3b)
                 use_tooth_local_attn: bool = False,
                 tooth_local_attn_heads: int = 8,
                 tooth_local_attn_layers: int = 1,
                 tooth_local_attn_ffn_dim: int = 1024,
                 tooth_local_attn_dropout: float = 0.0,
                 # logging
                 gate_log_interval: int = 500,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_tooth_classes = num_tooth_classes
        self.use_mask_modulation = use_mask_modulation
        self.use_semantic_cross_attn = use_semantic_cross_attn
        self.use_tooth_guided_queries = use_tooth_guided_queries
        self.max_tooth_queries = max_tooth_queries
        self.use_aux_seg = use_aux_seg
        self.aux_seg_weight = float(aux_seg_weight)
        self.use_roi_refine = use_roi_refine
        self.roi_refine_loss_weight = float(roi_refine_loss_weight)
        self.roi_refine_topk = roi_refine_topk
        self.roi_refine_iou_thr = roi_refine_iou_thr
        self.use_tooth_local_attn = use_tooth_local_attn

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

        self.tooth_query_init = None
        if use_tooth_guided_queries:
            if max_tooth_queries <= 0 or max_tooth_queries >= self.num_queries:
                raise ValueError(
                    f'max_tooth_queries must be in (0, {self.num_queries}), '
                    f'got {max_tooth_queries}')
            self.tooth_query_init = ToothGuidedQueryInit(
                max_tooth_queries=max_tooth_queries,
                num_tooth_classes=num_tooth_classes,
                embed_dims=self.embed_dims)

        self.aux_seg_head = None
        if use_aux_seg:
            self.aux_seg_head = AuxSegHead(
                in_channels=self.embed_dims,
                mid_channels=aux_seg_mid_channels,
                num_classes=num_tooth_classes,
                upsample_scale=aux_seg_upsample)

        self.roi_refine_head = None
        self._c2_proj = None
        if use_roi_refine:
            self.roi_refine_head = ToothRoIRefineHead(
                in_channels=roi_refine_c2_channels,
                feat_channels=roi_refine_feat_channels,
                roi_h=roi_refine_h,
                roi_w=roi_refine_w,
                c2_stride=4,
                num_dcn_layers=roi_refine_num_dcn,
                num_tooth_classes=num_tooth_classes,
                tooth_expand_ratio=roi_refine_tooth_expand)

        self.tooth_local_attn = None
        if use_tooth_local_attn:
            self.tooth_local_attn = ToothLocalAttention(
                embed_dims=self.embed_dims,
                num_heads=tooth_local_attn_heads,
                dropout=tooth_local_attn_dropout,
                ffn_dim=tooth_local_attn_ffn_dim,
                num_layers=tooth_local_attn_layers)

        self._cached_tooth_priors: Tuple[Tensor, ...] | None = None
        self._cached_neck_feats: Tuple[Tensor, ...] | None = None
        self._cached_tooth_mask: Tensor | None = None
        self._cached_c2_feat: Tensor | None = None

        log = MMLogger.get_current_instance()
        log.info(
            '[SADINO] modulation=%s(s=%s,bias=%s) cross_attn=%s(L=%d) '
            'tooth_queries=%s(k=%d) aux_seg=%s(w=%s) roi_refine=%s '
            'local_attn=%s(L=%d)',
            use_mask_modulation, modulation_strength, gate_init_bias,
            use_semantic_cross_attn, num_semantic_cross_attn_layers,
            use_tooth_guided_queries, max_tooth_queries,
            use_aux_seg, aux_seg_weight, use_roi_refine,
            use_tooth_local_attn, tooth_local_attn_layers)

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
        backbone_outs = self.backbone(batch_inputs)

        if self.use_roi_refine and len(backbone_outs) == 4:
            c2_feat = backbone_outs[0]
            self._cached_c2_feat = self.roi_refine_head.c2_proj(c2_feat)
            neck_inputs = backbone_outs[1:]
        else:
            neck_inputs = backbone_outs
            self._cached_c2_feat = None

        x = self.neck(neck_inputs) if self.with_neck else neck_inputs

        ref_hw = (batch_inputs.shape[2], batch_inputs.shape[3])
        tooth = self._tooth_tensor_from_samples(batch_data_samples, ref_hw)
        priors = self.tooth_encoder(tooth)
        self._cached_tooth_priors = priors
        self._cached_neck_feats = x
        self._cached_tooth_mask = tooth

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

    def pre_decoder(self,
                    memory: Tensor,
                    memory_mask: Tensor,
                    spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Override DINO pre_decoder to inject tooth-guided queries.

        When ``use_tooth_guided_queries`` is True, the last
        ``max_tooth_queries`` slots (in the matching part) are replaced by
        tooth-derived reference points and content embeddings.  This preserves
        the total number of queries and does not interfere with DN queries.
        """
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            batch_data_samples=batch_data_samples)

        if (self.use_tooth_guided_queries and
                self.tooth_query_init is not None and
                self._cached_tooth_mask is not None):
            tooth_refs, tooth_content = self.tooth_query_init(
                self._cached_tooth_mask)
            k = self.max_tooth_queries
            query = decoder_inputs_dict['query']
            ref_pts = decoder_inputs_dict['reference_points']

            if self.training:
                n_dn = query.shape[1] - self.num_queries
                match_query = query[:, n_dn:]
                match_refs = ref_pts[:, n_dn:]

                match_query = torch.cat(
                    [match_query[:, :-k], tooth_content], dim=1)
                match_refs = torch.cat(
                    [match_refs[:, :-k], tooth_refs], dim=1)

                decoder_inputs_dict['query'] = torch.cat(
                    [query[:, :n_dn], match_query], dim=1)
                decoder_inputs_dict['reference_points'] = torch.cat(
                    [ref_pts[:, :n_dn], match_refs], dim=1)
            else:
                decoder_inputs_dict['query'] = torch.cat(
                    [query[:, :-k], tooth_content], dim=1)
                decoder_inputs_dict['reference_points'] = torch.cat(
                    [ref_pts[:, :-k], tooth_refs], dim=1)

        return decoder_inputs_dict, head_inputs_dict

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

        if (self.use_tooth_local_attn and
                self.tooth_local_attn is not None and
                self._cached_tooth_mask is not None):
            decoder_outputs_dict = self._apply_tooth_local_attn(
                decoder_outputs_dict,
                encoder_outputs_dict,
                decoder_inputs_dict)

        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def _apply_tooth_local_attn(
        self,
        decoder_outputs_dict: Dict,
        encoder_outputs_dict: Dict,
        decoder_inputs_dict: Dict,
    ) -> Dict:
        """Apply tooth-local dense attention to the last decoder layer output."""
        hidden_states = decoder_outputs_dict['hidden_states']
        references = decoder_outputs_dict['references']

        last_hidden = hidden_states[-1]
        last_ref = references[-1]

        memory = decoder_inputs_dict['memory']
        spatial_shapes = encoder_outputs_dict['spatial_shapes']

        refined = self.tooth_local_attn(
            query=last_hidden,
            memory=memory,
            ref_points=last_ref,
            tooth_mask=self._cached_tooth_mask,
            spatial_shapes=spatial_shapes,
            num_tooth_classes=self.num_tooth_classes)

        hidden_states = list(hidden_states)
        hidden_states[-1] = refined
        hidden_states = torch.stack(hidden_states, dim=0)

        decoder_outputs_dict['hidden_states'] = hidden_states
        return decoder_outputs_dict

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

        if (self.use_roi_refine and self.roi_refine_head is not None
                and self._cached_c2_feat is not None):
            refine_losses = self._compute_refine_loss(
                head_inputs_dict, batch_inputs, batch_data_samples)
            losses.update(refine_losses)

        self._cached_tooth_priors = None
        self._cached_neck_feats = None
        self._cached_tooth_mask = None
        self._cached_c2_feat = None
        return losses

    def _compute_refine_loss(
        self,
        head_inputs_dict: Dict,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Compute refinement loss using the last decoder layer's outputs."""
        references = head_inputs_dict['references']
        last_ref = references[-1].detach()

        img_h, img_w = batch_inputs.shape[2], batch_inputs.shape[3]
        B = batch_inputs.shape[0]

        all_deltas = []
        all_iou_preds = []
        all_pred_cxcywh = []
        all_gt_cxcywh = []

        for b in range(B):
            gt_instances = batch_data_samples[b].gt_instances
            gt_bboxes = gt_instances.bboxes
            if gt_bboxes.numel() == 0:
                continue

            if self.training:
                dn_meta = head_inputs_dict.get('dn_meta')
                if dn_meta is not None:
                    n_dn = dn_meta.get('num_denoising_queries', 0)
                else:
                    n_dn = 0
                ref_b = last_ref[b, n_dn:]
            else:
                ref_b = last_ref[b]

            pred_cxcywh_norm = ref_b
            pred_cxcywh_abs = pred_cxcywh_norm.clone()
            pred_cxcywh_abs[:, 0::2] *= img_w
            pred_cxcywh_abs[:, 1::2] *= img_h
            pred_xyxy = cxcywh_to_xyxy(pred_cxcywh_abs)

            gt_xyxy = gt_bboxes
            gt_cxcywh = xyxy_to_cxcywh(gt_xyxy)

            iou_matrix = self._pairwise_iou(pred_xyxy, gt_xyxy)
            max_iou, matched_gt = iou_matrix.max(dim=1)
            pos_mask = max_iou >= self.roi_refine_iou_thr

            if self.roi_refine_topk > 0:
                topk_k = min(self.roi_refine_topk, pos_mask.sum().item())
                if topk_k == 0:
                    topk_k = min(
                        self.roi_refine_topk,
                        min(10, max_iou.shape[0]))
                    _, topk_idx = max_iou.topk(topk_k)
                    pos_mask = torch.zeros_like(pos_mask)
                    pos_mask[topk_idx] = True

            if not pos_mask.any():
                continue

            pos_pred_xyxy = pred_xyxy[pos_mask]
            pos_gt_idx = matched_gt[pos_mask]
            pos_gt_cxcywh = gt_cxcywh[pos_gt_idx]
            pos_pred_cxcywh = pred_cxcywh_abs[pos_mask]

            if self._cached_tooth_mask is not None:
                expanded, tooth_ids = \
                    self.roi_refine_head._expand_boxes_with_tooth(
                        pos_pred_xyxy, self._cached_tooth_mask[b:b + 1],
                        img_h, img_w)
            else:
                expanded = pos_pred_xyxy
                tooth_ids = None

            batch_idx = torch.full(
                (expanded.shape[0],), b,
                dtype=torch.long, device=expanded.device)
            delta, iou_pred = self.roi_refine_head(
                self._cached_c2_feat, expanded, batch_idx, tooth_ids)

            all_deltas.append(delta)
            all_iou_preds.append(iou_pred)
            all_pred_cxcywh.append(pos_pred_cxcywh)
            all_gt_cxcywh.append(pos_gt_cxcywh)

        if not all_deltas:
            zero = batch_inputs.sum() * 0.0
            return dict(
                loss_refine_delta=zero,
                loss_refine_giou=zero,
                loss_refine_iou=zero)

        cat_delta = torch.cat(all_deltas, dim=0)
        cat_iou = torch.cat(all_iou_preds, dim=0)
        cat_pred = torch.cat(all_pred_cxcywh, dim=0)
        cat_gt = torch.cat(all_gt_cxcywh, dim=0)

        norm_factor = cat_pred.new_tensor([img_w, img_h, img_w, img_h])
        cat_pred_norm = cat_pred / norm_factor
        cat_gt_norm = cat_gt / norm_factor

        r_losses = refine_loss(
            cat_delta, cat_iou, cat_pred_norm, cat_gt_norm)

        w = self.roi_refine_loss_weight
        for k in r_losses:
            r_losses[k] = r_losses[k] * w

        return r_losses

    @staticmethod
    def _pairwise_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """Compute NxM pairwise IoU between boxes1 [N,4] and boxes2 [M,4]."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        return inter / (area1[:, None] + area2[None, :] - inter).clamp(1e-6)

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

        if (self.use_roi_refine and self.roi_refine_head is not None
                and self._cached_c2_feat is not None):
            results_list = self._refine_predictions(
                results_list, batch_inputs, batch_data_samples, rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        self._cached_tooth_priors = None
        self._cached_neck_feats = None
        self._cached_tooth_mask = None
        self._cached_c2_feat = None
        return batch_data_samples

    def _refine_predictions(
        self,
        results_list: list,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool,
    ) -> list:
        """Apply RoI refinement to prediction results during inference."""
        img_h, img_w = batch_inputs.shape[2], batch_inputs.shape[3]

        for b, results in enumerate(results_list):
            bboxes = results.bboxes
            if bboxes.numel() == 0:
                continue

            if rescale:
                scale_factor = batch_data_samples[b].scale_factor
                sf = bboxes.new_tensor(scale_factor).repeat(2)
                bboxes_abs = bboxes * sf
            else:
                bboxes_abs = bboxes.clone()

            pred_cxcywh = xyxy_to_cxcywh(bboxes_abs)

            if self._cached_tooth_mask is not None:
                expanded, tooth_ids = \
                    self.roi_refine_head._expand_boxes_with_tooth(
                        bboxes_abs, self._cached_tooth_mask[b:b + 1],
                        img_h, img_w)
            else:
                expanded = bboxes_abs
                tooth_ids = None

            batch_idx = torch.full(
                (expanded.shape[0],), b,
                dtype=torch.long, device=expanded.device)
            delta, iou_pred = self.roi_refine_head(
                self._cached_c2_feat, expanded, batch_idx, tooth_ids)

            refined_cxcywh = apply_delta_to_boxes(pred_cxcywh, delta)
            refined_xyxy = cxcywh_to_xyxy(refined_cxcywh)

            refined_xyxy[:, 0].clamp_(min=0)
            refined_xyxy[:, 1].clamp_(min=0)
            refined_xyxy[:, 2].clamp_(max=img_w)
            refined_xyxy[:, 3].clamp_(max=img_h)

            if rescale:
                refined_xyxy = refined_xyxy / sf

            results.bboxes = refined_xyxy

            iou_score = iou_pred.squeeze(-1).sigmoid()
            results.scores = results.scores * iou_score

        return results_list

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
        self._cached_tooth_mask = None
        self._cached_c2_feat = None
        return out
