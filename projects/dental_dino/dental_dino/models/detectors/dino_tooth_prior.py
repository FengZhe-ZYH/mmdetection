# Copyright (c) OpenMMLab. DINO with tooth prior encoder, modulation, guided queries, aux losses.

from typing import Dict, List, Tuple, Union

import torch
from mmengine.logging import MMLogger
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from ..utils.dental_aux_losses import (HeatmapHead, gaussian_heatmap_target,
                                        heatmap_focal_loss, nwd_aux_loss)
from ..utils.tooth_prior_modules import (MaskGuidedFeatureModulation,
                                         ToothPriorEncoder)
from ..utils.tooth_query_generator import (batch_tooth_reference_points,
                                           invert_sigmoid_normalized)


@MODELS.register_module()
class DINOToothPrior(DINO):
    """DINO + tooth semantic prior (neck modulation, optional guided queries, aux losses).

    Requirements when using tooth mask paths: ``gt_sem_seg`` from
    ``LoadAnnotations(with_seg=True)`` and ``data_prefix['seg']``.
    """

    def __init__(self,
                 *args,
                 use_tooth_prior_encoder: bool = True,
                 use_mask_guided_modulation: bool = True,
                 modulation_strength: float = 1.0,
                 aux_prior_energy_weight: float = 1e-6,
                 tooth_prior_channels: int = 64,
                 gate_log_interval: int = 200,
                 use_tooth_guided_queries: bool = False,
                 queries_per_tooth: int = 2,
                 num_tooth_slots: int = 32,
                 use_heatmap_aux_loss: bool = False,
                 heatmap_loss_weight: float = 1.0,
                 use_nwd_loss: bool = False,
                 nwd_loss_weight: float = 1.0,
                 nwd_small_area_thr: float = 0.001,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.use_tooth_prior_encoder = use_tooth_prior_encoder
        self.use_mask_guided_modulation = use_mask_guided_modulation
        self.modulation_strength = float(modulation_strength)
        self.aux_prior_energy_weight = float(aux_prior_energy_weight)
        self.tooth_prior_channels = tooth_prior_channels
        self.gate_log_interval = gate_log_interval
        self.use_tooth_guided_queries = use_tooth_guided_queries
        self.queries_per_tooth = int(queries_per_tooth)
        self.num_tooth_slots = int(num_tooth_slots)
        self.use_heatmap_aux_loss = use_heatmap_aux_loss
        self.heatmap_loss_weight = float(heatmap_loss_weight)
        self.use_nwd_loss = use_nwd_loss
        self.nwd_loss_weight = float(nwd_loss_weight)
        self.nwd_small_area_thr = float(nwd_small_area_thr)

        if self.use_mask_guided_modulation and not self.use_tooth_prior_encoder:
            raise ValueError(
                'DINOToothPrior: use_mask_guided_modulation requires '
                'use_tooth_prior_encoder=True.')
        if self.use_tooth_guided_queries:
            if self.num_tooth_slots <= 0 or self.num_tooth_slots >= self.num_queries:
                raise ValueError(
                    f'num_tooth_slots must be in (0, num_queries={self.num_queries}), '
                    f'got {self.num_tooth_slots}')
        if self.use_heatmap_aux_loss and self.heatmap_loss_weight <= 0:
            raise ValueError('heatmap_loss_weight must be > 0 when enabled.')
        if self.use_nwd_loss and self.nwd_loss_weight <= 0:
            raise ValueError('nwd_loss_weight must be > 0 when enabled.')

        self.tooth_prior_encoder = None
        self.mask_modulation = None
        self.heatmap_head = None

        if self.use_tooth_prior_encoder:
            self.tooth_prior_encoder = ToothPriorEncoder(
                in_channels=1,
                prior_channels=tooth_prior_channels,
                num_levels=self.num_feature_levels)
        if self.use_mask_guided_modulation:
            self.mask_modulation = MaskGuidedFeatureModulation(
                feat_channels=self.embed_dims,
                prior_channels=tooth_prior_channels,
                num_levels=self.num_feature_levels,
                gate_log_interval=gate_log_interval,
                modulation_strength=self.modulation_strength)
        if self.use_heatmap_aux_loss:
            self.heatmap_head = HeatmapHead(in_channels=self.embed_dims)

        self._last_tooth_priors: Tuple[Tensor, ...] | None = None

        log = MMLogger.get_current_instance()
        log.info(
            '[DINOToothPrior] tooth_encoder=%s mask_modulation=%s strength=%s '
            'tooth_guided_queries=%s heatmap_aux=%s nwd=%s num_feature_levels=%d',
            self.use_tooth_prior_encoder, self.use_mask_guided_modulation,
            self.modulation_strength, self.use_tooth_guided_queries,
            self.use_heatmap_aux_loss, self.use_nwd_loss,
            self.num_feature_levels)

    def init_weights(self) -> None:
        super().init_weights()

        def _init_prior_conv(module: torch.nn.Module) -> None:
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        if self.tooth_prior_encoder is not None:
            self.tooth_prior_encoder.apply(_init_prior_conv)
        if self.mask_modulation is not None:
            self.mask_modulation.apply(_init_prior_conv)
        if self.heatmap_head is not None:
            self.heatmap_head.apply(_init_prior_conv)

    def _need_tooth_sem_seg(self) -> bool:
        return (self.use_tooth_prior_encoder or self.use_tooth_guided_queries)

    def _tooth_tensor_from_samples(
            self, batch_data_samples: SampleList,
            ref_hw: Tuple[int, int]) -> torch.Tensor:
        masks: List[Tensor] = []
        for s in batch_data_samples:
            if not hasattr(s, 'gt_sem_seg') or s.gt_sem_seg is None:
                raise KeyError(
                    'DINOToothPrior: missing gt_sem_seg. Use seg data_prefix and '
                    'LoadAnnotations(with_seg=True).')
            m = s.gt_sem_seg.sem_seg
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.dim() != 3:
                raise ValueError(
                    f'gt_sem_seg.sem_seg must be [H,W] or [1,H,W], got {tuple(m.shape)}')
            if m.shape[-2:] != ref_hw:
                raise RuntimeError(
                    f'tooth mask {tuple(m.shape[-2:])} != input {ref_hw}. '
                    'Use pad_seg in DetDataPreprocessor.')
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
        self._last_tooth_priors = None
        if not self.use_tooth_prior_encoder:
            return x
        ref_hw = (batch_inputs.shape[2], batch_inputs.shape[3])
        tooth = self._tooth_tensor_from_samples(batch_data_samples, ref_hw)
        priors = self.tooth_prior_encoder(tooth)
        if len(priors) != len(x):
            raise RuntimeError(
                f'prior levels {len(priors)} != neck levels {len(x)}')
        self._last_tooth_priors = priors
        if self.use_mask_guided_modulation and self.mask_modulation is not None:
            x = self.mask_modulation(x, priors)
        return x

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        bs, _, _ = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        num_tooth = self.num_tooth_slots if self.use_tooth_guided_queries else 0
        k_enc = self.num_queries - num_tooth
        if self.use_tooth_guided_queries:
            if k_enc <= 0:
                raise RuntimeError('k_enc <= 0; reduce num_tooth_slots.')

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=k_enc, dim=1)[1]
        topk_score_enc = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact_enc = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))

        if self.use_tooth_guided_queries:
            assert batch_data_samples is not None
            bh, bw = batch_data_samples[0].batch_input_shape
            tooth = self._tooth_tensor_from_samples(batch_data_samples,
                                                    (bh, bw))
            tooth_refs = batch_tooth_reference_points(
                tooth,
                self.queries_per_tooth,
                num_tooth,
                memory.device,
                memory.dtype)
            tooth_unact = invert_sigmoid_normalized(tooth_refs)
            template = topk_score_enc[:, -1:, :].expand(-1, num_tooth,
                                                         -1).clone().detach()
            topk_score = torch.cat([topk_score_enc, template], dim=1)
            topk_coords_unact = torch.cat(
                [topk_coords_unact_enc, tooth_unact], dim=1)
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            MMLogger.get_current_instance().info(
                '[tooth_guided_queries] num_tooth_slots=%d k_enc=%d', num_tooth,
                k_enc)
        else:
            topk_score = topk_score_enc
            topk_coords = topk_coords_unact_enc.sigmoid()
            topk_coords_unact = topk_coords_unact_enc.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self._extract_feat_with_tooth(batch_inputs,
                                                  batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.use_tooth_prior_encoder and not self.use_mask_guided_modulation:
            if self._last_tooth_priors is None:
                raise RuntimeError(
                    'tooth prior tensors missing; encoder should run in extract.')
            eng = sum((p**2).mean() for p in self._last_tooth_priors)
            losses['loss_tooth_prior_energy'] = self.aux_prior_energy_weight * eng

        if self.use_heatmap_aux_loss:
            if self.heatmap_head is None:
                raise RuntimeError('heatmap_head not built.')
            hm = self.heatmap_head(img_feats[0])
            metas = [s.metainfo for s in batch_data_samples]
            tgt = gaussian_heatmap_target(
                [s.gt_instances for s in batch_data_samples], metas, hm)
            losses['loss_heatmap'] = self.heatmap_loss_weight * heatmap_focal_loss(
                hm, tgt)

        if self.use_nwd_loss:
            hs = head_inputs_dict['hidden_states']
            refs = head_inputs_dict['references']
            _, bbox_preds = self.bbox_head(hs, refs)
            pred_last = bbox_preds[-1]
            nwd = nwd_aux_loss(
                pred_last, [s.gt_instances for s in batch_data_samples],
                [s.metainfo for s in batch_data_samples],
                small_area_thr=self.nwd_small_area_thr)
            losses['loss_nwd'] = self.nwd_loss_weight * nwd

        self._check_aux_in_total(losses)
        # Drop refs to prior tensors as soon as loss dict is built; the graph
        # is retained via losses / modulation path, not via this cache. Avoids
        # an extra strong reference across the optimizer step boundary.
        self._last_tooth_priors = None
        return losses

    def _check_aux_in_total(self, losses: dict) -> None:
        """Plan: if aux loss enabled it must appear in returned dict (not None)."""
        if self.use_heatmap_aux_loss:
            if 'loss_heatmap' not in losses or losses['loss_heatmap'] is None:
                raise RuntimeError(
                    'use_heatmap_aux_loss=True but loss_heatmap missing.')
        if self.use_nwd_loss:
            if 'loss_nwd' not in losses or losses['loss_nwd'] is None:
                raise RuntimeError('use_nwd_loss=True but loss_nwd missing.')

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
        self._last_tooth_priors = None
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self._extract_feat_with_tooth(batch_inputs,
                                                  batch_data_samples)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        out = self.bbox_head.forward(**head_inputs_dict)
        self._last_tooth_priors = None
        return out
