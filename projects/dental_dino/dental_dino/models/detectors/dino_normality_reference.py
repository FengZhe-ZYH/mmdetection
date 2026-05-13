# Copyright (c) OpenMMLab. DINO with tooth-interior normality reference.

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList


@MODELS.register_module()
class DINONormalityReference(DINO):
    """DINO plus tooth-interior normal-vs-lesion auxiliary learning.

    The detector keeps the original DINO detection path unchanged.  During
    training it uses tooth semantic masks to extract two feature sets from the
    highest-resolution neck feature:

    - lesion features: average pooled inside each GT lesion box
    - healthy features: same tooth mask minus all lesion boxes

    Healthy features update a global EMA normal prototype.  Auxiliary losses
    then push healthy features toward the prototype, lesion features away from
    it, and train a lightweight anomaly score head on lesion/healthy features.
    """

    def __init__(self,
                 *args,
                 use_normality_reference: bool = True,
                 normality_feat_level: int = 0,
                 normality_proj_dim: int = 128,
                 normality_ema_momentum: float = 0.05,
                 lesion_expand_ratio: float = 0.15,
                 min_region_pixels: int = 4,
                 loss_normality_weight: float = 0.2,
                 loss_anomaly_weight: float = 0.1,
                 lesion_proto_margin: float = 0.5,
                 lesion_healthy_margin: float = 0.2,
                 log_interval: int = 200,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_normality_reference = use_normality_reference
        self.normality_feat_level = int(normality_feat_level)
        self.normality_proj_dim = int(normality_proj_dim)
        self.normality_ema_momentum = float(normality_ema_momentum)
        self.lesion_expand_ratio = float(lesion_expand_ratio)
        self.min_region_pixels = int(min_region_pixels)
        self.loss_normality_weight = float(loss_normality_weight)
        self.loss_anomaly_weight = float(loss_anomaly_weight)
        self.lesion_proto_margin = float(lesion_proto_margin)
        self.lesion_healthy_margin = float(lesion_healthy_margin)
        self.log_interval = int(log_interval)

        if self.normality_feat_level < 0:
            raise ValueError('normality_feat_level must be non-negative.')
        if self.normality_ema_momentum <= 0 or self.normality_ema_momentum > 1:
            raise ValueError('normality_ema_momentum must be in (0, 1].')

        self.normality_projector = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.normality_proj_dim),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.normality_proj_dim, self.normality_proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.normality_proj_dim // 2, 1),
        )
        self.register_buffer(
            'healthy_prototype',
            torch.zeros(self.normality_proj_dim),
            persistent=True)
        self.register_buffer(
            'healthy_prototype_initialized',
            torch.tensor(False),
            persistent=True)

        MMLogger.get_current_instance().info(
            '[DINONormalityReference] enabled=%s feat_level=%d proj_dim=%d '
            'ema=%s loss_normality=%s loss_anomaly=%s',
            use_normality_reference, normality_feat_level, normality_proj_dim,
            normality_ema_momentum, loss_normality_weight,
            loss_anomaly_weight)

    def init_weights(self) -> None:
        super().init_weights()
        for module in self.normality_projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.anomaly_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _tooth_tensor_from_samples(
            self, batch_data_samples: SampleList,
            ref_hw: Tuple[int, int]) -> Tensor:
        masks: List[Tensor] = []
        for sample in batch_data_samples:
            if not hasattr(sample, 'gt_sem_seg') or sample.gt_sem_seg is None:
                raise KeyError(
                    'DINONormalityReference requires gt_sem_seg. Use '
                    'data_prefix["seg"], LoadAnnotations(with_seg=True), and '
                    'pad_seg=True in the data preprocessor.')
            mask = sample.gt_sem_seg.sem_seg
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dim() != 3 or mask.size(0) != 1:
                raise ValueError(
                    'gt_sem_seg.sem_seg must have shape [H,W] or [1,H,W], '
                    f'got {tuple(mask.shape)}')
            if mask.shape[-2:] != ref_hw:
                raise RuntimeError(
                    f'tooth mask shape {tuple(mask.shape[-2:])} != input '
                    f'shape {ref_hw}. Enable pad_seg in DetDataPreprocessor.')
            masks.append(mask)
        return torch.stack(masks, dim=0).float()

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.use_normality_reference:
            losses.update(
                self._normality_reference_loss(
                    img_feats, batch_inputs, batch_data_samples))
        return losses

    def _normality_reference_loss(self, img_feats: Tuple[Tensor, ...],
                                  batch_inputs: Tensor,
                                  batch_data_samples: SampleList
                                  ) -> Dict[str, Tensor]:
        if self.normality_feat_level >= len(img_feats):
            raise ValueError(
                f'normality_feat_level={self.normality_feat_level} but only '
                f'{len(img_feats)} feature levels are available.')

        feat = img_feats[self.normality_feat_level]
        tooth = self._tooth_tensor_from_samples(
            batch_data_samples, (batch_inputs.shape[2], batch_inputs.shape[3]))
        tooth_feat = F.interpolate(
            tooth, size=feat.shape[-2:], mode='nearest')[:, 0].long()

        lesion_raw, healthy_raw = self._collect_roi_features(
            feat, tooth_feat, batch_inputs, batch_data_samples)
        zero = feat.sum() * 0.0
        if lesion_raw.numel() == 0 or healthy_raw.numel() == 0:
            return dict(
                loss_normality_contrast=zero,
                loss_anomaly_score=zero,
                normality_healthy_proto=zero)

        lesion = F.normalize(self.normality_projector(lesion_raw), dim=1)
        healthy = F.normalize(self.normality_projector(healthy_raw), dim=1)
        self._update_healthy_prototype(healthy)
        proto = F.normalize(self.healthy_prototype.detach(), dim=0)

        healthy_proto_loss = (1.0 - (healthy * proto).sum(dim=1)).mean()
        lesion_proto_dist = 1.0 - (lesion * proto).sum(dim=1)
        lesion_proto_loss = F.relu(
            self.lesion_proto_margin - lesion_proto_dist).mean()
        lesion_healthy_sim = (lesion * healthy).sum(dim=1)
        lesion_healthy_loss = F.relu(
            lesion_healthy_sim - self.lesion_healthy_margin).mean()

        contrast_loss = (
            healthy_proto_loss + lesion_proto_loss + lesion_healthy_loss)
        anomaly_logits = torch.cat(
            [self.anomaly_head(lesion), self.anomaly_head(healthy)], dim=0)
        anomaly_targets = torch.cat([
            torch.ones(lesion.size(0), 1, device=lesion.device),
            torch.zeros(healthy.size(0), 1, device=healthy.device)
        ], dim=0)
        anomaly_loss = F.binary_cross_entropy_with_logits(
            anomaly_logits, anomaly_targets)

        return dict(
            loss_normality_contrast=self.loss_normality_weight * contrast_loss,
            loss_anomaly_score=self.loss_anomaly_weight * anomaly_loss,
            normality_healthy_proto=healthy_proto_loss.detach())

    def _collect_roi_features(self, feat: Tensor, tooth_feat: Tensor,
                              batch_inputs: Tensor,
                              batch_data_samples: SampleList
                              ) -> Tuple[Tensor, Tensor]:
        lesion_feats: List[Tensor] = []
        healthy_feats: List[Tensor] = []
        _, _, feat_h, feat_w = feat.shape
        img_h, img_w = batch_inputs.shape[2], batch_inputs.shape[3]

        for batch_idx, sample in enumerate(batch_data_samples):
            bboxes = sample.gt_instances.bboxes
            if bboxes.numel() == 0:
                continue
            lesion_remove = torch.zeros(
                (feat_h, feat_w), dtype=torch.bool, device=feat.device)
            scaled_boxes = []
            for bbox in bboxes:
                box = self._scale_box_to_feature(
                    bbox, img_w, img_h, feat_w, feat_h,
                    expand_ratio=self.lesion_expand_ratio)
                scaled_boxes.append(box)
                x1, y1, x2, y2 = box
                lesion_remove[y1:y2, x1:x2] = True

            for bbox, lesion_box in zip(bboxes, scaled_boxes):
                lesion_mask = torch.zeros_like(lesion_remove)
                x1, y1, x2, y2 = self._scale_box_to_feature(
                    bbox, img_w, img_h, feat_w, feat_h, expand_ratio=0.0)
                lesion_mask[y1:y2, x1:x2] = True
                if lesion_mask.sum() < self.min_region_pixels:
                    lesion_mask = torch.zeros_like(lesion_remove)
                    x1, y1, x2, y2 = lesion_box
                    lesion_mask[y1:y2, x1:x2] = True
                if lesion_mask.sum() < self.min_region_pixels:
                    continue

                tooth_mask = self._same_tooth_mask(
                    tooth_feat[batch_idx], lesion_mask)
                healthy_mask = tooth_mask & (~lesion_remove)
                if healthy_mask.sum() < self.min_region_pixels:
                    healthy_mask = (tooth_feat[batch_idx] > 0) & (~lesion_remove)
                if healthy_mask.sum() < self.min_region_pixels:
                    continue

                lesion_feats.append(
                    self._masked_average(feat[batch_idx], lesion_mask))
                healthy_feats.append(
                    self._masked_average(feat[batch_idx], healthy_mask))

        if not lesion_feats:
            empty = feat.new_zeros((0, feat.size(1)))
            return empty, empty
        return torch.stack(lesion_feats, dim=0), torch.stack(healthy_feats, dim=0)

    def _scale_box_to_feature(self, bbox: Tensor, img_w: int, img_h: int,
                              feat_w: int, feat_h: int,
                              expand_ratio: float = 0.0
                              ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox.float()
        if expand_ratio > 0:
            bw = (x2 - x1).clamp(min=1.0)
            bh = (y2 - y1).clamp(min=1.0)
            x1 = x1 - bw * expand_ratio
            x2 = x2 + bw * expand_ratio
            y1 = y1 - bh * expand_ratio
            y2 = y2 + bh * expand_ratio

        sx = feat_w / float(img_w)
        sy = feat_h / float(img_h)
        fx1 = int(torch.floor(x1 * sx).clamp(0, feat_w - 1).item())
        fy1 = int(torch.floor(y1 * sy).clamp(0, feat_h - 1).item())
        fx2 = int(torch.ceil(x2 * sx).clamp(fx1 + 1, feat_w).item())
        fy2 = int(torch.ceil(y2 * sy).clamp(fy1 + 1, feat_h).item())
        return fx1, fy1, fx2, fy2

    def _same_tooth_mask(self, tooth_map: Tensor, lesion_mask: Tensor) -> Tensor:
        lesion_ids = tooth_map[lesion_mask]
        lesion_ids = lesion_ids[lesion_ids > 0]
        if lesion_ids.numel() == 0:
            return tooth_map > 0
        unique_ids, counts = lesion_ids.unique(return_counts=True)
        tooth_id = unique_ids[counts.argmax()]
        return tooth_map == tooth_id

    @staticmethod
    def _masked_average(feat: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=feat.dtype).unsqueeze(0)
        return (feat * weights).sum(dim=(1, 2)) / weights.sum().clamp(min=1.0)

    @torch.no_grad()
    def _update_healthy_prototype(self, healthy: Tensor) -> None:
        batch_proto = F.normalize(healthy.detach().mean(dim=0), dim=0)
        if not bool(self.healthy_prototype_initialized.item()):
            self.healthy_prototype.copy_(batch_proto)
            self.healthy_prototype_initialized.fill_(True)
            return
        momentum = self.normality_ema_momentum
        updated = (1.0 - momentum) * self.healthy_prototype + momentum * batch_proto
        self.healthy_prototype.copy_(F.normalize(updated, dim=0))

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        return self.bbox_head.forward(**head_inputs_dict)
