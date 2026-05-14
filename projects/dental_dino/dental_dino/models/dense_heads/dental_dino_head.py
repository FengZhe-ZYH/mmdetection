from typing import Dict, List, Optional, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean


@MODELS.register_module()
class DentalDINOHead(DINOHead):
    """DINO head variants for dental lesion confidence diagnostics.

    The three options are intentionally independent so each experiment can be
    compared against the same baseline:

    - UPQC: extra positive-only under-confidence classification loss.
    - Quality calibration: an IoU-quality logit branch added to cls logits at
      inference.
    - Suppressed query recovery: soft classification targets for unmatched
      IoU-valid queries, without changing Hungarian matching.
    """

    def __init__(self,
                 *args,
                 enable_upqc: bool = False,
                 upqc_loss_weight: float = 0.5,
                 enable_quality_branch: bool = False,
                 quality_loss_weight: float = 0.5,
                 quality_lambda: float = 0.5,
                 quality_neg_weight: float = 0.25,
                 enable_suppressed_recovery: bool = False,
                 suppressed_loss_weight: float = 0.5,
                 suppressed_iou_thr: float = 0.5,
                 suppressed_topk: int = 3,
                 **kwargs) -> None:
        self.enable_upqc = enable_upqc
        self.upqc_loss_weight = upqc_loss_weight
        self.enable_quality_branch = enable_quality_branch
        self.quality_loss_weight = quality_loss_weight
        self.quality_lambda = quality_lambda
        self.quality_neg_weight = quality_neg_weight
        self.enable_suppressed_recovery = enable_suppressed_recovery
        self.suppressed_loss_weight = suppressed_loss_weight
        self.suppressed_iou_thr = suppressed_iou_thr
        self.suppressed_topk = suppressed_topk
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()
        if self.enable_quality_branch:
            fc_quality = Linear(self.embed_dims, self.cls_out_channels)
            if self.share_pred_layer:
                self.quality_branches = nn.ModuleList(
                    [fc_quality for _ in range(self.num_pred_layer)])
            else:
                self.quality_branches = nn.ModuleList([
                    copy.deepcopy(fc_quality)
                    for _ in range(self.num_pred_layer)
                ])

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, ...]:
        if not self.enable_quality_branch:
            return super().forward(hidden_states, references)

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_quality_logits = []

        for layer_id in range(hidden_states.shape[0]):
            reference = self._inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            outputs_quality = self.quality_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                tmp_reg_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_quality_logits.append(outputs_quality)

        return (torch.stack(all_layers_outputs_classes),
                torch.stack(all_layers_outputs_coords),
                torch.stack(all_layers_quality_logits))

    @staticmethod
    def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        all_layers_quality_logits = None
        if len(outs) == 3:
            all_layers_cls_scores, all_layers_bbox_preds, \
                all_layers_quality_logits = outs
        else:
            all_layers_cls_scores, all_layers_bbox_preds = outs

        return self.loss_by_feat(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas,
            dn_meta,
            all_layers_quality_logits=all_layers_quality_logits)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None,
        all_layers_quality_logits: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        loss_dict = super().loss_by_feat(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            enc_cls_scores,
            enc_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            dn_meta,
            batch_gt_instances_ignore)

        if not (self.enable_upqc or self.enable_quality_branch
                or self.enable_suppressed_recovery):
            return loss_dict

        (matching_cls_scores, matching_bbox_preds, _, _) = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        matching_quality_logits = None
        if all_layers_quality_logits is not None:
            (matching_quality_logits, _, _, _) = self.split_outputs(
                all_layers_quality_logits, all_layers_bbox_preds, dn_meta)

        if self.enable_upqc:
            upqc_losses = self.loss_upqc(
                matching_cls_scores, matching_bbox_preds, batch_gt_instances,
                batch_img_metas)
            self._add_layer_losses(loss_dict, 'loss_cls_upqc', upqc_losses)

        if self.enable_quality_branch and matching_quality_logits is not None:
            quality_losses = self.loss_quality_branch(
                matching_cls_scores, matching_bbox_preds,
                matching_quality_logits, batch_gt_instances, batch_img_metas)
            self._add_layer_losses(loss_dict, 'loss_quality', quality_losses)

        if self.enable_suppressed_recovery:
            suppressed_losses = self.loss_suppressed_recovery(
                matching_cls_scores, matching_bbox_preds, batch_gt_instances,
                batch_img_metas)
            self._add_layer_losses(
                loss_dict, 'loss_cls_suppressed', suppressed_losses)

        return loss_dict

    @staticmethod
    def _add_layer_losses(loss_dict: Dict[str, Tensor], name: str,
                          losses: List[Tensor]) -> None:
        loss_dict[name] = losses[-1]
        for layer_id, loss in enumerate(losses[:-1]):
            loss_dict[f'd{layer_id}.{name}'] = loss

    def _assign_for_aux(self, cls_score: Tensor, bbox_pred: Tensor,
                        gt_instances: InstanceData,
                        img_meta: dict):
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pred_bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        pred_instances = InstanceData(scores=cls_score, bboxes=pred_bboxes)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        return assign_result, pred_bboxes

    def loss_upqc(self, all_layers_cls_scores: Tensor,
                  all_layers_bbox_preds: Tensor,
                  batch_gt_instances: InstanceList,
                  batch_img_metas: List[dict]) -> List[Tensor]:
        return [
            self.loss_upqc_single(cls_scores, bbox_preds, batch_gt_instances,
                                  batch_img_metas)
            for cls_scores, bbox_preds in zip(all_layers_cls_scores,
                                              all_layers_bbox_preds)
        ]

    def loss_upqc_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                         batch_gt_instances: InstanceList,
                         batch_img_metas: List[dict]) -> Tensor:
        loss_terms = []
        num_pos = 0
        for img_id, gt_instances in enumerate(batch_gt_instances):
            if len(gt_instances) == 0:
                continue
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            assign_result, pred_bboxes = self._assign_for_aux(
                cls_score, bbox_pred, gt_instances, batch_img_metas[img_id])
            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0, as_tuple=False).squeeze(1)
            if pos_inds.numel() == 0:
                continue
            pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
            pos_labels = gt_instances.labels[pos_gt_inds.long()]
            pos_ious = bbox_overlaps(
                pred_bboxes[pos_inds],
                gt_instances.bboxes[pos_gt_inds.long()],
                mode='iou',
                is_aligned=True).clamp(min=0, max=1).detach()
            pos_logits = cls_score[pos_inds, pos_labels]
            pos_scores = pos_logits.sigmoid().detach()
            targets = 0.5 + 0.5 * pos_ious
            weights = pos_ious * (1 - pos_scores)
            loss_terms.append(
                F.binary_cross_entropy_with_logits(
                    pos_logits, targets, reduction='none') * weights)
            num_pos += int(pos_inds.numel())

        if not loss_terms:
            return cls_scores.sum() * 0
        avg_factor = self._distributed_avg_factor(
            cls_scores, max(num_pos, 1))
        return torch.cat(loss_terms).sum() / avg_factor * self.upqc_loss_weight

    def loss_quality_branch(self, all_layers_cls_scores: Tensor,
                            all_layers_bbox_preds: Tensor,
                            all_layers_quality_logits: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> List[Tensor]:
        losses = []
        for cls_scores, bbox_preds, quality_logits in zip(
                all_layers_cls_scores, all_layers_bbox_preds,
                all_layers_quality_logits):
            losses.append(
                self.loss_quality_branch_single(
                    cls_scores, bbox_preds, quality_logits,
                    batch_gt_instances, batch_img_metas))
        return losses

    def loss_quality_branch_single(self, cls_scores: Tensor,
                                   bbox_preds: Tensor,
                                   quality_logits: Tensor,
                                   batch_gt_instances: InstanceList,
                                   batch_img_metas: List[dict]) -> Tensor:
        targets = quality_logits.new_zeros(quality_logits.shape)
        weights = quality_logits.new_full(
            quality_logits.shape, self.quality_neg_weight)

        for img_id, gt_instances in enumerate(batch_gt_instances):
            if len(gt_instances) == 0:
                continue
            assign_result, pred_bboxes = self._assign_for_aux(
                cls_scores[img_id], bbox_preds[img_id], gt_instances,
                batch_img_metas[img_id])
            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0, as_tuple=False).squeeze(1)
            if pos_inds.numel() == 0:
                continue
            pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
            pos_labels = gt_instances.labels[pos_gt_inds.long()]
            pos_ious = bbox_overlaps(
                pred_bboxes[pos_inds],
                gt_instances.bboxes[pos_gt_inds.long()],
                mode='iou',
                is_aligned=True).clamp(min=0, max=1).detach()
            targets[img_id, pos_inds, pos_labels] = pos_ious
            weights[img_id, pos_inds, pos_labels] = 1.0

        loss = F.binary_cross_entropy_with_logits(
            quality_logits, targets, reduction='none') * weights
        avg_factor = self._distributed_avg_factor(
            quality_logits, max(float(quality_logits.numel()), 1.0))
        return loss.sum() / avg_factor * self.quality_loss_weight

    def loss_suppressed_recovery(self, all_layers_cls_scores: Tensor,
                                 all_layers_bbox_preds: Tensor,
                                 batch_gt_instances: InstanceList,
                                 batch_img_metas: List[dict]) -> List[Tensor]:
        return [
            self.loss_suppressed_recovery_single(
                cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)
            for cls_scores, bbox_preds in zip(all_layers_cls_scores,
                                              all_layers_bbox_preds)
        ]

    def loss_suppressed_recovery_single(self, cls_scores: Tensor,
                                        bbox_preds: Tensor,
                                        batch_gt_instances: InstanceList,
                                        batch_img_metas: List[dict]) -> Tensor:
        loss_terms = []
        num_aux = 0
        for img_id, gt_instances in enumerate(batch_gt_instances):
            if len(gt_instances) == 0:
                continue
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            assign_result, pred_bboxes = self._assign_for_aux(
                cls_score, bbox_pred, gt_instances, batch_img_metas[img_id])
            unmatched_mask = assign_result.gt_inds == 0
            if not bool(unmatched_mask.any().item()):
                continue

            overlaps = bbox_overlaps(
                pred_bboxes, gt_instances.bboxes, mode='iou')
            for gt_idx in range(len(gt_instances)):
                candidate_mask = (
                    unmatched_mask
                    & (overlaps[:, gt_idx] > self.suppressed_iou_thr))
                candidate_inds = torch.nonzero(
                    candidate_mask, as_tuple=False).squeeze(1)
                if candidate_inds.numel() == 0:
                    continue
                candidate_ious = overlaps[candidate_inds, gt_idx]
                topk = min(self.suppressed_topk, candidate_inds.numel())
                top_vals, top_order = torch.topk(candidate_ious, k=topk)
                selected = candidate_inds[top_order]
                label = int(gt_instances.labels[gt_idx].detach().item())
                logits = cls_score[selected, label]
                targets = (0.5 + 0.5 * top_vals.detach()).to(logits.dtype)
                loss_terms.append(
                    F.binary_cross_entropy_with_logits(
                        logits, targets, reduction='none'))
                num_aux += int(selected.numel())

        if not loss_terms:
            return cls_scores.sum() * 0
        avg_factor = self._distributed_avg_factor(
            cls_scores, max(num_aux, 1))
        return (torch.cat(loss_terms).sum() / avg_factor *
                self.suppressed_loss_weight)

    def _distributed_avg_factor(self, ref: Tensor, value: float) -> Tensor:
        avg_factor = ref.new_tensor([value])
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        return avg_factor.clamp(min=1.0)

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(hidden_states, references)
        if len(outs) == 3:
            cls_scores, bbox_preds, quality_logits = outs
        else:
            cls_scores, bbox_preds = outs
            quality_logits = None
        return self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            all_layers_quality_logits=quality_logits)

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False,
                        all_layers_quality_logits: Optional[Tensor] = None
                        ) -> InstanceList:
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        if self.enable_quality_branch and all_layers_quality_logits is not None:
            cls_scores = (
                cls_scores +
                self.quality_lambda * all_layers_quality_logits[-1])

        result_list = []
        for img_id in range(len(batch_img_metas)):
            results = self._predict_by_feat_single(
                cls_scores[img_id], bbox_preds[img_id],
                batch_img_metas[img_id], rescale)
            result_list.append(results)
        return result_list
