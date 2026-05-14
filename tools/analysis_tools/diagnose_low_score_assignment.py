# Copyright (c) OpenMMLab. All rights reserved.
"""Trace low-score IoU-valid DINO queries through Hungarian assignment.

The goal is to decide whether low_score_hit FNs are caused by:

1. the query being assigned positive by Hungarian matching but still receiving a
   low classification score, or
2. an IoU-valid query being excluded by one-to-one matching and therefore
   trained as background.

The script reproduces final-decoder Hungarian assignment on validation images
using the model's own assigner and writes both aggregate counts and per-GT
trace records.
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import register_all_modules


DEFAULT_CONFIG = (
    'work_dirs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline_decay/'
    '20260512_094157/vis_data/config.py')
DEFAULT_CHECKPOINT = (
    'work_dirs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline_decay/'
    'epoch_17.pth')
DEFAULT_OUT_DIR = (
    'work_dirs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline_decay/'
    'epoch17_low_score_assignment')


def _patch_torch_load_for_mmengine_checkpoints() -> None:
    orig_torch_load = torch.load

    def _wrapped(*args, **kwargs):
        try:
            kwargs.setdefault('weights_only', False)
            return orig_torch_load(*args, **kwargs)
        except TypeError:
            kwargs.pop('weights_only', None)
            return orig_torch_load(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[assignment]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, defaultdict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _scalar_int(x: Any) -> int:
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def _boxes_xyxy(inst) -> torch.Tensor:
    if inst is None or len(inst) == 0:
        return torch.zeros(0, 4)
    bboxes = inst.bboxes
    if hasattr(bboxes, 'tensor'):
        tensor = bboxes.tensor
    elif isinstance(bboxes, torch.Tensor):
        tensor = bboxes
    else:
        tensor = torch.as_tensor(np.asarray(bboxes))
    return tensor.detach().float().cpu()


def _bboxes_tensor(inst) -> torch.Tensor:
    bboxes = inst.bboxes
    if hasattr(bboxes, 'tensor'):
        return bboxes.tensor
    return bboxes


def _labels(inst) -> List[int]:
    if inst is None or len(inst) == 0 or not hasattr(inst, 'labels'):
        return []
    return [int(x) for x in inst.labels.detach().cpu().tolist()]


def _scale_factor_xyxy(img_meta: dict, device, dtype) -> torch.Tensor:
    scale_factor = torch.as_tensor(
        img_meta['scale_factor'], device=device, dtype=dtype)
    if scale_factor.numel() == 2:
        scale_factor = scale_factor.repeat(2)
    return scale_factor


def _query_assignment_status(assigned_gt: int, target_gt_idx: int) -> str:
    if assigned_gt == target_gt_idx + 1:
        return 'assigned_same_gt_positive'
    if assigned_gt > 0:
        return 'assigned_other_gt_positive'
    if assigned_gt == 0:
        return 'assigned_background'
    return 'assigned_ignore'


def _box_meta(box: Sequence[float], img_w: int, img_h: int) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    return dict(
        width=w,
        height=h,
        area=area,
        rel_area=area / max(float(img_w * img_h), 1.0),
        aspect_ratio=w / max(h, 1e-6),
        rel_center_x=((x1 + x2) * 0.5) / img_w,
        rel_center_y=((y1 + y2) * 0.5) / img_h)


def _rank_ascending(values: torch.Tensor, index: int) -> int:
    return int((values < values[index]).sum().item()) + 1


def _rank_descending(values: torch.Tensor, index: int) -> int:
    return int((values > values[index]).sum().item()) + 1


def _compute_match_cost_matrix(assigner, pred_instances: InstanceData,
                               gt_instances: InstanceData,
                               img_meta: dict) -> Optional[torch.Tensor]:
    if len(gt_instances) == 0 or len(pred_instances) == 0:
        return None
    costs = []
    for match_cost in assigner.match_costs:
        costs.append(match_cost(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta))
    return torch.stack(costs).sum(dim=0).detach().cpu()


def _candidate_record(query_idx: int, gt_idx: int, scores: torch.Tensor,
                      ious_for_gt: torch.Tensor, cost_for_gt: Optional[torch.Tensor],
                      gt_inds: torch.Tensor) -> Dict[str, Any]:
    assigned_gt = int(gt_inds[query_idx].item())
    record = dict(
        query_index=int(query_idx),
        score=float(scores[query_idx]),
        iou_to_gt=float(ious_for_gt[query_idx]),
        assigned_gt_ind=assigned_gt,
        assignment_status=_query_assignment_status(assigned_gt, gt_idx),
        score_rank_desc=_rank_descending(scores, query_idx),
        iou_rank_desc=_rank_descending(ious_for_gt, query_idx))
    if cost_for_gt is not None:
        record['hungarian_cost_to_gt'] = float(cost_for_gt[query_idx])
        record['cost_rank_asc'] = _rank_ascending(cost_for_gt, query_idx)
    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose whether low-score IoU-valid queries are positives or one-to-one background.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument(
        '--store-topk-candidates',
        type=int,
        default=8,
        help='Number of low-score IoU-valid candidate queries to store per GT.')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _patch_torch_load_for_mmengine_checkpoints()
    register_all_modules()
    init_default_scope('mmdet')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.resume = False
    if cfg.model.backbone.get('init_cfg', None) is not None:
        cfg.model.backbone.init_cfg = None

    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = osp.join(args.out_dir, 'low_score_assignment_traces.jsonl')
    summary_path = osp.join(args.out_dir, 'low_score_assignment_summary.json')

    loader = Runner.build_dataloader(cfg.val_dataloader)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    summary: Dict[str, Any] = dict(
        config=osp.abspath(args.config),
        checkpoint=osp.abspath(args.checkpoint),
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        total_images=0,
        total_gt=0,
        low_score_hit_gt=0,
        low_score_iou_valid_candidates=0,
        gt_conclusion_counts=Counter(),
        candidate_assignment_counts=Counter(),
        assigned_positive_low_score_stats=Counter(),
        one_to_one_suppressed_stats=Counter(),
        gt_size_conclusion_counts=defaultdict(Counter),
        hard_examples=[])

    with open(trace_path, 'w', encoding='utf-8') as trace_file:
        stop = False
        for data_batch in tqdm(loader, desc='assignment diagnostic'):
            if args.max_images is not None and summary['total_images'] >= args.max_images:
                break

            with torch.no_grad():
                data = model.data_preprocessor(data_batch, False)
                batch_inputs = data['inputs']
                batch_data_samples = data['data_samples']
                img_feats = model.extract_feat(batch_inputs)
                head_inputs = model.forward_transformer(
                    img_feats, batch_data_samples)
                all_cls_scores, all_bbox_preds = model.bbox_head(
                    head_inputs['hidden_states'], head_inputs['references'])

            final_cls_scores = all_cls_scores[-1]
            final_bbox_preds = all_bbox_preds[-1]

            for batch_idx, sample in enumerate(batch_data_samples):
                if args.max_images is not None and summary['total_images'] >= args.max_images:
                    stop = True
                    break

                img_meta = sample.metainfo
                img_h, img_w = img_meta['img_shape']
                ori_h, ori_w = img_meta['ori_shape']
                gt_instances_orig = sample.gt_instances
                gt_boxes_orig_cpu = _boxes_xyxy(gt_instances_orig)
                gt_labels = _labels(gt_instances_orig)
                num_gt = gt_boxes_orig_cpu.shape[0]

                cls_logits = final_cls_scores[batch_idx]
                scores = cls_logits.sigmoid().squeeze(-1).detach().cpu()
                bbox_pred = final_bbox_preds[batch_idx]
                pred_boxes_img = bbox_cxcywh_to_xyxy(bbox_pred)
                factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                pred_boxes_img = pred_boxes_img * factor
                pred_boxes_img[:, 0::2].clamp_(min=0, max=img_w)
                pred_boxes_img[:, 1::2].clamp_(min=0, max=img_h)

                scale_factor = _scale_factor_xyxy(
                    img_meta, pred_boxes_img.device, pred_boxes_img.dtype)
                pred_boxes_orig = pred_boxes_img / scale_factor

                # The validation pipeline loads annotations after Resize, so
                # GT boxes are original-scale. Reproduce the training assigner
                # in resized image coordinates, but diagnose IoU in original
                # coordinates to match failure_analysis_iou0.5 semantics.
                gt_instances_assign = InstanceData()
                gt_instances_assign.bboxes = (
                    _bboxes_tensor(gt_instances_orig).to(
                        device=pred_boxes_img.device,
                        dtype=pred_boxes_img.dtype) * scale_factor)
                gt_instances_assign.labels = gt_instances_orig.labels.to(
                    device=pred_boxes_img.device)

                pred_instances = InstanceData(
                    scores=cls_logits,
                    bboxes=pred_boxes_img)
                assign_result = model.bbox_head.assigner.assign(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances_assign,
                    img_meta=img_meta)
                gt_inds = assign_result.gt_inds.detach().cpu()
                cost_matrix = _compute_match_cost_matrix(
                    model.bbox_head.assigner, pred_instances,
                    gt_instances_assign, img_meta)

                pred_boxes_cpu = pred_boxes_orig.detach().float().cpu()
                ious = bbox_overlaps(
                    pred_boxes_cpu, gt_boxes_orig_cpu, mode='iou') if num_gt else torch.zeros(
                        pred_boxes_cpu.shape[0], 0)

                img_path = sample.img_path
                if isinstance(img_path, list):
                    img_path = img_path[0]

                summary['total_images'] += 1
                summary['total_gt'] += int(num_gt)

                for gt_idx in range(num_gt):
                    ious_for_gt = ious[:, gt_idx]
                    low_valid_mask = (ious_for_gt >= args.iou_thr) & (scores < args.score_thr)
                    high_valid_mask = (ious_for_gt >= args.iou_thr) & (scores >= args.score_thr)
                    if not bool(low_valid_mask.any()) or bool(high_valid_mask.any()):
                        continue

                    summary['low_score_hit_gt'] += 1
                    low_query_indices = torch.nonzero(
                        low_valid_mask, as_tuple=False).squeeze(1)
                    summary['low_score_iou_valid_candidates'] += int(
                        low_query_indices.numel())

                    assigned_query = torch.nonzero(
                        gt_inds == gt_idx + 1, as_tuple=False).squeeze(1)
                    assigned_query_idx = int(assigned_query[0].item()) if assigned_query.numel() else None
                    assigned_record = None
                    if assigned_query_idx is not None:
                        assigned_record = _candidate_record(
                            assigned_query_idx, gt_idx, scores, ious_for_gt,
                            cost_matrix[:, gt_idx] if cost_matrix is not None else None,
                            gt_inds)

                    same_assigned_low = [
                        int(q.item()) for q in low_query_indices
                        if int(gt_inds[int(q.item())].item()) == gt_idx + 1
                    ]
                    background_low = [
                        int(q.item()) for q in low_query_indices
                        if int(gt_inds[int(q.item())].item()) == 0
                    ]
                    other_gt_low = [
                        int(q.item()) for q in low_query_indices
                        if int(gt_inds[int(q.item())].item()) > 0
                        and int(gt_inds[int(q.item())].item()) != gt_idx + 1
                    ]

                    if same_assigned_low:
                        conclusion = 'assigned_positive_but_low_score'
                    else:
                        conclusion = 'iou_valid_candidates_suppressed_by_one_to_one'

                    summary['gt_conclusion_counts'][conclusion] += 1
                    for q in low_query_indices.tolist():
                        status = _query_assignment_status(
                            int(gt_inds[q].item()), gt_idx)
                        summary['candidate_assignment_counts'][status] += 1

                    if conclusion == 'assigned_positive_but_low_score':
                        summary['assigned_positive_low_score_stats']['gt'] += 1
                        summary['assigned_positive_low_score_stats']['low_candidates'] += len(same_assigned_low)
                    else:
                        summary['one_to_one_suppressed_stats']['gt'] += 1
                        summary['one_to_one_suppressed_stats']['background_candidates'] += len(background_low)
                        summary['one_to_one_suppressed_stats']['other_gt_candidates'] += len(other_gt_low)

                    gt_box = gt_boxes_orig_cpu[gt_idx].tolist()
                    meta = _box_meta(gt_box, ori_w, ori_h)
                    size_bucket = _size_bucket(meta['area'])
                    summary['gt_size_conclusion_counts'][conclusion][size_bucket] += 1

                    sorted_low = low_query_indices[
                        torch.argsort(ious_for_gt[low_query_indices], descending=True)]
                    candidate_records = [
                        _candidate_record(
                            int(query_idx.item()), gt_idx, scores, ious_for_gt,
                            cost_matrix[:, gt_idx] if cost_matrix is not None else None,
                            gt_inds)
                        for query_idx in sorted_low[:args.store_topk_candidates]
                    ]

                    record = dict(
                        img_id=_scalar_int(sample.img_id),
                        img_path=img_path,
                        img_shape=[int(img_h), int(img_w)],
                        ori_shape=[int(ori_h), int(ori_w)],
                        gt_index=gt_idx,
                        gt_label=gt_labels[gt_idx] if gt_idx < len(gt_labels) else None,
                        gt_bbox_xyxy=gt_box,
                        gt_meta=meta,
                        conclusion=conclusion,
                        num_low_score_iou_valid_candidates=int(low_query_indices.numel()),
                        num_low_candidates_assigned_same_gt=len(same_assigned_low),
                        num_low_candidates_assigned_background=len(background_low),
                        num_low_candidates_assigned_other_gt=len(other_gt_low),
                        assigned_query_for_gt=assigned_record,
                        low_score_iou_valid_candidates=candidate_records)
                    trace_file.write(json.dumps(_json_safe(record), ensure_ascii=False) + '\n')

                    if len(summary['hard_examples']) < 100:
                        summary['hard_examples'].append(record)

            if stop:
                break

    summary['trace_file'] = osp.abspath(trace_path)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    print(f'Done. Summary: {summary_path}')
    print(f'Traces: {trace_path}')


def _size_bucket(area: float) -> str:
    if area < 1500:
        return 'tiny'
    if area < 3000:
        return 'small'
    if area < 8000:
        return 'medium'
    return 'large'


if __name__ == '__main__':
    main()
