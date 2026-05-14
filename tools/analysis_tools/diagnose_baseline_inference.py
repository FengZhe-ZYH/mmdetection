# Copyright (c) OpenMMLab. All rights reserved.
"""Run validation inference and collect detailed detection diagnostics.

This is a deeper companion to ``analyze_val_failures.py``.  It keeps low-score
predictions and nearest-GT IoUs so later analysis can separate:

- confident true positives
- low-confidence hits that become false negatives under a score threshold
- localization/extent misses near a GT box
- duplicate/assignment false positives
- background or negative-image false positives
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import register_all_modules


DEFAULT_CONFIG = (
    'work_dirs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline_decay/'
    '20260512_094157/vis_data/config.py')
DEFAULT_CHECKPOINT = (
    'work_dirs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline_decay/'
    'epoch_17.pth')
DEFAULT_OUT_DIR = (
    'work_dirs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline_decay/'
    'epoch17_diagnostics')


def _patch_torch_load_for_mmengine_checkpoints() -> None:
    """PyTorch>=2.6 defaults to weights_only=True, which breaks mmengine ckpts."""
    orig_torch_load = torch.load

    def _wrapped(*args, **kwargs):
        try:
            kwargs.setdefault('weights_only', False)
            return orig_torch_load(*args, **kwargs)
        except TypeError:
            kwargs.pop('weights_only', None)
            return orig_torch_load(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[assignment]


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


def _labels(inst) -> List[int]:
    if inst is None or len(inst) == 0 or not hasattr(inst, 'labels'):
        return []
    return [int(x) for x in inst.labels.detach().cpu().tolist()]


def _scores(inst) -> torch.Tensor:
    if inst is None or len(inst) == 0 or not hasattr(inst, 'scores'):
        return torch.zeros(0)
    return inst.scores.detach().float().cpu()


def _scalar_int(x: Any) -> int:
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(',') if x.strip()]
    if not values:
        raise ValueError('Expected at least one threshold.')
    return sorted(set(values))


def greedy_match(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor,
                 pred_scores: torch.Tensor,
                 iou_thr: float) -> Tuple[List[int], List[int],
                                          List[Tuple[int, int]]]:
    """Score-ordered one-to-one matching used for fixed-threshold diagnostics."""
    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]
    if num_gt == 0 and num_pred == 0:
        return [], [], []
    if num_gt == 0:
        return list(range(num_pred)), [], []
    if num_pred == 0:
        return [], list(range(num_gt)), []

    ious = bbox_overlaps(pred_boxes, gt_boxes, mode='iou')
    order = torch.argsort(pred_scores, descending=True).tolist()
    matched_gt = set()
    fp_pred_idx: List[int] = []
    tp_pairs: List[Tuple[int, int]] = []

    for pred_idx in order:
        row = ious[pred_idx]
        best_iou, best_gt = float(row.max()), int(row.argmax())
        if best_iou >= iou_thr and best_gt not in matched_gt:
            matched_gt.add(best_gt)
            tp_pairs.append((pred_idx, best_gt))
        else:
            fp_pred_idx.append(pred_idx)

    fn_gt_idx = [gt_idx for gt_idx in range(num_gt) if gt_idx not in matched_gt]
    return fp_pred_idx, fn_gt_idx, tp_pairs


def box_metadata(box: Sequence[float], img_w: int, img_h: int) -> Dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x_band = 'left' if cx / img_w < 0.33 else 'center' if cx / img_w < 0.67 else 'right'
    y_band = 'top' if cy / img_h < 0.33 else 'middle' if cy / img_h < 0.67 else 'bottom'
    return dict(
        width=w,
        height=h,
        area=area,
        rel_area=area / max(float(img_w * img_h), 1.0),
        aspect_ratio=w / max(h, 1e-6),
        center_x=cx,
        center_y=cy,
        rel_center_x=cx / img_w,
        rel_center_y=cy / img_h,
        location=f'{y_band}_{x_band}')


def summarize_threshold(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor,
                        pred_scores: torch.Tensor, score_thr: float,
                        iou_thr: float) -> Dict[str, Any]:
    keep = pred_scores >= score_thr
    kept_boxes = pred_boxes[keep]
    kept_scores = pred_scores[keep]
    fp_idx, fn_idx, tp_pairs = greedy_match(
        gt_boxes, kept_boxes, kept_scores, iou_thr)
    tp = len(tp_pairs)
    fp = len(fp_idx)
    fn = len(fn_idx)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        score_thr=score_thr,
        num_kept=int(keep.sum().item()),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1)


def gt_diagnostics(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor,
                   pred_scores: torch.Tensor, matched_gt: Iterable[int],
                   iou_thr: float, score_thr: float,
                   near_iou_thr: float) -> List[Dict[str, Any]]:
    matched_gt = set(matched_gt)
    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]
    if num_gt == 0:
        return []

    ious = bbox_overlaps(pred_boxes, gt_boxes, mode='iou') if num_pred else torch.zeros(0, num_gt)
    records: List[Dict[str, Any]] = []
    for gt_idx in range(num_gt):
        if num_pred:
            gt_ious = ious[:, gt_idx]
            best_any_iou, best_any_pred = float(gt_ious.max()), int(gt_ious.argmax())
            best_any_score = float(pred_scores[best_any_pred])
            iou_hit = gt_ious >= iou_thr
            score_hit = pred_scores >= score_thr
            near_hit = gt_ious >= near_iou_thr
            best_score_iou_hit = float(pred_scores[iou_hit].max()) if bool(iou_hit.any()) else None
            best_iou_score_hit = float(gt_ious[score_hit].max()) if bool(score_hit.any()) else None
            best_score_near_hit = float(pred_scores[near_hit].max()) if bool(near_hit.any()) else None
        else:
            best_any_iou = 0.0
            best_any_pred = None
            best_any_score = None
            best_score_iou_hit = None
            best_iou_score_hit = None
            best_score_near_hit = None

        if gt_idx in matched_gt:
            status = 'tp'
        elif best_score_iou_hit is not None and best_score_iou_hit < score_thr:
            status = 'low_score_hit'
        elif best_iou_score_hit is not None and best_iou_score_hit >= iou_thr:
            status = 'assignment_or_duplicate_miss'
        elif best_iou_score_hit is not None and best_iou_score_hit >= near_iou_thr:
            status = 'localization_miss'
        elif best_score_near_hit is not None and best_score_near_hit < score_thr:
            status = 'low_score_localization_candidate'
        elif best_any_iou >= near_iou_thr:
            status = 'weak_near_candidate'
        else:
            status = 'no_near_candidate'

        records.append(dict(
            gt_index=gt_idx,
            status=status,
            best_any_pred_index=best_any_pred,
            best_any_iou=best_any_iou,
            best_any_score=best_any_score,
            best_score_iou_ge_thr=best_score_iou_hit,
            best_iou_score_ge_thr=best_iou_score_hit,
            best_score_iou_ge_near=best_score_near_hit))
    return records


def pred_diagnostics(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor,
                     pred_scores: torch.Tensor, tp_pred_indices: Iterable[int],
                     iou_thr: float, near_iou_thr: float) -> List[Dict[str, Any]]:
    tp_pred_indices = set(tp_pred_indices)
    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]
    if num_pred == 0:
        return []

    ious = bbox_overlaps(pred_boxes, gt_boxes, mode='iou') if num_gt else torch.zeros(num_pred, 0)
    records: List[Dict[str, Any]] = []
    for pred_idx in range(num_pred):
        if num_gt:
            row = ious[pred_idx]
            best_gt_iou, best_gt_idx = float(row.max()), int(row.argmax())
        else:
            best_gt_iou, best_gt_idx = 0.0, None

        if pred_idx in tp_pred_indices:
            status = 'tp'
        elif num_gt == 0:
            status = 'negative_image_fp'
        elif best_gt_iou >= iou_thr:
            status = 'duplicate_or_assignment_fp'
        elif best_gt_iou >= near_iou_thr:
            status = 'localization_or_extent_fp'
        else:
            status = 'background_fp'

        records.append(dict(
            pred_index=pred_idx,
            status=status,
            score=float(pred_scores[pred_idx]),
            best_gt_index=best_gt_idx,
            best_gt_iou=best_gt_iou))
    return records


def add_counter(counter: Counter, key: str, amount: int = 1) -> None:
    counter[key] += amount


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline validation inference and collect detailed diagnostics.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--primary-score-thr', type=float, default=0.3)
    parser.add_argument('--near-iou-thr', type=float, default=0.1)
    parser.add_argument(
        '--score-thrs',
        default='0.01,0.05,0.1,0.2,0.3,0.5,0.7',
        help='Comma-separated thresholds for aggregate precision/recall curves.')
    parser.add_argument(
        '--store-pred-score-thr',
        type=float,
        default=0.0,
        help='Only predictions at or above this score are stored in JSONL. '
        'All model-returned predictions are still used for GT diagnostics.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Optional cap for smoke tests. By default the full val set is used.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, same format as other MMDetection tools.')
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
    per_image_path = osp.join(args.out_dir, 'per_image_diagnostics.jsonl')
    summary_path = osp.join(args.out_dir, 'diagnostics_summary.json')

    score_thrs = _parse_float_list(args.score_thrs)
    if args.primary_score_thr not in score_thrs:
        score_thrs = sorted(score_thrs + [args.primary_score_thr])

    loader = Runner.build_dataloader(cfg.val_dataloader)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    summary: Dict[str, Any] = dict(
        config=osp.abspath(args.config),
        checkpoint=osp.abspath(args.checkpoint),
        iou_thr=args.iou_thr,
        primary_score_thr=args.primary_score_thr,
        near_iou_thr=args.near_iou_thr,
        store_pred_score_thr=args.store_pred_score_thr,
        total_images=0,
        total_gt=0,
        total_predictions_returned=0,
        total_predictions_stored=0,
        thresholds={str(thr): dict(tp=0, fp=0, fn=0, num_kept=0)
                    for thr in score_thrs},
        gt_status_counts=Counter(),
        pred_status_counts=Counter(),
        gt_size_status_counts=defaultdict(Counter),
        gt_location_status_counts=defaultdict(Counter),
        image_failure_counts=Counter())

    hard_cases: List[Dict[str, Any]] = []

    with open(per_image_path, 'w', encoding='utf-8') as jsonl:
        stop = False
        for data_batch in tqdm(loader, desc='diagnostic infer'):
            with torch.no_grad():
                outputs = model.test_step(data_batch)
            if not isinstance(outputs, list):
                outputs = [outputs]

            for out in outputs:
                if args.max_images is not None and summary['total_images'] >= args.max_images:
                    stop = True
                    break
                gt = out.gt_instances
                pred = out.pred_instances
                gt_boxes = _boxes_xyxy(gt)
                pred_boxes = _boxes_xyxy(pred)
                pred_scores = _scores(pred)
                pred_labels = _labels(pred)
                gt_labels = _labels(gt)

                img_path = out.img_path[0] if isinstance(out.img_path, list) else out.img_path
                img_id = _scalar_int(out.img_id)
                img_h, img_w = out.img_shape
                ori_h, ori_w = out.ori_shape

                summary['total_images'] += 1
                summary['total_gt'] += int(gt_boxes.shape[0])
                summary['total_predictions_returned'] += int(pred_boxes.shape[0])

                threshold_stats = {}
                primary_stats = None
                primary_tp_pairs: List[Tuple[int, int]] = []
                primary_fp_idx: List[int] = []
                primary_fn_idx: List[int] = []
                for thr in score_thrs:
                    stats = summarize_threshold(
                        gt_boxes, pred_boxes, pred_scores, thr, args.iou_thr)
                    threshold_stats[str(thr)] = stats
                    for key in ('tp', 'fp', 'fn', 'num_kept'):
                        summary['thresholds'][str(thr)][key] += stats[key]
                    if abs(thr - args.primary_score_thr) < 1e-12:
                        keep = pred_scores >= thr
                        primary_fp_idx, primary_fn_idx, primary_tp_pairs = greedy_match(
                            gt_boxes, pred_boxes[keep], pred_scores[keep], args.iou_thr)
                        kept_original = torch.nonzero(keep, as_tuple=False).squeeze(1).tolist()
                        primary_tp_pairs = [
                            (kept_original[pred_idx], gt_idx)
                            for pred_idx, gt_idx in primary_tp_pairs
                        ]
                        primary_fp_idx = [kept_original[pred_idx] for pred_idx in primary_fp_idx]
                        primary_stats = stats

                matched_gt = [gt_idx for _, gt_idx in primary_tp_pairs]
                tp_pred_indices = [pred_idx for pred_idx, _ in primary_tp_pairs]
                gt_records = gt_diagnostics(
                    gt_boxes, pred_boxes, pred_scores, matched_gt,
                    args.iou_thr, args.primary_score_thr, args.near_iou_thr)
                pred_records_all = pred_diagnostics(
                    gt_boxes, pred_boxes, pred_scores, tp_pred_indices,
                    args.iou_thr, args.near_iou_thr)

                pred_keep = pred_scores >= args.store_pred_score_thr
                stored_pred_indices = torch.nonzero(
                    pred_keep, as_tuple=False).squeeze(1).tolist()
                summary['total_predictions_stored'] += len(stored_pred_indices)

                gt_entries = []
                for gt_idx, box in enumerate(gt_boxes.tolist()):
                    meta = box_metadata(box, ori_w, ori_h)
                    status = gt_records[gt_idx]['status']
                    add_counter(summary['gt_status_counts'], status)
                    add_counter(summary['gt_size_status_counts'][status], _size_bucket(meta['area']))
                    add_counter(summary['gt_location_status_counts'][status], meta['location'])
                    gt_entries.append(dict(
                        gt_index=gt_idx,
                        label=gt_labels[gt_idx] if gt_idx < len(gt_labels) else None,
                        bbox_xyxy=box,
                        meta=meta,
                        diagnostic=gt_records[gt_idx]))

                pred_entries = []
                for pred_idx in stored_pred_indices:
                    box = pred_boxes[pred_idx].tolist()
                    record = pred_records_all[pred_idx]
                    add_counter(summary['pred_status_counts'], record['status'])
                    pred_entries.append(dict(
                        pred_index=pred_idx,
                        label=pred_labels[pred_idx] if pred_idx < len(pred_labels) else None,
                        score=float(pred_scores[pred_idx]),
                        bbox_xyxy=box,
                        meta=box_metadata(box, ori_w, ori_h),
                        diagnostic=record))

                assert primary_stats is not None
                failure_type = 'clean'
                if primary_stats['fp'] and primary_stats['fn']:
                    failure_type = 'both_fp_fn'
                elif primary_stats['fp']:
                    failure_type = 'fp_only'
                elif primary_stats['fn']:
                    failure_type = 'fn_only'
                add_counter(summary['image_failure_counts'], failure_type)

                image_record = dict(
                    img_id=img_id,
                    img_path=img_path,
                    img_shape=[int(img_h), int(img_w)],
                    ori_shape=[int(ori_h), int(ori_w)],
                    num_gt=int(gt_boxes.shape[0]),
                    num_predictions_returned=int(pred_boxes.shape[0]),
                    num_predictions_stored=len(stored_pred_indices),
                    primary=primary_stats,
                    primary_tp_pairs_pred_gt=primary_tp_pairs,
                    primary_fp_pred_indices=primary_fp_idx,
                    primary_fn_gt_indices=primary_fn_idx,
                    threshold_stats=threshold_stats,
                    gt=gt_entries,
                    predictions=pred_entries)
                jsonl.write(json.dumps(_json_safe(image_record), ensure_ascii=False) + '\n')

                if failure_type != 'clean':
                    hard_cases.append(dict(
                        img_id=img_id,
                        img_path=img_path,
                        failure_type=failure_type,
                        num_gt=int(gt_boxes.shape[0]),
                        tp=primary_stats['tp'],
                        fp=primary_stats['fp'],
                        fn=primary_stats['fn'],
                        gt_status_counts=Counter(r['diagnostic']['status'] for r in gt_entries)))
            if stop:
                break

    # Finalize threshold metrics and JSON-serializable counters.
    for thr, stats in summary['thresholds'].items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats.update(precision=precision, recall=recall, f1=f1)

    hard_cases.sort(key=lambda x: (x['fn'], x['fp'], x['num_gt']), reverse=True)
    summary['hard_cases_top50'] = hard_cases[:50]
    summary['per_image_diagnostics'] = osp.abspath(per_image_path)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    print(f'Done. Summary: {summary_path}')
    print(f'Per-image diagnostics: {per_image_path}')


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
