# Copyright (c) OpenMMLab. All rights reserved.
"""验证集推理 + 基于 IoU 的漏检/误检分析，导出可视化与 JSON。"""
import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import mmcv
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import register_all_modules


def _patch_torch_load_for_mmengine_checkpoints() -> None:
    """PyTorch>=2.6 默认 weights_only=True，mmengine 保存的 ckpt 含 HistoryBuffer 等对象会失败。"""
    _orig = torch.load

    def _wrapped(*args, **kwargs):
        try:
            kwargs.setdefault('weights_only', False)
            return _orig(*args, **kwargs)
        except TypeError:
            kwargs.pop('weights_only', None)
            return _orig(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[assignment]


def _boxes_xyxy(inst) -> torch.Tensor:
    """InstanceData -> float32 CPU tensor (N, 4) xyxy."""
    if inst is None or len(inst) == 0:
        return torch.zeros(0, 4)
    b = inst.bboxes
    if hasattr(b, 'tensor'):
        t = b.tensor
    elif isinstance(b, torch.Tensor):
        t = b
    else:
        t = torch.as_tensor(np.asarray(b))
    return t.detach().float().cpu()


def greedy_match(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_thr: float,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """按分数从高到低贪心匹配，IoU>=阈值一对一。

    Returns:
        fp_pred_idx: 未匹配到任何 GT 的预测下标
        fn_gt_idx: 未被任何预测匹配的 GT 下标
        tp_pairs: (pred_idx, gt_idx)
    """
    g, p = gt_boxes.shape[0], pred_boxes.shape[0]
    if g == 0 and p == 0:
        return [], [], []
    if g == 0:
        return list(range(p)), [], []
    if p == 0:
        return [], list(range(g)), []

    ious = bbox_overlaps(pred_boxes, gt_boxes, mode='iou')  # (P, G)
    order = torch.argsort(pred_scores, descending=True).tolist()
    matched_gt: set = set()
    tp_pairs: List[Tuple[int, int]] = []
    fp_pred_idx: List[int] = []

    for pi in order:
        row = ious[pi]
        best_iou, best_g = float(row.max()), int(row.argmax())
        if best_iou >= iou_thr and best_g not in matched_gt:
            matched_gt.add(best_g)
            tp_pairs.append((pi, best_g))
        else:
            fp_pred_idx.append(pi)

    fn_gt_idx = [gi for gi in range(g) if gi not in matched_gt]
    return fp_pred_idx, fn_gt_idx, tp_pairs


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


def parse_args():
    p = argparse.ArgumentParser(
        description='验证集推理并按 IoU 阈值统计失败样例（漏检 FN / 误检 FP）')
    p.add_argument('config', help='配置文件路径（可用训练 dump 的 vis_data/config.py）')
    p.add_argument('checkpoint', help='checkpoint 路径，例如 epoch_24.pth')
    p.add_argument(
        '--out-dir',
        required=True,
        help='输出根目录，将创建 all_vis/ failure_vis/ failure_cases.json')
    p.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='与 GT 判定为 TP 的 IoU 阈值，默认 0.5')
    p.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='参与匹配的预测置信度下限（与可视化一致）')
    p.add_argument(
        '--device', default='cuda:0', help='推理设备，默认 cuda:0')
    p.add_argument(
        '--save-all-vis',
        action='store_true',
        help='若为真，为验证集每一张保存 GT|Pred 拼图；默认仅保存失败样例图')
    p.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置项')
    return p.parse_args()


def main():
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
    all_vis_dir = osp.join(args.out_dir, 'all_vis')
    fail_vis_dir = osp.join(args.out_dir, 'failure_vis')
    os.makedirs(fail_vis_dir, exist_ok=True)
    if args.save_all_vis:
        os.makedirs(all_vis_dir, exist_ok=True)

    loader = Runner.build_dataloader(cfg.val_dataloader)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    summary = {'total': 0, 'failures': 0, 'iou_thr': args.iou_thr, 'score_thr': args.score_thr}
    failure_records: List[Dict[str, Any]] = []

    for data_batch in tqdm(loader, desc='val infer'):
        with torch.no_grad():
            outputs = model.test_step(data_batch)

        # test_step 返回与 batch 等长的 DetDataSample 列表
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            summary['total'] += 1
            gt = out.gt_instances
            pred = out.pred_instances

            scores = pred.scores
            keep = scores > args.score_thr
            pred_f = pred[keep]
            pred_boxes = _boxes_xyxy(pred_f)
            pred_scores = pred_f.scores.detach().float().cpu() if len(pred_f) else torch.zeros(0)
            gt_boxes = _boxes_xyxy(gt)

            fp_idx, fn_idx, tp_pairs = greedy_match(
                gt_boxes, pred_boxes, pred_scores, args.iou_thr)
            is_failure = (len(fp_idx) > 0) or (len(fn_idx) > 0)

            img_path = out.img_path
            if isinstance(img_path, list):
                img_path = img_path[0]

            stem = osp.splitext(osp.basename(img_path))[0]
            iid = _scalar_int(out.img_id)
            out_name = f'{iid:012d}_{stem}.jpg'

            img_bgr = mmcv.imread(img_path)
            img_rgb = mmcv.imconvert(img_bgr, 'bgr', 'rgb')

            if args.save_all_vis:
                vis_all = osp.join(all_vis_dir, out_name)
                visualizer.add_datasample(
                    name=stem,
                    image=img_rgb,
                    data_sample=out,
                    draw_gt=True,
                    draw_pred=True,
                    show=False,
                    out_file=vis_all,
                    pred_score_thr=args.score_thr,
                )

            if is_failure:
                summary['failures'] += 1
                vis_fail = osp.join(fail_vis_dir, out_name)
                visualizer.add_datasample(
                    name=stem,
                    image=img_rgb,
                    data_sample=out,
                    draw_gt=True,
                    draw_pred=True,
                    show=False,
                    out_file=vis_fail,
                    pred_score_thr=args.score_thr,
                )

                fp_boxes = pred_boxes[fp_idx].tolist() if fp_idx else []
                fn_boxes = gt_boxes[fn_idx].tolist() if fn_idx else []
                fp_scores = pred_scores[fp_idx].tolist() if fp_idx else []

                failure_records.append({
                    'img_path': img_path,
                    'img_id': iid,
                    'num_gt': int(gt_boxes.shape[0]),
                    'num_pred_matched': int(len(pred_f)),
                    'num_tp': len(tp_pairs),
                    'num_fp': len(fp_idx),
                    'num_fn': len(fn_idx),
                    'fp_pred_indices': fp_idx,
                    'fn_gt_indices': fn_idx,
                    'tp_pairs_pred_gt': tp_pairs,
                    'fp_bboxes_xyxy': fp_boxes,
                    'fp_scores': fp_scores,
                    'fn_bboxes_xyxy': fn_boxes,
                    'failure_vis': vis_fail,
                })

    out_json = osp.join(args.out_dir, 'failure_cases.json')
    payload = {
        'config': osp.abspath(args.config),
        'checkpoint': osp.abspath(args.checkpoint),
        'summary': summary,
        'failure_cases': failure_records,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)

    print(f'完成: 共 {summary["total"]} 张，失败 {summary["failures"]} 张 '
          f'(IoU>={args.iou_thr}, score>{args.score_thr})')
    print(f'JSON: {out_json}')
    print(f'失败可视化目录: {fail_vis_dir}')


if __name__ == '__main__':
    main()
