#!/usr/bin/env python
"""单张或目录图像推理与可视化（基于 mmdet DetInferencer）。

在 mmdetection 仓库根目录执行，需已设置 PYTHONPATH=.

示例:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python projects/dental_dino/scripts/image_demo_dental.py \\
    /path/to/pano.jpg \\
    projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py \\
    --weights epoch_12.pth \\
    --out-dir ./vis_out \\
    --pred-score-thr 0.3
"""
from argparse import ArgumentParser

from mmdet.apis import DetInferencer


def parse_args():
    p = ArgumentParser(description='Dental DINO / MMDet image demo')
    p.add_argument('inputs', help='Image path or directory')
    p.add_argument('config', help='Config path')
    p.add_argument('--weights', required=True, help='Checkpoint .pth')
    p.add_argument('--out-dir', default='./dental_vis')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--pred-score-thr', type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    inferencer = DetInferencer(
        model=args.config,
        weights=args.weights,
        device=args.device,
        scope='mmdet',
    )
    inferencer(
        inputs=args.inputs,
        out_dir=args.out_dir,
        no_save_vis=False,
        pred_score_thr=args.pred_score_thr,
    )


if __name__ == '__main__':
    main()
