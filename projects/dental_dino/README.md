# Dental DINO（MMDetection）

在 MMDetection 的 **DINO** 上延续 Plan.md 中的牙科全景片病灶检测与 **牙齿结构先验**：在 **neck 输出多尺度特征** 上做与 RT-DETRv3 思路一致的 **Mask-Guided Feature Modulation**（先验编码 + gate/bias 调制），**不把 mask 与 RGB 简单 concat 到输入**。

## 模块说明

- `projects/dental_dino/dental_dino/models/detectors/dino_tooth_prior.py`：`DINOToothPrior`，在 backbone+neck 之后、encoder 之前插入调制。
- `projects/dental_dino/dental_dino/models/utils/tooth_prior_modules.py`：`ToothPriorEncoder`、`MaskGuidedFeatureModulation`。
- 牙齿语义图沿用 **CocoDataset 的 `data_prefix['seg']` + `LoadAnnotations(with_seg=True)`**，读入为 `gt_seg_map` / `gt_sem_seg`，与图像共用 RandomFlip/Resize/Crop，**不做静默兜底**；开启先验时若缺失则报错。

## 配置（对应 Plan.md 实验 A–F，框架为 DINO）

| 实验 | 配置 |
|------|------|
| A baseline | `configs/dino-4scale_r50_8xb2-12e_perio_baseline.py` |
| B RGB+mask concat | `configs/dino-4scale_r50_8xb2-12e_perio_concat_mask.py`（`ConcatSemSegToImage` + `DetDataPreprocessor4Ch` + `in_channels=4`） |
| C 仅 Prior Encoder | `configs/dino-4scale_r50_8xb2-12e_perio_prior_encoder_only.py`（调制关、`loss_tooth_prior_energy`） |
| D Encoder + 调制 | `configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py` |
| E + 牙齿引导 query | `configs/dino-4scale_r50_8xb2-12e_perio_toothprior_queries.py` |
| F + heatmap / NWD | `configs/dino-4scale_r50_8xb2-12e_perio_toothprior_aux.py` |

数据集根目录默认：`/hdd1/zyh/Dental/mutil_repo/PerioXrays_Dataset/`（含 `train2017`、`SemanticsMask_train` 等）。

## 训练 / 测试 / 可视化

在 MMDetection 仓库根目录执行（`PYTHONPATH` 需包含仓库根，脚本已设置）：

```bash
CUDA_VISIBLE_DEVICES=0 bash projects/dental_dino/scripts/run_train.sh \
  projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py

CUDA_VISIBLE_DEVICES=0 bash projects/dental_dino/scripts/run_test.sh \
  projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py \
  work_dirs/.../epoch_12.pth

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python projects/dental_dino/scripts/image_demo_dental.py \
  /path/to/img.jpg \
  projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py \
  --weights work_dirs/.../epoch_12.pth --out-dir ./vis
```

## 与 RT-DETRv3（Paddle）分支的关系

Paddle 版 RT-DETRv3 实现保留在 `RT-DETRv3/` 子仓库；本目录为 **PyTorch/MMDet** 主线，便于与现有 mmcv/mmengine 生态对齐。
