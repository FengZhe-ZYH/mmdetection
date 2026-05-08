# Dental DINO（MMDetection）

在 MMDetection 的 **DINO** 上延续 Plan.md 中的牙科全景片病灶检测与 **牙齿结构先验**：在 **neck 输出多尺度特征** 上做与 RT-DETRv3 思路一致的 **Mask-Guided Feature Modulation**（先验编码 + gate/bias 调制），**不把 mask 与 RGB 简单 concat 到输入**。

## 模块说明

- `projects/dental_dino/dental_dino/models/detectors/dino_tooth_prior.py`：`DINOToothPrior`，在 backbone+neck 之后、encoder 之前插入调制。
- `projects/dental_dino/dental_dino/models/utils/tooth_prior_modules.py`：`ToothPriorEncoder`、`MaskGuidedFeatureModulation`。
- 牙齿语义图沿用 **CocoDataset 的 `data_prefix['seg']` + `LoadAnnotations(with_seg=True)`**，读入为 `gt_seg_map` / `gt_sem_seg`，与图像共用 RandomFlip/Resize/Crop，**不做静默兜底**；开启先验时若缺失则报错。

## 配置

| 配置 | 说明 |
|------|------|
| `configs/dino-4scale_r50_8xb2-12e_perio_baseline.py` | 单类 PerioXrays，标准 DINO |
| `configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py` | 同上 + 牙齿 PNG 目录 + `DINOToothPrior` |

数据集根目录默认：`/hdd1/zyh/Dental/mutil_repo/PerioXrays_Dataset/`（含 `train2017`、`SemanticsMask_train` 等）。

## 训练示例

在 MMDetection 仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py
```

## 与 RT-DETRv3（Paddle）分支的关系

Paddle 版 RT-DETRv3 实现保留在 `RT-DETRv3/` 子仓库；本目录为 **PyTorch/MMDet** 主线，便于与现有 mmcv/mmengine 生态对齐。
