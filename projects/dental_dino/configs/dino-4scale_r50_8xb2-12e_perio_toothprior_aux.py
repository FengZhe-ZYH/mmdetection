# Plan.md 实验 F：牙齿先验调制 + heatmap 辅助损失 + NWD（小框）
# 训练: CUDA_VISIBLE_DEVICES=0 bash projects/dental_dino/scripts/run_train.sh \
#   projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior_aux.py

_base_ = [
    'dino-4scale_r50_8xb2-12e_perio_toothprior.py',
]

model = dict(
    use_heatmap_aux_loss=True,
    heatmap_loss_weight=1.0,
    use_nwd_loss=True,
    nwd_loss_weight=1.0,
    nwd_small_area_thr=0.001,
)
