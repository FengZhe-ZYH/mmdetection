# Plan.md 实验 C：Tooth Prior Encoder 开启，Mask-Guided Modulation 关闭（仅 aux 能量正则）
# 训练: CUDA_VISIBLE_DEVICES=0 bash projects/dental_dino/scripts/run_train.sh \
#   projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_prior_encoder_only.py

_base_ = [
    'dino-4scale_r50_8xb2-12e_perio_toothprior.py',
]

model = dict(
    use_mask_guided_modulation=False,
    aux_prior_energy_weight=1e-6,
)
