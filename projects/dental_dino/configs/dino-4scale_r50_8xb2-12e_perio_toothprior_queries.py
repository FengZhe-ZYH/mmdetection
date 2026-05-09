# Plan.md 实验 E：在实验 D 基础上启用牙齿引导 query（pre_decoder 混合 top-k 与牙位参考）
# 训练: CUDA_VISIBLE_DEVICES=0 bash projects/dental_dino/scripts/run_train.sh \
#   projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior_queries.py

_base_ = [
    'dino-4scale_r50_8xb2-12e_perio_toothprior.py',
]

model = dict(
    use_tooth_guided_queries=True,
    queries_per_tooth=2,
    num_tooth_slots=32,
)
