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

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    )
)