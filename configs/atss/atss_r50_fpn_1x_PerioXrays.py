_base_ = [
    '../_base_/models/atss_r50_fpn.py',
    '../_base_/datasets/PerioXrays.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# ===== 模型设置：尽量对齐 Faster R-CNN 的 PerioXrays 配置 =====
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./pretrained_checkpoints/resnet50-0676ba61.pth')),
    bbox_head=dict(num_classes=1),
)

# ===== 训练/验证/测试 loop：对齐 max_epochs 与 val_interval =====
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===== 优化器：与 Faster R-CNN PerioXrays 保持一致 =====
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4))

# ===== 学习率策略：保留 Linear warmup，替换为 StepLR（与 Faster R-CNN 配置一致）=====
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='StepLR', begin=0, end=15, by_epoch=True, step_size=3, gamma=0.33),
]

# ===== 运行时配置（现代版）：对齐日志/保存策略 =====
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'),
)

# ===== 日志配置 =====
log_level = 'INFO'

# ===== 环境设置 =====
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ===== 可视化 =====
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

