_base_ = [
    '../_base_/datasets/PerioXrays.py',
    '../_base_/models/yolox_s.py',
    '../_base_/default_runtime.py',
    './yolox_tta.py'
]

# ====== 基础信息 ======
num_classes = 1  # 如果多类请改实际值

# ====== 只覆盖必须项：类别数 ======
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192, num_classes=num_classes)
)

# ====== 针对牙科影像做训练时长与增强策略调整 ======
# yolox_s_300e 对你这类任务通常过长；同时你说不需要复杂train pipeline，
# 所以这里把训练缩到100e并关闭后期强增强切换影响（由dataset base控制简单pipeline）
max_epochs = 100
num_last_epochs = 15
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)]
)

# 保留YOLOX默认SGD体系（通常优于随意改AdamW）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
)

# 对齐100e重设调度：warmup + cosine + 最后常量阶段
param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=False,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0005,
        begin=5,
        T_max=max_epochs - 5 - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1.0,
        begin=max_epochs - num_last_epochs,
        end=max_epochs
    ),
]

# YOLOX后期模式切换与EMA，沿用官方机制
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=num_last_epochs, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49
    )
]

# yolox_s_8xb8 的默认总batch通常对应64
auto_scale_lr = dict(base_batch_size=64)