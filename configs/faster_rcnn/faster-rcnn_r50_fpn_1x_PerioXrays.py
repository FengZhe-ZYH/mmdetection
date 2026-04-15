_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',  # 基本模型配置
    '../_base_/datasets/PerioXrays.py',         # 数据集配置，指向你自己的 PerioXrays.py
    '../_base_/schedules/schedule_1x.py',       # 学习率调度策略
    '../_base_/default_runtime.py'             # 默认运行时配置
]

# 模型设置
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained_checkpoints/resnet50-0676ba61.pth')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')           
# 优化器和学习率配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4))

# 保留 _base_ 中的 LinearLR（warmup），替换 MultiStepLR
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='StepLR', begin=0, end=15, by_epoch=True, step_size=3, gamma=0.33)
]

# ===== 运行时配置（现代版）=====
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),          # 每 50 步输出日志
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                                        # 每 1 个 epoch 保存
        max_keep_ckpts=3,                                  # 最多保留 3 个检查点
        save_best='auto'))                                 # 自动保存最佳模型

# ===== 日志配置 =====
log_level = 'INFO'

# ===== 环境设置 =====
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# ===== 可视化 =====
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')