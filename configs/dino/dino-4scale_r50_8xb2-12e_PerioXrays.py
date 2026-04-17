_base_ = [
    '../_base_/datasets/PerioXrays.py',
    '../_base_/models/dino.py',
    '../_base_/default_runtime.py'
]

# ====== 基础信息 ======
# 这里按牙科常见单类先写成1；如果你是多类，请改成实际类别数
num_classes = 1

# ====== 只覆盖必须项：类别数 ======
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained_checkpoints/resnet50-0676ba61.pth')),
    bbox_head=dict(
        num_classes=num_classes
    )
)

# ====== 训练策略（在原始DINO 12e基础上做更稳健的牙科场景设置） ======
# 理由：
# 1) 医学/牙科小目标场景往往收敛慢于通用场景，12e常偏短
# 2) 你不希望复杂pipeline，那么适当延长epoch提升泛化
max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 保留DINO默认AdamW思路，仅做温和优化
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,          # 对8xb2基线是稳定起点
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)  # 小数据医学影像更稳
        }
    )
)

# 继承原始step策略并对齐24e
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1
    )
]

# 自动学习率缩放：dino-4scale_r50_8xb2 的基线总batch一般按16设计
auto_scale_lr = dict(base_batch_size=16)