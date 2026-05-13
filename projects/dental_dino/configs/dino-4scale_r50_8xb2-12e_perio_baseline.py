# DINO 4-scale R50 — PerioXrays 单类病灶检测（baseline，不使用牙齿语义）
# 训练: CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#   projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_baseline.py

_base_ = ['../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py']

data_root = '/hdd1/zyh/Dental/mutil_repo/PerioXrays_Dataset/'
metainfo = dict(
    classes=('Apical Periodontitis', ),
    palette=[(220, 20, 60)],
)

model = dict(bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017'),
    ))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017'),
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

auto_scale_lr = dict(enable=False, base_batch_size=2)


