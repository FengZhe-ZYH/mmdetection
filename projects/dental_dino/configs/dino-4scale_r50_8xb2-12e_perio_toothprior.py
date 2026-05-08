# DINO + 牙齿语义先验（neck 特征 Mask-Guided Modulation，对齐 RT-DETRv3 思路）
# 依赖 data_prefix['seg'] 与标注同名的 PNG 语义图（与 COCO seg_map 流程一致）
# 训练: CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#   projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior.py

_base_ = ['../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py']

custom_imports = dict(
    imports=['projects.dental_dino.dental_dino'],
    allow_failed_imports=False,
)

data_root = '/hdd1/zyh/Dental/mutil_repo/PerioXrays_Dataset/'
metainfo = dict(
    classes=('Apical Periodontitis', ),
    palette=[(220, 20, 60)],
)

model = dict(
    type='DINOToothPrior',
    use_tooth_prior_encoder=True,
    use_mask_guided_modulation=True,
    tooth_prior_channels=64,
    gate_log_interval=50,
    bbox_head=dict(num_classes=1),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        pad_seg=True,
        seg_pad_value=0,
    ),
)

# 与图像同步几何变换；gt_seg_map 会随 Flip / Resize / Crop 更新
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
        ]),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017', seg='SemanticsMask_train'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017', seg='SemanticsMask_val'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
test_evaluator = val_evaluator
