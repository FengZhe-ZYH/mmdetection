_base_ = ['../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py']

data_root = '/hdd1/zyh/Dental/mutil_repo/PerioXrays_Dataset/'

metainfo = dict(
    classes=('Apical Periodontitis',),
    palette=[(220, 20, 60)],
)

backend_args = None

model = dict(
    bbox_head=dict(num_classes=1)
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
test_evaluator = val_evaluator