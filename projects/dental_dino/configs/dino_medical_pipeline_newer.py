# DINO 单类根尖周炎检测 — 医学影像友好 pipeline
#
# 相对默认 DINO COCO pipeline 的调整：
# - 仅水平翻转（全景/根尖片通常不做垂直翻转）
# - 去掉 RandomCrop 分支，避免过度裁掉小块根尖周病灶
# - 保留多尺度 RandomChoiceResize，与 DINO 原设定一致
# - train 保留 filter_empty_gt=False，便于含大量阴性样本的筛查数据
#
# 数据：COCO 检测格式，目录示例
#   data/
#     train2017/   images
#     val2017/
#     annotations/instances_train2017.json
#                instances_val2017.json
#
# 训练（在仓库根目录）:
#   CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#     projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline.py

custom_imports = dict(
    imports=['projects.dental_dino.dental_dino.datasets.transforms'],
    allow_failed_imports=False)

_base_ = ['../../../configs/dino/dino-4scale_r50_8xb2-24e_coco.py']

# 按你的实际数据根目录修改（可为绝对路径）
data_root = '/hdd1/zyh/Datasets/CariesXrays/coco_official/'
metainfo = dict(
    classes=('Decay', ),
    palette=[(220, 20, 60)],
)

backend_args = None

model = dict(bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8)),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
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
