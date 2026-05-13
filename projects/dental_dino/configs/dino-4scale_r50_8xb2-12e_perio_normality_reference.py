# DINO + tooth-interior normality reference prototype.
#
# This config starts from the clean medical-pipeline baseline and adds only:
# - tooth semantic mask loading for constructing normal/lesion regions
# - DINONormalityReference auxiliary contrast/anomaly losses
#
# Train:
#   conda run -n dinov3 python tools/train.py \
#     projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_normality_reference.py

_base_ = ['dino-4scale_r50_8xb2-12e_perio_singleclass_medical_pipeline.py']

custom_imports = dict(
    imports=['projects.dental_dino.dental_dino'],
    allow_failed_imports=False,
)

model = dict(
    type='DINONormalityReference',
    use_normality_reference=True,
    normality_feat_level=0,
    normality_proj_dim=128,
    normality_ema_momentum=0.05,
    lesion_expand_ratio=0.15,
    min_region_pixels=4,
    loss_normality_weight=0.2,
    loss_anomaly_weight=0.1,
    lesion_proto_margin=0.5,
    lesion_healthy_margin=0.2,
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

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
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
    dataset=dict(
        data_prefix=dict(img='train2017', seg='SemanticsMask_train'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='val2017', seg='SemanticsMask_val'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader
