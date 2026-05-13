# SA-DINO v2: Semantic-Aware DINO with enhanced query initialization and
# optimized training strategy for periapical periodontitis detection.
#
# Improvements over SA-DINO v1:
# - Tooth-guided query initialization (32 queries from mask centroids+offset)
# - Extended training: 24 epochs with MultiStepLR at [20, 23]
# - EMA model averaging
# - 500-iter linear warm-up
#
# Training:
#   CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#     projects/dental_dino/configs/dino-4scale_r50_8xb2-24e_perio_sadino_v2.py

_base_ = ['../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py']

custom_imports = dict(
    imports=['projects.dental_dino.dental_dino'],
    allow_failed_imports=False,
)

data_root = '/hdd1/zyh/Datasets/CariesXrays/coco_official/'
metainfo = dict(
    classes=('Decay', ),
    palette=[(220, 20, 60)],
)

backend_args = None

model = dict(
    type='SADINO',
    bbox_head=dict(num_classes=1),
    # tooth embedding encoder
    num_tooth_classes=34,
    tooth_embed_dim=32,
    tooth_prior_channels=64,
    # mask-guided feature modulation
    use_mask_modulation=True,
    modulation_strength=1.0,
    gate_init_bias=-2.0,
    # semantic cross-attention after encoder
    use_semantic_cross_attn=True,
    num_semantic_cross_attn_layers=3,
    semantic_cross_attn_heads=8,
    # tooth-guided query initialization
    use_tooth_guided_queries=True,
    max_tooth_queries=32,
    # auxiliary segmentation loss
    use_aux_seg=True,
    aux_seg_weight=0.4,
    aux_seg_mid_channels=128,
    aux_seg_upsample=2,
    # logging
    gate_log_interval=500,
    # data preprocessor: pad seg map for alignment
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

# Training pipeline: medical-friendly (no ColorJitter — X-ray is pseudo-3ch grayscale)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
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
    dict(type='LoadImageFromFile', backend_args=backend_args),
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

# --- Training Strategy ---

# Extended training: 24 epochs
max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# LR schedule: MultiStepLR with 0.1x decay at epoch 20 and 23
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1),
]

# Optimizer: same base, with slightly higher lr for new modules
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'tooth_encoder': dict(lr_mult=1.0),
            'tooth_query_init': dict(lr_mult=1.0),
            'mask_modulation': dict(lr_mult=1.0),
            'semantic_cross_attn_layers': dict(lr_mult=1.0),
            'aux_seg_head': dict(lr_mult=1.0),
        }),
)

# EMA
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]

# Checkpointing: save best and every 4 epochs
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        save_best='coco/bbox_mAP',
        rule='greater',
    ),
)

auto_scale_lr = dict(enable=False, base_batch_size=2)
