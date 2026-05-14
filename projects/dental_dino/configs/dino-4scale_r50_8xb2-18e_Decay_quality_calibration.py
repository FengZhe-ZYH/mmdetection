_base_ = './dino-4scale_r50_8xb2-18e_Decay_baseline.py'

custom_imports = dict(
    imports=['projects.dental_dino.dental_dino'],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='DentalDINOHead',
        enable_quality_branch=True,
        quality_loss_weight=0.5,
        quality_lambda=0.5,
        quality_neg_weight=0.25))
