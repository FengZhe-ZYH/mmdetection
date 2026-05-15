_base_ = './dino-4scale_r50_8xb2-18e_Decay_baseline.py'

custom_imports = dict(
    imports=['projects.dental_dino.dental_dino'],
    allow_failed_imports=False)

model = dict(
    type='DINOEvidence',
    bbox_head=dict(
        type='DentalDINOHead',
        enable_evidence_branch=True,
        lambda_evidence=0.15,
        evidence_loss_weight=0.25,
        evidence_neg_weight=0.1,
        evidence_rank_loss_weight=0.05,
        evidence_pos_iou_thr=0.5,
        evidence_hard_neg_score_thr=0.3,
        evidence_hard_neg_topk=50,
        evidence_rank_neg_topk=32,
        evidence_rank_margin=0.2,
        evidence_expand_ratio=0.5,
        evidence_roi_output_size=7,
        evidence_roi_featmap_strides=(8, 16, 32, 64)))
