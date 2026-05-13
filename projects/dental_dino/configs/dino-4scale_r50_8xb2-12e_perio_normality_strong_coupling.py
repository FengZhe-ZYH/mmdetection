# DINO + tooth-interior normality reference with prediction coupling.
#
# This keeps the normality-reference training losses from the prototype config,
# and additionally uses the learned anomaly / healthy-prototype signal during
# inference to re-score final DINO predictions.
#
# Train:
#   conda run -n dinov3 python tools/train.py \
#     projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_normality_strong_coupling.py

_base_ = ['dino-4scale_r50_8xb2-12e_perio_normality_reference.py']

model = dict(
    use_prediction_coupling=True,
    anomaly_score_weight=0.35,
    proto_score_weight=0.15,
    prediction_score_eps=1e-4,
)
