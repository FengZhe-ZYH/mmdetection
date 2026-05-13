# DINO + tooth-interior normality reference with query-level injection.
#
# Compared with the prediction-coupling variant, this version injects the
# normality / anomaly signal into the decoder matching queries before DINO
# prediction heads run.  Prediction-level re-scoring stays disabled so the
# experiment isolates query-level coupling.
#
# Train:
#   conda run -n dinov3 python tools/train.py \
#     projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_normality_query_coupling.py

_base_ = ['dino-4scale_r50_8xb2-12e_perio_normality_reference.py']

model = dict(
    use_query_coupling=True,
    query_coupling_weight=0.10,
    query_anomaly_weight=0.50,
    query_proto_weight=0.20,
    use_prediction_coupling=False,
)
