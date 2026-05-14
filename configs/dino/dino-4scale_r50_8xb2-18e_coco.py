_base_ = './dino-4scale_r50_8xb2-12e_coco.py'
max_epochs = 18
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]
