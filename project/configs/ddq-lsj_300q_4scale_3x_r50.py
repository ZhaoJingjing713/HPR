_base_ = [
    './_base_/datasets/lsj_data_re_aug_coco_detection.py',
]

randomness=dict(seed=681328528)

batch_augments = [
    dict(type='BatchFixedSizePad', size=_base_.image_size, pad_mask=True)
]
model = dict(
    type='HPRDDQDETR',
    num_queries=300,  # num_matching_queries
    aux_weights=[0.5,0.5],
    # ratio of num_dense queries to num_queries
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_mask=True,
                batch_augments=batch_augments)),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='backbone_pth/backbone.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=0,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        bbox_roi_extractor = dict(
            type='SingleRoIExtractor',
            finest_scale=56,
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64]),
        layer_cfg=dict(
            merge_method='learnable_channel_aware',
            initial_weights=[1,1,1], 
            merge_dropout=0.,# only be used  when method='cross_attn'
            dy_conv_cfg=dict(
                in_channels=256,
                feat_channels=64,
                out_channels=256,
                input_feat_shape=7,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')),  # 0.1 for DeformDETR
            regional_ca_cfg=dict(
                sample_num=5,
                embed_dims=256,
                num_heads=8,
                use_key_pos=True,
                positional_encoding=dict(
                    num_feats=128,
                    normalize=True,
                    offset=0.0,  # -0.5 for DeformDETR
                    temperature=20),
                attn_drop=0.,
                proj_drop=0.,
                dropout_layer=dict(type='Dropout', drop_prob=0.),
                init_cfg=None,
                batch_first=True,
                norm_cfg=dict(type='LN'),
                act_cfg = dict(type='ReLU', inplace=True),),  # 0.1 for DeformDETR
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),

    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='HPRDDQDETRHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

train_dataloader = dict(
    batch_size=1) #  (16 GPUs) x (1 samples per GPU)


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# learning policy
max_iters = 88000*3   # 88000/(118287-coco train set/(16-bs))≈12ep
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,    
    val_interval=8000)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=False)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[205300, 249300],
        gamma=0.1)
]
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=8000))

vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
