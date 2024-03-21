_base_ = [
    './_base_/datasets/data_re_aug_coco_detection.py', 
    './_base_/default_runtime.py'
]
randomness=dict(seed=681328528)

model = dict(
    type='HPRDETR',
    num_queries=900,  # num_matching_queries
    aux_weights=[0.5,0.5],
    with_box_refine=True,
    as_two_stage=True,
    use_dn=False,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.335, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=1)),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
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
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0), 
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  
                ffn_drop=0.0))), 
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
            merge_dropout=0.,
            dy_conv_cfg=dict(
                in_channels=256,
                feat_channels=64,
                out_channels=256,
                input_feat_shape=7,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')),
            regional_ca_cfg=dict(
                sample_num=5,
                embed_dims=256,
                num_heads=8,
                use_key_pos=True,
                positional_encoding=dict(
                    num_feats=128,
                    normalize=True,
                    offset=0.0,
                    temperature=20),
                attn_drop=0.,
                proj_drop=0.,
                dropout_layer=dict(type='Dropout', drop_prob=0.),
                init_cfg=None,
                batch_first=True,
                norm_cfg=dict(type='LN'),
                act_cfg = dict(type='ReLU', inplace=True),),  
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0), 
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0), 
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048, 
                ffn_drop=0.0)), 
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='HPRDETRHead',
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
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001, 
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
) 

max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16,22],
        gamma=0.1)
]

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