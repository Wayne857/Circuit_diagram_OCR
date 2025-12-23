_base_ = [
    '../_base_/models/segman.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# 数据集配置
dataset_type = 'CustomDataset'
data_root = 'C:/Users/11/Desktop/pj/image_extract/segman_dataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline,
        classes=('resistor', 'motor', 'ground', 'line', 'arrow', 'line_connector', 
                'chip', 'capacitor', 'zener_diode', 'mov', 'fuse', 'inductor'),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline,
        classes=('resistor', 'motor', 'ground', 'line', 'arrow', 'line_connector', 
                'chip', 'capacitor', 'zener_diode', 'mov', 'fuse', 'inductor'),
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline,
        classes=('resistor', 'motor', 'ground', 'line', 'arrow', 'line_connector', 
                'chip', 'capacitor', 'zener_diode', 'mov', 'fuse', 'inductor'),
    ))

# 模型设置
model = dict(
    decode_head=dict(
        num_classes=12,  # 您的数据集有12个类别
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0)
    ),
    # 训练和测试设置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器设置
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00006, 
    betas=(0.9, 0.999), 
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

lr_config = dict(
    _delete_=True, 
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False
)

evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')