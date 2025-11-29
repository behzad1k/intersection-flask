# YOLOv8-nano 640x384 Config for Inference

default_scope = 'mmyolo'

# Image size
img_scale = (448, 256)  # width x height
pad_val = 114

# YOLOv8-nano parameters
deepen_factor = 0.33
widen_factor = 0.25

# 9 classes (8 vehicles + mask)
num_classes = 9

metainfo = dict(
  classes=('bicycle', 'motorcycle', 'car', 'transporter', 'bus', 'truck', 'trailer', 'unknown', 'mask')
)

last_stage_out_channels = 768

strides = [8, 16, 32]

# Model definition
model = dict(
  type='YOLODetector',
  data_preprocessor=dict(
    type='YOLOv5DetDataPreprocessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True),

  # YOLOv8 CSPDarknet backbone (nano size)
  backbone=dict(
    type='YOLOv8CSPDarknet',
    arch='P5',
    last_stage_out_channels=last_stage_out_channels,
    deepen_factor=deepen_factor,
    widen_factor=widen_factor,
    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    act_cfg=dict(type='SiLU', inplace=True)),

  # YOLO neck
  neck=dict(
    type='YOLOv8PAFPN',
    deepen_factor=deepen_factor,
    widen_factor=widen_factor,
    in_channels=[256, 512, last_stage_out_channels],
    out_channels=[256, 512, last_stage_out_channels],
    num_csp_blocks=3,
    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    act_cfg=dict(type='SiLU', inplace=True)),

  # Detection head
  bbox_head=dict(
    type='YOLOv8Head',
    head_module=dict(
      type='YOLOv8HeadModule',
      num_classes=num_classes,
      in_channels=[256, 512, last_stage_out_channels],
      widen_factor=widen_factor,
      reg_max=16,
      norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
      act_cfg=dict(type='SiLU', inplace=True),
      featmap_strides=strides),
    prior_generator=dict(
      type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
    bbox_coder=dict(type='DistancePointBBoxCoder'),
    loss_cls=dict(
      type='mmdet.CrossEntropyLoss',
      use_sigmoid=True,
      reduction='none',
      loss_weight=0.5),
    loss_bbox=dict(
      type='IoULoss',
      iou_mode='ciou',
      bbox_format='xyxy',
      reduction='sum',
      loss_weight=7.5,
      return_iou=False),
    loss_dfl=dict(
      type='mmdet.DistributionFocalLoss',
      reduction='mean',
      loss_weight=1.5 / 4)),

  test_cfg=dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.7),
    max_per_img=300))

# Test pipeline for inference
test_pipeline = [
  dict(type='LoadImageFromFile'),
  dict(type='YOLOv5KeepRatioResize', scale=img_scale),
  dict(type='LetterResize',
       scale=img_scale,
       allow_scale_up=False,
       pad_val=dict(img=pad_val)),
  dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
  dict(type='mmdet.PackDetInputs',
       meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                  'scale_factor', 'pad_param'))
]

# Dummy test dataloader (required by MMDetection API)
test_dataloader = dict(
  batch_size=1,
  num_workers=1,
  persistent_workers=False,
  drop_last=False,
  sampler=dict(type='DefaultSampler', shuffle=False),
  dataset=dict(
    type='YOLOv5CocoDataset',
    data_root='.',
    metainfo=metainfo,
    ann_file='annotations.json',
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=test_pipeline))

test_evaluator = dict(
  type='mmdet.CocoMetric',
  ann_file='annotations.json',
  metric='bbox',
  format_only=False)