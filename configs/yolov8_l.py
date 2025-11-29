# YOLOv8-large MobileNetV2 Config for Inference
# Based on: yolov8_l_mobilenet_v2_512x288_indices_246.py

default_scope = 'mmyolo'

# Image size
img_scale = (512, 288)  # width x height
pad_val = 114

# MobileNetV2 architecture settings
arch_settings = [[1, 16, 1, 1],
                 [6, 24, 2, 2],
                 [6, 32, 3, 2],
                 [6, 64, 4, 2],
                 [6, 96, 3, 1],
                 [6, 160, 3, 2],
                 [6, 320, 1, 1]]
arch_settings.append([0, 1280, 0, 0])

mobilenet_out_indices = (2, 4, 6)
deepen_factor = 1
channels = [arch_settings[i][1] for i in mobilenet_out_indices]  # [32, 96, 320]

# 9 classes (8 vehicles + mask)
num_classes = 9

metainfo = dict(
  classes=('bicycle', 'motorcycle', 'car', 'transporter', 'bus', 'truck', 'trailer', 'unknown', 'mask')
)

strides = [8, 16, 32]

# Model definition
model = dict(
  type='YOLODetector',
  data_preprocessor=dict(
    type='YOLOv5DetDataPreprocessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True),

  # MobileNetV2 backbone
  backbone=dict(
    type='mmdet.MobileNetV2',
    out_indices=mobilenet_out_indices,  # (2, 4, 6)
    act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    # Note: init_cfg with pretrained checkpoint is only needed for training
    # For inference, the weights are loaded from the .pth file
  ),

  # YOLO neck adapted for MobileNetV2 channels
  neck=dict(
    type='YOLOv8PAFPN',
    in_channels=channels,  # [32, 96, 320] from MobileNetV2
    out_channels=channels,  # Keep same channels
    deepen_factor=deepen_factor,
    widen_factor=1,
    num_csp_blocks=3,
    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    act_cfg=dict(type='SiLU', inplace=True)),

  # Detection head
  bbox_head=dict(
    type='YOLOv8Head',
    head_module=dict(
      type='YOLOv8HeadModule',
      num_classes=num_classes,
      in_channels=channels,  # [32, 96, 320]
      widen_factor=1,
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

# Dummy test dataloader (required by MMDetection API but not actually used)
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
    ann_file='annotations.json',  # Dummy file
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=test_pipeline))

test_evaluator = dict(
  type='mmdet.CocoMetric',
  ann_file='annotations.json',  # Dummy file
  metric='bbox',
  format_only=False)

# Allow unused parameters (needed for this architecture)
find_unused_parameters = True