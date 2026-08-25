"""Plug DAH-NMS into an MMDetection instance-seg model (weights not included)."""

from mmdet.apis import inference_detector, init_detector

from dahnms import DAHNMSConfig
from dahnms.adapters.mmdet import attach_dahnms

CONFIG = "path/to/htc_config.py"
CHECKPOINT = "path/to/checkpoint.pth"
IMAGE = "path/to/image.jpg"

cfg = DAHNMSConfig(
    density_thre=0.5,
    density_overlap=0.5,
    density_dilate=3,
    mask_dilate=4,
    mask_iou_thres=0.7,
    box_iou_thres=0.7,
    cross_box_iou=0.55,
)

model = init_detector(CONFIG, CHECKPOINT, device="cuda:0")
model = attach_dahnms(model, cfg, max_candidates=1000)

result = inference_detector(model, IMAGE)
# result.pred_instances: bboxes / scores / labels / masks (DAH-NMS filtered)
print(len(result.pred_instances))
