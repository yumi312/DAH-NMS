"""Plug DAH-NMS into your own YOLO-Seg model + data (weights/data not included)."""

from ultralytics import YOLO

from dahnms import DAHNMSConfig, make_dahnms_validator_class

WEIGHTS = "path/to/best.pt"
DATA = "path/to/data.yaml"

cfg = DAHNMSConfig(
    density_thre=0.5,
    density_overlap=0.5,
    density_dilate=3,
    mask_dilate=4,
    mask_iou_thres=0.7,
    box_iou_thres=0.7,
    cross_box_iou=0.55,
)

model = YOLO(WEIGHTS)
model.val(
    data=DATA,
    conf=0.1,
    max_det=1000,
    retina_masks=True,
    validator=make_dahnms_validator_class(cfg),
)
