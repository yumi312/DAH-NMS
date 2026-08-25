"""MMDetection adapter for DAH-NMS.

Replaces the detector's standard box NMS with DAH-NMS after masks are
available. Works with stock instance-segmentation models (Mask R-CNN, HTC,
Cascade Mask R-CNN, …). GE-CAN / BM-RM are not required.
"""

from __future__ import annotations

from typing import Any

import torch

from ..config import DAHNMSConfig
from ..core import dah_nms


def _as_mask_tensor(masks: Any, device: torch.device) -> torch.Tensor:
    """Convert MMDet mask containers to a ``[N, H, W]`` tensor."""
    if masks is None:
        return torch.zeros(0, 0, 0, device=device)
    if torch.is_tensor(masks):
        return masks.to(device=device)
    if hasattr(masks, "to_tensor"):
        # BitmapMasks / PolygonMasks
        try:
            return masks.to_tensor(dtype=torch.bool, device=device)
        except TypeError:
            arr = masks.to_ndarray()
            return torch.as_tensor(arr, device=device)
    if hasattr(masks, "masks"):
        return torch.as_tensor(masks.masks, device=device)
    if hasattr(masks, "to_ndarray"):
        return torch.as_tensor(masks.to_ndarray(), device=device)
    raise TypeError(f"Unsupported mask type: {type(masks)!r}")


def apply_dahnms_to_instances(
    instances: Any,
    cfg: DAHNMSConfig,
    *,
    max_det: int | None = None,
) -> Any:
    """Filter a single-image ``InstanceData`` (or similar) with DAH-NMS.

    Expects ``bboxes``, ``scores``, ``labels``, and ``masks`` fields.
    """
    if instances is None or len(instances) == 0:
        return instances
    if not hasattr(instances, "masks"):
        raise ValueError("DAH-NMS requires instance masks; got no `masks` field")

    boxes = instances.bboxes
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    boxes = boxes.float()
    scores = instances.scores.float()
    classes = instances.labels.long()
    masks = _as_mask_tensor(instances.masks, boxes.device).float()

    keep = dah_nms(boxes, scores, classes, masks, cfg)
    if max_det is not None and max_det > 0:
        keep = keep[:max_det]
    return instances[keep]


def _soften_box_nms(test_cfg: Any, max_candidates: int) -> None:
    """Disable IoU suppression so mask head sees score-filtered candidates."""
    if test_cfg is None:
        return
    # roi_head.test_cfg is the rcnn dict: score_thr / nms / max_per_img
    nms_cfg = dict(type="nms", iou_threshold=1.0)
    if hasattr(test_cfg, "nms"):
        test_cfg.nms = nms_cfg
    elif isinstance(test_cfg, dict):
        test_cfg["nms"] = nms_cfg

    cur = 0
    if hasattr(test_cfg, "max_per_img"):
        cur = int(test_cfg.max_per_img)
        test_cfg.max_per_img = max(cur, max_candidates)
    elif isinstance(test_cfg, dict) and "max_per_img" in test_cfg:
        cur = int(test_cfg["max_per_img"])
        test_cfg["max_per_img"] = max(cur, max_candidates)


def attach_dahnms(
    model: Any,
    cfg: DAHNMSConfig | None = None,
    *,
    max_candidates: int = 1000,
    max_det: int | None = None,
) -> Any:
    """Attach DAH-NMS to an MMDetection instance-segmentation model.

    Softens the RoI head's box NMS (keep top-``max_candidates`` by score),
    runs the mask head as usual, then applies DAH-NMS on the surviving
    instances. Compatible with ``init_detector`` + ``inference_detector``.

    Args:
        model: An MMDet detector with ``roi_head.with_mask``.
        cfg: DAH-NMS hyper-parameters (defaults if omitted).
        max_candidates: Max score-filtered boxes passed to the mask head.
        max_det: Optional cap after DAH-NMS (defaults to ``max_candidates``).
    """
    cfg = cfg or DAHNMSConfig()
    if max_det is None:
        max_det = max_candidates

    if not hasattr(model, "roi_head"):
        raise ValueError("attach_dahnms expects a two-stage detector with `roi_head`")
    roi_head = model.roi_head
    if not getattr(roi_head, "with_mask", False):
        raise ValueError(
            "DAH-NMS requires an instance-segmentation model (mask head missing)"
        )

    _soften_box_nms(getattr(roi_head, "test_cfg", None), max_candidates)
    # Also soften top-level model.test_cfg.rcnn if present (some configs share it)
    model_test_cfg = getattr(model, "test_cfg", None)
    if model_test_cfg is not None and hasattr(model_test_cfg, "rcnn"):
        _soften_box_nms(model_test_cfg.rcnn, max_candidates)

    original_predict = roi_head.predict

    def predict_with_dahnms(*args: Any, **kwargs: Any):
        results_list = original_predict(*args, **kwargs)
        return [
            apply_dahnms_to_instances(res, cfg, max_det=max_det) for res in results_list
        ]

    roi_head.predict = predict_with_dahnms  # type: ignore[method-assign]
    model._dahnms_cfg = cfg
    model._dahnms_attached = True
    return model
