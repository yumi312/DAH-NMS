"""Ultralytics YOLO-Seg adapter for DAH-NMS."""

from __future__ import annotations

import torch
import torchvision

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression as ultralytics_nms

from ..config import DAHNMSConfig
from ..core import (
    MAX_WH,
    _empty,
    box_nms_by_class,
    cross_partition_score_dedup,
    density_split,
    dilate_masks,
    mask_nms_by_class,
)

MAX_NMS = 30000


class DAHNMSegmentationValidator(SegmentationValidator):
    """SegmentationValidator with DAH-NMS instead of standard box NMS."""

    dahnms_cfg: DAHNMSConfig = DAHNMSConfig()

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        det = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        agnostic = bool(self.args.single_cls or self.args.agnostic_nms)
        cfg = self.dahnms_cfg

        cands = ultralytics_nms(
            det,
            self.args.conf,
            iou_thres=1.0,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=agnostic,
            max_det=MAX_NMS,
            end2end=self.end2end,
            rotated=False,
        )

        imgsz = [4 * x for x in proto.shape[2:]]
        out: list[dict[str, torch.Tensor]] = []
        for i, x in enumerate(cands):
            if x.numel():
                keep = _dah_nms_yolo(x, proto=proto[i], cfg=cfg, agnostic=agnostic)[
                    : self.args.max_det
                ]
                x = x[keep]
            pred = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
            coeff = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coeff, pred["bboxes"], shape=imgsz)
                if coeff.shape[0]
                else torch.zeros((0, *imgsz), dtype=torch.uint8, device=x.device)
            )
            out.append(pred)
        return out


def _dah_nms_yolo(
    x: torch.Tensor,
    *,
    proto: torch.Tensor,
    cfg: DAHNMSConfig,
    agnostic: bool,
) -> torch.Tensor:
    """YOLO path: decode masks only for high-density candidates."""
    boxes_xyxy = x[:, :4]
    scores = x[:, 4]
    classes = x[:, 5]
    mask_coeffs = x[:, 6:]

    high_id, low_id = density_split(
        boxes_xyxy,
        cfg.density_thre,
        cfg.density_overlap,
        cfg.density_dilate,
        max_wh=MAX_WH,
    )

    if high_id.numel():
        mh, mw = int(proto.shape[1]), int(proto.shape[2])
        spatial = ops.process_mask(
            proto,
            mask_coeffs[high_id],
            boxes_xyxy[high_id],
            (mh * 4, mw * 4),
            upsample=False,
        )
        spatial = dilate_masks(spatial, cfg.mask_dilate)
        local = mask_nms_by_class(
            scores[high_id], spatial, cfg.mask_iou_thres, classes[high_id]
        )
        high_keep = high_id[local]
    else:
        high_keep = _empty(boxes_xyxy.device)

    if low_id.numel():
        if agnostic:
            local = torchvision.ops.nms(
                boxes_xyxy[low_id], scores[low_id], cfg.box_iou_thres
            )
        else:
            local = box_nms_by_class(
                boxes_xyxy[low_id], scores[low_id], classes[low_id], cfg.box_iou_thres
            )
        low_keep = low_id[local]
    else:
        low_keep = _empty(boxes_xyxy.device)

    if cfg.cross_box_iou > 0 and high_keep.numel() and low_keep.numel():
        high_keep, low_keep = cross_partition_score_dedup(
            high_keep, low_keep, scores, classes, boxes_xyxy, cfg.cross_box_iou
        )

    kept = torch.cat([high_keep, low_keep])
    if kept.numel() == 0:
        return kept
    return kept[scores[kept].argsort(descending=True)]


def make_dahnms_validator_class(cfg: DAHNMSConfig) -> type[DAHNMSegmentationValidator]:
    """Bind a config for ``model.val(validator=...)``."""

    class _V(DAHNMSegmentationValidator):
        dahnms_cfg = cfg

    return _V
