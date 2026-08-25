"""DAH-NMS configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DAHNMSConfig:
    """Hyper-parameters for Density-Aware Hybrid NMS.

    Defaults follow the configuration reported in the paper (Section IV.B:
    density threshold 0.5, IoU threshold 0.7). Other models or datasets may
    benefit from retuning; see Table III of the paper for sensitivity.
    """

    density_thre: float = 0.5
    density_overlap: float = 0.5
    mask_iou_thres: float = 0.7
    box_iou_thres: float = 0.7
    mask_dilate: int = 4
    # Dilate high-density cells before the split (reduces border split-offs)
    density_dilate: int = 3
    # After split NMS: same-class high↔low pairs with box IoU > this compete by score
    cross_box_iou: float = 0.55
