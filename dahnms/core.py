"""Framework-agnostic Density-Aware Hybrid NMS (DAH-NMS).

Operates on plain tensors (boxes, scores, classes, masks) with no detector-
or class-name coupling. High-density → class-aware Mask-NMS; low-density →
class-aware Box-NMS; then cross-partition same-class score competition.
"""

from __future__ import annotations

import torch
import torchvision

from .config import DAHNMSConfig

GRID = 160
MAX_WH = 7680


def _empty(device: torch.device) -> torch.Tensor:
    return torch.zeros(0, dtype=torch.int64, device=device)


def _grid_xyxy(boxes: torch.Tensor, max_wh: int, grid: int) -> tuple[torch.Tensor, ...]:
    coords = (boxes % max_wh) * (float(grid) / float(max_wh))
    return coords.T.int().clamp(0, grid - 1)


def dilate_masks(masks: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilate masks of shape [N, H, W] with a square kernel."""
    if radius <= 0 or masks.numel() == 0:
        return masks
    k = radius * 2 + 1
    x = masks.float().unsqueeze(1)
    kernel = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype)
    return (torch.nn.functional.conv2d(x, kernel, padding=radius).squeeze(1) > 0).to(masks.dtype)


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """Pairwise mask IoU. ``mask1`` [N, HW], ``mask2`` [M, HW] → [N, M]."""
    intersection = torch.matmul(mask1, mask2.T).clamp(min=0)
    area1 = mask1.sum(dim=1).clamp(min=1e-6)
    area2 = mask2.sum(dim=1).clamp(min=1e-6)
    return intersection / (area1[:, None] + area2[None, :] - intersection)


def _mask_nms(scores: torch.Tensor, masks: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if masks.numel() == 0:
        return _empty(scores.device)
    feats = masks.flatten(1).float()
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel():
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[mask_iou(feats[i : i + 1], feats[rest]).view(-1) <= iou_thres]
    return torch.tensor(keep, dtype=torch.int64, device=scores.device)


def mask_nms_by_class(
    scores: torch.Tensor,
    masks: torch.Tensor,
    iou_thres: float,
    classes: torch.Tensor | None,
) -> torch.Tensor:
    """Class-aware Mask-NMS (standard per-class NMS; not tied to any class name)."""
    if classes is None:
        return _mask_nms(scores, masks, iou_thres)
    kept: list[torch.Tensor] = []
    for cls_id in classes.unique():
        idx = torch.nonzero(classes == cls_id, as_tuple=True)[0]
        local = _mask_nms(scores[idx], masks[idx], iou_thres)
        if local.numel():
            kept.append(idx[local])
    return torch.cat(kept) if kept else _empty(scores.device)


def box_nms_by_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_thres: float,
) -> torch.Tensor:
    """Class-aware hard Box-NMS."""
    kept: list[torch.Tensor] = []
    for cls_id in classes.unique():
        idx = torch.nonzero(classes == cls_id, as_tuple=True)[0]
        local = torchvision.ops.nms(boxes[idx], scores[idx], iou_thres)
        if local.numel():
            kept.append(idx[local])
    return torch.cat(kept) if kept else _empty(scores.device)


def density_split(
    boxes: torch.Tensor,
    density_thre: float,
    density_overlap: float,
    density_dilate: int = 0,
    *,
    max_wh: int = MAX_WH,
    grid: int = GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition boxes into high- / low-density index sets."""
    n = len(boxes)
    if n == 0:
        e = _empty(boxes.device)
        return e, e
    dens = torch.zeros((grid, grid), device=boxes.device)
    x1, y1, x2, y2 = _grid_xyxy(boxes, max_wh, grid)
    for j in range(n):
        dens[y1[j] : y2[j], x1[j] : x2[j]] += 1
    if dens.max() > 0:
        dens = dens / dens.max()
    high_cells = (dens > density_thre).float()
    if density_dilate > 0:
        high_cells = dilate_masks(high_cells.unsqueeze(0), density_dilate).squeeze(0).float()
    is_high = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    for j in range(n):
        if y2[j] > y1[j] and x2[j] > x1[j]:
            is_high[j] = high_cells[y1[j] : y2[j], x1[j] : x2[j]].mean() > density_overlap
    return torch.nonzero(is_high, as_tuple=True)[0], torch.nonzero(~is_high, as_tuple=True)[0]


def cross_partition_score_dedup(
    high_keep: torch.Tensor,
    low_keep: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    boxes: torch.Tensor,
    iou_thres: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-class high/low pairs with box IoU > iou_thres: keep the higher score."""
    if iou_thres <= 0 or high_keep.numel() == 0 or low_keep.numel() == 0:
        return high_keep, low_keep

    cross = torchvision.ops.box_iou(boxes[low_keep], boxes[high_keep])
    same = classes[low_keep][:, None] == classes[high_keep][None, :]
    cross = cross.masked_fill(~same, 0.0)

    n_h = int(high_keep.numel())
    n_l = int(low_keep.numel())
    all_scores = torch.cat([scores[high_keep], scores[low_keep]])
    order = all_scores.argsort(descending=True)
    alive_h = torch.ones(n_h, dtype=torch.bool, device=high_keep.device)
    alive_l = torch.ones(n_l, dtype=torch.bool, device=low_keep.device)

    for idx in order.tolist():
        if idx < n_h:
            if not bool(alive_h[idx]):
                continue
            alive_l &= ~((cross[:, idx] > iou_thres) & alive_l)
        else:
            j = idx - n_h
            if not bool(alive_l[j]):
                continue
            alive_h &= ~((cross[j, :] > iou_thres) & alive_h)

    return high_keep[alive_h], low_keep[alive_l]


def dah_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    masks: torch.Tensor,
    cfg: DAHNMSConfig,
    *,
    max_wh: int = MAX_WH,
) -> torch.Tensor:
    """Run DAH-NMS and return keep indices sorted by descending score.

    Args:
        boxes: ``[N, 4]`` xyxy in image coordinates.
        scores: ``[N]`` confidence scores.
        classes: ``[N]`` class ids (any integer labels; no name coupling).
        masks: ``[N, H, W]`` binary / soft masks aligned with ``boxes``.
        cfg: DAH-NMS hyper-parameters.
        max_wh: Coordinate scale used by the density grid (legacy YOLO-compatible).
    """
    if boxes.numel() == 0:
        return _empty(boxes.device)

    high_id, low_id = density_split(
        boxes,
        cfg.density_thre,
        cfg.density_overlap,
        cfg.density_dilate,
        max_wh=max_wh,
    )

    if high_id.numel():
        spatial = dilate_masks(masks[high_id], cfg.mask_dilate)
        local = mask_nms_by_class(
            scores[high_id], spatial, cfg.mask_iou_thres, classes[high_id]
        )
        high_keep = high_id[local]
    else:
        high_keep = _empty(boxes.device)

    if low_id.numel():
        local = box_nms_by_class(
            boxes[low_id], scores[low_id], classes[low_id], cfg.box_iou_thres
        )
        low_keep = low_id[local]
    else:
        low_keep = _empty(boxes.device)

    if cfg.cross_box_iou > 0 and high_keep.numel() and low_keep.numel():
        high_keep, low_keep = cross_partition_score_dedup(
            high_keep, low_keep, scores, classes, boxes, cfg.cross_box_iou
        )

    kept = torch.cat([high_keep, low_keep])
    if kept.numel() == 0:
        return kept
    return kept[scores[kept].argsort(descending=True)]
