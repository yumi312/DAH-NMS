# DAH-NMS

**Density-Aware Hybrid NMS** for instance segmentation in dense, heavily overlapping scenes.

This repository provides the standalone DAH-NMS module from the paper:

> **Edge-Cloud Collaborative Framework for Density-Aware Hair Shaft and Follicle Segmentation in Distributed Hair Loss Inspection** (under review, 2026)

DAH-NMS routes candidate detections by local density: crowded regions are resolved with class-aware **mask-based NMS** (shape-accurate, tolerant of the heavy box overlap between thin, elongated instances), sparse regions keep fast **box-based NMS**, and a cross-partition step removes duplicates that straddle the two branches. The core is **framework-agnostic** — it operates on plain tensors (boxes, scores, classes, masks) with no hardcoded class names — and ships with thin adapters for Ultralytics YOLO-Seg and MMDetection.

## Scope of this release

The full system described in the paper was developed jointly with a hospital and an industry partner. The clinical images are covered by an IRB protocol and patient-privacy agreements, and the edge-cloud deployment stack contains partner-proprietary code, so those parts cannot be released, and requests for the clinical data cannot be accommodated. The post-processing method itself is independent of that data and infrastructure, and is released here in full.

**Included**

- DAH-NMS core (framework-agnostic, no dataset- or class-specific logic)
- Ultralytics YOLO-Seg adapter
- MMDetection adapter (works with stock detectors such as HTC and Mask R-CNN)

**Not released** (clinical-data and partner agreements)

- the clinical dataset, annotations, and trained weights
- training code and the GE-CAN / BM-RM network modules
- the edge-cloud deployment stack

DAH-NMS is detector-agnostic: it can be attached to any instance-segmentation model and evaluated on any dataset with dense, overlapping instances.

## How it works

Input: all candidate detections from a model, confidence-filtered but **before** any NMS.

1. **Density map** — accumulate predicted boxes into a global density map, threshold it (`density_thre`) to isolate high-density regions, then dilate them (`density_dilate`).
2. **Partition** — each candidate is routed by its intersection ratio with the dilated high-density regions (`density_overlap`): above the ratio → high-density branch; otherwise → low-density branch.
3. **High-density branch** — class-aware mask-based NMS: masks are dilated (`mask_dilate`) before suppression by mask IoU (`mask_iou_thres`).
4. **Low-density branch** — traditional hard box-NMS (`box_iou_thres`).
5. **Cross-partition dedup** — for a same-class pair with one survivor from each branch and box IoU above `cross_box_iou`, keep the higher score.
6. Concatenate survivors and sort by score.

See Section III.C and Fig. 6 of the paper for the method, and Table III for parameter sensitivity. Steps 1–4 follow Section III.C of the paper; the dilation steps and the cross-partition dedup (step 5) are release-level implementation details.

## Install

```bash
pip install -r requirements.txt   # core: torch, torchvision, numpy
# optional adapters — install only what you need:
pip install "ultralytics>=8.3.0"  # YOLO-Seg adapter (AGPL-3.0); tested with 8.4.128
pip install "mmdet>=3.0"          # MMDetection adapter (Apache-2.0)
# MMDetection is easiest to install via OpenMMLab's mim; see the MMDetection docs.
```

Add this repository to `PYTHONPATH`, or install editable:

```bash
pip install -e .
```

Tested with ultralytics 8.4.128 and mmdet 3.x.

## Usage

```bash
python smoke_test.py  # runs on random tensors, no weights needed
```

### Ultralytics YOLO-Seg

```python
from ultralytics import YOLO
from dahnms import DAHNMSConfig, make_dahnms_validator_class

cfg = DAHNMSConfig()
model = YOLO("your_model.pt")
model.val(
    data="your_data.yaml",
    conf=0.1,
    max_det=1000,
    retina_masks=True,
    validator=make_dahnms_validator_class(cfg),
)
```

Works with any YOLO-Seg model and any class list. See `example_val.py`.

### MMDetection (HTC, Mask R-CNN, ...)

The paper's full model is an HTC-based network implemented on MMDetection, so this adapter follows the same post-processing path as the paper: it replaces the detector's standard NMS with DAH-NMS.

```python
from mmdet.apis import init_detector, inference_detector
from dahnms import DAHNMSConfig
from dahnms.adapters.mmdet import attach_dahnms

model = init_detector("htc_config.py", "checkpoint.pth", device="cuda:0")
model = attach_dahnms(model, DAHNMSConfig(), max_candidates=1000)

result = inference_detector(model, "image.jpg")
# result.pred_instances — DAH-NMS filtered
```

The paper's complete network additionally contains GE-CAN and BM-RM, which are not part of this release; the adapter works with unmodified MMDetection models. See `example_mmdet.py`.

### Core API (any framework)

```python
from dahnms import DAHNMSConfig, dah_nms

keep = dah_nms(boxes, scores, classes, masks, DAHNMSConfig())
# boxes: [N,4] xyxy, scores: [N], classes: [N], masks: [N,H,W]
```

## Defaults


| Param             | Default | Role                                                                                      |
| ----------------- | ------- | ----------------------------------------------------------------------------------------- |
| `density_thre`    | `0.5`   | Threshold on the density map to isolate high-density regions                              |
| `density_overlap` | `0.5`   | Box → high-density candidate if intersection ratio with high-density regions exceeds this |
| `density_dilate`  | `3`     | Dilate high-density regions before classifying candidates                                 |
| `mask_dilate`     | `4`     | Dilate masks before Mask-based NMS on high-density candidates                             |
| `mask_iou_thres`  | `0.7`   | Mask IoU threshold for high-density Mask-based NMS                                        |
| `box_iou_thres`   | `0.7`   | Box IoU threshold for low-density traditional NMS                                         |
| `cross_box_iou`   | `0.55`  | Cross-partition same-class box IoU; keep higher score                                     |


### Correspondence to the paper

| Paper (Sections III.C / IV.B, Table III)                               | This repo                                                       | Default |
| ---------------------------------------------------------------------- | --------------------------------------------------------------- | ------- |
| density threshold — intersection ratio routing a box to mask-based NMS | `density_overlap`                                               | `0.5`   |
| IoU threshold applied within the suppression step                      | `mask_iou_thres` (high-density) / `box_iou_thres` (low-density) | `0.7`   |

The remaining parameters are implementation details not swept in the paper:
`density_thre` (map-level threshold isolating high-density regions),
`density_dilate`, `mask_dilate`, and `cross_box_iou`.

Other detectors or datasets may benefit from retuning (see Table III).

## Repository structure

```
dahnms/
  __init__.py
  core.py              # framework-agnostic DAH-NMS
  config.py            # DAHNMSConfig
  adapters/
    ultralytics.py     # YOLO-Seg validator
    mmdet.py           # MMDetection attach_dahnms
example_val.py
example_mmdet.py
smoke_test.py
requirements.txt
```

## Citation

If you use this code, please cite the paper (BibTeX will be updated upon publication):

```bibtex
@article{jhong2026dahnms,
  title={Edge-Cloud Collaborative Framework for Density-Aware Hair Shaft and Follicle Segmentation in Distributed Hair Loss Inspection},
  author={Jhong, Sin-Ye and Wu, Yi-Chen and Hsia, Chih-Hsien},
  note={Under review},
  year={2026}
}
```

## License

- DAH-NMS core (`dahnms/`): **MIT**. Depends only on PyTorch, torchvision, and NumPy.
- Ultralytics adapter: imports `ultralytics` (**AGPL-3.0**); use of that adapter is subject to AGPL terms.
- MMDetection adapter: designed for `mmdet` (**Apache-2.0**).

Neither adapter is required to use the core.
