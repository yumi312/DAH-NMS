"""DAH-NMS: Density-Aware Hybrid NMS for instance segmentation."""

from .adapters.mmdet import apply_dahnms_to_instances, attach_dahnms
from .config import DAHNMSConfig
from .core import dah_nms, density_split

__all__ = [
    "DAHNMSConfig",
    "dah_nms",
    "density_split",
    "attach_dahnms",
    "apply_dahnms_to_instances",
    "DAHNMSegmentationValidator",
    "make_dahnms_validator_class",
]

try:
    from .adapters.ultralytics import (
        DAHNMSegmentationValidator,
        make_dahnms_validator_class,
    )
except ImportError:  # ultralytics optional

    DAHNMSegmentationValidator = None  # type: ignore[misc, assignment]

    def make_dahnms_validator_class(*_a, **_k):  # type: ignore[misc]
        raise ImportError(
            "Ultralytics adapter requires `pip install ultralytics`"
        ) from None
