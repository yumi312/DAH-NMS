"""Core DAH-NMS smoke test on random tensors (no weights/dataset needed)."""

import torch

from dahnms import DAHNMSConfig, dah_nms

g = torch.Generator().manual_seed(0)
n = 200
cxy = torch.rand(n, 2, generator=g) * 512
wh = torch.rand(n, 2, generator=g) * 60 + 8
boxes = torch.cat([cxy - wh / 2, cxy + wh / 2], 1).clamp(0, 512)
scores = torch.rand(n, generator=g)
classes = torch.randint(0, 3, (n,), generator=g)
masks = torch.zeros(n, 128, 128)
for i, b in enumerate(boxes / 4):
    x1, y1, x2, y2 = b.int().tolist()
    masks[i, y1 : y2 + 1, x1 : x2 + 1] = 1

keep = dah_nms(boxes, scores, classes, masks, DAHNMSConfig())
print(f"kept {len(keep)} / {n}")
