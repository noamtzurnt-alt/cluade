"""Paste the real faces from the original photos into the composite.

Only pixels inside a soft ellipse around each face change; every other
pixel of the composite stays bit-identical.

Usage:
  pip install opencv-python-headless numpy
  python3 replace_faces.py COMPOSITE.png HELICOPTER.webp BASKETBALL.jpg MIRROR.jpg OUT.png
The three photos are the ORIGINAL full photos (not the face crops).
"""
import sys
import cv2
import numpy as np

comp_path, heli, ball, mirror, out_path = sys.argv[1:6]
comp = cv2.imread(comp_path)
H, W = comp.shape[:2]
sx = W / 1881  # panel ranges below were measured on a 1881px-wide composite

# photo -> (panel x-range in composite, face ellipse center/axes in photo coords)
jobs = [
    (heli,   (360, 930),  (668, 795), (64, 86)),
    (ball,   (0, 380),    (447, 770), (56, 72)),
    (mirror, (920, 1340), (614, 470), (76, 100)),
]

sift = cv2.SIFT_create(8000)
gray_comp = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
out = comp.astype(np.float32)
for path, (x0, x1), center, axes in jobs:
    src = cv2.imread(path)
    region = np.zeros((H, W), np.uint8)
    region[:, int(x0 * sx):int(x1 * sx)] = 255
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(gray_comp, region)
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3)
    print(f"{path}: {int(inliers.sum())} inlier matches")

    mask = np.zeros(src.shape[:2], np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), axes[0] * 0.18)
    warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.warpAffine(mask, M, (W, H))[..., None]
    out = out * (1 - alpha) + warped * alpha

out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
cv2.imwrite(out_path, out)
changed = (out != comp).any(axis=2).sum()
print(f"saved {out_path}; changed {changed} of {H * W} pixels")
