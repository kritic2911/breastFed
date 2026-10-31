"""Shared image preprocessing used by every site's ingest script.

Two versions are produced per image so the harmonization ablation is a flag later:
  raw512  : full image, percentile intensity scaling, resized + padded to 512x512
  proc512 : breast-cropped, orientation-aligned (chest wall on the left),
            intensity-normalised, CLAHE contrast, 512x512
"""
import cv2
import numpy as np

SIZE = 512
MASK_THR = 15  # 0-255 threshold used to find the breast region; tune if crops look wrong


def scale_u8(arr, lo, hi):
    if hi <= lo:
        hi = lo + 1.0
    arr = np.asarray(arr, dtype=np.float32)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_square(img, size=SIZE):
    """Scale the longest side to `size`, keep aspect ratio, centre on a black square."""
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = img
    return out


def make_raw(arr):
    lo, hi = np.percentile(arr, (0.5, 99.5))
    return to_square(scale_u8(arr, lo, hi))


def make_proc(arr):
    lo, hi = np.percentile(arr, (0.5, 99.5))
    u8 = scale_u8(arr, lo, hi)
    H, W = u8.shape

    # 1) find the breast: threshold, clean up, keep the largest connected blob
    mask = (cv2.GaussianBlur(u8, (5, 5), 0) > MASK_THR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(v) for v in stats[k, :4])
    else:
        x, y, w, h = 0, 0, W, H

    # 2) crop with a small margin
    m = int(0.02 * max(w, h))
    x0, x1 = max(x - m, 0), min(x + w + m, W)
    y0, y1 = max(y - m, 0), min(y + h + m, H)
    crop = u8[y0:y1, x0:x1]

    # 3) orientation: put the bright (chest-wall) side on the left
    cw = crop.shape[1]
    if crop[:, : cw // 2].sum(dtype=np.int64) < crop[:, cw // 2:].sum(dtype=np.int64):
        crop = crop[:, ::-1]

    # 4) intensity normalisation inside tissue, then CLAHE
    tissue = crop > MASK_THR
    if tissue.sum() > 100:
        lo2, hi2 = np.percentile(crop[tissue], (1, 99.5))
        crop = scale_u8(crop, lo2, hi2)
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.ascontiguousarray(crop))
    return to_square(crop)
