"""Experiment lab: coverage maps from cached renders vs finished bitmaps."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from sock_extractor.raster import rgb_to_lab

EXCLUDE = ("Intergration", "Scramble", "Toyota")   # finished bitmap != mockup (revised art)


def load(name_filter=""):
    out = []
    for f in sorted((ROOT / "output" / "cache").glob("*.npz")):
        if any(e in f.name for e in EXCLUDE) or name_filter.lower() not in f.name.lower():
            continue
        z = np.load(f, allow_pickle=True)
        gt = np.array(Image.open(str(z["bmp"])).convert("RGB"))
        out.append((f.stem, z["arr"], [tuple(c) for c in z["pal"]], gt))
    return out


def nearest_idx(arr, pal):
    pl = rgb_to_lab(np.array(pal, float))
    flat = arr.reshape(-1, 3)
    n = np.empty(len(flat), np.int16); dm = np.empty(len(flat))
    for i in range(0, len(flat), 400_000):
        d = ((rgb_to_lab(flat[i:i+400_000].astype(float))[:, None] - pl[None]) ** 2).sum(-1)
        n[i:i+400_000] = d.argmin(1); dm[i:i+400_000] = np.sqrt(d.min(1))
    return n.reshape(arr.shape[:2]), dm.reshape(arr.shape[:2])


def coverage(nidx, P, H, W, dx=0.0, dy=0.0):
    """(P,H,W) fraction of each output cell covered by each palette colour.
    dx, dy shift the output grid in OUTPUT pixels."""
    h, w = nidx.shape
    sy, sx = h / H, w / W
    cov = np.zeros((P, H, W), np.float32)
    ys = ((np.arange(h) + 0.5) / sy - dy).astype(int).clip(0, H - 1)
    xs = ((np.arange(w) + 0.5) / sx - dx).astype(int).clip(0, W - 1)
    cell = ys[:, None] * W + xs[None, :]
    for k in range(P):
        cov[k] = np.bincount(cell[nidx == k], minlength=H * W).reshape(H, W)
    return cov / np.maximum(cov.sum(0, keepdims=True), 1)


def gt_labels(gt, pal):
    gcols, ginv = np.unique(gt.reshape(-1, 3), axis=0, return_inverse=True)
    d = ((rgb_to_lab(gcols.astype(float))[:, None] - rgb_to_lab(np.array(pal, float))[None]) ** 2).sum(-1)
    return d.argmin(1)[ginv.ravel()].reshape(gt.shape[:2])
