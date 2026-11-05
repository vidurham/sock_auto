"""Score the automatic bitmap against the designer's finished bitmaps.

    python tools/score.py            # prints accuracy, writes output/score/*.png

Each pair = a mockup (design_pdf/*.png or *.pdf) + the finished BMP made by hand.
Accuracy = % of pixels whose colour matches (palette colours matched by nearest
CIELAB so tiny RGB differences between screenshot and PDF don't count).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sock_extractor.raster import extract_bitmap, rgb_to_lab  # noqa: E402

PAIRS = {
    "chipdf": "Chicago Backflow St Patricks Bitmap.bmp",
    "jarritospdf": "Jarritos Crew Bitmap (Color).bmp",
    "quietmtnpdf": "Quiet Mountain_2502_Bitmap R1.bmp",
    "sedonapdf": "OB Sedona_2603_Bitmap R1 (1).bmp",
}


def gt_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def score_pair(mock: Path, gt_path: Path, out_dir: Path, force_size=True):
    gt = gt_rgb(gt_path)
    th = gt.shape[0] if force_size else None
    r = extract_bitmap(str(mock), target_h=th)
    pal = np.array(r["palette"], np.uint8)
    ours = pal[r["indices"]]
    if ours.shape != gt.shape:
        return {"name": mock.stem, "size": f"{ours.shape[1]}x{ours.shape[0]} vs GT {gt.shape[1]}x{gt.shape[0]}",
                "acc": None}
    # label both images by nearest colour in the union palette
    gt_cols = np.unique(gt.reshape(-1, 3), axis=0)
    allc = np.vstack([pal, gt_cols]).astype(float)
    lab_all = rgb_to_lab(allc)
    def lbl(img):
        lab = rgb_to_lab(img.reshape(-1, 3).astype(float))
        # map every pixel to nearest of OUR palette (so GT colours fold onto ours)
        d = ((lab[:, None] - lab_all[None, : len(pal)]) ** 2).sum(-1)
        return d.argmin(1).reshape(img.shape[:2])
    lo, lg = lbl(ours), lbl(gt)
    acc = float((lo == lg).mean())
    diff = np.where((lo != lg)[..., None], np.array([255, 0, 255], np.uint8), (ours // 3 + 170).astype(np.uint8))
    S = 3
    strip = np.concatenate([gt, ours, diff], axis=1)
    Image.fromarray(strip).resize((strip.shape[1] * S, strip.shape[0] * S), Image.NEAREST) \
        .save(out_dir / f"{mock.stem}_gt_ours_diff.png")
    g = r["geometry"]
    return {"name": mock.stem, "acc": acc, "bad_px": int((lo != lg).sum()),
            "prop_h": round(r["proportional_height"], 1), "gt_h": gt.shape[0],
            "pal_ours": [tuple(int(v) for v in c) for c in pal],
            "pal_gt": [tuple(int(v) for v in c) for c in gt_cols],
            "notes": g.notes}


def main():
    out_dir = ROOT / "output" / "score"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = []
    for stem, bmp in PAIRS.items():
        mock = next((p for p in (ROOT / "design_pdf").glob(stem + ".*")), None)
        if mock is None:
            continue
        try:
            s = score_pair(mock, ROOT / "finished_bit" / bmp, out_dir)
        except Exception as e:  # keep scoring the rest
            print(f"{stem:14s} ERROR {e}")
            continue
        if s["acc"] is None:
            print(f"{stem:14s} SIZE MISMATCH {s['size']}")
            continue
        total.append(s["acc"])
        print(f"{stem:14s} {100*s['acc']:6.2f}%  wrong px={s['bad_px']:5d}  "
              f"aspect-height={s['prop_h']} (GT {s['gt_h']})  {s['notes'] or ''}")
        print(f"{'':14s} ours {s['pal_ours']}\n{'':14s} gt   {s['pal_gt']}")
    if total:
        print(f"MEAN {100*np.mean(total):.2f}%")


if __name__ == "__main__":
    main()
