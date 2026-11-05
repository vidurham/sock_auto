"""Chip name + chip RGB from a mockup PDF's colour column."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import fitz


def chips(pdf):
    pg = fitz.open(pdf)[0]
    spans = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = " ".join(s["text"].strip() for s in l["spans"]).strip()
            if t:
                spans.append((t, fitz.Rect(l["bbox"])))
    out = []
    for d in pg.get_drawings():
        r = d.get("rect")
        if r is None or r.x1 > 210 or not (30 <= r.width <= 90 and 0.85 <= r.width / r.height <= 1.15):
            continue
        rgb = tuple(int(round(c * 255)) for c in d["fill"][:3]) if d.get("type") in ("f", "fs") and d.get("fill") else (255, 255, 255)
        above = [(r.y0 - s[1].y1, s[0], s[1]) for s in spans if s[1].y1 <= r.y0 + 2 and r.y0 - s[1].y1 < 45 and s[1].x1 < 230]
        above.sort(key=lambda t: t[0])
        # label may be 2 lines: take lines within 30pt above
        name = " ".join(t for _, t, rr in sorted(above, key=lambda t: t[2].y0) if r.y0 - rr.y1 < 40)
        out.append((r.y0, name, rgb))
    out.sort()
    seen, res = set(), []
    for y, n, c in out:
        if (n, c) not in seen:
            seen.add((n, c)); res.append((n, c))
    return res


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    from tools.pairs import pairs
    from sock_extractor.raster import rgb_to_lab
    for pdf, bmp in pairs():
        ch = chips(pdf)
        gcols = np.unique(np.array(Image.open(bmp).convert("RGB")).reshape(-1, 3), axis=0)
        for n, c in ch:
            if not len(gcols):
                continue
            d = np.sqrt(((rgb_to_lab(gcols.astype(float)) - rgb_to_lab(np.array([c], float))) ** 2).sum(-1))
            k = d.argmin()
            flag = "" if d[k] < 3 else ("  <-- DIFFERENT" if d[k] < 25 else "  (unused?)")
            print(f"{pdf.stem[:22]:22s} {n[:32]:32s} chip={c} gt={tuple(int(v) for v in gcols[k])} dE={d[k]:.1f}{flag}")
