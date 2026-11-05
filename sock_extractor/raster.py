"""
Image-based (raster) mockup -> production bitmap.

Works on a screenshot / PNG / JPG export of a Custom Sock Lab mockup page, or on
any PDF page rendered to pixels (useful when a PDF is a flattened image and the
vector path in core.py can't find the FLAT VIEW shapes).

Rules learned from the designer's finished bitmaps (see finished_bit/):
  * Only the FLAT VIEW sock body is knit into the bitmap.
  * The CUFF band (thin strip at the top of the flat view) is NOT included --
    the bitmap starts directly below the cuff line.
  * The TOE scallops are NOT included -- the bitmap ends at the toe line.
  * Heel ovals and the blue heel guide line are erased (artwork continued).
  * Width maps to 168 px; height comes from the product spec table (or from the
    body's aspect ratio when the product can't be read from the page).
  * Colours are exactly the swatch chips in the left column, and only the chips
    the artwork actually uses end up in the BMP palette.

Template geometry (stable across the examples, as fractions of flat width W):
  cuff band height ~0.088 W, heel ovals ~0.25 W wide x ~0.175 W tall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

PAGE_WHITE = 245          # every channel >= this -> blank page background
CUFF_FRAC = 0.088         # cuff band height / flat width
OUTLINE_FRAC = 0.003      # outline stroke to trim / flat width (calibrated on finished bitmaps)
TOE_FRAC = 0.226          # toe scallop height / flat width
LINE_SEARCH_FRAC = 0.035  # +/- search window for cuff / toe lines (x flat width)
HEEL_W_FRAC = 0.26        # heel half-oval reach from each edge (x flat width)
HEEL_H_FRAC = 0.095       # heel half-oval half-height (x flat width)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def load_image(path_or_img, page: int = 0, dpi: int = 200) -> np.ndarray:
    """RGB uint8 array from an image file, a PIL image, or a PDF page."""
    if isinstance(path_or_img, Image.Image):
        return np.array(path_or_img.convert("RGB"))
    p = str(path_or_img)
    if p.lower().endswith(".pdf"):
        import fitz  # PyMuPDF

        doc = fitz.open(p)
        pix = doc[page].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    img = Image.open(p)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return np.array(bg)
    return np.array(img.convert("RGB"))


def _is_page(a: np.ndarray) -> np.ndarray:
    return (a >= PAGE_WHITE).all(axis=-1)


def _is_blue_line(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (b > 140) & (b - r > 70) & (b - g > 15)


def rgb_to_lab(rgb) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    e = (6 / 29) ** 3
    f = np.where(xyz > e, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


# --------------------------------------------------------------------------
# geometry detection
# --------------------------------------------------------------------------


@dataclass
class FlatGeometry:
    x0: int            # first column inside the left outline
    x1: int            # one past the last column inside the right outline
    y_top: int         # top outline of the flat view (top of cuff)
    cuff_y: int        # first body row (just below the cuff line)
    toe_y: int         # one past the last body row (top of toe line)
    heel_y: int        # heel guide line row
    heel_rows: tuple[int, int] = (0, 0)   # rows covered by the blue guide line
    notes: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def body_height(self) -> int:
        return self.toe_y - self.cuff_y


def _find_heel_line(a: np.ndarray) -> tuple[int, int, int, tuple[int, int]]:
    """Row of the blue heel guide line and its horizontal extent."""
    blue = _is_blue_line(a)
    counts = blue.sum(axis=1)
    y = int(counts.argmax())
    if counts[y] < 0.05 * a.shape[1]:
        raise RuntimeError("No blue heel guide line found (is this a sock mockup page?).")
    xs = np.where(blue[y])[0]
    xs = xs[xs > 0.15 * a.shape[1]]          # ignore the swatch column
    # the line can be interrupted by artwork/ovals: take its full extent
    lx0, lx1 = int(xs.min()), int(xs.max())
    rows = np.where(counts >= 0.5 * counts[y])[0]
    rows = rows[np.abs(rows - y) <= 4]
    return y, lx0, lx1, (int(rows.min()), int(rows.max()) + 1)


def _column_is_sock(a: np.ndarray, x: int, y0: int, y1: int) -> bool:
    col = a[y0:y1, x]
    return float((~_is_page(col)).mean()) > 0.9


def _line_score(a: np.ndarray, x0: int, x1: int, r: int, d: int = 3) -> float:
    """How strongly row r looks like a thin full-width outline: it differs from
    the rows a few px above AND below it across most of the width."""
    H = a.shape[0]
    if r - d < 0 or r + d >= H:
        return 0.0
    row = a[r, x0:x1].astype(np.int16)
    up = a[r - d, x0:x1].astype(np.int16)
    dn = a[r + d, x0:x1].astype(np.int16)
    du = np.abs(row - up).sum(1) > 60
    dd = np.abs(row - dn).sum(1) > 60
    return float((du & dd).mean())


def _best_line(a, x0, x1, lo, hi) -> tuple[int, int] | None:
    """Rows [r0, r1) of the strongest thin horizontal outline in [lo, hi)."""
    lo, hi = max(1, lo), min(a.shape[0] - 1, hi)
    if hi <= lo:
        return None
    scores = np.array([_line_score(a, x0, x1, r) for r in range(lo, hi)])
    if scores.size == 0 or scores.max() < 0.5:
        return None
    r = lo + int(scores.argmax())
    thr = 0.5 * scores.max()
    r0 = r
    while r0 - 1 >= lo and scores[r0 - 1 - lo] >= thr:
        r0 -= 1
    r1 = r + 1
    while r1 < hi and scores[r1 - lo] >= thr:
        r1 += 1
    return r0, r1


def detect_flat_view(a: np.ndarray) -> FlatGeometry:
    H, W = a.shape[:2]
    notes: list[str] = []
    heel_y, lx0, lx1, heel_rows = _find_heel_line(a)

    # Columns: the flat view sits inside the overhanging heel line. Sample a band
    # above the line (inside the sock leg) to find the outer outline columns.
    span = lx1 - lx0
    by0, by1 = max(0, heel_y - int(0.5 * span)), max(1, heel_y - 6)
    xl = next((x for x in range(lx0, lx0 + span // 2) if _column_is_sock(a, x, by0, by1)), None)
    xr = next((x for x in range(lx1, lx1 - span // 2, -1) if _column_is_sock(a, x, by0, by1)), None)
    if xl is None or xr is None:
        raise RuntimeError("Could not find the FLAT VIEW left/right edges.")

    # The outer columns are the template's outline stroke, a constant ~0.6% of the
    # flat width (2 px on a 2400-px-wide screenshot). Step inside it.
    t = max(1, int(round(OUTLINE_FRAC * (xr - xl))))
    x0 = xl + t
    x1 = xr + 1 - t
    w = x1 - x0

    # Top / bottom: walk rows away from the heel until a row is blank page across
    # the whole flat view (outline columns included, so a white cuff or white
    # body inside a black outline doesn't count as page).
    def outside(y):
        if not (_is_page(a[y, xl]) and _is_page(a[y, xr])):
            return False
        return float(_is_page(a[y, x0:x1]).mean()) > 0.97

    def walk(y, step, need=4):
        run = 0
        while 0 <= y + step < H:
            y += step
            run = run + 1 if outside(y) else 0
            if run >= need:
                return y - step * need
        return y
    y_top = walk(heel_y - 6, -1)
    y_end = walk(heel_y + 6, +1)      # bottom of the toe scallops

    # Cuff line: thin outline about CUFF_FRAC*w below the top.
    win = int(LINE_SEARCH_FRAC * w) + 2
    exp_cuff = y_top + int(round(CUFF_FRAC * w))
    cl = _best_line(a, x0, x1, exp_cuff - win, exp_cuff + win)
    if cl is None:
        notes.append("cuff line not found; used template ratio")
        cuff_y = exp_cuff + 1
    else:
        cuff_y = cl[1]

    # Toe line: the scallops below it are ~TOE_FRAC*w tall. Search the strongest
    # full-width outline in a window around that prior.
    exp_toe = y_end - int(round(TOE_FRAC * w))
    twin = int(0.06 * w) + 2
    tl = _best_line(a, x0, x1, exp_toe - twin, exp_toe + twin)
    if tl is None:
        notes.append("toe line not found; used template ratio")
        toe_y = exp_toe
    else:
        toe_y = tl[0] + 1   # calibrated: designer bitmaps keep the row touching the toe line

    return FlatGeometry(x0, x1, y_top, cuff_y, toe_y, heel_y, heel_rows, notes)


# --------------------------------------------------------------------------
# swatches -> palette
# --------------------------------------------------------------------------


def detect_swatches(a: np.ndarray, left_frac: float = 0.13) -> list[tuple[int, int, int]]:
    """Swatch chip colours from the left column, top to bottom."""
    H, W = a.shape[:2]
    lw = int(left_frac * W)
    region = a[:, :lw]
    nonpage = ~_is_page(region)
    # the swatch centre column: most non-page pixels (ignore the far-right divider)
    # ignore page borders / frame bars at the very edges of the screenshot
    nonpage[: int(0.01 * H)] = False
    nonpage[H - int(0.01 * H) - 1:] = False
    lo = int(lw * 0.10)
    cnt = nonpage[:, lo: int(lw * 0.85)].sum(0)
    xc = lo + int(cnt.argmax())
    col = nonpage[:, xc]
    runs = []
    y = 0
    while y < H:
        if col[y]:
            y0 = y
            while y < H and col[y]:
                y += 1
            runs.append((y0, y))
        else:
            y += 1
    if not runs:
        return []
    runs = [r for r in runs if r[1] - r[0] < 0.2 * H]   # chips are small circles
    if not runs:
        return []
    def hwidth(y):
        """horizontal extent of the non-page run through (y, xc)"""
        row = nonpage[y]
        l = xc
        while l > 0 and row[l - 1]:
            l -= 1
        r = xc
        while r < lw - 1 and row[r + 1]:
            r += 1
        return r - l + 1

    # Filled chips are circles: vertical run through the centre column ~= the
    # horizontal width at its middle row. Header bands / text fail this test.
    circles = []
    for y0, y1 in runs:
        L = y1 - y0
        if L < 0.02 * H:
            continue
        if 0.8 <= hwidth((y0 + y1) // 2) / L <= 1.25:
            circles.append((y0, y1))
    if not circles:
        return []
    D = float(np.median([y1 - y0 for y0, y1 in circles]))
    out: list[tuple[int, tuple[int, int, int]]] = []
    for y0, y1 in circles:
        m0, m1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        k = max(2, (y1 - y0) // 8)
        patch = region[m0:m1, max(0, xc - k): xc + k + 1].reshape(-1, 3)
        out.append((y0, tuple(int(v) for v in np.median(patch, 0))))
    # white chips are outline-only: two short arcs ~D apart with page between
    short = [r for r in runs if r[1] - r[0] <= 6]
    for (a0, a1), (b0, b1) in zip(short, short[1:]):
        gap = b0 - a1
        if 0.8 * D <= gap + (a1 - a0) + (b1 - b0) <= 1.15 * D:
            if not any(abs(o[0] - a0) < D for o in out):
                out.append((a0, (255, 255, 255)))
    out.sort()
    pal: list[tuple[int, int, int]] = []
    for _, c in out:
        if all(sum(abs(p - q) for p, q in zip(c, k)) > 12 for k in pal):
            pal.append(c)
    return pal


# --------------------------------------------------------------------------
# heel removal
# --------------------------------------------------------------------------


def remove_heel(body: np.ndarray, heel_y: int, heel_rows: tuple[int, int]) -> np.ndarray:
    """Erase the heel guide line and the two edge half-ovals in the body crop.

    Each column inside a half-oval is refilled from just outside the oval: the top
    half from the pixel above, the bottom half from the pixel below. On a solid
    body that is exact; on a patterned body it continues horizontal structure
    without stamping in a distant motif.
    """
    out = body.copy()
    H, W = out.shape[:2]
    reach = int(round(HEEL_W_FRAC * W))
    hh = HEEL_H_FRAC * W + 2
    margin = 2
    yy, xx = np.mgrid[0:H, 0:W]
    for side in (0, 1):
        xe = 0 if side == 0 else W - 1
        lens = ((xx - xe) / (reach + 1.0)) ** 2 + ((yy - heel_y) / hh) ** 2 <= 1.0
        if not lens.any():
            continue
        # ring just outside the lens
        ring = (((xx - xe) / (reach + 5.0)) ** 2 + ((yy - heel_y) / (hh + 4)) ** 2 <= 1.0) & ~lens
        ring_px = out[ring].astype(np.int16)
        q = ring_px // 8 * 8
        u, c = np.unique(q, axis=0, return_counts=True)
        mode = u[c.argmax()]
        near = np.abs(ring_px - mode).sum(1) < 30
        if near.mean() > 0.75:
            # solid surround: paint the lens with the body colour (median of the
            # ring pixels that match the mode, so we keep the exact shade)
            out[lens] = np.median(ring_px[near], axis=0).astype(np.uint8)
            continue
        for i in range(reach + 1):
            x = i if side == 0 else W - 1 - i
            t = i / max(1, reach)
            half = hh * np.sqrt(max(0.0, 1 - t * t))
            if half < 1:
                continue
            ya = int(np.floor(heel_y - half)) - margin
            yb = int(np.ceil(heel_y + half)) + margin
            ya, yb = max(0, ya), min(H - 1, yb)
            mid = heel_y
            if ya > 0:
                out[ya:mid, x] = out[ya - 1, x]
            if yb < H - 1:
                out[mid:yb + 1, x] = out[yb + 1, x]
    # the guide line across the middle
    r0, r1 = heel_rows
    r0, r1 = max(0, r0 - 2), min(H, r1 + 2)
    if 0 < r0 and r1 < H:
        m = (r0 + r1) // 2
        out[r0:m] = out[r0 - 1]
        out[m:r1] = out[r1]
    return out


# --------------------------------------------------------------------------
# quantise + downsample
# --------------------------------------------------------------------------


def quantize_downsample(body: np.ndarray, palette, out_w: int, out_h: int) -> np.ndarray:
    """Per-palette coverage voting.

    Every source pixel is assigned to its nearest palette colour (CIELAB). Pixels
    that are far from every chip (anti-aliased edges, blends) get a low weight so
    they don't invent a third colour on a two-colour edge. Each palette channel's
    weighted coverage is then box-resampled to the output grid and the winner per
    output pixel is taken.
    """
    H, W = body.shape[:2]
    pal_lab = rgb_to_lab(np.array(palette, dtype=np.float64))
    flat = body.reshape(-1, 3)
    nearest = np.empty(flat.shape[0], np.int32)
    dmin = np.empty(flat.shape[0], np.float64)
    step = 250_000                                  # chunk to bound memory
    for i in range(0, flat.shape[0], step):
        lab = rgb_to_lab(flat[i:i + step].astype(np.float64))
        d = np.sqrt(((lab[:, None, :] - pal_lab[None]) ** 2).sum(-1))
        nearest[i:i + step] = d.argmin(1)
        dmin[i:i + step] = d.min(1)
    weight = np.where(dmin < 12, 1.0, np.where(dmin < 25, 0.4, 0.1))
    P = len(palette)
    votes = np.zeros((P, out_h, out_w), np.float32)
    for k in range(P):
        m = np.where(nearest == k, weight, 0.0).reshape(H, W).astype(np.float32)
        votes[k] = np.array(Image.fromarray(m, mode="F").resize((out_w, out_h), Image.BOX))
    return votes.argmax(0).astype(np.uint8)


def compact_palette(indices: np.ndarray, palette, min_px: int = 1):
    """Drop palette entries the bitmap never uses; renumber indices."""
    used = [k for k in range(len(palette)) if int((indices == k).sum()) >= min_px]
    remap = np.zeros(len(palette), np.uint8)
    for new, old in enumerate(used):
        remap[old] = new
    return remap[indices], [tuple(palette[k]) for k in used]


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------


def snap_height(prop_h: float, known: tuple[int, ...]) -> int:
    return int(min(known, key=lambda k: abs(k - prop_h))) if known else int(round(prop_h))


def extract_bitmap(src, target_w: int = 168, target_h: int | None = None,
                   known_heights: tuple[int, ...] | None = None,
                   palette=None) -> dict:
    """Mockup image (path / PIL / array / PDF path) -> production bitmap.

    target_h: force a height (from the product spec). If None, the height is
    derived from the body's aspect ratio and snapped to the nearest known
    production height.
    Returns dict(indices, palette, geometry, design (RGB crop), target_h, ...).
    """
    a = src if isinstance(src, np.ndarray) else load_image(src)
    g = detect_flat_view(a)
    body = a[g.cuff_y:g.toe_y, g.x0:g.x1]
    body = remove_heel(body, g.heel_y - g.cuff_y,
                       (g.heel_rows[0] - g.cuff_y, g.heel_rows[1] - g.cuff_y))
    prop_h = target_w * body.shape[0] / body.shape[1]
    if target_h is None:
        if known_heights is None:
            from .product_specs import KNOWN_HEIGHTS as known_heights  # noqa: N811
        target_h = snap_height(prop_h, known_heights)
    if palette is None:
        palette = detect_swatches(a)
    if len(palette) < 1:
        raise RuntimeError("No swatch colours found in the left column.")
    idx = quantize_downsample(body, palette, target_w, target_h)
    idx, pal = compact_palette(idx, palette)
    return {
        "indices": idx,
        "palette": pal,
        "swatches": list(palette),
        "geometry": g,
        "design": body,
        "target_w": target_w,
        "target_h": target_h,
        "proportional_height": prop_h,
    }


def process_image(src_path: str, out_dir: str, target_w: int = 168,
                  target_h: int | None = None, page: int = 0) -> dict:
    """Same outputs as core.process_full_pdf, from an image (or a raster PDF page):
    <base>.bmp, <base>_design.png, <base>_colors.png, <base>_palette.json."""
    import json
    import os

    from .core import save_paletted_bmp

    base = os.path.splitext(os.path.basename(src_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    a = load_image(src_path, page=page, dpi=300)
    r = extract_bitmap(a, target_w=target_w, target_h=target_h)
    idx, pal = r["indices"], r["palette"]

    bmp_path = os.path.join(out_dir, f"{base}.bmp")
    save_paletted_bmp(idx, pal, bmp_path)
    design_path = os.path.join(out_dir, f"{base}_design.png")
    Image.fromarray(r["design"]).resize((target_w, r["target_h"]), Image.LANCZOS).save(design_path)
    colors_path = os.path.join(out_dir, f"{base}_colors.png")
    Image.fromarray(a[:, : int(0.13 * a.shape[1])]).save(colors_path)
    palette_path = os.path.join(out_dir, f"{base}_palette.json")
    with open(palette_path, "w", encoding="utf-8") as f:
        json.dump({"palette": [list(c) for c in pal]}, f, indent=2)
    return {
        "pdf": src_path,
        "bmp_path": bmp_path,
        "design_path": design_path,
        "colors_path": colors_path,
        "palette_path": palette_path,
        "palette": pal,
        "target_w": target_w,
        "target_h": r["target_h"],
        "proportional_height": r["proportional_height"],
        "method": "raster",
        "notes": r["geometry"].notes,
    }
