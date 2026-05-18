"""
Sock mockup extraction from Custom Sock Lab PDFs: vector-based FLAT VIEW crop,
heel-guide removal, CIELAB paletted BMP output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# ---- Configuration ----
RENDER_DPI = 300
TARGET_W, TARGET_H = 168, 402
COLOR_CROP_PADDING = 14
SUPERSAMPLE_FACTOR = 4


# ---------------- helpers ----------------


@dataclass
class TextSpan:
    text: str
    bbox: fitz.Rect


def get_text_spans(page) -> list[TextSpan]:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    spans.append(TextSpan(t, fitz.Rect(span["bbox"])))
    return spans


def find_span(spans, target_text):
    for s in spans:
        if s.text.upper() == target_text.upper():
            return s
    return None


def find_flat_view_rect(page, spans):
    """Return the Rect of the FLAT VIEW sock design body (stroke-only rectangle)."""
    flat_label = find_span(spans, "FLAT VIEW")
    if flat_label is None:
        raise RuntimeError("Could not find 'FLAT VIEW' label on page.")
    label_cx = (flat_label.bbox.x0 + flat_label.bbox.x1) / 2

    best = None
    best_dx = float("inf")
    for d in page.get_drawings():
        if d.get("type") != "s":
            continue
        items = d.get("items") or []
        if len(items) != 1:
            continue
        if items[0][0] != "re":
            continue
        rect = d.get("rect")
        if rect is None or rect.height < 200 or rect.width < 80:
            continue
        cx = (rect.x0 + rect.x1) / 2
        dx = abs(cx - label_cx)
        if dx < best_dx:
            best_dx = dx
            best = rect
    if best is None:
        raise RuntimeError("Could not locate the FLAT VIEW sock body rectangle.")
    return fitz.Rect(best)


def find_heel_shapes(page, flat_rect):
    """Find heel oval halves + heel guide line; return list of fitz.Rect in PDF points."""
    heel_rects = []
    mid_y = (flat_rect.y0 + flat_rect.y1) / 2

    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue
        dtype = d.get("type")

        if dtype == "s" and rect.height < 4:
            ovl = min(rect.x1, flat_rect.x1) - max(rect.x0, flat_rect.x0)
            if ovl > flat_rect.width * 0.5:
                cy = (rect.y0 + rect.y1) / 2
                if abs(cy - mid_y) < 0.20 * flat_rect.height:
                    heel_rects.append(fitz.Rect(rect))
            continue

        if dtype not in ("f", "fs"):
            continue
        if not (rect.y0 < mid_y < rect.y1):
            continue
        rel_w = rect.width / flat_rect.width
        rel_h = rect.height / flat_rect.height
        if not (0.15 < rel_w < 0.40):
            continue
        if not (0.04 < rel_h < 0.12):
            continue
        tol = 2
        touches_left = abs(rect.x0 - flat_rect.x0) < tol and rect.x1 < flat_rect.x1 - 10
        touches_right = abs(rect.x1 - flat_rect.x1) < tol and rect.x0 > flat_rect.x0 + 10
        if touches_left or touches_right:
            heel_rects.append(fitz.Rect(rect))
    return heel_rects


def find_color_column_crop(page, spans, page_rect):
    """Tight Rect for the left color column (labels + circles)."""
    LEFT_LIMIT = 210

    circle_rects = []
    for d in page.get_drawings():
        if d.get("type") not in ("f", "fs"):
            continue
        rect = d.get("rect")
        if rect is None:
            continue
        if rect.x1 > LEFT_LIMIT:
            continue
        if 30 <= rect.width <= 90 and 30 <= rect.height <= 90:
            ratio = rect.width / rect.height
            if 0.85 <= ratio <= 1.15:
                circle_rects.append(rect)

    if not circle_rects:
        raise RuntimeError("Could not detect color circles in left column.")

    label_rects = []
    for s in spans:
        if s.bbox.x1 > LEFT_LIMIT:
            continue
        if s.bbox.y0 < 90:
            continue
        if s.text.strip() == "":
            continue
        label_rects.append(s.bbox)

    all_rects = circle_rects + label_rects
    x0 = min(r.x0 for r in all_rects)
    y0 = min(r.y0 for r in all_rects)
    x1 = max(r.x1 for r in all_rects)
    y1 = max(r.y1 for r in all_rects)

    p = COLOR_CROP_PADDING
    crop = fitz.Rect(
        max(0, x0 - p),
        max(0, y0 - p),
        min(page_rect.x1, x1 + p),
        min(page_rect.y1, y1 + p),
    )
    return crop


def process_pdf(pdf_path: str, out_dir: str, dpi: int = RENDER_DPI) -> dict:
    """Crop FLAT VIEW (PNG), color column (PNG); heel artifacts patched."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[0]
    page_rect = page.rect
    spans = get_text_spans(page)

    flat_rect = find_flat_view_rect(page, spans)
    heel_rects = find_heel_shapes(page, flat_rect)
    colors_rect = find_color_column_crop(page, spans, page_rect)

    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_full = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def to_px(rect: fitz.Rect):
        return (
            int(round(rect.x0 * scale)),
            int(round(rect.y0 * scale)),
            int(round(rect.x1 * scale)),
            int(round(rect.y1 * scale)),
        )

    fx0, fy0, fx1, fy1 = to_px(flat_rect)
    design_img = img_full.crop((fx0, fy0, fx1, fy1)).copy()

    arr = np.array(design_img)
    h, w = arr.shape[:2]

    stroke_widths = {}
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None:
            stroke_widths[
                (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
            ] = d.get("width", 0) or 0

    def to_design_px(hr, sw_pdf):
        sw_half_px = max(2, int(round((sw_pdf / 2.0) * scale)) + 2)
        cx0 = max(hr.x0, flat_rect.x0)
        cy0 = max(hr.y0, flat_rect.y0)
        cx1 = min(hr.x1, flat_rect.x1)
        cy1 = min(hr.y1, flat_rect.y1)
        rx0 = int(round((cx0 - flat_rect.x0) * scale)) - sw_half_px
        ry0 = int(round((cy0 - flat_rect.y0) * scale)) - sw_half_px
        rx1 = int(round((cx1 - flat_rect.x0) * scale)) + sw_half_px
        ry1 = int(round((cy1 - flat_rect.y0) * scale)) + sw_half_px
        rx0 = max(0, rx0)
        ry0 = max(0, ry0)
        rx1 = min(w, rx1)
        ry1 = min(h, ry1)
        return rx0, ry0, rx1, ry1

    if heel_rects:
        heel_top_pdf = min(r.y0 for r in heel_rects)
        heel_top_px = max(0, int(round((heel_top_pdf - flat_rect.y0) * scale)) - 4)
        sample_y0 = max(0, heel_top_px - 12)
        sample_y1 = heel_top_px
    else:
        sample_y0, sample_y1 = int(h * 0.45), int(h * 0.50)
    strip_w = max(2, int(w * 0.04))
    if sample_y1 > sample_y0:
        edge_pixels = np.vstack(
            [
                arr[sample_y0:sample_y1, 0:strip_w].reshape(-1, 3),
                arr[sample_y0:sample_y1, w - strip_w : w].reshape(-1, 3),
            ]
        )
        body_color = tuple(int(c) for c in np.median(edge_pixels, axis=0))
    else:
        body_color = (128, 128, 128)

    def patch_with_source(target, source):
        tx0, ty0, tx1, ty1 = target
        sx0, sy0, sx1, sy1 = source
        th, tw = ty1 - ty0, tx1 - tx0
        sh, sw_ = sy1 - sy0, sx1 - sx0
        if th <= 0 or tw <= 0 or sh <= 0 or sw_ <= 0:
            return
        src = arr[sy0:sy1, sx0:sx1]
        if sh < th:
            src = np.tile(src, (int(np.ceil(th / sh)), 1, 1))[:th]
        else:
            src = src[:th]
        if sw_ < tw:
            src = np.tile(src, (1, int(np.ceil(tw / sw_)), 1))[:, :tw]
        else:
            src = src[:, :tw]
        arr[ty0:ty1, tx0:tx1] = src

    for hr in heel_rects:
        sw = stroke_widths.get(
            (round(hr.x0, 2), round(hr.y0, 2), round(hr.x1, 2), round(hr.y1, 2)), 0
        )
        rx0, ry0, rx1, ry1 = to_design_px(hr, sw)
        if rx1 <= rx0 or ry1 <= ry0:
            continue

        is_line = hr.height < 4
        touches_left = abs(hr.x0 - flat_rect.x0) < 2
        touches_right = abs(hr.x1 - flat_rect.x1) < 2

        if is_line:
            band_h = ry1 - ry0
            src_y1 = max(0, ry0 - 4)
            src_y0 = max(0, src_y1 - band_h)
            if src_y1 - src_y0 < band_h:
                src_y0 = min(h, ry1 + 4)
                src_y1 = min(h, src_y0 + band_h)
            if src_y1 - src_y0 > 0:
                patch_with_source((rx0, ry0, rx1, ry1), (rx0, src_y0, rx1, src_y1))
            else:
                arr[ry0:ry1, rx0:rx1] = body_color
        elif touches_left:
            src_x0 = min(w - 1, rx1 + 4)
            src_x1 = min(w, src_x0 + 8)
            if src_x1 - src_x0 > 0:
                patch_with_source((rx0, ry0, rx1, ry1), (src_x0, ry0, src_x1, ry1))
            else:
                arr[ry0:ry1, rx0:rx1] = body_color
        elif touches_right:
            src_x1 = max(1, rx0 - 4)
            src_x0 = max(0, src_x1 - 8)
            if src_x1 - src_x0 > 0:
                patch_with_source((rx0, ry0, rx1, ry1), (src_x0, ry0, src_x1, ry1))
            else:
                arr[ry0:ry1, rx0:rx1] = body_color
        else:
            arr[ry0:ry1, rx0:rx1] = body_color

    design_img = Image.fromarray(arr)

    design_final = design_img.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    design_path = os.path.join(out_dir, f"{base}_design.png")
    design_final.save(design_path, "PNG")

    cx0, cy0, cx1, cy1 = to_px(colors_rect)
    colors_img = img_full.crop((cx0, cy0, cx1, cy1))
    colors_path = os.path.join(out_dir, f"{base}_colors.png")
    colors_img.save(colors_path, "PNG")

    return {
        "pdf": pdf_path,
        "design_path": design_path,
        "colors_path": colors_path,
        "flat_rect_pts": tuple(flat_rect),
        "colors_rect_pts": tuple(colors_rect),
        "heel_rects_pts": [tuple(r) for r in heel_rects],
        "body_color_rgb": body_color,
    }


# ---------- color science ----------


def rgb_to_lab(rgb):
    """sRGB (0-255) -> CIELAB."""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    if rgb_lin.ndim == 1:
        xyz = M @ rgb_lin
    else:
        xyz = rgb_lin @ M.T
    ref = np.array([0.95047, 1.0, 1.08883])
    xyz_n = xyz / ref
    eps = (6 / 29) ** 3
    f = np.where(
        xyz_n > eps,
        xyz_n ** (1 / 3),
        (xyz_n / (3 * (6 / 29) ** 2)) + 4 / 29,
    )
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b_ = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b_], axis=-1)


def extract_palette_from_pdf(pdf_path: str) -> list[tuple[int, int, int]]:
    """Swatch RGB triples in top-to-bottom column order."""
    LEFT_LIMIT = 210
    doc = fitz.open(pdf_path)
    page = doc[0]
    swatches = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue
        if rect.x1 > LEFT_LIMIT:
            continue
        if not (30 <= rect.width <= 90 and 30 <= rect.height <= 90):
            continue
        ratio = rect.width / rect.height
        if not (0.85 <= ratio <= 1.15):
            continue

        dtype = d.get("type")
        if dtype in ("f", "fs"):
            fill = d.get("fill")
            if fill:
                rgb = tuple(int(round(c * 255)) for c in fill[:3])
                swatches.append((rect.y0, rgb))
        elif dtype == "s":
            swatches.append((rect.y0, (255, 255, 255)))

    swatches.sort()
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for _, rgb in swatches:
        if rgb not in seen:
            seen.add(rgb)
            out.append(rgb)
    return out


def render_clean_design(pdf_path: str, dpi: int) -> tuple[np.ndarray, fitz.Rect]:
    """Render FLAT VIEW clip at dpi with heel guides patched out."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    spans = get_text_spans(page)
    flat_rect = find_flat_view_rect(page, spans)
    heel_rects = find_heel_shapes(page, flat_rect)

    stroke_w = 0.0
    for d in page.get_drawings():
        if d.get("type") != "s":
            continue
        items = d.get("items") or []
        if len(items) != 1 or items[0][0] != "re":
            continue
        r = d.get("rect")
        if r is None:
            continue
        if (
            abs(r.x0 - flat_rect.x0) < 0.5
            and abs(r.y0 - flat_rect.y0) < 0.5
            and abs(r.x1 - flat_rect.x1) < 0.5
            and abs(r.y1 - flat_rect.y1) < 0.5
        ):
            stroke_w = d.get("width") or 0.0
            break
    inset = stroke_w / 3.0
    render_rect = fitz.Rect(
        flat_rect.x0 + inset,
        flat_rect.y0 + inset,
        flat_rect.x1 - inset,
        flat_rect.y1 - inset,
    )

    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=render_rect, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    h, w = arr.shape[:2]

    stroke_widths = {}
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None:
            stroke_widths[
                (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
            ] = d.get("width", 0) or 0

    def to_design_px(hr, sw_pdf):
        sw_half_px = max(2, int(round((sw_pdf / 2.0) * scale)) + 2)
        cx0 = max(hr.x0, render_rect.x0)
        cy0 = max(hr.y0, render_rect.y0)
        cx1 = min(hr.x1, render_rect.x1)
        cy1 = min(hr.y1, render_rect.y1)
        rx0 = int(round((cx0 - render_rect.x0) * scale)) - sw_half_px
        ry0 = int(round((cy0 - render_rect.y0) * scale)) - sw_half_px
        rx1 = int(round((cx1 - render_rect.x0) * scale)) + sw_half_px
        ry1 = int(round((cy1 - render_rect.y0) * scale)) + sw_half_px
        return max(0, rx0), max(0, ry0), min(w, rx1), min(h, ry1)

    if heel_rects:
        heel_top_pdf = min(r.y0 for r in heel_rects)
        heel_top_px = max(0, int(round((heel_top_pdf - render_rect.y0) * scale)) - 4)
        sample_y0 = max(0, heel_top_px - 12)
        sample_y1 = heel_top_px
    else:
        sample_y0, sample_y1 = int(h * 0.45), int(h * 0.50)
    strip_w = max(2, int(w * 0.04))
    if sample_y1 > sample_y0:
        edge_pixels = np.vstack(
            [
                arr[sample_y0:sample_y1, 0:strip_w].reshape(-1, 3),
                arr[sample_y0:sample_y1, w - strip_w : w].reshape(-1, 3),
            ]
        )
        body_color = tuple(int(c) for c in np.median(edge_pixels, axis=0))
    else:
        body_color = (128, 128, 128)

    def patch_with_source(target, source):
        tx0, ty0, tx1, ty1 = target
        sx0, sy0, sx1, sy1 = source
        th, tw = ty1 - ty0, tx1 - tx0
        sh, sw_ = sy1 - sy0, sx1 - sx0
        if th <= 0 or tw <= 0 or sh <= 0 or sw_ <= 0:
            return
        src = arr[sy0:sy1, sx0:sx1]
        if sh < th:
            src = np.tile(src, (int(np.ceil(th / sh)), 1, 1))[:th]
        else:
            src = src[:th]
        if sw_ < tw:
            src = np.tile(src, (1, int(np.ceil(tw / sw_)), 1))[:, :tw]
        else:
            src = src[:, :tw]
        arr[ty0:ty1, tx0:tx1] = src

    for hr in heel_rects:
        sw = stroke_widths.get(
            (round(hr.x0, 2), round(hr.y0, 2), round(hr.x1, 2), round(hr.y1, 2)), 0
        )
        rx0, ry0, rx1, ry1 = to_design_px(hr, sw)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        is_line = hr.height < 4
        touches_left = abs(hr.x0 - flat_rect.x0) < 2
        touches_right = abs(hr.x1 - flat_rect.x1) < 2
        if is_line:
            band_h = ry1 - ry0
            src_y1 = max(0, ry0 - 4)
            src_y0 = max(0, src_y1 - band_h)
            if src_y1 - src_y0 < band_h:
                src_y0 = min(h, ry1 + 4)
                src_y1 = min(h, src_y0 + band_h)
            if src_y1 - src_y0 > 0:
                patch_with_source((rx0, ry0, rx1, ry1), (rx0, src_y0, rx1, src_y1))
            else:
                arr[ry0:ry1, rx0:rx1] = body_color
        elif touches_left:
            src_x0 = min(w - 1, rx1 + 4)
            src_x1 = min(w, src_x0 + 8)
            if src_x1 - src_x0 > 0:
                patch_with_source((rx0, ry0, rx1, ry1), (src_x0, ry0, src_x1, ry1))
            else:
                arr[ry0:ry1, rx0:rx1] = body_color
        elif touches_right:
            src_x1 = max(1, rx0 - 4)
            src_x0 = max(0, src_x1 - 8)
            if src_x1 - src_x0 > 0:
                patch_with_source((rx0, ry0, rx1, ry1), (src_x0, ry0, src_x1, ry1))
            else:
                arr[ry0:ry1, rx0:rx1] = body_color
        else:
            arr[ry0:ry1, rx0:rx1] = body_color

    return arr, flat_rect


def snap_and_downsample(
    hi_arr: np.ndarray,
    palette: list,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """LAB nearest-color snap + mode pool to palette indices."""
    H, W, _ = hi_arr.shape
    P = len(palette)
    pal_lab = rgb_to_lab(np.array(palette))

    nearest = np.empty((H, W), dtype=np.int8)
    chunk_rows = max(1, 4_000_000 // (W * P))
    for y0 in range(0, H, chunk_rows):
        y1 = min(H, y0 + chunk_rows)
        block = hi_arr[y0:y1].reshape(-1, 3)
        block_lab = rgb_to_lab(block)
        diff = block_lab[:, None, :] - pal_lab[None, :, :]
        dist = (diff * diff).sum(axis=2)
        nearest[y0:y1] = np.argmin(dist, axis=1).reshape(y1 - y0, W).astype(np.int8)

    sy = H / target_h
    sx = W / target_w
    result = np.zeros((target_h, target_w), dtype=np.uint8)
    for ty in range(target_h):
        y0 = int(ty * sy)
        y1 = max(y0 + 1, int((ty + 1) * sy))
        for tx in range(target_w):
            x0 = int(tx * sx)
            x1 = max(x0 + 1, int((tx + 1) * sx))
            block = nearest[y0:y1, x0:x1]
            counts = np.bincount(block.flatten(), minlength=P)
            result[ty, tx] = np.argmax(counts)
    return result


def to_paletted_image(indices: np.ndarray, palette: list) -> Image.Image:
    H, W = indices.shape
    img = Image.new("P", (W, H))
    flat: list[int] = []
    for c in palette:
        flat.extend(c)
    flat += [0] * (768 - len(flat))
    img.putpalette(flat)
    img.putdata(indices.flatten().tolist())
    return img


def convert_pdf_to_bmp(
    pdf_path: str,
    out_dir: str,
    dpi: int | None = None,
    save_intermediates: bool = False,
) -> dict:
    """Produce 168×402 paletted BMP + metadata."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    if dpi is None:
        doc = fitz.open(pdf_path)
        page = doc[0]
        spans = get_text_spans(page)
        flat_rect = find_flat_view_rect(page, spans)
        native_dpi = TARGET_W / flat_rect.width * 72.0
        dpi = int(round(native_dpi * SUPERSAMPLE_FACTOR))

    hi_arr, flat_rect = render_clean_design(pdf_path, dpi=dpi)

    palette = extract_palette_from_pdf(pdf_path)
    if not palette:
        raise RuntimeError("No swatch colors found in PDF.")

    indices = snap_and_downsample(hi_arr, palette, TARGET_W, TARGET_H)
    pal_img = to_paletted_image(indices, palette)

    bmp_path = os.path.join(out_dir, f"{base}.bmp")
    pal_img.save(bmp_path)

    palette_path = os.path.join(out_dir, f"{base}_palette.json")
    with open(palette_path, "w", encoding="utf-8") as f:
        json.dump({"palette": [list(c) for c in palette]}, f, indent=2)

    info = {
        "pdf": pdf_path,
        "bmp_path": bmp_path,
        "palette_path": palette_path,
        "palette": palette,
        "flat_rect_pts": tuple(flat_rect),
        "render_dpi": dpi,
    }

    if save_intermediates:
        preview_path = os.path.join(out_dir, f"{base}_preview.png")
        pal_img.convert("RGB").save(preview_path)
        info["preview_path"] = preview_path

    return info


def process_full_pdf(pdf_path: str, out_dir: str) -> dict:
    """Run preview extraction + BMP conversion; merged result dict."""
    preview = process_pdf(pdf_path, out_dir)
    bmp_info = convert_pdf_to_bmp(pdf_path, out_dir)
    preview.update(bmp_info)
    return preview
