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

from .product_specs import (
    DEFAULT_SPEC,
    KNOWN_HEIGHTS,
    ProductSpec,
    _PREF_TYPE,
    candidate_specs,
    detect_material,
    detect_style_length,
    resolve_spec,
    snap_height,
)

# ---- Configuration ----
RENDER_DPI = 300
TARGET_W, TARGET_H = 168, 402
COLOR_CROP_PADDING = 14
SUPERSAMPLE_FACTOR = 6   # 6 supersamples per output px: uniformly >= 4x on every
# validated design (Coffman +0.7, Rice +0.8, others flat), better fine-detail edges.
# Revert to 4 for ~2.25x faster rendering if throughput matters more than crispness.


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
    best_key = (float("inf"), 0.0)
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
        # Pick the rect nearest the label centre; on a tie (concentric rectangles,
        # e.g. an inner panel inside the full sock outline) take the TALLEST, which
        # is the full sock body — otherwise a short inner panel gets scaled wrong.
        key = (round(dx, 0), -rect.height)
        if key < best_key:
            best_key = key
            best = rect
    if best is None:
        raise RuntimeError("Could not locate the FLAT VIEW sock body rectangle.")
    return fitz.Rect(best)


def find_heel_shapes(page, flat_rect):
    """Find heel oval halves + heel guide line; return list of fitz.Rect in PDF points.

    The heel break sits at different heights per sock length (near the top on an
    ankle, ~50% on a crew, ~68% on a knee-high), so shapes are located at ANY y
    and then filtered to a band around the detected heel row.

    Oval detection is deliberately permissive because the ovals are drawn
    differently across templates:
      * filled ("f"/"fs") OR stroke-only ("s") ellipses,
      * sized relative to the flat WIDTH (constant 168-px mapping) rather than the
        height, so a short ankle sock isn't rejected for having a "tall" oval.
    The reliable signal is geometry: a wide-ish shape attached to the left or
    right edge at the heel row. The raster pass in the render step (see
    heel_band_from_raster) is the safety net that catches anything missed here.
    """
    flat_w = flat_rect.width
    flat_h = flat_rect.height
    tol = 2.5

    oval_cands = []
    line_cands = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue
        dtype = d.get("type")

        # Heel guide line: thin, near-horizontal, spanning most of the width. The
        # real heel line OVERHANGS the sock (pokes out past both edges), which
        # uniquely distinguishes it from the cuff/toe bands that stop at the edge.
        span = min(rect.x1, flat_rect.x1) - max(rect.x0, flat_rect.x0)
        if rect.height < 4 and span > flat_w * 0.5:
            overhang = (rect.x0 < flat_rect.x0 - 3) or (rect.x1 > flat_rect.x1 + 3)
            line_cands.append((fitz.Rect(rect), overhang, span))
            continue

        # Heel oval (filled or stroked), edge-attached, sized relative to WIDTH.
        if dtype not in ("f", "fs", "s"):
            continue
        # A heel oval is an ellipse — its path contains Bézier curves. Pattern
        # shapes that merely look oval-ish in their bounding box (argyle diamonds,
        # chevrons, triangles) are built from straight segments only. Requiring a
        # curve keeps the body pattern from being mistaken for a heel oval and then
        # erased (e.g. Rice's blue argyle diamonds touch the edge at the heel row).
        items = d.get("items") or []
        if not any(it[0] == "c" for it in items):
            continue
        rel_w = rect.width / flat_w
        rel_hw = rect.height / flat_w  # height relative to width -> length-independent
        if not (0.10 < rel_w < 0.55):
            continue
        if not (0.06 < rel_hw < 0.60):
            continue
        touches_left = abs(rect.x0 - flat_rect.x0) < tol and rect.x1 < flat_rect.x1 - 8
        touches_right = abs(rect.x1 - flat_rect.x1) < tol and rect.x0 > flat_rect.x0 + 8
        if touches_left or touches_right:
            oval_cands.append(fitz.Rect(rect))

    # Anchor the heel row. The heel LINE is the most reliable anchor: prefer the
    # one that overhangs the sock, then the widest. Only if there's no line do we
    # fall back to the ovals — and even then we must avoid the toe scallops, which
    # also look like edge ovals. (Anchoring on the median oval centre is wrong:
    # there can be more toe scallops than heel ovals, dragging the anchor down.)
    if line_cands:
        line_cands.sort(key=lambda t: (t[1], t[2]), reverse=True)  # overhang, then span
        anchor_line = line_cands[0][0]
        heel_y = (anchor_line.y0 + anchor_line.y1) / 2
    elif oval_cands:
        # Without a line, take the cluster of ovals nearest the geometric heel
        # position for this length (ankle≈top, crew≈middle), not the median.
        approx = flat_rect.y0 + 0.45 * flat_h
        centers = sorted(oval_cands, key=lambda o: abs((o.y0 + o.y1) / 2 - approx))
        heel_y = (centers[0].y0 + centers[0].y1) / 2
    else:
        return []

    band = 0.12 * flat_w  # group only shapes near the heel row
    near = [o for o in oval_cands if abs((o.y0 + o.y1) / 2 - heel_y) <= band]
    # dedupe ovals that came from multiple overlapping draw paths
    seen, heel_rects = set(), []
    for o in near:
        key = (round(o.x0 / 4), round(o.y0 / 4), round(o.x1 / 4), round(o.y1 / 4))
        if key not in seen:
            seen.add(key)
            heel_rects.append(o)
    # Add ONLY the heel guide line (the one that overhangs the sock). Full-width
    # but NON-overhanging thin strokes are pattern elements (e.g. the horizontal
    # bands in a Navajo print), not the heel line — including them would heal and
    # damage the artwork.
    for (l, overhang, _) in line_cands:
        if overhang and abs((l.y0 + l.y1) / 2 - heel_y) <= band:
            heel_rects.append(l)
    return heel_rects


def heel_band_from_raster(arr, heel_y_px, body_color, search_frac=0.18):
    """Measure the heel ovals' vertical extent directly from the rendered image.

    The ovals hug the left/right edges at the heel row; this finds the run of
    edge rows around heel_y whose colour differs from the body. It works no matter
    how the ovals were drawn in the PDF (fill, stroke, any colour), which is the
    robust way to catch ovals the vector pass misses. Returns (top, bot) pixel
    rows, or None if no clear edge band is found.
    """
    H, W = arr.shape[:2]
    e = max(3, int(0.05 * W))
    bg = np.array(body_color, dtype=int)
    left = arr[:, :e].reshape(H, -1, 3).astype(int)
    right = arr[:, W - e:].reshape(H, -1, 3).astype(int)
    nb_left = np.abs(left - bg).sum(axis=2).min(axis=1) > 45
    nb_right = np.abs(right - bg).sum(axis=2).min(axis=1) > 45
    nb = nb_left | nb_right

    y = int(np.clip(heel_y_px, 1, H - 2))
    if not nb[y]:
        rng = int(search_frac * H)
        cand = [yy for yy in range(max(0, y - rng), min(H, y + rng)) if nb[yy]]
        if not cand:
            return None
        y = min(cand, key=lambda c: abs(c - heel_y_px))
    top = y
    while top > 0 and nb[top - 1]:
        top -= 1
    bot = y
    while bot < H - 1 and nb[bot + 1]:
        bot += 1
    if bot - top > 0.45 * H:  # merged with edge pattern -> unreliable
        return None
    return top, bot


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


def heal_region(arr, y0, y1, x0, x1, body_color, var_thresh=40):
    """Remove a heel artifact (oval or guide line) from a rendered design array by
    refilling rows [y0:y1] x cols [x0:x1] so the surrounding pattern continues.

    Two regimes, chosen automatically:
      * Near-solid neighbourhood -> fill with body_color (perfect for solid bodies;
        letters never sit in a solid heel zone, so nothing is harmed).
      * Patterned neighbourhood  -> copy a same-size block from the vertical offset
        whose top/bottom edges best continue the rows bordering the hole
        (seam-minimising). This locks onto the artwork's true vertical period
        (e.g. logo/text rows), so repeating patterns line up and letters stay
        intact — instead of smearing a few adjacent pixels as the old patch did.

    Decided at run time from the image alone (no template metadata needed).
    """
    H, W = arr.shape[:2]
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if y1 <= y0 or x1 <= x0:
        return
    cols = slice(x0, x1)
    h = y1 - y0

    # Decide solid-vs-patterned from the BORDER rows just OUTSIDE the hole, not the
    # hole itself (which still holds the oval/line and would always look "busy").
    # If the border is dominated by one colour (allowing a few stray pattern
    # pixels), fill with that background colour. This is the common case and
    # exactly the "replace the oval with the bg colour" behaviour we want, and it
    # avoids seam-copying a pattern block (which would duplicate a motif).
    top_b = arr[max(0, y0 - 4):y0, cols]
    bot_b = arr[y1:min(H, y1 + 4), cols]
    parts = [b.reshape(-1, 3) for b in (top_b, bot_b) if b.size]
    border = np.vstack(parts) if parts else np.empty((0, 3), np.uint8)
    if border.shape[0]:
        q = (border // 8 * 8)
        uniq, cnt = np.unique(q, axis=0, return_counts=True)
        mode = uniq[cnt.argmax()].astype(np.int16)
        frac = float(np.mean(np.abs(border.astype(np.int16) - mode).sum(1) < 24))
        if frac > 0.6:
            arr[y0:y1, cols] = mode.astype(np.uint8)
            return

    # Patterned surround (e.g. Honorlock penguins, Rice argyle): continue the
    # pattern by copying a same-size block from the vertical offset whose borders
    # best continue the rows bordering the hole. A small penalty on offset
    # distance keeps it continuing the LOCAL pattern instead of grabbing a distant
    # matching region (e.g. a logo near the cuff).
    search = min(H // 2, max(48, h * 8))
    top_ref = arr[y0 - 1, cols].astype(np.int16) if y0 > 0 else None
    bot_ref = arr[y1, cols].astype(np.int16) if y1 < H else None

    best, best_cost = None, float("inf")
    offsets = list(range(h, search)) + list(range(-search, -h + 1))
    for off in offsets:
        sy0, sy1 = y0 + off, y1 + off
        if sy0 < 0 or sy1 > H:
            continue
        if not (sy1 <= y0 or sy0 >= y1):  # source must not overlap the hole
            continue
        cost, n = 0.0, 0
        if top_ref is not None and sy0 > 0:
            cost += float(np.abs(top_ref - arr[sy0 - 1, cols].astype(np.int16)).mean())
            n += 1
        if bot_ref is not None and sy1 < H:
            cost += float(np.abs(bot_ref - arr[sy1, cols].astype(np.int16)).mean())
            n += 1
        if n == 0:
            continue
        cost = cost / n + 0.20 * (abs(off) / max(1, h))  # prefer local offsets
        if cost < best_cost:
            best_cost, best = cost, (sy0, sy1)

    if best is not None:
        arr[y0:y1, cols] = arr[best[0]:best[1], cols]
    else:
        arr[y0:y1, cols] = body_color


def _dominant_body_color(arr, heel_band_px=None):
    """Most common colour along the left+right edge columns (excluding the heel
    band). Using the mode over the whole height — rather than a median of a small
    strip — keeps a stray edge element (e.g. a toothbrush poking to the edge) from
    being mistaken for the body, which otherwise breaks the heel fill."""
    H, W = arr.shape[:2]
    e = max(3, int(0.05 * W))
    rows = np.ones(H, dtype=bool)
    if heel_band_px:
        t, b = heel_band_px
        rows[max(0, int(t)):min(H, int(b))] = False
    samp = np.vstack([arr[rows, :e].reshape(-1, 3), arr[rows, W - e:].reshape(-1, 3)])
    if samp.size == 0:
        return (128, 128, 128)
    q = (samp // 8 * 8)
    cols, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(c) for c in cols[counts.argmax()])


def _heal_line(arr, y0, y1, x0, x1, body_color):
    """Heal the thin heel guide line by CONTINUING the surrounding image into it.

    The line is only a few pixels tall and interrupts whatever pattern runs
    through it (solid body, stripes, scattered icons, logo text). We rebuild it by
    extending the rows just ABOVE downward over the top half and the rows just
    BELOW upward over the bottom half, so both sides grow inward and meet in the
    middle. Copying (never blending) keeps the exact palette colours, so a white
    background stays white and an icon edge continues seamlessly — no flat streak.
    """
    H, W = arr.shape[:2]
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if y1 <= y0 or x1 <= x0:
        return
    cols = slice(x0, x1)
    h = y1 - y0
    mid = (h + 1) // 2  # rows filled from above; the rest from below
    above, below = y0, H - y1
    did_top = did_bot = False
    if above >= mid:
        arr[y0:y0 + mid, cols] = arr[y0 - mid:y0, cols]
        did_top = True
    if below >= (h - mid):
        arr[y0 + mid:y1, cols] = arr[y1:y1 + (h - mid), cols]
        did_bot = True
    if not did_top and did_bot and below >= h:          # no room above: all from below
        arr[y0:y1, cols] = arr[y1:y1 + h, cols]
    elif not did_bot and did_top and above >= h:        # no room below: all from above
        arr[y0:y1, cols] = arr[y0 - h:y0, cols]
    elif not did_top and not did_bot:                   # nowhere to borrow: solid
        arr[y0:y1, cols] = np.array(body_color, np.uint8)


def _aware_block(arr, y0, y1, cols, search=None, cost_cap=60.0):
    """Find a same-height source block that continues the pattern through rows
    [y0:y1] (within `cols`), and return it as a FULL-WIDTH slice so a caller can
    mask it into a lens shape.

    The block is taken from the vertical offset whose top/bottom border rows best
    match the rows just outside the hole — i.e. an integer multiple of the
    artwork's vertical period — so a repeating motif (argyle diamonds, scattered
    logos) lines up *in phase* instead of being stamped over with a flat colour.
    A small penalty on offset distance prefers continuing the LOCAL pattern over
    grabbing a distant lookalike. Returns None if nothing continues cleanly
    (border cost above `cost_cap`), letting the caller fall back to a flat fill.
    """
    H, W = arr.shape[:2]
    h = y1 - y0
    if h <= 0:
        return None
    if search is None:
        search = min(H // 2, max(64, h * 10))
    top_ref = arr[y0 - 1, cols].astype(np.int16) if y0 > 0 else None
    bot_ref = arr[y1, cols].astype(np.int16) if y1 < H else None
    if top_ref is None and bot_ref is None:
        return None
    best, best_cost = None, float("inf")
    offsets = list(range(h, search)) + list(range(-search, -h + 1))
    for off in offsets:
        sy0, sy1 = y0 + off, y1 + off
        if sy0 < 0 or sy1 > H:
            continue
        if not (sy1 <= y0 or sy0 >= y1):  # source must not overlap the hole
            continue
        cost, n = 0.0, 0
        if top_ref is not None and sy0 > 0:
            cost += float(np.abs(top_ref - arr[sy0 - 1, cols].astype(np.int16)).mean())
            n += 1
        if bot_ref is not None and sy1 < H:
            cost += float(np.abs(bot_ref - arr[sy1, cols].astype(np.int16)).mean())
            n += 1
        if n == 0:
            continue
        cost = cost / n + 0.20 * (abs(off) / max(1, h))  # prefer local offsets
        if cost < best_cost:
            best_cost, best = cost, (sy0, sy1)
    if best is None or best_cost > cost_cap:
        return None
    return arr[best[0]:best[1], :]


def _drawing_maps(page):
    """Per-drawing stroke widths and (for filled shapes) fill colours, keyed by the
    drawing's rounded rect — used to identify heel shapes and their fill colour."""
    stroke_widths, heel_fills = {}, {}
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        key = (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
        stroke_widths[key] = d.get("width", 0) or 0
        f = d.get("fill")
        if d.get("type") in ("f", "fs") and f:
            heel_fills[key] = tuple(int(round(v * 255)) for v in f[:3])
    return stroke_widths, heel_fills


def _remove_heel(arr, heel_rects, to_design_px, stroke_widths, body_color, heel_fills=None):
    """Remove the heel guide line + semi-ovals from a rendered design array.

    Stage 1 (precise): heal every detected vector heel shape. The thin guide line
    is healed in place; each oval is padded (vertically, and out to the edge it
    hugs) so it's fully covered even if its box is a touch small or a stroke-only
    outline. heal_region continues the surrounding pattern, so padding is safe.

    Stage 2 (fallback): only if the vector pass found NO ovals (just a line, or
    nothing), measure the oval band straight from the rendered pixels and heal the
    oval's bounding reach at each edge. This catches ovals the vector pass missed
    while staying off designs whose ovals were already handled precisely — so a
    heavily patterned design isn't disturbed when it doesn't need to be.
    """
    h, w = arr.shape[:2]
    heel_centers = []
    n_ovals = 0
    # Handle ovals BEFORE the guide line: the line continues the rows around it, so
    # those rows must already be clean (oval removed) when the line is healed.
    for hr in sorted(heel_rects, key=lambda r: r.height < 4):
        sw = stroke_widths.get(
            (round(hr.x0, 2), round(hr.y0, 2), round(hr.x1, 2), round(hr.y1, 2)), 0
        )
        rx0, ry0, rx1, ry1 = to_design_px(hr, sw)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        if hr.height < 4:  # guide line — continue the surrounding image into it
            _heal_line(arr, ry0, ry1, rx0, rx1, body_color)
        else:              # oval: replace ONLY the lens shape with the LOCAL bg
            fill_rgb = (heel_fills or {}).get(
                (round(hr.x0, 2), round(hr.y0, 2), round(hr.x1, 2), round(hr.y1, 2))
            )
            # An oval whose fill colour equals the body is a flat body-coloured heel
            # patch sitting on top of the artwork. Reconstructing it with the content-
            # aware copy can stamp a wrong motif (Wayne Sanderson's camo grabbed a
            # black-chicken block); a plain body-colour fill is the correct, clean
            # result and removes the outline. Force the flat-fill path for it.
            oval_is_body = fill_rgb is not None and \
                sum(abs(a - b) for a, b in zip(fill_rgb, body_color)) <= 40
            n_ovals += 1
            cy = (ry0 + ry1) / 2.0
            half_h = max(1.0, (ry1 - ry0) / 2.0) + 1.0   # +1px for anti-alias rim
            wdt = max(1.0, float(rx1 - rx0)) + 1.0
            hugs_left = rx0 <= (w - rx1)
            x_edge = rx0 if hugs_left else rx1
            y0m, y1m = max(0, ry0 - 1), min(h, ry1 + 1)
            yy, xx = np.mgrid[y0m:y1m, 0:w]
            mask = ((xx - x_edge) / wdt) ** 2 + ((yy - cy) / half_h) ** 2 <= 1.0
            lx0 = 0 if hugs_left else max(0, int(x_edge - wdt))
            lx1 = min(w, int(x_edge + wdt)) if hugs_left else w
            cols = slice(lx0, lx1)
            # Decide solid-vs-patterned from the rows immediately bordering the lens
            # (within its own columns). `mode`/`frac` describe that local surround.
            bt = arr[max(0, y0m - 3):y0m, cols].reshape(-1, 3)
            bb = arr[y1m:min(h, y1m + 3), cols].reshape(-1, 3)
            border = np.vstack([b for b in (bt, bb) if b.size]) if (bt.size or bb.size) else np.empty((0, 3), np.uint8)
            mode = np.array(body_color, np.uint8)
            frac = 0.0
            if border.shape[0]:
                q = (border // 8 * 8)
                u, c = np.unique(q, axis=0, return_counts=True)
                mode = u[c.argmax()].astype(np.uint8)
                frac = float(np.mean(np.abs(border.astype(np.int16) - mode.astype(np.int16)).sum(1) < 24))
            if frac > 0.92 or oval_is_body:
                # Near-solid surround, or a body-coloured oval patch: a flat fill is
                # exact and cheapest. Use the body colour for the patch case (the
                # design under it is unknown; body colour is the clean choice).
                arr[y0m:y1m][mask] = np.array(body_color, np.uint8) if oval_is_body else mode
            else:
                # Mixed surround. Pick the continuation method from the LOCAL pattern's
                # orientation, measured on the wide interior region beside the lens:
                #   * horizontally banded (Coffman's guide stripe + black body, plain
                #     backgrounds) — little horizontal gradient, edges run vertically.
                #     Extend each row's interior pixel across the lens. This continues
                #     horizontal features and can NEVER pull in a distant motif — the bug
                #     where a vertical search reached the cuff logo and stamped "COFFMAN
                #     ENGINEERS" into the heel.
                #   * vertically-varying texture (Rice argyle, Honorlock penguins) —
                #     comparable horizontal and vertical gradient. Phase-matched vertical
                #     block copy via _aware_block.
                if hugs_left:
                    ref = arr[y0m:y1m, lx1:w]
                    ref_x = min(w - 1, lx1)
                else:
                    ref = arr[y0m:y1m, 0:lx0]
                    ref_x = max(0, lx0 - 1)
                banded = False
                if ref.shape[0] >= 3 and ref.shape[1] >= 3:
                    r16 = ref.astype(np.int16)
                    hg = float(np.abs(np.diff(r16, axis=1)).sum(2).mean())
                    vg = float(np.abs(np.diff(r16, axis=0)).sum(2).mean())
                    banded = hg < 5.0 and hg < 0.30 * (vg + 1e-3)
                if banded:
                    # The lens hugs an edge; fill it by reflecting the adjacent INTERIOR
                    # block horizontally into it. The patterns these ovals sit on (Navajo
                    # borders) are horizontally symmetric, so the mirror continues the 2D
                    # motif structure — instead of tiling a single interior column, which
                    # fuses short horizontal motif-lines into a solid blob at the edge.
                    rx_lo, rx_hi = (0, lx1) if hugs_left else (lx0, w)
                    width = rx_hi - rx_lo
                    if hugs_left:
                        src = arr[y0m:y1m, rx_hi:min(w, rx_hi + width)][:, ::-1]
                        sw = src.shape[1]
                        if sw:
                            arr[y0m:y1m, rx_hi - sw:rx_hi] = src
                            if sw < width:  # ran out of interior; extend nearest column
                                arr[y0m:y1m, :rx_hi - sw] = arr[y0m:y1m, rx_hi - sw:rx_hi - sw + 1]
                    else:
                        src = arr[y0m:y1m, max(0, rx_lo - width):rx_lo][:, ::-1]
                        sw = src.shape[1]
                        if sw:
                            arr[y0m:y1m, rx_lo:rx_lo + sw] = src
                            if sw < width:
                                arr[y0m:y1m, rx_lo + sw:] = arr[y0m:y1m, rx_lo + sw - 1:rx_lo + sw]
                    # The oval can dip past the band's lower white separator line into the
                    # region below, where the side edges are plain background (the motif
                    # there is centred). Don't let the mirror bleed colour below that line:
                    # find the full-width separator inside the oval and reset the side
                    # block beneath it to the body colour.
                    sep_bot = None
                    for yy in range(y0m, y1m):
                        if hugs_left:
                            inter = arr[yy, rx_hi:min(w, rx_hi + 40)]
                        else:
                            inter = arr[yy, max(0, rx_lo - 40):rx_lo]
                        if inter.shape[0] and float(np.mean(
                                np.abs(inter.astype(np.int16) - 255).sum(1) <= 60)) > 0.7:
                            sep_bot = yy
                    if sep_bot is not None and sep_bot + 1 < y1m:
                        arr[sep_bot + 1:y1m, rx_lo:rx_hi] = np.array(body_color, dtype=arr.dtype)
                else:
                    src = _aware_block(arr, y0m, y1m, cols)
                    if src is not None:
                        arr[y0m:y1m][mask] = src[mask]
                    else:
                        arr[y0m:y1m][mask] = mode
        heel_centers.append((ry0 + ry1) // 2)

    if heel_centers and n_ovals == 0:
        heel_y = int(np.median(heel_centers))
        band = heel_band_from_raster(arr, heel_y, body_color)
        if band:
            top, bot = band
            m = max(1, int(0.10 * (bot - top)))
            top, bot = max(0, top - m), min(h, bot + m)
            bg = np.array(body_color)
            maxreach = int(0.42 * w)
            sub = arr[top:bot].astype(int)
            nb = np.abs(sub - bg).sum(axis=2) > 45
            lr = rr = 0
            for row in nb:
                l = 0
                while l < maxreach and row[l]:
                    l += 1
                lr = max(lr, l)
                r = 0
                while r < maxreach and row[w - 1 - r]:
                    r += 1
                rr = max(rr, r)
            if lr > 1:
                heal_region(arr, top, bot, 0, lr, body_color)
            if rr > 1:
                heal_region(arr, top, bot, w - rr, w, body_color)


def process_pdf(
    pdf_path: str,
    out_dir: str,
    dpi: int = RENDER_DPI,
    target_w: int = TARGET_W,
    target_h: int = TARGET_H,
) -> dict:
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

    stroke_widths, heel_fills = _drawing_maps(page)

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
        ys = []
        for hr in heel_rects:
            _, ry0, _, ry1 = to_design_px(hr, 0)
            ys += [ry0, ry1]
        heel_band_px = (min(ys), max(ys)) if ys else None
    else:
        heel_band_px = None
    body_color = _dominant_body_color(arr, heel_band_px)

    _remove_heel(arr, heel_rects, to_design_px, stroke_widths, body_color, heel_fills)

    design_img = Image.fromarray(arr)

    design_final = design_img.resize((target_w, target_h), Image.LANCZOS)

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


def _clean_channels(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Snap a colour to pure black/white only when the WHOLE colour is near-pure
    (production bitmaps use clean (0,0,0)/(255,255,255), e.g. an artwork (3,4,4)
    black). Per-channel cleaning is avoided — it would corrupt colours that merely
    have one near-pure channel, e.g. a cream (248,236,210) must stay (248,...)."""
    if all(v < 8 for v in rgb):
        return (0, 0, 0)
    if all(v > 247 for v in rgb):
        return (255, 255, 255)
    return rgb


def _swatch_palette(page) -> list[tuple[int, int, int]]:
    """Fallback: the colour squares in the left column, top-to-bottom."""
    LEFT_LIMIT = 210
    swatches = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None or rect.x1 > LEFT_LIMIT:
            continue
        if not (30 <= rect.width <= 90 and 30 <= rect.height <= 90):
            continue
        if not (0.85 <= rect.width / rect.height <= 1.15):
            continue
        dtype = d.get("type")
        if dtype in ("f", "fs") and d.get("fill"):
            swatches.append((rect.y0, tuple(int(round(c * 255)) for c in d["fill"][:3])))
        elif dtype == "s":
            swatches.append((rect.y0, (255, 255, 255)))
    swatches.sort()
    seen, out = set(), []
    for _, rgb in swatches:
        if rgb not in seen:
            seen.add(rgb)
            out.append(rgb)
    return out


def _palette_from_render(arr, swatches, min_frac: float = 0.0008,
                         bin_q: int = 4) -> list[tuple[int, int, int]]:
    """Palette = the colours actually painted in the FLAT VIEW artwork, clustered
    onto the declared swatches.

    Every rendered pixel is assigned to its nearest swatch. A swatch is kept only
    if its cluster has a substantial painted colour (>= ``min_frac`` of the image),
    so unused swatches and pure anti-alias clusters drop out. Each kept swatch is
    represented by the artwork's TRUE shade for it: the most-painted cluster colour
    within ``repr_tol`` of the swatch, or — when the artwork renders that swatch a
    little off-chip (e.g. a background slightly lighter than its swatch square) —
    the most-painted cluster colour overall.

    This reads colour straight from the left-hand flat view ("the colours we
    use"): it keeps tiny multi-colour logos (the COMCAST peacock's six feather
    colours, ~0.2% area each, exact swatch matches) while folding shading tints
    and edge blends back onto their base colour (ROWEL's mid-brown heel panel; the
    peacock's grey edge blends, which sit nearest a swatch but far from it).
    """
    if not swatches:
        return []
    H, W = arr.shape[:2]
    tot = float(H * W) or 1.0
    sw = np.array(swatches, np.int16)
    q = (arr.reshape(-1, 3) // bin_q * bin_q).astype(np.int16)
    u, c = np.unique(q, axis=0, return_counts=True)
    d = np.abs(u[:, None, :] - sw[None, :, :]).sum(2)  # (N_colours, N_swatches)
    nearest = d.argmin(1)
    ndist = d.min(1)
    pal: list[tuple[int, tuple[int, int, int]]] = []
    NEAR_TOL = 20          # within this of a chip => an intentional use of that colour
    NEAR_MIN = 0.00015     # ...kept even when tiny, so small declared accents survive
    for si in range(len(sw)):
        idx = np.where(nearest == si)[0]
        if idx.size == 0:
            continue
        # Substantial colours in this cluster (>= 30% of its busiest colour); this
        # drops anti-alias specks. Among them, the rendition is the one CLOSEST to
        # the swatch — so a tiny near-swatch speck can't beat the true colour, and a
        # blend that merely shares the cluster (e.g. the peacock's neutral grey in
        # the LILAC cluster) loses to the real feather, while a big background drawn
        # a little off-chip is still chosen because it's the only substantial colour.
        mx = c[idx].max()
        keep = idx[c[idx] >= 0.30 * mx]
        rep = keep[np.argmin(ndist[keep])]
        rep_cnt = int(c[rep])
        rep_frac = rep_cnt / tot
        rep_dist = int(ndist[rep])
        # Keep this swatch if it is genuinely used in the artwork:
        #   * its rendition closely matches the chip (the designer painted that exact
        #     colour) — kept even when tiny, so small accents like Chicago's red
        #     flag-star survive, OR
        #   * it covers a meaningful area even if rendered a little off-chip (e.g. a
        #     background a touch lighter than its square — Honorlock's grey).
        # Unused chips (only far anti-alias in their cluster — e.g. Ballard's LEMON,
        # which the artwork never paints) satisfy neither and drop out.
        if not ((rep_dist <= NEAR_TOL and rep_frac >= NEAR_MIN) or rep_frac >= min_frac):
            continue
        pal.append((rep_cnt, tuple(int(x) for x in u[rep])))
    pal.sort(key=lambda t: -t[0])
    return [rgb for _, rgb in pal]


def extract_palette_from_pdf(pdf_path: str, max_colors: int = 16, merge: int = 16,
                             arr: "np.ndarray | None" = None) -> list[tuple[int, int, int]]:
    """Design palette taken from the FLAT-VIEW artwork colours (the colours that
    actually appear in the knit).

    Primary path: render the flat view and read the painted colours, clustered
    onto the declared swatches (see ``_palette_from_render``). This is the source
    of truth — the colours the design actually uses — and it correctly keeps small
    multi-colour logos while dropping anti-alias blends and shading tints. If a
    rendered array is already on hand (the BMP step has one), pass it as ``arr`` to
    avoid a second render.

    Fallbacks, in order: the older bounding-box area estimate (used only if the
    flat view renders but yields too few clustered colours), then the raw swatch
    squares (if the flat view can't be located at all)."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    try:
        spans = get_text_spans(page)
        fr = find_flat_view_rect(page, spans)
    except Exception:
        fr = None

    # --- Primary: colours read from the rendered flat-view artwork ---
    sw = _swatch_palette(page)
    if fr is not None and len(sw) >= 2:
        if arr is None:
            try:
                arr, _ = render_clean_design(pdf_path, dpi=200)
            except Exception:
                arr = None
        if arr is not None:
            pal = _palette_from_render(arr, sw)
            if len(pal) >= 2:
                return pal

    if fr is not None:
        area: dict[tuple[int, int, int], float] = {}
        for d in page.get_drawings():
            r = d.get("rect")
            if r is None:
                continue
            ox = max(0.0, min(r.x1, fr.x1) - max(r.x0, fr.x0))
            oy = max(0.0, min(r.y1, fr.y1) - max(r.y0, fr.y0))
            a = ox * oy
            if a <= 0:
                continue
            for key in ("fill", "color"):
                c = d.get(key)
                if c:
                    rgb = _clean_channels(tuple(int(round(v * 255)) for v in c[:3]))
                    area[rgb] = area.get(rgb, 0.0) + a
        # Drop negligible-area colours (anti-alias intermediates / stray fills, e.g.
        # ROWEL's mid-brown, Rice's stray grey) before merging, so they don't end up
        # as palette entries the production bitmap never uses. Keep everything with a
        # meaningful footprint — this does NOT cap the count, so designs with many
        # genuine tints (Virginia, Maryland) keep all their colours.
        total = sum(area.values()) or 1.0
        sig = {c: a for c, a in area.items() if a / total >= 0.004}
        if len(sig) >= 2:
            area = sig
        # most-used first, dropping colours that are within `merge` of one already kept
        pal: list[tuple[int, int, int]] = []
        for rgb, _a in sorted(area.items(), key=lambda kv: -kv[1]):
            if all(sum(abs(x - y) for x, y in zip(rgb, p)) > merge for p in pal):
                pal.append(rgb)
            if len(pal) >= max_colors:
                break
        if len(pal) >= 2:
            return pal

    return _swatch_palette(page)



def _redact_heel_paths(page, heel_rects, body_color=None, heel_fills=None):
    """Remove heel-oval and guide-line *vector paths* so the artwork under them
    renders through — recovering the true heel pattern exactly instead of
    reconstructing it after rasterisation (Toyota: cream with the black swirl
    carrying through).

    Gated for safety: an oval whose fill equals the body colour is a flat
    body-coloured heel patch (e.g. Wayne Sanderson's camo) — those are KEPT and
    left to the existing flat-fill path, so body-heel designs are unchanged.
    Accent-coloured ovals (Toyota red, Coffman/Honorlock blue) and the guide
    line are revealed.

    Returns (n_paths_removed, kept_rects) where kept_rects still need the raster
    ``_remove_heel`` (body ovals; everything on a flattened/raster PDF).
    """
    if not heel_rects:
        return 0, [], 0
    before = len(page.get_drawings())
    kept = []
    annotated = 0
    for r in heel_rects:
        is_line = r.height < 4
        if is_line:
            # The guide line is a blue stroke; redaction leaves a residue that
            # snaps to cream. Heal it in raster instead (clean vertical fill).
            kept.append(r)
            continue
        if body_color is not None and heel_fills is not None:
            fill = heel_fills.get((round(r.x0, 2), round(r.y0, 2),
                                   round(r.x1, 2), round(r.y1, 2)))
            if fill is not None and sum(abs(a - b) for a, b in zip(fill, body_color)) <= 40:
                kept.append(r)            # body-coloured patch -> keep flat-fill
                continue
        # accent oval: margin 8 removes BOTH the fill and the 0.75pt outline stroke
        page.add_redact_annot(fitz.Rect(r.x0 - 8, r.y0 - 8, r.x1 + 8, r.y1 + 8),
                              fill=None)
        annotated += 1
    if annotated == 0:
        return 0, kept, 0
    page.apply_redactions(graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    return before - len(page.get_drawings()), kept, annotated


def _reveal_left_void(arr, revealed_rects, to_design_px, body_color,
                      min_density_gain=0.12):
    """Decide whether a heel reveal should be REJECTED (True == reject, fall back
    to the constructed raster fill).

    A reveal is only trustworthy when the artwork genuinely continues under the
    oval and fills the lens with pattern — which shows up as the lens coming back
    *denser* (less body colour) than the rows around it (Toyota's swirls converge
    at the heel: ~0.20 less body inside than outside). When the artwork does NOT
    continue — small-motif or interrupted patterns (argyle, stripes, repeating
    icons) — the lens comes back equal-or-emptier (void/gap), and the old
    constructed fill is the correct, verified result.

    Conservative by design: anything that isn't a confident, well-filled reveal
    is rejected, so this can only ever fall back to the proven behaviour, never
    regress it. Requires EVERY revealed oval to clear the bar."""
    H, W = arr.shape[:2]
    bc = np.array(body_color, np.int16)

    def body_frac(block):
        if block.size == 0:
            return 1.0
        d = np.abs(block.reshape(-1, 3).astype(np.int16) - bc).sum(1)
        return float((d <= 40).mean())

    if not revealed_rects:
        return False
    for r in revealed_rects:
        rx0, ry0, rx1, ry1 = to_design_px(r, 0)
        if rx1 <= rx0 or ry1 <= ry0:
            return True
        inside = arr[ry0:ry1, rx0:rx1]
        band = max(4, (ry1 - ry0) // 2)
        above = arr[max(0, ry0 - band):ry0, rx0:rx1]
        below = arr[ry1:min(H, ry1 + band), rx0:rx1]
        parts = [b.reshape(-1, 3) for b in (above, below) if b.size]
        surround = np.vstack(parts) if parts else np.empty((0, 3), arr.dtype)
        density_gain = body_frac(surround) - body_frac(inside)
        if density_gain < min_density_gain:
            return True   # not a confident, well-filled reveal -> reject
    return False


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

    stroke_widths, heel_fills = _drawing_maps(page)
    # cheap low-res probe to learn the body colour BEFORE deciding which ovals
    # are body-coloured patches (kept) vs accent overlays (revealed).
    _pp = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), clip=render_rect, alpha=False)
    _parr = np.frombuffer(_pp.samples, dtype=np.uint8).reshape(_pp.height, _pp.width, 3)
    _body_probe = _dominant_body_color(_parr, None)
    heel_removed, heel_kept, n_accent = _redact_heel_paths(
        page, heel_rects, _body_probe, heel_fills)
    revealed = [r for r in heel_rects
                if r.height >= 4 and not any(r == k for k in heel_kept)]

    scale = dpi / 72.0

    def _render(pg):
        px = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=render_rect, alpha=False)
        return np.frombuffer(px.samples, dtype=np.uint8).reshape(px.height, px.width, 3).copy()

    arr = _render(page)
    h, w = arr.shape[:2]

    stroke_widths, heel_fills = _drawing_maps(page)

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
        ys = []
        for hr in heel_rects:
            _, ry0, _, ry1 = to_design_px(hr, 0)
            ys += [ry0, ry1]
        heel_band_px = (min(ys), max(ys)) if ys else None
    else:
        heel_band_px = None
    body_color = _dominant_body_color(arr, heel_band_px)

    # Void check: a reveal is only correct when the artwork genuinely continues
    # under the oval (Toyota's swirls). When it doesn't — small-motif or
    # interrupted patterns (argyle, stripes, repeating icons) — removing the oval
    # leaves a body-coloured hole the surrounding pattern doesn't have. Detect that
    # and revert to the original page + the raster reconstruction (old behaviour).
    if heel_removed and revealed and _reveal_left_void(arr, revealed, to_design_px, body_color):
        page = fitz.open(pdf_path)[0]
        arr = _render(page)
        stroke_widths, heel_fills = _drawing_maps(page)
        body_color = _dominant_body_color(arr, heel_band_px)
        heel_removed, heel_kept = 0, heel_rects

    # Revealed ovals + guide line are already correct from the render. Only the
    # KEPT rects (body-coloured patches, or everything on a raster PDF where
    # nothing could be removed) still need the raster reconstruction.
    rects_for_raster = heel_kept if heel_removed else heel_rects
    if rects_for_raster:
        _remove_heel(arr, rects_for_raster, to_design_px, stroke_widths, body_color, heel_fills)

    return arr, flat_rect


def snap_and_downsample(
    hi_arr: np.ndarray,
    palette: list,
    target_w: int,
    target_h: int,
    straighten_stripe_edges: bool = True,
    edge_margin: int = 4,
    equalize_stripes: bool = True,
    mirror_symmetry: bool = False,
) -> np.ndarray:
    """LAB nearest-color snap + mode pool to palette indices.

    If straighten_stripe_edges is set, each output row whose off-color pixels are
    confined to within edge_margin of the left/right edge is committed entirely to
    its dominant color. This straightens stripe lines to full width and removes
    ragged edge pixels WITHOUT moving any band up or down or changing its height.
    Rows whose minority pixels reach the interior (logo art, scenery, true stripe
    boundaries) are left exactly as the per-pixel mode produced them.
    """
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
    do_straighten = straighten_stripe_edges and target_w > 2 * edge_margin
    for ty in range(target_h):
        y0 = int(ty * sy)
        y1 = max(y0 + 1, int((ty + 1) * sy))
        row = np.empty(target_w, dtype=np.uint8)
        for tx in range(target_w):
            x0 = int(tx * sx)
            x1 = max(x0 + 1, int((tx + 1) * sx))
            block = nearest[y0:y1, x0:x1]
            counts = np.bincount(block.flatten(), minlength=P)
            row[tx] = np.argmax(counts)

        # straighten: full-width commit only when stray pixels are pure edge noise
        if do_straighten:
            d = int(np.argmax(np.bincount(row, minlength=P)))
            off = np.where(row != d)[0]
            if off.size and np.all((off < edge_margin) | (off >= target_w - edge_margin)):
                row[:] = d

        result[ty] = row

    if equalize_stripes:
        _equalize_equal_width_stripes(result, nearest, H, target_h, W, P)
    if mirror_symmetry:
        result = _mirror_symmetric_bands(result, P)
    return result


def _mirror_symmetric_bands(idx, P, min_sym: float = 0.70, min_panel: int = 8,
                            sep_frac: float = 0.92) -> np.ndarray:
    """Enforce vertical mirror-symmetry on bands that are *meant* to be symmetric
    (e.g. Navajo borders), by reflecting the top half of each near-symmetric panel
    onto the bottom.

    Downsampling a symmetric design to 168px breaks the symmetry slightly (the
    motif's centre line doesn't land on the pixel grid the same way top vs bottom),
    so equal halves render a touch differently. This finds horizontal panels —
    delimited by full-width solid separator rows — that are already MOSTLY
    symmetric (so the symmetry is intended, not coincidental) and makes them
    exactly symmetric about their best axis. Panels that aren't near-symmetric
    (logos, text, asymmetric art) fall below ``min_sym`` and are left untouched.

    Opt-in: it intentionally diverges from a ground-truth bitmap when that bitmap
    is itself asymmetric, so it is only applied when the caller asks for it.
    """
    H, W = idx.shape
    out = idx.copy()
    is_sep = np.zeros(H, bool)
    for y in range(H):
        c = np.bincount(idx[y], minlength=P)
        if c.max() >= sep_frac * W:
            is_sep[y] = True
    bounds = [0]
    for y in range(1, H):
        if is_sep[y] != is_sep[y - 1]:
            bounds.append(y)
    bounds.append(H)
    for i in range(len(bounds) - 1):
        p0, p1 = bounds[i], bounds[i + 1]
        if p1 - p0 < min_panel or is_sep[p0]:
            continue
        best_s, best_ax = -1.0, None
        for ax in range(p0 + 4, p1 - 3):
            n = min(ax - p0, p1 - ax)
            if n < 4:
                continue
            top = out[ax - n:ax]
            bot = out[ax:ax + n][::-1]
            k = min(len(top), len(bot))
            s = float(np.mean(top[:k] == bot[:k]))
            if s > best_s:
                best_s, best_ax = s, ax
        if best_ax is not None and best_s >= min_sym:
            n = min(best_ax - p0, p1 - best_ax)
            out[best_ax:best_ax + n] = out[best_ax - n:best_ax][::-1]
    return out


def _equalize_equal_width_stripes(result, nearest, H, target_h, W, P,
                                  max_out_px: int = 10, full_frac: float = 0.95,
                                  equal_tol: float = 1.6) -> None:
    """Give a group of EQUAL-width full-width horizontal stripes a uniform output
    thickness.

    Non-integer downsample pitch forces equal stripes into a mix of 1/2-px (here
    3/3/4-px) runs, which reads as "lines of different sizes". This finds runs of
    adjacent thin bands that span the full width AND are equal-height in the
    high-res artwork (so they were *meant* to match), and re-lays them at one
    shared rounded thickness. It deliberately does NOT touch bands whose true
    heights differ (e.g. a Navajo print's intentionally-varied bands) or anything
    that isn't a full-width solid line (logos, scenery), so it can't distort art.
    """
    scale = target_h / float(H)
    min_out = 2.0   # ignore sub-2px bands (image-edge outlines, aa slivers — not design stripes)
    # Full-width dominant palette index per high-res row (-1 if no colour ≥ full_frac).
    rowdom = np.full(H, -1, dtype=np.int32)
    thr = full_frac * W
    for y in range(H):
        counts = np.bincount(nearest[y], minlength=P)
        k = int(counts.argmax())
        if counts[k] >= thr:
            rowdom[y] = k
    # Collapse to high-res bands [start, length, index].
    bands = []
    i = 0
    while i < H:
        if rowdom[i] < 0:
            i += 1
            continue
        j = i
        while j < H and rowdom[j] == rowdom[i]:
            j += 1
        bands.append([i, j - i, int(rowdom[i])])
        i = j
    n = len(bands)
    k = 0
    while k < n:
        # Grow a group of consecutive, ADJACENT, thin bands of similar thickness.
        if not (min_out <= bands[k][1] * scale <= max_out_px):
            k += 1
            continue
        g = [bands[k]]
        m = k + 1
        while m < n and (min_out <= bands[m][1] * scale <= max_out_px) and \
                abs(bands[m][0] - (g[-1][0] + g[-1][1])) <= max(2.0, scale) and \
                bands[m][1] <= equal_tol * g[-1][1] and g[-1][1] <= equal_tol * bands[m][1]:
            g.append(bands[m])
            m += 1
        k = m
        if len(g) < 2 or len({b[2] for b in g}) < 2:
            continue
        hs = [b[1] for b in g]
        if max(hs) > equal_tol * min(hs):   # heights differ -> meant to differ; leave alone
            continue
        nb = len(g)
        r0 = int(round(g[0][0] * scale))
        r1 = int(round((g[-1][0] + g[-1][1]) * scale))
        region = max(nb, r1 - r0)
        t = max(1, int(round(region / nb)))      # equal per-band thickness (rounds 3.5->4)
        block = nb * t
        center = (r0 + r1) / 2.0
        start = int(round(center - block / 2.0))
        start = max(0, min(start, target_h - block)) if block <= target_h else 0
        y = start
        for b in g:
            y2 = min(target_h, y + t)
            if y2 > y:
                result[y:y2, :] = b[2]
            y = y2


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
    target_w: int = TARGET_W,
    target_h: int = TARGET_H,
    mirror_symmetry: bool = False,
) -> dict:
    """Produce paletted BMP at target_w×target_h + metadata."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    if dpi is None:
        doc = fitz.open(pdf_path)
        page = doc[0]
        spans = get_text_spans(page)
        flat_rect = find_flat_view_rect(page, spans)
        native_dpi = target_w / flat_rect.width * 72.0
        dpi = int(round(native_dpi * SUPERSAMPLE_FACTOR))

    hi_arr, flat_rect = render_clean_design(pdf_path, dpi=dpi)

    palette = extract_palette_from_pdf(pdf_path, arr=hi_arr)
    if not palette:
        raise RuntimeError("No swatch colors found in PDF.")

    indices = snap_and_downsample(hi_arr, palette, target_w, target_h,
                                  mirror_symmetry=mirror_symmetry)
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


def process_full_pdf(
    pdf_path: str,
    out_dir: str,
    target_w: int = TARGET_W,
    target_h: int = TARGET_H,
    mirror_symmetry: bool = False,
) -> dict:
    """Run preview extraction + BMP conversion; merged result dict."""
    preview = process_pdf(pdf_path, out_dir, target_w=target_w, target_h=target_h)
    bmp_info = convert_pdf_to_bmp(pdf_path, out_dir, target_w=target_w, target_h=target_h,
                                  mirror_symmetry=mirror_symmetry)
    preview.update(bmp_info)
    return preview

# ---------- automatic size detection ----------


def _detect_material_from_drawings(page) -> str | None:
    """Best-effort: read the selected material from the filled header circle.

    The three material indicators (COMBED COTTON / PERFORMANCE NYLON / MERINO
    WOOL) sit in the top band of the template; the selected one is a filled
    circle, the others are stroke-only. We find the filled header circle and map
    it to the nearest material label. Returns 'cotton'|'wool'|'nylon'|None.

    Material does not affect the bitmap size, so any failure here is harmless —
    it only refines the product-type label.
    """
    try:
        spans = get_text_spans(page)
        labels = {}
        for s in spans:
            t = s.text.upper()
            cx = (s.bbox.x0 + s.bbox.x1) / 2
            if "COMBED COTTON" in t or t.strip() == "COTTON":
                labels["cotton"] = cx
            elif "PERFORMANCE NYLON" in t or t.strip() == "NYLON":
                labels["nylon"] = cx
            elif "MERINO WOOL" in t or t.strip() == "WOOL":
                labels["wool"] = cx
        if not labels:
            return None

        top_band = 140  # header lives near the top of the page
        filled = []
        for d in page.get_drawings():
            if d.get("type") not in ("f", "fs"):
                continue
            r = d.get("rect")
            if r is None or r.y0 > top_band:
                continue
            if not (6 <= r.width <= 26 and 6 <= r.height <= 26):
                continue
            if not (0.8 <= r.width / r.height <= 1.25):
                continue
            filled.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        if not filled:
            return None

        # choose the filled circle nearest (in x) to any material label
        best_mat, best_dx = None, float("inf")
        for cx, _cy in filled:
            for mat, lx in labels.items():
                dx = abs(cx - lx)
                if dx < best_dx:
                    best_dx, best_mat = dx, mat
        # only trust it if the circle is reasonably close to a label
        return best_mat if best_dx < 120 else None
    except Exception:
        return None


def detect_spec_from_pdf(pdf_path: str) -> dict:
    """Detect product type / style / length / material from a mockup PDF.

    Returns a dict with keys: style, length, material, spec (ProductSpec|None),
    text_found (bool). Size is fully determined by (style, length); material is
    best-effort and only refines the label.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    text = page.get_text("text") or ""
    style, length = detect_style_length(text)
    material = detect_material(text) or _detect_material_from_drawings(page)
    spec = resolve_spec(style, length, material)
    return {
        "style": style,
        "length": length,
        "material": material,
        "spec": spec,
        "text_found": bool(text.strip()),
    }


def resolve_output_size(pdf_path: str, override_spec: ProductSpec | None = None) -> dict:
    """Decide the output (width, height) for a PDF.

    Width is always 168. Height is chosen as follows:
      1. override_spec (manual selection) — used verbatim.
      2. Detected (style, length) with a single production size — used verbatim.
      3. Detected (style, length) with two sizes that differ only by material
         (e.g. athletic Quarter: cotton 310 vs nylon 254) — the candidate whose
         height best matches the FLAT-VIEW artwork proportion is chosen. This
         resolves material from geometry, so a misread material circle can't pick
         the wrong size.
      4. Known style but untabulated length (e.g. Ski on some types) — height is
         derived proportionally from the artwork, then snapped to the nearest
         known production height when one is close.
      5. Detection failed entirely — proportional height if the flat view was
         found, else DEFAULT_SPEC (Crew), flagged as a fallback.

    Returns: width, height, spec, source, detection, derived(bool),
             proportional_height.
    """
    detection = detect_spec_from_pdf(pdf_path)

    if override_spec is not None:
        return {
            "width": override_spec.width, "height": override_spec.height,
            "spec": override_spec, "source": "manual",
            "detection": detection, "derived": False, "proportional_height": None,
        }

    style, length = detection["style"], detection["length"]
    cands = candidate_specs(style, length)

    # Proportional height from the flat-view artwork (used for disambiguation /
    # untabulated lengths). Computed once; tolerant of detection failure.
    prop_h = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        spans = get_text_spans(page)
        flat_rect = find_flat_view_rect(page, spans)
        if flat_rect is not None and flat_rect.width > 0:
            prop_h = 168.0 * flat_rect.height / flat_rect.width
    except Exception:
        prop_h = None

    # Case 2/3: one or more tabulated candidates for this (style, length).
    if cands:
        if len(cands) == 1:
            spec = cands[0]
            source = "auto"
        elif prop_h is not None:
            spec = min(cands, key=lambda s: abs(s.height - prop_h))
            source = "auto (geometry-resolved)"
        else:
            # No geometry to disambiguate: prefer material hint, else first.
            spec = resolve_spec(style, length, detection["material"]) or cands[0]
            source = "auto (material-hint)"
        return {
            "width": spec.width, "height": spec.height, "spec": spec,
            "source": source, "detection": detection, "derived": False,
            "proportional_height": prop_h,
        }

    # Case 4/5: no tabulated size. Use the artwork proportion.
    if prop_h is not None:
        width = 168
        snapped = snap_height(prop_h)
        if snapped is not None:
            height, source, derived = snapped, "auto-snapped", False
        else:
            height, source, derived = int(round(prop_h)), "auto-derived", True
        ptype = _PREF_TYPE.get(style or "", {}).get(detection["material"] or "", "Custom")
        out_spec = ProductSpec(ptype, length or "Custom", width, height)
        return {
            "width": width, "height": height, "spec": out_spec, "source": source,
            "detection": detection, "derived": derived, "proportional_height": prop_h,
        }

    # Total failure: fall back to the default size.
    return {
        "width": DEFAULT_SPEC.width, "height": DEFAULT_SPEC.height,
        "spec": DEFAULT_SPEC, "source": "fallback-default",
        "detection": detection, "derived": False, "proportional_height": None,
    }


def process_pdf_autosize(
    pdf_path: str,
    out_dir: str,
    override_spec: ProductSpec | None = None,
) -> dict:
    """Universal entry point: detect the correct size for THIS PDF (or honor a
    manual override) and run the full extraction at that size.

    The returned dict is the usual process_full_pdf result, plus:
      target_w, target_h, size_source, product_label, detection.
    """
    sized = resolve_output_size(pdf_path, override_spec=override_spec)
    w, h = sized["width"], sized["height"]
    info = process_full_pdf(pdf_path, out_dir, target_w=w, target_h=h)
    info["target_w"] = w
    info["target_h"] = h
    info["size_source"] = sized["source"]
    info["product_label"] = sized["spec"].label if sized["spec"] else f"{w}×{h} px"
    info["detection"] = sized["detection"]
    return info
