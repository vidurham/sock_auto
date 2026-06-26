"""Click-to-paint grid editor for the production bitmap (Streamlit component).

Rides entirely on ``edit_layer``: each click/drag gesture becomes a ``set_pixels``
op on a per-design ``EditStack``. The brush can only paint colours that are
already in the palette, so the edited file is always production-valid. ``base`` is
never touched — the editor renders ``replay(base, ops)`` live.

Viewport canvas for laptop editing: **Fit all** to see the whole sock, zoom into a
spot (overview click, double-click, or Detail), fix pixels, **Fit all** to zoom
back out. Pan/zoom is preserved across edits.

Use standalone via ``editor_app.py`` to test in isolation, or call
``bitmap_editor(indices, palette, stack_key=...)`` from the main app.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
import streamlit as st

import edit_layer as el
from pixel_grid import pixel_grid


# ---------------------------------------------------------------------------
# Pure helpers (testable without Streamlit)
# ---------------------------------------------------------------------------
def indices_to_rgb(indices: np.ndarray, palette) -> np.ndarray:
    """(H,W) palette indices -> (H,W,3) uint8 RGB."""
    pal = np.array(palette, dtype=np.uint8)
    return pal[indices]


def _palette_picker(palette, stack_key: str) -> int:
    """Grid of small colour squares; returns selected palette index."""
    sk = f"color_idx::{stack_key}"
    if sk not in st.session_state:
        st.session_state[sk] = 0
    sel = int(st.session_state[sk])
    n = len(palette)
    per_row = min(12, max(n, 1))

    st.caption("Brush colour — click a square")
    for row_start in range(0, n, per_row):
        cols = st.columns(per_row)
        for col, i in enumerate(range(row_start, min(row_start + per_row, n))):
            c = palette[i]
            picked = i == sel
            with cols[col]:
                st.markdown(
                    f'<div title="index {i} · rgb{tuple(int(v) for v in c)}" '
                    f'style="width:2.25rem;height:2.25rem;margin:0 auto 0.15rem;'
                    f'background:rgb({int(c[0])},{int(c[1])},{int(c[2])});'
                    f'border:{"3px solid #4da3ff" if picked else "2px solid #666"};'
                    f'border-radius:4px;box-shadow:{"0 0 0 2px #1a1a1a" if picked else "none"};">'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    str(i),
                    key=f"pick::{stack_key}::{i}",
                    type="primary" if picked else "secondary",
                    width="stretch",
                ):
                    st.session_state[sk] = i

    c = palette[sel]
    st.markdown(
        f"**Selected:** index **{sel}** · rgb({int(c[0])}, {int(c[1])}, {int(c[2])})"
    )
    return sel


def paletted_bmp_bytes(indices: np.ndarray, palette) -> bytes:
    """Encode (H,W) indices + palette as a paletted BMP (same format the
    pipeline writes)."""
    h, w = indices.shape
    img = Image.new("P", (w, h))
    flat: list[int] = []
    for c in palette:
        flat.extend([int(c[0]), int(c[1]), int(c[2])])
    flat += [0] * (768 - len(flat))
    img.putpalette(flat)
    img.putdata(indices.flatten().tolist())
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The editor component
# ---------------------------------------------------------------------------
@st.fragment
def bitmap_editor(base_indices: np.ndarray, palette, *, stack_key: str,
                  design_name: str = "bitmap") -> np.ndarray:
    """Render the grid editor for one bitmap. Returns the current edited index
    array (= replay(base, ops))."""
    H, W = base_indices.shape

    sk = f"editstack::{stack_key}"
    if sk not in st.session_state:
        st.session_state[sk] = el.EditStack.new_for(
            base_indices, palette, meta={"design": design_name, "size": [W, H]}
        )
    stack: el.EditStack = st.session_state[sk]

    if not stack.can_replay_on(base_indices, palette):
        st.warning(
            f"The bitmap changed shape/palette since these {len(stack)} edit(s) "
            "were made, so they can't be re-applied. Starting a fresh edit layer."
        )
        st.session_state[sk] = stack = el.EditStack.new_for(
            base_indices, palette, meta={"design": design_name, "size": [W, H]}
        )
    elif stack.is_stale_for(base_indices, palette) and len(stack):
        st.info("Bitmap palette colours drifted slightly; edits still apply by index.")

    sel_idx = _palette_picker(palette, stack_key)

    view_key = f"grid_view::{stack_key}"
    edited = stack.render(base_indices, palette)
    result = pixel_grid(
        edited,
        palette,
        brush_idx=sel_idx,
        view=st.session_state.get(view_key),
        show_grid=True,
        key=f"grid::{stack_key}",
    )

    if result:
        if result.get("view"):
            st.session_state[view_key] = result["view"]
        if result.get("pixels"):
            cells = [(int(p["x"]), int(p["y"])) for p in result["pixels"]]
            tag = (sel_idx, tuple(sorted(cells)))
            last_key = f"last_paint::{stack_key}"
            if st.session_state.get(last_key) != tag:
                st.session_state[last_key] = tag
                stack.add(el.op_set_pixels([(x, y, sel_idx) for x, y in cells]))

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Undo last edit", key=f"undo::{stack_key}", disabled=len(stack) == 0):
            stack.undo()
    with b2:
        if st.button("Clear all edits", key=f"clearall::{stack_key}",
                     disabled=len(stack) == 0):
            stack.clear()
    with b3:
        st.caption(f"{len(stack)} edit op(s) on the stack")

    final = stack.render(base_indices, palette)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download edited BMP",
            data=paletted_bmp_bytes(final, palette),
            file_name=f"{design_name}_edited.bmp",
            mime="image/bmp",
            key=f"dlbmp::{stack_key}",
        )
    with d2:
        st.download_button(
            "Download edits (.json)",
            data=stack.to_json(),
            file_name=f"{design_name}_edits.json",
            mime="application/json",
            key=f"dljson::{stack_key}",
        )

    return final
