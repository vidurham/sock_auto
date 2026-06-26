"""Standalone bitmap brush — test the 1px palette editor on its own.

    python -m streamlit run editor_app.py

Drop in one of the pipeline's output ``.bmp`` files (optionally its
``*_palette.json``). If no palette file is given, the palette is read from the
BMP itself. Paint with palette-locked colours, Apply / Undo, download the
edited BMP and the replayable edit stack.

This is deliberately separate from app.py so the brush can be validated before
it's wired into the main flow.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
import streamlit as st

from bitmap_editor import bitmap_editor

st.set_page_config(page_title="Bitmap Brush", layout="wide")
st.title("Bitmap brush — palette-locked pixel editor")
st.caption("Edits are a non-destructive op-stack on top of the automation output. "
           "The base bitmap is never changed.")


def load_indices_palette(bmp_file, pal_file):
    img = Image.open(bmp_file)
    if img.mode != "P":
        # accept RGB too: quantize to the palette file if present, else error
        if pal_file is None:
            raise ValueError("RGB BMP needs a *_palette.json so colours can be indexed.")
        pal = [tuple(c) for c in json.load(pal_file)["palette"]]
        a = np.array(img.convert("RGB")).reshape(-1, 3).astype(int)
        P = np.array(pal)
        idx = np.abs(a[:, None, :] - P[None, :, :]).sum(2).argmin(1)
        indices = idx.reshape(img.size[1], img.size[0]).astype(np.uint8)
        return indices, pal

    indices = np.array(img, dtype=np.uint8)
    if pal_file is not None:
        palette = [tuple(int(v) for v in c) for c in json.load(pal_file)["palette"]]
    else:
        flat = img.getpalette() or []
        n = int(indices.max()) + 1
        palette = [tuple(flat[i * 3:i * 3 + 3]) for i in range(n)]
    return indices, palette


up1, up2 = st.columns(2)
with up1:
    bmp_file = st.file_uploader("Production BMP", type=["bmp"])
with up2:
    pal_file = st.file_uploader("Palette JSON (optional)", type=["json"])

if bmp_file is None:
    st.info("Upload a `.bmp` from the pipeline's output to start painting.")
    st.stop()

try:
    indices, palette = load_indices_palette(bmp_file, pal_file)
except Exception as e:
    st.error(f"Could not load: {e}")
    st.stop()

name = os.path.splitext(bmp_file.name)[0]
H, W = indices.shape
st.write(f"**{name}** · {W}×{H} · {len(palette)} palette colours")

final = bitmap_editor(indices, palette, stack_key=name, design_name=name)
