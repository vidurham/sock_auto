"""Scrollable viewport pixel-grid Streamlit component."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import streamlit.components.v1 as components

_FRONTEND = Path(__file__).resolve().parent / "frontend"
_pixel_grid = components.declare_component("pixel_grid", path=str(_FRONTEND))


def pixel_grid(
    indices: np.ndarray,
    palette,
    *,
    brush_idx: int,
    view: dict[str, Any] | None = None,
    show_grid: bool = True,
    key: str | None = None,
) -> dict | None:
    """Viewport grid editor. Pass ``view`` ``{cell_px, pan_x, pan_y}`` to restore
    zoom/pan after a paint rerun; ``cell_px < 0`` means fit-all on first open.

    Returns ``{"pixels": [...], "view": {...}}`` when the user paints, else
    ``None``."""
    h, w = indices.shape
    pal = [[int(c[0]), int(c[1]), int(c[2])] for c in palette]
    v = view or {}
    return _pixel_grid(
        width=int(w),
        height=int(h),
        cell_px=int(v.get("cell_px", -1)),
        pan_x=float(v.get("pan_x", 0)),
        pan_y=float(v.get("pan_y", 0)),
        indices_b64=base64.b64encode(indices.astype(np.uint8).tobytes()).decode("ascii"),
        palette=pal,
        brush_idx=int(brush_idx),
        show_grid=bool(show_grid),
        key=key,
        default=None,
    )
