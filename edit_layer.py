"""Non-destructive edit layer for the production bitmap.

The automation produces a paletted index array (``base``) that is what we score
against ground truth. Human edits are kept *separate* as an ordered list of ops
and replayed on top at render time:

    final = replay(base, ops, palette)

``base`` is never mutated, so the automation's GT scores stay honest and an edit
can never silently corrupt the pipeline output. An edit is explicitly "a human
decided this beats GT for production."

Every op is a small JSON-serialisable dict with a ``kind`` and its params, and
every op transforms a palette-index array in -> palette-index array out, so ops
compose in any order and the file is always production-valid mid-stack (no pixel
can ever hold an index outside the palette).

This module is the substrate the 1px brush and (later) the box tools all ride
on. It has no Streamlit / PIL dependency so it can be unit-tested headlessly.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Base signature + staleness
# ---------------------------------------------------------------------------
def base_signature(indices: np.ndarray, palette: Sequence) -> dict:
    """Identity of the base an edit stack was drawn against.

    Ops live in *final-bitmap pixel coordinates* and reference *palette indices*,
    so they are only valid against a base of the same shape and palette. If a
    re-run changes either (e.g. the Toyota size fix flipping 550 -> 625, or a
    palette merge), the old edits must NOT be replayed blind.
    """
    h, w = indices.shape
    return {
        "h": int(h),
        "w": int(w),
        "n_palette": int(len(palette)),
        "palette": [[int(c[0]), int(c[1]), int(c[2])] for c in palette],
    }


def replay_compatible(sig: dict, indices: np.ndarray, palette: Sequence) -> bool:
    """True if ops recorded against ``sig`` can be replayed *safely* (no crash,
    no out-of-range index) against this base. Requires same shape and same
    palette length; exact RGB values may differ (indices still valid)."""
    h, w = indices.shape
    return (sig.get("h") == int(h)
            and sig.get("w") == int(w)
            and sig.get("n_palette") == int(len(palette)))


# ---------------------------------------------------------------------------
# Op registry  (box tools register here later without touching replay)
# ---------------------------------------------------------------------------
OpHandler = Callable[[np.ndarray, dict, dict], None]
_OPS: dict[str, OpHandler] = {}


def register(kind: str) -> Callable[[OpHandler], OpHandler]:
    def deco(fn: OpHandler) -> OpHandler:
        _OPS[kind] = fn
        return fn
    return deco


def op_kinds() -> tuple[str, ...]:
    return tuple(sorted(_OPS))


@register("set_pixels")
def _apply_set_pixels(idx: np.ndarray, op: dict, ctx: dict) -> None:
    """The 1px brush. ``pixels`` is a list of [x, y, palette_index]. Any pixel
    out of bounds or any index outside the palette is silently skipped, so the
    result is always production-valid."""
    P = ctx["n_palette"]
    H, W = idx.shape
    for x, y, v in op["pixels"]:
        if 0 <= x < W and 0 <= y < H and 0 <= v < P:
            idx[y, x] = v


def replay(base_indices: np.ndarray, ops: Iterable[dict], palette: Sequence) -> np.ndarray:
    """Apply ``ops`` to a *copy* of ``base_indices`` and return the result.
    ``base_indices`` is never mutated."""
    idx = np.ascontiguousarray(base_indices.copy())
    ctx = {"n_palette": int(len(palette))}
    for op in ops:
        kind = op.get("kind")
        fn = _OPS.get(kind)
        if fn is None:
            raise ValueError(f"unknown op kind: {kind!r}")
        fn(idx, op, ctx)
    return idx


# ---------------------------------------------------------------------------
# Op constructors
# ---------------------------------------------------------------------------
def op_set_pixels(pixels: Iterable[tuple[int, int, int]]) -> dict:
    """Build a set_pixels op from an iterable of (x, y, palette_index)."""
    pl = [[int(x), int(y), int(v)] for (x, y, v) in pixels]
    return {"kind": "set_pixels", "pixels": pl}


def validate_ops(ops: Iterable[dict], palette: Sequence,
                 shape: tuple[int, int]) -> list[str]:
    """Return a list of human-readable problems (empty == clean). Used to warn
    before replaying a loaded stack."""
    issues: list[str] = []
    P = len(palette)
    H, W = shape
    for i, op in enumerate(ops):
        kind = op.get("kind")
        if kind not in _OPS:
            issues.append(f"op {i}: unknown kind {kind!r}")
            continue
        if kind == "set_pixels":
            for (x, y, v) in op.get("pixels", []):
                if not (0 <= x < W and 0 <= y < H):
                    issues.append(f"op {i}: pixel ({x},{y}) outside {W}x{H}")
                    break
                if not (0 <= v < P):
                    issues.append(f"op {i}: index {v} outside palette[0..{P-1}]")
                    break
    return issues


# ---------------------------------------------------------------------------
# EditStack: per-design store of ops + provenance
# ---------------------------------------------------------------------------
class EditStack:
    """Ordered ops plus the signature of the base they were drawn against."""

    def __init__(self, signature: dict, ops: list[dict] | None = None,
                 meta: dict | None = None):
        self.signature = signature
        self.ops: list[dict] = list(ops or [])
        self.meta: dict = dict(meta or {})

    # --- editing ---
    def add(self, op: dict) -> None:
        self.ops.append(op)

    def undo(self) -> dict | None:
        return self.ops.pop() if self.ops else None

    def clear(self) -> None:
        self.ops = []

    def __len__(self) -> int:
        return len(self.ops)

    # --- staleness ---
    def is_stale_for(self, indices: np.ndarray, palette: Sequence) -> bool:
        """Strict: any difference from the recorded signature (shape OR palette,
        including RGB drift). Use for a 'these edits were drawn against a
        different output' warning."""
        return self.signature != base_signature(indices, palette)

    def can_replay_on(self, indices: np.ndarray, palette: Sequence) -> bool:
        """Loose: safe to replay without crashing or producing invalid indices
        (same shape + palette length)."""
        return replay_compatible(self.signature, indices, palette)

    # --- render ---
    def render(self, base_indices: np.ndarray, palette: Sequence) -> np.ndarray:
        return replay(base_indices, self.ops, palette)

    # --- persistence (one stack per design, saved next to its BMP) ---
    def to_json(self) -> str:
        return json.dumps(
            {"schema": SCHEMA_VERSION, "signature": self.signature,
             "ops": self.ops, "meta": self.meta},
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "EditStack":
        d = json.loads(s)
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported edit-stack schema {d.get('schema')}")
        return cls(d["signature"], d.get("ops", []), d.get("meta", {}))

    @classmethod
    def new_for(cls, indices: np.ndarray, palette: Sequence,
                meta: dict | None = None) -> "EditStack":
        return cls(base_signature(indices, palette), [], meta)
