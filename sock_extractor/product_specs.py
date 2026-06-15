"""Production output sizes by product type and style.

Sizes are the real production bitmap dimensions, verified against a set of
known-good production bitmaps supplied by the team. (An earlier revision tried a
"cheat sheet" PDF whose numbers turned out to disagree with the actual bitmaps;
those have been reverted. The one correction from the verified bitmaps is
Cotton Dress Casual / Over the Calf, which is 450, not 590.)

Note: material matters for athletic sizing — e.g. Quarter is 310 in cotton/wool
but 254 in performance nylon. Rather than rely on reading the selected-material
circle, the pipeline disambiguates by the flat-view artwork's proportions (see
resolve_output_size in core.py): it picks whichever candidate height best matches
the drawing, so the correct size is chosen even when material can't be read.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSpec:
    product_type: str
    style: str            # this is the LENGTH name (kept as `style` for API compat)
    width: int
    height: int
    heel_break: int | None = None

    @property
    def label(self) -> str:
        return f"{self.product_type} — {self.style} ({self.width}×{self.height} px)"


# Verified production sizes (width is always 168).
PRODUCT_SPECS: tuple[ProductSpec, ...] = (
    ProductSpec("Coozie / Can Cooler", "12 oz. Standard", 168, 88),
    ProductSpec("Athletic - Cotton & Wool", "Ankle", 168, 230),
    ProductSpec("Athletic - Cotton & Wool", "Quarter", 168, 310),
    ProductSpec("Athletic - Cotton & Wool", "Mini Crew", 168, 362),
    ProductSpec("Athletic - Cotton & Wool", "Crew", 168, 402),
    ProductSpec("Athletic - Cotton & Wool", "Over the Calf", 168, 550),
    ProductSpec("Athletic - Cotton & Wool", "Ski", 168, 625),
    ProductSpec("Athletic - Cotton & Wool", "Knee High", 168, 614),
    ProductSpec("Cotton Dress Casual", "Mini Crew", 168, 390),
    ProductSpec("Cotton Dress Casual", "Crew", 168, 420),
    ProductSpec("Cotton Dress Casual", "Over the Calf", 168, 450),  # verified (was 590)
    ProductSpec("Knit Compression Socks", "Over the Calf", 168, 543),
    ProductSpec("Performance Athletic Nylon", "Quarter", 168, 254),
    ProductSpec("Performance Athletic Nylon", "Mini Crew", 168, 372),
    ProductSpec("Performance Athletic Nylon", "Crew", 168, 402),
    ProductSpec("Performance Athletic Nylon", "Over the Calf", 168, 450),
    ProductSpec("Merino Wool Dress / Flat Knit", "Crew", 168, 413),
)

DEFAULT_SPEC = next(
    s for s in PRODUCT_SPECS
    if s.product_type == "Athletic - Cotton & Wool" and s.style == "Crew"
)
DEFAULT_SPEC_INDEX = PRODUCT_SPECS.index(DEFAULT_SPEC)

# All distinct production heights — used to snap a proportionally-derived height
# to an exact production value for lengths not otherwise resolved.
KNOWN_HEIGHTS: tuple[int, ...] = tuple(sorted({s.height for s in PRODUCT_SPECS}))


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
# style class -> the product types that belong to it
_TYPES_BY_STYLE: dict[str, tuple[str, ...]] = {
    "coozie": ("Coozie / Can Cooler",),
    "athletic": ("Athletic - Cotton & Wool", "Performance Athletic Nylon"),
    "dress": ("Cotton Dress Casual", "Merino Wool Dress / Flat Knit"),
    "compression": ("Knit Compression Socks",),
}

# material -> preferred product type, per style class (used only for labelling /
# tie-breaks; size is resolved geometrically)
_PREF_TYPE: dict[str, dict[str, str]] = {
    "athletic": {"nylon": "Performance Athletic Nylon",
                 "cotton": "Athletic - Cotton & Wool",
                 "wool": "Athletic - Cotton & Wool"},
    "dress": {"wool": "Merino Wool Dress / Flat Knit",
              "cotton": "Cotton Dress Casual"},
}


def detect_style_length(text: str) -> tuple[str | None, str | None]:
    """Parse (style_class, length) from a mockup's text layer."""
    u = (text or "").upper()
    if "COMPRESSION" in u:
        style: str | None = "compression"
    elif "DRESS CASUAL" in u:
        style = "dress"
    elif "ATHLETIC" in u:
        style = "athletic"
    elif "COOZIE" in u or "CAN COOLER" in u:
        style = "coozie"
    else:
        style = None

    if "KNEE HIGH" in u or "KNEE-HIGH" in u:
        length: str | None = "Knee High"
    elif "OVER THE CALF" in u:
        length = "Over the Calf"
    elif "SKI" in u:
        length = "Ski"
    elif "MINI CREW" in u:
        length = "Mini Crew"
    elif "QUARTER" in u:
        length = "Quarter"
    elif "ANKLE" in u or "FOOTIE" in u:
        length = "Ankle"
    elif "CREW" in u:
        length = "Crew"
    elif "12 OZ" in u:
        length = "12 oz. Standard"
    else:
        length = None

    if style == "compression" and length is None:
        length = "Over the Calf"
    return style, length


def detect_material(text: str) -> str | None:
    """Best-effort material from the style line (only if a single one is named)."""
    u = (text or "").upper()
    line = ""
    for ln in u.splitlines():
        if "STYLE," in ln or "COMPRESSION SOCK" in ln:
            line = ln
            break
    src = line or u
    named = []
    if "MERINO WOOL" in src or " WOOL" in src:
        named.append("wool")
    if "NYLON" in src:
        named.append("nylon")
    if "COMBED COTTON" in src or "COTTON" in src:
        named.append("cotton")
    return named[0] if len(named) == 1 else None


def candidate_specs(style: str | None, length: str | None) -> list[ProductSpec]:
    """All production specs matching a detected (style, length) — may be 1 or 2
    (e.g. athletic Quarter exists in both cotton/wool 310 and nylon 254)."""
    if style is None or length is None:
        return []
    types = _TYPES_BY_STYLE.get(style, ())
    return [s for s in PRODUCT_SPECS if s.product_type in types and s.style == length]


def snap_height(h: float, tol_frac: float = 0.04) -> int | None:
    """Snap a derived height to the nearest known production height within tol."""
    if not KNOWN_HEIGHTS:
        return None
    nearest = min(KNOWN_HEIGHTS, key=lambda k: abs(k - h))
    return nearest if abs(nearest - h) <= tol_frac * h else None


def resolve_spec(style: str | None, length: str | None,
                 material: str | None = None) -> ProductSpec | None:
    """Pick a spec from (style, length[, material]) without geometry.

    Used as a fallback / for the manual path. When two candidates differ only by
    material and material is unknown, returns None so the caller can disambiguate
    by artwork proportion (the preferred route)."""
    cands = candidate_specs(style, length)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    if material is not None:
        pref = _PREF_TYPE.get(style or "", {}).get(material)
        for c in cands:
            if c.product_type == pref:
                return c
    return None
