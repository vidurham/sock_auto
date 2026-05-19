"""Production output sizes by product type and style."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSpec:
    product_type: str
    style: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.product_type} — {self.style} ({self.width}×{self.height} px)"


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
    ProductSpec("Cotton Dress Casual", "Over the Calf", 168, 590),
    ProductSpec("Knit Compression Socks", "Over the Calf", 168, 543),
    ProductSpec("Performance Athletic Nylon", "Quarter", 168, 254),
    ProductSpec("Performance Athletic Nylon", "Mini Crew", 168, 372),
    ProductSpec("Performance Athletic Nylon", "Crew", 168, 402),
    ProductSpec("Performance Athletic Nylon", "Over the Calf", 168, 450),
    ProductSpec("Merino Wool Dress / Flat Knit", "Crew", 168, 413),
)

DEFAULT_SPEC = PRODUCT_SPECS[4]  # Athletic - Cotton & Wool, Crew

DEFAULT_SPEC_INDEX = PRODUCT_SPECS.index(DEFAULT_SPEC)
