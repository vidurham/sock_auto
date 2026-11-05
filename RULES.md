# Mockup → Bitmap rules

These rules come from comparing the designer's mockups (`design_pdf/`) with the
finished production bitmaps (`finished_bit/`). Check them with:

```bash
python tools/score.py        # % of pixels matching each finished bitmap + diff images in output/score/
```

Convert mockups in bulk:

```bash
python batch.py path/to/mockups/        # PDFs and/or PNG/JPG → output/batch/<name>/<name>.bmp + summary.csv
```

## What goes into the bitmap

| Rule | Detail |
|---|---|
| Only the FLAT VIEW | Front / back / side views are ignored. |
| No cuff | The thin band at the top of the flat view is cut off. The bitmap starts right under the cuff line. |
| No toe | The scalloped toe below the toe line is cut off. |
| Heel removed | The two half-ovals at the heel and the blue guide line are erased. A plain surround gets a flat fill in the body color. A patterned surround is continued from above and below. |
| Width | Always 168 px = the full flat-view width, with the outline stroke trimmed. |
| Height | Taken from the product spec table (`sock_extractor/product_specs.py`). If the product can't be read (for example on image mockups), the body's proportions are snapped to the nearest production height. These proportions came out at exactly 402 on all 5 example crews. |
| Colors | Exactly the swatch chips from the left column. Pixels snap to the nearest chip in CIELAB. Non-chip shading, such as Sedona's light-brown panels, folds into the nearest chip. |
| Palette | Only chips the artwork actually uses. Written as a tight 8-bit BMP (palette size = number of colors). |

## Template geometry (fractions of flat-view width W)

- Cuff band height ≈ 0.088 W
- Toe scallop height ≈ 0.226 W
- Heel half-ovals ≈ 0.26 W reach × 0.19 W tall
- Outline stroke ≈ 0.003–0.006 W

## Two extraction paths

1. **Vector PDF** (`sock_extractor/core.py`) is the most exact. Shapes and colors are read from the PDF itself.
   - A cuff guard trims the cuff only when a full-width line sits at the cuff position *and* trimming makes the shape match a production size better.
2. **Image** (`sock_extractor/raster.py`) works on PNG/JPG screenshots, and on flattened PDFs automatically when the vector path fails.
   - It finds the flat view from the blue heel line, the outline edges, the cuff/toe lines and the scallop bottoms.

## Current accuracy on the examples (image path, 2400-px screenshots)

| Design | Match |
|---|---|
| Chicago Backflow | 96.4% |
| Jarritos | 97.8% |
| Quiet Mountain | 97.8% |
| OB Sedona | 97.7% |

Remaining differences are mostly 1-px anti-aliased edges. A screenshot gives only about 2 source pixels per bitmap pixel, while the PDF path renders at 6×, so real PDFs should score higher.

## Known open questions

- **Yarn RGB vs chip RGB.** Sedona's finished bitmap uses Marigold (246,223,47), but the chip on the mockup is (252,185,40). A yarn-name → production-RGB table would fix cases like this.
- **Palette order.** The finished BMPs don't follow chip order (Chicago starts with white, black…). If the knitting machine cares about index order, we need that rule.
