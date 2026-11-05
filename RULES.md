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
   - The FLAT VIEW rect is already the body without the cuff, so no cuff trimming is needed. An earlier cuff guard was removed because it cut Spencers at a white stripe.
2. **Image** (`sock_extractor/raster.py`) works on PNG/JPG screenshots, and on flattened PDFs automatically when the vector path fails.
   - It finds the flat view from the blue heel line, the outline edges, the cuff/toe lines and the scallop bottoms.

## Accuracy on 30 real PDF → finished bitmap pairs

```bash
python tools/cache_renders.py   # once: high-res renders for experiments
python tools/score_pdf.py       # full scorecard + diff images in output/score_pdf/
```

Three metrics, strictest first:

| Metric | Meaning | Mean (28 designs) | ≥99% |
|---|---|---|---|
| strict | same RGB in every pixel | 94.5% | 1 |
| yarn | same yarn in every pixel (what the machine knits) | 94.8% | 1 |
| edge1 | same yarn, allowing edges to differ by 1 stitch | 97.9% | 11 |

Size detection is correct on 27 of 30. The misses are Coffman (309 vs 310 for the same template Donovan made at 310), Integration Health (544 vs table 543) and Toyota (the mockup says Over the Calf, the bitmap is Ski).

### What the remaining difference is (measured, not guessed)

- **97% of all wrong pixels sit exactly on a colour boundary.** Only 0.15% of pixels are "content" differences. The design, colours, crop, size and heel are right; the disagreement is which side of an edge a boundary pixel lands on.
- The finished bitmaps are **not a fixed function of the mockup**. Each of these was tested against all 27 bitmaps, and none reproduces them:
  - area averaging (current)
  - point sampling at all 64 sub-pixel phases
  - plain 72-dpi export, with and without anti-aliasing
  - global offsets
  - circular wrap shifts
  - a gradient-boosted model of the designers' choices (`tools/learn.py`). Tested leave-one-design-out, it scores 94.58% against 94.69% for plain coverage.

  The best shift per design ranges from −1 to +1 row and −0.5 to +0.5 column, so placement is done by hand.
- **Bitmap RGB is not canonical either.** The same yarn has different RGB in different bitmaps (LEMON has 4 values, GRASS 2), so palette RGB comes from the chips.
- **Two finished bitmaps are revised designs** that are not in their mockup PDFs (Integration Health "Updated", Scramble Monkey R3). They are excluded from the means.

### Rules added from the data

- **Flat view at 1 pt = 1 px.** Every FLAT VIEW is drawn 168 pt wide, and its rect height is the bitmap height (crew 402.66 pt). The vector rect already excludes the cuff.
- **Tints become a checkerboard.** A fine two-colour texture is rendered as a 1-px checkerboard, darker yarn on (row + col) even, as on ROWEL (+2.9%). It only fires where a whole 3×3 area is mixed, so other designs are unchanged.
- **Image-only PDFs** (Toyota) are handled by the image extractor. The size comes from the body proportions, and the chips are found by circle shape.
- **The real bitmaps follow the code's size table, not the cheat sheet.** For example: ankle 230, quarter 310 (one 309), nylon mini crew 372, OTC 550, knee high 614, ski 625.

### Designer rules tested and rejected (so nobody re-tries them blind)

| Idea | Result on 30 designs |
|---|---|
| Thin lines win the pixel: a minority colour forming a thin line takes the pixel (whiskers, outlines) | Worse at every setting tried: hurt 9–26 designs (up to −1.7% on one), best gain +0.24%. Designers thicken some lines and drop others. |
| Repeat lock: copy one half of the sock onto the other | Only Maryland, where the artwork itself repeats 98.5%, is safe, and it gains just +0.15%. Alameda gains +1%, but only because the designer forced a repeat the artwork doesn't have; at that threshold Chicago, Sedona and others lose 4–7%. |
| Pixel-art stars: round dots become plus/diamond stamps (Christian Encounter's habit) | Helps Christian (765 → 726 wrong px). Hurts Comunidad (−0.48%), Honorlock (−0.29%) and Quiet Mountain (−0.11%), whose designers kept dots round. Shipped as an opt-in **Pixel-art stars** checkbox (`--pixel-stars` in batch.py). |
| Learned per-pixel model, tested leave-one-design-out | 94.58% vs 94.69% for plain coverage |
| Nearest-neighbour sampling at the best of 64 phases | 94.95% even when the phase is picked per design with hindsight |

Kosmic Kitty raw vs final: the designer's own raw export matches their final bitmap only **94.55%**. Our output matches the final at **95.94%**. The remaining gap is hand artwork done after export.

## Earlier screenshot results (image path, `python tools/score.py`)

Chicago 96.4%, Jarritos 97.8%, Quiet Mountain 97.8%, OB Sedona 97.7%.
