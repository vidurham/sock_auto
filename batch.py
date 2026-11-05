"""Batch: turn a folder (or list) of mockups into production bitmaps.

    python batch.py mockups/            -> output/batch/<name>/<name>.bmp ...
    python batch.py a.pdf b.png -o out/  --size "Athletic - Cotton & Wool/Crew"

PDFs use the vector extractor (size auto-detected from the page text), falling
back to the image extractor for flattened PDFs. Images (PNG/JPG) use the image
extractor with size from the body's proportions. A summary CSV is written.
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sock_extractor.core import process_full_pdf, resolve_output_size  # noqa: E402
from sock_extractor.product_specs import PRODUCT_SPECS  # noqa: E402
from sock_extractor.raster import process_image  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def find_spec(name: str):
    t, _, l = name.partition("/")
    for s in PRODUCT_SPECS:
        if s.product_type.lower() == t.strip().lower() and s.style.lower() == l.strip().lower():
            return s
    raise SystemExit(f"Unknown size {name!r}. Options:\n" +
                     "\n".join(f"  {s.product_type}/{s.style}" for s in PRODUCT_SPECS))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=str(ROOT / "output" / "batch"))
    ap.add_argument("--pixel-stars", action="store_true",
                    help="draw small round dots as pixel-art plus/diamond stamps")
    ap.add_argument("--size", help='force a size, e.g. "Athletic - Cotton & Wool/Crew"')
    a = ap.parse_args(argv)
    forced = find_spec(a.size) if a.size else None

    files: list[Path] = []
    for inp in a.inputs:
        p = Path(inp)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT | {".pdf"})
        else:
            files.append(p)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        job = out / f.stem
        try:
            if f.suffix.lower() == ".pdf":
                if forced:
                    w, h, src = forced.width, forced.height, "manual"
                else:
                    r = resolve_output_size(str(f))
                    w, h, src = r["width"], r["height"], r["source"]
                info = process_full_pdf(str(f), str(job), target_w=w, target_h=h, pixel_stars=a.pixel_stars)
            else:
                info = process_image(str(f), str(job),
                                     target_h=forced.height if forced else None)
                h, src = info["target_h"], "manual" if forced else "auto (proportions)"
            rows.append([f.name, "ok", f"168x{h}", src, info.get("method", ""),
                         len(info["palette"]), info["bmp_path"], "; ".join(info.get("notes", []))])
            print(f"OK    {f.name:40s} 168x{h}  {src}  [{info.get('method','')}]")
        except Exception as e:
            rows.append([f.name, "ERROR", "", "", "", "", "", str(e)])
            print(f"ERROR {f.name:40s} {e}")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["file", "status", "size", "size_source", "method",
                                   "colors", "bmp", "notes"], *rows])
    print(f"\n{sum(r[1]=='ok' for r in rows)}/{len(rows)} converted -> {out}")


if __name__ == "__main__":
    main()
