"""Cache high-res heel-cleaned flat-view renders + palettes for fast experiments."""
import sys, warnings, os
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from tools.pairs import pairs
S = 8
def one(pb):
    pdf, bmp = pb
    from sock_extractor.core import render_clean_design, extract_palette_from_pdf, find_flat_view_rect, get_text_spans
    import fitz
    out = ROOT / "output" / "cache" / (pdf.stem[:40] + ".npz")
    if out.exists(): return pdf.name, "cached"
    try:
        pg = fitz.open(pdf)[0]; fr = find_flat_view_rect(pg, get_text_spans(pg))
        dpi = 168 / fr.width * 72 * S
        arr, fr = render_clean_design(str(pdf), dpi=dpi)
        pal = extract_palette_from_pdf(str(pdf), arr=arr)
        np.savez_compressed(out, arr=arr, pal=np.array(pal), bmp=str(bmp))
        return pdf.name, arr.shape
    except Exception as e:
        return pdf.name, repr(e)
if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(one, pairs()): print(*r)
