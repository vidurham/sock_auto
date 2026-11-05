"""Pair each mockup PDF in design_pdf/ with its finished bitmap in finished_bit/."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = {"sock", "mockup", "mockups", "final", "bitmap", "pdf", "bmp", "r1", "r2", "r3", "the", "inc",
        "1", "2", "b", "c", "a", "updated", "new"}


def toks(name: str) -> set[str]:
    n = re.sub(r"\(\d\)", " ", name.lower())
    n = re.sub(r"_\d{4}_", " ", n)
    return {t for t in re.findall(r"[a-z0-9]+", n) if t not in STOP and not re.fullmatch(r"\d{4}", t)}


MANUAL = {  # ambiguous names
    "Buck Bald Brewing_2603_Sock Mockups Final.pdf": "Buck Bald Brewing Kosmic Kitty Final Bitmap.bmp",
    "Comunidad_2604_Mockups Final 1.pdf": "Comunidad_2604_Bitmap 1 Adult Dress Casual Mini Crew.bmp",
    "Jarritos_2602_Sock Mockups FINAL Ankle.pdf": "Jarritos Ankle Bitmap.bmp",
    "Jarritos_2602_Sock Mockups FINAL Crew (Color).pdf": "Jarritos Crew Bitmap (Color).bmp",
    "Buck Bald Brewing_2603_Sock Mockups Kosmic Kitty FINAL.pdf": "Buck Bald Brewing Kosmic Kitty Bitmap.bmp",
    "Buck Bald Brewing_2603_Sock Mockups Shesquatch FINAL (1).pdf": "Buck Bald Brewing Shesquatch Bitmap.bmp",
    "Rice University Biotech Launch Pad_2605_SockMockups_FINAL.pdf": "RUBiotechLaunchPad_2605_Bitmap.bmp",
    "IntergrationHealth_2604_FINAL MOCKUP.pdf": "Integration Health_2604_Bitmap Updated.bmp",
    "Ojai Valley Land Conservancy_2602_Mockups Final.pdf": "Ojai Land Conservancy_2602_Bitmap.bmp",
}


def pairs() -> list[tuple[Path, Path]]:
    pdfs = sorted((ROOT / "design_pdf").glob("*.pdf"))
    bmps = sorted(b for b in (ROOT / "finished_bit").glob("*.bmp") if "(1)" not in b.name or True)
    out = []
    for p in pdfs:
        if "Cheat Sheet" in p.name:
            continue
        if p.name in MANUAL:
            out.append((p, ROOT / "finished_bit" / MANUAL[p.name]))
            continue
        tp = toks(p.stem)
        best, score = None, 0.0
        for b in bmps:
            tb = toks(b.stem)
            s = len(tp & tb) / max(1, len(tp | tb))
            if s > score:
                best, score = b, s
        if best is not None and score >= 0.2:
            out.append((p, best))
    return out


if __name__ == "__main__":
    from PIL import Image
    ps = pairs()
    for p, b in ps:
        im = Image.open(b)
        print(f"{p.name[:55]:55s} -> {b.name[:45]:45s} {im.size} {im.mode} n={len(im.getcolors() or [])}")
    print(len(ps), "pairs")
