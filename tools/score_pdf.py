"""Score the PDF pipeline against every finished bitmap.

    python tools/score_pdf.py [filter]    # writes output/score_pdf/*.png diffs
"""
from __future__ import annotations
import sys, os, tempfile, warnings, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pairs import pairs  # noqa: E402

OUT = ROOT / "output" / "score_pdf"
# finished bitmap was made from revised artwork that is not in the mockup PDF
REVISED = ("Intergration", "Scramble")


def rgb_to_lab(x):
    from sock_extractor.raster import rgb_to_lab as f
    return f(x)


def compare(ours_idx, ours_pal, gt_img, tol=10.0):
    """Three accuracies, strictest first:
      strict : our pixel RGB == GT pixel RGB (CIELAB < tol). A designer picking a
               different RGB for the same yarn counts as wrong.
      yarn   : same yarn in the same pixel (GT colours matched to our palette by
               overlap). This is what the knitting machine sees.
      edge1  : yarn match allowing a 1-px edge shift (our colour appears in GT's
               3x3 neighbourhood and vice versa) -- measures 'same design, edges
               within one stitch'.
    """
    gt = np.array(gt_img.convert("RGB"))
    H, W = gt.shape[:2]
    pal = np.array(ours_pal, np.uint8)
    ours = pal[ours_idx]
    gcols, ginv = np.unique(gt.reshape(-1, 3), axis=0, return_inverse=True)
    ginv = ginv.reshape(H, W)
    d = np.sqrt(((rgb_to_lab(gcols.astype(float))[:, None] - rgb_to_lab(pal.astype(float))[None]) ** 2).sum(-1))
    ok_strict = (d < tol)[ginv, ours_idx]
    # yarn correspondence: each GT colour -> our colour it overlaps most
    co = np.zeros((len(gcols), len(pal)), np.int64)
    np.add.at(co, (ginv.ravel(), ours_idx.ravel()), 1)
    gmap = co.argmax(1)
    gl = gmap[ginv]
    ok_yarn = gl == ours_idx
    def nb(a):
        p = np.pad(a, 1, mode="edge")
        return [p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]
    in_gt = np.zeros((H, W), bool); in_ours = np.zeros((H, W), bool)
    for x in nb(gl): in_gt |= x == ours_idx
    for x in nb(ours_idx): in_ours |= x == gl
    ok_edge = ok_yarn | (in_gt & in_ours)
    return dict(acc=float(ok_strict.mean()), yarn=float(ok_yarn.mean()), edge1=float(ok_edge.mean()),
                ok=ok_yarn, gt=gt, ours=ours, pal_err=float(d.min(1).max()), ncol=f"{len(pal)}/{len(gcols)}")


def run_one(args):
    pdf, bmp, mode = args
    from sock_extractor.core import process_full_pdf, resolve_output_size
    gt_img = Image.open(bmp)
    W, H = gt_img.size
    res = {"name": pdf.stem[:40], "gt_h": H}
    try:
        r = resolve_output_size(str(pdf))
        res["auto_h"] = r["height"]
        th = H if mode == "gt" else r["height"]
        with tempfile.TemporaryDirectory() as td:
            info = process_full_pdf(str(pdf), td, target_w=168, target_h=th)
            idx = np.array(Image.open(info["bmp_path"]))
            pal = info["palette"]
            res["method"] = info.get("method")
        if idx.shape != (H, W):
            res["acc"] = 0.0
            res["err"] = f"size {idx.shape[::-1]}"
            return res
        c = compare(idx, pal, gt_img)
        ok, gt, ours = c.pop("ok"), c.pop("gt"), c.pop("ours")
        res.update(c, wrong=int((~ok).sum()), pal_err=round(c["pal_err"], 1))
        diff = np.where(ok[..., None], (ours // 3 + 170).astype(np.uint8), np.array([255, 0, 255], np.uint8))
        strip = np.concatenate([gt, ours, diff], 1)
        Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST) \
            .save(OUT / f"{pdf.stem[:40]}.png")
    except Exception as e:
        res["acc"] = 0.0
        res["err"] = repr(e)[:120]
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flt = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ""
    mode = "auto" if "--auto" in sys.argv else "gt"
    ps = [(p, b, mode) for p, b in pairs() if flt.lower() in p.name.lower()]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rs = list(ex.map(run_one, ps))
    rs.sort(key=lambda r: r.get("yarn", 0))
    print(f"{'design':40s} {'strict':>7s} {'yarn':>7s} {'edge1':>7s}  size")
    for r in rs:
        size_ok = "OK " if r.get("auto_h") == r["gt_h"] else "BAD"
        flag = "  [GT is a revised design]" if any(e in r["name"] for e in REVISED) else ""
        print(f"{r['name']:40s} {100*r['acc']:6.2f}% {100*r.get('yarn',0):6.2f}% {100*r.get('edge1',0):6.2f}%  "
              f"gt={r['gt_h']} auto={r.get('auto_h')} {size_ok} {r.get('method','')} {r.get('err','')}{flag}")
    ev = [r for r in rs if not any(e in r["name"] for e in REVISED)]
    for k in ("acc", "yarn", "edge1"):
        v = [r.get(k, 0) for r in ev]
        print(f"MEAN {k:6s} {100*np.mean(v):6.2f}%   >=99%: {sum(a>=0.99 for a in v)}/{len(v)}")
    print(f"size correct: {sum(r.get('auto_h')==r['gt_h'] for r in rs)}/{len(rs)}   "
          f"(excluding {len(rs)-len(ev)} designs whose finished bitmap is a revised design)")
    json.dump(rs, open(OUT / "results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
