"""Learn the designer's pixel decisions from finished bitmaps.

For every output pixel and every candidate palette colour we compute features
(how much of the pixel / its quarters / its 3x3 neighbourhood that colour
covers, its contrast to the surroundings, whether it is the background, the
checkerboard parity ...). A gradient-boosted model scores candidates; the
highest-scoring colour wins. Evaluated leave-one-design-out.
"""
import sys, warnings, pickle; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from tools.fitlab import load, nearest_idx, coverage, gt_labels
from sock_extractor.raster import rgb_to_lab

FEATS = ["c", "q0", "q1", "q2", "q3", "n_mean", "n_max", "n_min", "c_rank", "c_gap",
         "glob", "glob_rank", "L", "dL_dom", "is_dark", "is_light", "mid", "par", "par_mid",
         "row_frac", "col_frac", "n5_mean", "thin_h", "thin_v"]


def box3(a):
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    H, W = a.shape[1:]
    s = [p[:, dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]
    return np.stack(s, 0)       # (9,P,H,W)


def features(nidx, pal, H, W):
    P = len(pal)
    cov = coverage(nidx, P, H, W)                  # (P,H,W)
    q = coverage(nidx, P, 2 * H, 2 * W)            # quarters
    q = q.reshape(P, H, 2, W, 2)
    qs = [q[:, :, 0, :, 0], q[:, :, 0, :, 1], q[:, :, 1, :, 0], q[:, :, 1, :, 1]]
    nb = box3(cov)
    n_mean = nb.mean(0); n_max = nb.max(0); n_min = nb.min(0)
    p5 = np.pad(cov, ((0, 0), (2, 2), (2, 2)), mode="edge")
    n5 = np.stack([p5[:, dy:dy + H, dx:dx + W] for dy in range(5) for dx in range(5)], 0).mean(0)
    order = np.argsort(-cov, axis=0)
    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(P)[:, None, None], 0)
    top = np.take_along_axis(cov, order[:1], 0)[0]
    gap = cov - top
    glob = cov.reshape(P, -1).mean(1)
    grank = np.argsort(np.argsort(-glob))
    L = rgb_to_lab(np.array(pal, float))[:, 0]
    dom = n5.argmax(0)                              # locally dominant colour
    dL = L[:, None, None] - L[dom][None]
    yy, xx = np.mgrid[0:H, 0:W]
    par = ((yy + xx) % 2).astype(np.float32)
    mid = 4 * cov * (1 - cov)
    # thin-ness: present here but weak in both horizontal (or vertical) neighbours
    pc = np.pad(cov, ((0, 0), (1, 1), (1, 1)), mode="edge")
    thin_h = cov - 0.5 * (pc[:, 1:-1, :-2] + pc[:, 1:-1, 2:])
    thin_v = cov - 0.5 * (pc[:, :-2, 1:-1] + pc[:, 2:, 1:-1])
    F = np.stack([
        cov, *qs, n_mean, n_max, n_min, rank.astype(np.float32), gap,
        np.broadcast_to(glob[:, None, None], cov.shape), np.broadcast_to(grank[:, None, None], cov.shape).astype(np.float32),
        np.broadcast_to(L[:, None, None], cov.shape), dL,
        np.broadcast_to((L == L.min())[:, None, None], cov.shape).astype(np.float32),
        np.broadcast_to((L == L.max())[:, None, None], cov.shape).astype(np.float32),
        mid, np.broadcast_to(par, cov.shape), par * mid,
        np.broadcast_to(yy / H, cov.shape), np.broadcast_to(xx / W, cov.shape), n5, thin_h, thin_v,
    ], -1)                                          # (P,H,W,F)
    cand = n_max > 0                                # colour present in 3x3
    return F.astype(np.float32), cand, cov


def build(designs):
    rows = []
    for name, arr, pal, gt in designs:
        H, W = gt.shape[:2]
        n, _ = nearest_idx(arr, pal)
        F, cand, cov = features(n, pal, H, W)
        g = gt_labels(gt, pal)
        ncand = cand.sum(0)
        rows.append(dict(name=name, F=F, cand=cand, cov=cov, g=g, ncand=ncand, pal=pal))
    return rows


def train_xy(rows, max_easy=0):
    X, y = [], []
    for r in rows:
        hard = r["ncand"] > 1
        P = r["F"].shape[0]
        for k in range(P):
            m = r["cand"][k] & hard
            X.append(r["F"][k][m]); y.append((r["g"] == k)[m])
    return np.concatenate(X), np.concatenate(y)


def predict(model, r):
    P, H, W, nf = r["F"].shape
    s = np.full((P, H, W), -1e9, np.float32)
    hard = r["ncand"] > 1
    for k in range(P):
        m = r["cand"][k] & hard
        if m.any():
            s[k][m] = model.predict_proba(r["F"][k][m])[:, 1]
    pred = r["cov"].argmax(0)
    pred[hard] = s.argmax(0)[hard]
    return pred


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = build(load())
    base, learned = [], []
    for i, r in enumerate(rows):
        tr = [x for j, x in enumerate(rows) if j != i]
        X, y = train_xy(tr)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)
        m.fit(X, y)
        b = (r["cov"].argmax(0) == r["g"]).mean()
        l = (predict(m, r) == r["g"]).mean()
        base.append(b); learned.append(l)
        print(f"{r['name'][:38]:38s} argmax={b:.4f}  learned(LOO)={l:.4f}  {'+' if l>b else ''}{100*(l-b):.2f}")
    print(f"MEAN argmax={np.mean(base):.4f} learned={np.mean(learned):.4f}")
    X, y = train_xy(rows)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31).fit(X, y)
    pickle.dump(m, open(ROOT / "sock_extractor" / "pixel_model.pkl", "wb"))
