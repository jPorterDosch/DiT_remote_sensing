"""The n=5000 curvature battery -- the frozen-model baseline for pre-registered
prediction 3, on the dataset where it matters (RESISC45: powered instrument, hard classes).

Everything paired by construction: states, pred, pred_mid come from the SAME chain pass.
All arms probed identically (in-fold PCA-512 for tractability at n=5000 -- symmetric across
every block, so margins are comparable; levels are not the object). C=0.1 (n=500 showed
C-stability of all deltas). Image-level bootstrap.

Blocks: the carrier decomposition (norms / pattern / full) plus the three validity controls
(iid noise, image-shuffled curvature, synthetic s=4 detection anchor).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
C = 0.1
CACHES = {
    "resisc45": "models/solver_curvature_n5000/resisc45_solvercurv_n50.npz",
    "eurosat": "models/solver_curvature_n5000/eurosat_solvercurv_n50.npz",
}


def pool_tokens(x, grid):
    n, k, t, d = x.shape
    hw = int(round(t ** 0.5))
    sp = torch.from_numpy(x).reshape(n * k, hw, hw, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(sp, (grid, grid)).reshape(n, k * d * grid * grid).numpy()


def _fold(x, y, tr, va):
    sc = StandardScaler().fit(x[tr])
    a, b = sc.transform(x[tr]), sc.transform(x[va])
    if a.shape[1] > 512:
        p = PCA(n_components=512, svd_solver="randomized", random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
    return va, m.predict(b)


def correct(x, y, n_jobs=7):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(x, y))
        for va, pred in Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va) for tr, va in jobs):
            out[va] += (pred == y[va]).astype(float)
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    for ds, path in CACHES.items():
        d = np.load(path)
        y = d["labels"]
        n = len(y)
        rng = np.random.default_rng(0)
        states = pool_tokens(d["states"], 2)          # 7 x 256
        pred = pool_tokens(d["pred"], 2)              # 6 x 256
        raw = d["pred_mid"] - d["pred"]
        curv = pool_tokens(raw, 2)
        norms = np.linalg.norm(raw.reshape(n, 6, -1), axis=2)
        raw_n = raw / (norms[:, :, None, None] + 1e-8)
        pattern = pool_tokens(raw_n, 2)
        D = curv.shape[1]
        onehot = np.eye(len(np.unique(y)))[y]
        blocks = [
            ("pred (velocity)", pred),
            ("curv full", curv),
            ("curv pattern (L2-normed)", pattern),
            ("curv norms only (6d)", norms),
            ("noise iid", rng.standard_normal((n, D))),
            ("curv SHUFFLED rows", curv[rng.permutation(n)]),
            ("synthetic s=4", np.hstack([onehot] * (D // onehot.shape[1] + 1))[:, :D]
             + 4 * rng.standard_normal((n, D))),
        ]
        base_x = np.hstack([states, pred])
        base = correct(base_x, y)
        print(f"\n=== {ds}  n={n}  chance={1/len(np.unique(y)):.4f}  "
              f"base states+pred {base.mean():.4f}  (C={C}, PCA-512) ===", flush=True)
        for name, blk in blocks:
            if name == "pred (velocity)":
                st = correct(states, y)
                dd = base - st
                lo, hi = ci(dd)
                print(f"  states alone {st.mean():.4f}; +pred d {dd.mean():+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]", flush=True)
                continue
            plus = correct(np.hstack([base_x, blk]), y)
            dd = plus - base
            lo, hi = ci(dd)
            sig = "SIGNIF" if lo > 0 or hi < 0 else ""
            print(f"  +{name:<26} {plus.mean():.4f}  d {dd.mean():+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {sig}", flush=True)


if __name__ == "__main__":
    main()
