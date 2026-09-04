"""Probe the solver curvature (pred_mid - pred): is the wall signal class-informative,
and does it add anything OUTSIDE the cached-state span?

Checklist applied before running (memory: verify-before-reporting):
- span validity: curvature is a nonlinear function of the state through the network, so
  `states + curv vs states` is a legitimate non-redundancy test (unlike 6g's increments).
- harness power: the states+X harness at this n and D was validated with graded synthetic
  blocks (section 8 / S2, detection floor +0.002..+0.008). Same protocol reused.
- no operating-point selection: all of C in {0.01, 0.1, 1.0} reported.
- pairing: hard-fail on labels/subset_indices vs the invstate cache.
- bootstrap: image-level, paired.
Blocks probed ALONE and as additions: curv (the wall), pred (velocity, expected ~section-3
null), pred_mid (midpoint velocity). Pooled per t: token-mean (64d) and 2x2 (256d).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
CS = [0.01, 0.1, 1.0]
DATA = {
    "eurosat": ("models/solver_curvature/eurosat_solvercurv_n50.npz",
                "models/raw_xt/eurosat_invstate_n50_full.npz"),
    "resisc45": ("models/solver_curvature/resisc45_solvercurv_n50.npz",
                 "models/raw_xt/resisc45_invstate_n50_full.npz"),
}


def pool_tokens(x, grid):
    """(N, K, T, 64) packed -> (N, K*64*grid*grid) via spatial pooling on the token grid."""
    n, k, t, d = x.shape
    hw = int(round(t ** 0.5))
    sp = torch.from_numpy(x).reshape(n * k, hw, hw, d).permute(0, 3, 1, 2)
    p = F.adaptive_avg_pool2d(sp, (grid, grid))
    return p.reshape(n, k * d * grid * grid).numpy()


def _fold(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
    return va, m.predict(sc.transform(x[va]))


def correct(x, y, c, n_jobs=7):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(x, y))
        for va, pred in Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va, c) for tr, va in jobs):
            out[va] += (pred == y[va]).astype(float)
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    for ds, (curv_path, state_path) in DATA.items():
        dc, dst = np.load(curv_path), np.load(state_path)
        y = dc["labels"]
        if not np.array_equal(y, dst["labels"]) or not np.array_equal(
                dc["subset_indices"], dst["subset_indices"]):
            raise SystemExit(f"FATAL: {ds} curvature and state caches not paired")
        curv = dc["pred_mid"] - dc["pred"]  # fp32, offline difference
        blocks = {"curv": curv, "pred": dc["pred"], "pred_mid": dc["pred_mid"]}
        # states: 4x4-pooled full latents, 7 t (2g protocol), for the non-redundancy base
        f = dst["feats"]
        n, k, dim = f.shape
        hw = int(round((dim // 16) ** 0.5)); ch = dim // (hw * hw)
        st = torch.from_numpy(f).reshape(n, k, ch, hw, hw)
        states = F.adaptive_avg_pool2d(st.reshape(n * k, ch, hw, hw), (4, 4)).reshape(n, -1).numpy()
        chance = 1 / len(np.unique(y))
        print(f"\n=== {ds}  n={n}  chance={chance:.4f}  (curv over 6 non-terminal t) ===", flush=True)
        for c in CS:
            base = correct(states, y, c)
            print(f"  C={c}   states(7x256d) {base.mean():.4f}", flush=True)
            for name, blk in blocks.items():
                for g, tag in [(1, "mean64"), (2, "2x2")]:
                    x = pool_tokens(blk, g)
                    alone = correct(x, y, c)
                    plus = correct(np.hstack([states, x]), y, c)
                    d = plus - base
                    lo, hi = ci(d)
                    sig = "SIGNIF" if lo > 0 or hi < 0 else ""
                    print(f"    {name:<9}{tag:<7} alone {alone.mean():.4f}   "
                          f"states+ {plus.mean():.4f}  d {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {sig}",
                          flush=True)


if __name__ == "__main__":
    main()
