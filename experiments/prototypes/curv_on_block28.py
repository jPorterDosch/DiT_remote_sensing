"""Does solver curvature add anything on top of the BEST base -- the block-28 DiT features?

The n=5000 curvature battery (curv5000_probe) deliberately uses a latent-space base
(states+velocity) so "+curvature" is attributable to TRAJECTORY GEOMETRY -- see 6g/6p. That
answers the mechanism question and leaves the PRACTICAL one open: if curvature is redundant
with the operating-point features, it has explanatory value but no deployment value, and the
paper must not imply otherwise (JD, 2026-09-16).

Design: v3 discipline. Base = block-28 pooled inversion features (7 t, DiTF-normalized
offline, concat 21504d -> base-only PCA-512). Block = curvature pattern (per-step L2-normed,
2x2-pooled, 1536d), appended standardized-raw. Null = row-SHUFFLED pattern (width- and
distribution-matched). Both bars per CLAUDE.md rule 3b: raw delta > 0 AND vs-null delta > 0.
Same chain settings and the SAME 5000-image subset in both caches (verified at load); the
two runs drew different VAE posteriors, so pairing is at image level, not trajectory level
-- noted, conservative (unpaired trajectory noise dilutes, cannot manufacture, a gain).
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

C = 0.1
SEEDS = [0, 1, 2]
PCA_DIM = 512
DISCARD = [154, 1446]
CACHES = {
    "resisc45": (
        "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
        "models/solver_curvature_n5000/resisc45_solvercurv_n50.npz",
    ),
    "eurosat": (
        "models/n5000_eurosat_inversion/eurosat_flux_f4ee81c8+42/multistep_train_feats_inversion_g1.0_n50.npz",
        "models/solver_curvature_n5000/eurosat_solvercurv_n50.npz",
    ),
}


def apply_ditf(feats, mods, discard):
    # mirrors traj_readout.apply_ditf_normalization (copy: experiments must stay standalone)
    x = feats.astype(np.float64).copy()
    if discard:
        x[:, :, discard] = 0.0
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mu) / np.sqrt(var + 1e-6)
    x = (1.0 + mods[None, :, 1, :]) * x + mods[None, :, 0, :]
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def pool_tokens(x, grid=2):
    n, k, t, d = x.shape
    hw = int(round(t**0.5))
    sp = torch.from_numpy(x).reshape(n * k, hw, hw, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(sp, (grid, grid)).reshape(n, k * d * grid * grid).numpy()


def _fold(base, blk, y, tr, va):
    sc = StandardScaler().fit(base[tr])
    a, b = sc.transform(base[tr]), sc.transform(base[va])
    k = min(PCA_DIM, a.shape[1], len(tr) - 1)
    if k < a.shape[1]:
        p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    if blk is not None:
        s2 = StandardScaler().fit(blk[tr])
        a = np.hstack([a, s2.transform(blk[tr])])
        b = np.hstack([b, s2.transform(blk[va])])
    m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
    return va, m.predict(b)


def correct(base, blk, y, n_jobs=7):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(base, y))
        for va, pred in Parallel(n_jobs=n_jobs)(delayed(_fold)(base, blk, y, tr, va) for tr, va in jobs):
            out[va] += (pred == y[va]).astype(float)
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    for ds, (fpath, cpath) in CACHES.items():
        df, dc = np.load(fpath), np.load(cpath)
        if not np.array_equal(df["subset_indices"], dc["subset_indices"]):
            raise SystemExit(f"UNPAIRED caches for {ds}")
        y = df["labels"]
        n = len(y)
        mods = df["mods"] if "mods" in df.files else None
        feats = apply_ditf(df["feats"], mods, DISCARD) if mods is not None else df["feats"]
        base = feats.reshape(n, -1)  # 7 x 3072 concat, DiTF-normalized
        raw = dc["pred_mid"] - dc["pred"]
        norms = np.linalg.norm(raw.reshape(n, raw.shape[1], -1), axis=2)
        pattern = pool_tokens(raw / (norms[:, :, None, None] + 1e-8))
        rng = np.random.default_rng(0)
        b = correct(base, None, y)
        pa = correct(base, pattern, y)
        pn = correct(base, pattern[rng.permutation(n)], y)
        raw_d = pa - b
        dod = pa - pn
        rl, rh = ci(raw_d)
        dl, dh = ci(dod)
        print(
            f"\n=== {ds}  n={n}  base = block-28 inversion concat (DiTF, PCA-512): {b.mean():.4f} ===",
            flush=True,
        )
        print(
            f"  +curv pattern        {pa.mean():.4f}   raw d {raw_d.mean():+.4f} [{rl:+.4f},{rh:+.4f}]",
            flush=True,
        )
        print(
            f"  vs SHUFFLED pattern  {pn.mean():.4f}   DoD   {dod.mean():+.4f} [{dl:+.4f},{dh:+.4f}]",
            flush=True,
        )
        both = raw_d.mean() > 0 and rl > 0 and dl > 0
        print(
            f"  RULE 3b (raw>0 AND DoD>0): {'PASS -- curvature adds beyond block-28' if both else 'FAIL -- no evidence beyond block-28'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
