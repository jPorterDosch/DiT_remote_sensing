"""Does aggregating block-28 features ACROSS TIMESTEPS add beyond the best single t --
measured with the v3/rule-3b harness instead of concat_trajectory's fixed PCA-512-on-concat?

WHY THIS EXISTS (JD challenge, 2026-09-16). The standing "concat <= best single t"
conclusion (6h/6q) rests on an instrument with a RECORDED asymmetry: PCA-512 compresses the
21504-dim concat 42x but the 3072-dim single-t only 6x -- conservative for positive gains,
and flagged in 6h as possibly manufacturing the ens1 negative. The depth axis (section 11)
was re-run under the corrected discipline (base-only PCA, block appended raw, row-shuffled
width-matched null, raw>0 AND DoD>0); the time axis never was. This closes that gap.

Base = block-28 features at the arm's best single t (fixed a priori from the 6h tables --
slightly optimistic base, i.e. conservative against a concat gain). Block = the OTHER six
timesteps' features (18432d). Null = the same block row-shuffled. DiTF normalization applied
offline to everything, as in every offline probe.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

C = 0.1
SEEDS = [0, 1, 2]
PCA_DIM = 512
DISCARD = [154, 1446]
# (cache path, best-t index per the 6h per-timestep tables: ens8 peaks t=260 on RESISC45,
# t=180 on EuroSAT)
ARMS = {
    ("resisc45", "ens8"): (
        "models/n5000_resisc45_oneshot_ens8/resisc45_flux_4118f153+42/multistep_train_feats_oneshot_g1.0.npz",
        2,
    ),
    ("eurosat", "ens8"): (
        "models/n5000_eurosat_oneshot_ens8/eurosat_flux_23ee82c7+42/multistep_train_feats_oneshot_g1.0.npz",
        1,
    ),
    ("resisc45", "inv"): (
        "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
        1,
    ),
    ("eurosat", "inv"): (
        "models/n5000_eurosat_inversion/eurosat_flux_f4ee81c8+42/multistep_train_feats_inversion_g1.0_n50.npz",
        1,
    ),
}


def apply_ditf(feats, mods, discard):
    x = feats.astype(np.float64).copy()
    if discard:
        x[:, :, discard] = 0.0
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mu) / np.sqrt(var + 1e-6)
    x = (1.0 + mods[None, :, 1, :]) * x + mods[None, :, 0, :]
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


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
    for (ds, arm), (path, tbest) in ARMS.items():
        d = np.load(path)
        y = d["labels"]
        n = len(y)
        ts = [int(t) for t in d["timesteps"]]
        feats = apply_ditf(d["feats"], d["mods"], DISCARD)
        base = feats[:, tbest, :]
        others = np.concatenate([feats[:, i, :] for i in range(len(ts)) if i != tbest], axis=1)
        rng = np.random.default_rng(0)
        b = correct(base, None, y)
        pa = correct(base, others, y)
        pn = correct(base, others[rng.permutation(n)], y)
        raw_d = pa - b
        dod = pa - pn
        rl, rh = ci(raw_d)
        dl, dh = ci(dod)
        both = rl > 0 and dl > 0
        print(
            f"\n=== {ds} {arm}  n={n}  base = t{ts[tbest]} alone: {b.mean():.4f} ===\n"
            f"  +other 6 timesteps  {pa.mean():.4f}   raw d {raw_d.mean():+.4f} [{rl:+.4f},{rh:+.4f}]\n"
            f"  vs SHUFFLED others  {pn.mean():.4f}   DoD   {dod.mean():+.4f} [{dl:+.4f},{dh:+.4f}]\n"
            f"  RULE 3b: {'PASS -- time aggregation adds beyond best single t' if both else 'FAIL -- no evidence beyond best single t'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
