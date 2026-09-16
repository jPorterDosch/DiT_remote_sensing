"""Do OTHER blocks add class information beyond block 28? (Hyperfeatures' depth axis,
tested with the v2 discipline: base-only PCA, width-matched row-shuffled null, paired
image-level CI.) n=500 blocksweep caches, t=100, C=0.1."""

import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

C = 0.1
SEEDS = [0, 1, 2]
TIDX = 0  # t=100
K28 = {
    "eurosat": "models/paired_500_eurosat_ens1/eurosat_flux_c54fd240+42/multistep_train_feats_oneshot_g1.0.npz",
    "resisc45": "models/paired_500_resisc45_ens1/resisc45_flux_1c1a9d90+42/multistep_train_feats_oneshot_g1.0.npz",
}
for ds in list(K28):
    if not os.path.exists(K28[ds]):
        c = sorted(glob.glob(f"models/paired_500_{ds}*/*/multistep_train_feats_oneshot_g1.0.npz"))
        K28[ds] = c[0]


def fold(base, blk, y, tr, va):
    sc = StandardScaler().fit(base[tr])
    a, b = sc.transform(base[tr]), sc.transform(base[va])
    k = min(512, a.shape[1], len(tr) - 1)
    if k < a.shape[1]:
        p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    if blk is not None:
        s2 = StandardScaler().fit(blk[tr])
        a = np.hstack([a, s2.transform(blk[tr])])
        b = np.hstack([b, s2.transform(blk[va])])
    m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
    return va, m.predict(b)


def correct(base, blk, y):
    out = np.zeros(len(y))
    for s in SEEDS:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(base, y):
            va_, pred = fold(base, blk, y, tr, va)
            out[va_] += pred == y[va_]
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


for ds in ("eurosat", "resisc45"):
    d28 = np.load(K28[ds])
    y = d28["labels"]
    idx = d28["subset_indices"]
    base = d28["feats"][:, TIDX, :]
    others = {}
    for k in (19, 24, 33, 38, 48):
        pth = glob.glob(f"models/blocksweep_{ds}_k{k}/*/multistep_train_feats_oneshot_g1.0.npz")[0]
        dk = np.load(pth)
        if not np.array_equal(dk["subset_indices"], idx):
            raise SystemExit(f"UNPAIRED {ds} k{k}")
        others[k] = dk["feats"][:, TIDX, :]
    rng = np.random.default_rng(0)
    plateau = np.hstack([others[19], others[24], others[33]])  # 9216d, plateau blocks
    allb = np.hstack([others[k] for k in (19, 24, 33, 38, 48)])  # 15360d, all
    b = correct(base, None, y)
    print(f"\n=== {ds}  n={len(y)}  base = block-28 alone {b.mean():.4f} (t=100, C={C}) ===")
    for name, blk in (("plateau blocks {19,24,33}", plateau), ("all blocks {19..48}", allb)):
        shuf = blk[rng.permutation(len(y))]
        pa = correct(base, blk, y)
        pn = correct(base, shuf, y)
        dd = pa - pn
        lo, hi = ci(dd)
        sig = "SIGNIF" if lo > 0 or hi < 0 else "ns"
        print(
            f"  +{name:<26} {pa.mean():.4f}  vs SHUFFLED null d {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {sig}"
        )
