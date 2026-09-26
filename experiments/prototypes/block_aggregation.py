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
T_GRID = [100, 180, 260, 340, 420, 500, 580]
DS_ARGS = {
    "eurosat": "data/eurosat/EuroSAT_RGB --img-size 224 224",
    "resisc45": "data/resisc45/NWPU-RESISC45 --img-size 256 256",
}


def regen_cmd(ds, k, save_dir):
    # The extraction that produced every cache here (experiments/overnight_gpu.sh blocksweep).
    return (
        f"python run.py --task extract --model.name flux --model.ensemble-size 1 --dataset.name {ds} "
        f"--dataset.path {DS_ARGS[ds]} --t {' '.join(map(str, T_GRID))} --extraction-mode ONESHOT "
        f"--subset-size 500 --subset-seed 42 --eps-seed 42 --k {k} --guidance-scale 1.0 --save-dir {save_dir}"
    )


def load_pinned(pth, ds, k, save_dir):
    """Fail fast (rules 9/11): no fallback to a look-alike cache, and the t-grid must match."""
    if not os.path.exists(pth):
        raise SystemExit(f"missing cache {pth}\nregenerate with:\n  {regen_cmd(ds, k, save_dir)}")
    d = np.load(pth)
    if [int(t) for t in d["timesteps"]] != T_GRID:
        raise SystemExit(
            f"{pth}: timesteps {list(d['timesteps'])} != {T_GRID} (TIDX={TIDX} would not be t=100)\n"
            f"regenerate with:\n  {regen_cmd(ds, k, save_dir)}"
        )
    return d


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
    d28 = load_pinned(K28[ds], ds, 28, f"models/paired_500_{ds}_ens1")
    y = d28["labels"]
    idx = d28["subset_indices"]
    base = d28["feats"][:, TIDX, :]
    others = {}
    for k in (19, 24, 33, 38, 48):
        hits = glob.glob(f"models/blocksweep_{ds}_k{k}/*/multistep_train_feats_oneshot_g1.0.npz")
        if len(hits) != 1:
            raise SystemExit(
                f"expected exactly 1 blocksweep cache for {ds} k{k}, found {hits}\n"
                f"regenerate with:\n  {regen_cmd(ds, k, f'models/blocksweep_{ds}_k{k}')}"
            )
        dk = load_pinned(hits[0], ds, k, f"models/blocksweep_{ds}_k{k}")
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
