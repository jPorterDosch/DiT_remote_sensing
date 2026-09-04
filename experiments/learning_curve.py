"""Accuracy vs n, subsampled from the n=5000 caches. Is n=5000 converged or still climbing?

n=500 -> n=5000 moved RESISC45 DiT from 0.595 to 0.840 -- a 25-point swing that invalidated a
round of conclusions. Before quoting any n=5000 number as final, this measures whether the
curve has flattened. Class-stratified subsampling; each n evaluated with the same 3x5-fold
protocol; each arm at its frozen C.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
NS = [500, 1000, 2000, 3500, 5000]
ARMS = {
    "eurosat DiT ens1 (C=0.01)": ("models/n5000_eurosat_oneshot_ens1/eurosat_flux_0b91191d+42/multistep_train_feats_oneshot_g1.0.npz", 0.01),
    "resisc45 DiT ens1 (C=0.1)": ("models/n5000_resisc45_oneshot_ens1/resisc45_flux_19044381+42/multistep_train_feats_oneshot_g1.0.npz", 0.1),
}
T_IDX = 0  # t=100


def _fold(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
    return float((m.predict(sc.transform(x[va])) == y[va]).mean())


def sub(y, n, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        keep.append(rng.choice(idx, min(n // len(np.unique(y)), len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def main():
    for name, (path, c) in ARMS.items():
        d = np.load(path)
        f, y = d["feats"][:, T_IDX, :], d["labels"]
        print(f"\n{name}  t=100", flush=True)
        prev = None
        for n in NS:
            idx = sub(y, n)
            xs, ys = f[idx], y[idx]
            jobs = [(tr, va) for s in SEEDS
                    for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(xs, ys)]
            a = float(np.mean(Parallel(n_jobs=5)(
                delayed(_fold)(xs, ys, tr, va, c) for tr, va in jobs)))
            gain = "" if prev is None else f"  (+{a - prev:.4f})"
            print(f"  n={len(idx):<6} acc={a:.4f}{gain}", flush=True)
            prev = a


if __name__ == "__main__":
    main()
