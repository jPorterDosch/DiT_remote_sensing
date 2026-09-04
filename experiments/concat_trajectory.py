"""Trajectory-as-SET: does concatenating the 7 timesteps beat the best single timestep,
and does the inversion trajectory's concat differ from one-shot's, at n=5000?

The arm-level (single-t) advantage is settled as eps-averaging (section 6c/6e). The SET
question is not: concat won everywhere it appeared but was never separated from "more
features". This measures the concat margins at n=5000 with the image-level machinery.
In-fold PCA to 512 keeps the 21504-dim concat comparable to a 3072-dim single t under the
same probe (lossy but equally lossy for every arm; margins, not levels, are the object).
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
C = 0.1
ARMS = {
    "eurosat": {
        "ens1": "models/n5000_eurosat_oneshot_ens1/eurosat_flux_0b91191d+42/multistep_train_feats_oneshot_g1.0.npz",
        "ens8": "models/n5000_eurosat_oneshot_ens8/eurosat_flux_23ee82c7+42/multistep_train_feats_oneshot_g1.0.npz",
        "inv": "models/n5000_eurosat_inversion/eurosat_flux_f4ee81c8+42/multistep_train_feats_inversion_g1.0_n50.npz",
    },
    "resisc45": {
        "ens1": "models/n5000_resisc45_oneshot_ens1/resisc45_flux_19044381+42/multistep_train_feats_oneshot_g1.0.npz",
        "ens8": "models/n5000_resisc45_oneshot_ens8/resisc45_flux_4118f153+42/multistep_train_feats_oneshot_g1.0.npz",
        "inv": "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
    },
}


def _fold(x, y, tr, va):
    sc = StandardScaler().fit(x[tr])
    a, b = sc.transform(x[tr]), sc.transform(x[va])
    if x.shape[1] > 512:
        pca = PCA(n_components=512, svd_solver="randomized", random_state=0).fit(a)
        a, b = pca.transform(a), pca.transform(b)
    m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
    return va, m.predict(b)


def correct(x, y, n_jobs=7):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(x, y))
        for va, pred in Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va) for tr, va in jobs):
            out[va] += (pred == y[va]).astype(float)
    return out / len(SEEDS)


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    for ds, paths in ARMS.items():
        d = {k: np.load(v) for k, v in paths.items()}
        y = d["ens1"]["labels"]
        for v in d.values():
            assert np.array_equal(v["labels"], y)
        print(f"\n=== {ds}  n={len(y)}  C={C}  PCA-512 on concat ===", flush=True)
        res = {}
        for k, v in d.items():
            f = v["feats"]
            ts = [int(t) for t in v["timesteps"]]
            # CORRECTED 2026-08-23: t=100 is NOT the best single t for every arm (ens8 peaks
            # at t=180/260, RESISC45 inv at t=180). Assuming it was handed the concat a
            # handicapped baseline -- the same wrong-baseline defect the audit catalogued.
            # Compute per-image correctness at EVERY t under this protocol and take the
            # best-mean t as the single-t baseline.
            per_t = {t: correct(f[:, i, :], y) for i, t in enumerate(ts)}
            best_t = max(per_t, key=lambda t: per_t[t].mean())
            res[f"{k}_single"] = per_t[best_t]
            res[f"{k}_concat"] = correct(f.reshape(len(y), -1), y)
            print(f"  {k}: best single t={best_t} {res[f'{k}_single'].mean():.4f}  "
                  f"concat {res[f'{k}_concat'].mean():.4f}", flush=True)
        for name, a, b in [("concat gain, ens1", "ens1_concat", "ens1_single"),
                           ("concat gain, ens8", "ens8_concat", "ens8_single"),
                           ("concat gain, inv", "inv_concat", "inv_single"),
                           ("inv concat - ens8 concat", "inv_concat", "ens8_concat"),
                           ("inv concat - ens1 concat", "inv_concat", "ens1_concat")]:
            diff = res[a] - res[b]
            lo, hi = boot(diff)
            sig = "SIGNIF" if lo > 0 or hi < 0 else ""
            print(f"    {name}: {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}", flush=True)


if __name__ == "__main__":
    main()
