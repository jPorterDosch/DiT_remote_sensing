"""THE trajectory-as-set test: 7 timesteps x 1 draw vs 1 timestep x 7 draws, at identical
NFE and identical dimensionality.

Concat across the trajectory is worth ~1-1.4 points once eps-noise is handled (concat
analysis, 2026-08-22), but that gain was never separated from generic multi-view diversity.
This is the separation:

  trajectory set: the ens1 cache's 7 timesteps (one eps draw carried through all t)
  draw set:       7 independent eps draws all at t=100 (eps_seed 42..48), same images

Both are 7 forwards and 7x3072 dims. If the trajectory set wins, the multi-t structure
(input SNR ladder + adaLN operator diversity) carries signal beyond redundant views. If the
draw set wins or ties, the concat gain is generic averaging and the trajectory framing dies
at the set level too. Also reported: mean-of-draws (the ens7 analogue) so averaging vs
concatenation is visible.
"""
from __future__ import annotations

import glob

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
C = 0.1
TRAJ = {
    "eurosat": "models/n5000_eurosat_oneshot_ens1/eurosat_flux_0b91191d+42/multistep_train_feats_oneshot_g1.0.npz",
    "resisc45": "models/n5000_resisc45_oneshot_ens1/resisc45_flux_19044381+42/multistep_train_feats_oneshot_g1.0.npz",
}


def _fold(x, y, tr, va):
    sc = StandardScaler().fit(x[tr])
    a, b = sc.transform(x[tr]), sc.transform(x[va])
    if x.shape[1] > 512:
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


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    for ds, traj_path in TRAJ.items():
        dt = np.load(traj_path)
        y, idx = dt["labels"], dt["subset_indices"]
        traj = dt["feats"]  # (N, 7, 3072), one eps across all t
        members = [traj[:, 0, :]]  # member 1: eps_seed 42 at t=100
        for p in sorted(glob.glob(f"models/divctl_{ds}_eps*/*/multistep_train_feats_oneshot_g1.0.npz")):
            dm = np.load(p)
            if not (np.array_equal(dm["labels"], y) and np.array_equal(dm["subset_indices"], idx)):
                raise SystemExit(f"FATAL: {p} not paired")
            members.append(dm["feats"][:, 0, :])  # t=100 slice
        assert len(members) == 7, f"expected 7 members, got {len(members)}"
        draws = np.stack(members, axis=1)  # (N, 7, 3072)

        print(f"\n=== {ds}  n={len(y)}  7 NFE both arms, PCA-512, C={C} ===", flush=True)
        res = {
            "traj_concat (7t x 1eps)": correct(traj.reshape(len(y), -1), y),
            "draw_concat (1t x 7eps)": correct(draws.reshape(len(y), -1), y),
            "draw_mean   (ens7@t100)": correct(draws.mean(axis=1), y),
            "single      (1t x 1eps)": correct(traj[:, 0, :], y),
        }
        for k, v in res.items():
            print(f"  {k}: {v.mean():.4f}", flush=True)
        for name, a, b in [("traj_concat - draw_concat", "traj_concat (7t x 1eps)", "draw_concat (1t x 7eps)"),
                           ("traj_concat - draw_mean", "traj_concat (7t x 1eps)", "draw_mean   (ens7@t100)"),
                           ("draw_concat - draw_mean", "draw_concat (1t x 7eps)", "draw_mean   (ens7@t100)")]:
            d = res[a] - res[b]
            lo, hi = boot(d)
            sig = "SIGNIF" if lo > 0 or hi < 0 else ""
            print(f"    {name}: {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}", flush=True)


if __name__ == "__main__":
    main()
