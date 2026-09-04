"""Paired arm comparison at n=5000 with an IMAGE-LEVEL bootstrap.

The section-4 headline was established with `boot_ci` over 15 (seed, fold) accuracies, which
finding A1 (section 8) shows is anti-conservative: the 3 seeds are re-partitions of the same
images, so the CI is over CV splits of a fixed sample, not over images. Under an image-level
bootstrap the per-timestep ens8-vs-inversion verdicts died at n=500. This re-runs the
comparison at n=5000 with the correct unit.

Design: for each arm, out-of-fold predictions per (seed, image); a per-image correctness score
averaged over the 3 seeds; PAIRED per-image differences between arms (same images, hard-fail
on pairing); bootstrap over the 5000 images. One shared C across arms per dataset (0.1 -- the
modal frozen choice; section 8's C-sweep shows the ORDERING is C-invariant, and a shared C
avoids the different-operating-point problem for this paired test).

Comparisons, per timestep and endpoint: ens8 - inversion, inversion - ens1, ens8 - ens1.
"""
from __future__ import annotations

import sys

import numpy as np
from joblib import Parallel, delayed
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


def _fold(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
    return va, m.predict(sc.transform(x[va]))


def per_image_correct(x, y, n_jobs):
    """(N,) correctness in [0,1], averaged over seeds; each image out-of-fold once per seed."""
    correct = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(x, y))
        out = Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va, C) for tr, va in jobs)
        for va, pred in out:
            correct[va] += (pred == y[va]).astype(float)
    return correct / len(SEEDS)


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for ds, paths in ARMS.items():
        d = {k: np.load(v) for k, v in paths.items()}
        ref = d["ens1"]
        for k, v in d.items():
            if not np.array_equal(v["labels"], ref["labels"]) or not np.array_equal(
                v["subset_indices"], ref["subset_indices"]
            ):
                raise SystemExit(f"FATAL: {ds}/{k} not paired")
        y = ref["labels"]
        ts = [int(t) for t in ref["timesteps"]]
        print(f"\n=== {ds}  n={len(y)}  C={C} (shared)  seeds={SEEDS} ===", flush=True)
        for t_idx in (0, len(ts) - 1):
            corr = {k: per_image_correct(v["feats"][:, t_idx, :], y, n_jobs) for k, v in d.items()}
            print(f"  t={ts[t_idx]}  acc: " + "  ".join(f"{k}={corr[k].mean():.4f}" for k in corr),
                  flush=True)
            for a, b in [("ens8", "inv"), ("inv", "ens1"), ("ens8", "ens1")]:
                diff = corr[a] - corr[b]
                lo, hi = boot(diff)
                verdict = ("SIGNIF " if lo > 0 or hi < 0 else "       ")
                print(f"    {a}-{b}: {diff.mean():+.4f}  image-CI [{lo:+.4f}, {hi:+.4f}] {verdict}",
                      flush=True)


if __name__ == "__main__":
    main()
