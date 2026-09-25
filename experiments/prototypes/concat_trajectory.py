"""Trajectory-as-SET: does concatenating the 7 timesteps beat the best single timestep,
and does the inversion trajectory's concat differ from one-shot's, at n=5000?

The arm-level (single-t) advantage is settled as eps-averaging (section 6c/6e). The SET
question is not: concat won everywhere it appeared but was never separated from "more
features". This measures the concat margins at n=5000 with the image-level machinery.
In-fold PCA to 512 keeps the 21504-dim concat comparable to a 3072-dim single t under the
same probe (lossy but equally lossy for every arm; margins, not levels, are the object).

FINDINGS (v2, 2026-08-23, RESEARCH_NOTES 6h). With honest best-t baselines, concatenating
the trajectory adds NOTHING significant on any arm, either dataset (ens8 +0.0016/+0.0025 ns,
inv +0.0036/+0.0029 ns) and HURTS ens1 (-0.0061/-0.0103 SIG -- but see the CAVEAT below:
these two SIG cells are NOT established). v1's "+0.014 SIG concat gain"
was a wrong-baseline artifact (t=100 assumed best; ens8 peaks at t=180/260) -- see 6g.

CAVEAT (2026-09-06 review, RESEARCH_NOTES "Concat v2"): best_t is chosen by argmax over the
SAME per-image correctness vectors the paired bootstrap then consumes -- a winner's curse
over 7 correlated timesteps that biases the single-t baseline UP and the concat delta DOWN,
plausibly by the size of the ens1 effects. The ns rows are safe (a downward-biased estimate
that is still ns supports "no gain" conservatively); the ens1 SIG-negative cells must not be
cited until best_t is selected NESTED (argmax on training folds only).
"""

from __future__ import annotations

import os

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


def correct_per_seed(x, y, n_jobs=7):
    """-> (S, N) per-seed per-image correctness, plus per-seed fold membership.
    Same fits as before; the per-seed detail is retained so best-t can be selected
    NESTED (leave-fold-out) instead of on the evaluation data."""
    out = np.zeros((len(SEEDS), len(y)))
    folds = []
    for si, s in enumerate(SEEDS):
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(x, y))
        folds.append([va for _, va in jobs])
        for va, pred in Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va) for tr, va in jobs):
            out[si, va] = (pred == y[va]).astype(float)
    return out, folds


def correct(x, y, n_jobs=7):
    out, _ = correct_per_seed(x, y, n_jobs)
    return out.mean(axis=0)


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    for ds, paths in ARMS.items():
        d = {k: np.load(v) for k, v in paths.items()}
        y = d["ens1"]["labels"]
        ref_idx = d["ens1"]["subset_indices"]
        for k, v in d.items():
            # labels alone cannot detect disjoint subsets (6o-F): sorted stratified labels
            # are determined by (subset_size, n_classes) regardless of subset_seed.
            if not np.array_equal(v["subset_indices"], ref_idx):
                raise RuntimeError(f"arm {k} not paired: subset_indices differ")
            if not np.array_equal(v["labels"], y):
                raise RuntimeError(f"arm {k} not paired: labels differ")
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
            # v2 fold structure depends only on (y, seed), so it is IDENTICAL across
            # timesteps of one arm -- folds from any t serve all t.
            per_t_seed = {}
            folds = None
            for i, t in enumerate(ts):
                per_t_seed[t], folds = correct_per_seed(f[:, i, :], y)
            # NESTED best-t (6m item 1 / CLAUDE.md rule 1): for each (seed, fold), t* is
            # the argmax of mean correctness over the images NOT in that fold, then fold
            # images are scored at THEIR OWN t*. The old argmax over the full vectors
            # selected on the same images it then evaluated -- a winner's curse over 7
            # correlated timesteps, biased against concat. Kept below as *_single_BIASED
            # for comparison only.
            nested = np.zeros((len(SEEDS), len(y)))
            picks = []
            for si in range(len(SEEDS)):
                for va in folds[si]:
                    mask = np.ones(len(y), bool)
                    mask[va] = False
                    t_star = max(ts, key=lambda t: per_t_seed[t][si, mask].mean())
                    nested[si, va] = per_t_seed[t_star][si, va]
                    picks.append(t_star)
            res[f"{k}_single"] = nested.mean(axis=0)
            mean_per_t = {t: per_t_seed[t].mean() for t in ts}
            best_t_biased = max(mean_per_t, key=mean_per_t.get)
            res[f"{k}_single_BIASED"] = per_t_seed[best_t_biased].mean(axis=0)
            res[f"{k}_concat"] = correct(f.reshape(len(y), -1), y)
            from collections import Counter

            print(
                f"  {k}: nested single-t {res[f'{k}_single'].mean():.4f} "
                f"(picks {dict(Counter(picks))}; biased argmax t={best_t_biased} "
                f"{res[f'{k}_single_BIASED'].mean():.4f})  concat {res[f'{k}_concat'].mean():.4f}",
                flush=True,
            )
        for name, a, b in [
            ("concat gain, ens1", "ens1_concat", "ens1_single"),
            ("concat gain, ens8", "ens8_concat", "ens8_single"),
            ("concat gain, inv", "inv_concat", "inv_single"),
            ("inv concat - ens8 concat", "inv_concat", "ens8_concat"),
            ("inv concat - ens1 concat", "inv_concat", "ens1_concat"),
        ]:
            diff = res[a] - res[b]
            lo, hi = boot(diff)
            sig = "SIGNIF" if lo > 0 or hi < 0 else ""
            print(f"    {name}: {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}", flush=True)
        os.makedirs("results", exist_ok=True)
        np.savez(
            f"results/concat_v2_{ds}.npz",
            labels=y,
            **{f"correct__{k}": v for k, v in res.items()},
        )
        print(f"  wrote results/concat_v2_{ds}.npz", flush=True)


if __name__ == "__main__":
    main()
