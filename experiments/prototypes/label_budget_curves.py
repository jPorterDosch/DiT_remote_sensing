"""Q3 (9.3): label-budget curves, FLUX vs DINOv2, RESISC45, cached features only (CPU).

Question: does the 6ac ordering (DINOv2 > FLUX by +6.4 linear) hold at low label budgets,
or do flow features have a few-shot niche? 10/25/50/100 labels-per-class, 3 seeds; train
rows drawn stratified from the 6ac 5,000; eval = the remaining images (identical eval set
across arms within a (budget, seed) cell -> per-image PAIRED deltas). Instrument otherwise
= section-13: in-fold scaler, LR C=0.1, PCA-512 protection on the FLUX t-best block only.
Pre-registered read in RESEARCH_NOTES 9.3 Q3.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from experiments.prototypes._absorption_harness import C as HC, RESISC45_BASE, RESISC45_TBEST_IDX, ci, ditf  # noqa: E402

BUDGETS = (10, 25, 50, 100)
SEEDS = (0, 1, 2)


def draw(y, per_class, seed):
    rng = np.random.default_rng(seed)
    tr = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                                 for c in np.unique(y)]))
    ev = np.setdiff1d(np.arange(len(y)), tr)
    return tr, ev


def fit_flux(base, others, y, tr, ev):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(base[tr])
        a, b = sc.transform(base[tr]), sc.transform(base[ev])
        k = min(512, a.shape[1], len(tr) - 1)
        if k < a.shape[1]:
            p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
            a, b = p.transform(a), p.transform(b)
        s2 = StandardScaler().fit(others[tr])
        a = np.hstack([a, s2.transform(others[tr])])
        b = np.hstack([b, s2.transform(others[ev])])
        m = LogisticRegression(C=HC, max_iter=2000).fit(a, y[tr])
        nw = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return (m.predict(b) == y[ev]).astype(np.int8), nw


def fit_plain(X, y, tr, ev):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=HC, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        nw = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return (m.predict(sc.transform(X[ev])) == y[ev]).astype(np.int8), nw


def cell(base, others, dino, y, per_class, seed):
    tr, ev = draw(y, per_class, seed)
    cf, wf = fit_flux(base, others, y, tr, ev)
    cd, wd = fit_plain(dino, y, tr, ev)
    d = cd.astype(int) - cf.astype(int)
    lo, hi = ci(d)
    return per_class, seed, cf.mean(), cd.mean(), d.mean(), lo, hi, wf + wd, cf, cd, ev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-jobs", type=int, default=6)
    args = p.parse_args()

    inv = np.load(RESISC45_BASE)
    y = inv["labels"].astype(np.int64)
    fi = ditf(inv["feats"], inv["mods"])
    base = fi[:, RESISC45_TBEST_IDX, :]
    others = np.concatenate([fi[:, i, :] for i in range(fi.shape[1]) if i != RESISC45_TBEST_IDX], axis=1)
    dn = np.load("results/dinov2_resisc45_feats_n5000.npz", allow_pickle=True)
    dino = np.concatenate([dn["cls"], dn["mp"]], axis=1).astype(np.float32)
    r = np.load("results/dinov2_resisc45_paired.npz", allow_pickle=True)
    assert np.array_equal(r["labels"], y)

    jobs = [(b, s) for b in BUDGETS for s in SEEDS]
    out = Parallel(n_jobs=args.n_jobs)(
        delayed(cell)(base, others, dino, y, b, s) for b, s in jobs)
    print(f"{'labels/cls':>10} {'seed':>4} {'FLUX':>7} {'DINOv2':>7} {'delta':>8} {'95% CI':>20} warn")
    store = {"labels": y}
    for per_class, seed, af, ad, dm, lo, hi, nw, cf, cd, ev in out:
        print(f"{per_class:>10} {seed:>4} {af:7.4f} {ad:7.4f} {dm:+8.4f} [{lo:+.4f},{hi:+.4f}]  {nw}")
        store[f"flux_b{per_class}_s{seed}"] = cf
        store[f"dino_b{per_class}_s{seed}"] = cd
        store[f"ev_b{per_class}_s{seed}"] = ev
    for b in BUDGETS:
        ds = [dm for pc, s, af, ad, dm, lo, hi, nw, cf, cd, ev in out if pc == b]
        print(f"budget {b:>3}: mean delta over seeds {np.mean(ds):+.4f}")
    np.savez("results/label_budget_curves_resisc45.npz", **store,
             protocol=np.array("Q3: 10/25/50/100 per class x 3 seeds, sec-13 arms, paired on shared eval"))
    print("cached to results/label_budget_curves_resisc45.npz")


if __name__ == "__main__":
    main()
