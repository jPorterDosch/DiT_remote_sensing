"""Accuracy-vs-t for ONE feature arm at a frozen probe operating point.

The companion to protocol_shape.py, for arms that come from a single cache (no pooling grid
to choose): DiT one-shot, DiT inversion, raw z_t. Same discipline -- C is selected ONCE on
the endpoint timesteps and frozen for the whole curve, never re-selected per t, because the
curve's SHAPE is the quantity being measured.

No PCA: for a DiT cache D=3072 < n_train, so any cap below 3072 would be lossy rather than
the lossless rotation it is in the wide-D raw-x_t case.

Reports decay under three metrics because they disagree, and the disagreement is load-bearing:
relative accuracy is ceiling-compressed, and the EuroSAT DiT arm sits near 0.91.

FINDINGS. Produced every per-arm n=5000 curve in RESEARCH_NOTES 6c/6e/6f: one-shot ens1
(EuroSAT 0.9564->0.8555, RESISC45 0.8405->0.6127), ens8 and inversion (~-3.5%/-3.7% EuroSAT,
~-11%/-12% RESISC45 -- the two arms trace the SAME curve), untrained (6b), fixed-cond (6f).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_geometry_probe import boot_ci  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
SEEDS = [0, 1, 2]
N_FOLDS = 5
MAX_ITER = 2000


def _fold_acc(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    a, b = sc.transform(x[tr]), sc.transform(x[va])
    m = LogisticRegression(C=c, max_iter=MAX_ITER).fit(a, y[tr])
    return float((m.predict(b) == y[va]).mean())


def fold_accs(x, y, c, seeds, n_jobs):
    jobs = [
        (tr, va)
        for s in seeds
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(x, y)
    ]
    return Parallel(n_jobs=n_jobs)(delayed(_fold_acc)(x, y, tr, va, c) for tr, va in jobs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out-npz", default=None, help="save per-fold accs for later pairing")
    args = p.parse_args()

    d = np.load(args.cache)
    f, y = d["feats"], d["labels"]
    ts = [int(t) for t in d["timesteps"]]
    n_cls = len(np.unique(y))
    label = args.label or os.path.basename(args.cache)
    print(
        f"{label}\n  {args.cache}\n  feats={f.shape} n={len(y)} classes={n_cls} "
        f"chance={1 / n_cls:.4f} ts={ts}\n",
        flush=True,
    )

    ends = [0, len(ts) - 1]
    print("selecting C (endpoints, seed 0):", flush=True)
    best, best_acc = None, -1.0
    for c in C_GRID:
        a = float(np.mean([np.mean(fold_accs(f[:, k, :], y, c, [0], args.n_jobs)) for k in ends]))
        print(f"  C={c:<8} endpoint-mean={a:.4f}", flush=True)
        if a > best_acc:
            best, best_acc = c, a
    print(f"\nFROZEN: C={best}\n", flush=True)

    print(f"{'t':>6}{'acc':>10}", flush=True)
    print("-" * 16, flush=True)
    per_t, curve = {}, {}
    for k, t in enumerate(ts):
        fa = fold_accs(f[:, k, :], y, best, SEEDS, args.n_jobs)
        per_t[t], curve[t] = fa, float(np.mean(fa))
        print(f"{t:>6}{curve[t]:>10.4f}", flush=True)

    lo_t, hi_t = ts[0], ts[-1]
    a, b = curve[lo_t], curve[hi_t]
    ch = 1 / n_cls
    d_ends = [x - z for z, x in zip(per_t[lo_t], per_t[hi_t], strict=True)]
    ci = boot_ci(d_ends)
    print(
        f"\ndecay {lo_t} -> {hi_t}:  {a:.4f} -> {b:.4f}   paired 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]",
        flush=True,
    )
    print(
        f"  acc rel {(b - a) / a * 100:+.1f}%   above-ch rel "
        f"{((b - ch) / (a - ch) - 1) * 100:+.1f}%   err ratio {(1 - b) / (1 - a):.2f}x",
        flush=True,
    )

    if args.out_npz:
        np.savez(
            args.out_npz, ts=np.array(ts), C=np.array(best), accs=np.array([per_t[t] for t in ts]), labels=y
        )
        print(f"\nwrote {args.out_npz}", flush=True)


if __name__ == "__main__":
    main()
