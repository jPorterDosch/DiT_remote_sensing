"""Protocol-only accuracy-vs-t curve at a FROZEN probe operating point.

WHY THIS EXISTS. The earlier raw-x_t numbers were read at best-over-pooling and best-over-C
chosen SEPARATELY AT EACH TIMESTEP. That is a selection rule applied to the very quantity
being measured -- the shape of accuracy vs t -- so it can flatten or steepen the curve it is
reporting. Measured on EuroSAT raw x_t (16-d), the decay t=100 -> t=580 is:

    C=0.01   -1.7%      C=1.0   -7.3%      C=10  -19.5%      C=100  -25.6%      best-per-t  -22.5%

i.e. the headline "protocol decay" spans an order of magnitude depending on regularization.
The DiT arm, by contrast, is nearly C-invariant (EuroSAT -11.8% to -12.3% across four decades),
so the asymmetry is real and lives entirely on the raw side. This script therefore:

  1. selects ONE (pooling grid, C) globally -- averaged over the endpoint timesteps, on a
     single CV seed -- and freezes it;
  2. reports the whole curve at that frozen operating point, for ens1 and ens8;
  3. reports the arithmetic positive control ens8-ens1 with paired bootstrap CIs, because a
     null curve is uninterpretable unless the instrument is shown to see an effect that is
     guaranteed to exist (averaging M draws scales eps by 1/sqrt(M));
  4. reports the decay under THREE metrics, because they disagree. Relative accuracy is
     ceiling-compressed and EuroSAT's DiT arm sits at 0.91; in error terms the ordering
     between the protocol and DiT arms can invert. Quoting one metric alone is not honest.

Pooling grids are re-derived offline from the saved `full` array, so no re-extraction is
needed to change the grid.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_geometry_probe import boot_ci  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GRIDS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
N_FOLDS = 5
MAX_ITER = 2000
D_CAP = 512  # in-fold PCA cap; lossless for L2 whenever <= n_train-1 and >= rank


def repool(path: str, g: int):
    """Re-derive a g x g average-pooled feature from the saved unpooled `full` cache."""
    d = np.load(path)
    f = d["feats"]
    n, k, dim = f.shape
    hw = int(round((dim // 16) ** 0.5))
    c = dim // (hw * hw)
    x = torch.from_numpy(f).reshape(n, k, c, hw, hw)
    p = F.adaptive_avg_pool2d(x.reshape(n * k, c, hw, hw), (g, g))
    return p.reshape(n, k, c * g * g).numpy(), d["labels"], [int(t) for t in d["timesteps"]]


def _fold_acc(x, y, tr, va, c, d_cap):
    sc = StandardScaler().fit(x[tr])
    a, b = sc.transform(x[tr]), sc.transform(x[va])
    k = min(d_cap, x.shape[1], len(tr) - 1)
    if k < x.shape[1]:
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
        a, b = pca.transform(a), pca.transform(b)
    m = LogisticRegression(C=c, max_iter=MAX_ITER).fit(a, y[tr])
    return float((m.predict(b) == y[va]).mean())


def fold_accs(x, y, c, seeds, n_jobs):
    """Per-(seed,fold) accuracies, in a fixed order so arms can be paired elementwise."""
    jobs = [
        (tr, va)
        for s in seeds
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(x, y)
    ]
    return Parallel(n_jobs=n_jobs)(delayed(_fold_acc)(x, y, tr, va, c, D_CAP) for tr, va in jobs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dir", default="models/raw_xt_n5000")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    paths = {m: f"{args.dir}/{args.dataset}_rawxt_ens{m}_full.npz" for m in (1, 8)}
    for m, path in paths.items():
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: missing {path}")

    print(f"{args.dataset}  dir={args.dir}", flush=True)
    # Hard-fail on pairing, matching arm_paired_ci.py / state_baseline_compare.py /
    # raw_xt_probe.py / velocity_probe.py. The ens1 and ens8 caches are written by a single
    # raw_xt_baseline.py run so they are paired by construction today -- but this script
    # carries the project's only arithmetic-guaranteed positive control (ens8-ens1), and it
    # was the one comparison script without the assert every other one has.
    ref = np.load(paths[1])
    for m in (1, 8):
        d = np.load(paths[m])
        if not np.array_equal(d["labels"], ref["labels"]):
            raise SystemExit(f"FATAL: {paths[m]} disagrees on labels -- not paired")
        if not np.array_equal(d["subset_indices"], ref["subset_indices"]):
            raise SystemExit(f"FATAL: {paths[m]} covers different images -- not paired")
        if [int(t) for t in d["timesteps"]] != [int(t) for t in ref["timesteps"]]:
            raise SystemExit(f"FATAL: {paths[m]} disagrees on timesteps -- not paired")

    pooled = {}
    for m in (1, 8):
        for g in GRIDS:
            pooled[(m, g)], y, ts = repool(paths[m], g)
    n_cls = len(np.unique(y))
    print(f"n={len(y)} classes={n_cls} chance={1 / n_cls:.4f} ts={ts}\n", flush=True)

    # --- Stage 1: freeze (grid, C) ONCE, on the endpoints, on a single CV seed, ens1 only.
    ends = [ts.index(ts[0]), ts.index(ts[-1])]
    print("selecting operating point (endpoints, seed 0, ens1):", flush=True)
    best, best_acc = None, -1.0
    for g in GRIDS:
        x = pooled[(1, g)]
        for c in C_GRID:
            a = float(np.mean([np.mean(fold_accs(x[:, k, :], y, c, [0], args.n_jobs))
                               for k in ends]))
            print(f"  grid={g}x{g} dims={x.shape[2]:<5} C={c:<8} endpoint-mean={a:.4f}", flush=True)
            if a > best_acc:
                best, best_acc = (g, c), a
    g_star, c_star = best
    print(f"\nFROZEN: grid={g_star}x{g_star} ({pooled[(1, g_star)].shape[2]} dims)  C={c_star}\n",
          flush=True)

    # --- Stage 2: the full curve at that frozen point, both ensemble sizes.
    print(f"{'t':>6}{'eta':>7}{'ens1':>9}{'ens8':>9}{'ens8-ens1':>11}{'95% CI':>24}", flush=True)
    print("-" * 66, flush=True)
    curve = {}
    for k, t in enumerate(ts):
        f1 = fold_accs(pooled[(1, g_star)][:, k, :], y, c_star, SEEDS, args.n_jobs)
        f8 = fold_accs(pooled[(8, g_star)][:, k, :], y, c_star, SEEDS, args.n_jobs)
        d = [b - a for a, b in zip(f1, f8, strict=True)]
        lo, hi = boot_ci(d)
        curve[t] = (float(np.mean(f1)), float(np.mean(f8)))
        eta = (t / 1000) / (1 - t / 1000)
        print(f"{t:>6}{eta:>7.2f}{np.mean(f1):>9.4f}{np.mean(f8):>9.4f}"
              f"{np.mean(d):>+11.4f}   [{lo:+.4f}, {hi:+.4f}]", flush=True)

    # --- Decay, three ways. They disagree; that disagreement is the finding.
    lo_t, hi_t = ts[0], ts[-1]
    chance = 1 / n_cls
    print(f"\ndecay {lo_t} -> {hi_t}   (chance={chance:.4f})", flush=True)
    print(f"{'arm':>6}{'acc rel':>11}{'above-ch rel':>15}{'err ratio':>12}", flush=True)
    for i, name in ((0, "ens1"), (1, "ens8")):
        a, b = curve[lo_t][i], curve[hi_t][i]
        print(f"{name:>6}{(b - a) / a * 100:>+10.1f}%"
              f"{((b - chance) / (a - chance) - 1) * 100:>+14.1f}%"
              f"{(1 - b) / (1 - a):>11.2f}x", flush=True)


if __name__ == "__main__":
    main()
