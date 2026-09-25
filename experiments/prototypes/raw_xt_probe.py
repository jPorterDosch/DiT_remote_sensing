"""Probe the raw x_t baseline and compare it, paired, against the DiT one-shot arm.

Reports per timestep and for the concatenated trajectory:
  - raw x_t accuracy at each pooling (mean / q2 / full)
  - DiT one-shot accuracy at the SAME ensemble size
  - the paired delta (DiT - raw) with a bootstrap 95% CI over (seed, fold)

REGULARIZATION. The arms differ in dimensionality by up to 3 orders of magnitude (16 vs
12544 vs 3072), and section 2 established that a frozen penalty across unequal D is exactly
how a dimensionality confound sneaks in. So C is swept per feature set and the BEST is
reported. That is deliberately generous to the baseline -- an upper bound on what is
linearly available in x_t is the conservative choice when the conclusion of interest is
"the DiT features beat it".

BASELINE STRENGTH.
MEASURED (2026-08-16) -- `full` is NOT an upper bound, contrary to the obvious expectation.
Spatial mean-pooling AVERAGES eps over h*w latent positions, denoising by ~sqrt(h*w) (~28x at
28x28), while `full` keeps every position's noise and hands 12,544 dims to 500 samples. At
t=100 on EuroSAT: mean 0.666 > q2 0.594 > full 0.478 -- monotonically WORSE with more
dimensions, the opposite of the section-2 confound. The baseline's strength is therefore the
MAX over poolings, and which pooling wins shifts with t.

So the paired delta is taken against the BEST pooling per comparison, which is the
conservative choice; taking it against `full` alone overstates the DiT advantage by ~0.19 at
t=100 on EuroSAT ens=1.

PAIRING. Raw and DiT arms are built from the same stratified subset in the same order, so
folds are identical and deltas are paired per (seed, fold); the loader asserts this.

FINDINGS (RESEARCH_NOTES 6). EuroSAT: the Tier-1 pattern reproduces against the raw-latent
baseline. Audit S3/S4: the best-over-18-configs selection is NOT materially inflating the
baseline (nested check: bias -0.007, wrong sign for the worry). Terminology caveat: this
baseline is VAE-latent, not model-free -- pooled pixels reach only 43% of above-chance vs
the latent's 74% (EuroSAT).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from token_geometry_probe import C_GRID, MAX_ITER, N_FOLDS, boot_ci  # noqa: E402


def _assert_ens(path: str, expected: int) -> None:
    """ens1 and ens8 one-shot caches share a filename; labels/subset_indices/timesteps all
    match, so a swapped --dit-one-ens1/--dit-one-ens8 passes every existing guard and flips
    the sign of the reported columns. The npz already stores the ensemble size -- read it."""
    import json
    import os

    meta = os.path.splitext(path)[0] + "_meta.json"
    if os.path.exists(meta):
        got = json.load(open(meta)).get("ensemble_size")
        if got is not None and int(got) != expected:
            raise SystemExit(f"FATAL: {path} has ensemble_size={got}, expected {expected} — arms swapped?")


SEEDS = [0, 1, 2]
POOLINGS = ("mean", "q2", "full")


def load(path_or_glob: str) -> dict:
    paths = sorted(glob.glob(path_or_glob))
    if len(paths) != 1:
        raise SystemExit(f"FATAL: {len(paths)} files matched {path_or_glob}: {paths}")
    d = np.load(paths[0], allow_pickle=False)
    return {
        "feats": d["feats"],
        "labels": d["labels"],
        "ts": [int(t) for t in d["timesteps"]],
        "idx": d["subset_indices"],
        "path": paths[0],
    }


def best_over_c(x: np.ndarray, y: np.ndarray, d_cap: int) -> tuple[float, float, list[float]]:
    """-> (best mean acc over C_GRID, best C, per-fold accs at best C).

    d_cap drives an in-fold PCA. At d_cap = n_train - 1 the projection is LOSSLESS for any
    feature set wider than that: an L2-regularized linear solution lies in the span of the
    training rows, so directions outside it carry zero coefficient, and the penalty is
    invariant to the orthonormal change of basis. It is skipped outright when the feature set
    is already narrower (the 16- and 64-dim poolings). Without it the concat at full pooling
    is 87,808 dims and the sweep is intractable for no statistical gain.

    The scaler and PCA do NOT depend on C, so they are fitted ONCE per fold and the C grid is
    swept on the projected data. Folding them inside the C loop (the obvious way, via
    cv_fold_accs) refits the projection 6x per fold, and at 87,808 dims that projection is
    the dominant cost -- it is what made the first attempt exceed its wall clock.
    """
    per_c: dict[float, list[float]] = {c: [] for c in C_GRID}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, va in skf.split(x, y):
            sc = StandardScaler().fit(x[tr])
            a, b = sc.transform(x[tr]), sc.transform(x[va])
            k = min(d_cap, x.shape[1], len(tr) - 1) if d_cap is not None else None
            if k is not None and k < x.shape[1]:
                pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
                a, b = pca.transform(a), pca.transform(b)
            for c in C_GRID:
                m = LogisticRegression(C=c, max_iter=MAX_ITER).fit(a, y[tr])
                per_c[c].append(float((m.predict(b) == y[va]).mean()))
    best_c = max(C_GRID, key=lambda c: float(np.mean(per_c[c])))
    return float(np.mean(per_c[best_c])), best_c, per_c[best_c]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="models/raw_xt")
    p.add_argument("--dataset", required=True)
    p.add_argument("--ens", type=int, required=True)
    p.add_argument("--dit", required=True, help="glob for the DiT one-shot pooled cache")
    p.add_argument(
        "--pca", type=int, default=400, help="in-fold PCA cap; 400 = n_train-1 at n=500/5-fold, i.e. lossless"
    )
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    dit = load(args.dit)

    # The DiT cache's filename does not encode ensemble size; only meta.json does.

    _assert_ens(dit["path"], args.ens)
    y, ts = dit["labels"], dit["ts"]
    n_cls = len(np.unique(y))
    print(f"{args.dataset} ens={args.ens}  n={len(y)}  classes={n_cls}  chance={1 / n_cls:.4f}")
    print(f"  DiT: {dit['path']}\n")

    raws = {}
    for how in POOLINGS:
        r = load(os.path.join(args.raw_dir, f"{args.dataset}_rawxt_ens{args.ens}_{how}.npz"))
        if not np.array_equal(r["labels"], y):
            raise SystemExit(f"FATAL: labels differ for pooling {how} — not paired")
        if not np.array_equal(r["idx"], dit["idx"]):
            raise SystemExit(f"FATAL: subset_indices differ for pooling {how} — not paired")
        if r["ts"] != ts:
            raise SystemExit(f"FATAL: timesteps differ for pooling {how}")
        raws[how] = r

    dims = {h: raws[h]["feats"].shape[2] for h in POOLINGS}
    print(
        f"dims: raw mean={dims['mean']}  q2={dims['q2']}  full={dims['full']}  DiT={dit['feats'].shape[2]}\n"
    )

    rows = []
    hdr = f"{'t':<8}{'eta':>7}" + "".join(f"{'raw ' + h:>11}" for h in POOLINGS)
    hdr += f"{'best':>7}{'DiT':>10}{'DiT-best':>10}{'95% CI':>22}"
    print(hdr)
    print("-" * len(hdr))

    def one(name: str, t_nom: int | None, sl):
        eta = "" if t_nom is None else f"{(t_nom / 1000) / (1 - t_nom / 1000):>7.2f}"
        accs, folds = {}, {}
        for how in POOLINGS:
            m, c, f = best_over_c(sl(raws[how]["feats"]), y, args.pca)
            accs[how], folds[how] = m, f
        dm, dc, df = best_over_c(sl(dit["feats"]), y, args.pca)
        # Conservative baseline: the strongest pooling, not a fixed one (see MEASURED above).
        win = max(POOLINGS, key=lambda h: accs[h])
        d = [a - b for a, b in zip(df, folds[win], strict=True)]
        lo, hi = boot_ci(d)
        print(
            f"{name:<8}{eta}"
            + "".join(f"{accs[h]:>11.4f}" for h in POOLINGS)
            + f"{win:>7}{dm:>10.4f}{np.mean(d):>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
        )
        rows.append(
            {
                "dataset": args.dataset,
                "ens": args.ens,
                "comparison": name,
                "eta": None if t_nom is None else round((t_nom / 1000) / (1 - t_nom / 1000), 4),
                **{f"raw_{h}": round(accs[h], 6) for h in POOLINGS},
                "raw_best_pooling": win,
                "raw_best": round(accs[win], 6),
                "dit": round(dm, 6),
                "delta_dit_minus_rawbest": round(float(np.mean(d)), 6),
                "ci_lo": round(lo, 6),
                "ci_hi": round(hi, 6),
                "verdict": ("DiT better" if lo > 0 else "RAW better" if hi < 0 else "not distinguishable"),
            }
        )

    for i, t in enumerate(ts):
        one(f"t={t}", t, lambda f, i=i: f[:, i, :])
    one("concat(all t)", None, lambda f: f.reshape(f.shape[0], -1))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
