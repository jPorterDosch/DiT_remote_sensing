"""Paired inversion-vs-oneshot comparison with bootstrap CIs.

experiments/linear_probes.py reports point estimates per arm, which is enough to see a
direction but not to claim one. The arms are extracted from the SAME images with the same
subset seed, so every comparison here is PAIRED per (seed, fold): both arms are trained and
evaluated on identical splits and only the features differ. That removes fold-to-fold
difficulty variance, which is the dominant noise term at n=500 with 45 classes.

Reports, per timestep and for the concatenated trajectory:
    oneshot - inversion, mean + bootstrap 95% CI over (seed, fold) pairs.

A positive interval means one-shot genuinely beats the chain; an interval straddling zero
means the observed gap is not distinguishable from noise, regardless of its sign.

FINDINGS + CAVEATS (RESEARCH_NOTES 4 + audit). Established the original three-arm result
(ens8 >= inversion >= ens1). Audit updates: the (seed,fold) bootstrap unit is
anti-conservative (use paired_image_bootstrap.py at n=5000 for the corrected verdicts); the
ordering is C-invariant across four decades (S4) but per-timestep ens8-vs-inv significance
did not survive the image-level unit.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_geometry_probe import boot_ci, cv_fold_accs  # noqa: E402

SEEDS = [0, 1, 2]


def load_arm(pattern: str) -> dict:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"FATAL: no cache matched {pattern}")
    if len(paths) > 1:
        raise SystemExit(f"FATAL: {len(paths)} caches matched {pattern}: {paths}")
    d = np.load(paths[0], allow_pickle=False)
    return {
        "feats": d["feats"],
        "labels": d["labels"],
        "ts": [int(t) for t in d["timesteps"]],
        "idx": d["subset_indices"],
        "path": paths[0],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inversion", required=True, help="glob for the inversion pooled cache")
    p.add_argument("--oneshot", required=True, help="glob for the oneshot pooled cache")
    p.add_argument("--c", type=float, default=0.01, help="L2 penalty, frozen across arms")
    p.add_argument("--dim", type=int, default=None, help="optional in-fold PCA cap")
    p.add_argument("--label", default="dataset")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    inv, one = load_arm(args.inversion), load_arm(args.oneshot)
    if not np.array_equal(inv["labels"], one["labels"]):
        raise SystemExit("FATAL: arms disagree on labels — not paired")
    if not np.array_equal(inv["idx"], one["idx"]):
        raise SystemExit("FATAL: arms cover different images — not paired")
    if inv["ts"] != one["ts"]:
        raise SystemExit(f"FATAL: timesteps differ: {inv['ts']} vs {one['ts']}")

    y, ts = inv["labels"], inv["ts"]
    n_cls = len(np.unique(y))
    print(
        f"{args.label}: n={len(y)} classes={n_cls} chance={1 / n_cls:.4f} "
        f"timesteps={ts}  C={args.c}  PCA={args.dim}"
    )
    print(f"  inversion: {inv['path']}\n  oneshot:   {one['path']}\n")

    rows = []
    print(f"{'comparison':<18}{'inversion':>11}{'oneshot':>10}{'one-inv':>10}{'95% CI':>22}  verdict")
    print("-" * 88)

    def compare(name: str, xi: np.ndarray, xo: np.ndarray):
        ai = [a for s in SEEDS for a in cv_fold_accs(xi, y, args.c, s, args.dim)]
        ao = [a for s in SEEDS for a in cv_fold_accs(xo, y, args.c, s, args.dim)]
        d = [o - i for o, i in zip(ao, ai, strict=True)]
        lo, hi = boot_ci(d)
        if lo > 0:
            v = "ONESHOT better"
        elif hi < 0:
            v = "INVERSION better"
        else:
            v = "not distinguishable"
        print(
            f"{name:<18}{np.mean(ai):>11.4f}{np.mean(ao):>10.4f}{np.mean(d):>+10.4f}"
            f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}  {v}"
        )
        rows.append(
            {
                "comparison": name,
                "inversion": round(float(np.mean(ai)), 6),
                "oneshot": round(float(np.mean(ao)), 6),
                "delta": round(float(np.mean(d)), 6),
                "ci_lo": round(lo, 6),
                "ci_hi": round(hi, 6),
                "verdict": v,
                "n_pairs": len(d),
                "C": args.c,
                "dataset": args.label,
            }
        )

    for i, t in enumerate(ts):
        compare(f"t={t}", inv["feats"][:, i, :], one["feats"][:, i, :])
    nimg = inv["feats"].shape[0]
    compare("concat(all t)", inv["feats"].reshape(nimg, -1), one["feats"].reshape(nimg, -1))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
