"""Does the model's VELOCITY carry class signal beyond the states it is evaluated at?

WHY THIS IS THE ONE PLACE THE REDUNDANCY ARGUMENT DOES NOT APPLY
---------------------------------------------------------------
The inversion chain is a deterministic ODE: x(t) = Phi_t(x_0). If the flow is
approximately invertible, ANY single state determines the entire path, so every feature
derived from cached states is mutually redundant — which is exactly what the ordering null
and the token-geometry null both look like. Reparameterizing block-28 hidden states cannot
escape that.

The velocity v(x, t) is different in kind. It is the MODEL'S OUTPUT at that point — where
the learned prior says the sample should go — not a description of where the sample is.
It is a function of the model's weights as well as the state, so it is not recoverable
from the cached state alone. For an out-of-distribution input (remote sensing is heavily
OOD for FLUX) the velocity is where the model's disagreement with the data shows up.

THE LOAD-BEARING COMPARISON
---------------------------
Not "is velocity above chance" — it trivially will be, since velocity is a function of an
image that is itself classifiable. The question is whether velocity adds anything the
states do not already have:

    states_plus_vel  vs  states        <-- non-redundancy. This is the test.
    vel_*            vs  chance        <-- sanity: does velocity carry signal at all.

A positive delta on the first is the only result that supports "the model's judgment
carries information the trajectory's states do not".

Probe/fold/seed conventions and the in-fold PCA parity scheme are inherited from
experiments/token_geometry_probe.py.

FINDINGS, updated (RESEARCH_NOTES 3 + 2026-08-19 audit S2). The null is real AND
instrument-validated: a synthetic block at the same D is detected down to +0.008 (an order
of magnitude below the observed -0.005), and a block WEAKER than velocity is still caught.
Scope caveat: vel_pooled is R^2=0.9988 with the pooled VAE latent -- the claim is about the
SPATIALLY POOLED velocity; token-resolution velocity was only ever tested at two extremes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

# EFFECTIVE_D_NOTE: the reported D column is the REQUESTED cap; cv_fold_accs applies
# min(d_cap, n_features, n_train-1), so at n=500 any cap above 399 is silently 399.
# Read the CSV's D as an upper bound, not the components actually used.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_geometry_probe import (  # noqa: E402
    C_GRID,
    DISCARD_CHANNELS,
    SEEDS,
    apply_ditf_pooled,
    boot_ci,
    cv_fold_accs,
)

CANDIDATES = ["states_plus_vel_pooled", "states_plus_vel_tokens"]
REFERENCE = "states"


def load_one(pattern: str, what: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"FATAL: no {what} cache matched {pattern}")
    if len(paths) > 1:
        raise SystemExit(f"FATAL: {len(paths)} {what} caches matched {pattern}: {paths}")
    return np.load(paths[0], allow_pickle=False), paths[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="models/paired_500_resisc45_vel")
    p.add_argument("--dim", type=int, default=400, help="in-fold PCA cap (default %(default)s)")
    p.add_argument("--out-csv", default="results/velocity_probe.csv")
    p.add_argument(
        "--compare-pooled-to",
        default="models/paired_500_resisc45/*/multistep_train_feats_inversion_g1.0_n50.npz",
        help="pooled cache from the pre-velocity run; verifies want_velocity is inert wrt features",
    )
    args = p.parse_args()

    vd, vpath = load_one(os.path.join(args.cache_dir, "*", "multistep_train_vels_*.npz"), "velocity")
    pd_, ppath = load_one(os.path.join(args.cache_dir, "*", "multistep_train_feats_*.npz"), "pooled")
    vels, labels = vd["vels"], vd["labels"]  # N, K, T, d
    pool = pd_["feats"]  # N, K, C
    if not np.array_equal(labels, pd_["labels"]):
        raise SystemExit("FATAL: velocity and pooled caches disagree on labels — not paired")
    if not np.array_equal(vd["subset_indices"], pd_["subset_indices"]):
        raise SystemExit("FATAL: velocity and pooled caches cover different images")

    n_cls = len(np.unique(labels))
    print(f"vels {vels.shape}  pooled {pool.shape}  classes={n_cls}  chance={1.0 / n_cls:.4f}")
    print(f"  vels:   {vpath}\n  pooled: {ppath}")

    # --- Is want_velocity inert with respect to the features? The terminal step swaps an
    # early-exiting forward_feat for a full forward_velocity_feat, documented to yield the
    # same block-k features. If that is true the pooled caches from the two runs must match;
    # if not, every velocity-vs-states comparison below is confounded by a feature change.
    ref = sorted(glob.glob(args.compare_pooled_to))
    if ref:
        old = np.load(ref[0], allow_pickle=False)
        if np.array_equal(old["subset_indices"], pd_["subset_indices"]):
            same = np.array_equal(old["feats"], pool)
            if same:
                print(
                    "\nfeature-identity check: pooled caches are BIT-IDENTICAL to the "
                    "pre-velocity run — want_velocity is inert wrt features. OK"
                )
            else:
                diff = np.abs(old["feats"].astype(np.float64) - pool.astype(np.float64))
                rel = diff.sum() / (np.abs(old["feats"]).sum() + 1e-12)
                print(
                    f"\nfeature-identity check: NOT bit-identical. max abs {diff.max():.3e}, "
                    f"rel-L1 {rel:.3e} — inspect before trusting velocity results"
                )
        else:
            # Silence here would be indistinguishable from a pass to anyone skimming the log,
            # and RESEARCH_NOTES cites this very line as evidence the check passed.
            raise SystemExit(
                "FATAL: feature-identity reference covers a DIFFERENT image subset "
                f"({len(old['subset_indices'])} vs {len(pd_['subset_indices'])} indices, "
                "or different values) — the check cannot run and must not be reported as passed."
            )
    else:
        print(f"\nfeature-identity check: skipped (no cache matched {args.compare_pooled_to})")

    n, k, t_tok, d = vels.shape
    vel_pooled = vels.mean(axis=2).reshape(n, -1)  # N, K*d   — spatially pooled
    vel_tokens = vels.reshape(n, -1)  # N, K*T*d — full spatial detail

    rows: list[dict] = []
    for norm in ("raw", "normalized"):
        p_use = apply_ditf_pooled(pool, pd_["mods"], DISCARD_CHANNELS) if norm == "normalized" else pool
        states = p_use.reshape(n, -1)
        feats = {
            "states": states,
            "vel_pooled": vel_pooled,
            "vel_tokens": vel_tokens,
            "states_plus_vel_pooled": np.concatenate([states, vel_pooled], axis=1),
            "states_plus_vel_tokens": np.concatenate([states, vel_tokens], axis=1),
        }
        raw_dims = {kk: v.shape[1] for kk, v in feats.items()}
        print(f"\n[{norm}] raw dims {raw_dims}")

        best_c, best_m = C_GRID[0], -1.0
        for c in C_GRID:
            m = float(
                np.mean([np.mean(cv_fold_accs(feats[kk], labels, c, SEEDS[0], args.dim)) for kk in feats])
            )
            print(f"  C={c:<7} mean-across-sets acc {m:.4f}", flush=True)
            if m > best_m:
                best_c, best_m = c, m
        print(f"  frozen C={best_c}")

        for fs in feats:
            for seed in SEEDS:
                for fold, acc in enumerate(cv_fold_accs(feats[fs], labels, best_c, seed, args.dim)):
                    rows.append(
                        {
                            "feature_set": fs,
                            "norm": norm,
                            "seed": seed,
                            "fold": fold,
                            "acc": round(acc, 6),
                            "D": min(args.dim, raw_dims[fs]),
                            "D_cap": args.dim,
                            "raw_D": raw_dims[fs],
                            "C": best_c,
                            "t_set": "|".join(str(x) for x in vd["timesteps"]),
                            "subset_seed": int(vd["subset_seed"]),
                            "n_classes": n_cls,
                            "chance": round(1.0 / n_cls, 6),
                        }
                    )
            a = [r["acc"] for r in rows if r["feature_set"] == fs and r["norm"] == norm]
            print(f"  {fs:<24} acc {np.mean(a):.4f} +/- {np.std(a):.4f} (raw D {raw_dims[fs]})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")

    # --- the load-bearing test: does velocity ADD anything to the states?
    print("\n" + "=" * 78)
    print("NON-REDUNDANCY: (states + velocity) - states, paired per (seed, fold)")
    print("=" * 78)
    verdicts = []
    for norm in ("raw", "normalized"):
        print(f"\n[{norm}]")
        key = lambda r: (r["seed"], r["fold"])  # noqa: E731
        base = {key(r): r["acc"] for r in rows if r["feature_set"] == REFERENCE and r["norm"] == norm}
        for cand in CANDIDATES:
            cm = {key(r): r["acc"] for r in rows if r["feature_set"] == cand and r["norm"] == norm}
            shared = sorted(set(base) & set(cm))
            diffs = [cm[kk] - base[kk] for kk in shared]
            lo, hi = boot_ci(diffs)
            adds = lo > 0
            verdicts.append(adds)
            print(
                f"  {cand:<24} mean {np.mean(diffs):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                f"n={len(diffs)}  {'ADDS SIGNAL' if adds else 'adds nothing'}"
            )
    print("\n" + "-" * 78)
    print(
        f"VERDICT: velocity {'IS' if any(verdicts) else 'is NOT'} non-redundant with the states "
        f"({sum(verdicts)}/{len(verdicts)} candidate-x-norm comparisons cleared)"
    )
    print("-" * 78)


if __name__ == "__main__":
    main()
