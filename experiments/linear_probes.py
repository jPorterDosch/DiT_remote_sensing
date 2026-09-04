"""Tiers 1-2 (+ delta supplement) — linear probes on the paired multi-timestep caches.

Login-node-safe: numpy + sklearn only, no torch, no GPU, no FLUX. Consumes the paired
500-image caches (one per arm: inversion g1.0 / oneshot g1.0 / oneshot g3.5) written by
task='extract', and reports linear-probe accuracy three ways:

  Tier 1  per-timestep     — one logistic-regression probe per (arm, timestep). The
                             accuracy-vs-t curve: WHERE along the path each construction
                             is separable. Also prints the strongest single t for the
                             inversion arm, which sets traj_readout.py --best-t.
  Tier 2  concatenated     — one probe per arm on the flattened (N, K*C) trajectory.
                             Does the SET of states help jointly. A single L2 penalty is
                             swept ONCE (max cross-arm mean CV accuracy) and applied
                             identically to every arm, so the comparison measures features
                             not per-arm tuning; a robustness table across the penalty grid
                             is printed alongside.
  Delta   consecutive Δ    — one probe per arm on concatenated increments
                             x(t_{i+1}) - x(t_i), (N, (K-1)*C). The parameter-free
                             ordering-adjacent row (cf. B2). Uses the frozen Tier-2 penalty.

Evaluation is 5-fold stratified CV on the 500-image cache (same protocol as the B2
harness), so all tiers are directly comparable. Features are per-channel standardized
inside each CV fold (train-fold statistics only, via a Pipeline — no leakage).

Usage:
    python experiments/linear_probes.py                       # globs models/paired_500/*/...
    python experiments/linear_probes.py <cacheA.npz> <cacheB.npz> ...
    python experiments/linear_probes.py --out-csv results/linear_probes.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_GLOB = "models/paired_500/*/multistep_train_feats_*.npz"
N_FOLDS = 5
SEED = 0
# L2 penalty grid for Tier 2 / delta. C is inverse strength: smaller = stronger
# regularization. With ~500 images and up to 21504 features the fit is deeply
# underdetermined, so the ranking should be read together with the robustness table.
C_GRID = [0.001, 0.01, 0.1, 1.0]
MAX_ITER = 5000


def make_probe(c: float):
    """Standardize (train-fold stats only) -> multinomial L2 logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=MAX_ITER),
    )


def cv_acc(x: np.ndarray, y: np.ndarray, c: float, seed: int) -> tuple[float, float]:
    """Mean, std of stratified 5-fold CV accuracy for an L2 probe at penalty c."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_score(make_probe(c), x, y, cv=skf, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def tag_of(c: dict) -> str:
    mode = str(c["extraction_mode"]) if "extraction_mode" in c else "oneshot?(legacy)"
    g = str(c["guidance_scale"]) if "guidance_scale" in c else "3.5?(unrecorded)"
    extra = f" n={c['num_inversion_steps']}" if "num_inversion_steps" in c else ""
    return f"{mode.lower()}_g{g}{extra}"


def load_caches(paths: list[str]) -> list[dict]:
    caches = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        c = {k: d[k] for k in d.files}
        c["__path__"] = p
        c["__tag__"] = tag_of(c)
        caches.append(c)
    # Sort for stable arm order in the tables (inversion first, then oneshot by guidance).
    return sorted(caches, key=lambda c: c["__tag__"])


def assert_paired(caches: list[dict]) -> None:
    """Fail loudly if the caches are not the same images / timesteps — the whole
    comparison is per-image-paired, so a mismatch invalidates every table below."""
    ref = caches[0]
    for c in caches[1:]:
        for key in ("labels", "timesteps", "subset_indices"):
            if key in ref and key in c and not np.array_equal(ref[key], c[key]):
                raise SystemExit(
                    f"FATAL: '{key}' differs between {ref['__tag__']} and {c['__tag__']} — "
                    "caches are not paired; run experiments/verify_paired_caches.py first."
                )


def is_inversion(c: dict) -> bool:
    return "extraction_mode" in c and str(c["extraction_mode"]).lower() == "inversion"


# =======================================================================================
# Tier 1 — per-timestep
# =======================================================================================
def tier1(caches: list[dict], rows: list[dict]) -> None:
    ts = caches[0]["timesteps"].tolist()
    print("\n" + "=" * 72)
    print("TIER 1 — per-timestep linear probe (5-fold CV accuracy, mean±std)")
    print("=" * 72)
    header = f"{'arm':<26}" + "".join(f"{f't={t}':>12}" for t in ts)
    print(header)
    print("-" * len(header))

    best_t_inversion: tuple[float, int] | None = None
    for c in caches:
        y = c["labels"]
        feats = c["feats"]  # (N, K, C)
        line = f"{c['__tag__']:<26}"
        per_t = []
        for ti, t in enumerate(ts):
            m, s = cv_acc(feats[:, ti, :], y, c=1.0, seed=SEED)
            per_t.append(m)
            line += f"{f'{m:.3f}':>12}"
            rows.append(
                {"tier": "1_per_timestep", "arm": c["__tag__"], "key": f"t={t}",
                 "acc_mean": round(m, 4), "acc_std": round(s, 4)}
            )
        print(line)
        if is_inversion(c):
            bi = int(np.argmax(per_t))
            best_t_inversion = (per_t[bi], ts[bi])

    if best_t_inversion is not None:
        acc, t = best_t_inversion
        print("-" * len(header))
        print(f">>> best single t (inversion): t={t}  (CV acc {acc:.3f})")
        print(f">>> feed this to the B2 sweep:  BEST_T={t} sbatch experiments/ordering_traj_readout_inversion.sh")
    else:
        print("-" * len(header))
        print(">>> no inversion cache among inputs — cannot recommend --best-t")


# =======================================================================================
# Tier 2 — concatenated trajectory, single frozen penalty across arms
# =======================================================================================
def tier2(caches: list[dict], rows: list[dict]) -> float:
    print("\n" + "=" * 72)
    print("TIER 2 — concatenated-trajectory linear probe (single penalty across arms)")
    print("=" * 72)

    def flat(c: dict) -> np.ndarray:
        return c["feats"].reshape(c["feats"].shape[0], -1)  # (N, K*C)

    dim = flat(caches[0]).shape[1]
    print(f"feature dim per arm: {dim} (K*C); {caches[0]['feats'].shape[0]} images — heavily "
          "underdetermined, read with the robustness table.\n")

    # Robustness grid + penalty selection: pick C* maximizing the cross-arm MEAN CV acc.
    grid = {c["__tag__"]: {} for c in caches}
    mean_by_c = {}
    for cval in C_GRID:
        accs = []
        for c in caches:
            m, s = cv_acc(flat(c), c["labels"], c=cval, seed=SEED)
            grid[c["__tag__"]][cval] = (m, s)
            accs.append(m)
        mean_by_c[cval] = float(np.mean(accs))
    c_star = max(mean_by_c, key=lambda cv: mean_by_c[cv])

    header = f"{'arm':<26}" + "".join(f"{f'C={cv}':>14}" for cv in C_GRID)
    print(header)
    print("-" * len(header))
    for c in caches:
        line = f"{c['__tag__']:<26}"
        for cval in C_GRID:
            m, s = grid[c["__tag__"]][cval]
            mark = "*" if cval == c_star else " "
            line += f"{f'{m:.3f}{mark}':>14}"
            rows.append(
                {"tier": "2_concat", "arm": c["__tag__"], "key": f"C={cval}{'(frozen)' if cval == c_star else ''}",
                 "acc_mean": round(m, 4), "acc_std": round(s, 4)}
            )
        print(line)
    print("-" * len(header))
    print(f">>> frozen penalty C*={c_star} (max cross-arm mean CV acc = {mean_by_c[c_star]:.3f}); "
          "starred column is the headline Tier-2 comparison.")
    return c_star


# =======================================================================================
# Delta supplement — consecutive increments, frozen penalty
# =======================================================================================
def tier_delta(caches: list[dict], c_star: float, rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"DELTA SUPPLEMENT — consecutive increments x(t+1)-x(t), frozen penalty C*={c_star}")
    print("=" * 72)
    header = f"{'arm':<26}{'acc (mean±std)':>20}"
    print(header)
    print("-" * len(header))
    for c in caches:
        feats = c["feats"]  # (N, K, C)
        deltas = np.diff(feats, axis=1)  # (N, K-1, C)
        x = deltas.reshape(deltas.shape[0], -1)  # (N, (K-1)*C)
        m, s = cv_acc(x, c["labels"], c=c_star, seed=SEED)
        print(f"{c['__tag__']:<26}{f'{m:.3f}±{s:.3f}':>20}")
        rows.append(
            {"tier": "delta", "arm": c["__tag__"], "key": f"C={c_star}",
             "acc_mean": round(m, 4), "acc_std": round(s, 4)}
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("caches", nargs="*", help=f"cache npz paths (default: glob {DEFAULT_GLOB})")
    ap.add_argument("--out-csv", default=None, help="also write all rows to this CSV")
    args = ap.parse_args()

    paths = args.caches or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        raise SystemExit(f"ERROR: no caches found (glob: {DEFAULT_GLOB}) — nothing extracted yet?")

    caches = load_caches(paths)
    assert_paired(caches)
    print(f"loaded {len(caches)} arm(s): " + ", ".join(c["__tag__"] for c in caches))
    print(f"images: {caches[0]['feats'].shape[0]}  timesteps: {caches[0]['timesteps'].tolist()}  "
          f"feat_dim: {caches[0]['feats'].shape[-1]}")

    rows: list[dict] = []
    tier1(caches, rows)
    c_star = tier2(caches, rows)
    tier_delta(caches, c_star, rows)

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tier", "arm", "key", "acc_mean", "acc_std"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
