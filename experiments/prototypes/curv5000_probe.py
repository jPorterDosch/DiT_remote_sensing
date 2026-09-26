"""The n=5000 curvature battery -- the frozen-model baseline for pre-registered
prediction 3, on the dataset where it matters (RESISC45: powered instrument, hard classes).

Everything paired by construction: states, pred, pred_mid come from the SAME chain pass.
C=0.1 (n=500 showed C-stability of all deltas). Image-level bootstrap.

Blocks: the carrier decomposition (norms / pattern / full) plus the validity controls
(iid noise, image-shuffled curvature, synthetic detection anchor, and a width-matched noise
block for the 6-dim norms arm).

HOW THE PCA IS FITTED, and why it was changed (2026-09-10, RESEARCH_NOTES 6o-A)
------------------------------------------------------------------------------
v1 standardized [base | block] TOGETHER and then fitted PCA-512 on the concatenation. That
made the two arms "base squeezed into 512" vs "base AND block squeezed into the same 512":
the base's 3328 columns hold only ~112 directions with eigenvalue > 1, so PCA-512 on the
base alone also kept ~400 LOW-variance base directions -- and appending 1536 unit-variance
block columns EVICTED that tail. Measured at n=1500: of the 512-direction budget, the base
retained 90 directions under an iid-noise block and 155 under curvature, down from 512.
Since PCA is unsupervised, low variance does NOT mean low class information, so every v1
delta was (information the block adds) - (information the base lost making room). Proof the
eviction term dominated: the synthetic anchor -- literally the one-hot LABELS plus noise --
read NEGATIVE (-0.015 on RESISC45). An instrument that scores a block containing the answer
below zero cannot be read as an information measure.

v2 (default) fits the PCA on the BASE ONLY, then appends the standardized block raw. The
base representation is then IDENTICAL in every arm, so a delta measures addition. Cost: arms
differ in width (512 vs 512+block), which is the section-2 dimensionality confound -- so the
matched-width controls stop being sanity checks and become the CALIBRATION. Pass
--legacy-pca to reproduce v1 exactly.

WHAT THE CONTROLS ARE FOR, corrected after the first v2 run (2026-09-10)
-----------------------------------------------------------------------
v2's first gate demanded that the iid-noise block read ~0. That is the WRONG criterion and it
failed for a reason that is not a bug: with the base PCA protected, appending 1536 junk
columns still moves the arm from 512 to 2048 features at fixed C, and 1536 noise dimensions
at n_train=4000 with K=45 cost real generalization (measured -0.0395 on RESISC45). The
width-matched 6-dim noise control proves it is width and not breakage: -0.0001, dead null.

So a block's delta against the BASE conflates its information with the cost of its own width.
The reported statistic is therefore the block against its WIDTH-MATCHED NULL, which is a
paired per-image quantity -- (plus_block - base) - (plus_null - base) collapses to
plus_block - plus_null -- so it gets a proper image-level bootstrap CI rather than arithmetic
on two independent intervals. For the curvature blocks the primary null is ROW-SHUFFLED
curvature (same width AND same marginal distribution, only the image correspondence
destroyed); iid noise is reported as a secondary null.

GATE (per the section-8 rule that controls are measured, not assumed):
  (a) probe sanity -- the 6-dim noise block must straddle 0. If it does not, the probe itself
      is broken and nothing else matters.
  (b) detection power -- the synthetic anchor must beat ITS OWN width-matched null
      significantly. This is what v1 failed (anchor -0.015, i.e. a block containing the
      literal one-hot labels scored below baseline).
The width penalty itself is reported as a measured calibration, never as a pass/fail.

Per-image correctness vectors for every arm are cached to results/ so the statistics can be
recomputed without re-running the probes (this table has now needed recomputing three times).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
C = 0.1
PCA_DIM = 512
MAX_ITER = 2000
CACHES = {
    "resisc45": "models/solver_curvature_n5000/resisc45_solvercurv_n50.npz",
    "eurosat": "models/solver_curvature_n5000/eurosat_solvercurv_n50.npz",
}


def pool_tokens(x, grid):
    n, k, t, d = x.shape
    hw = int(round(t**0.5))
    sp = torch.from_numpy(x).reshape(n * k, hw, hw, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(sp, (grid, grid)).reshape(n, k * d * grid * grid).numpy()


def _fold(base, blk, y, tr, va, legacy):
    """One (train, val) fit. legacy=True reproduces v1 (PCA after concatenation)."""
    if legacy:
        x = base if blk is None else np.hstack([base, blk])
        sc = StandardScaler().fit(x[tr])
        a, b = sc.transform(x[tr]), sc.transform(x[va])
        k = min(PCA_DIM, a.shape[1], len(tr) - 1)
        if k < a.shape[1]:
            p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
            a, b = p.transform(a), p.transform(b)
    else:
        # v2: the base's representation is fitted WITHOUT the block, so it is identical in
        # every arm and cannot be evicted by the block's columns.
        sc = StandardScaler().fit(base[tr])
        a, b = sc.transform(base[tr]), sc.transform(base[va])
        # Rank guard: n_components must not exceed n_train-1 or the feature count. Without
        # it a small --max-n self-check dies inside joblib, and at any n where PCA_DIM
        # approaches n_train the component count silently differs between folds.
        k = min(PCA_DIM, a.shape[1], len(tr) - 1)
        if k < a.shape[1]:
            p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
            a, b = p.transform(a), p.transform(b)
        if blk is not None:
            scb = StandardScaler().fit(blk[tr])
            a = np.hstack([a, scb.transform(blk[tr])])
            b = np.hstack([b, scb.transform(blk[va])])
    m = LogisticRegression(C=C, max_iter=MAX_ITER).fit(a, y[tr])
    # A fit that hit the cap is effectively MORE regularized than one that converged, so a
    # non-convergence rate that differs by arm is a silent operating-point difference. v2's
    # arms differ in width (512 vs 512+D), so this is tracked per arm and printed; the
    # width-matched-null statistic is the one that controls for it, since block and null
    # share a width.
    hit_cap = bool(np.any(np.atleast_1d(m.n_iter_) >= MAX_ITER))
    return va, m.predict(b), hit_cap


def correct(base, blk, y, legacy, n_jobs=7):
    """-> (per-image correctness averaged over SEEDS repeats of 5-fold CV,
    fraction of folds whose lbfgs hit the iteration cap)."""
    out = np.zeros(len(y))
    caps = []
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(base, y))
        for va, pred, hit_cap in Parallel(n_jobs=n_jobs)(
            delayed(_fold)(base, blk, y, tr, va, legacy) for tr, va in jobs
        ):
            out[va] += (pred == y[va]).astype(float)
            caps.append(hit_cap)
    return out / len(SEEDS), float(np.mean(caps))


def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def pca_budget(base, blk):
    """Diagnostic: how much of the PCA-512 budget the base keeps once blk is appended.
    This is the displacement that v2 removes; printed so it is visible, not argued."""
    if blk is None:
        return None
    x = np.hstack([base, blk])
    z = StandardScaler().fit_transform(x)
    comp = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0).fit(z).components_
    return float((comp[:, : base.shape[1]] ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-pca", action="store_true", help="reproduce v1 (PCA after concat)")
    ap.add_argument("--max-n", type=int, default=None, help="subsample for a fast self-check")
    ap.add_argument("--datasets", nargs="+", default=list(CACHES))
    ap.add_argument("--n-jobs", type=int, default=7)
    ap.add_argument("--show-budget", action="store_true", help="print the PCA displacement diagnostic")
    args = ap.parse_args()
    legacy = args.legacy_pca
    print(
        f"HARNESS: {'v1 LEGACY (PCA after concat)' if legacy else 'v2 (PCA on base only)'}  C={C}  PCA={PCA_DIM}"
    )

    for ds in args.datasets:
        d = np.load(CACHES[ds])
        y = d["labels"]
        sl = slice(0, args.max_n) if args.max_n else slice(None)
        y = y[sl]
        n = len(y)
        rng = np.random.default_rng(0)
        states = pool_tokens(d["states"][sl], 2)
        pred = pool_tokens(d["pred"][sl], 2)
        raw = (d["pred_mid"] - d["pred"])[sl]
        curv = pool_tokens(raw, 2)
        norms = np.linalg.norm(raw.reshape(n, raw.shape[1], -1), axis=2)
        pattern = pool_tokens(raw / (norms[:, :, None, None] + 1e-8), 2)
        D = curv.shape[1]
        K = len(np.unique(y))
        onehot = np.eye(K)[y]

        base_x = np.hstack([states, pred])
        base, cap_base = correct(base_x, None, y, legacy, args.n_jobs)
        st, cap_st = correct(states, None, y, legacy, args.n_jobs)
        print(
            f"\n=== {ds}  n={n}  K={K}  chance={1 / K:.4f}  base(states+pred) {base.mean():.4f} ===",
            flush=True,
        )
        # VELOCITY measured v2-style (2026-09-11 review, finding 9): base(states+pred) vs
        # states alone compares two DIFFERENTLY-fitted PCAs -- the v1 eviction mechanism.
        # Instead: states is the base, pred is a block, judged against its width-matched
        # shuffled null, exactly like every other block.
        st_pred, _ = correct(states, pred, y, legacy, args.n_jobs)
        st_pshuf, _ = correct(states, pred[rng.permutation(n)], y, legacy, args.n_jobs)
        dd = st_pred - st_pshuf
        lo, hi = ci(dd)
        print(
            f"  states alone {st.mean():.4f};  velocity vs SHUFFLED-pred null "
            f"d {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  (v2-style; the old states-vs-"
            f"states+pred delta was eviction-confounded)",
            flush=True,
        )

        blocks = [
            ("curv full", curv, "CTRL curv SHUFFLED rows"),
            # pattern's null is SHUFFLED PATTERN (2026-09-11 review, finding 10): shuffled
            # RAW curv matches width but not the per-column distribution shape of the
            # L2-normalized block, and the width penalty at fixed C depends on shape too.
            ("curv pattern (L2-normed)", pattern, "CTRL pattern SHUFFLED rows"),
            ("curv norms only (6d)", norms, "CTRL noise iid (6d)"),
            # --- nulls / controls. 3rd field = the width-matched null for THIS arm ---
            ("CTRL noise iid (matched D)", rng.standard_normal((n, D)), None),
            ("CTRL noise iid (6d)", rng.standard_normal((n, norms.shape[1])), None),
            ("CTRL curv SHUFFLED rows", curv[rng.permutation(n)], None),
            ("CTRL pattern SHUFFLED rows", pattern[rng.permutation(n)], None),
            (
                "CTRL synthetic anchor s=4",
                np.hstack([onehot] * (D // K + 1))[:, :D] + 4 * rng.standard_normal((n, D)),
                "CTRL noise iid (matched D)",
            ),
        ]

        # --- run every arm, keeping the PER-IMAGE correctness vectors ---
        vec = {
            "base": base,
            "states": st,
            "states+pred (v2)": st_pred,
            "states+SHUFFLED pred": st_pshuf,
        }
        width = {}
        caps = {"base": cap_base, "states": cap_st}
        for name, blk, _null in blocks:
            vec[name], caps[name] = correct(base_x, blk, y, legacy, args.n_jobs)
            width[name] = blk.shape[1]
            dd = vec[name] - base
            lo, hi = ci(dd)
            budget = f"  [base keeps {pca_budget(base_x, blk):.0f}/{PCA_DIM}]" if args.show_budget else ""
            print(
                f"  +{name:<32} {vec[name].mean():.4f}  vs base {dd.mean():+.4f} "
                f"[{lo:+.4f},{hi:+.4f}]  maxiter={caps[name]:.0%}{budget}",
                flush=True,
            )

        # --- the REPORTED statistic: block vs its width-matched null, paired per image ---
        print("\n  REPORTED STATISTIC -- block vs WIDTH-MATCHED null (paired, image-level CI):", flush=True)
        corrected = {}
        for name, blk, null in blocks:
            if null is None:
                continue
            dd = vec[name] - vec[null]  # (plus_blk - base) - (plus_null - base)
            lo, hi = ci(dd)
            corrected[name] = (dd.mean(), lo, hi)
            sig = "SIGNIF" if lo > 0 or hi < 0 else "ns"
            print(
                f"    {name:<32} d {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {sig}"
                f"   (null: {null}, D={width[name]}, maxiter {caps[name]:.0%} vs {caps[null]:.0%})",
                flush=True,
            )

        # --- width-penalty calibration, measured not assumed ---
        print("\n  WIDTH PENALTY (cost of adding junk columns at fixed C, measured):", flush=True)
        for nm in ("CTRL noise iid (matched D)", "CTRL noise iid (6d)", "CTRL curv SHUFFLED rows"):
            dd = vec[nm] - base
            lo, hi = ci(dd)
            print(f"    {nm:<32} D={width[nm]:<5} {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

        # ---------------------------- GATE ----------------------------
        d6 = vec["CTRL noise iid (6d)"] - base
        lo6, hi6 = ci(d6)
        sane = lo6 <= 0 <= hi6
        anch = corrected["CTRL synthetic anchor s=4"]
        powered = anch[1] > 0
        print(f"\n  GATE for {ds}:", flush=True)
        print(
            f"    (a) probe sanity: 6d noise straddles 0      {d6.mean():+.4f} "
            f"[{lo6:+.4f},{hi6:+.4f}]   {'PASS' if sane else 'FAIL'}",
            flush=True,
        )
        print(
            f"    (b) detection power: anchor beats its null  {anch[0]:+.4f} "
            f"[{anch[1]:+.4f},{anch[2]:+.4f}]   {'PASS' if powered else 'FAIL'}",
            flush=True,
        )
        if sane and powered:
            print("    => GATE PASS: the REPORTED STATISTIC rows above are readable.", flush=True)
        else:
            print(f"    => GATE FAIL on {ds}: do NOT quote any cell above.", flush=True)

        # --- cache per-image vectors so this never needs re-running to re-analyse ---
        os.makedirs("results", exist_ok=True)
        tag = "legacy" if legacy else "v2"
        out = f"results/curv5000_{tag}_{ds}{'' if args.max_n is None else f'_n{n}'}.npz"
        np.savez(
            out,
            labels=y,
            arms=np.array(list(vec), dtype=object),
            **{f"correct__{k}": v for k, v in vec.items()},
            **{f"width__{k}": np.array(v) for k, v in width.items()},
            **{f"maxiter__{k}": np.array(v) for k, v in caps.items()},
            harness=np.array(tag),
            C=np.array(C),
            pca_dim=np.array(PCA_DIM),
            seeds=np.array(SEEDS),
        )
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
