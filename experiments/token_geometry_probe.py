"""Job 3 — does joint space x time structure carry class signal that pooling discards?

Hypothesis: how the trajectory evolves DIFFERENTLY across image regions ("which parts
resolve when") is class-relevant, and every prior aggregation destroys it by pooling
tokens spatially before the time axis is introduced.

THE DESIGN CONSTRAINT THAT SHAPES EVERYTHING BELOW
--------------------------------------------------
Spatial mean commutes with any LINEAR per-token operation:

    mean_i ( x[b,i] - x[a,i] )  ==  pooled[b] - pooled[a]

So a feature built from pooled per-token DISPLACEMENT VECTORS is not a new feature at
all — it is exactly what the time-no-space baseline already sees, and the alignment null
would reproduce it bit-for-bit, making the comparison vacuous. Every statistic used here
is therefore NONLINEAR in the token features before pooling (cosines, magnitudes,
elementwise |delta|). Those do not commute with the spatial mean, which is precisely why
they can express something pooling throws away — and why shuffling token correspondence
across t (baseline 3) can destroy them while leaving per-timestep marginals intact.

FEATURE SETS (all probed identically, all projected to a common D)
-----------------------------------------------------------------
  token_geom      per-token temporal statistics, pooled AFTER computation:
                    per t-pair: cos(x[a,i], x[b,i]) and ||x[b,i]-x[a,i]||, each summarized
                                over tokens by mean/std/9 deciles (distributional, not
                                just central — "which parts resolve when" is a spread)
                    per t-pair: elementwise mean_i |x[b,i]-x[a,i]|  (C dims, nonlinear)
                    per t:      ||x[t,i]|| summarized over tokens
  time_no_space   BASELINE 1: pooled (N, K, C) flattened. Existing Tier-2 protocol.
  space_no_time   BASELINE 2: tokens at the single best t, each token projected to a few
                              dims by a shared seeded matrix then CONCATENATED over
                              tokens — spatial structure retained, no time.
  align_null      BASELINE 3: token_geom recomputed after independently permuting token
                              order at each timestep per image. Preserves every
                              per-timestep marginal; destroys the space x time coupling.

Pre-committed decision rule: token_geom is real only if it beats ALL THREE at matched D,
outside noise, on paired per-fold deltas with bootstrap CIs.

Probe/fold/seed conventions are inherited from experiments/linear_probes.py (StandardScaler
-> multinomial L2 logistic regression, stratified 5-fold). The L2 penalty is swept ONCE on
the mean across feature sets and then frozen, so the comparison measures features, not
per-feature-set tuning.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

# EFFECTIVE_D_NOTE: the reported D column is the REQUESTED cap; cv_fold_accs applies
# min(d_cap, n_features, n_train-1), so at n=500 any cap above 399 is silently 399.
# Read the CSV's D as an upper bound, not the components actually used.
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5
SEEDS = [0, 1, 2]
# Extended past 1.0: the first run froze C at the top of the old grid in BOTH norms, so
# the optimum may have been outside it and every feature set judged under the wrong penalty.
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MAX_ITER = 2000
DISCARD_CHANNELS = [154, 1446]
QUANTILES = np.linspace(0.1, 0.9, 9)

# token_geom_dist is the distributional summaries ALONE (99 dims). The full variant is
# 9315 dims of which 9216 (98.9%) are the elementwise |delta| block, so in the first run
# the 99 dims that actually encode "which parts resolve when" were diluted 93:1 and then
# mixed by the random projection. The dist variant cannot be swamped and needs no
# projection at all. Both are probed; the full variant is kept for comparison.
# Spatial partitions of the token grid. (1,1) = pool all tokens (global summaries);
# (2,2) = per-quadrant. Finer splits (4,4)/(8,8)/full map are deliberately NOT run yet —
# at n=500 the training fold has rank <= 400 so a linear probe cannot exploit more, and
# each decile would be estimated from too few tokens. Revisit when n grows; see the
# ablation note in RESEARCH_NOTES.md.
PARTITIONS = {
    "global": [(1, 1)],          # 99 dims
    "q2": [(2, 2)],              # 396 dims
    "q2g": [(1, 1), (2, 2)],     # 495 dims
}
CANDIDATES = [f"token_geom_{k}" for k in PARTITIONS]
NULLS = [f"align_null_{k}" for k in PARTITIONS]
BASELINES = ["time_no_space", "space_no_time"] + NULLS
FEATURE_SETS = CANDIDATES + BASELINES


# ---------------------------------------------------------------------------------------
# shared probe conventions (mirrors experiments/linear_probes.py)
# ---------------------------------------------------------------------------------------
def make_probe(c: float, n_components: int | None = None):
    """StandardScaler -> [PCA to a common D] -> multinomial L2 logistic regression.

    PARITY. The first run matched dimensionality with a seeded Gaussian random projection,
    which turned out to be the dominant confound: random projection preserves pairwise
    distances but NOT class-discriminative structure, so features whose signal is
    concentrated in few dimensions were destroyed while diffuse ones were untouched
    (measured: from D=1024 to native, time_no_space gained 5% while token_geom gained 84%).

    PCA fixes it on principle rather than by tuning. A linear probe's weights live in the
    row space of its TRAINING fold, so projecting onto that span costs the trained
    classifier nothing — test predictions depend only on the component of x_test inside it.
    With a 400-sample training fold, PCA to <=400 components is therefore simultaneously
    matched across feature sets AND lossless for each of them. Fitted inside the fold on
    train only, so there is no leakage.
    """
    steps: list = [StandardScaler()]
    if n_components is not None:
        steps.append(PCA(n_components=n_components, svd_solver="randomized", random_state=0))
    steps.append(LogisticRegression(C=c, max_iter=MAX_ITER))
    return make_pipeline(*steps)


def cv_fold_accs(
    x: np.ndarray, y: np.ndarray, c: float, seed: int, d_cap: int | None = None
) -> list[float]:
    """Per-FOLD accuracies (not just the mean) — the deltas are paired per fold."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    out = []
    for tr, va in skf.split(x, y):
        # PCA cannot return more components than min(n_train, n_features); centering costs
        # one more. Below that bound the projection is a no-op and is skipped entirely.
        k = None
        if d_cap is not None:
            k = min(d_cap, x.shape[1], len(tr) - 1)
            if k >= x.shape[1]:
                k = None
        p = make_probe(c, k).fit(x[tr], y[tr])
        out.append(float((p.predict(x[va]) == y[va]).mean()))
    return out


def apply_ditf_tokens(tok: np.ndarray, mods: np.ndarray, discard: list[int]) -> np.ndarray:
    """DiTF normalization applied PER TOKEN. tok (N, S, L, C), mods (S, 3, C).

    Same sequence as traj_readout.apply_ditf_normalization, but LayerNorm is taken over C
    within each token rather than over a spatially-pooled vector — which is the more
    faithful application (the pooled version is the approximation, cf. the caveat in
    plot_multistep_diagnostics).

    NOTE: the final F.normalize makes every token unit-norm, so under --norm normalized the
    norm-profile block of token_geom is constant by construction and carries no signal.
    That is expected, not a bug; the cosine and elementwise-|delta| blocks still vary.
    """
    x = tok.astype(np.float32).copy()
    if discard:
        x[:, :, :, discard] = 0.0
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mu) / np.sqrt(var + 1e-6)
    x = (1.0 + mods[None, :, 1, :][:, :, None, :]) * x + mods[None, :, 0, :][:, :, None, :]
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return x


def apply_ditf_pooled(feats: np.ndarray, mods: np.ndarray, discard: list[int]) -> np.ndarray:
    """DiTF normalization on POOLED features (N, K, C) — mirrors
    traj_readout.apply_ditf_normalization so baseline 1 gets the same treatment its
    existing Tier-2 protocol would give it."""
    x = feats.astype(np.float64).copy()
    if discard:
        x[:, :, discard] = 0.0
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mu) / np.sqrt(var + 1e-6)
    x = (1.0 + mods[None, :, 1, :]) * x + mods[None, :, 0, :]
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return x.astype(np.float32)


def summarize(v: np.ndarray) -> np.ndarray:
    """(N, L) per-token scalars -> (N, 11) distributional summary over tokens.

    Mean alone would discard the spread, and the hypothesis is precisely that regions
    differ from one another — so the deciles are the part that carries it.
    """
    return np.concatenate(
        [v.mean(1, keepdims=True), v.std(1, keepdims=True), np.quantile(v, QUANTILES, axis=1).T],
        axis=1,
    )


def region_ids(h: int, w: int, gh: int, gw: int) -> np.ndarray:
    """Assign each row-major token of an (h, w) grid to one of gh*gw spatial cells."""
    idx = np.arange(h * w)
    r, c = idx // w, idx % w
    return ((r * gh) // h) * gw + ((c * gw) // w)


def per_token_fields(x: np.ndarray) -> np.ndarray:
    """(N, S, L, C) -> (N, F, L): the F per-token scalar fields the hypothesis is about.

    F = S norms + 2 * n_pairs (cosine and displacement magnitude per timestep pair).
    These are the quantities whose SPREAD ACROSS REGIONS is "which parts resolve when".
    Computed once here so any spatial partition can summarize them without recomputing.
    """
    n, s, ell, _ = x.shape
    fields = [np.linalg.norm(x[:, t], axis=-1) for t in range(s)]
    for a in range(s):
        for b in range(a + 1, s):
            xa, xb = x[:, a], x[:, b]
            na = np.maximum(np.linalg.norm(xa, axis=-1), 1e-12)
            nb = np.maximum(np.linalg.norm(xb, axis=-1), 1e-12)
            fields.append(np.einsum("nlc,nlc->nl", xa, xb) / (na * nb))
            fields.append(np.linalg.norm(xb - xa, axis=-1))
    return np.stack(fields, axis=1)  # N, F, L


def build_token_geom_regional(
    tok: np.ndarray,
    grid_hw: tuple[int, int],
    partitions: list[tuple[int, int]],
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Per-token scalar fields summarized within each cell of each spatial partition.

    partitions=[(1,1)]        -> global summaries only              (99 dims)
    partitions=[(2,2)]        -> per-quadrant summaries             (396 dims)
    partitions=[(1,1),(2,2)]  -> both                               (495 dims)

    Quantile summaries are permutation-invariant WITHIN a cell, so they encode "how much
    do regions here differ in timing" without asserting that a given token index means the
    same thing across images — which matters for overhead imagery with no canonical
    orientation. Finer partitions trade that robustness for locality and estimate each
    decile from fewer tokens; see the ablation note in RESEARCH_NOTES.md.
    """
    h, w = grid_hw
    x = tok
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        x = np.empty_like(tok)
        for i in range(tok.shape[0]):
            for t in range(tok.shape[1]):
                x[i, t] = tok[i, t, rng.permutation(tok.shape[2])]

    fields = per_token_fields(x)  # N, F, L
    n, f, _ = fields.shape
    blocks: list[np.ndarray] = []
    for gh, gw in partitions:
        rid = region_ids(h, w, gh, gw)
        for r in range(gh * gw):
            sub = fields[:, :, rid == r]  # N, F, L_r
            for j in range(f):
                blocks.append(summarize(sub[:, j, :]))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_token_geom(
    tok: np.ndarray, shuffle_seed: int | None = None, include_delta_block: bool = True
) -> np.ndarray:
    """tok (N, S, L, C) -> (N, D). shuffle_seed set => alignment null (baseline 3).

    The null permutes token order INDEPENDENTLY per (image, timestep), so every
    per-timestep marginal is untouched and only the cross-timestep correspondence dies.

    include_delta_block=False drops the (S_pairs, C) elementwise mean|delta| block and
    returns only the 99-dim distributional summaries. Two reasons that variant matters:
    it is not diluted 93:1 by the delta block, and the delta block is the part where the
    shuffle does not act as a true null — shuffling turns "how far did token i travel"
    into "how far apart are two random tokens", i.e. within-image spatial dispersion,
    which is itself an informative texture statistic rather than an absence of signal.
    """
    n, s, ell, _ = tok.shape
    x = tok
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        x = np.empty_like(tok)
        for i in range(n):
            for t in range(s):
                x[i, t] = tok[i, t, rng.permutation(ell)]

    blocks: list[np.ndarray] = []
    # per-timestep: token norm profile
    for t in range(s):
        blocks.append(summarize(np.linalg.norm(x[:, t], axis=-1)))
    # per t-pair: cosine, displacement magnitude, elementwise mean |delta|
    for a in range(s):
        for b in range(a + 1, s):
            xa, xb = x[:, a], x[:, b]
            na = np.maximum(np.linalg.norm(xa, axis=-1), 1e-12)
            nb = np.maximum(np.linalg.norm(xb, axis=-1), 1e-12)
            blocks.append(summarize(np.einsum("nlc,nlc->nl", xa, xb) / (na * nb)))
            d = xb - xa
            blocks.append(summarize(np.linalg.norm(d, axis=-1)))
            if include_delta_block:
                blocks.append(np.abs(d).mean(axis=1))  # (N, C) — nonlinear, does NOT collapse
            del d
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_space_no_time(tok: np.ndarray, t_idx: int, per_token_dim: int, seed: int) -> np.ndarray:
    """Tokens at ONE timestep, each projected to per_token_dim then concatenated.

    Keeps spatial structure (token identity survives in the concatenation) while removing
    time entirely. Projecting before concatenating keeps the raw dim tractable; the same
    seeded matrix is shared by all tokens so the projection cannot encode position.
    """
    x = tok[:, t_idx]  # N, L, C
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(x.shape[-1]), size=(x.shape[-1], per_token_dim))
    return (x @ proj).reshape(x.shape[0], -1).astype(np.float32)


def project_to(x: np.ndarray, d: int, seed: int) -> np.ndarray:
    """Seeded Gaussian random projection to a common D.

    Random rather than PCA on purpose: it needs no fitting, so there is no train/test
    leakage to reason about and every feature set is treated identically.
    """
    if x.shape[1] <= d:
        return x
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], d))
    return (x @ p).astype(np.float32)


def boot_ci(diffs: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not diffs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.asarray(diffs, dtype=float)
    m = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="models/paired_500_resisc45")
    p.add_argument("--arm", default="inversion", choices=["inversion", "oneshot"])
    p.add_argument("--dim", type=int, default=1024, help="common D for parity (default %(default)s)")
    p.add_argument("--per-token-dim", type=int, default=8, help="space_no_time per-token proj dim")
    p.add_argument("--best-t", type=int, default=260, help="timestep for space_no_time")
    p.add_argument("--out-csv", default="results/token_geometry.csv")
    p.add_argument("--proj-seed", type=int, default=0)
    args = p.parse_args()

    tok_glob = os.path.join(args.cache_dir, "*", f"multistep_train_tokens_{args.arm}_*.npz")
    pool_glob = os.path.join(args.cache_dir, "*", f"multistep_train_feats_{args.arm}_*.npz")
    tok_paths, pool_paths = sorted(glob.glob(tok_glob)), sorted(glob.glob(pool_glob))
    if not tok_paths or not pool_paths:
        raise SystemExit(f"FATAL: missing caches for arm={args.arm} under {args.cache_dir}")
    td = np.load(tok_paths[0], allow_pickle=False)
    pd_ = np.load(pool_paths[0], allow_pickle=False)

    tok, labels = td["tokens"], td["labels"]
    tok_ts = [int(t) for t in td["timesteps"]]
    pool, pool_ts = pd_["feats"], [int(t) for t in pd_["timesteps"]]
    if not np.array_equal(labels, pd_["labels"]):
        raise SystemExit("FATAL: token and pooled caches disagree on labels — not paired")
    if args.best_t not in tok_ts:
        raise SystemExit(f"FATAL: --best-t {args.best_t} not in token cache timesteps {tok_ts}")
    mods_all = pd_["mods"]
    mods_tok = np.stack([mods_all[pool_ts.index(t)] for t in tok_ts])  # S, 3, C
    grid_hw = (int(td["grid_hw"][0]), int(td["grid_hw"][1]))
    if grid_hw[0] * grid_hw[1] != tok.shape[2]:
        raise SystemExit(f"FATAL: grid {grid_hw} does not match L={tok.shape[2]}")

    n_cls = len(np.unique(labels))
    print(f"arm={args.arm}  tokens {tok.shape}  pooled {pool.shape}  classes={n_cls}")
    print(f"token t={tok_ts}  pooled t={pool_ts}  chance={1.0 / n_cls:.4f}")

    rows: list[dict] = []
    for norm in ("raw", "normalized"):
        if norm == "normalized":
            t_use = apply_ditf_tokens(tok, mods_tok, DISCARD_CHANNELS)
            p_use = apply_ditf_pooled(pool, mods_all, DISCARD_CHANNELS)
        else:
            t_use, p_use = tok, pool

        feats: dict[str, np.ndarray] = {}
        for name, parts in PARTITIONS.items():
            feats[f"token_geom_{name}"] = build_token_geom_regional(t_use, grid_hw, parts)
            feats[f"align_null_{name}"] = build_token_geom_regional(
                t_use, grid_hw, parts, shuffle_seed=1234
            )
        feats["space_no_time"] = build_space_no_time(
            t_use, tok_ts.index(args.best_t), args.per_token_dim, args.proj_seed
        )
        feats["time_no_space"] = p_use.reshape(p_use.shape[0], -1)

        raw_dims = {k: v.shape[1] for k, v in feats.items()}
        print(f"\n[{norm}] raw dims {raw_dims}")
        print(f"[{norm}] parity: in-fold PCA to D<={args.dim} (no-op where raw D <= cap)")

        # freeze one L2 penalty across all feature sets (measure features, not tuning)
        best_c, best_m = C_GRID[0], -1.0
        for c in C_GRID:
            m = float(np.mean([
                np.mean(cv_fold_accs(feats[k], labels, c, SEEDS[0], args.dim)) for k in FEATURE_SETS
            ]))
            print(f"  C={c:<7} mean-across-sets acc {m:.4f}", flush=True)
            if m > best_m:
                best_c, best_m = c, m
        print(f"  frozen C={best_c}")

        for fs in FEATURE_SETS:
            for seed in SEEDS:
                for fold, acc in enumerate(cv_fold_accs(feats[fs], labels, best_c, seed, args.dim)):
                    rows.append({
                        "arm": args.arm, "feature_set": fs, "norm": norm, "seed": seed,
                        "fold": fold, "acc": round(acc, 6),
                        "D": min(args.dim, raw_dims[fs]),
                        "D_cap": args.dim, "raw_D": raw_dims[fs], "C": best_c,
                        "t_set": "|".join(map(str, tok_ts)),
                        "best_t": args.best_t, "subset_seed": int(td["subset_seed"]),
                        "n_classes": n_cls, "chance": round(1.0 / n_cls, 6),
                    })
            a = [r["acc"] for r in rows if r["feature_set"] == fs and r["norm"] == norm]
            print(f"  {fs:<20} acc {np.mean(a):.4f} +/- {np.std(a):.4f}  "
                  f"(raw D {raw_dims[fs]}, used {min(args.dim, raw_dims[fs])})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")
    summarize_deltas(rows)


def summarize_deltas(rows: list[dict]) -> None:
    """Paired candidate - baseline per (norm, seed, fold), with bootstrap CI.

    Each candidate is matched to the appropriate alignment null: the full feature against
    the full null, the dist-only feature against the dist-only null. Pairing the dist
    candidate against the full null would compare across two differences at once.
    """
    def key(r):
        return (r["seed"], r["fold"])

    print("\n" + "=" * 78)
    print("PAIRED DELTAS  (positive => candidate better than baseline)")
    print("=" * 78)
    # Verdicts are PER NORM: the rule is "beats all three baselines", evaluated separately
    # under raw and normalized. A candidate that passes in one norm and not the other is
    # not an ambiguous result to be averaged away — it is evidence that the feature (or a
    # baseline) is normalization-dependent, which is itself worth reporting.
    per_norm: dict[tuple[str, str], list[bool]] = {}
    for cand in CANDIDATES:
        # each candidate is matched to the null built from the SAME spatial partition;
        # pairing across partitions would confound two differences at once.
        null_for = cand.replace("token_geom_", "align_null_")
        bases = ["time_no_space", "space_no_time", null_for]
        print(f"\n### candidate: {cand}   (alignment null: {null_for})")
        for norm in ("raw", "normalized"):
            print(f"  [{norm}]")
            verdicts: list[bool] = []
            cm = {key(r): r["acc"] for r in rows if r["feature_set"] == cand and r["norm"] == norm}
            for base in bases:
                bm = {key(r): r["acc"] for r in rows if r["feature_set"] == base and r["norm"] == norm}
                shared = sorted(set(cm) & set(bm))
                if not shared:
                    continue
                d = [cm[k] - bm[k] for k in shared]
                lo, hi = boot_ci(d)
                beats = lo > 0
                verdicts.append(beats)
                print(f"    vs {base:<17} mean {np.mean(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                      f"n={len(d)}  {'BEATS' if beats else 'does NOT beat'}")
            per_norm[(cand, norm)] = verdicts

    print("\n" + "-" * 78)
    print("DECISION RULE (per norm): real only if it beats ALL THREE baselines, outside noise.")
    for cand in CANDIDATES:
        results = {}
        for norm in ("raw", "normalized"):
            v = per_norm.get((cand, norm), [])
            results[norm] = all(v) and bool(v)
            print(f"  {cand:<18} [{norm:<10}] {'PASSES' if results[norm] else 'FAILS'}  "
                  f"({sum(v)}/{len(v)} baselines cleared)")
        if results["raw"] != results["normalized"]:
            print(f"  >>> {cand}: NORM-DEPENDENT — passes under "
                  f"{'raw' if results['raw'] else 'normalized'} only. Report this rather than "
                  "averaging it away; it localizes the effect to the normalization.")
    print("-" * 78)


if __name__ == "__main__":
    main()
