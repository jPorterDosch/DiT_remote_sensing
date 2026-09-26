"""Full-split frozen linear probe: fit on the ENTIRE official train split, score once on
the ENTIRE official test split. The headline "frozen FLUX + linear probe" number and the
measured check of the learning-curve extrapolation (RESEARCH_NOTES 6r: EuroSAT predicted
0.9765 at n=21,600, asymptote 0.9832 — that prediction is anchored at t=180).

PROTOCOL. Matches learning_curve_extrapolation.py exactly so the number is comparable to
the extrapolation it validates: StandardScaler fit on train -> PCA-512 (randomized,
random_state=0, fit on train) -> LogisticRegression(C=0.1, max_iter=2000). C and the PCA
width are the standing a-priori values (CLAUDE.md rule 1: nothing here is selected on the
test split — every t in the cache is reported, no max is taken, C is pinned). A no-PCA
variant (scaler -> LR on raw 3072) is reported alongside; PCA at n=21,600 is a holdover
from the small-n protocol, not a necessity, and the delta between the two is itself
informative.

RULE 8: LogisticRegression convergence is counted per arm (ConvergenceWarning) and
printed; a non-converged arm is flagged, not silently reported.
RULE 12: per-image correctness vectors for every (t, variant) go to results/.

Inputs are the two caches written by experiments/extract_full_eurosat.sh:
  train: models/full_eurosat_oneshot_ens8/<run>/multistep_train_feats_oneshot_g1.0.npz
  test:  models/full_eurosat_oneshot_ens8/<run>+test/multistep_test_feats_oneshot_g1.0.npz
Cache identity is validated (ensemble size, timesteps, guidance, block) before any fit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

C = 0.1  # standing a-priori probe C, pinned across the battery
# The instrument, pinned absolutely (rule 9/11): both caches must be THIS configuration.
EXPECT_META = {
    "ensemble_size": 8,
    "guidance_scale": 1.0,
    "k": 28,
    "weights": "flux-dev",
    "extraction_mode": "ONESHOT",
    "dataset": "eurosat",
}
PCA_DIM = 512
MAX_ITER = 2000


def load_cache(pattern: str, expect_split: str, expect_n: int):
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 cache for {pattern}, found {hits}")
    d = np.load(hits[0])
    meta_path = hits[0].replace(".npz", "_meta.json")
    meta = json.load(open(meta_path))
    if meta.get("split", "train") != expect_split:
        raise SystemExit(f"{hits[0]}: meta split={meta.get('split')!r}, expected {expect_split!r}")
    # Absolute pins, not just cross-arm equality: a consistently-wrong PAIR of caches
    # (driver edit, wrong dataset) must not pass silently.
    if d["feats"].shape[0] != expect_n:
        raise SystemExit(f"{hits[0]}: N={d['feats'].shape[0]}, expected the full split ({expect_n})")
    if not np.array_equal(d["subset_indices"], np.arange(expect_n)):
        raise SystemExit(
            f"{hits[0]}: subset_indices is not arange({expect_n}) — this is a SUBSET cache, not the full split"
        )
    for key, want in EXPECT_META.items():
        if meta.get(key) != want:
            raise SystemExit(f"{hits[0]}: meta {key}={meta.get(key)!r}, expected {want!r}")
    return d["feats"], d["labels"], d["timesteps"], meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-glob", default="models/full_eurosat_oneshot_ens8/*+42/multistep_train_feats_oneshot_g1.0.npz"
    )
    p.add_argument(
        "--test-glob",
        default="models/full_eurosat_oneshot_ens8/*+43+test/multistep_test_feats_oneshot_g1.0.npz",
    )
    p.add_argument("--expect-train-n", type=int, default=21600)
    p.add_argument("--expect-test-n", type=int, default=5400)
    p.add_argument("--out", default="results/full_split_probe_eurosat.npz")
    args = p.parse_args()

    Xtr, ytr, t_tr, m_tr = load_cache(args.train_glob, "train", args.expect_train_n)
    Xte, yte, t_te, m_te = load_cache(args.test_glob, "test", args.expect_test_n)

    # Cache-identity validation (rule 11): the two caches must be the same instrument.
    if not np.array_equal(t_tr, t_te):
        raise SystemExit(f"timestep mismatch: train {t_tr} vs test {t_te}")
    for key in (
        "ensemble_size",
        "guidance_scale",
        "k",
        "weights",
        "dataset",
        "img_size",
        "extraction_mode",
        "fixed_cond_t",
        "degrade_to",
        "pooling",
    ):
        if m_tr.get(key) != m_te.get(key):
            raise SystemExit(f"meta mismatch on {key}: {m_tr.get(key)} vs {m_te.get(key)}")
    # eps realizations must be INDEPENDENT across splits, not equal: with a shared
    # eps_seed and unshuffled loaders, test image #k reuses train image #k's exact eps
    # draws (position-paired; 600 of those pairs are same-class), a structural
    # optimistic coupling. The protocol must match across arms; the noise must not.
    if m_tr.get("eps_seed") == m_te.get("eps_seed"):
        raise SystemExit(
            f"train and test caches share eps_seed={m_tr.get('eps_seed')}: position-paired "
            "noise reuse couples the splits. Re-extract the test split with a different "
            "--eps-seed (and --seed)."
        )
    num_classes = int(max(ytr.max(), yte.max())) + 1
    print(
        f"train {Xtr.shape} test {Xte.shape} t={list(t_tr)} ens={m_tr['ensemble_size']} classes={num_classes}"
    )

    results: dict = {"labels_test": yte, "timesteps": t_tr}
    summary = []
    for ti, t_nom in enumerate(t_tr):
        A_raw = Xtr[:, ti, :].astype(np.float64)
        B_raw = Xte[:, ti, :].astype(np.float64)
        sc = StandardScaler().fit(A_raw)
        A, B = sc.transform(A_raw), sc.transform(B_raw)
        for variant in ("pca512", "raw3072"):
            if variant == "pca512":
                pca = PCA(
                    n_components=min(PCA_DIM, A.shape[1], A.shape[0] - 1),
                    svd_solver="randomized",
                    random_state=0,
                ).fit(A)
                Af, Bf = pca.transform(A), pca.transform(B)
            else:
                Af, Bf = A, B
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Af, ytr)
                n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wlist)
            pred = clf.predict(Bf)
            correct = (pred == yte).astype(np.int8)
            acc = float(correct.mean())
            f1 = float(f1_score(yte, pred, average="macro", labels=np.arange(num_classes)))
            results[f"correct_t{int(t_nom)}_{variant}"] = correct
            summary.append((int(t_nom), variant, acc, f1, n_warn))
            conv = "CONVERGED" if n_warn == 0 else f"NOT CONVERGED ({n_warn} warnings)"
            print(f"t={int(t_nom):>3} {variant:>8}: top1={acc:.4f} macro_f1={f1:.4f}  [{conv}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        **results,
        summary=np.array([f"{t},{v},{a:.6f},{f:.6f},{w}" for t, v, a, f, w in summary]),
        C=np.array(C),
        max_iter=np.array(MAX_ITER),
        protocol=np.array(
            "scaler(train)->PCA512(train)->LR C=0.1 | scaler->LR raw; full train fit, full test eval"
        ),
    )
    print(f"per-image vectors cached to {args.out}")


if __name__ == "__main__":
    main()
