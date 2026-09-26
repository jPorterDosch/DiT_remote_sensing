"""GEO-Bench m-eurosat probe: the SatDiFuser-comparable number, on their exact partitions.

PRE-REGISTERED PROTOCOL (write nothing test-derived back into selection):
  For EACH arm (oneshot_ens8, inversion) independently:
    candidates = 7 single-t features  +  concat over all 7 t (per-t standardized)
    C grid     = {0.01, 0.1, 1.0}
    scaler fit on train only; LogisticRegression(max_iter=3000)
    SELECT (candidate, C) by VALIDATION accuracy (996 imgs — the official val split;
    rule 1 satisfied: selection data is disjoint from the reported test split)
    EVALUATE the argmax ONCE on the official test split (996 imgs).
  Exactly two test evaluations happen, one per arm, both reported. No cross-arm max is
  quoted as "the" result — the arms are different instruments, reported side by side.

Cache identity (rule 11): absolute pins per arm; the train arm of oneshot must carry
eps_seed distinct from val/test (RESEARCH_NOTES 6t audit F1); inversion shards are
merged only after verifying disjoint, complete coverage of arange(N) via each shard's
subset_indices. Per-image correctness vectors go to results/ (rule 12); convergence is
counted per fitted cell (rule 8) and any NOT-CONVERGED selected cell is flagged.

Comparator: SatDiFuser global-fusion 97.7 top-1 — same partitions (their loader uses the
geobench package default = 16,200/996/996), 996-image test, binomial 95% CI ~±0.9 pt.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0)
MAX_ITER = 3000
EXPECT_N = {"train": 16200, "val": 996, "test": 996}
SMOKE = False  # set by --smoke-expect; relaxes the arange-coverage shard check only
PINS_COMMON = {"guidance_scale": 1.0, "k": 28, "weights": "flux-dev", "dataset": "m_eurosat"}
ARMS = {
    "oneshot_ens8": {
        "dir": "models/m_eurosat_oneshot_ens8",
        "fname": "multistep_{split}_feats_oneshot_g1.0.npz",
        "pins": {**PINS_COMMON, "extraction_mode": "ONESHOT", "ensemble_size": 8},
    },
    "inversion": {
        "dir": "models/m_eurosat_inversion",
        "fname": "multistep_{split}_feats_inversion_g1.0_n50.npz",
        "pins": {**PINS_COMMON, "extraction_mode": "INVERSION", "num_inversion_steps": 50},
    },
}


def _load_one(path: str, pins: dict, split: str):
    d = np.load(path)
    meta = json.load(open(path.replace(".npz", "_meta.json")))
    if meta.get("split", "train") != split:
        raise SystemExit(f"{path}: meta split={meta.get('split')!r} != {split!r}")
    for k, want in pins.items():
        if meta.get(k) != want:
            raise SystemExit(f"{path}: meta {k}={meta.get(k)!r}, expected {want!r}")
    return d, meta


def load_split(arm: dict, split: str):
    """Load one split of one arm, merging shard caches when present."""
    pattern = os.path.join(arm["dir"], "*", arm["fname"].format(split=split))
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no caches for {pattern}")
    parts = [_load_one(h, arm["pins"], split) for h in hits]
    n_expect = EXPECT_N[split]
    if len(parts) == 1 and parts[0][1].get("num_shards", 1) in (1, None):
        d, meta = parts[0]
        feats, labels, idx = d["feats"], d["labels"], d["subset_indices"]
    else:
        # shard merge: order by shard_index, verify disjoint + complete coverage
        parts.sort(key=lambda p: p[1]["shard_index"])
        n_shards = parts[0][1]["num_shards"]
        got = [p[1]["shard_index"] for p in parts]
        if got != list(range(n_shards)):
            raise SystemExit(f"{split}: have shards {got}, expected 0..{n_shards - 1}")
        idx = np.concatenate([p[0]["subset_indices"] for p in parts])
        if len(np.unique(idx)) != len(idx) or len(idx) != n_expect:
            raise SystemExit(f"{split}: merged shard indices are not disjoint / complete (n={len(idx)})")
        if not SMOKE and not np.array_equal(np.sort(idx), np.arange(n_expect)):
            # real (full-split) shards must cover the whole split exactly; only smoke
            # caches (stratified subsets) are exempt
            raise SystemExit(f"{split}: merged shard indices are not a cover of arange({n_expect})")
        feats = np.concatenate([p[0]["feats"] for p in parts])
        labels = np.concatenate([p[0]["labels"] for p in parts])
        meta = parts[0][1]
    if feats.shape[0] != n_expect:
        raise SystemExit(f"{split}: N={feats.shape[0]}, expected {n_expect}")
    order = np.argsort(idx)  # dataset order, so val/test vectors align across arms
    ts = [int(t) for t in parts[0][0]["timesteps"]]
    return feats[order].astype(np.float64), labels[order], ts, meta


def fit_eval(Xtr, ytr, Xev, C):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Xtr, ytr)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    pred = clf.predict(Xev)
    return clf, pred, n_warn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--out", default="results/m_eurosat_probe.npz")
    p.add_argument(
        "--smoke-expect",
        default=None,
        help="SMOKE ONLY: 'train,val,test' counts to expect instead of the official "
        "16200,996,996 (e.g. '20,20,20'). The official numbers are the default and the "
        "only quotable configuration.",
    )
    args = p.parse_args()
    if args.smoke_expect:
        global SMOKE
        SMOKE = True
        tr, va, te = (int(x) for x in args.smoke_expect.split(","))
        EXPECT_N.update({"train": tr, "val": va, "test": te})
        print(f"SMOKE MODE: expected sizes overridden to {EXPECT_N} — NOT quotable")

    results: dict = {}
    for arm_name in args.arms:
        arm = ARMS[arm_name]
        Xtr, ytr, ts, m_tr = load_split(arm, "train")
        Xva, yva, ts_v, m_va = load_split(arm, "val")
        Xte, yte, ts_t, m_te = load_split(arm, "test")
        if not (ts == ts_v == ts_t):
            raise SystemExit(f"{arm_name}: timestep mismatch {ts} {ts_v} {ts_t}")
        if arm_name == "oneshot_ens8":
            seeds = {m_tr.get("eps_seed"), m_va.get("eps_seed"), m_te.get("eps_seed")}
            if len(seeds) != 3:
                raise SystemExit(
                    f"{arm_name}: eps seeds not pairwise distinct across splits: {seeds} (6t audit F1)"
                )
        num_classes = int(ytr.max()) + 1
        print(f"\n== {arm_name}: train {Xtr.shape} val {Xva.shape} test {Xte.shape} t={ts}")

        def make(X, ti):  # single-t slice or concat
            return X.reshape(X.shape[0], -1) if ti is None else X[:, ti, :]

        candidates = [(f"t{t}", i) for i, t in enumerate(ts)] + [(f"concat{len(ts)}t", None)]
        best = None  # (val_acc, name, C, scaler(s), clf, n_warn)
        val_table = {}
        for name, ti in candidates:
            Atr, Ava = make(Xtr, ti), make(Xva, ti)
            sc = StandardScaler().fit(Atr)
            Atr_s, Ava_s = sc.transform(Atr), sc.transform(Ava)
            for C in C_GRID:
                _, pred, n_warn = fit_eval(Atr_s, ytr, Ava_s, C)
                acc = float((pred == yva).mean())
                val_table[f"{name},C={C}"] = (acc, n_warn)
                print(
                    f"  val {name:>9} C={C:<5} acc={acc:.4f}"
                    + ("" if n_warn == 0 else f"  [NOT CONVERGED x{n_warn}]")
                )
                if best is None or acc > best[0]:
                    best = (acc, name, ti, C, n_warn)

        assert best is not None
        val_acc, name, ti, C, sel_warn = best
        print(
            f"  SELECTED on val: {name}, C={C} (val acc {val_acc:.4f})"
            + ("" if sel_warn == 0 else "  [selected cell NOT CONVERGED — diagnose before quoting]")
        )
        # single test evaluation for this arm
        Atr, Ate = make(Xtr, ti), make(Xte, ti)
        sc = StandardScaler().fit(Atr)
        _, pred, n_warn = fit_eval(sc.transform(Atr), ytr, sc.transform(Ate), C)
        correct = (pred == yte).astype(np.int8)
        top1 = float(correct.mean())
        f1 = float(f1_score(yte, pred, average="macro", labels=np.arange(num_classes)))
        se = (top1 * (1 - top1) / len(yte)) ** 0.5
        conv = "CONVERGED" if n_warn == 0 else f"NOT CONVERGED x{n_warn}"
        print(f"  TEST ({name}, C={C}): top1={top1:.4f} ±{1.96 * se:.4f}  macro_f1={f1:.4f}  [{conv}]")
        results[f"{arm_name}_correct_test"] = correct
        results[f"{arm_name}_selected"] = np.array(
            f"{name},C={C},val={val_acc:.6f},test={top1:.6f},f1={f1:.6f},warn={n_warn}"
        )
        results[f"{arm_name}_val_table"] = np.array([f"{k}:{v[0]:.6f},{v[1]}" for k, v in val_table.items()])
        results[f"{arm_name}_labels_test"] = yte

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        **results,
        protocol=np.array(
            "per-arm: select {single-t x7, concat7t} x C{0.01,0.1,1} on official val; one test eval per arm; "
            "scaler train-fit; LR max_iter=3000; GEO-Bench default partition 16200/996/996"
        ),
    )
    print(f"\nper-image vectors cached to {args.out}")
    print("Comparator: SatDiFuser global-fusion 97.7 top-1, same partitions, 996-img test (95% CI ~±0.9 pt).")


if __name__ == "__main__":
    main()
