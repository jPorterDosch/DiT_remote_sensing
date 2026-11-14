"""Label-efficiency matrix: model x dataset x label fraction x seed, from cached features.

    python -m eval.matrix run    --spec eval/matrix_specs/phase1.json [--models ...] [--datasets ...]
    python -m eval.matrix report --spec eval/matrix_specs/phase1.json [--out-dir results/matrix]

One cell = the official protocol of eval.probe (StandardScaler on the probe-training rows,
logistic regression, candidate view x C selected on the FULL official val split, ONE test
evaluation) with the probe trained on a label subset of the official train split:
  * fractions are NESTED per (dataset, seed): 1 % c 5 % c 10 % c 25 % c 100 %;
  * single-label tasks: per-class permutation, the first ceil(f * n_c) of every class (so every
    class is present at every fraction); multi-label (m-bigearthnet): one random permutation;
  * seeds 0, 1, 2 (the 100 % cell is seed-free and computed once);
  * val and test are the full official splits: the "label budget" is the probe-training labels
    only, selection on val uses all val labels, identically for every model -- unless the spec
    sets "selection": "val-budget", which subsamples the val split for the selection with the
    same nested draw at the cell's fraction and seed (selection_stamp; 2026-11-03).
Cells are cached as results/matrix/cells/<model>/<dataset>/f<frac>_s<seed>.npz with the
per-image test vector (rule 12), the selection, and the sha1 of every feature file read; a
cell whose inputs (feature files) or protocol (c_grid, max_iter, resolved model entry) changed
is refused unless --redo. `report` reads the cells, prints every
MISSING cell (features absent, or never run) explicitly, and writes per-dataset tables,
label-efficiency curves, AULC (unweighted mean of the metric over the five label budgets, 100 %
included; AULC<100 % = the same mean over 1/5/10/25 % only; equal weight per
log-spaced budget) and teacher-relative deltas (spec: "teacher": "<model name>").

Spec (JSON): {"fractions": [...], "seeds": [...], "c_grid": [...], "datasets": [...],
  "models": [{"name", "kind": "flux"|"dino"|"satdifuser", "features": pattern with {split} (and {dataset}),
              "view": "cands,comma,separated" | null, "transform": null | "ditf" (flux only; the
              DiTF view of the pooled tokens instead of the raw protocol view, stamped in the cell),
              "expect": {...flux pins... | satdifuser: {"scale": ...}},
              "teacher": name | null, "per_dataset": {dataset: {overrides}}}]}
Metrics: top-1 (single-label), micro-F1 (multi-label); never averaged across datasets.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval import features as F  # noqa: E402
from eval import probe as PR  # noqa: E402
from eval import protocols as P  # noqa: E402
from eval import wb  # noqa: E402

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_C_GRID = P.C_GRID_WIDE  # the wide official grid; phase1.json extends it to 10 and 100 (2026-10-13b)
OUT_DIR = "results/matrix"


# ----------------------------------------------------------------------------- subsets
def label_subsets(y: np.ndarray, fractions, seed: int) -> dict[float, np.ndarray]:
    """Nested label subsets of the train split, {fraction: sorted row indices}. Single-label:
    per-class permutation, first ceil(f * n_c) rows of each class. Multi-hot: one permutation,
    first ceil(f * n). Nested by construction (prefixes of the same permutation)."""
    fractions = sorted(set(float(f) for f in fractions))
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError(f"fractions must be in (0, 1]: {fractions}")
    rng = np.random.default_rng(seed)
    if y.ndim == 1:
        perms = {c: rng.permutation(np.flatnonzero(y == c)) for c in np.unique(y)}
        out = {}
        for f in fractions:
            out[f] = np.sort(np.concatenate([p[: math.ceil(f * len(p))] for p in perms.values()]))
    else:
        perm = rng.permutation(len(y))
        out = {f: np.sort(perm[: math.ceil(f * len(y))]) for f in fractions}
    for a, b in zip(fractions, fractions[1:]):
        if not set(out[a].tolist()) <= set(out[b].tolist()):
            raise RuntimeError("subsets are not nested")
    return out


# ----------------------------------------------------------------------------- one cell
BLAS_THREADS_CAP = 8  # part of the protocol: see blas_threads
# Bumped whenever a code change alters what a cell computes without touching the spec (subset
# draw, fit, scaler, selection, metric, thread policy); stored in every cell's config, so old
# cells are refused by `run` / `report` until recomputed (--redo).
PROTOCOL_VERSION = "2026-10-15"  # one-vs-rest fits at a FIXED OVR_FIT_THREADS (was cpus // workers)
# 2026-10-13b: spec C grid to 10 / 100, BLAS cap fixed at 8 (was the allocation). Cells written
# under it are reused where the 2026-10-15 change alters nothing (config_current).


SELECTIONS = ("full-val", "val-budget")
FULL_VAL_STAMP = "full official val"


def selection_stamp(selection: str, frac: float) -> str:
    """The `selection` entry of a cell's config. "full-val" (the default and every cell before
    2026-11-03): candidate view and C are selected on the FULL official val split at every
    fraction. "val-budget": the val split used for that selection is subsampled with the SAME
    nested per-class draw as the probe-training labels (label_subsets at the cell's fraction and
    seed), so the whole pipeline, probe training AND model selection, sees labels at the budget
    (reviewer objection 2026-11-03: full-val selection at a 1 % training budget is not a uniformly
    label-constrained pipeline). At 100 % the two are the same computation and carry the same
    stamp, so the 100 % cells are shared between the two protocols."""
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    if selection == "full-val" or float(frac) == 1.0:
        return FULL_VAL_STAMP
    return "val-budget: official val subsampled by label_subsets at the cell's fraction and seed"


def val_rows(selection: str, yva: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Val rows a cell selects on: all of them, or the budget draw (see selection_stamp)."""
    if selection_stamp(selection, frac) == FULL_VAL_STAMP:
        return np.arange(len(yva))
    return label_subsets(yva, [frac], seed)[float(frac)]


def blas_threads(n_rows: int) -> int:
    """BLAS threads for an lbfgs fit on n_rows. Tiny designs are overhead-bound (forestnet 1 %,
    69 x 4096: 793 s with 48 unbounded threads on a login node, 12 s with 8, 8 s with 1), so the
    count scales with the rows. It is capped at a CONSTANT, not at the job's CPU allocation:
    lbfgs on the wide FLUX designs is only reproducible for a fixed thread count (OpenBLAS sums
    in a thread-dependent order; forestnet 1 % x k=56 moved 10/70 val entries by up to 0.003 and
    1/993 test predictions between 1 and 4 threads, review 2026-10-13), so a cell's numbers must
    not depend on --cpus-per-task. The fit oversubscribes when the allocation is smaller than
    the cap (slower, same numbers). Multi-hot labels: every one-vs-rest label fit runs at the
    constant protocols.OVR_FIT_THREADS (= 1) BLAS threads, serial or in the OVR_N_JOBS worker
    pool (protocols._CountingLR); the pool size (matrix sbatch: cpus, lowered by
    budget_ovr_jobs when memory is short) only changes wall time. Until 2026-10-15 the share
    was cpus // workers, i.e. it followed the allocation (stamped as `ovr_threads`)."""
    return int(min(BLAS_THREADS_CAP, max(1, n_rows // 256)))


def fit_cell(cands: dict, tr: np.ndarray, ytr, yva, yte, c_grid):
    """eval.protocols.run_official on the probe-training rows `tr` (scaler fit on them), the
    same selection rule (first max over candidate order then C order) and a C grid argument.
    Multi-hot labels: the one-vs-rest fits spread over OVR_N_JOBS workers sized to the job's
    memory first (protocols.budget_ovr_jobs, like run_official; the matrix sbatch exports
    OVR_N_JOBS=cpus) and each candidate's design is shared with them as ONE memmap
    (protocols._shared_for_workers). Predictions do not depend on the WORKER count
    (tests/test_matrix.py) but do depend on the BLAS thread count for wide designs, hence the
    fixed policies in blas_threads and protocols.OVR_FIT_THREADS; the thread and worker counts
    used are returned and stamped in the cell. Wall time 2026-10-13: the m-bigearthnet 25 % cells (43 labels x 4 views x 7 C,
    serial) took 1.5-3 h each before the worker pool."""
    from threadpoolctl import threadpool_limits

    n_jobs = P.budget_ovr_jobs(cands, ytr)
    threads = blas_threads(len(tr))
    with threadpool_limits(limits=threads, user_api="blas"):
        out = _fit_cell(cands, tr, ytr, yva, yte, c_grid)
    cpus = len(os.sched_getaffinity(0))
    # ovr_threads = the BLAS threads of each one-vs-rest label fit (protocols.OVR_FIT_THREADS, a
    # constant since 2026-10-15; before: cpus // workers). Wide designs move at the 1e-3 level
    # with the thread count, so it is stamped per cell and tables_markdown reports a dataset
    # whose cells mix shares (config_current should keep that from happening).
    multilabel = np.asarray(ytr).ndim == 2
    return (*out, {"blas_threads": threads, "ovr_n_jobs": int(n_jobs), "cpus": cpus,
                   "ovr_threads": P.OVR_FIT_THREADS if multilabel else threads})  # fmt: skip


def _fit_cell(cands, tr, ytr, yva, yte, c_grid):
    from sklearn.preprocessing import StandardScaler

    y = ytr[tr]
    best, table = None, []
    for name, (Atr, Ava, _) in cands.items():
        sc = StandardScaler().fit(Atr[tr])
        V = sc.transform(Ava)
        with P._shared_for_workers(lambda: sc.transform(Atr[tr]), y) as A:
            for Cv in c_grid:
                pred, n_warn = P.official_fit_eval(A, y, V, Cv)
                acc = P.metric(P.per_image(yva, pred))
                table.append({"candidate": name, "C": Cv, "val_acc": acc, "conv_warnings": n_warn})
                if best is None or acc > best[0]:
                    best = (acc, name, Cv, n_warn)
            del A  # release the (deleted) memmap before the next candidate writes its own
    val_acc, name, Cv, sel_warn = best
    Atr, _, Ate = cands[name]
    sc = StandardScaler().fit(Atr[tr])
    with P._shared_for_workers(lambda: sc.transform(Atr[tr]), y) as A:
        pred, n_warn = P.official_fit_eval(A, y, sc.transform(Ate), Cv)
        del A
    correct = P.per_image(yte, pred)
    sel = {"candidate": name, "C": Cv, "val_acc": val_acc, "sel_warn": sel_warn, "test_warn": n_warn}
    return correct, pred, sel, table


# ----------------------------------------------------------------------------- models
def _resolve(model: dict, dataset: str) -> dict:
    m = {k: v for k, v in model.items() if k != "per_dataset"}
    m.update((model.get("per_dataset") or {}).get(dataset, {}))
    if "features" not in m:
        raise SystemExit(f"model {model['name']}: no features pattern for {dataset}")
    m["features"] = m["features"].replace("{dataset}", dataset)
    return m


def load_cands(model: dict, dataset: str):
    """(cands, ytr, yva, yte, test_paths, input_files) via eval.probe's official loaders."""
    m = _resolve(model, dataset)
    kind = m["kind"]
    args = SimpleNamespace(dataset=dataset, features=m["features"], view=m.get("view"), sizes=None)
    if m.get("transform") and kind != "flux":
        raise SystemExit(f"model {m['name']}: transform is only defined for kind 'flux'")
    if kind == "flux":
        args.pins = F.flux_pins(dataset, dict(m.get("expect") or {}))
        args.transform = m.get("transform")  # null (raw pooled tokens, the protocol) | "ditf"
        cands, ytr, yva, yte, paths, inputs, _ = PR.official_flux(args)
    elif kind == "dino":
        cands, ytr, yva, yte, paths, inputs, _ = PR.official_dino(args)
    elif kind == "satdifuser":
        # SatDiFuser's DiffusionSat features (eval.satdifuser): expect.scale pins the pyramid scale
        # per dataset, as expect.k pins a FLUX arm's block.
        args.expect = dict(m.get("expect") or {})
        cands, ytr, yva, yte, paths, inputs, _ = PR.official_satdifuser(args)
    else:
        raise SystemExit(f"model {m['name']}: kind {kind!r} not supported (flux, dino, satdifuser)")
    return cands, ytr, yva, yte, paths, inputs


def nonfinite_rows(cands: dict) -> tuple[np.ndarray, list[str]]:
    """Train rows with a non-finite value in ANY candidate view, and the names of candidates with
    non-finite val/test rows (those are never droppable: the reported split must be whole)."""
    bad = set()
    bad_eval = []
    for name, (Atr, Ava, Ate) in cands.items():
        bad.update(np.flatnonzero(~np.isfinite(Atr).all(axis=1)).tolist())
        if not np.isfinite(Ava).all() or not np.isfinite(Ate).all():
            bad_eval.append(name)
    return np.array(sorted(bad), dtype=int), bad_eval


def features_present(model: dict, dataset: str) -> tuple[bool, str]:
    m = _resolve(model, dataset)
    for split in F.official_spec(dataset)["sizes"]:
        pat = m["features"].format(split=split)
        if not glob.glob(pat):
            return False, f"no features for split {split}: {pat}"
    return True, ""


def input_files(model: dict, dataset: str) -> list[str]:
    """Every feature file a cell of (model, dataset) reads: the official loaders' own globs."""
    m = _resolve(model, dataset)
    return sorted(
        h for split in F.official_spec(dataset)["sizes"] for h in glob.glob(m["features"].format(split=split))
    )


# ----------------------------------------------------------------------------- cells on disk
def cell_path(out_dir: str, model: str, dataset: str, frac: float, seed: int) -> str:
    return os.path.join(out_dir, "cells", model, dataset, f"f{frac:g}_s{seed}.npz")


def inputs_sha(files) -> dict:
    return {wb.rel(p): wb.file_identity(p)["sha1"] for p in files}


def cell_config(
    model: dict, dataset: str, frac: float, seed: int, c_grid, selection: str = "full-val"
) -> dict:
    """Everything besides the feature files that determines a cell's numbers; stored in the cell
    and compared on reuse, so a spec edit (C grid, view list, pins) cannot silently keep old cells."""
    # drop_nonfinite_train_rows is an opt-in for cells whose sampled rows hold non-finite features;
    # it leaves every other cell's numbers untouched (those record dropped_train_rows = []), so it
    # is not part of the identity -- setting it on a spec must not invalidate finished cells.
    resolved = {k: v for k, v in _resolve(model, dataset).items() if k != "drop_nonfinite_train_rows"}
    return {
        "model": resolved,
        "dataset": dataset,
        "frac": frac,
        "seed": seed,
        "c_grid": list(c_grid),
        "max_iter": P.OFFICIAL_MAX_ITER,
        "selection": selection_stamp(selection, frac),
        "protocol": PROTOCOL_VERSION,
        "blas_threads_cap": BLAS_THREADS_CAP,
    }


def ovr_share(compute: dict):
    """BLAS threads each one-vs-rest fit of a cell ran with: the `ovr_threads` stamp (cells from
    2026-10-15 on), else the pre-2026-10-15 rule (workers: cpus // ovr_n_jobs; serial:
    blas_threads). None if the stamp is missing altogether."""
    if "ovr_threads" in compute:
        return compute["ovr_threads"]
    n, cpus = int(compute.get("ovr_n_jobs") or 0), int(compute.get("cpus") or 0)
    if n > 1:
        return max(1, cpus // n)
    return compute.get("blas_threads")


def config_current(
    old_cfg: dict, cfg: dict, old_compute: dict, multilabel: bool, accept_thread_share: bool = False
) -> bool:
    """True if a stored cell (its config + compute stamp) is what the CURRENT protocol computes.
    Exact config match, or the one accepted cross-version case: 2026-10-13b -> 2026-10-15 only
    fixed the BLAS threads of the one-vs-rest fits at protocols.OVR_FIT_THREADS, so a
    single-label cell, or a multi-label cell whose fits already ran at that share (ovr_share),
    is bit-identical under both. Anything else is stale: refused by `run` / `collect`,
    recomputed with --redo. accept_thread_share: ALSO accept a 2026-10-13b multi-label cell at
    another share (--accept-thread-share; user decision 2026-10-15: the 13 flux-pretrained x
    m-bigearthnet phase-1 cells, fit at 2 threads, are kept, not recomputed; the effect is ~1e-3
    and the report prints the mixed-share NOTE for that dataset)."""
    if old_cfg == cfg:
        return True
    if {k for k in set(old_cfg) | set(cfg) if old_cfg.get(k) != cfg.get(k)} != {"protocol"}:
        return False
    if old_cfg.get("protocol") != "2026-10-13b" or cfg.get("protocol") != "2026-10-15":
        return False
    if not multilabel:
        return True
    return accept_thread_share or ovr_share(old_compute) == P.OVR_FIT_THREADS


def run(
    spec: dict, models, datasets, out_dir: str, redo: bool, accept_thread_share: bool = False
) -> list[dict]:
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    seeds = [int(s) for s in spec.get("seeds", DEFAULT_SEEDS)]
    c_grid = tuple(float(c) for c in spec.get("c_grid", DEFAULT_C_GRID))
    selection = spec.get("selection", "full-val")
    selection_stamp(selection, 1.0)  # validate
    log = []
    for model in models:
        for dataset in datasets:
            ok, why = features_present(model, dataset)
            if not ok:
                print(f"MISSING {model['name']} x {dataset}: {why}", flush=True)
                log.append({"model": model["name"], "dataset": dataset, "status": "missing", "reason": why})
                continue
            sha = inputs_sha(input_files(model, dataset))  # sidecar sha1s: no feature load for cached cells
            cands = None
            for seed in seeds:
                for frac in fractions:
                    if frac == 1.0 and seed != seeds[0]:
                        continue  # the full split has no draw
                    path = cell_path(out_dir, model["name"], dataset, frac, seed)
                    cfg = cell_config(model, dataset, frac, seed, c_grid, selection)
                    if os.path.exists(path):
                        old = np.load(path, allow_pickle=True)
                        old_sha, old_cfg = (json.loads(str(old[k])) for k in ("inputs", "config"))
                        old_compute = json.loads(str(old["compute"])) if "compute" in old.files else {}
                        if old_sha == sha and config_current(
                            old_cfg, json.loads(json.dumps(cfg)), old_compute, old["labels"].ndim == 2,
                            accept_thread_share,
                        ):  # fmt: skip
                            log.append(
                                {
                                    "model": model["name"],
                                    "dataset": dataset,
                                    "frac": frac,
                                    "seed": seed,
                                    "status": "cached",
                                }
                            )
                            continue
                        if not redo:
                            what = "inputs" if old_sha != sha else "protocol/config"
                            raise SystemExit(
                                f"{path}: {what} changed since it was written (pass --redo to recompute)"
                            )
                    if cands is None:
                        cands, ytr, yva, yte, paths, inputs = load_cands(model, dataset)
                        bad_train, bad_eval = nonfinite_rows(cands)
                        if bad_eval:
                            raise SystemExit(
                                f"{model['name']} x {dataset}: non-finite val/test features in {bad_eval}; re-extract"
                            )
                        got = inputs_sha(inputs)
                        if got != sha:
                            raise SystemExit(
                                f"{model['name']} x {dataset}: loader read {sorted(got)} but the glob lists {sorted(sha)}"
                            )
                    if os.path.exists(path):
                        # --redo: keep the superseded cell (per-image vectors) under stale/.
                        keep = os.path.join(
                            out_dir, "stale", model["name"], dataset,
                            f"{os.path.basename(path)[:-4]}.{old_cfg.get('protocol', 'noproto')}.{int(time.time())}.npz",
                        )  # fmt: skip
                        os.makedirs(os.path.dirname(keep), exist_ok=True)
                        os.replace(path, keep)
                        print(f"stale cell kept at {keep}", flush=True)
                    subs = label_subsets(ytr, fractions, seed)
                    tr = subs[frac]
                    # Non-finite cached train rows (a bf16 overflow in one forward pass) are refused
                    # unless the spec's model entry opts in with "drop_nonfinite_train_rows": true;
                    # the dropped rows are then recorded in the cell (dropped_train_rows) and the
                    # fit uses n_train minus them. Never silent (2026-11-02: semt-lo m_bigearthnet
                    # train image 19538 at t=180, blocks >= 8).
                    dropped = np.intersect1d(tr, bad_train)
                    if dropped.size:
                        # Read through _resolve so a per_dataset override counts (phase7 semt-lo x
                        # m_bigearthnet sets it for that dataset only; 2026-11-08 cell 6561197 refused it).
                        if not _resolve(model, dataset).get("drop_nonfinite_train_rows"):
                            raise SystemExit(
                                f"{model['name']} x {dataset} f={frac} s={seed}: {dropped.size} sampled train row(s) "
                                f"{dropped.tolist()[:8]} hold non-finite features; re-extract, or set "
                                '"drop_nonfinite_train_rows": true on the spec\'s model entry to fit without them'
                            )
                        tr = np.setdiff1d(tr, dropped)
                        print(
                            f"{model['name']} x {dataset} f={frac} s={seed}: dropping {dropped.size} non-finite train row(s) {dropped.tolist()[:8]}",
                            flush=True,
                        )
                    # Selection rows of the val split: all of them, or the budget draw (val-budget).
                    va = val_rows(selection, yva, frac, seed)
                    if len(va) == len(yva):
                        cands_fit, yva_fit = cands, yva
                    else:
                        cands_fit = {k: (Atr, Ava[va], Ate) for k, (Atr, Ava, Ate) in cands.items()}
                        yva_fit = yva[va]
                    correct, pred, sel, table, compute = fit_cell(cands_fit, tr, ytr, yva_fit, yte, c_grid)
                    score = P.metric(correct)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp = path + ".tmp.npz"
                    np.savez(
                        tmp,
                        correct=correct,
                        pred=pred,
                        labels=yte,
                        paths=np.array(paths),
                        train_rows=tr,
                        dropped_train_rows=dropped,
                        n_train=np.array(len(tr)),
                        val_rows=va,
                        n_val=np.array(len(va)),
                        score=np.array(score),
                        selected=np.array(json.dumps(sel)),
                        val_table=np.array(json.dumps(table)),
                        inputs=np.array(json.dumps(sha, sort_keys=True)),
                        config=np.array(json.dumps(cfg, sort_keys=True)),
                        compute=np.array(json.dumps(compute, sort_keys=True)),
                    )
                    os.replace(tmp, path)
                    print(
                        f"{model['name']:>18} {dataset:<14} f={frac:<5g} s={seed} n_train={len(tr):>6} n_val={len(va):>5} "
                        f"sel={sel['candidate']}/C={sel['C']:g} val={sel['val_acc']:.4f} TEST={score:.4f}"
                        + ("" if sel["test_warn"] == 0 else f" [NOT CONVERGED x{sel['test_warn']}]"),
                        flush=True,
                    )
                    log.append(
                        {
                            "model": model["name"],
                            "dataset": dataset,
                            "frac": frac,
                            "seed": seed,
                            "status": "done",
                            "score": score,
                        }
                    )
    return log


# ----------------------------------------------------------------------------- report
def _load_cell(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "score": float(d["score"]),
        "n_train": int(d["n_train"]),
        "correct": d["correct"],
        "labels": d["labels"],
        "paths": [os.path.basename(p) for p in d["paths"].tolist()],
        "selected": json.loads(str(d["selected"])),
        "inputs": json.loads(str(d["inputs"])),
        "config": json.loads(str(d["config"])),
        "compute": json.loads(str(d["compute"])) if "compute" in d.files else {},
    }


def collect(spec: dict, out_dir: str, accept_thread_share: bool = False) -> tuple[dict, list[dict]]:
    """cells[model][dataset][frac] = list of {seed, score, n_train, correct, ...}; plus missing
    list. A cell is re-validated against the CURRENT feature files (sidecar sha1) and the spec's
    protocol exactly as `run` would: a stale cell is refused, never reported. Feature files that
    no longer exist cannot be checked; the cell keeps the sha it was computed from and the report
    says so ("inputs_absent")."""
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    seeds = [int(s) for s in spec.get("seeds", DEFAULT_SEEDS)]
    c_grid = tuple(float(c) for c in spec.get("c_grid", DEFAULT_C_GRID))
    selection = spec.get("selection", "full-val")
    cells, missing = {}, []
    for model in spec["models"]:
        for dataset in spec["datasets"]:
            present, _ = features_present(model, dataset)
            sha = inputs_sha(input_files(model, dataset)) if present else None
            for frac in fractions:
                for seed in seeds:
                    if frac == 1.0 and seed != seeds[0]:
                        continue
                    p = cell_path(out_dir, model["name"], dataset, frac, seed)
                    if not os.path.exists(p):
                        missing.append(
                            {"model": model["name"], "dataset": dataset, "frac": frac, "seed": seed}
                        )
                        continue
                    c = _load_cell(p)
                    cfg = json.loads(json.dumps(cell_config(model, dataset, frac, seed, c_grid, selection)))
                    multilabel = c["labels"].ndim == 2
                    if multilabel and "ovr_threads" not in c["compute"]:
                        # pre-2026-10-15 stamp: derive the share so the mixed-share NOTE can name it
                        c["compute"] = {**c["compute"], "ovr_threads": ovr_share(c["compute"])}
                    if not config_current(c["config"], cfg, c["compute"], multilabel):
                        if not (
                            accept_thread_share
                            and config_current(c["config"], cfg, c["compute"], multilabel, True)
                        ):
                            raise SystemExit(
                                f"{p}: protocol/config differs from the spec -- rerun with --redo"
                            )
                        c["thread_share_accepted"] = True  # kept on purpose (--accept-thread-share)
                        print(
                            f"ACCEPTED at {c['compute']['ovr_threads']} one-vs-rest threads (not {P.OVR_FIT_THREADS}): {p}"
                        )
                    if sha is not None and c["inputs"] != sha:
                        raise SystemExit(f"{p}: inputs changed since it was written -- rerun with --redo")
                    c["seed"] = seed
                    c["inputs_absent"] = sha is None
                    cells.setdefault(model["name"], {}).setdefault(dataset, {}).setdefault(frac, []).append(c)
    return cells, missing


def summarize(spec: dict, cells: dict) -> dict:
    """Per model x dataset: mean/std over seeds per fraction, AULC, and (if a teacher is named)
    the per-fraction delta to the teacher with an image-level bootstrap CI at 100 %."""
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    teacher_of = {m["name"]: m.get("teacher") for m in spec["models"]}
    out = {}
    for model, per_ds in cells.items():
        for dataset, per_f in per_ds.items():
            row = {
                "fractions": {},
                "aulc": None,
                "cells": [{"compute": c.get("compute", {})} for cs in per_f.values() for c in cs],
            }
            for f in fractions:
                if f in per_f:
                    s = np.array([c["score"] for c in per_f[f]])
                    row["fractions"][f] = {
                        "mean": float(s.mean()),
                        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,  # sample std over seeds
                        "n_seeds": len(s),
                        "n_train": per_f[f][0]["n_train"],
                    }
            if all(f in row["fractions"] for f in fractions):
                row["aulc"] = float(np.mean([row["fractions"][f]["mean"] for f in fractions]))
            low = [f for f in fractions if f < 1.0]  # low-label AULC: same mean, 100 % point left out
            if low and all(f in row["fractions"] for f in low):
                row["aulc_low"] = float(np.mean([row["fractions"][f]["mean"] for f in low]))
            else:
                row["aulc_low"] = None
            row["inputs_absent"] = any(c["inputs_absent"] for cs in per_f.values() for c in cs)
            t = teacher_of.get(model)
            if t and t in cells and dataset in cells[t]:
                row["teacher"] = t
                row["delta_vs_teacher"] = {}
                for f in fractions:
                    if f in per_f and f in cells[t][dataset]:
                        a = np.mean([c["score"] for c in cells[t][dataset][f]])
                        b = np.mean([c["score"] for c in per_f[f]])
                        d = {"delta": float(b - a)}
                        if f == 1.0:
                            ct, cs = cells[t][dataset][f][0], per_f[f][0]
                            if not np.array_equal(ct["labels"], cs["labels"]) or ct["paths"] != cs["paths"]:
                                raise SystemExit(
                                    f"{model} vs teacher {t} on {dataset}: test images differ or are "
                                    "ordered differently -- the paired delta would be meaningless"
                                )
                            d["delta"], d["ci_lo"], d["ci_hi"] = P.paired_delta(ct["correct"], cs["correct"])
                        row["delta_vs_teacher"][f] = d
                ta = summarize_aulc(cells[t][dataset], fractions)
                if row["aulc"] is not None and ta is not None:
                    row["delta_aulc_vs_teacher"] = row["aulc"] - ta
                ta_low = summarize_aulc(cells[t][dataset], [f for f in fractions if f < 1.0])
                if row.get("aulc_low") is not None and ta_low is not None:
                    row["delta_aulc_low_vs_teacher"] = row["aulc_low"] - ta_low
            out.setdefault(model, {})[dataset] = row
    return out


def summarize_aulc(per_f: dict, fractions) -> float | None:
    if not all(f in per_f for f in fractions):
        return None
    return float(np.mean([np.mean([c["score"] for c in per_f[f]]) for f in fractions]))


def tables_markdown(spec: dict, summary: dict, missing: list[dict]) -> str:
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    if spec.get("selection", "full-val") == "val-budget":
        lines = [
            "Protocol (val-budget): label fraction = share of the official train split used to fit the "
            "probe (nested per seed) AND the share of the official val split the candidate view and C "
            "are selected on (same nested per-class draw, same seed), identically for every model; one "
            "evaluation on the full official test split. Probe training and model selection both see "
            "labels at the budget; the 100 % cells are the full-val ones.",
        ]
    else:
        lines = [
            "Protocol: label fraction = share of the official train split used to fit the probe "
            "(nested per seed); candidate view and C are selected on the FULL official val split at "
            "every fraction (identically for every model), one evaluation on the full official test "
            "split. The budget is probe-training labels only; val labels are not counted.",
        ]
    for dataset in spec["datasets"]:
        metric = "micro-F1" if F.official_spec(dataset).get("multilabel") else "top-1"
        lines.append(
            f"\n### {dataset} ({metric}; mean ± sample std (ddof=1) over seeds; 100 % = single fit)\n"
        )
        shares = sorted(
            {
                str(c.get("compute", {}).get("ovr_threads", "unrecorded"))
                for r in summary.values()
                for c in (r.get(dataset) or {}).get("cells", [])
            }
        )
        if F.official_spec(dataset).get("multilabel") and len(shares) > 1:
            lines.append(
                f"NOTE: the one-vs-rest fits of this dataset ran with different BLAS thread counts "
                f"across cells ({', '.join(shares)} threads; `ovr_threads` in each cell's compute record): "
                "wide designs move at the 1e-3 level with it. The count is a protocol constant since "
                "2026-10-15 (protocols.OVR_FIT_THREADS); cells at another share are pre-2026-10-15 ones "
                "kept with --accept-thread-share (recompute with --redo before quoting sub-1e-2 "
                "differences between models).\n"
            )
        lines.append(
            "| model | "
            + " | ".join(f"{f:g}" for f in fractions)
            + " | AULC | Δ AULC vs teacher | AULC<100 % | Δ AULC<100 % |"
        )
        lines.append("|---|" + "---|" * (len(fractions) + 4))
        for m in spec["models"]:
            row = summary.get(m["name"], {}).get(dataset)
            if row is None:
                lines.append(f"| {m['name']} | " + " | ".join("—" for _ in fractions) + " | — | — | — | — |")
                continue
            cells = []
            for f in fractions:
                c = row["fractions"].get(f)
                cells.append(
                    "—"
                    if c is None
                    else (f"{c['mean']:.4f}" if c["n_seeds"] == 1 else f"{c['mean']:.4f} ± {c['std']:.4f}")
                )
            aulc = "—" if row["aulc"] is None else f"{row['aulc']:.4f}"
            dt = row.get("delta_aulc_vs_teacher")
            aulc_low, dt_low = row.get("aulc_low"), row.get("delta_aulc_low_vs_teacher")
            lines.append(
                f"| {m['name']} | "
                + " | ".join(cells)
                + f" | {aulc} | {'—' if dt is None else f'{dt:+.4f}'}"
                + f" | {'—' if aulc_low is None else f'{aulc_low:.4f}'}"
                + f" | {'—' if dt_low is None else f'{dt_low:+.4f}'} |"
            )
    if missing:
        lines.append(f"\n**MISSING cells: {len(missing)}**\n")
        by = {}
        for c in missing:
            by.setdefault((c["model"], c["dataset"]), []).append(f"f{c['frac']:g}/s{c['seed']}")
        for (m, d), v in sorted(by.items()):
            lines.append(f"- {m} × {d}: {len(v)} cells ({', '.join(v[:6])}{', ...' if len(v) > 6 else ''})")
    return "\n".join(lines) + "\n"


def plot_curves(spec: dict, summary: dict, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    outs = []
    for dataset in spec["datasets"]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        drawn = 0
        for m in spec["models"]:
            row = summary.get(m["name"], {}).get(dataset)
            if not row:
                continue
            xs = [f for f in fractions if f in row["fractions"]]
            ys = [row["fractions"][f]["mean"] for f in xs]
            es = [row["fractions"][f]["std"] for f in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=m["name"])
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xscale("log")
        ax.set_xticks(fractions)
        ax.set_xticklabels([f"{f:g}" for f in fractions])
        ax.set_xlabel("fraction of train labels (probe)")
        ax.set_ylabel("micro-F1" if F.official_spec(dataset).get("multilabel") else "top-1")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out_dir, f"curve_{dataset}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        outs.append(p)
    return outs


def report(spec: dict, out_dir: str, accept_thread_share: bool = False) -> str:
    cells, missing = collect(spec, out_dir, accept_thread_share)
    summary = summarize(spec, cells)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "missing": missing, "spec": spec}, f, indent=1, default=str)
    md = tables_markdown(spec, summary, missing)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    plots = plot_curves(spec, summary, out_dir)
    print(md)
    print(f"-> {out_dir}/summary.json, summary.md, {len(plots)} curves; MISSING cells: {len(missing)}")
    return md


# ----------------------------------------------------------------------------- CLI
def load_spec(path: str) -> dict:
    spec = json.load(open(path))
    for k in ("models", "datasets"):
        if not spec.get(k):
            raise SystemExit(f"{path}: spec needs a non-empty {k!r}")
    names = [m["name"] for m in spec["models"]]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate model names")
    for m in spec["models"]:
        if m.get("teacher") and m["teacher"] not in names:
            raise SystemExit(f"{path}: {m['name']} names teacher {m['teacher']!r} which is not a model")
    for d in spec["datasets"]:
        F.official_spec(d)
    if spec.get("selection", "full-val") not in SELECTIONS:
        raise SystemExit(f"{path}: selection must be one of {SELECTIONS}, got {spec.get('selection')!r}")
    return spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--models", nargs="*", default=None, help="subset of the spec's model names")
    r.add_argument("--datasets", nargs="*", default=None)
    r.add_argument("--out-dir", default=OUT_DIR)
    r.add_argument("--redo", action="store_true", help="recompute cells whose inputs changed")
    r.add_argument(
        "--fractions", type=float, nargs="*", default=None, help="SMOKE: subset of the spec's fractions"
    )
    r.add_argument("--seeds", type=int, nargs="*", default=None, help="SMOKE: subset of the spec's seeds")
    p = sub.add_parser("report")
    p.add_argument("--spec", required=True)
    p.add_argument("--out-dir", default=OUT_DIR)
    for x in (r, p):
        x.add_argument(
            "--accept-thread-share",
            action="store_true",
            help="keep pre-2026-10-15 multi-label cells whose one-vs-rest fits ran at another BLAS "
            "thread count than protocols.OVR_FIT_THREADS (the report prints the mixed-share NOTE)",
        )
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if a.cmd == "report":
        report(spec, a.out_dir, a.accept_thread_share)
        return 0
    models = spec["models"]
    if a.models:
        unknown = set(a.models) - {m["name"] for m in models}
        if unknown:
            raise SystemExit(f"--models not in spec: {sorted(unknown)}")
        models = [m for m in models if m["name"] in a.models]
    datasets = a.datasets or spec["datasets"]
    unknown = set(datasets) - set(spec["datasets"])
    if unknown:
        raise SystemExit(f"--datasets not in spec: {sorted(unknown)}")
    warnings.filterwarnings("ignore", category=UserWarning)
    for key, want in (("fractions", a.fractions), ("seeds", a.seeds)):
        if want:
            have = [
                float(x) for x in spec.get(key, DEFAULT_FRACTIONS if key == "fractions" else DEFAULT_SEEDS)
            ]
            bad = [x for x in want if float(x) not in have]
            if bad:
                raise SystemExit(f"--{key} {bad} not in the spec's {key} {have}")
            if (
                key == "seeds"
                and int(have[0]) not in [int(x) for x in want]
                and 1.0 in [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
            ):
                # The seed-free 100 % cell is filed under the spec's FIRST seed; a subset without
                # it would compute that cell under another seed's name, which `report` never reads.
                raise SystemExit(
                    f"--seeds {want} omits the spec's first seed {int(have[0])}, which owns the 100 % "
                    "cell: add it, or drop 1.0 with --fractions"
                )
            spec = {**spec, key: list(want)}
    log = run(spec, models, datasets, a.out_dir, a.redo, a.accept_thread_share)
    n = {s: sum(1 for r in log if r["status"] == s) for s in ("done", "cached", "missing")}
    print(f"MATRIX_RUN_OK {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
