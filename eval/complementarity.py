"""Teacher-complementarity diagnostics (reviewer priority 2, 2026-11-10): is adapted FLUX merely
the DINOv3-SAT teacher distilled into a larger model, or do its features carry predictive
information the teacher's do not?

    python -m eval.complementarity agree  [--datasets ...] [--out-dir results/complementarity]
    python -m eval.complementarity run    --dataset m_so2sat [--arms T,T+FA,...] [--smoke-train 600 --c-grid 1e-3,1e-1]
    python -m eval.complementarity report [--datasets ...] [--smoke]

Roles (default; --roles overrides): T = dinov3-sat (the alignment teacher), F0 = flux-pretrained-k28
(frozen FLUX at the block the adapted arms use), FA = flux-proj (joint adaptation, seed 42). Every
role is read through the label-efficiency matrix's own 100 % cell
(results/matrix/cells/<model>/<dataset>/f1_s0.npz): the feature files, pins and sha1s are the
cell's, so this instrument and the paper's tables cannot drift apart.

agree   Prediction agreement and error overlap between the 100 % cells of two models on the SAME
        official test images (paths + labels verified equal, rule 6): agreement, both right /
        A only / B only / both wrong, oracle (either right), and the error-consistency kappa
        (Geirhos et al. 2020: observed consistency of the correct/wrong pattern against the
        consistency two independent classifiers of the same accuracies would show). Image-level
        bootstrap (rule 4; multi-label entries are clustered by image). Reference pairs give the
        scale: two seeds of the adapted arm (the noise ceiling) and the two DINOv3 teachers.
        No fit, no selection: it re-reads cached per-image vectors.

run     The concatenation probe, on the official splits, under the matrix protocol
        (StandardScaler on train, logistic regression, max_iter 3000, the matrix C grid and thread
        policy). One arm = one design matrix [base | block], fit at EVERY C; val metric and the
        test predictions of every C are stored (rule 12), nothing is selected here.
          view      one FIXED view per role, read from that role's 100 % cell: the candidate with
                    the best VAL metric among the non-concatenated candidates (first max in the
                    cell's order). For DINO that is the cell's selected view; for FLUX a single
                    timestep (3072-d), so the candidate's capacity is controlled. Chosen on val,
                    never on test (rule 1).
          base      identical in every arm (rule 3): the only transform is the per-column
                    StandardScaler, which cannot let a block displace base columns. No PCA
                    anywhere: neither side is compressed (rule 3a).
          null      the block ROW-SHUFFLED within each split (same width, same marginals, image
                    correspondence destroyed); a transformed block is shuffled after the transform.
          controls  T+noise6 (6 iid-noise columns), T+anchor (labels tiled to the FLUX width plus
                    noise, sd 4: a block that contains the answer) and its shuffled null, and a
                    weak anchor (aggregate SNR 1) that is reported as measured detection power.
        Draws are keyed by (purpose, block, split), so an arm is the same whichever job fits it.

report  For base B and candidate X: raw = metric(B+X) - metric(B) and vs-null = metric(B+X) -
        metric(B+shuffled X), each a paired per-image statistic with a 10k image-level bootstrap
        CI. "X adds beyond B" requires BOTH > 0 (rule 3b: a redundant copy also beats its null).
        Primary operating point: every arm at its own val-selected C (the paper's protocol: the
        selection rule is identical for all arms and disjoint from test). Because arms differ in
        width, the same deltas are also reported at every FIXED C (rule 8) and the convergence
        warnings of every arm are printed.
        GATE (executed, rule 13), per dataset:
          (a) probe sanity     T+noise6 - T straddles 0;
          (b) detection power  T+anchor beats its own shuffled null;
          (c) reproduction     every single-role arm reproduces its matrix cell's val column.
        On FAIL the dataset's rows are printed with "do NOT quote".

What a positive result does NOT show: that generative pretraining CAUSED the extra information
(FLUX differs from the teacher in data, objective, architecture and size at once). What a null
shows: no linearly accessible addition at this test-set size; the weak anchor gives the scale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval import features as F  # noqa: E402
from eval import matrix as M  # noqa: E402
from eval import protocols as P  # noqa: E402

OUT_DIR = "results/complementarity"
CELLS_DIR = "results/matrix"
DATASETS = ("m_eurosat", "m_brick_kiln", "m_pv4ger", "m_so2sat", "m_forestnet", "m_bigearthnet")
ROLES = {"T": "dinov3-sat", "F0": "flux-pretrained-k28", "FA": "flux-proj"}
SPLITS = ("train", "val", "test")
SEED = 0
N_BOOT = 10000
NOISE_WIDTH = 6
ANCHOR_SD = {"anchor": 4.0, "anchorw": None}  # None: sd = sqrt(copies per label), aggregate SNR 1
VERSION = "2026-11-10"  # bump when a code change alters what an arm computes

# arm -> (base role, block). Blocks: ("feat", role) | ("null", role) | ("noise", width) |
# ("anchor", name) | ("null_anchor", name). The reverse pairs reuse the forward concatenation
# (F0+T is the same design as T+F0 with the columns permuted).
ARMS = {
    "T": ("T", None),
    "F0": ("F0", None),
    "FA": ("FA", None),
    "T+F0": ("T", ("feat", "F0")),
    "T+null(F0)": ("T", ("null", "F0")),
    "T+FA": ("T", ("feat", "FA")),
    "T+null(FA)": ("T", ("null", "FA")),
    "F0+null(T)": ("F0", ("null", "T")),
    "FA+null(T)": ("FA", ("null", "T")),
    "T+noise6": ("T", ("noise", NOISE_WIDTH)),
    "T+anchor": ("T", ("anchor", "anchor")),
    "T+null(anchor)": ("T", ("null_anchor", "anchor")),
    "T+anchorw": ("T", ("anchor", "anchorw")),
    "T+null(anchorw)": ("T", ("null_anchor", "anchorw")),
}
# (label, base arm, base+candidate arm, base+null arm)
PAIRS = (
    ("teacher + frozen FLUX", "T", "T+F0", "T+null(F0)"),
    ("teacher + adapted FLUX", "T", "T+FA", "T+null(FA)"),
    ("frozen FLUX + teacher", "F0", "T+F0", "F0+null(T)"),
    ("adapted FLUX + teacher", "FA", "T+FA", "FA+null(T)"),
)
CONTROL_PAIRS = (
    ("CTRL strong anchor (sd 4)", "T", "T+anchor", "T+null(anchor)"),
    ("CTRL weak anchor (SNR 1)", "T", "T+anchorw", "T+null(anchorw)"),
)
# reference pairs of `agree`: (label, model A, model B)
AGREE_PAIRS = (
    ("teacher vs frozen FLUX", "dinov3-sat", "flux-pretrained-k28"),
    ("teacher vs adapted FLUX", "dinov3-sat", "flux-proj"),
    ("frozen vs adapted FLUX", "flux-pretrained-k28", "flux-proj"),
    ("REF adapted seed 42 vs 43", "flux-proj", "flux-proj-s43"),
    ("REF teacher vs DINOv3-web", "dinov3-sat", "dinov3-web"),
)
AGREE_CONTRAST = ("teacher vs adapted FLUX", "teacher vs frozen FLUX")  # kappa(first) - kappa(second)


# ----------------------------------------------------------------------------- cells
def load_cell(cells_dir: str, model: str, dataset: str) -> dict:
    path = M.cell_path(cells_dir, model, dataset, 1.0, 0)
    if not os.path.exists(path):
        raise SystemExit(f"no 100 % matrix cell for {model} x {dataset}: {path}")
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in ("correct", "pred", "labels", "paths")}
    for k in ("selected", "val_table", "config", "inputs"):
        out[k] = json.loads(str(d[k]))
    out["score"] = float(d["score"])
    out["path"] = path
    return out


def check_same_images(a: dict, b: dict, what: str) -> None:
    """Rule 6: pair on the image list, never on labels alone (sorted labels pass for disjoint
    subsets). Both the ordered path keys and the labels must be equal."""
    pa, pb = ([F.path_key(str(p)) for p in x["paths"]] for x in (a, b))
    if pa != pb:
        n_same = sum(x == y for x, y in zip(pa, pb))
        raise SystemExit(
            f"{what}: test image lists differ ({n_same}/{max(len(pa), len(pb))} positions agree)"
        )
    if a["labels"].shape != b["labels"].shape or not np.array_equal(a["labels"], b["labels"]):
        raise SystemExit(f"{what}: test labels differ on identical image lists")


# ----------------------------------------------------------------------------- agreement
def agreement_counts(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Per-image sufficient counts (N, 7): units, agree, both right, A only, B only, both wrong,
    both wrong with the SAME prediction. Single-label: one unit per image. Multi-hot: one unit
    per (image, label) decision, summed per image so the bootstrap resamples images (rule 4)."""
    y, pa, pb = (np.asarray(x) for x in (y, pa, pb))
    if not (y.shape == pa.shape == pb.shape):
        raise ValueError(f"shape mismatch: labels {y.shape}, preds {pa.shape} {pb.shape}")
    ca, cb = pa == y, pb == y
    cols = [np.ones_like(ca), pa == pb, ca & cb, ca & ~cb, ~ca & cb, ~ca & ~cb, ~ca & ~cb & (pa == pb)]
    S = np.stack([c.astype(np.int64) for c in cols], axis=-1)
    return S if y.ndim == 1 else S.sum(axis=1)


def agreement_from_sums(s: np.ndarray) -> dict:
    """Statistics from summed counts (..., 7); vectorized over leading axes (bootstrap)."""
    n, agree, bc, ao, bo, bw, same = (s[..., i].astype(np.float64) for i in range(7))
    acc_a, acc_b = (bc + ao) / n, (bc + bo) / n
    obs = (bc + bw) / n
    exp = acc_a * acc_b + (1 - acc_a) * (1 - acc_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(exp < 1, (obs - exp) / (1 - exp), np.nan)
        same_wrong = np.where(bw > 0, same / bw, np.nan)
    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "agreement": agree / n,
        "both_right": bc / n,
        "a_only": ao / n,
        "b_only": bo / n,
        "both_wrong": bw / n,
        "oracle": 1 - bw / n,
        "kappa": kappa,
        "same_wrong_frac": same_wrong,
    }


def boot_sums(S: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, 7) summed counts of image-level resamples. The index stream depends only on
    (seed, N), so two pairs on the same images get the SAME resamples (paired contrasts)."""
    idx = np.random.default_rng(seed).integers(0, len(S), (n_boot, len(S)))
    return np.concatenate([S[i].sum(axis=1) for i in np.array_split(idx, 20)])


def _q(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return (
        (float("nan"), float("nan"))
        if not len(x)
        else (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
    )


def agreement(y, pa, pb, n_boot: int = N_BOOT) -> tuple[dict, dict]:
    """Point statistics with bootstrap CIs, and the raw bootstrap arrays (for paired contrasts)."""
    S = agreement_counts(y, pa, pb)
    point = agreement_from_sums(S.sum(axis=0))
    boot = agreement_from_sums(boot_sums(S, n_boot))
    out = {k: {"value": float(v), "ci": list(_q(boot[k]))} for k, v in point.items()}
    out["n_images"] = int(len(S))
    out["n_units"] = int(S[:, 0].sum())
    return out, boot


def run_agree(args) -> int:
    res = {"version": VERSION, "n_boot": N_BOOT, "pairs": [list(p) for p in AGREE_PAIRS], "datasets": {}}
    lines = [
        "# Prediction agreement and error overlap (100 % cells, official test split)",
        "",
        "Image-level bootstrap 95 % CIs (10k). kappa = error consistency (0: the overlap two",
        "independent classifiers of these accuracies would show; 1: identical right/wrong pattern).",
        "oracle = share of decisions at least one of the two gets right. m_bigearthnet: one unit per",
        "(image, label) decision, resampled by image; its accuracies are label-wise, not micro-F1.",
        "",
    ]
    for ds in args.datasets:
        cells, boots, rows = {}, {}, {}
        for label, a, b in AGREE_PAIRS:
            for m in (a, b):
                if m not in cells:
                    cells[m] = load_cell(args.cells_dir, m, ds)
            check_same_images(cells[a], cells[b], f"{ds}: {a} vs {b}")
            rows[label], boots[label] = agreement(cells[a]["labels"], cells[a]["pred"], cells[b]["pred"])
            rows[label]["models"] = [a, b]
            rows[label]["cells"] = [cells[a]["path"], cells[b]["path"]]
            rows[label]["selected"] = [cells[a]["selected"], cells[b]["selected"]]
        hi, lo = AGREE_CONTRAST
        point = rows[hi]["kappa"]["value"] - rows[lo]["kappa"]["value"]
        contrast = {"value": point, "ci": list(_q(boots[hi]["kappa"] - boots[lo]["kappa"]))}
        res["datasets"][ds] = {"pairs": rows, "kappa_contrast": {"minus": [hi, lo], **contrast}}
        lines += [
            f"## {ds}",
            "",
            "| pair | acc A | acc B | agreement | both right | A only | B only | both wrong | oracle | kappa |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for label, r in rows.items():

            def f(k, r=r):
                return f"{100 * r[k]['value']:.1f}"

            ci = r["kappa"]["ci"]
            lines.append(
                f"| {label} | {f('acc_a')} | {f('acc_b')} | {f('agreement')} | {f('both_right')} | {f('a_only')} | "
                f"{f('b_only')} | {f('both_wrong')} | {f('oracle')} | {r['kappa']['value']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |"
            )
        lines += [
            "",
            f"kappa({hi}) - kappa({lo}) = {point:+.3f} [{contrast['ci'][0]:+.3f}, {contrast['ci'][1]:+.3f}] "
            "(paired bootstrap; > 0: adaptation moved FLUX's errors toward the teacher's)",
            "",
        ]
        print("\n".join(lines[-(len(rows) + 6) :]), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "agreement.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(args.out_dir, "agreement.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"AGREE_OK {len(args.datasets)} datasets -> {args.out_dir}/agreement.{{json,md}}", flush=True)
    return 0


# ----------------------------------------------------------------------------- features
def pick_view(cell: dict) -> dict:
    """The role's fixed view: best VAL metric among the non-concatenated candidates of its 100 %
    cell, first max in the cell's (candidate, C) order -- the cell's own tie rule."""
    rows = [r for r in cell["val_table"] if not r["candidate"].startswith("concat")]
    if not rows:
        raise SystemExit(f"{cell['path']}: no non-concatenated candidate in the val table")
    best = rows[0]
    for r in rows[1:]:
        if r["val_acc"] > best["val_acc"]:
            best = r
    col = {r["C"]: r["val_acc"] for r in cell["val_table"] if r["candidate"] == best["candidate"]}
    return {
        "view": best["candidate"],
        "cell_C": best["C"],
        "cell_val": best["val_acc"],
        "cell_val_column": col,
    }


def flux_split_paths(files: list[str]) -> dict[str, list[str]]:
    """Ordered image path keys per split, straight from the FLUX cache shards (the loader returns
    only the test paths)."""
    by_split: dict[str, list] = {}
    for h in files:
        by_split.setdefault(F.cache_meta(h).get("split", "train"), []).append(h)
    out = {}
    for split, hits in by_split.items():
        idx, paths = [], []
        for h in hits:
            d = np.load(h, allow_pickle=True)
            idx.append(np.asarray(d["subset_indices"]))
            paths.append(np.asarray(d["paths"]))
        order = np.argsort(np.concatenate(idx))
        out[split] = [F.path_key(str(p)) for p in np.concatenate(paths)[order]]
    return out


def load_role(cells_dir: str, model: str, dataset: str) -> dict:
    """One role's fixed-view features on the three official splits, through the matrix loaders
    with the model entry stored in the role's 100 % cell. Refuses if a feature file changed since
    the cell was written, if a row is non-finite, or (FLUX) if any split's cached image list is
    not the official listing in order."""
    cell = load_cell(cells_dir, model, dataset)
    pick = pick_view(cell)
    entry = {**cell["config"]["model"], "view": pick["view"]}
    cands, ytr, yva, yte, paths, inputs = M.load_cands(entry, dataset)
    if list(cands) != [pick["view"]]:
        raise SystemExit(f"{model} x {dataset}: asked for view {pick['view']!r}, loader built {list(cands)}")
    sha = M.inputs_sha(inputs)
    if sha != cell["inputs"]:
        raise SystemExit(
            f"{model} x {dataset}: feature files differ from the ones {cell['path']} was computed on"
        )
    bad_train, bad_eval = M.nonfinite_rows(cands)
    if len(bad_train) or bad_eval:
        raise SystemExit(
            f"{model} x {dataset}: non-finite features (train rows {bad_train.tolist()[:8]}, eval {bad_eval})"
        )
    if entry["kind"] == "flux":
        got = flux_split_paths(inputs)
        for split in SPLITS:
            want = [F.path_key(p) for p in F.list_official_split(dataset, split)[0]]
            if got.get(split) != want:
                raise SystemExit(
                    f"{model} x {dataset}: cached {split} image list is not the official listing in order"
                )
    return {
        "model": model,
        "X": cands[pick["view"]],
        "y": (ytr, yva, yte),
        "paths": [F.path_key(str(p)) for p in paths],
        "inputs": sha,
        "pick": pick,
        "entry": entry,
        "cell": cell["path"],
    }


def check_roles(roles: dict) -> None:
    """Every role must hold the same ordered images on every split (rules 6/7/11). Train and val
    order is the official listing for each role (DINO: loader identity check; FLUX: load_role);
    here: equal row counts and labels on all splits and equal test path keys."""
    names = list(roles)
    ref = roles[names[0]]
    for n in names[1:]:
        r = roles[n]
        if r["paths"] != ref["paths"]:
            raise SystemExit(f"roles {names[0]} and {n}: test image lists differ")
        for s, ya, yb, Xa, Xb in zip(SPLITS, ref["y"], r["y"], ref["X"], r["X"]):
            if len(Xa) != len(Xb) or ya.shape != yb.shape or not np.array_equal(ya, yb):
                raise SystemExit(f"roles {names[0]} and {n}: {s} rows / labels differ")


# ----------------------------------------------------------------------------- blocks
def _rng(*key) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32("/".join(str(k) for k in key).encode())])


def null_of(block, key: str):
    """Row-shuffled block: one permutation per split (rows never cross splits)."""
    return tuple(B[_rng("null", key, s).permutation(len(B))] for B, s in zip(block, SPLITS))


def noise_block(ns, width: int):
    return tuple(_rng("noise", width, s).standard_normal((n, width)) for n, s in zip(ns, SPLITS))


def label_matrix(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[y] if y.ndim == 1 else y.astype(np.float64)


def anchor_sd(name: str, width: int, n_labels: int) -> float:
    """Noise sd of an anchor: the fixed strong value, or sqrt(copies per label) so that the mean
    over a label's copies has signal 1 against noise sd 1."""
    sd = ANCHOR_SD[name]
    return float(sd) if sd is not None else float(np.sqrt(max(1, width // n_labels)))


def anchor_block(ys, width: int, name: str):
    """A block that contains the answer: the label matrix tiled to `width` columns plus iid noise
    (curv5000_probe's construction). Uses the split's own labels on val and test by design -- it is
    the positive control of the instrument, never a result."""
    n_classes = int(max(y.max() for y in ys)) + 1 if ys[0].ndim == 1 else ys[0].shape[1]
    sd = anchor_sd(name, width, n_classes)
    out = []
    for y, s in zip(ys, SPLITS):
        L = label_matrix(y, n_classes)
        tiled = np.hstack([L] * (width // L.shape[1] + 1))[:, :width]
        out.append(tiled + sd * _rng("anchor", name, s).standard_normal((len(y), width)))
    return tuple(out)


def arm_roles(arm: str) -> set[str]:
    base, block = ARMS[arm]
    need = {base}
    if block and block[0] in ("feat", "null"):
        need.add(block[1])
    if block and block[0] in ("anchor", "null_anchor"):
        need.add("FA")  # the anchor is width-matched to the adapted-FLUX block
    return need


def build_arm(arm: str, roles: dict):
    """([base | block] per split, width of base, width of block)."""
    base, block = ARMS[arm]
    B = roles[base]["X"]
    ys = roles[base]["y"]
    if block is None:
        return B, B[0].shape[1], 0
    kind, what = block
    if kind == "feat":
        X = roles[what]["X"]
    elif kind == "null":
        X = null_of(roles[what]["X"], what)
    elif kind == "noise":
        X = noise_block([len(b) for b in B], int(what))
    elif kind == "anchor":
        X = anchor_block(ys, roles["FA"]["X"][0].shape[1], what)
    elif kind == "null_anchor":
        X = null_of(anchor_block(ys, roles["FA"]["X"][0].shape[1], what), what)
    else:
        raise ValueError(f"unknown block {block}")
    for b, x in zip(B, X):
        if len(b) != len(x):
            raise SystemExit(f"{arm}: base and block row counts differ ({len(b)} vs {len(x)})")
    return tuple(np.hstack([b, x]) for b, x in zip(B, X)), B[0].shape[1], X[0].shape[1]


# ----------------------------------------------------------------------------- fit
def fit_arm(X, ys, c_grid) -> dict:
    """Every C of the grid for one design, at the matrix operating point (blas_threads; the
    one-vs-rest pool is memory-budgeted inside official_all_cells). No selection."""
    from threadpoolctl import threadpool_limits

    ytr, yva, yte = ys
    threads = M.blas_threads(len(ytr))
    t0 = time.time()
    with threadpool_limits(limits=threads, user_api="blas"):
        rows, test_preds = P.official_all_cells({"arm": X}, ytr, yva, yte, c_grid)
    if [r["C"] for r in rows] != list(c_grid):
        raise RuntimeError("official_all_cells returned an unexpected C order")
    return {
        "val_metric": np.array([r["val_acc"] for r in rows]),
        "conv_warnings": np.array([r["conv_warnings"] for r in rows]),
        "test_pred": np.stack([test_preds[("arm", c)] for c in c_grid]).astype(np.int16),
        "compute": {
            "blas_threads": threads,
            "cpus": len(os.sched_getaffinity(0)),
            "ovr_n_jobs": int(os.environ.get("OVR_N_JOBS", "1")) if ytr.ndim == 2 else 1,
            "ovr_threads": P.OVR_FIT_THREADS if ytr.ndim == 2 else threads,
            "seconds": round(time.time() - t0, 1),
        },
    }


def slug(arm: str) -> str:
    return arm.replace("+", "__").replace("(", "-").replace(")", "")


def arm_path(out_dir: str, dataset: str, arm: str) -> str:
    return os.path.join(out_dir, "arms", dataset, f"{slug(arm)}.npz")


def strided(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


def arm_config(arm: str, dataset: str, roles: dict, c_grid, smoke_train: int | None) -> dict:
    return {
        "arm": arm,
        "definition": [ARMS[arm][0], list(ARMS[arm][1]) if ARMS[arm][1] else None],
        "dataset": dataset,
        "roles": {
            r: {
                "model": roles[r]["model"],
                "view": roles[r]["pick"]["view"],
                "cell": roles[r]["cell"],
                "entry": roles[r]["entry"],
            }
            for r in sorted(arm_roles(arm))
        },
        "c_grid": list(c_grid),
        "max_iter": P.OFFICIAL_MAX_ITER,
        "matrix_protocol": M.PROTOCOL_VERSION,
        "blas_threads_cap": M.BLAS_THREADS_CAP,
        "seed": SEED,
        "version": VERSION,
        "smoke_train": smoke_train,
    }


def run_arms(args) -> int:
    ds = args.dataset
    F.official_spec(ds)
    arms = args.arms or list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; have {list(ARMS)}")
    smoke = args.smoke_train is not None
    out_dir = os.path.join(args.out_dir, "smoke") if smoke else args.out_dir
    if args.c_grid and not smoke:
        raise SystemExit("--c-grid is a smoke option; quotable arms use the matrix grid of the role cells")
    model_of = {**ROLES, **dict(kv.split("=", 1) for kv in (args.roles or []))}
    need = sorted(set().union(*(arm_roles(a) for a in arms)))
    roles = {r: load_role(args.cells_dir, model_of[r], ds) for r in need}
    check_roles(roles)
    grids = {r: tuple(load_cell(args.cells_dir, model_of[r], ds)["config"]["c_grid"]) for r in need}
    if len(set(grids.values())) != 1:
        raise SystemExit(f"{ds}: the role cells were computed on different C grids: {grids}")
    c_grid = tuple(float(c) for c in args.c_grid.split(",")) if args.c_grid else next(iter(grids.values()))
    if smoke:  # strided train subset (labels are class-sorted, so every class appears); val/test whole
        keep = strided(len(roles[need[0]]["y"][0]), args.smoke_train)
        for r in roles.values():
            r["X"] = (r["X"][0][keep], r["X"][1], r["X"][2])
            r["y"] = (r["y"][0][keep], r["y"][1], r["y"][2])
    for r in need:
        p = roles[r]["pick"]
        print(f"{ds} role {r} = {roles[r]['model']} view {p['view']} width {roles[r]['X'][0].shape[1]} "
              f"(cell val {p['cell_val']:.4f} at C={p['cell_C']:g}) rows {[len(x) for x in roles[r]['X']]}", flush=True)  # fmt: skip
    done = 0
    for arm in arms:
        path = arm_path(out_dir, ds, arm)
        cfg = json.loads(json.dumps(arm_config(arm, ds, roles, c_grid, args.smoke_train)))
        if os.path.exists(path):
            old = json.loads(str(np.load(path, allow_pickle=True)["config"]))
            if old == cfg:
                print(f"{ds} {arm}: cached", flush=True)
                continue
            if not args.redo:
                raise SystemExit(f"{path}: config changed since it was written (pass --redo to recompute)")
            keep_at = os.path.join(out_dir, "stale", ds, f"{slug(arm)}.{int(time.time())}.npz")
            os.makedirs(os.path.dirname(keep_at), exist_ok=True)
            os.replace(path, keep_at)
            print(f"stale arm kept at {keep_at}", flush=True)
        X, w_base, w_block = build_arm(arm, roles)
        ys = roles[ARMS[arm][0]]["y"]
        fit = fit_arm(X, ys, c_grid)
        del X
        repro = None
        if ARMS[arm][1] is None and not smoke:  # gate (c): the cell's val column for this view
            col = roles[arm]["pick"]["cell_val_column"]
            repro = float(max(abs(v - col[c]) for v, c in zip(fit["val_metric"], c_grid)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(
            tmp,
            c_grid=np.array(c_grid),
            val_metric=fit["val_metric"],
            conv_warnings=fit["conv_warnings"],
            test_pred=fit["test_pred"],
            labels=ys[2],
            paths=np.array(roles[ARMS[arm][0]]["paths"]),
            n_train=np.array(len(ys[0])),
            n_val=np.array(len(ys[1])),
            width_base=np.array(w_base),
            width_block=np.array(w_block),
            repro_max_abs_val_diff=np.array(np.nan if repro is None else repro),
            config=np.array(json.dumps(cfg, sort_keys=True)),
            inputs=np.array(
                json.dumps({r: roles[r]["inputs"] for r in sorted(arm_roles(arm))}, sort_keys=True)
            ),
            compute=np.array(json.dumps(fit["compute"], sort_keys=True)),
        )
        os.replace(tmp, path)
        i = int(np.argmax(fit["val_metric"]))
        test = P.metric(P.per_image(ys[2], fit["test_pred"][i].astype(np.int64)))
        print(
            f"{ds} {arm:<16} width {w_base}+{w_block} sel C={c_grid[i]:g} val={fit['val_metric'][i]:.4f} TEST={test:.4f} "
            f"warn@sel={fit['conv_warnings'][i]} warnC={int((fit['conv_warnings'] > 0).sum())}/{len(c_grid)} "
            f"{fit['compute']['seconds']:.0f}s" + ("" if repro is None else f" repro_dval={repro:.2e}"),
            flush=True,
        )
        done += 1
    print(f"COMPLEMENTARITY_RUN_OK {ds} done={done} cached={len(arms) - done} out={out_dir}", flush=True)
    return 0


# ----------------------------------------------------------------------------- report
def load_arm(out_dir: str, dataset: str, arm: str) -> dict | None:
    path = arm_path(out_dir, dataset, arm)
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    a = {k: d[k] for k in ("c_grid", "val_metric", "conv_warnings", "test_pred", "labels", "paths")}
    a.update(
        {k: float(d[k]) for k in ("n_train", "n_val", "width_base", "width_block", "repro_max_abs_val_diff")}
    )
    a["config"] = json.loads(str(d["config"]))
    a["compute"] = json.loads(str(d["compute"]))
    a["sel"] = int(np.argmax(a["val_metric"]))  # first max in C order: the protocol's tie rule
    a["vec"] = [P.per_image(a["labels"], p.astype(np.int64)) for p in a["test_pred"]]
    return a


def verdict(raw: tuple, vs_null: tuple) -> str:
    """Rule 3b: 'adds' needs both the raw and the null-corrected CI above zero."""
    if raw[1] > 0 and vs_null[1] > 0:
        return "ADDS beyond the base"
    if raw[2] < 0:
        return "hurts"
    if vs_null[1] > 0:
        return "image-linked, no net gain (redundant with the base)"
    return "no detectable addition"


def pair_stats(arms: dict, base: str, plus: str, null: str) -> dict | None:
    if any(arms.get(a) is None for a in (base, plus, null)):
        return None
    B, X, N = arms[base], arms[plus], arms[null]
    for other in (X, N):
        if list(other["paths"]) != list(B["paths"]) or not np.array_equal(other["labels"], B["labels"]):
            raise SystemExit(f"arms {base} / {plus} / {null}: test images differ")
        if not np.array_equal(other["c_grid"], B["c_grid"]):
            raise SystemExit(f"arms {base} / {plus} / {null}: C grids differ")

    def at(ib, ix, inull):
        raw = P.paired_delta(B["vec"][ib], X["vec"][ix], N_BOOT)
        vs = P.paired_delta(N["vec"][inull], X["vec"][ix], N_BOOT)
        pen = P.paired_delta(B["vec"][ib], N["vec"][inull], N_BOOT)
        return {
            "raw": list(raw),
            "vs_null": list(vs),
            "width_penalty": list(pen),
            "verdict": verdict(raw, vs),
        }

    out = at(B["sel"], X["sel"], N["sel"])
    out["scores"] = {k: P.metric(a["vec"][a["sel"]]) for k, a in (("base", B), ("plus", X), ("null", N))}
    out["C"] = {k: float(a["c_grid"][a["sel"]]) for k, a in (("base", B), ("plus", X), ("null", N))}
    out["warn_at_sel"] = {
        k: int(a["conv_warnings"][a["sel"]]) for k, a in (("base", B), ("plus", X), ("null", N))
    }
    fixed = [at(i, i, i) for i in range(len(B["c_grid"]))]
    out["fixed_C"] = [{"C": float(c), **f} for c, f in zip(B["c_grid"], fixed)]
    out["fixed_C_at_base_sel"] = fixed[B["sel"]]
    out["n_C_adds"] = sum(f["verdict"].startswith("ADDS") for f in fixed)
    return out


def _d(t) -> str:
    return f"{100 * t[0]:+.2f} [{100 * t[1]:+.2f}, {100 * t[2]:+.2f}]"


def gate(arms: dict, smoke: bool) -> dict:
    g = {}
    if arms.get("T") is not None and arms.get("T+noise6") is not None:
        d = P.paired_delta(
            arms["T"]["vec"][arms["T"]["sel"]], arms["T+noise6"]["vec"][arms["T+noise6"]["sel"]], N_BOOT
        )
        g["a_probe_sanity"] = {"delta": list(d), "pass": bool(d[1] <= 0 <= d[2])}
    if arms.get("T+anchor") is not None and arms.get("T+null(anchor)") is not None:
        A, Nn = arms["T+anchor"], arms["T+null(anchor)"]
        d = P.paired_delta(Nn["vec"][Nn["sel"]], A["vec"][A["sel"]], N_BOOT)
        g["b_detection_power"] = {"delta": list(d), "pass": bool(d[1] > 0)}
    singles = {a: arms[a] for a in ("T", "F0", "FA") if arms.get(a) is not None}
    if singles and not smoke:
        multilabel = next(iter(singles.values()))["labels"].ndim == 2
        # single-label: the same deterministic fit, so at most one val image of slack; multi-label:
        # the documented ~1e-3 thread-share effect of the one-vs-rest fits (matrix.config_current).
        tol = {a: (2e-3 if multilabel else 1.0 / s["n_val"] + 1e-12) for a, s in singles.items()}
        diffs = {a: s["repro_max_abs_val_diff"] for a, s in singles.items()}
        g["c_reproduction"] = {
            "max_abs_val_diff": diffs,
            "tolerance": tol,
            "pass": all(diffs[a] <= tol[a] for a in singles),
        }
    need = ("a_probe_sanity", "b_detection_power") + (() if smoke else ("c_reproduction",))
    g["complete"] = all(k in g for k in need)
    g["pass"] = g["complete"] and all(g[k]["pass"] for k in need)
    return g


def run_report(args) -> int:
    out_dir = os.path.join(args.out_dir, "smoke") if args.smoke else args.out_dir
    res = {"version": VERSION, "n_boot": N_BOOT, "smoke": bool(args.smoke), "datasets": {}}
    lines = [
        "# Teacher complementarity: concatenation probes (official splits, 100 % labels)",
        "",
        "raw = metric(base + X) - metric(base); vs null = metric(base + X) - metric(base + row-shuffled X);",
        "width penalty = metric(base + shuffled X) - metric(base). Points x100, paired image-level",
        "bootstrap 95 % CI (10k). Every arm at its own val-selected C; 'fixed C' repeats the test with",
        "all three arms at one C (the count of grid values where the verdict is ADDS). 'ADDS' needs",
        "raw > 0 AND vs null > 0 (CI excluding 0). Per-comparison intervals, no multiplicity correction.",
        "Metric: top-1; m_bigearthnet micro-F1.",
        "",
    ]
    if args.smoke:
        lines[:0] = ["**SMOKE RUN (strided train subset, reduced C grid): no number here is quotable.**", ""]
    n_fail = 0
    for ds in args.datasets:
        arms = {a: load_arm(out_dir, ds, a) for a in ARMS}
        have = [a for a, v in arms.items() if v is not None]
        if not have:
            lines += [f"## {ds}", "", "MISSING: no arm computed.", ""]
            continue
        missing = [a for a in ARMS if a not in have]
        g = gate(arms, args.smoke)
        pairs = {label: pair_stats(arms, b, x, n) for label, b, x, n in PAIRS + CONTROL_PAIRS}
        roles = {r: v for a in have for r, v in arms[a]["config"]["roles"].items()}
        lines += [f"## {ds}", ""]
        lines += [f"- role {r}: {v['model']}, view `{v['view']}`" for r, v in sorted(roles.items())]
        if missing:
            lines.append(f"- MISSING arms: {', '.join(missing)}")
        lines += [
            "",
            "| arm | width | C | val | test | warnings at C | C values with warnings |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in have:
            v = arms[a]
            i = v["sel"]
            lines.append(
                f"| {a} | {int(v['width_base'])}+{int(v['width_block'])} | {v['c_grid'][i]:g} | {100 * v['val_metric'][i]:.2f} | "
                f"{100 * P.metric(v['vec'][i]):.2f} | {int(v['conv_warnings'][i])} | {int((v['conv_warnings'] > 0).sum())}/{len(v['c_grid'])} |"
            )
        lines += [
            "",
            "| base + X | base | + X | + null | raw | vs null | width penalty | fixed C: ADDS | verdict |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for label, _, _, _ in PAIRS + CONTROL_PAIRS:
            p = pairs[label]
            if p is None:
                lines.append(f"| {label} | MISSING | | | | | | | |")
                continue
            s = p["scores"]
            lines.append(
                f"| {label} | {100 * s['base']:.2f} | {100 * s['plus']:.2f} | {100 * s['null']:.2f} | {_d(p['raw'])} | "
                f"{_d(p['vs_null'])} | {_d(p['width_penalty'])} | {p['n_C_adds']}/{len(p['fixed_C'])} | {p['verdict']} |"
            )
        lines += ["", f"GATE for {ds}:"]
        if "a_probe_sanity" in g:
            lines.append(
                f"- (a) probe sanity: 6 noise columns straddle 0: {_d(g['a_probe_sanity']['delta'])} {'PASS' if g['a_probe_sanity']['pass'] else 'FAIL'}"
            )
        if "b_detection_power" in g:
            lines.append(
                f"- (b) detection power: strong anchor beats its null: {_d(g['b_detection_power']['delta'])} {'PASS' if g['b_detection_power']['pass'] else 'FAIL'}"
            )
        if "c_reproduction" in g:
            c = g["c_reproduction"]
            worst = ", ".join(f"{a} {v:.1e}" for a, v in c["max_abs_val_diff"].items())
            lines.append(
                f"- (c) reproduction of the matrix cells' val columns (max abs diff): {worst} {'PASS' if c['pass'] else 'FAIL'}"
            )
        if not g["complete"]:
            lines.append("- => GATE INCOMPLETE (control arms missing): do NOT quote any cell above.")
            n_fail += 1
        elif g["pass"]:
            lines.append("- => GATE PASS: the rows above are readable.")
        else:
            lines.append(f"- => GATE FAIL on {ds}: do NOT quote any cell above.")
            n_fail += 1
        lines.append("")
        res["datasets"][ds] = {
            "roles": roles,
            "missing_arms": missing,
            "arms": {
                a: {
                    "C": float(arms[a]["c_grid"][arms[a]["sel"]]),
                    "val": float(arms[a]["val_metric"][arms[a]["sel"]]),
                    "test": P.metric(arms[a]["vec"][arms[a]["sel"]]),
                    "test_by_C": [P.metric(v) for v in arms[a]["vec"]],
                    "val_by_C": arms[a]["val_metric"].tolist(),
                    "conv_warnings": arms[a]["conv_warnings"].tolist(),
                    "width": [int(arms[a]["width_base"]), int(arms[a]["width_block"])],
                    "compute": arms[a]["compute"],
                }
                for a in have
            },
            "pairs": pairs,
            "gate": g,
        }
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "concat.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out_dir, "concat.md"), "w") as fh:
        fh.write(text)
    print(
        f"COMPLEMENTARITY_REPORT_OK datasets={len(args.datasets)} gate_not_passed={n_fail} -> {out_dir}/concat.{{json,md}}",
        flush=True,
    )
    return 0


# ----------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("agree", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default=OUT_DIR)
        p.add_argument("--cells-dir", default=CELLS_DIR, help="the matrix out dir holding cells/")
        if name == "run":
            p.add_argument("--dataset", required=True)
            p.add_argument(
                "--arms", type=lambda s: [a for a in s.split(",") if a], help=f"comma list of {list(ARMS)}"
            )
            p.add_argument(
                "--roles",
                nargs="*",
                help="override role models, e.g. FA=flux-proj-s43 (stamped in every arm)",
            )
            p.add_argument(
                "--smoke-train",
                type=int,
                help="SMOKE: strided train subset of this size; writes under <out>/smoke",
            )
            p.add_argument("--c-grid", help="SMOKE only: comma list of C values")
            p.add_argument(
                "--redo",
                action="store_true",
                help="recompute arms whose config changed (old arm kept under stale/)",
            )
        else:
            p.add_argument("--datasets", nargs="+", default=list(DATASETS))
        if name == "report":
            p.add_argument(
                "--smoke", action="store_true", help="report the smoke arms (<out>/smoke); never quotable"
            )
    args = ap.parse_args(argv)
    return {"agree": run_agree, "run": run_arms, "report": run_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
