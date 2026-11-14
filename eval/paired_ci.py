"""Paired image-level bootstrap intervals for arm-vs-arm differences (review finding 8, 2026-11-12).

The matrix report gives a paired interval only for "model vs its teacher at 100 %", and the
segmentation report gives each arm its own interval. Two per-arm intervals that overlap say
nothing about the DIFFERENCE between the arms, because both arms are scored on the same test
images and their errors are correlated. This tool reads the stored per-image results of two arms,
resamples the TEST IMAGES (the independent unit, CLAUDE.md rule 4) with the same draw for both
arms, recomputes the metric of each arm on every draw and reports the 2.5 / 97.5 percentiles of
the difference.

    seg  cells (eval.seg_probe):  test_conf_per_image (N, C, C)  -> mIoU of the summed confusion
    cls  cells (eval.matrix):     correct (N,)                   -> top-1 accuracy
                                  correct (N, 3) = tp, fp, fn    -> micro-F1 (m_bigearthnet)

For a fraction with several seeds the statistic is mean over seeds of metric(arm) minus the same
for the base, and each bootstrap draw reweights the same test images in every seed. The interval
is therefore CONDITIONAL on the probe-training subsets and the fitted probes: it measures test-set
sampling noise, not the variance from drawing another training subset (the seed std in the
summary tables covers that, with 3 seeds). Nothing is selected here: every (dataset, fraction,
arm) present is reported, and the number of intervals is printed so the reader can discount for
multiplicity.

Gates, executed every run (rule 13), printed as `GATE ... PASS|FAIL`:
  identity      a pair is refused (listed, no number) unless both arms hold the same test paths
                in the same order (and labels, for cls); a refusal is counted and printed
  reproduction  the point metric recomputed from the per-image data equals the value the harness
                stored in the cell (test_miou / score), every cell
  null          base vs base gives delta 0 and the interval [0, 0]

    python -m eval.paired_ci --kind seg --cells results/seg/phase3_lr/cells \
        --base flux-pretrained --out results/paired/seg_phase3_lr
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

CELL_RE = re.compile(r"^f([0-9.]+)_s(\d+)\.npz$")
N_BOOT = 10_000
REPRO_TOL = 1e-9


class Refused(Exception):
    """The two arms cannot be paired (different test images / order / seeds)."""


# ------------------------------------------------------------------------------ metrics
def miou_batch(conf: np.ndarray) -> np.ndarray:
    """mIoU of (R, C, C) summed confusions (rows = reference); classes with an empty union are
    skipped per replicate, NaN when none is left. Same rule as eval.seg_probe.miou."""
    conf = np.asarray(conf, np.float64)
    inter = np.einsum("rcc->rc", conf)
    union = conf.sum(1) + conf.sum(2) - inter
    ok = union > 0
    iou = np.where(ok, inter / np.where(ok, union, 1.0), 0.0)
    n = ok.sum(1)
    return np.where(n > 0, iou.sum(1) / np.maximum(n, 1), np.nan)


def micro_f1_batch(sums: np.ndarray) -> np.ndarray:
    """micro-F1 of (R, 3) summed tp, fp, fn. Same rule as eval.protocols._micro_f1."""
    tp, fp, fn = np.moveaxis(np.asarray(sums, np.float64), -1, 0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def metric_of(per_image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Metric of one arm under (R, N) image weights. per_image: (N,) correctness, (N, 3) counts
    or (N, C, C) confusions."""
    w = np.asarray(weights, np.float64)
    x = np.asarray(per_image, np.float64)
    if x.ndim == 1:
        return (w @ x) / w.sum(1)
    if x.ndim == 2 and x.shape[1] == 3:
        return micro_f1_batch(w @ x)
    if x.ndim == 3 and x.shape[1] == x.shape[2]:
        return miou_batch((w @ x.reshape(len(x), -1)).reshape(len(w), *x.shape[1:]))
    raise ValueError(f"per-image data of shape {x.shape} is not correctness, tp/fp/fn or confusions")


def bootstrap_weights(n_images: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n_images) resampling counts: each row is one draw of n_images with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n_boot).astype(np.float32)


def paired_interval(base: list[np.ndarray], arm: list[np.ndarray], weights: np.ndarray) -> dict:
    """mean over seeds of metric(arm) - metric(base), with the percentile interval over the image
    draws in `weights`. base / arm: one per-image array per seed, same seeds in the same order."""
    if len(base) != len(arm) or not base:
        raise Refused(f"seed counts differ ({len(base)} vs {len(arm)})")
    n = base[0].shape[0]
    if any(x.shape[0] != n for x in base + arm) or weights.shape[1] != n:
        raise Refused("per-image arrays and weights disagree on the number of test images")
    ones = np.ones((1, n))
    point = float(
        np.mean([metric_of(b, ones)[0] for b in arm]) - np.mean([metric_of(a, ones)[0] for a in base])
    )
    draws = np.mean([metric_of(b, weights) for b in arm], axis=0) - np.mean(
        [metric_of(a, weights) for a in base], axis=0
    )
    n_nan = int(np.isnan(draws).sum())
    if n_nan == len(draws):
        raise Refused("every bootstrap draw is NaN")
    lo, hi = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))
    return {"delta": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "n_images": int(n), "n_seeds": len(base), "n_boot": int(len(draws)), "nan_draws": n_nan}  # fmt: skip


# ------------------------------------------------------------------------------ cells
KINDS = {
    "seg": {"per_image": "test_conf_per_image", "stored": "test_miou", "paths": "test_paths",
            "labels": None, "train": "train_indices"},
    "cls": {"per_image": "correct", "stored": "score", "paths": "paths", "labels": "labels", "train": "train_rows"},
}  # fmt: skip


def list_cells(cells_dir: str, arm: str, dataset: str) -> dict[float, dict[int, str]]:
    d = os.path.join(cells_dir, arm, dataset)
    out: dict[float, dict[int, str]] = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        m = CELL_RE.match(f)
        if m:
            out.setdefault(float(m.group(1)), {})[int(m.group(2))] = os.path.join(d, f)
    return out


def load_cell(path: str, kind: str) -> dict:
    k = KINDS[kind]
    with np.load(path, allow_pickle=False) as z:
        c = {
            "path": path,
            "per_image": z[k["per_image"]],
            "stored": float(z[k["stored"]]),
            "paths": z[k["paths"]],
            "labels": z[k["labels"]] if k["labels"] else None,
            "train": z[k["train"]],
        }
    recomputed = float(metric_of(c["per_image"], np.ones((1, len(c["per_image"]))))[0])
    c["repro_abs_diff"] = abs(recomputed - c["stored"])
    return c


def check_pairable(a: dict, b: dict) -> None:
    """Refuse unless the two cells score the same test images in the same order (rule 6: pair on
    identity, not on shapes)."""
    if a["paths"].shape != b["paths"].shape or not np.array_equal(a["paths"], b["paths"]):
        raise Refused(f"test paths differ: {a['path']} vs {b['path']}")
    if (a["labels"] is None) != (b["labels"] is None):
        raise Refused("one cell has labels and the other has none")
    if a["labels"] is not None and not np.array_equal(a["labels"], b["labels"]):
        raise Refused(f"test labels differ: {a['path']} vs {b['path']}")
    if a["per_image"].shape != b["per_image"].shape:
        raise Refused(f"per-image shapes differ: {a['per_image'].shape} vs {b['per_image'].shape}")


def compare(
    cells_dir: str, kind: str, base: str, arms: list[str], datasets: list[str], n_boot: int, seed: int
) -> dict:
    rows, refused, repro_max, n_cells = [], [], 0.0, 0
    null_ok, null_n = True, 0
    weights: dict[int, np.ndarray] = {}
    for dataset in datasets:
        base_cells = list_cells(cells_dir, base, dataset)
        if not base_cells:
            refused.append(
                {"dataset": dataset, "arm": "*", "frac": None, "reason": f"no cells for base {base}"}
            )
            continue
        loaded_base = {
            f: {s: load_cell(p, kind) for s, p in by_seed.items()} for f, by_seed in base_cells.items()
        }
        for f, by_seed in loaded_base.items():
            for c in by_seed.values():
                repro_max, n_cells = max(repro_max, c["repro_abs_diff"]), n_cells + 1
            n = len(next(iter(by_seed.values()))["per_image"])
            if n not in weights:
                weights[n] = bootstrap_weights(n, n_boot, seed)
            per = [by_seed[s]["per_image"] for s in sorted(by_seed)]
            r = paired_interval(per, per, weights[n])
            null_n += 1
            null_ok &= r["delta"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0
        for arm in arms:
            arm_cells = list_cells(cells_dir, arm, dataset)
            if not arm_cells:
                refused.append({"dataset": dataset, "arm": arm, "frac": None, "reason": "no cells"})
                continue
            for f in sorted(set(loaded_base) | set(arm_cells)):
                entry = {"dataset": dataset, "arm": arm, "base": base, "frac": f}
                try:
                    if f not in loaded_base or f not in arm_cells:
                        raise Refused("fraction present in only one arm")
                    if set(loaded_base[f]) != set(arm_cells[f]):
                        raise Refused(
                            f"seeds differ: base {sorted(loaded_base[f])} vs arm {sorted(arm_cells[f])}"
                        )
                    seeds = sorted(arm_cells[f])
                    a = [loaded_base[f][s] for s in seeds]
                    b = [load_cell(arm_cells[f][s], kind) for s in seeds]
                    for c in b:
                        repro_max, n_cells = max(repro_max, c["repro_abs_diff"]), n_cells + 1
                    for ca, cb in zip(a, b):
                        check_pairable(ca, cb)
                    for c in a[1:] + b[1:]:  # the same test images in every seed
                        check_pairable(a[0], c)
                    n = len(a[0]["per_image"])
                    r = paired_interval([c["per_image"] for c in a], [c["per_image"] for c in b], weights[n])
                    r["same_train_subsets"] = bool(
                        all(np.array_equal(ca["train"], cb["train"]) for ca, cb in zip(a, b))
                    )
                    r["base_metric"] = float(np.mean([c["stored"] for c in a]))
                    r["arm_metric"] = float(np.mean([c["stored"] for c in b]))
                    rows.append({**entry, **r})
                except Refused as e:
                    refused.append({**entry, "reason": str(e)})
    gates = {
        "identity": {"pass": True, "refused": len(refused)},  # a refusal is the guard firing, reported below
        "reproduction": {
            "pass": bool(n_cells > 0 and repro_max <= REPRO_TOL),
            "max_abs_diff": repro_max,
            "cells": n_cells,
        },
        "null": {"pass": bool(null_n > 0 and null_ok), "comparisons": null_n},
    }
    return {"kind": kind, "cells_dir": cells_dir, "base": base, "arms": arms, "datasets": datasets,
            "n_boot": n_boot, "seed": seed, "unit": "test image", "rows": rows, "refused": refused, "gates": gates}  # fmt: skip


def markdown(res: dict) -> str:
    metric = "mIoU" if res["kind"] == "seg" else "top-1 / micro-F1"
    lines = [
        f"# Paired intervals vs `{res['base']}` ({metric}, {res['kind']} cells `{res['cells_dir']}`)",
        "",
        f"Delta = arm - base, mean over seeds. 95 % percentile interval from {res['n_boot']} paired resamples of the",
        "test images (same draw for both arms and every seed). Conditional on the probe-training subsets; it does not",
        f"include seed-to-seed variance. {len(res['rows'])} intervals are listed and none is corrected for multiplicity.",
        "`*` = interval excludes 0.",
        "",
    ]
    for g, v in res["gates"].items():
        lines.append(
            f"- GATE {g} {'PASS' if v['pass'] else 'FAIL'}: {json.dumps({k: x for k, x in v.items() if k != 'pass'})}"
        )
    lines.append("")
    for dataset in res["datasets"]:
        rows = [r for r in res["rows"] if r["dataset"] == dataset]
        if not rows:
            continue
        lines += [
            f"## {dataset}",
            "",
            "| arm | frac | seeds | n | base | arm | delta | 95 % interval | |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (r["arm"], r["frac"])):
            lines.append(
                f"| {r['arm']} | {r['frac']:g} | {r['n_seeds']} | {r['n_images']} | {r['base_metric']:.6f} | {r['arm_metric']:.6f} "
                f"| {r['delta']:+.6f} | [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}] | {'*' if r['excludes_zero'] else ''} |"
            )
        lines.append("")
    if res["refused"]:
        lines += ["## Refused (no number)", ""]
        lines += [f"- {r['dataset']} / {r['arm']} / frac {r['frac']}: {r['reason']}" for r in res["refused"]]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument(
        "--cells", required=True, help="the cells directory (<cells>/<arm>/<dataset>/f<frac>_s<seed>.npz)"
    )
    ap.add_argument("--base", required=True, help="arm every other arm is compared against")
    ap.add_argument("--arms", nargs="*", default=None, help="default: every other arm directory")
    ap.add_argument("--datasets", nargs="*", default=None, help="default: every dataset the base has")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.json and <out>.md")
    a = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(a.cells, a.base)):
        raise SystemExit(f"no base arm directory {os.path.join(a.cells, a.base)}")
    arms = (
        a.arms
        if a.arms
        else sorted(d for d in os.listdir(a.cells) if d != a.base and os.path.isdir(os.path.join(a.cells, d)))
    )
    unknown = [m for m in arms if not os.path.isdir(os.path.join(a.cells, m))]
    if unknown:
        raise SystemExit(f"no cells directory for arms {unknown}")
    datasets = a.datasets if a.datasets else sorted(os.listdir(os.path.join(a.cells, a.base)))
    for ext in (".json", ".md"):
        if os.path.exists(a.out + ext):
            raise SystemExit(f"{a.out + ext} exists: refusing to overwrite")
    res = compare(a.cells, a.kind, a.base, arms, datasets, a.n_boot, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write(markdown(res))
    ok = True
    for g, v in res["gates"].items():
        ok &= v["pass"]
        print(
            f"GATE {g} {'PASS' if v['pass'] else 'FAIL'}: {json.dumps({k: x for k, x in v.items() if k != 'pass'})}",
            flush=True,
        )
    for r in res["refused"]:
        print(f"REFUSED {r['dataset']} / {r['arm']} / frac {r['frac']}: {r['reason']}", flush=True)
    print(f"{len(res['rows'])} intervals, {len(res['refused'])} refused -> {a.out}.json / .md", flush=True)
    if not ok:
        print("GATE FAIL: do NOT quote any interval above", flush=True)
        return 1
    print("PAIRED_CI_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
