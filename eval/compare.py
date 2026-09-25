"""Paired per-image comparison of two eval.probe result files (B - A), image-level
bootstrap (rule 4), cell by cell.

    python -m eval.compare results/eval/<A>.npz results/eval/<B>.npz

Pairing guards (rule 6, fire-tested in eval/gates.py): same dataset and protocol, the same
cells, identical evaluated images per cell. For cv/budget/mlp the ordered path lists must
match exactly (folds/splits depend on order). For official the test sets must be the SAME
set of images; they are aligned by split/class/file (the tail of the path), because FLUX
caches and DINO feature files list the images in different orders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # noqa: E402

from eval import wb  # noqa: E402
from eval.protocols import ci  # noqa: E402


def _key(p: str) -> str:
    return "/".join(os.path.normpath(p).split(os.sep)[-3:])


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    cells = [str(c) for c in d["cells"]]
    return {
        "path": path,
        "arm": str(d["arm"]),
        "dataset": str(d["dataset"]),
        "protocol": str(d["protocol"]),
        "smoke": bool(d["smoke"]),
        "paths": [str(p) for p in d["paths"]],
        "labels": d["labels"],
        "cells": {c: (d[f"correct__{c}"].astype(np.float64), d[f"ev__{c}"]) for c in cells},
    }


def pair(a: dict, b: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {cell: (a_vec, b_vec)} aligned per image, or raises SystemExit."""
    for k in ("dataset", "protocol"):
        if a[k] != b[k]:
            raise SystemExit(f"UNPAIRED: {k} {a[k]!r} vs {b[k]!r}")
    if a["smoke"] or b["smoke"]:
        print("WARNING: comparing SMOKE results -- not quotable")
    if set(a["cells"]) != set(b["cells"]):
        raise SystemExit(f"UNPAIRED: cells {sorted(a['cells'])} vs {sorted(b['cells'])}")
    out = {}
    if a["protocol"] == "official":
        ka, kb = [_key(p) for p in a["paths"]], [_key(p) for p in b["paths"]]
        if len(set(ka)) != len(ka) or set(ka) != set(kb):
            raise SystemExit("UNPAIRED: official test sets are not the same images")
        pos_b = {k: i for i, k in enumerate(kb)}
        perm = np.array([pos_b[k] for k in ka])
        if not np.array_equal(a["labels"], b["labels"][perm]):
            raise SystemExit("UNPAIRED: labels disagree for the same test images")
        for c, (va, ea) in a["cells"].items():
            vb, eb = b["cells"][c]
            if not (np.array_equal(ea, np.arange(len(ka))) and np.array_equal(eb, np.arange(len(kb)))):
                raise SystemExit(f"UNPAIRED: official cell {c} does not cover the test split")
            out[c] = (va, vb[perm])
        return out
    if a["paths"] != b["paths"]:
        n = sum(x == y for x, y in zip(a["paths"], b["paths"]))
        raise SystemExit(f"UNPAIRED: identity paths differ ({n}/{len(a['paths'])} positions agree)")
    if not np.array_equal(a["labels"], b["labels"]):
        raise SystemExit("UNPAIRED: labels differ")
    for c, (va, ea) in a["cells"].items():
        vb, eb = b["cells"][c]
        if not np.array_equal(ea, eb):
            raise SystemExit(f"UNPAIRED: cell {c} evaluated different images")
        out[c] = (va, vb)
    return out


def compare(a_path: str, b_path: str) -> list[dict]:
    a, b = load(a_path), load(b_path)
    rows = []
    for c, (va, vb) in sorted(pair(a, b).items()):
        d = vb - va
        lo, hi = ci(d)
        verdict = "PARITY (CI spans 0)" if lo <= 0 <= hi else ("B > A" if lo > 0 else "A > B")
        rows.append(
            {
                "cell": c,
                "acc_a": float(va.mean()),
                "acc_b": float(vb.mean()),
                "delta": float(d.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "n": len(d),
                "verdict": verdict,
            }
        )
        print(
            f"  {c:<10} A {va.mean():.4f}  B {vb.mean():.4f}  B-A {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  -> {verdict}"
        )
    return rows


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="baseline result .npz (A)")
    p.add_argument("b", help="candidate result .npz (B); reported delta is B - A")
    args = p.parse_args(argv)
    a, b = load(args.a), load(args.b)
    config = {
        "a": wb.file_identity(args.a),
        "b": wb.file_identity(args.b),
        "bootstrap": {"n": 10000, "seed": 0},
    }
    _, name = wb.init("compare", a["dataset"], f"{b['arm']}-vs-{a['arm']}", config, job_type="compare")
    print(f"== {name}: B={b['arm']} vs A={a['arm']} ({a['protocol']})")
    wb.use_results(args.a)
    wb.use_results(args.b)
    rows = compare(args.a, args.b)
    if wandb.run is not None:
        wb.log_table("paired_deltas", rows)
        for r in rows:
            wandb.run.summary[f"delta/{r['cell']}"] = r["delta"]
            wandb.run.summary[f"ci/{r['cell']}"] = json.dumps([r["ci_lo"], r["ci_hi"]])
        wandb.run.summary["delta/mean_over_cells"] = float(np.mean([r["delta"] for r in rows]))
        wandb.finish()
    return rows


if __name__ == "__main__":
    main()
