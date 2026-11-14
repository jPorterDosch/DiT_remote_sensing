"""PASS markers that tie a control gate to the feature files it judged (review 2026-11-12,
staged-delta finding 3).

phase11_weak_teacher.json said "Gated by dino_probe.sbatch GATE_ONLY=1 before any cell runs" and
nothing enforced it: experiments/isaac/eval/matrix.sbatch would score a row whose gate had never
run, or had run on features that were extracted again afterwards (CLAUDE.md rule 13: a gate is
executed code that prints PASS / FAIL, not prose).

    # dino_probe.sbatch, after the control gate passed on a full (non-smoke) cache:
    python3 -m eval.gate_marker write --gate dino_control --preset P --dataset D \
        --features 'models/dino_feats/P_D_{split}.npz' --job $SLURM_JOB_ID
    # matrix.sbatch, before a task is scored:
    python3 -m eval.gate_marker require --spec SPEC --model M --dataset D

A spec opts a row in with a TOP-LEVEL map (top-level keys are not part of a cell's identity,
eval.matrix.cell_config; a key inside the model dict would be):

    "requires_gate": {"<model name>": {"gate": "dino_control", "preset": "<extract_dino preset>"}}

`require` exits 0 for a model the map does not name. For a named model it exits 0 only when a
marker exists whose gate / preset / dataset match AND whose recorded size and mtime of every
split's feature file equal the files on disk now. A marker for files that have since changed does
not count: the gate judged other features. Markers are never overwritten.

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

SPLITS = ("train", "val", "test")
DEFAULT_DIR = "results/gates"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _safe(what: str, value: str) -> str:
    if not _NAME.match(value or ""):
        raise SystemExit(f"gate marker: {what}={value!r} is not a plain name ([A-Za-z0-9_.-])")
    return value


def file_stamps(pattern: str, dataset: str | None = None) -> dict:
    """{split: {"path": real path, "size": bytes, "mtime_ns": int}} for every split. A missing or
    empty file is fatal: a gate cannot have judged it."""
    if "{split}" not in pattern:
        raise SystemExit(f"gate marker: features pattern {pattern!r} has no {{split}}")
    out = {}
    for split in SPLITS:
        p = pattern.format(split=split, dataset=dataset)
        try:
            st = os.stat(p)
        except OSError as e:
            raise SystemExit(f"gate marker: feature file {p}: {e.strerror}") from e
        if st.st_size <= 0:
            raise SystemExit(f"gate marker: feature file {p} is empty")
        out[split] = {"path": os.path.realpath(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    return out


def write(gate_dir: str, gate: str, preset: str, dataset: str, pattern: str, job: str) -> str:
    """Write the PASS marker; -> its path. Call ONLY after the gate printed PASS."""
    name = "_".join(
        _safe(k, v) for k, v in (("gate", gate), ("preset", preset), ("dataset", dataset), ("job", job))
    )
    marker = {
        "gate": gate,
        "verdict": "PASS",
        "preset": preset,
        "dataset": dataset,
        "features": file_stamps(pattern),
        "job": job,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(gate_dir, exist_ok=True)
    path = os.path.join(gate_dir, name + ".json")
    try:
        with open(path, "x") as f:
            json.dump(marker, f, indent=2)
    except FileExistsError:
        raise SystemExit(f"{path} exists; a gate marker is never overwritten") from None
    return path


def marker_matches(marker: object, gate: str, preset: str, dataset: str, now: dict) -> tuple[bool, str]:
    """-> (ok, why not). `now` = file_stamps of the features on disk."""
    if not isinstance(marker, dict):
        return False, "not a json object"
    for k, want in (("verdict", "PASS"), ("gate", gate), ("preset", preset), ("dataset", dataset)):
        if marker.get(k) != want:
            return False, f"{k}={marker.get(k)!r}, not {want!r}"
    then = marker.get("features")
    if not isinstance(then, dict) or set(then) != set(now):
        return False, f"splits {sorted(then) if isinstance(then, dict) else then!r}, not {sorted(now)}"
    for split, cur in now.items():
        if then[split] != cur:
            return False, f"{split} features changed since the gate ran ({then[split]} then, {cur} now)"
    return True, ""


def require(spec_path: str, model: str, dataset: str, gate_dir: str) -> tuple[bool, str]:
    """-> (ok, message) for one (model, dataset) task of a matrix spec."""
    with open(spec_path) as f:
        spec = json.load(f)
    gated = spec.get("requires_gate", {})
    if not isinstance(gated, dict):
        raise SystemExit(f"{spec_path}: requires_gate must be a map of model name -> {{gate, preset}}")
    by_name = {m["name"]: m for m in spec["models"]}
    unknown = sorted(set(gated) - set(by_name))
    if unknown:  # a misspelt name would silently gate nothing
        raise SystemExit(f"{spec_path}: requires_gate names {unknown}, not among the spec's models")
    if model not in by_name:
        raise SystemExit(f"{spec_path}: no model {model!r}")
    if model not in gated:
        return True, f"{model} is not listed under requires_gate in {spec_path}"
    entry = gated[model]
    if not isinstance(entry, dict) or set(entry) != {"gate", "preset"}:
        raise SystemExit(
            f"{spec_path}: requires_gate[{model!r}] must be exactly {{gate, preset}}, got {entry!r}"
        )
    gate, preset = _safe("gate", entry["gate"]), _safe("preset", entry["preset"])
    now = file_stamps(by_name[model]["features"], _safe("dataset", dataset))
    files = sorted(glob.glob(os.path.join(gate_dir, f"{gate}_{preset}_{dataset}_*.json")))
    reasons = []
    for p in files:
        try:
            with open(p) as f:
                ok, why = marker_matches(json.load(f), gate, preset, dataset, now)
        except (OSError, ValueError) as e:
            ok, why = False, f"unreadable ({type(e).__name__})"
        if ok:
            return True, f"{gate} PASS for {preset} x {dataset} on the current feature files ({p})"
        reasons.append(f"{p}: {why}")
    detail = "; ".join(reasons) if reasons else f"no marker {gate}_{preset}_{dataset}_*.json under {gate_dir}"
    return (
        False,
        f"{model} x {dataset} needs gate {gate} ({preset}) to have passed on the current features: {detail}",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--gate", required=True)
    w.add_argument("--preset", required=True)
    w.add_argument("--dataset", required=True)
    w.add_argument("--features", required=True, help="feature cache pattern with {split}")
    w.add_argument("--job", required=True)
    w.add_argument("--dir", default=DEFAULT_DIR)
    r = sub.add_parser("require")
    r.add_argument("--spec", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--dir", default=DEFAULT_DIR)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        print(
            f"GATE_MARKER written: {write(a.dir, a.gate, a.preset, a.dataset, a.features, a.job)}", flush=True
        )
        return 0
    ok, msg = require(a.spec, a.model, a.dataset, a.dir)
    print(f"GATE required-marker {'PASS' if ok else 'FAIL'}: {msg}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
