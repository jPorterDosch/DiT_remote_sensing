"""Arm-config gate (review 2026-11-12, item 6): an ablation run may differ from its reference run
in the DECLARED fields only.

arm_full.sh checks four fields of a finished run (target, sem_t_weights, seed, arm). Every other
field is assumed equal to the seed-42 joint run that the tables compare against. This tool
replaces that assumption with a check: it flattens `cfg` and `train_data` of each run's meta.json,
adds the resolved `align` block as `resolved.*`, and compares all of it with the reference run.

  declared     the expectation file names the key AND the value the run must hold. A declared key
               that does not differ, or holds another value, FAILS (a guard must be able to fail in
               both directions).
  logging_only AlignConfig.LOGGING_ONLY keys (checkpoint cadence, diagnostics). Printed, never a
               failure: the training code excludes them from the run identity.
  default      a key that one meta.json lacks (written by older code) and the other holds at the
               AlignConfig default (top-level cfg keys: None). Counted as equal and printed.
  undeclared   anything else. FAILS.

A run directory under --models-dir that the expectation file does not list FAILS, and so does a
listed run without a meta.json: nothing is skipped silently.

What this does NOT prove: that two runs ran the same code. meta.json records no commit, and the
reference predates several config keys. Equal configs are necessary for a fair arm comparison,
not sufficient.

    python3 experiments/isaac/git10m/check_arm_cfg.py --models-dir $STORE/models/git10m

Reads a few dozen small json files: no torch, seconds. Prints `GATE arm cfg PASS|FAIL` and
returns 1 on FAIL. --out writes the full report (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config_types import AlignConfig  # noqa: E402  (stdlib-only module: no torch import)

ABSENT = "<absent>"
# Where the run was written, and what it is called: never part of the comparison.
IGNORED = ("cfg.save_dir", "cfg.align.resume")
# resolved.* keys that restate a cfg.align.* key (compared there) or are derived from the arm.
RESOLVED_RESTATED = (
    "arm",
    "feat",
    "weight",
    "target_shuffle",
    "fm_weight",
    "stop_step",
    "sem_t_weights",
    "k",
)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def view(meta: dict) -> dict:
    """The compared fields of one meta.json, flat."""
    for need in ("cfg", "train_data", "align"):
        if not isinstance(meta.get(need), dict):
            raise ValueError(f"meta.json has no '{need}' block")
    if not isinstance(meta["cfg"].get("align"), dict):
        raise ValueError("meta.json has no cfg.align block")
    out = flatten(meta["cfg"], "cfg.")
    out.update(flatten(meta["train_data"], "train_data."))
    out.update(flatten({k: v for k, v in meta["align"].items() if k not in RESOLVED_RESTATED}, "resolved."))
    return {k: v for k, v in out.items() if k not in IGNORED}


def _default(key: str):
    """The value an absent key stands for, or ABSENT when no default is known."""
    if key.startswith("cfg.align."):
        return getattr(AlignConfig(), key[len("cfg.align.") :], ABSENT)
    if key.startswith("cfg.") and "." not in key[len("cfg.") :]:
        return None
    return ABSENT


def compare(ref: dict, run: dict, declared: dict) -> dict:
    """ref / run: view() dicts. declared: {key: value the run must hold}."""
    logging_only = {f"cfg.align.{k}" for k in AlignConfig.LOGGING_ONLY}
    res = {"declared": {}, "logging_only": {}, "default": {}, "undeclared": {}, "declared_not_met": {}}
    for key in sorted(set(ref) | set(run)):
        a, b = ref.get(key, ABSENT), run.get(key, ABSENT)
        if key in declared:
            if a != b and b == declared[key]:
                res["declared"][key] = [a, b]
            else:
                why = "equals the reference" if a == b else f"run holds {b!r}, expected {declared[key]!r}"
                res["declared_not_met"][key] = why
        elif a == b:
            continue
        elif key in logging_only:
            res["logging_only"][key] = [a, b]
        elif ABSENT in (a, b) and (b if a == ABSENT else a) == _default(key):
            res["default"][key] = [a, b]
        else:
            res["undeclared"][key] = [a, b]
    for key in declared:
        if key not in ref and key not in run:
            res["declared_not_met"][key] = "key is in neither meta.json"
    res["pass"] = not res["undeclared"] and not res["declared_not_met"]
    return res


def check(models_dir: str, expect: dict) -> dict:
    ref_name = expect["reference"]
    runs = expect["runs"]
    if ref_name in runs:
        raise ValueError(f"the reference {ref_name} must not be listed under runs")
    out = {"reference": ref_name, "runs": {}, "unlisted": [], "missing": []}
    ref_path = os.path.join(models_dir, ref_name, "meta.json")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"reference meta.json missing: {ref_path}")
    ref = view(json.load(open(ref_path)))
    on_disk = sorted(d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)))
    out["unlisted"] = [d for d in on_disk if d != ref_name and d not in runs]
    for name, spec in sorted(runs.items()):
        path = os.path.join(models_dir, name, "meta.json")
        if not os.path.isfile(path):
            out["missing"].append(name)
            continue
        out["runs"][name] = compare(ref, view(json.load(open(path))), spec["differs"]) | {
            "role": spec.get("role", "")
        }
    out["pass"] = (
        bool(out["runs"])
        and not out["unlisted"]
        and not out["missing"]
        and all(r["pass"] for r in out["runs"].values())
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", required=True, help="directory holding the git10m_flux_* run directories")
    ap.add_argument("--expect", default=os.path.join(HERE, "arm_cfg_expect.json"))
    ap.add_argument("--out", default="", help="write the full report here (refuses to overwrite)")
    args = ap.parse_args(argv)
    if args.out and os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    res = check(args.models_dir, json.load(open(args.expect)))
    print(f"reference {res['reference']}")
    for name, r in res["runs"].items():
        print(f"{'ok  ' if r['pass'] else 'FAIL'} {name}  [{r['role']}]")
        print(f"       declared: {json.dumps(r['declared'])}")
        for kind in ("undeclared", "declared_not_met"):
            if r[kind]:
                print(f"       {kind.upper()}: {json.dumps(r[kind])}")
        if r["logging_only"] or r["default"]:
            print(
                f"       not compared as differences: logging-only {sorted(r['logging_only'])}; absent = default {sorted(r['default'])}"
            )
    for name in res["unlisted"]:
        print(f"FAIL {name}  UNLISTED: add it to {args.expect} with its declared differences")
    for name in res["missing"]:
        print(f"FAIL {name}  LISTED BUT NO meta.json under {args.models_dir}")
    n_ok = sum(r["pass"] for r in res["runs"].values())
    print(
        f"GATE arm cfg {'PASS' if res['pass'] else 'FAIL'}: {n_ok}/{len(res['runs'])} runs differ from the reference "
        f"in declared fields only; {len(res['unlisted'])} unlisted, {len(res['missing'])} missing"
    )
    if not res["pass"]:
        print("do NOT compare a FAIL run with the reference as if only its declared field changed")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
