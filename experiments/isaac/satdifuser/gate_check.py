"""PASS / FAIL for the SatDiFuser m-eurosat reproduction gate (review 2026-11-12, item 10).

gate_eurosat.sbatch used to end with an unconditional `echo SDF_GATE_OK`: whatever their run.py
measured, the log said OK, and "nothing of theirs is dumped until this passes" was a sentence in
the README. This module reads the output of their run.py and decides.

PASS needs all of:
  - exactly one "Final test metric: <x>" line (their run.py prints it once, after loading the
    best-on-val checkpoint; none = the run died or found no checkpoint, two = two runs in one log),
  - no "No best checkpoint found" line,
  - x a fraction in [0, 1],
  - |100 x - published| <= tolerance, on either side. A number well ABOVE theirs is as much an
    instrument discrepancy as one below.

Published = SatDiFuser Table 2, m-eurosat top-1, per fuser. Tolerance 0.5 points: README.md's
"within ~0.5 of their gw 97.7", written 2026-10-17, before any gate run existed (none has run as
of 2026-11-12).

    gate_check.py verdict --log RUN.log --fuser gw --out $SDF/gate/eurosat_gw_<jobid>.json
    gate_check.py require --dir $SDF/gate --fuser gw      # dump.sbatch: exit 1 unless the gate passed

`verdict` writes the json for PASS and for FAIL (never over an existing file) and exits 1 on
FAIL. `require` passes only when at least one verdict for the fuser exists and every one of them
is PASS: a FAIL stays in the way until someone has diagnosed it and moved the file aside.
`require` does not trust the stored "gate" field: it re-judges the stored metric against
PUBLISHED and refuses a verdict whose tolerance is looser than TOL_PP. `--tolerance` can only
tighten the gate.

Stdlib only (runs in their conda env and under tests/test_satdifuser_gate.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

PUBLISHED = {"gw": 97.7, "lw": 96.8, "moe": 97.3}  # Table 2, m-eurosat top-1 (%)
TOL_PP = 0.5
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FINAL = re.compile(r"Final test metric:\s*(\S+)")


def verdict(text: str, fuser: str, tolerance: float = TOL_PP) -> dict:
    """-> {"gate": "PASS" | "FAIL", "reason", "metric", "published", "tolerance_pp", "fuser"}."""
    if fuser not in PUBLISHED:
        raise SystemExit(f"fuser {fuser!r} has no published m-eurosat number (known: {sorted(PUBLISHED)})")
    ref = PUBLISHED[fuser]
    out = {"fuser": fuser, "published": ref, "tolerance_pp": tolerance, "metric": None, "gate": "FAIL"}
    text = _ANSI.sub("", text)
    found = _FINAL.findall(text)
    if "No best checkpoint found" in text:
        return {**out, "reason": "run.py reported 'No best checkpoint found': no test evaluation happened"}
    if len(found) != 1:
        return {**out, "reason": f"expected exactly one 'Final test metric' line, found {len(found)}"}
    try:
        m = float(found[0])
    except ValueError:
        return {**out, "reason": f"unparseable metric {found[0]!r}"}
    out["metric"] = m
    if not (math.isfinite(m) and 0.0 <= m <= 1.0):
        return {**out, "reason": f"metric {m!r} is not a fraction in [0, 1]"}
    if not (math.isfinite(tolerance) and tolerance > 0):
        return {**out, "reason": f"tolerance {tolerance!r} is not a positive number"}
    gap = 100.0 * m - ref
    if abs(gap) > tolerance:
        return {
            **out,
            "reason": f"{100 * m:.2f} vs published {ref} (gap {gap:+.2f}, tolerance +/-{tolerance})",
        }
    return {
        **out,
        "gate": "PASS",
        "reason": f"{100 * m:.2f} vs published {ref} (gap {gap:+.2f}, tolerance +/-{tolerance})",
    }


def line(v: dict) -> str:
    s = f"GATE satdifuser-eurosat {v['gate']} [{v['fuser']}]: {v['reason']}"
    return s if v["gate"] == "PASS" else s + " -- do NOT dump or quote any SatDiFuser feature"


def stored_ok(v: object, fuser: str) -> tuple[bool, str]:
    """Re-judge a stored verdict against THIS module's constants instead of trusting its "gate"
    field (review 2026-11-12, staged-delta finding 2: a verdict written with --tolerance 5 said
    PASS and `require` believed it). -> (ok, why not)."""
    if not isinstance(v, dict):
        return False, "not a json object"
    if v.get("gate") != "PASS":
        return False, f"gate={v.get('gate')!r}"
    if v.get("fuser") != fuser:
        return False, f"fuser={v.get('fuser')!r}, not {fuser!r}"
    if fuser not in PUBLISHED or v.get("published") != PUBLISHED[fuser]:
        return False, f"published={v.get('published')!r}, not {PUBLISHED.get(fuser)!r}"
    tol, m = v.get("tolerance_pp"), v.get("metric")
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not (math.isfinite(tol) and 0 < tol <= TOL_PP)
    ):
        return False, f"tolerance_pp={tol!r} is not in (0, {TOL_PP}]"
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not (math.isfinite(m) and 0.0 <= m <= 1.0):
        return False, f"metric={m!r} is not a fraction in [0, 1]"
    gap = 100.0 * m - PUBLISHED[fuser]
    if abs(gap) > tol:
        return False, f"stored PASS but gap {gap:+.2f} exceeds its tolerance +/-{tol}"
    return True, ""


def require(gate_dir: str, fuser: str) -> tuple[bool, str]:
    """-> (ok, message). ok iff >= 1 verdict file for the fuser and every one of them is a PASS
    that still holds when re-judged by `stored_ok` (tolerance no looser than TOL_PP)."""
    files = sorted(glob.glob(os.path.join(gate_dir, f"eurosat_{fuser}_*.json")))
    if not files:
        return False, f"no gate verdict for fuser {fuser} under {gate_dir} (run gate_eurosat.sbatch first)"
    bad = []
    for f in files:
        try:
            with open(f) as fh:
                ok, why = stored_ok(json.load(fh), fuser)
        except (OSError, ValueError) as e:
            ok, why = False, f"unreadable ({type(e).__name__})"
        if not ok:
            bad.append(f"{f}: {why}")
    if bad:
        return False, f"gate verdict(s) not PASS: {bad} (diagnose, then move aside; a FAIL is never ignored)"
    return True, f"{len(files)} verdict(s), all PASS: {files}"


def _tolerance(text: str) -> float:
    """argparse type: the gate may be made STRICTER on the command line, never looser."""
    x = float(text)
    if not (math.isfinite(x) and 0 < x <= TOL_PP):
        raise argparse.ArgumentTypeError(f"tolerance must be in (0, {TOL_PP}], got {text!r}")
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--log", required=True, help="captured stdout+stderr of their run.py")
    v.add_argument("--fuser", required=True)
    v.add_argument("--tolerance", type=_tolerance, default=TOL_PP, help=f"points, in (0, {TOL_PP}]")
    v.add_argument("--out", required=True, help="verdict json (refused if it exists)")
    v.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", ""))
    r = sub.add_parser("require")
    r.add_argument("--dir", required=True)
    r.add_argument("--fuser", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "require":
        ok, msg = require(args.dir, args.fuser)
        print(f"GATE satdifuser-eurosat required: {'PASS' if ok else 'FAIL'}: {msg}", flush=True)
        return 0 if ok else 1
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; a verdict is never overwritten")
    with open(args.log, errors="replace") as f:
        res = verdict(f.read(), args.fuser, args.tolerance)
    res.update(log=os.path.abspath(args.log), job=args.job)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x") as f:
        json.dump(res, f, indent=2)
    print(line(res), flush=True)
    return 0 if res["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
