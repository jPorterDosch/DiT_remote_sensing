"""6w battery probe: evaluate candidates B, C, D against the section-13 strongest base
(RESISC45). Candidate A runs inside eps_spread_probe.py (it also validates the harness).

Each block is appended standardized-raw to the protected base and judged by the rule-3b
double bar via the shared harness. Pairing is by subset_indices values, asserted, never
assumed. D (n=1500) subselects the base rows by recorded keep positions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _absorption_harness import load_base, run_block  # noqa: E402

CANDIDATES = {
    "B: guidance directions (46 prompts, t260)": "results/guidance_direction_resisc45.npz",
    "C: field FD probes (3t x 4 dirs)": "results/field_probes_resisc45.npz",
    "D: chain contraction (n1500)": "results/lyapunov_chains_resisc45.npz",
}


def main() -> None:
    base, others, y, si = load_base()
    results: dict = {}
    for name, path in CANDIDATES.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} missing")
            continue
        d = np.load(path)
        blk, bsi, by = d["block"], d["subset_indices"], d["labels"]
        assert np.isfinite(blk).all(), f"{name}: non-finite block"
        if len(bsi) == len(si):
            assert np.array_equal(bsi, si), f"{name}: subset_indices mismatch"
            assert np.array_equal(by, y), f"{name}: labels mismatch"
            b, o, yy = base, others, y
        else:  # D: paired subsample — subselect base rows by recorded positions
            keep = d["keep_positions"]
            assert np.array_equal(si[keep], bsi), f"{name}: keep positions do not map into base"
            assert np.array_equal(y[keep], by), f"{name}: labels mismatch after keep"
            b, o, yy = base[keep], others[keep], by
        r = run_block(name, b, o, blk.astype(np.float32), yy)
        key = name.split(":")[0]
        results[f"{key}_base"] = r["base"]
        results[f"{key}_plus"] = r["plus"]
        results[f"{key}_null"] = r["null"]
        if "zero_shot_acc" in d:
            print(f"    (zero-shot argmin-L, descriptive: {float(d['zero_shot_acc']):.4f})")

    os.makedirs("results", exist_ok=True)
    np.savez("results/flow_battery_probe_resisc45.npz", **results, labels=y, subset_indices=si)
    print("per-image vectors cached to results/flow_battery_probe_resisc45.npz")


if __name__ == "__main__":
    main()
