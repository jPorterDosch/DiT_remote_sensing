"""6z probe: FM loss profile (I) and explicit dt-partial (II) vs the section-13 base."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _absorption_harness import load_base, run_block  # noqa: E402

base, others, y, si = load_base()
d = np.load("results/loss_profile_resisc45.npz")
assert np.array_equal(d["subset_indices"], si) and np.array_equal(d["labels"], y)
results: dict = {"labels": y, "subset_indices": si}
for tag, name in (("I", "I: FM loss profile (7 t)"), ("II", "II: explicit dv/dt at fixed x")):
    blk = d[f"{tag}_block"].astype(np.float32)
    assert np.isfinite(blk).all()
    r = run_block(name, base, others, blk, y)
    for k in ("base", "plus", "null"):
        results[f"{tag}_{k}"] = r[k]
np.savez("results/probe_ij_resisc45.npz", **results)
print("cached to results/probe_ij_resisc45.npz")
