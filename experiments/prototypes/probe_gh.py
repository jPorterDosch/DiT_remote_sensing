"""6y candidates G (Fisher sketch) and H (equivariance defect) vs the section-13 base."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _absorption_harness import load_base, run_block  # noqa: E402

CANDIDATES = {
    "G: Fisher sketch (block-9 grads, 4096d)": "results/fisher_sketch_resisc45.npz",
    "H: equivariance defect (D4-ish, 264d)": "results/equivariance_defect_resisc45.npz",
}

base, others, y, si = load_base()
results: dict = {"labels": y, "subset_indices": si}
for name, path in CANDIDATES.items():
    if not os.path.exists(path):
        print(f"SKIP {name}: {path} missing")
        continue
    d = np.load(path)
    assert np.array_equal(d["subset_indices"], si) and np.array_equal(d["labels"], y), name
    blk = d["block"].astype(np.float32)
    assert np.isfinite(blk).all(), name
    r = run_block(name, base, others, blk, y)
    key = name.split(":")[0]
    for k in ("base", "plus", "null"):
        results[f"{key}_{k}"] = r[k]
np.savez("results/probe_gh_resisc45.npz", **results)
print("cached to results/probe_gh_resisc45.npz")
