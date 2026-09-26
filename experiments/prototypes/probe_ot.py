"""6z-III: OT-geometry block (tortuosity + segment lengths + anchor ratios) vs the
section-13 base. Zero GPU — computed from solver_curvature_n5000 cached states.
Anchor rows excluded from the probe so no image is its own reference."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _absorption_harness import load_base, run_block  # noqa: E402

N_ANCHORS = 64

base, others, y, si = load_base()
d = np.load("models/solver_curvature_n5000/resisc45_solvercurv_n50.npz")
assert np.array_equal(d["subset_indices"], si)
states = d["states"].astype(np.float64)  # (N, 7, 256, 64)
n = len(y)
flat = states.reshape(n, 7, -1)  # (N, 7, 16384)

seg = np.linalg.norm(np.diff(flat, axis=1), axis=2)  # (N, 6) segment lengths
chord = np.linalg.norm(flat[:, -1] - flat[:, 0], axis=1)  # (N,)
tort = seg.sum(1) / np.maximum(chord, 1e-8)

disp = flat[:, -1] - flat[:, 0]  # (N, 16384) displacement z580 - z100
rng = np.random.default_rng(0)
anchors = []
for cls in np.unique(y):
    pos = np.flatnonzero(y == cls)
    anchors.append(rng.choice(pos, size=max(1, N_ANCHORS // len(np.unique(y))), replace=False))
anchors = np.sort(np.concatenate(anchors))[:N_ANCHORS]
A = disp[anchors]  # (K, D)
An = A / np.linalg.norm(A, axis=1, keepdims=True)
dn = disp / np.linalg.norm(disp, axis=1, keepdims=True)
cos = dn @ An.T  # (N, K)
dist = np.linalg.norm(disp[:, None, :] - A[None, :, :], axis=2) if False else None
# memory-safe pairwise distances: ||d_i - a_k||^2 = ||d_i||^2 + ||a_k||^2 - 2 d_i.a_k
di2 = (disp**2).sum(1)[:, None]
ak2 = (A**2).sum(1)[None, :]
dist = np.sqrt(np.maximum(di2 + ak2 - 2 * (disp @ A.T), 0))

block = np.concatenate([tort[:, None], seg, chord[:, None], cos, dist], axis=1).astype(np.float32)
print("OT block:", block.shape, "finite:", bool(np.isfinite(block).all()),
      "| tortuosity mean/range:", tort.mean().round(3), (tort.min().round(3), tort.max().round(3)))

keep = np.setdiff1d(np.arange(n), anchors)  # anchors excluded from the probe
r = run_block("6z-III: OT geometry (tortuosity + anchors)", base[keep], others[keep], block[keep], y[keep])
np.savez("results/probe_ot_resisc45.npz", labels=y[keep], subset_indices=si[keep],
         block=block, anchors=anchors, tortuosity=tort.astype(np.float32),
         base=r["base"], plus=r["plus"], null=r["null"])
print("cached to results/probe_ot_resisc45.npz")
