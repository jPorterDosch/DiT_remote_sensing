"""6w candidate A: eps-ensemble SPREAD as a feature block (RESISC45 only).

The ens-8 mean deliberately averages away the across-eps variability of the trunk's
features; the per-channel std across independent eps draws is a per-image noise-
sensitivity fingerprint the cached mean cannot contain. Zero GPU: computed from the six
divctl ens1 caches (eps seeds 43..48, t=[100,580], subset verified identical to the base).

Pre-registered block (RESEARCH_NOTES 6w-A): per-channel std over the 6 seeds at both
cached t's -> 2x3072 = 6144 d. Side-arm (diagnostic only): the 6-seed MEAN block.

STEP 0 of this script validates the copied harness by reproducing the section-13
curvature-pattern numbers (base 0.8536, raw -0.0037, DoD +0.0024 ns) before any new
block is evaluated; a mismatch aborts the battery.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _absorption_harness import RESISC45_BASE, ci, load_base, run_block  # noqa: E402


def pool(x, g=2):  # verbatim from curv_absorption_recheck.py, for the validation arm
    n, k, t, d = x.shape
    hw = int(round(t**0.5))
    sp = torch.from_numpy(x).reshape(n * k, hw, hw, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(sp, (g, g)).reshape(n, k * d * g * g).numpy()


def main() -> None:
    base, others, y, si = load_base()
    n = len(y)

    # ---- STEP 0: harness validation against the recorded section-13 result ----
    dc = np.load("models/solver_curvature_n5000/resisc45_solvercurv_n50.npz")
    assert np.array_equal(dc["subset_indices"], si), "curvature cache pairing broke"
    raw = dc["pred_mid"] - dc["pred"]
    nr = np.linalg.norm(raw.reshape(n, raw.shape[1], -1), axis=2)
    pattern = pool(raw / (nr[:, :, None, None] + 1e-8))
    val = run_block("VALIDATION curv-pattern (must match section 13)", base, others, pattern, y)
    rd_mean = float((val["plus"] - val["base"]).mean())
    if not (abs(val["base"].mean() - 0.8536) < 0.002 and abs(rd_mean - (-0.0037)) < 0.002):
        raise SystemExit(
            f"HARNESS VALIDATION FAILED: base {val['base'].mean():.4f} (want 0.8536), "
            f"raw d {rd_mean:+.4f} (want -0.0037). Battery aborted."
        )
    print("harness validation PASSED\n", flush=True)

    # ---- candidate A ----
    seeds_feats = []
    for seed in range(43, 49):
        p = glob.glob(f"models/divctl_resisc45_eps{seed}/*/multistep_train_feats_oneshot_g1.0.npz")
        assert len(p) == 1, p
        d = np.load(p[0])
        assert np.array_equal(d["subset_indices"], si), f"divctl eps{seed} pairing broke"
        seeds_feats.append(d["feats"])  # (n, 2, 3072)
    stack = np.stack(seeds_feats)  # (6, n, 2, 3072)
    spread = stack.std(axis=0).reshape(n, -1)  # (n, 6144)
    mean6 = stack.mean(axis=0).reshape(n, -1)  # diagnostic side-arm

    results = {"labels": y, "subset_indices": si}
    r = run_block("A: eps-spread std (t100+t580, 6 seeds)", base, others, spread, y)
    results["A_spread_base"] = r["base"]
    results["A_spread_plus"] = r["plus"]
    results["A_spread_null"] = r["null"]
    r2 = run_block("A-side: 6-seed MEAN block (diagnostic)", base, others, mean6, y)
    results["A_mean_plus"] = r2["plus"]
    results["A_mean_null"] = r2["null"]

    os.makedirs("results", exist_ok=True)
    np.savez("results/eps_spread_probe_resisc45.npz", **results,
             base_path=np.array(RESISC45_BASE),
             protocol=np.array("6w-A: std/mean over divctl eps43-48 at t100+t580; section-13 base; rule-3b"))
    print("per-image vectors cached to results/eps_spread_probe_resisc45.npz")


if __name__ == "__main__":
    main()
