"""Synthetic validation of the forward-vs-reversed direction control.

CPU-only, seconds (C=32 instead of 3072). Establishes two things before any GPU run:

  1. plumbing  — grouped folds, standardize, train, predict, diagnostics, CSV.
  2. semantics — on data where direction IS readable, the control returns PASS.
     If traj sits at chance even here, the encoder is order-blind and a run on the
     real cache would be uninterpretable.

Usage:  python tests/test_direction_control.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import traj_readout as T  # noqa: E402

T.EPOCHS = 60  # tiny model + tiny C -> converges fast
T.N_FOLDS = 3


def make_ordered(n: int = 200, k: int = 7, c: int = 32, seed: int = 0) -> np.ndarray:
    """Trajectories with a deterministic drift along the time axis.

    x[i,k] = content_i + k * drift. Reversing flips the sign of the k-gradient, so
    direction is readable — but only by something order-aware: the mean over k is
    identical forward and reversed, so mean-pool alone cannot do it.
    """
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(n, 1, c)).astype(np.float32)
    drift = rng.normal(size=(1, 1, c)).astype(np.float32)
    steps = np.arange(k, dtype=np.float32)[None, :, None]
    noise = 0.05 * rng.normal(size=(n, k, c))
    return (content + steps * drift + noise).astype(np.float32)


def test_dataset_builder() -> None:
    f = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)
    x, y, g = T.build_direction_dataset(f)
    assert x.shape == (6, 5, 2) and y.tolist() == [0, 1, 0, 1, 0, 1]
    assert g.tolist() == [0, 0, 1, 1, 2, 2]
    assert np.array_equal(x[0], f[0]) and np.array_equal(x[1], f[0][::-1])
    assert x.flags["C_CONTIGUOUS"], "negative stride must be materialized for torch"
    for i in range(3):  # both classes hold the same states -> content is label-free
        assert np.array_equal(np.sort(x[2 * i].ravel()), np.sort(x[2 * i + 1].ravel()))
    mean_feat = x.mean(axis=1)  # a permutation-invariant readout is exactly tied
    assert np.allclose(mean_feat[0::2], mean_feat[1::2])
    print("OK  dataset builder: reversal, grouping, content-balance, mean-pool tie")


def test_control_detects_ordering() -> None:
    x, y, groups = T.build_direction_dataset(make_ordered())
    print(f"\nsynthetic: {x.shape}  classes {np.bincount(y).tolist()}  chance 0.5000\n")

    dev = torch.device("cpu")
    by_arm = {}
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "synth_control.csv")
        for arm in ("traj", "shuffle"):
            print(f"--- arm={arm} ---")
            rows = T.run_control_arm(arm, x, y, groups, 2, 42, dev)
            by_arm[arm] = rows
            T.append_control_rows(out_csv, rows, "raw")
    T.report_control_verdict(by_arm)

    traj = float(np.mean([r["acc"] for r in by_arm["traj"]]))
    shuf = float(np.mean([r["acc"] for r in by_arm["shuffle"]]))
    print("\n" + "=" * 62)
    assert shuf < 0.62, f"shuffle should be near chance on direction, got {shuf:.4f}"
    print(f"OK  shuffle at chance ({shuf:.4f}) — permutation erases direction as designed")
    assert traj > 0.75, (
        f"traj = {traj:.4f} on TRIVIALLY ordered data — the encoder is order-blind, so a "
        "run on the real cache cannot be interpreted until this is fixed"
    )
    print(f"OK  traj reads ordering ({traj:.4f}) — control can return PASS")
    print("=" * 62)


if __name__ == "__main__":
    test_dataset_builder()
    test_control_detects_ordering()
