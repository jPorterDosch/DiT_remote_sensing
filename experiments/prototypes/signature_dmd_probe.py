"""6y candidates E (path signatures / Levy areas) and F (Koopman/DMD spectra), RESISC45.

Zero GPU: both blocks are computed from solver_curvature_n5000's cached chain states
(5000, 7, 256, 64) and evaluated against the section-13 base via the shared harness.
See RESEARCH_NOTES 6y for the exact pre-registered block definitions.

E block (2080 d): Levy areas A_ij over the time-augmented pooled path (65 coords: nominal
t/1000 + 64 channel means), i<j. Level-1 and symmetric level-2 excluded by design.
F block (134 d): per-image DMD operator A from 6 transitions x 256 token pairs;
sorted |eig| (64) + sorted Re(eig) (64) + per-transition residual norms (6).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _absorption_harness import load_base, run_block  # noqa: E402

CURV = "models/solver_curvature_n5000/resisc45_solvercurv_n50.npz"


def levy_areas(paths: np.ndarray) -> np.ndarray:
    """paths: (N, K, D). Levy area A_ij = 1/2 sum_k (p_i dp_j - p_j dp_i), i<j -> (N, D*(D-1)/2)."""
    n, k, d = paths.shape
    dp = np.diff(paths, axis=1)  # (N, K-1, D)
    pm = 0.5 * (paths[:, :-1, :] + paths[:, 1:, :])  # midpoint rule, (N, K-1, D)
    # M_ij = sum_k pm_i * dp_j  (N, D, D); Levy = (M - M^T)/2, upper triangle
    M = np.einsum("nki,nkj->nij", pm, dp)
    A = 0.5 * (M - np.transpose(M, (0, 2, 1)))
    iu = np.triu_indices(d, k=1)
    return A[:, iu[0], iu[1]].astype(np.float32)


def dmd_block(states: np.ndarray) -> np.ndarray:
    """states: (N, K, L, D). Per image: A = argmin ||Z' - A Z|| over the K-1 transitions
    stacked across L tokens; block = sorted |eig|, sorted Re(eig), per-transition residuals."""
    n, k, tokens, d = states.shape
    out = np.zeros((n, 2 * d + (k - 1)), dtype=np.float32)
    for i in range(n):
        Z = states[i, :-1].reshape(-1, d)  # ((K-1)*L, D)
        Zp = states[i, 1:].reshape(-1, d)
        A, *_ = np.linalg.lstsq(Z, Zp, rcond=None)  # maps row-vectors: z' ~= z @ A
        eig = np.linalg.eigvals(A.T)
        out[i, :d] = np.sort(np.abs(eig))[::-1]
        out[i, d : 2 * d] = np.sort(eig.real)[::-1]
        R = states[i, 1:] - np.einsum("klj,ji->kli", states[i, :-1], A)
        out[i, 2 * d :] = np.linalg.norm(R.reshape(k - 1, -1), axis=1)
        if i % 500 == 0:
            print(f"  dmd {i}/{n}", flush=True)
    return out


def main() -> None:
    base, others, y, si = load_base()
    dc = np.load(CURV)
    assert np.array_equal(dc["subset_indices"], si), "curvature cache pairing broke"
    states = dc["states"].astype(np.float64)  # (5000, 7, 256, 64)
    t_nom = np.array(dc["state_timesteps"], dtype=np.float64) / 1000.0

    # E: time-augmented pooled path -> Levy areas
    pooled = states.mean(axis=2)  # (N, 7, 64)
    # standardize channels across the whole set so no channel's scale dominates the areas
    mu, sd = pooled.mean((0, 1), keepdims=True), pooled.std((0, 1), keepdims=True) + 1e-8
    paths = np.concatenate(
        [np.broadcast_to(t_nom[None, :, None], (len(y), 7, 1)), (pooled - mu) / sd], axis=2
    )  # (N, 7, 65)
    E = levy_areas(paths)
    print("E block:", E.shape, "finite:", bool(np.isfinite(E).all()))

    # F: DMD spectra on per-token dynamics (channel-standardized the same way)
    F = dmd_block(((states - mu[:, :, None, :]) / sd[:, :, None, :]))
    print("F block:", F.shape, "finite:", bool(np.isfinite(F).all()))

    results: dict = {"labels": y, "subset_indices": si, "E_block": E, "F_block": F}
    rE = run_block("E: Levy areas (time-aug pooled path)", base, others, E, y)
    rF = run_block("F: DMD/Koopman spectra (token dynamics)", base, others, F, y)
    for tag, r in (("E", rE), ("F", rF)):
        results[f"{tag}_base"] = r["base"]
        results[f"{tag}_plus"] = r["plus"]
        results[f"{tag}_null"] = r["null"]

    os.makedirs("results", exist_ok=True)
    np.savez("results/signature_dmd_probe_resisc45.npz", **results)
    print("per-image vectors cached to results/signature_dmd_probe_resisc45.npz")


if __name__ == "__main__":
    main()
