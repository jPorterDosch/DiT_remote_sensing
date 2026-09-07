"""Find a substrate on which the traj-vs-shuffle ablation is ADMISSIBLE.

THE PROBLEM (section 1 CORRECTION). `per_sample_time_shuffle` permutes slots but preserves
the multiset of per-timestep vectors, and timestep is 99.5% decodable from a single vector
(87% even after per-timestep z-scoring, with an MLP). A content-attending transformer re-sorts
the tokens internally, so the ablation removes nothing and traj~=shuffle is uninformative.

THE FIX MUST BE MEASURED, NOT ASSUMED (section 8 rule). A transform T is an admissible
substrate only if BOTH gates pass:
  GATE 1  timestep is unrecoverable from T(x) by a probe at least as expressive as the
          readout (transformer readout => MLP probe), on held-out data, split by image.
  GATE 2  class signal SURVIVES T. A transform that destroys the label too makes the
          subsequent comparison vacuous.

Mechanisms tried, cheapest first:
  z      per-timestep z-score            (known to fail gate 1: MLP ~0.87)
  white  per-timestep ZCA whitening in a shared PCA basis -- removes per-t covariance,
         not just mean/scale
  inlp   iterative nullspace projection against a linear t-classifier (Ravfogel et al.
         style): fit t-classifier, project features onto the null space of its weights,
         repeat until linear t-accuracy hits chance
  both   white -> inlp

All transforms are fit on the TRAIN fold only and applied to val (no leakage).

FINDINGS (2026-09-01, RESEARCH_NOTES 1). Only per-timestep ZCA whitening passes the
nonlinear gate (MLP t-ID 0.176 vs chance 0.143); z-scoring leaves 0.646, INLP 0.739 --
removing per-t COVARIANCE is what matters, not means/scales/linear discriminants. Class
signal survives at 0.873 (from 0.940). See timestep_removal_verify.py for why even this
substrate fails the test that matters.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CACHES = {
    "eurosat": "models/paired_500_eurosat_inv_redo/eurosat_flux_5cad8aad+42/multistep_train_feats_inversion_g1.0_n50.npz",
}
PCA_DIM = 256
INLP_ROUNDS = 12


def fit_shared_pca(tr):  # (N, K, C) -> PCA on all timesteps pooled
    n, k, c = tr.shape
    p = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
    p.fit(tr.reshape(n * k, c))
    return p


def apply_pca(p, x):
    n, k, c = x.shape
    return p.transform(x.reshape(n * k, c)).reshape(n, k, -1)


def fit_white(tr):  # per-timestep ZCA in the shared basis
    stats = []
    for i in range(tr.shape[1]):
        z = tr[:, i, :]
        mu = z.mean(0)
        cov = np.cov((z - mu).T) + 1e-4 * np.eye(z.shape[1])
        w, V = np.linalg.eigh(cov)
        W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ V.T
        stats.append((mu, W))
    return stats


def apply_white(stats, x):
    return np.stack([(x[:, i, :] - mu) @ W for i, (mu, W) in enumerate(stats)], axis=1)


def fit_inlp(tr_flat, t_lab, rounds=INLP_ROUNDS):
    """Return a projection P removing the linear t-discriminant subspace."""
    d = tr_flat.shape[1]
    P = np.eye(d)
    x = tr_flat.copy()
    for _ in range(rounds):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(x, t_lab)
        W = clf.coef_  # (7, d)
        Q, _ = np.linalg.qr(W.T)  # orthonormal basis of the subspace
        Pi = np.eye(d) - Q @ Q.T
        P = P @ Pi
        x = x @ Pi
    return P


def t_recoverable(tr, va, probe):
    """7-way timestep identification from a single vector, held out, split by image."""
    n_tr, k, _ = tr.shape
    xtr, ytr = tr.reshape(-1, tr.shape[2]), np.tile(np.arange(k), n_tr)
    xva, yva = va.reshape(-1, va.shape[2]), np.tile(np.arange(k), va.shape[0])
    sc = StandardScaler().fit(xtr)
    a, b = sc.transform(xtr), sc.transform(xva)
    m = (
        LogisticRegression(C=0.01, max_iter=2000)
        if probe == "linear"
        else MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=0)
    )
    m.fit(a, ytr)
    return float((m.predict(b) == yva).mean())


def class_acc(tr, va, ytr, yva):
    """Concat-over-timesteps linear probe -- does class signal survive the transform?"""
    a, b = tr.reshape(len(tr), -1), va.reshape(len(va), -1)
    sc = StandardScaler().fit(a)
    m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(a), ytr)
    return float((m.predict(sc.transform(b)) == yva).mean())


def main():
    for ds, path in CACHES.items():
        d = np.load(path)
        f, y = d["feats"].astype(np.float64), d["labels"]
        k = f.shape[1]
        itr, iva = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
        chance_t, chance_y = 1 / k, 1 / len(np.unique(y))
        print(
            f"\n=== {ds}  n={len(y)}  K={k}  chance(t)={chance_t:.4f} chance(class)={chance_y:.4f} ===",
            flush=True,
        )
        print(f"{'substrate':<26}{'t-ID linear':>12}{'t-ID MLP':>10}{'class acc':>11}  gates", flush=True)

        pca = fit_shared_pca(f[itr])
        base_tr, base_va = apply_pca(pca, f[itr]), apply_pca(pca, f[iva])
        variants = {"raw (PCA only)": (base_tr, base_va)}

        # z-score
        mu = base_tr.mean(0, keepdims=True)
        sd = base_tr.std(0, keepdims=True) + 1e-8
        variants["z per-timestep"] = ((base_tr - mu) / sd, (base_va - mu) / sd)

        # whitening
        st = fit_white(base_tr)
        wtr, wva = apply_white(st, base_tr), apply_white(st, base_va)
        variants["ZCA white per-timestep"] = (wtr, wva)

        # inlp on raw, and on whitened
        for name, (a, b) in [("INLP", (base_tr, base_va)), ("white+INLP", (wtr, wva))]:
            flat = a.reshape(-1, a.shape[2])
            tl = np.tile(np.arange(k), a.shape[0])
            P = fit_inlp(flat, tl)
            variants[name] = (a @ P, b @ P)

        for name, (a, b) in variants.items():
            lin = t_recoverable(a, b, "linear")
            mlp = t_recoverable(a, b, "mlp")
            ca = class_acc(a, b, y[itr], y[iva])
            g1 = mlp < chance_t * 1.5
            print(
                f"{name:<26}{lin:>12.4f}{mlp:>10.4f}{ca:>11.4f}  {'GATE1 PASS' if g1 else 'gate1 fail'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
