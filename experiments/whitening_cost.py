"""Does per-timestep whitening destroy SEMANTIC information, or only change conditioning?

The concern: per-t whitening equalizes the timesteps -- exactly the systematic cross-timestep
structure an ordering effect might live in. If it removes class signal along with timestep
identity, the traj-vs-shuffle test on that substrate is vacuous in a new way.

THE DECOMPOSITION. Compare three transforms that differ in ONE respect:
  raw          no transform                              t recoverable, no conditioning change
  GLOBAL white one whitening fit on ALL timesteps pooled  t STILL recoverable, same KIND of
                                                          conditioning change
  PER-T white  a separate whitening per timestep          t removed, same kind of change
Global whitening is the control that isolates the cost of WHITENING from the cost of
REMOVING TIMESTEP. If class accuracy under global ~= per-t, the drop is the whitening
operation (conditioning); if global ~= raw and per-t is lower, then equalizing the timesteps
specifically cost class signal -- and the substrate is suspect.

Also reported per timestep, so a loss concentrated at high or low t (which would bias an
ordering test) is visible rather than hidden in an average, and with a NONLINEAR probe, since
a purely-conditioning loss should shrink when the probe is not L2-regularized in the whitened
coordinates.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PCA_DIM = 256
CACHE = "models/paired_500_eurosat_inv_redo/eurosat_flux_5cad8aad+42/multistep_train_feats_inversion_g1.0_n50.npz"


def zca(z, ridge=1e-4):
    mu = z.mean(0)
    cov = np.cov((z - mu).T) + ridge * np.eye(z.shape[1])
    w, V = np.linalg.eigh(cov)
    return mu, V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T


def acc(a, b, ytr, yva, probe="linear"):
    sc = StandardScaler().fit(a)
    A, B = sc.transform(a), sc.transform(b)
    m = (LogisticRegression(C=0.1, max_iter=2000) if probe == "linear"
         else MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=0))
    m.fit(A, ytr)
    return float((m.predict(B) == yva).mean())


def t_id(tr, va):
    n, k, _ = tr.shape
    xtr, ytr = tr.reshape(-1, tr.shape[2]), np.tile(np.arange(k), n)
    xva, yva = va.reshape(-1, va.shape[2]), np.tile(np.arange(k), va.shape[0])
    sc = StandardScaler().fit(xtr)
    m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=0)
    m.fit(sc.transform(xtr), ytr)
    return float((m.predict(sc.transform(xva)) == yva).mean())


def main():
    d = np.load(CACHE)
    f, y = d["feats"].astype(np.float64), d["labels"]
    ts = [int(t) for t in d["timesteps"]]
    itr, iva = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    k = f.shape[1]
    p = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
    p.fit(f[itr].reshape(-1, f.shape[2]))
    tr = p.transform(f[itr].reshape(-1, f.shape[2])).reshape(len(itr), k, -1)
    va = p.transform(f[iva].reshape(-1, f.shape[2])).reshape(len(iva), k, -1)

    variants = {"raw (PCA only)": (tr, va)}
    mu, W = zca(tr.reshape(-1, tr.shape[2]))                     # GLOBAL
    variants["global whitening"] = ((tr - mu) @ W, (va - mu) @ W)
    st = [zca(tr[:, i, :]) for i in range(k)]                     # PER-TIMESTEP
    variants["per-timestep whitening"] = (
        np.stack([(tr[:, i, :] - m_) @ W_ for i, (m_, W_) in enumerate(st)], 1),
        np.stack([(va[:, i, :] - m_) @ W_ for i, (m_, W_) in enumerate(st)], 1))

    print(f"n={len(y)} K={k} chance={1/len(np.unique(y)):.3f}\n")
    print(f"{'substrate':<26}{'t-ID MLP':>10}{'class lin':>11}{'class MLP':>11}", flush=True)
    for name, (a, b) in variants.items():
        A, B = a.reshape(len(a), -1), b.reshape(len(b), -1)
        print(f"{name:<26}{t_id(a, b):>10.4f}{acc(A, B, y[itr], y[iva]):>11.4f}"
              f"{acc(A, B, y[itr], y[iva], 'mlp'):>11.4f}", flush=True)

    print(f"\nper-timestep class accuracy (linear), is the loss concentrated?", flush=True)
    print(f"  {'t':>6}" + "".join(f"{n[:14]:>16}" for n in variants), flush=True)
    for i, t in enumerate(ts):
        row = "".join(f"{acc(a[:, i, :], b[:, i, :], y[itr], y[iva]):>16.4f}"
                      for a, b in variants.values())
        print(f"  {t:>6}{row}", flush=True)


if __name__ == "__main__":
    main()
