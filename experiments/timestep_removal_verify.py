"""Does whitening TRULY remove timestep information? Stronger, and testing the right thing.

TWO PROBLEMS with the original gate (7-way MLP-256, 0.176 vs chance 0.143):
  (a) UNDERPOWERED CLAIM. With ~1050 held-out vectors the binomial SE is ~0.011, so 0.176 sits
      ~3 SE ABOVE chance -- residually recoverable, not removed.
  (b) WRONG CAPABILITY. The shuffle is defeated by SORTING, and sorting needs only PAIRWISE
      judgements ("this vector is noisier than that one"), never absolute identification.
      A substrate can be at chance on 7-way ID and still be perfectly sortable.

So this measures both, at several probe capacities, split by image:
  ABSOLUTE   7-way timestep ID from one vector           (the original gate, + binomial CI)
  PAIRWISE   given two vectors OF THE SAME IMAGE, which came first?   <- the capability that
             actually defeats per_sample_time_shuffle. Chance 0.5.
Both on raw, z-scored, global-whitened and per-timestep-whitened substrates.

FINDINGS (2026-09-01, RESEARCH_NOTES 1). The admissible-substrate window is EMPTY: even
per-t whitened, PAIRWISE order ("which of two same-image states came first") stays 73.2%
recoverable (z=+37) -- and sorting needs only pairwise judgements, so the shuffle remains
undone. On raw features pairwise is 1.000 over 6,300 held-out pairs (delta <= 5e-4), which
is the measured premise of the section-1 dissolution proof: ordering carries provably zero
information beyond the set.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PCA_DIM = 256
CACHE = (
    "models/paired_500_eurosat_inv_redo/eurosat_flux_5cad8aad+42/multistep_train_feats_inversion_g1.0_n50.npz"
)


def zca(z, ridge=1e-4):
    mu = z.mean(0)
    cov = np.cov((z - mu).T) + ridge * np.eye(z.shape[1])
    w, V = np.linalg.eigh(cov)
    return mu, V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T


def probes():
    return {
        "linear": LogisticRegression(C=0.01, max_iter=2000),
        "mlp256": MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=0),
        "mlp1024x2": MLPClassifier(hidden_layer_sizes=(1024, 512), max_iter=400, random_state=0),
        "gbdt": HistGradientBoostingClassifier(max_iter=300, random_state=0),
    }


def run(xtr, ytr, xva, yva, chance):
    sc = StandardScaler().fit(xtr)
    a, b = sc.transform(xtr), sc.transform(xva)
    best, out = 0.0, []
    for name, m in probes().items():
        m.fit(a, ytr)
        acc = float((m.predict(b) == yva).mean())
        best = max(best, acc)
        out.append(f"{name} {acc:.4f}")
    se = (chance * (1 - chance) / len(yva)) ** 0.5
    z = (best - chance) / se
    return best, z, "  ".join(out)


def main():
    d = np.load(CACHE)
    f, y = d["feats"].astype(np.float64), d["labels"]
    k = f.shape[1]
    itr, iva = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    p = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
    p.fit(f[itr].reshape(-1, f.shape[2]))
    tr = p.transform(f[itr].reshape(-1, f.shape[2])).reshape(len(itr), k, -1)
    va = p.transform(f[iva].reshape(-1, f.shape[2])).reshape(len(iva), k, -1)

    subs = {"raw (PCA only)": (tr, va)}
    mu = tr.mean(0, keepdims=True)
    sd = tr.std(0, keepdims=True) + 1e-8
    subs["z per-timestep"] = ((tr - mu) / sd, (va - mu) / sd)
    m_, W_ = zca(tr.reshape(-1, tr.shape[2]))
    subs["global whitening"] = ((tr - m_) @ W_, (va - m_) @ W_)
    st = [zca(tr[:, i, :]) for i in range(k)]
    subs["per-timestep whitening"] = (
        np.stack([(tr[:, i, :] - a) @ b for i, (a, b) in enumerate(st)], 1),
        np.stack([(va[:, i, :] - a) @ b for i, (a, b) in enumerate(st)], 1),
    )

    print(f"n={len(y)} K={k}   ABSOLUTE 7-way t-ID (chance {1 / k:.4f})", flush=True)
    for name, (a, b) in subs.items():
        xtr, ytr_ = a.reshape(-1, a.shape[2]), np.tile(np.arange(k), len(a))
        xva, yva_ = b.reshape(-1, b.shape[2]), np.tile(np.arange(k), len(b))
        best, z, det = run(xtr, ytr_, xva, yva_, 1 / k)
        print(f"  {name:<26} best {best:.4f}  z={z:+5.1f}   [{det}]", flush=True)

    print(
        "\nPAIRWISE ordering: same image, two timesteps -- which came first? "
        "(chance 0.5)  <- the capability that defeats the shuffle",
        flush=True,
    )
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    for name, (a, b) in subs.items():

        def build(arr):
            X, Y = [], []
            for i, j in pairs:
                X.append(np.hstack([arr[:, i, :], arr[:, j, :]]))
                Y.append(np.full(len(arr), int(i < j)))
            return np.vstack(X), np.concatenate(Y)

        xtr, ytr_ = build(a)
        xva, yva_ = build(b)
        best, z, det = run(xtr, ytr_, xva, yva_, 0.5)
        print(f"  {name:<26} best {best:.4f}  z={z:+5.1f}   [{det}]", flush=True)


if __name__ == "__main__":
    main()
