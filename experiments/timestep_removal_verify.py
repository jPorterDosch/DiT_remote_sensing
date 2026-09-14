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


SHRINK_ALPHA = 0.05  # scale-relative shrinkage toward (trace/d)*I; the old absolute 1e-4
# ridge regularized nothing against eigenvalues spanning 6 decades and injected a held-out
# covariate shift the gate could read as removed-t (6o-D). Matches timestep_removal.py.


def zca(z, alpha=SHRINK_ALPHA):
    mu = z.mean(0)
    cov = np.cov((z - mu).T)
    d = cov.shape[0]
    cov = (1 - alpha) * cov + alpha * (np.trace(cov) / d) * np.eye(d)
    w, V = np.linalg.eigh(cov)
    return mu, V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T


def probes():
    return {
        "linear": LogisticRegression(C=0.01, max_iter=2000),
        "mlp256": MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=0),
        "mlp1024x2": MLPClassifier(hidden_layer_sizes=(1024, 512), max_iter=400, random_state=0),
        "gbdt": HistGradientBoostingClassifier(max_iter=300, random_state=0),
    }


def run(xtr, ytr, xva, yva, chance, groups_va=None):
    """Fit the four probe families; report each with an image-clustered z, plus the MAX
    with its two corrections.

    Two fixes over v1 (6o-E, 6n-A):
      (a) v1 z-scored max(4 probes) against a single-test binomial SE with no correction --
          on a truly null substrate the max of 4 probes on ~1050 vectors sits 1.5-2 SE
          above chance, so v1 printed z ~ +2 for pure noise. The max is now selected on a
          SELECTION HALF of val and z-scored on the disjoint EVALUATION half (selection is
          then honest), and per-probe z's are printed so no max is needed at all.
      (b) rows derived from the same image are not independent (7 vectors, 42 pairs per
          image). SEs now cluster by image: z = mean / SE(per-image mean correctness).
    """
    sc = StandardScaler().fit(xtr)
    a, b = sc.transform(xtr), sc.transform(xva)
    if groups_va is None:
        groups_va = np.arange(len(yva))
    uniq = np.unique(groups_va)
    rng = np.random.default_rng(0)
    sel_groups = rng.choice(uniq, size=len(uniq) // 2, replace=False)
    sel_mask = np.isin(groups_va, sel_groups)

    def cluster_z(correct_vec, mask):
        g = groups_va[mask]
        per_img = np.array([correct_vec[mask][g == u].mean() for u in np.unique(g)])
        se = per_img.std(ddof=1) / len(per_img) ** 0.5
        if se == 0:
            return float("inf") if per_img.mean() > chance else 0.0  # perfect separation
        return (per_img.mean() - chance) / se

    accs, out = {}, []
    correct_by_probe = {}
    for name, m in probes().items():
        m.fit(a, ytr)
        correct = (m.predict(b) == yva).astype(float)
        correct_by_probe[name] = correct
        accs[name] = correct.mean()
        out.append(f"{name} {accs[name]:.4f} z={cluster_z(correct, np.ones(len(yva), bool)):+.1f}")
    # honest max: choose on the selection half, score on the evaluation half
    pick = max(accs, key=lambda nm: correct_by_probe[nm][sel_mask].mean())
    ev = ~sel_mask
    best_eval = correct_by_probe[pick][ev].mean()
    z = cluster_z(correct_by_probe[pick], ev)
    return best_eval, z, f"sel->{pick}; " + "  ".join(out)


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
        gva = np.repeat(np.arange(len(b)), k)  # cluster rows by source image (6n-A)
        best, z, det = run(xtr, ytr_, xva, yva_, 1 / k, groups_va=gva)
        print(f"  {name:<26} best(eval half) {best:.4f}  z_img={z:+5.1f}   [{det}]", flush=True)

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
        # 42 ordered pairs per image are functions of the same 7 vectors: independent n is
        # the IMAGE count, not the pair count (6n-A -- v1's binomial-n=6300 z's were
        # inflated up to sqrt(42)~6.5x).
        gva = np.tile(np.arange(len(b)), len(pairs))
        best, z, det = run(xtr, ytr_, xva, yva_, 0.5, groups_va=gva)
        print(f"  {name:<26} best(eval half) {best:.4f}  z_img={z:+5.1f}   [{det}]", flush=True)


if __name__ == "__main__":
    main()
