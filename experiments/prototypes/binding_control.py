"""The ordering question, asked in a form that IS answerable.

WHY NOT PERMUTATION. If timestep is decodable from a single feature vector (it is, ~99.5%),
then permuting slots does not remove ordering -- the readout re-sorts by content. No amount of
shuffling fixes that; the information is genuinely still present. Permutation can only test
ordering on a substrate where timestep is unrecoverable, which is what timestep_removal.py
searches for and may not find.

THE ANSWERABLE QUESTION. Does the trajectory carry class information in its JOINT structure --
the fact that all K states describe the SAME image -- beyond what the per-timestep marginals
carry? That is testable by an ablation that cannot be undone:

  real        [f_1(x_i), f_2(x_i), ..., f_K(x_i)]      one image, all timesteps
  mismatched  [f_1(x_a), f_2(x_b), ..., f_K(x_g)]      K DIFFERENT images OF THE SAME CLASS

The mismatched arm preserves, exactly: every per-timestep marginal, the timestep of every
slot, the class of every slot, and the dimensionality. It destroys only the cross-timestep
BINDING to one image. Nothing in the surviving features identifies which images were combined,
so the readout cannot undo it -- the failure mode that killed the shuffle ablation cannot recur.

READING THE RESULT, both directions stated in advance:
  real >> mismatched  binding carries class signal beyond the marginals -- the trajectory is
                      more than a bag of independent views. POSITIVE trajectory result.
  real ~= mismatched  the trajectory is exactly its marginals; no joint structure is used.
  real << mismatched  binding HURTS: K views of a class beat K views of an instance, i.e. the
                      mismatched arm is doing within-class ensembling. Consistent with the
                      diversity-matched finding (7 draws > 7 timesteps).

Mismatch is drawn WITHOUT replacement within class and re-drawn per seed; labels are unchanged.

RETRACTED (2026-09-01, RESEARCH_NOTES 1). The mismatched arm is constructed USING the class
labels (same-class draws per slot), so its near-perfect accuracy (0.99+) is label leakage by
design, not a finding. Kept for the retraction record; do not cite its numbers.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
C = 0.1
CACHES = {
    "eurosat_inv": "models/n5000_eurosat_inversion/eurosat_flux_f4ee81c8+42/multistep_train_feats_inversion_g1.0_n50.npz",
    "resisc45_inv": "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
}


def mismatch(f, y, rng):
    """Slot k of sample i <- slot k of a random same-class sample. Slot 0 keeps the original
    image so the arm is anchored (otherwise every sample is a different image entirely)."""
    out = f.copy()
    by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    for k in range(1, f.shape[1]):
        src = np.empty(len(y), dtype=int)
        for c, idx in by_class.items():
            src[idx] = rng.permutation(idx)
        out[:, k, :] = f[src, k, :]
    return out


def _fold(x, y, tr, va):
    sc = StandardScaler().fit(x[tr])
    a, b = sc.transform(x[tr]), sc.transform(x[va])
    if a.shape[1] > 512:
        p = PCA(n_components=512, svd_solver="randomized", random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
    return va, m.predict(b)


def correct(x, y, n_jobs=7):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(x, y))
        for va, p in Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va) for tr, va in jobs):
            out[va] += (p == y[va]).astype(float)
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    for name, path in CACHES.items():
        d = np.load(path)
        f, y = d["feats"], d["labels"]
        rng = np.random.default_rng(0)
        real = correct(f.reshape(len(y), -1), y)
        mis = correct(mismatch(f, y, rng).reshape(len(y), -1), y)
        diff = real - mis
        lo, hi = ci(diff)
        sig = "SIGNIF" if lo > 0 or hi < 0 else "ns"
        print(f"\n{name}  n={len(y)}  K={f.shape[1]}", flush=True)
        print(f"  real (bound)      {real.mean():.4f}", flush=True)
        print(f"  mismatched (same-class, per-slot)  {mis.mean():.4f}", flush=True)
        print(f"  real - mismatched {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}", flush=True)


if __name__ == "__main__":
    main()
