"""Is the random-weight dissociation driven by CLASS STRUCTURE?

Section 6d eliminated resolution: degrading RESISC45 to EuroSAT's effective bandwidth moved
the untrained-network fraction ~2 points (57% -> 55%), nowhere near EuroSAT's 84%. The leading
remaining candidate is class structure: EuroSAT is 10 coarse land-cover classes, RESISC45 is
45 fine-grained scene classes.

TEST. Restrict RESISC45 to its 10 most-separable classes (per-class recall of a linear probe
on the TRAINED features at t=100, seed-0 CV) and recompute the untrained/trained above-chance
fraction on that 10-class subproblem, chance 0.1 -- matching EuroSAT's class count and chance.
If the fraction jumps toward 84%, class granularity drives the dissociation. If it stays near
57%, neither resolution nor class count explains it and the content/domain explanation is all
that is left standing.

Both a top-10 (easiest) and a random-10 (median) restriction are reported, because "most
separable" biases toward classes that random features may also find easy -- the random-10
figure guards that read.

FINDINGS (2026-08-22, RESEARCH_NOTES 6e). Restricting RESISC45 to its 10 most-separable
classes reproduces EuroSAT's untrained fraction (87.9% vs 84.4%); random-10 moves it only to
61.7%. The random-weight dissociation is CLASS DIFFICULTY -- not class count, not resolution
(6d), not domain.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

TRAINED = (
    "models/n5000_resisc45_oneshot_ens1/resisc45_flux_19044381+42/multistep_train_feats_oneshot_g1.0.npz"
)
UNTRAINED = "models/n5000_resisc45_randinit_ens1/resisc45_flux_19044381+42+randinit/multistep_train_feats_oneshot_g1.0_RANDINIT.npz"
T_IDX = 0  # t=100
C_BY_ARM = {"trained": 0.1, "untrained": 10.0}  # each arm's frozen C from arm_shape


def fold_preds(x, y, c, seed=0):
    """Out-of-fold predictions, seed-0 5-fold."""
    pred = np.empty_like(y)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        sc = StandardScaler().fit(x[tr])
        m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
        pred[va] = m.predict(sc.transform(x[va]))
    return pred


def acc_on(x, y, keep, c):
    mask = np.isin(y, keep)
    xs, ys = x[mask], y[mask]
    remap = {k: i for i, k in enumerate(sorted(keep))}
    ys = np.array([remap[v] for v in ys])
    return float((fold_preds(xs, ys, c) == ys).mean())


def main():
    dt = np.load(TRAINED)
    du = np.load(UNTRAINED)
    if not (np.array_equal(dt["labels"], du["labels"])):
        raise RuntimeError("arms not paired")
    y = dt["labels"]
    xt, xu = dt["feats"][:, T_IDX, :], du["feats"][:, T_IDX, :]

    # Per-class recall of the trained probe -> class separability ranking.
    pred = fold_preds(xt, y, C_BY_ARM["trained"])
    classes = np.unique(y)
    recall = {c: float((pred[y == c] == c).mean()) for c in classes}
    order = sorted(classes, key=lambda c: -recall[c])
    top10 = order[:10]
    rng = np.random.default_rng(0)
    rand10 = sorted(rng.choice(classes, 10, replace=False))
    print("top-10 classes by trained recall:", [int(c) for c in top10], flush=True)
    print("random-10 classes:", [int(c) for c in rand10], flush=True)

    print(f"\n{'restriction':<22}{'chance':>8}{'trained':>10}{'untrained':>11}{'untr frac':>11}", flush=True)
    for name, keep, ch in [
        ("all 45", classes, 1 / 45),
        ("top-10 separable", top10, 0.1),
        ("random-10", rand10, 0.1),
    ]:
        at = acc_on(xt, y, list(keep), C_BY_ARM["trained"])
        au = acc_on(xu, y, list(keep), C_BY_ARM["untrained"])
        frac = (au - ch) / (at - ch)
        print(f"{name:<22}{ch:>8.4f}{at:>10.4f}{au:>11.4f}{frac:>10.1%}", flush=True)
    print("\nEuroSAT reference: untrained fraction 84.4% (10 classes, chance 0.1)", flush=True)


if __name__ == "__main__":
    main()
