"""Clean-latent (t=0) reference line, recovered exactly from the cached x_t arrays.

Audit finding W6/W9: the `inv state` column's flatness in t is near-tautological because z_t
is an invertible recoding of x0, so it should be drawn as an x0 REFERENCE LINE, not an arm.
This computes that line. No new extraction: raw_xt_baseline reuses ONE (lat, eps) pair across
all K timesteps, so x0 is exactly recoverable from any two cached timesteps by solving
    x_a = a*eps + (1-a)*x0
    x_b = b*eps + (1-b)*x0
(the adversarial review verified this reconstruction to <=2.5e-4 relative and recovers eps at
std 1.0005).

FINDINGS (2026-08-22, RESEARCH_NOTES 6e). Clean-latent x0 reference: EuroSAT 0.7397,
RESISC45 0.3145 -- the raw x_t arm at t=100 (0.7366/0.3159) is already AT the x0 ceiling, so
the entire raw decay is descent from clean-latent performance, as audit W6/W9 predicted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
CFG = {  # dataset -> (full cache, grid, frozen C from protocol_shape)
    "eurosat": ("models/raw_xt_n5000/eurosat_rawxt_ens1_full.npz", 1, 1000.0),
    "resisc45": ("models/raw_xt_n5000/resisc45_rawxt_ens1_full.npz", 4, 0.01),
}


def _fold(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
    return float((m.predict(sc.transform(x[va])) == y[va]).mean())


def main():
    for ds, (path, g, c) in CFG.items():
        d = np.load(path)
        f, y = d["feats"], d["labels"]
        ts = [int(t) for t in d["timesteps"]]
        a, b = ts[0] / 1000, ts[-1] / 1000
        xa, xb = f[:, 0, :], f[:, -1, :]
        # Solve the 2x2 linear system per element.
        x0 = (b * xa - a * xb) / (b - a)
        n, dim = x0.shape
        hw = int(round((dim // 16) ** 0.5))
        ch = dim // (hw * hw)
        t0 = torch.from_numpy(x0).reshape(n, ch, hw, hw)
        pooled = F.adaptive_avg_pool2d(t0, (g, g)).reshape(n, ch * g * g).numpy()
        jobs = [
            (tr, va)
            for s in SEEDS
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(pooled, y)
        ]
        accs = Parallel(n_jobs=5)(delayed(_fold)(pooled, y, tr, va, c) for tr, va in jobs)
        print(
            f"{ds}: clean-latent x0 reference ({g}x{g}, {pooled.shape[1]}d, C={c}): "
            f"{np.mean(accs):.4f} +- {np.std(accs):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
