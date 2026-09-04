"""Path geometry of the inversion chain: do the INCREMENTS or the CURVATURE of the raw
latent trajectory carry linearly decodable class signal beyond the states themselves?

The wall metaphor, formalized: v(x,t) ~= E[eps - x0 | x_t] averages over crossing training
paths. For an ideally rectified flow every path is straight, so BENDS (second differences)
mark where neighbouring density pulls on the path -- the model's density map, externalized
into a probe-readable object. First differences are step directions (mostly the straight/
content part, velocity-adjacent -- section 3's null, but that was measured at a degenerate
pooling, R^2=0.99 with the pooled latent, so it gets one honest retry at 4x4).

Data: the raw inversion-state caches (n=500, full spatial resolution, fp32), 7 states at
t=100..580 on a 50-step grid. Cached states are ~4 integration steps apart, so the second
difference here is COARSE curvature at the cached scale, not the per-step solver curvature
(that is the separate pred_mid - pred experiment).

MAGNITUDE GATE (section 8 rule: measure the ablation/feature, do not assume). States were
computed in bf16 (rel eps ~2^-8 ~ 0.004). The gate prints ||d2z|| / ||z|| per t-pair; if
coarse curvature sits at the quantization floor the probe reads noise and a null is
uninterpretable.

Design: non-redundancy against the states, mirroring the velocity harness the audit
validated (S2): states_concat vs states+increments vs states+curvature, plus each block
alone. 4x4 pooling per state (256d x 7 = 1792d states; 6x256 incr; 5x256 curv), paired
folds, C in {0.01, 0.1, 1.0} reported at each so no operating-point games.
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
CS = [0.01, 0.1, 1.0]
CACHES = {
    "eurosat": "models/raw_xt/eurosat_invstate_n50_full.npz",
    "resisc45": "models/raw_xt/resisc45_invstate_n50_full.npz",
}


def pool4(f):
    n, k, dim = f.shape
    hw = int(round((dim // 16) ** 0.5)); c = dim // (hw * hw)
    x = torch.from_numpy(f).reshape(n, k, c, hw, hw)
    return F.adaptive_avg_pool2d(x.reshape(n * k, c, hw, hw), (4, 4)).reshape(n, k, c * 16).numpy()


def _fold(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
    return float((m.predict(sc.transform(x[va])) == y[va]).mean())


def accs(x, y, c, n_jobs=7):
    jobs = [(tr, va) for s in SEEDS
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(x, y)]
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va, c) for tr, va in jobs))


def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    for ds, path in CACHES.items():
        d = np.load(path)
        f, y = d["feats"].astype(np.float64), d["labels"]
        ts = [int(t) for t in d["timesteps"]]
        # magnitude gate on the UNPOOLED states
        z = f
        d1 = np.diff(z, axis=1)
        d2 = np.diff(z, axis=1, n=2)
        rms = lambda a: float(np.sqrt((a ** 2).mean()))
        print(f"\n=== {ds}  n={len(y)}  ts={ts} ===", flush=True)
        print(f"  gate: rms(z)={rms(z):.4f}  rms(d1)={rms(d1):.4f}  rms(d2)={rms(d2):.4f}  "
              f"d2/z={rms(d2)/rms(z):.4f}  (bf16 floor ~0.004)", flush=True)
        p = pool4(f)
        states = p.reshape(len(y), -1)
        incr = np.diff(p, axis=1).reshape(len(y), -1)
        curv = np.diff(p, axis=1, n=2).reshape(len(y), -1)
        chance = 1 / len(np.unique(y))
        for c in CS:
            a_states = accs(states, y, c)
            rows = [("states (7x256)", a_states, None),
                    ("incr alone (6x256)", accs(incr, y, c), None),
                    ("curv alone (5x256)", accs(curv, y, c), None),
                    ("states+incr", accs(np.hstack([states, incr]), y, c), a_states),
                    ("states+curv", accs(np.hstack([states, curv]), y, c), a_states)]
            print(f"  C={c}  (chance {chance:.4f})", flush=True)
            for name, a, base in rows:
                if base is None:
                    print(f"    {name:<22} {a.mean():.4f}", flush=True)
                else:
                    dd = a - base
                    lo, hi = ci(dd)
                    sig = "SIGNIF" if lo > 0 or hi < 0 else ""
                    print(f"    {name:<22} {a.mean():.4f}  delta {dd.mean():+.4f} "
                          f"[{lo:+.4f}, {hi:+.4f}] {sig}", flush=True)


if __name__ == "__main__":
    main()
