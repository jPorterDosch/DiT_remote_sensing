"""Section-13 follow-up: re-check the absorption result (section 12) on the STRONGEST base
-- best-t block-28 protected by PCA + the other six timesteps appended raw -- since the
original used the 42x-compressed concat base."""

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

C = 0.1
SEEDS = [0, 1, 2]
DISCARD = [154, 1446]
CFG = {
    "resisc45": (
        "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
        "models/solver_curvature_n5000/resisc45_solvercurv_n50.npz",
        1,
    ),
    "eurosat": (
        "models/n5000_eurosat_inversion/eurosat_flux_f4ee81c8+42/multistep_train_feats_inversion_g1.0_n50.npz",
        "models/solver_curvature_n5000/eurosat_solvercurv_n50.npz",
        1,
    ),
}


def ditf(f, m):
    x = f.astype(np.float64).copy()
    x[:, :, DISCARD] = 0.0
    mu = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    x = (x - mu) / np.sqrt(v + 1e-6)
    x = (1 + m[None, :, 1, :]) * x + m[None, :, 0, :]
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def pool(x, g=2):
    n, k, t, d = x.shape
    hw = int(round(t**0.5))
    sp = torch.from_numpy(x).reshape(n * k, hw, hw, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(sp, (g, g)).reshape(n, k * d * g * g).numpy()


def _fold(base, extra, blk, y, tr, va):
    sc = StandardScaler().fit(base[tr])
    a, b = sc.transform(base[tr]), sc.transform(base[va])
    k = min(512, a.shape[1], len(tr) - 1)
    if k < a.shape[1]:
        p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    for e in (extra, blk):
        if e is not None:
            s2 = StandardScaler().fit(e[tr])
            a = np.hstack([a, s2.transform(e[tr])])
            b = np.hstack([b, s2.transform(e[va])])
    m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
    return va, m.predict(b)


def correct(base, extra, blk, y):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(base, y))
        for va, pred in Parallel(n_jobs=7)(delayed(_fold)(base, extra, blk, y, tr, va) for tr, va in jobs):
            out[va] += pred == y[va]
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


for ds, (fp, cp, tb) in CFG.items():
    df, dc = np.load(fp), np.load(cp)
    assert np.array_equal(df["subset_indices"], dc["subset_indices"])
    y = df["labels"]
    n = len(y)
    feats = ditf(df["feats"], df["mods"])
    base = feats[:, tb, :]
    others = np.concatenate([feats[:, i, :] for i in range(feats.shape[1]) if i != tb], axis=1)
    raw = dc["pred_mid"] - dc["pred"]
    nr = np.linalg.norm(raw.reshape(n, raw.shape[1], -1), axis=2)
    pattern = pool(raw / (nr[:, :, None, None] + 1e-8))
    rng = np.random.default_rng(0)
    b = correct(base, others, None, y)
    pa = correct(base, others, pattern, y)
    pn = correct(base, others, pattern[rng.permutation(n)], y)
    rd = pa - b
    dd = pa - pn
    rl, rh = ci(rd)
    dl, dh = ci(dd)
    both = rl > 0 and dl > 0
    print(
        f"\n=== {ds}: STRONGEST base (t-best + raw others) {b.mean():.4f} ===\n"
        f"  +curv pattern {pa.mean():.4f}  raw d {rd.mean():+.4f} [{rl:+.4f},{rh:+.4f}]  "
        f"DoD {dd.mean():+.4f} [{dl:+.4f},{dh:+.4f}]  -> {'ADDS' if both else 'ABSORBED'}",
        flush=True,
    )
