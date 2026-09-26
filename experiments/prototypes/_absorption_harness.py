"""Shared 'does block X add beyond the strongest base' harness for the 6w candidate
battery. Functions copied VERBATIM from experiments/curv_absorption_recheck.py (→ experiments/prototypes/) (the
section-13 validated instrument) — do not 'improve' them here; changes belong in a
versioned copy with a re-validation run.

Base = section-13 honest strongest base: inversion cache t-best (index 1 = t180 for
RESISC45) DiTF-normalized and PCA-512-protected inside the fold, the other six
timesteps appended raw. Blocks are standardized and appended raw (never enter the
base's PCA). Verdict = rule-3b double bar.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

C = 0.1
SEEDS = [0, 1, 2]
DISCARD = [154, 1446]

RESISC45_BASE = "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz"
RESISC45_TBEST_IDX = 1  # t=180, the a-priori best-t of the inversion arm (section 13)


def ditf(f, m):
    x = f.astype(np.float64).copy()
    x[:, :, DISCARD] = 0.0
    mu = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    x = (x - mu) / np.sqrt(v + 1e-6)
    x = (1 + m[None, :, 1, :]) * x + m[None, :, 0, :]
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


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


def correct(base, extra, blk, y, n_jobs=7):
    out = np.zeros(len(y))
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(base, y))
        for va, pred in Parallel(n_jobs=n_jobs)(delayed(_fold)(base, extra, blk, y, tr, va) for tr, va in jobs):
            out[va] += pred == y[va]
    return out / len(SEEDS)


def ci(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def load_base(base_path=RESISC45_BASE, tbest=RESISC45_TBEST_IDX):
    """Returns (base, others, labels, subset_indices)."""
    df = np.load(base_path)
    feats = ditf(df["feats"], df["mods"])
    base = feats[:, tbest, :]
    others = np.concatenate([feats[:, i, :] for i in range(feats.shape[1]) if i != tbest], axis=1)
    return base, others, df["labels"], df["subset_indices"]


def run_block(name, base, others, block, y, null_seed=0, n_jobs=7):
    """Standard three-arm evaluation: base / +block / +shuffled-null. Prints and returns
    the rule-3b verdict plus per-image correctness vectors for caching (rule 12)."""
    rng = np.random.default_rng(null_seed)
    b = correct(base, others, None, y, n_jobs)
    pa = correct(base, others, block, y, n_jobs)
    pn = correct(base, others, block[rng.permutation(len(y))], y, n_jobs)
    rd, dd = pa - b, pa - pn
    rl, rh = ci(rd)
    dl, dh = ci(dd)
    both = rl > 0 and dl > 0
    print(
        f"=== {name}: base {b.mean():.4f}  +block {pa.mean():.4f}  "
        f"raw d {rd.mean():+.4f} [{rl:+.4f},{rh:+.4f}]  "
        f"DoD {dd.mean():+.4f} [{dl:+.4f},{dh:+.4f}]  -> {'ADDS' if both else 'ABSORBED/FAIL'}",
        flush=True,
    )
    return {"name": name, "base": b, "plus": pa, "null": pn,
            "raw_ci": (rl, rh), "dod_ci": (dl, dh), "adds": both}
