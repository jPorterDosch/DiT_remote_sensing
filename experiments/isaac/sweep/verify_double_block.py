"""Evidence that a multi-block cache holds the POST-2026-10-05 double-block feature, for caches
whose `double_block_feature` stamp was backfilled by hand (2026-10-08) instead of written by the
extractor. eval.features refuses such a cache unless this script's record vouches for its sha1.

Pre-fix, DoubleStreamBlock.forward_feat returned (1 + scale2) * LN(x) + shift2 per token, where
LN has no affine and scale2/shift2 depend on t only, not on the image. Every pooled feature f then
satisfies sum_c (f[c] - shift2[c]) / (1 + scale2[c]) = 0: all images of a split lie on ONE affine
hyperplane. The post-fix feature (the raw block input) has no such constraint. Checks, per double
block at a few timesteps:

  train  smallest / median singular value of the centred N x C matrix (N > C) >= HYPER_MIN.
         Positive control (must FAIL): the same rows pushed through the pre-fix construction
         (LayerNorm, then the cache's own stored mod as the affine) fall below HYPER_MIN.
  val/test (N < C, so no hyperplane test): median per-image L2 norm within NORM_RATIO of train's
         at the same (t, block). Positive control (must FAIL): the pre-fix construction of the
         split falls outside it.

Writes <cache_root>/double_block_verification.json and prints PASS/FAIL (rule 13). CPU only.

  python experiments/isaac/sweep/verify_double_block.py models/m_forestnet_multiblock_ens8
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

from config_types import DOUBLE_BLOCK_FEATURE, NUM_DOUBLE_BLOCKS  # noqa: E402
from eval import wb  # noqa: E402
from eval.features import npz_member_memmap  # noqa: E402

HYPER_MIN = 1e-3  # real post-fix double blocks measured 0.05-0.2, the pre-fix construction ~1e-5
NORM_RATIO = (0.8, 1.25)
T_CHECK = (100, 500, 900)
RECORD = "double_block_verification.json"


def _rows(path: str, ti: list[int], js: list[int]) -> np.ndarray:
    """feats[:, ti, js, :] as float64 (N, len(ti), len(js), C); a contiguous j-range per t."""
    F = npz_member_memmap(path, "feats")
    lo, hi = min(js), max(js) + 1
    out = np.stack([np.asarray(F[:, t, lo:hi, :]) for t in ti], axis=1).astype(np.float64)
    return out[:, :, [j - lo for j in js], :]


def _pre_fix(X: np.ndarray, mod: np.ndarray) -> np.ndarray:
    """The pre-fix construction on pooled vectors: LN over channels, then (1 + scale) * . + shift."""
    ln = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return (1.0 + mod[1]) * ln + mod[0]


def hyper(X: np.ndarray) -> float:
    Xc = X - X.mean(0)
    ev = np.clip(np.linalg.eigvalsh(Xc.T @ Xc), 0, None)  # singular values^2, ascending
    s = np.sqrt(ev)
    return float(s[0] / np.median(s))


def med_norm(X: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(X, axis=1)))


def verify(root: str) -> dict:
    caches = sorted(glob.glob(os.path.join(root, "*", "multistep_*_feats_*.npz")))
    by_split = {}
    for p in caches:
        meta = json.load(open(p.replace(".npz", "_meta.json")))
        if meta.get("layout") != "N,K,B,C":
            raise SystemExit(f"{p}: not a multi-block cache")
        if meta["split"] in by_split:
            raise SystemExit(f"{root}: two {meta['split']} caches -- verify one extraction at a time")
        by_split[meta["split"]] = (p, meta)
    if "train" not in by_split:
        raise SystemExit(f"{root}: no train cache")
    tp, tmeta = by_split["train"]
    ks = tmeta["k"]
    js = [j for j, k in enumerate(ks) if k < NUM_DOUBLE_BLOCKS]
    if not js:
        raise SystemExit(f"{root}: no double-stream block in {ks}")
    ti = [tmeta["t"].index(t) for t in T_CHECK if t in tmeta["t"]]
    mods = np.load(tp)["mods"].astype(np.float64)  # (K, B, 3, C)
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}", flush=True)

    Xtr = _rows(tp, ti, js)
    n, C = Xtr.shape[0], Xtr.shape[-1]
    if n <= C:
        raise SystemExit(f"{tp}: N={n} <= C={C}, the hyperplane test needs more images than channels")
    stats = {"train": {}}
    for a, t in enumerate(ti):
        for b, j in enumerate(js):
            X = Xtr[:, a, b]
            real, ctrl = hyper(X), hyper(_pre_fix(X, mods[t, j]))
            stats["train"][f"t{tmeta['t'][t]}_k{ks[j]}"] = {"hyper": real, "pre_fix_control": ctrl}
            check(
                f"train t={tmeta['t'][t]} k={ks[j]}: no pre-fix hyperplane",
                real >= HYPER_MIN,
                f"({real:.3e})",
            )
            check(f"train t={tmeta['t'][t]} k={ks[j]}: control detected", ctrl < HYPER_MIN, f"({ctrl:.3e})")
    for split in sorted(set(by_split) - {"train"}):
        p, meta = by_split[split]
        if meta["k"] != ks or meta["t"] != tmeta["t"]:
            raise SystemExit(f"{p}: t/k differ from the train cache")
        Xs = _rows(p, ti, js)
        stats[split] = {}
        for a, t in enumerate(ti):
            for b, j in enumerate(js):
                ref = med_norm(Xtr[:, a, b])
                r = med_norm(Xs[:, a, b]) / ref
                rc = med_norm(_pre_fix(Xs[:, a, b], mods[t, j])) / ref
                stats[split][f"t{tmeta['t'][t]}_k{ks[j]}"] = {"norm_ratio": r, "pre_fix_control": rc}
                inside = NORM_RATIO[0] <= r <= NORM_RATIO[1]
                check(f"{split} t={tmeta['t'][t]} k={ks[j]}: norm ratio to train", inside, f"({r:.3f})")
                out = not (NORM_RATIO[0] <= rc <= NORM_RATIO[1])
                check(f"{split} t={tmeta['t'][t]} k={ks[j]}: control detected", out, f"({rc:.3f})")
    print(f"DOUBLE-BLOCK VERIFY {root}:", "PASS" if ok else "FAIL")
    return {
        "feature": DOUBLE_BLOCK_FEATURE,
        "pass": ok,
        "date": datetime.date.today().isoformat(),
        "criteria": {"hyper_min": HYPER_MIN, "norm_ratio": NORM_RATIO, "timesteps": T_CHECK},
        "caches": {wb.rel(p): {"split": s, **wb.file_identity(p)} for s, (p, _) in sorted(by_split.items())},
        "stats": stats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="multi-block cache roots, e.g. models/m_eurosat_multiblock_ens8")
    args = ap.parse_args(argv)
    bad = 0
    for root in args.roots:
        rec = verify(root)
        tmp = os.path.join(root, f".{RECORD}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, os.path.join(root, RECORD))
        bad += not rec["pass"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
