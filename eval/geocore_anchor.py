"""GeoCore's own EuroSAT readout rebuilt from our cache: the harness-validation anchor (spec §4.6).

GeoCore (eval/frozen_probe.py upstream) scores EuroSAT with: one noise draw per image, block
OUTPUTS of double 3, double 7, single 7, single 15 (= the INPUTS of our k = 4, 8, 16, 24),
mean-pooled tokens, the four blocks concatenated (16384-d), `nn.Linear` trained with Adam
lr 1e-3 for 40 epochs at batch 1024, a random 80/20 split (`np.random.default_rng(0)`), and the
best tau of {0.05, 0.1, 0.25, 0.5, 0.75} (0.25 in their table).

This script takes the ANCHOR extraction (ensemble_size 1, t includes 260 -- the grid point
nearest their 0.25) and reports that readout. Caveats printed with the number: t = 260 not 250,
GEO-Bench m_eurosat (the image count is printed; 18,192 in the shipped partition) is a subset of
the 27,000-image EuroSAT they split, and the resolution is whatever the cache was extracted at
(224 for the table run, 256 for the check).

The anchor is a gate, not a printout (review 2026-11-12, item 10: the script used to return 0
whatever it measured). It prints `GATE anchor PASS` and returns 0 only when the accuracy is within
`--tolerance` points of GeoCore's 97.5, on either side; otherwise `GATE anchor FAIL`, exit 1, and
the GeoCore row does not ship until the gap is explained (spec, goal 3). `--check-result` applies
the same verdict to a stored result json without recomputing it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

THEIR_BLOCKS = [4, 8, 16, 24]
THEIR_T = 260
THEIR_EUROSAT = 97.5  # GeoCore-9B frozen, 256 px, tau 0.25 (their Table; DINOv3-Sat 98.0)
# Tolerance in accuracy points, two-sided. Source: the spec's goal 3, "within about one point of
# their reported 97.5" (docs/superpowers/specs/2026-11-02-geocore-eval-design.md, dated the day the
# anchor ran). Turned into an executed gate on 2026-11-12, AFTER the result was known (98.05 at
# 224 px, job 6533080: +0.55); the value is the spec's, not one chosen to fit that result.
ANCHOR_TOL_PP = 1.0


def anchor_verdict(accuracy: float, reference: float = THEIR_EUROSAT, tolerance: float = ANCHOR_TOL_PP) -> tuple[bool, str]:  # fmt: skip
    """accuracy as a fraction in [0, 1]. -> (passed, the GATE line). A non-finite or out-of-range
    accuracy fails (a NaN compares False, so it can never pass)."""
    acc = float(accuracy)
    gap = 100.0 * acc - reference
    ok = 0.0 <= acc <= 1.0 and tolerance > 0 and abs(gap) <= tolerance
    line = (
        f"GATE anchor {'PASS' if ok else 'FAIL'}: {100 * acc:.2f} vs GeoCore {reference} "
        f"(gap {gap:+.2f}, tolerance +/-{tolerance})"
    )
    if not ok:
        line += " -- do NOT ship the GeoCore row until the gap is explained"
    return ok, line


def _tolerance(text: str) -> float:
    """argparse type: the gate may be made STRICTER on the command line, never looser than the
    spec's ANCHOR_TOL_PP (review 2026-11-12, staged-delta finding 8: `--tolerance 50` passed
    anything)."""
    import math

    x = float(text)
    if not (math.isfinite(x) and 0 < x <= ANCHOR_TOL_PP):
        raise argparse.ArgumentTypeError(f"tolerance must be in (0, {ANCHOR_TOL_PP}], got {text!r}")
    return x


def anchor_accuracy(X: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 40, lr: float = 1e-3,
                    batch: int = 1024, train_frac: float = 0.8) -> float:  # fmt: skip
    """Their linear-probe recipe: random train_frac split by `default_rng(seed)`, Adam, CE loss,
    no scaler, no weight decay, last-epoch accuracy on the held-out part."""
    import torch

    torch.manual_seed(seed)
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(train_frac * n))
    tr, va = perm[:n_tr], perm[n_tr:]
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64))
    head = torch.nn.Linear(Xt.shape[1], int(yt.max()) + 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.as_tensor(tr)[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(Xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        pred = head(Xt[va]).argmax(1)
    return float((pred == yt[va]).float().mean())


def _load_all_splits(pattern: str, blocks: list[int], t: int):
    from eval import features as F

    Xs, ys = [], []
    img_size, weights = None, None
    for split in ("train", "val", "test"):
        path_glob = pattern.format(split=split)
        import glob

        hits = sorted(glob.glob(path_glob))
        if len(hits) != 1:
            raise SystemExit(f"{split}: expected exactly one cache for {path_glob}, found {hits}")
        meta = F.cache_meta(hits[0])
        if meta.get("ensemble_size") != 1:
            raise SystemExit(
                f"{hits[0]}: ensemble_size={meta.get('ensemble_size')} -- the anchor needs the ensemble_size=1 extraction"
            )
        if t not in meta.get("t", []):
            raise SystemExit(f"{hits[0]}: t={t} not among cached timesteps {meta.get('t')}")
        if meta.get("layout") != "N,K,B,C":
            raise SystemExit(f"{hits[0]}: anchor needs a multi-block cache (layout N,K,B,C)")
        ti = list(meta["t"]).index(t)
        parts = []
        for k in blocks:
            d = F._block_slice(hits[0], meta, {"k": k})
            parts.append(d["feats"][:, ti, :])
            labels = d["labels"]
        Xs.append(np.concatenate(parts, axis=1))
        ys.append(labels)
        img_size, weights = meta.get("img_size"), meta.get("weights")
    return np.concatenate(Xs), np.concatenate(ys), img_size, weights


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="anchor cache pattern with {split}")
    src.add_argument("--check-result", metavar="JSON", help="verdict on a stored result (no recomputation)")
    ap.add_argument("--blocks", type=int, nargs="+", default=THEIR_BLOCKS)
    ap.add_argument("--t", type=int, default=THEIR_T)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=_tolerance, default=ANCHOR_TOL_PP,
                    help="allowed |accuracy - 97.5| in points, in (0, 1.0] (default and maximum: the spec's one point)")  # fmt: skip
    ap.add_argument("--out", default=None, help="json result path (refused if it exists)")
    args = ap.parse_args(argv)
    if args.check_result:
        with open(args.check_result) as f:
            stored = json.load(f)
        if stored.get("geocore_reported") != THEIR_EUROSAT:
            raise SystemExit(
                f"{args.check_result}: geocore_reported={stored.get('geocore_reported')!r}, not {THEIR_EUROSAT}"
            )
        ok, line = anchor_verdict(stored["accuracy"], THEIR_EUROSAT, args.tolerance)
        print(f"{line} [{args.check_result}, {stored.get('img_size')} px, {stored.get('n_images')} images]")
        return 0 if ok else 1
    if args.out and os.path.exists(args.out):
        raise SystemExit(
            f"{args.out} exists; a result file is never overwritten (move it aside or pick another --out)"
        )
    X, y, img_size, weights = _load_all_splits(args.features, args.blocks, args.t)
    acc = anchor_accuracy(X, y, args.seed, args.epochs, args.lr, args.batch)
    ok, line = anchor_verdict(acc, THEIR_EUROSAT, args.tolerance)
    res = {"weights": weights, "img_size": img_size, "t": args.t, "blocks": args.blocks, "n_images": int(len(X)),
           "dim": int(X.shape[1]), "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "seed": args.seed,
           "accuracy": acc, "geocore_reported": THEIR_EUROSAT, "tolerance_pp": args.tolerance, "gate": "PASS" if ok else "FAIL"}  # fmt: skip
    print(f"ANCHOR eurosat acc={100 * acc:.2f} (GeoCore reports {THEIR_EUROSAT} at 256 px, tau 0.25, full EuroSAT 80/20; "
          f"here t={args.t}, {img_size} px, GEO-Bench m_eurosat {len(X)} images, blocks {args.blocks}, {weights})")  # fmt: skip
    if args.out:
        with open(args.out, "x") as f:
            json.dump(res, f, indent=2)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
