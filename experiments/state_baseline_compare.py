"""One table per dataset: raw inputs vs DiT features, across all arms.

Columns are the six things that can be probed at each timestep:

  raw x_t ens1/ens8   the one-shot NETWORK INPUT, closed form (1-t)*x0 + t*eps, no DiT
  inv state           the inversion NETWORK INPUT, the ODE state z_t, no block features
  DiT one ens1/ens8   block-28 features from the one-shot arm
  DiT inv             block-28 features from the inversion chain

Each raw arm is reported at its BEST pooling over {mean, q2, full} and its best C, which is
the conservative choice when the question is whether the DiT features add anything.

The two raw arms differ in exactly one way that matters: the one-shot input carries a
per-image eps at relative magnitude eta = t/(1-t), while the inversion state carries none --
it is a deterministic function of x0. So the contrast between the two raw columns isolates
the eps-variance term with the DiT held out of the picture entirely.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_xt_probe import POOLINGS, best_over_c, load  # noqa: E402
from token_geometry_probe import boot_ci  # noqa: E402


def _assert_ens(path: str, expected: int) -> None:
    """ens1 and ens8 one-shot caches share a filename; labels/subset_indices/timesteps all
    match, so a swapped --dit-one-ens1/--dit-one-ens8 passes every existing guard and flips
    the sign of the reported columns. The npz already stores the ensemble size -- read it."""
    import json
    import os

    meta = os.path.splitext(path)[0] + "_meta.json"
    if os.path.exists(meta):
        got = json.load(open(meta)).get("ensemble_size")
        if got is not None and int(got) != expected:
            raise SystemExit(
                f"FATAL: {path} has ensemble_size={got}, expected {expected} — arms swapped?"
            )



def raw_arm(stem: str, raw_dir: str, y: np.ndarray, idx: np.ndarray, ts: list[int]) -> dict:
    """Load one raw arm's three poolings, asserting pairing."""
    out = {}
    for how in POOLINGS:
        r = load(os.path.join(raw_dir, f"{stem}_{how}.npz"))
        if not np.array_equal(r["labels"], y) or not np.array_equal(r["idx"], idx):
            raise SystemExit(f"FATAL: {stem}_{how} is not paired with the DiT arms")
        if r["ts"] != ts:
            raise SystemExit(f"FATAL: {stem}_{how} timesteps differ")
        out[how] = r["feats"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--raw-dir", default="models/raw_xt")
    p.add_argument("--dit-one-ens1", required=True)
    p.add_argument("--dit-one-ens8", required=True)
    p.add_argument("--dit-inv", required=True)
    p.add_argument("--invstate-stem", default=None, help="default {dataset}_invstate_n50")
    p.add_argument("--pca", type=int, default=400)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    dits = {"DiT one ens1": load(args.dit_one_ens1),
            "DiT one ens8": load(args.dit_one_ens8),
            "DiT inv": load(args.dit_inv)}
    ref = dits["DiT one ens1"]
    y, ts, idx = ref["labels"], ref["ts"], ref["idx"]
    for k, v in dits.items():
        if not np.array_equal(v["labels"], y) or not np.array_equal(v["idx"], idx):
            raise SystemExit(f"FATAL: {k} is not paired with the reference arm")
        if v["ts"] != ts:
            raise SystemExit(f"FATAL: {k} timesteps differ")

    stem = args.invstate_stem or f"{args.dataset}_invstate_n50"
    raws = {"raw x_t ens1": raw_arm(f"{args.dataset}_rawxt_ens1", args.raw_dir, y, idx, ts),
            "raw x_t ens8": raw_arm(f"{args.dataset}_rawxt_ens8", args.raw_dir, y, idx, ts),
            "inv state": raw_arm(stem, args.raw_dir, y, idx, ts)}

    n_cls = len(np.unique(y))
    print(f"{args.dataset}: n={len(y)} classes={n_cls} chance={1/n_cls:.4f}  PCA={args.pca}\n")

    cols = ["raw x_t ens1", "raw x_t ens8", "inv state", "DiT one ens1", "DiT one ens8", "DiT inv"]
    hdr = f"{'t':<8}{'eta':>7}" + "".join(f"{c:>14}" for c in cols)
    print(hdr); print("-" * len(hdr))

    rows = []

    def line(name: str, t_nom: int | None, sl):
        acc, folds, pooled = {}, {}, {}
        for c in cols:
            if c in raws:
                best = max(
                    ((best_over_c(sl(raws[c][h]), y, args.pca), h) for h in POOLINGS),
                    key=lambda z: z[0][0])
                (m, _cc, f), how = best
                pooled[c] = how
            else:
                m, _cc, f = best_over_c(sl(dits[c]["feats"]), y, args.pca)
                pooled[c] = "-"
            acc[c], folds[c] = m, f
        eta = "" if t_nom is None else f"{(t_nom/1000)/(1-t_nom/1000):>7.2f}"
        print(f"{name:<8}{eta}" + "".join(f"{acc[c]:>14.4f}" for c in cols))
        row = {"dataset": args.dataset, "comparison": name,
               "eta": None if t_nom is None else round((t_nom/1000)/(1-t_nom/1000), 4)}
        for c in cols:
            row[c.replace(" ", "_")] = round(acc[c], 6)
            if c in raws:
                row[c.replace(" ", "_") + "_pool"] = pooled[c]
        # The contrast of interest: does removing eps from the INPUT alone (one-shot -> ODE
        # state) reproduce the direction the DiT arms show?
        for a, b in [("inv state", "raw x_t ens1"), ("raw x_t ens8", "raw x_t ens1"),
                     ("DiT inv", "DiT one ens1"), ("DiT one ens8", "DiT one ens1")]:
            d = [u - v for u, v in zip(folds[a], folds[b], strict=True)]
            lo, hi = boot_ci(d)
            key = f"{a}_minus_{b}".replace(" ", "_")
            row[key] = round(float(np.mean(d)), 6)
            row[key + "_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)

    for i, t in enumerate(ts):
        line(f"t={t}", t, lambda f, i=i: f[:, i, :])
    line("concat", None, lambda f: f.reshape(f.shape[0], -1))

    print("\nPaired deltas (mean, 95% CI over 3 seeds x 5 folds):")
    for key in ["inv_state_minus_raw_x_t_ens1", "raw_x_t_ens8_minus_raw_x_t_ens1",
                "DiT_inv_minus_DiT_one_ens1", "DiT_one_ens8_minus_DiT_one_ens1"]:
        print(f"\n  {key}")
        for r in rows:
            print(f"    {r['comparison']:<10}{r[key]:>+10.4f}  {r[key + '_ci']}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
