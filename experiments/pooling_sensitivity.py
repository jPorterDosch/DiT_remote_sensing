"""Does the raw-input baseline become SENSITIVE at finer pooling?

Section 6 found every RESISC45 input-level contrast null, including `ens8 - ens1` -- which
cannot truly be zero, since averaging 8 draws reduces eps by sqrt(8) by arithmetic. So the
baseline there is an instrument that misses a manipulation known to be occurring, and its
nulls carry no weight.

The obvious diagnosis (too few dimensions for 45 classes) is WRONG: q2 at 64 dims already
exceeds K-1=44 and scores WORSE than mean at 16. The real limits are that mean/q2 average away
the spatial layout RESISC45 classes turn on, while `full` (16,384 dims at n=500, ~11 img/class)
overfits. This sweeps the middle -- 4x4 (256) and 8x8 (1024) -- re-pooled offline from the
saved `full` arrays, so it costs no re-extraction.

Read the ens8-ens1 column as the POSITIVE CONTROL. A pooling that cannot see it cannot be
trusted to report a null on anything else.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_xt_probe import best_over_c
from token_geometry_probe import boot_ci

_PAIR_REF: dict = {}


def _assert_paired(path, d):
    """boot_ci below pairs ens8/ens1 fold accuracies POSITIONALLY with strict=True, which
    passes for any two caches of equal length -- including disjoint image sets. Assert the
    caches actually cover the same images, as every other comparison script does."""
    for key in ("labels", "subset_indices"):
        if key not in d.files:
            continue
        if key in _PAIR_REF:
            if not np.array_equal(_PAIR_REF[key], d[key]):
                raise SystemExit(f"FATAL: {path} disagrees on {key} -- arms are not paired")
        else:
            _PAIR_REF[key] = d[key]


def repool(path, g):
    d = np.load(path); _assert_paired(path, d); f = d["feats"]; N, K, D = f.shape
    hw = int(round((D // 16) ** 0.5)); C = D // (hw * hw)
    x = torch.from_numpy(f).reshape(N, K, C, hw, hw)
    p = F.adaptive_avg_pool2d(x.reshape(N * K, C, hw, hw), (g, g))
    return p.reshape(N, K, C * g * g).numpy(), d["labels"]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-dir", default="models/raw_xt")
    ap.add_argument("--t-index", type=int, default=6, help="6 = t=580, where eps is largest")
    ap.add_argument("--grids", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pca", type=int, default=400)
    args = ap.parse_args()

    arms = {"ens1": f"{args.dataset}_rawxt_ens1_full.npz",
            "ens8": f"{args.dataset}_rawxt_ens8_full.npz",
            "inv":  f"{args.dataset}_invstate_n50_full.npz"}
    # The inversion-state arm costs a full chain per image, so it does not exist at every n.
    # Its absence must not block the positive control, which needs only the two raw arms.
    arms = {k: v for k, v in arms.items() if os.path.exists(os.path.join(args.raw_dir, v))}
    print(f"{args.dataset}  t-index={args.t_index}  dir={args.raw_dir}  arms={list(arms)}")
    print("(positive control = ens8 - ens1; it CANNOT truly be zero)\n")
    print(f"{'grid':>6}{'dims':>7}{'ens1':>9}{'ens8':>9}{'inv':>9}"
          f"{'ens8-ens1':>12}{'95% CI':>22}{'  sensitive?':>13}")
    print("-" * 90)
    for g in args.grids:
        acc, fold = {}, {}
        for k, fn in arms.items():
            x, y = repool(os.path.join(args.raw_dir, fn), g)
            m, _c, f = best_over_c(x[:, args.t_index, :], y, args.pca)
            acc[k], fold[k] = m, f
            n_img = x.shape[0]
        d = [a - b for a, b in zip(fold["ens8"], fold["ens1"], strict=True)]
        lo, hi = boot_ci(d)
        sens = "YES" if lo > 0 else "no (blind)"
        iv = f"{acc['inv']:>9.4f}" if 'inv' in acc else f"{'-':>9}"
        print(f"{f'{g}x{g}':>6}{x.shape[2]:>7}{acc['ens1']:>9.4f}{acc['ens8']:>9.4f}"
              f"{iv}{np.mean(d):>+12.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{sens:>13}")

if __name__ == "__main__":
    main()
