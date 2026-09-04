"""Job 2 — does token i at t_a describe the same patch as token i at t_b?

The token-geometry hypothesis reads per-token temporal statistics: it subtracts,
correlates, or profiles token i ACROSS timesteps. That is only meaningful if the token
index tracks a fixed image region along the chain. Row-major (h, w) order guarantees the
index set is the same at every timestep; it guarantees nothing about what the latent at
that index has become after the ODE has moved it.

Two measures, per arm and per timestep pair:

  cosine   same-token cos(x[n,i,ta], x[n,i,tb]) vs a null that pairs token i with a
           RANDOM token j != i of the SAME image at t_b. The null is essential: features
           in a 3072-d block share a large common component, so absolute cosine runs high
           even for unrelated tokens. Only the gap over the null is evidence of identity.

  retrieval for each token i at t_a, rank every token at t_b by cosine. If the index
           tracks a region, token i should retrieve itself. top-1 accuracy (chance = 1/L)
           and the median percentile rank of the true match. This is the sharper test:
           cosine means can stay high through a uniform drift that destroys identity,
           whereas retrieval cannot.

If retrieval collapses to chance at large t-gaps, per-token temporal statistics over that
range are computed across mismatched regions, and the rider must be restricted or dropped.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

# dataviz reference palette, categorical slots 1 and 2 (validated: CVD dE 24.7, normal 33.6)
C_SIGNAL = "#2a78d6"  # same token
C_NULL = "#eb6834"  # random-token null
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_SURFACE = "#fcfcfb"


def unit(x: np.ndarray) -> np.ndarray:
    """Row-normalize the last axis."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def analyze_pair(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """a, b: (N, L, C) token features at two timesteps. Returns alignment stats."""
    n, ell, _ = a.shape
    ah, bh = unit(a.astype(np.float32)), unit(b.astype(np.float32))

    same = np.einsum("nlc,nlc->nl", ah, bh)  # N, L

    # Null: pair token i with a random OTHER token of the same image (derangement-ish;
    # a fixed roll per image would alias with spatial autocorrelation, so sample).
    null = np.empty_like(same)
    top1 = np.empty(n)
    pct = np.empty(n)
    for i in range(n):
        perm = rng.permutation(ell)
        clash = perm == np.arange(ell)
        if clash.any():  # resample only the self-pairs
            perm[clash] = (perm[clash] + 1 + rng.integers(0, ell - 1, clash.sum())) % ell
        null[i] = np.einsum("lc,lc->l", ah[i], bh[i][perm])

        s = ah[i] @ bh[i].T  # L, L similarity
        top1[i] = float((s.argmax(axis=1) == np.arange(ell)).mean())
        # percentile rank of the true match among all candidates (1.0 = best)
        diag = s[np.arange(ell), np.arange(ell)][:, None]
        pct[i] = float((s < diag).mean(axis=1).mean())

    return {
        "same_mean": float(same.mean()),
        "same_std": float(same.std()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "gap": float(same.mean() - null.mean()),
        "top1": float(top1.mean()),
        "top1_chance": 1.0 / ell,
        "pct_rank": float(pct.mean()),
        "L": ell,
        "N": n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token-dir", default="models/token_align", help="dir holding *_tokens_*.npz")
    p.add_argument("--out-dir", default="results/token_alignment")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.token_dir, "*", "multistep_train_tokens_*.npz")))
    if not paths:
        raise SystemExit(f"FATAL: no token caches under {args.token_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows: list[dict] = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        toks = d["tokens"]  # N, S, L, C
        ts = [int(t) for t in d["timesteps"]]
        arm = str(d["extraction_mode"])
        h, w = (int(v) for v in d["grid_hw"])
        print(f"\n=== {arm}  {toks.shape}  grid {h}x{w}  t={ts}  ({os.path.basename(path)})")
        rng = np.random.default_rng(args.seed)
        for ia in range(len(ts)):
            for ib in range(ia + 1, len(ts)):
                st = analyze_pair(toks[:, ia], toks[:, ib], rng)
                st |= {"arm": arm, "t_a": ts[ia], "t_b": ts[ib], "t_gap": ts[ib] - ts[ia]}
                rows.append(st)
                print(
                    f"  t {ts[ia]:>3}->{ts[ib]:<3} (gap {st['t_gap']:>3}): "
                    f"same {st['same_mean']:.4f}  null {st['null_mean']:.4f}  "
                    f"gap {st['gap']:+.4f}   top1 {st['top1']:.3f} "
                    f"(chance {st['top1_chance']:.3f})  pct_rank {st['pct_rank']:.4f}"
                )

    fields = [
        "arm", "t_a", "t_b", "t_gap", "same_mean", "same_std", "null_mean", "null_std",
        "gap", "top1", "top1_chance", "pct_rank", "L", "N",
    ]
    csv_path = os.path.join(args.out_dir, "alignment_table.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r[k] for k in fields})
    print(f"\nwrote {csv_path}")

    plot(rows, os.path.join(args.out_dir, "alignment.png"))


def plot(rows: list[dict], out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted({r["arm"] for r in rows})
    # sharey on the cosine row: the arms are only comparable on a common scale, and the
    # inversion-vs-oneshot difference in the same-vs-null gap is the point of the panel.
    fig, axes = plt.subplots(2, len(arms), figsize=(5.6 * len(arms), 8.0), squeeze=False,
                             sharey="row")
    fig.patch.set_facecolor(C_SURFACE)

    for col, arm in enumerate(arms):
        rs = sorted([r for r in rows if r["arm"] == arm], key=lambda r: (r["t_gap"], r["t_a"]))
        labels = [f"{r['t_a']}→{r['t_b']}\ngap {r['t_gap']}" for r in rs]
        x = np.arange(len(rs))

        # --- row 0: cosine, same-token vs random-token null
        ax = axes[0][col]
        ax.set_facecolor(C_SURFACE)
        ax.errorbar(x - 0.06, [r["same_mean"] for r in rs], yerr=[r["same_std"] for r in rs],
                    fmt="o", ms=9, lw=2, capsize=4, color=C_SIGNAL, label="same token")
        ax.errorbar(x + 0.06, [r["null_mean"] for r in rs], yerr=[r["null_std"] for r in rs],
                    fmt="s", ms=9, lw=2, capsize=4, color=C_NULL, label="random-token null")
        # selective direct labels: the gap only, clear of the error-bar cap
        for xi, r in zip(x, rs, strict=True):
            ax.annotate(f"Δ{r['gap']:+.3f}", (xi, r["same_mean"] + r["same_std"]),
                        textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=9, color=C_MUTED)
        ax.set_title(f"{arm} — token self-similarity", color=C_TEXT, fontsize=12, pad=26)
        if col == 0:
            ax.set_ylabel("cosine", color=C_MUTED, fontsize=10)
        # legend above the axes so it cannot land on the null markers
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0, 1.0, 0.14), mode="expand", borderaxespad=0.0)

        # --- row 1: retrieval top-1, the sharper identity test
        ax = axes[1][col]
        ax.set_facecolor(C_SURFACE)
        ax.plot(x, [r["top1"] for r in rs], "o-", ms=9, lw=2, color=C_SIGNAL, label="top-1 retrieval")
        ch = rs[0]["top1_chance"]
        ax.axhline(ch, ls="--", lw=2, color=C_NULL, label=f"chance (1/L = {ch:.4f})")
        for xi, r in zip(x, rs, strict=True):
            ax.annotate(f"{r['top1']:.2f}", (xi, r["top1"]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, color=C_MUTED)
        ax.set_ylim(-0.04, 1.12)
        ax.set_title(f"{arm} — self-retrieval", color=C_TEXT, fontsize=12, pad=14)
        if col == 0:
            ax.set_ylabel("top-1 accuracy", color=C_MUTED, fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="center left")

        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, color=C_MUTED)
            ax.grid(axis="y", color="#e6e5e1", lw=1)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#d8d7d2")
            ax.tick_params(colors=C_MUTED)

    # headroom on the cosine row so the delta labels clear the legend strip above the axes
    lo, hi = axes[0][0].get_ylim()
    axes[0][0].set_ylim(lo, hi + 0.28 * (hi - lo))

    fig.suptitle("Token alignment across the denoising chain (EuroSAT, block 28, g=1.0)",
                 color=C_TEXT, fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150, facecolor=C_SURFACE)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
