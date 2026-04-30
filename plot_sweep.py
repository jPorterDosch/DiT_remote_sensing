"""
Plot PCK@0.1 results across a sweep of timestep (t) and block index (k).

Reads all JSON files matching layers_cat/{model}/t*_b*_e*.json and produces:
  1. Heatmap of Mean PCK@0.1 over (timestep × block_index)
  2. Line plots: PCK vs timestep (per block) and PCK vs block (per timestep)

Usage:
    python plot_sweep.py [--model flux] [--metric image|point] [--out sweep_plot.png]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_filename(path: Path) -> dict | None:
    """Extract t, k, ensemble_size from filename like t260_b[28]_e8.json"""
    name = path.stem  # e.g. "t260_b[28]_e8"
    m = re.match(r"t(\d+)_b\[?([^\]]+)\]?_e(\d+)", name)
    if not m:
        return None
    t = int(m.group(1))
    # k may be "28" or "10, 28" (multi-block); use tuple for multi, int for single
    k_ints = [int(x.strip()) for x in m.group(2).split(",")]
    k = k_ints[0] if len(k_ints) == 1 else tuple(k_ints)
    e = int(m.group(3))
    return {"t": t, "k": k, "e": e}


def load_results(model: str, metric: str) -> pd.DataFrame:
    base = Path("layers_cat") / model
    rows = []
    for json_path in sorted(base.glob("t*_b*_e*.json")):
        meta = parse_filename(json_path)
        if meta is None:
            print(f"  skipping (unparseable): {json_path.name}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        pck_mean = data[metric].get("Mean")
        pck_all = data[metric].get("All")
        if pck_mean is None:
            print(f"  skipping (no Mean key): {json_path.name}")
            continue
        rows.append({**meta, "pck_mean": pck_mean, "pck_all": pck_all})
    if not rows:
        raise FileNotFoundError(f"No result JSONs found in layers_cat/{model}/. Run eval_spair.py first.")
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, metric: str, ax: plt.Axes) -> None:
    pivot = df.pivot_table(index="t", columns="k", values="pck_mean", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)  # highest t at top
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=df["pck_mean"].min() * 0.97,
        vmax=df["pck_mean"].max() * 1.01,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Block index (k)")
    ax.set_ylabel("Timestep (t)")
    ax.set_title(f"Mean PCK@0.1 ({metric}) — heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i, t in enumerate(pivot.index):
        for j, k in enumerate(pivot.columns):
            v = pivot.loc[t, k]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="black")


def plot_lines_by_block(df: pd.DataFrame, metric: str, ax: plt.Axes) -> None:
    for k, grp in df.groupby("k"):
        grp = grp.sort_values("t")
        ax.plot(grp["t"], grp["pck_mean"], marker="o", label=f"k={k}")
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Mean PCK@0.1")
    ax.set_title(f"PCK@0.1 ({metric}) vs timestep")
    ax.legend(title="block index")
    ax.grid(True, alpha=0.3)


def plot_lines_by_timestep(df: pd.DataFrame, metric: str, ax: plt.Axes) -> None:
    for t, grp in df.groupby("t"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["pck_mean"], marker="o", label=f"t={t}")
    ax.set_xlabel("Block index (k)")
    ax.set_ylabel("Mean PCK@0.1")
    ax.set_title(f"PCK@0.1 ({metric}) vs block index")
    ax.legend(title="timestep", ncol=2)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flux")
    parser.add_argument("--metric", choices=["image", "point"], default="image")
    parser.add_argument("--out", default="sweep_plot.png")
    args = parser.parse_args()

    print(f"Loading results from layers_cat/{args.model}/ ...")
    df = load_results(args.model, args.metric)
    print(f"  {len(df)} result files found")
    print(df.to_string(index=False))

    unique_t = df["t"].nunique()
    unique_k = df["k"].nunique()

    if unique_t == 1 and unique_k == 1:
        print("Only one (t, k) found — nothing to sweep over. Exiting.")
        return

    n_axes = 1 + (unique_k > 1) + (unique_t > 1)
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]

    ax_idx = 0
    if unique_t > 1 and unique_k > 1:
        plot_heatmap(df, args.metric, axes[ax_idx])
        ax_idx += 1

    if unique_t > 1:
        plot_lines_by_block(df, args.metric, axes[ax_idx])
        ax_idx += 1

    if unique_k > 1:
        plot_lines_by_timestep(df, args.metric, axes[ax_idx])
        ax_idx += 1

    fig.suptitle(f"SPair-71k sweep — {args.model}", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
