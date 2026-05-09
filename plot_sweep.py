"""
Plot sweep results across timestep (t) and block index (k).

Supports both correspondence (PCK@0.1) and classification (accuracy/F1) tasks.
Task type is auto-detected from the JSON structure, or can be forced with --task.

Results are searched recursively under --results-dir, so per-(t,k) subdirectories
produced by the classification sweep are handled transparently.

JSON filenames must match: t<int>_b<...>_e<int>[_seed<int>].json

Usage:
    # Correspondence (SPair) — backward-compatible
    python plot_sweep.py --results-dir layers_cat/flux --metric image

    # Correspondence, point metric
    python plot_sweep.py --results-dir layers_cat/spair_flux --metric point

    # Classification (EuroSAT), 100% label fraction
    python plot_sweep.py --results-dir layers_cat/eurosat_flux --metric top1_accuracy

    # Classification, low-shot (10% labels)
    python plot_sweep.py --results-dir layers_cat/eurosat_flux --metric macro_f1 --frac 10

    # Override title and output prefix
    python plot_sweep.py --results-dir layers_cat/eurosat_flux --metric top1_accuracy \\
        --title "EuroSAT sweep — FLUX" --out plots/eurosat_sweep
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> dict | None:
    """
    Extract (t, k, ensemble_size, seed) from filenames like:
      t260_b[28]_e8.json
      t260_b[28]_e8_seed42.json
      t260_b28_e8_seed42.json
    """
    name = path.stem
    m = re.match(r"t(\d+)_b\[?([^\]_]+)\]?_e(\d+)(?:_seed(\d+))?", name)
    if not m:
        return None
    t = int(m.group(1))
    k_ints = [int(x.strip()) for x in m.group(2).split(",")]
    k = str(k_ints[0]) if len(k_ints) == 1 else str(tuple(k_ints))
    k_numeric = k_ints[0] if len(k_ints) == 1 else None
    e = int(m.group(3))
    seed = int(m.group(4)) if m.group(4) else None
    return {"t": t, "k": k, "k_numeric": k_numeric, "e": e, "seed": seed}


# ---------------------------------------------------------------------------
# Task detection and metric extraction
# ---------------------------------------------------------------------------

def detect_task(data: dict) -> str:
    """Infer task type from JSON structure."""
    if "image" in data or "point" in data:
        return "correspondence"
    return "classification"


def get_metric_label(task: str, metric: str, frac: float | None) -> str:
    """Human-readable y-axis / title label."""
    if task == "correspondence":
        return f"Mean PCK@0.1 ({metric})"
    frac_str = f" ({frac:.0f}% labels)" if frac is not None else ""
    labels = {
        "top1_accuracy": f"Top-1 Accuracy{frac_str}",
        "macro_f1": f"Macro F1{frac_str}",
        "weighted_f1": f"Weighted F1{frac_str}",
    }
    return labels.get(metric, f"{metric}{frac_str}")


def extract_value(data: dict, task: str, metric: str, frac: float) -> float | None:
    """Pull the scalar metric value out of a result JSON."""
    if task == "correspondence":
        section = data.get(metric, {})
        return section.get("Mean")

    # classification: keys are label fractions (stored as floats or stringified floats)
    for key, entry in data.items():
        try:
            if abs(float(key) - frac) < 0.01:
                return entry.get(metric)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(
    results_dir: str,
    task: str,
    metric: str,
    frac: float,
) -> pd.DataFrame:
    base = Path(results_dir)
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    rows = []
    seen: set[tuple] = set()  # deduplicate on (t, k, seed)

    for json_path in sorted(base.rglob("t*_b*_e*.json")):
        meta = parse_filename(json_path)
        if meta is None:
            print(f"  skipping (unparseable filename): {json_path.name}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        detected = detect_task(data)

        # enforce task filter if explicitly set
        if task != "auto" and detected != task:
            print(f"  skipping (task mismatch — expected {task}, got {detected}): {json_path.name}")
            continue
        actual_task = detected

        val = extract_value(data, actual_task, metric, frac)
        if val is None:
            print(f"  skipping (metric '{metric}' not found at frac={frac}): {json_path.name}")
            continue

        dedup_key = (meta["t"], meta["k"], meta.get("seed"))
        if dedup_key in seen:
            print(f"  skipping (duplicate t/k/seed): {json_path.name}")
            continue
        seen.add(dedup_key)

        rows.append({**meta, "value": val, "task": actual_task})

    if not rows:
        raise FileNotFoundError(
            f"No usable result JSONs found under '{results_dir}' "
            f"(task={task}, metric='{metric}', frac={frac})."
        )

    df = pd.DataFrame(rows)
    print(f"  {len(df)} result(s) loaded  [task={df['task'].iloc[0]}]")
    print(df[["t", "k", "e", "seed", "value"]].sort_values(["t", "k"]).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, ylabel: str, ax: plt.Axes) -> None:
    pivot = df.pivot_table(index="t", columns="k", values="value", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    vmin = df["value"].min()
    vmax = df["value"].max()
    padding = (vmax - vmin) * 0.03 if vmax > vmin else 1.0

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin - padding,
        vmax=vmax + padding,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Block index (k)")
    ax.set_ylabel("Timestep (t)")
    ax.set_title(f"{ylabel} — heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, t_val in enumerate(pivot.index):
        for j, k_val in enumerate(pivot.columns):
            v = pivot.loc[t_val, k_val]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="black")


def plot_lines_by_block(df: pd.DataFrame, ylabel: str, ax: plt.Axes) -> None:
    for k, grp in df.groupby("k"):
        grp = grp.sort_values("t")
        ax.plot(grp["t"], grp["value"], marker="o", label=f"k={k}")
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs timestep")
    ax.legend(title="block index")
    ax.grid(True, alpha=0.3)


def plot_lines_by_timestep(df: pd.DataFrame, ylabel: str, ax: plt.Axes) -> None:
    x_col = "k_numeric" if df["k_numeric"].notna().all() else "k"
    for t_val, grp in df.groupby("t"):
        grp = grp.sort_values(x_col)
        ax.plot(grp[x_col], grp["value"], marker="o", label=f"t={t_val}")
    ax.set_xlabel("Block index (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs block index")
    ax.legend(title="timestep", ncol=2)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot sweep results (correspondence or classification) over t × k.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", default="layers_cat/flux",
        help="Directory (searched recursively) containing result JSON files. "
             "Default: layers_cat/flux",
    )
    parser.add_argument(
        "--task", default="auto", choices=["auto", "correspondence", "classification"],
        help="Force task type, or 'auto' to detect from JSON structure.",
    )
    parser.add_argument(
        "--metric", default="image",
        help="Metric to plot. "
             "Correspondence: 'image' or 'point' (→ Mean PCK@0.1). "
             "Classification: 'top1_accuracy', 'macro_f1', 'weighted_f1'. "
             "Default: image",
    )
    parser.add_argument(
        "--frac", type=float, default=100.0,
        help="Label fraction %% for classification results (e.g. 10, 50, 100). "
             "Ignored for correspondence. Default: 100.0",
    )
    parser.add_argument(
        "--title", default="",
        help="Override the figure suptitle. If empty, auto-generated from --results-dir.",
    )
    parser.add_argument(
        "--out", default="sweep_plot",
        help="Output filename prefix (no extension). "
             "Final file: <out>_<metric>.png. Default: sweep_plot",
    )
    args = parser.parse_args()

    print(f"Loading results from '{args.results_dir}' ...")
    df = load_results(args.results_dir, args.task, args.metric, args.frac)

    task = df["task"].iloc[0]
    ylabel = get_metric_label(task, args.metric, args.frac if task == "classification" else None)

    unique_t = df["t"].nunique()
    unique_k = df["k"].nunique()

    if unique_t == 1 and unique_k == 1:
        t0, k0, v0 = df.iloc[0][["t", "k", "value"]]
        print(f"Only one (t={t0}, k={k0}) found — {ylabel}: {v0:.2f}. Nothing to plot.")
        return

    show_heatmap = unique_t > 1 and unique_k > 1
    show_by_block = unique_t > 1
    show_by_timestep = unique_k > 1
    n_axes = show_heatmap + show_by_block + show_by_timestep

    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]

    ax_idx = 0
    if show_heatmap:
        plot_heatmap(df, ylabel, axes[ax_idx]); ax_idx += 1
    if show_by_block:
        plot_lines_by_block(df, ylabel, axes[ax_idx]); ax_idx += 1
    if show_by_timestep:
        plot_lines_by_timestep(df, ylabel, axes[ax_idx]); ax_idx += 1

    suptitle = args.title if args.title else f"{Path(args.results_dir).name} sweep"
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()

    out_path = f"{args.out}_{args.metric}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()