"""Timestep x block sweep on an OFFICIAL-split dataset: select (t, k, C) on val, report
test ONCE, and plot the val grid.

    # per block (one ISAAC array task each; writes results/eval/sweep/<run>.npz)
    python -m eval.sweep block --dataset m_eurosat --k 33 --arm flux-oneshot-ens8 \\
        --features 'models/m_eurosat_sweep_ens8_k33/*/multistep_{split}_feats_oneshot_g1.0.npz' \\
        --expect extraction_mode=ONESHOT ensemble_size=8
    # after every block landed
    python -m eval.sweep select --dataset m_eurosat --arm flux-oneshot-ens8 \\
        --blocks results/eval/sweep/sweep-block_m_eurosat_flux-oneshot-ens8-k*.npz

BLOCK scores every (candidate x C) cell of the official protocol at one block k
(candidates = the 7 single timesteps + the 7-t concat; eval.protocols.official_all_cells).
It stores val accuracies in the clear and the TEST predictions SEALED, so the feature
caches can be deleted afterwards (the ISAAC job does, once this file is verified).

SELECT picks the cell with the highest val accuracy across all blocks (first max wins, in
block / candidate / C order, as run_official does), then unseals exactly ONE test vector
-- the selected cell's -- and writes it in eval.probe's result format, so eval.compare
can pair it with any other official-protocol arm (rule 1: nothing about test influenced
the choice). The plot shows val accuracy only; test appears as the one reported number.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # noqa: E402

from eval import features as F  # noqa: E402
from eval import probe as PR  # noqa: E402
from eval import protocols as P  # noqa: E402
from eval import wb  # noqa: E402

OUT_DIR = "results/eval/sweep"


def _sealed_key(cand: str, C: float) -> str:
    return f"sealed__{cand}__C{C}"


# ------------------------------------------------------------------------------ block
def run_block(args) -> str:
    pins = {**F.flux_pins(args.dataset, F.parse_expect(args.expect)), "k": args.k}
    if args.dataset not in F.OFFICIAL:
        raise SystemExit(f"{args.dataset}: no OFFICIAL entry (count the shipped partition first, rule 16)")
    files = {
        s: sorted(glob.glob(args.features.format(split=s))) for s in F.official_spec(args.dataset)["sizes"]
    }
    # keys of the per-split file map are split names, identical in smoke and real runs
    if not all(files.values()):
        raise SystemExit(f"no feature files for some split: { {s: len(v) for s, v in files.items()} }")
    config = {
        "protocol": "official-sweep",
        "protocol_constants": PR.PROTOCOL_CONSTANTS["official"],
        "k": args.k,
        "pins": pins,
        "features": {s: [wb.file_identity(f) for f in v] for s, v in files.items()},
        "smoke_sizes": args.smoke_sizes,
    }
    sizes = PR.parse_smoke_sizes(args.smoke_sizes)
    exp = "sweep-block" + ("-smoke" if sizes else "")
    _, name = wb.init(exp, args.dataset, f"{args.arm}-k{args.k}", config, job_type="probe")
    print(f"== {name}")

    # Same loader + cross-split checks as eval.probe --protocol official (eps seeds, meta).
    ns = SimpleNamespace(dataset=args.dataset, features=args.features, pins=pins, view="all", sizes=sizes)
    cands, ytr, yva, yte, test_paths, inputs, extra = PR.official_flux(ns)
    for p in inputs:
        wb.use_features(p)
    rows, preds = P.official_all_cells(cands, ytr, yva, yte)
    for r in rows:
        warn = "" if r["conv_warnings"] == 0 else f"  [NOT CONVERGED x{r['conv_warnings']}]"
        print(f"  k={args.k} val {r['candidate']:>9} C={r['C']:<5} acc={r['val_acc']:.4f}{warn}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    np.savez(
        out,
        k=np.array(args.k),
        dataset=np.array(args.dataset),
        arm=np.array(args.arm),
        candidates=np.array(list(cands)),
        C_grid=np.array(P.C_GRID),
        val_table=np.array(json.dumps(rows)),
        timesteps=np.array(extra["timesteps"]),
        test_paths=np.array(test_paths),
        test_labels=yte,
        pins=np.array(json.dumps(pins)),
        run_name=np.array(name),
        smoke=np.array(bool(sizes)),
        **{_sealed_key(c, Cv): pr for (c, Cv), pr in preds.items()},
    )
    verify_block(out)
    print(f"BLOCK OK: {len(rows)} cells (val in clear, test sealed) -> {out}")
    if wandb.run is not None:
        wb.log_table("val_grid", [{"k": args.k, **r} for r in rows])
        wb.log_results(out)
        wandb.finish()
    return out


def verify_block(path: str) -> None:
    """The ISAAC job deletes the caches only after this passes. The expected cells come
    from the cache's own t-grid (each t + the all-t concat, x C_GRID), not from the stored
    candidate list, so a truncated file cannot validate itself."""
    d = np.load(path, allow_pickle=True)
    ts = [int(t) for t in d["timesteps"]]
    want_cands = [f"t{t}" for t in ts] + [f"concat{len(ts)}t"]
    want_keys = {_sealed_key(c, Cv) for c in want_cands for Cv in P.C_GRID}
    rows = json.loads(str(d["val_table"]))
    got_rows = {(r["candidate"], r["C"]) for r in rows}
    sealed = {f for f in d.files if f.startswith("sealed__")}
    if sealed != want_keys or got_rows != {(c, Cv) for c in want_cands for Cv in P.C_GRID}:
        raise SystemExit(
            f"{path}: cells incomplete -- {len(rows)} val rows / {len(sealed)} sealed vectors, "
            f"expected {len(want_keys)} ({len(want_cands)} candidates x {len(P.C_GRID)} C)"
        )
    n = len(d["test_labels"])
    if any(d[s].shape != (n,) for s in sealed):
        raise SystemExit(f"{path}: a sealed test vector has the wrong length")


# ----------------------------------------------------------------------------- select
class SealedTests:
    """Test predictions of every sweep cell; exactly ONE may be opened (rule 1)."""

    def __init__(self, blocks: dict):
        self._blocks = blocks
        self.opened: tuple | None = None

    def open(self, k: int, cand: str, C: float) -> np.ndarray:
        if self.opened is not None:
            raise RuntimeError(
                f"seal violation: test vector {self.opened} already opened; refusing {(k, cand, C)}"
            )
        self.opened = (k, cand, C)
        return self._blocks[k][_sealed_key(cand, C)]


def load_blocks(paths: list[str], dataset: str):
    blocks, block_paths, ref = {}, {}, None
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        k = int(d["k"])
        if str(d["dataset"]) != dataset:
            raise SystemExit(f"{p}: dataset {d['dataset']} != {dataset}")
        if k in blocks:
            raise SystemExit(f"two result files for block k={k} -- pass exactly one per block")
        ident = (
            [str(x) for x in d["test_paths"]],
            d["test_labels"].tolist(),
            list(d["candidates"]),
            json.loads(str(d["pins"])) | {"k": None},
        )
        if ref is None:
            ref = ident
        elif ident != ref:
            raise SystemExit(
                f"{p}: test set / candidates / pins differ from the other blocks -- not one sweep"
            )
        blocks[k] = d
        block_paths[k] = p
    return dict(sorted(blocks.items())), dict(sorted(block_paths.items())), ref


def select(blocks: dict) -> tuple[dict, list[dict]]:
    grid, best = [], None
    for k, d in blocks.items():
        for r in json.loads(str(d["val_table"])):  # candidate order, then C order
            grid.append({"k": k, **r})
            if best is None or r["val_acc"] > best["val_acc"]:
                best = {"k": k, **r}
    return best, grid


def plot(grid: list[dict], best: dict, cands: list[str], title: str, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    ink, muted, surface = "#1f1e1c", "#6b6a66", "#fcfcfb"
    blue_ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", blue_ramp)
    ks = sorted({g["k"] for g in grid})
    # Display value per (k, candidate) = best val acc over C (a val-only statistic).
    M = np.full((len(ks), len(cands)), np.nan)
    for g in grid:
        i, j = ks.index(g["k"]), cands.index(g["candidate"])
        M[i, j] = np.nanmax([M[i, j], g["val_acc"]])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 0.55 * len(ks) + 2.2), facecolor=surface, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
    M = M * 100  # percent everywhere: cells, colorbar, curves
    im = ax1.imshow(M, cmap=cmap, aspect="auto")
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(len(ks)):
        for j in range(len(cands)):
            v = M[i, j]
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
            ax1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if dark else ink)
    bi, bj = ks.index(best["k"]), cands.index(best["candidate"])
    # Two-tone ring: visible on the darkest (typically selected) cell and on light ones.
    ax1.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=ink, linewidth=4.5))
    ax1.add_patch(Rectangle((bj - 0.44, bi - 0.44), 0.88, 0.88, fill=False, edgecolor="white", linewidth=2.0))
    ax1.set_xticks(
        range(len(cands)),
        [c.replace("concat", "concat ") for c in cands],
        rotation=0,
        fontsize=8,
        color=muted,
    )
    ax1.set_yticks(range(len(ks)), [f"k={k}" for k in ks], fontsize=8, color=muted)
    ax1.set_title("Val top-1 (%), best C per cell; outlined = selected", fontsize=10, color=ink, loc="left")
    for s in ax1.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02, label="val top-1 (%)")
    cb.ax.yaxis.label.set_color(muted)
    cb.ax.yaxis.label.set_fontsize(8)
    cb.ax.tick_params(labelsize=7, colors=muted)
    cb.outline.set_visible(False)

    # Focus pattern: every block as thin gray context; the selected block and the banked
    # default k=28 highlighted and direct-labelled (8 ordinal hues fail the ΔL check).
    tcands = [c for c in cands if not c.startswith("concat")]
    ts = [int(c[1:]) for c in tcands]
    for i, k in enumerate(ks):
        y = [M[i, cands.index(c)] for c in tcands]
        if k == best["k"]:
            ax2.plot(ts, y, color="#256abf", linewidth=2.2, marker="o", markersize=5, zorder=3)
            ax2.annotate(
                f"k={k} (selected)",
                (ts[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=ink,
                va="center",
            )
        elif k == 28:
            ax2.plot(ts, y, color=ink, linewidth=1.6, linestyle="--", zorder=2)
            ax2.annotate(
                "k=28 (banked)",
                (ts[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=muted,
                va="center",
            )
        else:
            ax2.plot(ts, y, color="#c9c7c1", linewidth=1.0, zorder=1)
    ax2.set_xticks(ts, [str(t) for t in ts], fontsize=8, color=muted)
    ax2.tick_params(axis="y", labelsize=8, colors=muted)
    ax2.set_xlabel("timestep t", fontsize=9, color=muted)
    ax2.set_ylabel("val top-1 (%)", fontsize=9, color=muted)
    ax2.grid(axis="y", color="#e6e4df", linewidth=0.8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color("#d6d4ce")
    ax2.set_title(
        "Val top-1 vs t (gray: other blocks, values in the heatmap)", fontsize=10, color=ink, loc="left"
    )
    ax2.set_xlim(ts[0] - 20, ts[-1] + 110)
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=surface)
    plt.close(fig)


def run_select(args) -> str:
    from sklearn.metrics import f1_score

    paths = sorted({p for pat in args.blocks for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no block results match {args.blocks}")
    blocks, block_paths, (test_paths, test_labels, cands, _) = load_blocks(paths, args.dataset)
    smoke = any(bool(d["smoke"]) for d in blocks.values())
    config = {
        "protocol": "official-sweep-select",
        "blocks": {k: wb.file_identity(p) for k, p in block_paths.items()},
        "C_grid": P.C_GRID,
    }
    exp = "sweep-select" + ("-smoke" if smoke else "")
    _, name = wb.init(exp, args.dataset, args.arm, config, job_type="probe")
    print(f"== {name}  blocks {list(blocks)}")
    for p in paths:
        wb.use_results(p)

    best, grid = select(blocks)
    print(f"SELECTED on val: k={best['k']} {best['candidate']} C={best['C']} (val {best['val_acc']:.4f})")
    sealed = SealedTests(blocks)
    pred = sealed.open(best["k"], best["candidate"], best["C"])
    yte = np.asarray(test_labels)
    correct = (pred == yte).astype(np.int8)
    n_cls = len(F.official_classes(args.dataset))
    f1 = float(f1_score(yte, pred, average="macro", labels=np.arange(n_cls)))
    top1 = float(correct.mean())
    se = (top1 * (1 - top1) / len(yte)) ** 0.5
    print(f"TEST (reported once): top1={top1:.4f} ±{1.96 * se:.4f}  macro_f1={f1:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    info = {"selected": best, "macro_f1": f1, "n_cells": len(grid), "blocks": list(blocks)}
    np.savez(
        out,
        cells=np.array(["test"]),
        correct__test=correct,
        ev__test=np.arange(len(yte)),
        paths=np.array(test_paths),
        labels=yte,
        protocol=np.array("official"),
        arm=np.array(args.arm),
        dataset=np.array(args.dataset),
        run_name=np.array(name),
        config=np.array(json.dumps(config, default=str)),
        info=np.array(json.dumps(info, default=str)),
        val_grid=np.array(json.dumps(grid)),
        smoke=np.array(smoke),
    )
    png = out.replace(".npz", ".png")
    title = (
        f"{args.dataset} t x k sweep ({args.arm}): selected k={best['k']} {best['candidate']} C={best['C']}, "
        f"test top-1 {100 * top1:.1f}% (reported once)"
    )
    plot(grid, best, cands, title, png)
    print(f"result -> {out}\nplot   -> {png}")
    if wandb.run is not None:
        wandb.run.summary.update({"acc/test": top1, "macro_f1": f1, "selected": json.dumps(best)})
        wb.log_table("val_grid", grid)
        wandb.log({"sweep_plot": wandb.Image(png)})
        wb.log_results(out)
        wandb.finish()
    return out


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("block", help="score every (candidate x C) cell at one block k")
    b.add_argument("--dataset", required=True)
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--arm", required=True)
    b.add_argument("--features", required=True, help="cache pattern with {split}")
    b.add_argument("--expect", nargs="*", default=None, metavar="KEY=VALUE")
    b.add_argument(
        "--smoke-sizes", default=None, metavar="TRAIN,VAL,TEST", help="SMOKE ONLY: subset split sizes"
    )
    b.add_argument("--out-dir", default=OUT_DIR)
    s = sub.add_parser("select", help="select on val across blocks, report test once, plot")
    s.add_argument("--dataset", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--blocks", nargs="+", required=True, help="block result files or globs")
    s.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)
    return run_block(args) if args.cmd == "block" else run_select(args)


if __name__ == "__main__":
    main()
