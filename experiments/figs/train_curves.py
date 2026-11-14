"""Training curves of Git-10M alignment runs from their own logs (no wandb needed).

Per run: models/git10m/<run>/stats.npz (one row per optimizer step: step/flow_loss,
step/align_loss, step/grad_norm, ...) and val.jsonl (every VAL_EVERY steps). Writes
results/figs/train_curves.png (all runs, one column each) and prints a spike summary.

    python experiments/figs/train_curves.py [--runs name=run_dir ...] [--out results/figs/train_curves.png]
"""

import argparse
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT = {
    "fm s42": "git10m_flux_ca8e6ce7+42+fm-8c098506",
    "fm s43": "git10m_flux_ca8e6ce7+43+fm-c0aa0eab",
    "naive s42": "git10m_flux_ca8e6ce7+42+naive-11e56b1e",
    "proj s42": "git10m_flux_ca8e6ce7+42+proj-ffcd7963",
    "proj s43": "git10m_flux_ca8e6ce7+43+proj-ffcd7963",
}


def smooth(x, w=100):
    x = np.asarray(x, float)
    if w <= 1 or x.size < w:
        return x
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), k, mode="same")


def load(run_dir):
    """Finished run: stats.npz + val.jsonl. In-flight run (no stats.npz yet): the resumable ckpt.pt
    (written every align.ckpt_every steps) carries the same per-step arrays and the val rows."""
    npz = os.path.join(run_dir, "stats.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        s = {k[5:]: np.asarray(d[k], float) for k in d.files if k.startswith("step/")}
        rows = [json.loads(line) for line in open(os.path.join(run_dir, "val.jsonl"))]
        return s, rows, None
    import sys
    import torch  # only for in-flight runs; the pickle references src/config_types

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    n = int(ck["global_step"])
    s = {k: np.asarray(v, float)[:n] for k, v in ck["stats"]["sc"].items()}
    return s, ck["val_rows"], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/git10m")
    ap.add_argument("--runs", nargs="*", default=None, help="name=run_dir_basename")
    ap.add_argument(
        "--inflight",
        action="store_true",
        help="every run dir under --models that has a ckpt.pt but no stats.npz",
    )
    ap.add_argument("--out", default="results/figs/train_curves.png")
    ap.add_argument("--smooth", type=int, default=100)
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs) if a.runs else DEFAULT
    if a.inflight:
        runs = {
            d.replace("git10m_flux_", ""): d
            for d in sorted(os.listdir(a.models))
            if os.path.exists(os.path.join(a.models, d, "ckpt.pt"))
            and not os.path.exists(os.path.join(a.models, d, "stats.npz"))
        }
        if not runs:
            print("no in-flight runs")
            return
    n = len(runs)
    fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 11), sharex=True, squeeze=False)
    for j, (name, rd) in enumerate(runs.items()):
        s, rows, inflight = load(os.path.join(a.models, rd))
        if inflight is not None:
            name = f"{name} (in flight, step {inflight})"
        steps = np.arange(1, len(s["flow_loss"]) + 1)
        vs = np.array([r["step"] for r in rows])
        vflow = np.array([r["val/flow_mse_loss"] for r in rows])
        vcos = np.array([-r["val/align_loss_mean"] for r in rows])
        g = s["grad_norm"]
        burst = np.flatnonzero(g > 1.0)
        ax = axes[0, j]
        ax.plot(steps, s["flow_loss"], lw=0.3, alpha=0.35, color="gray")
        ax.plot(steps, smooth(s["flow_loss"], a.smooth), color="C0", label="train (smoothed)")
        ax.plot(vs, vflow, "o-", ms=3, color="C3", label="val")
        ax.set_ylim(0.45, 1.2)
        ax.set_title(f"{name}\n{rd}", fontsize=8)
        ax.set_ylabel("flow loss")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        ax.plot(steps, -s["align_loss"], lw=0.3, alpha=0.35, color="gray")
        ax.plot(steps, smooth(-s["align_loss"], a.smooth), color="C0")
        ax.plot(vs, vcos, "o-", ms=3, color="C3")
        ax.set_ylim(0.3, 0.7)
        ax.set_ylabel("teacher cosine")
        ax = axes[2, j]
        ax.semilogy(steps, g, lw=0.4, color="C0")
        ax.axhline(1.0, color="k", ls="--", lw=0.7)
        ax.set_ylabel("grad norm (pre-clip)")
        ax = axes[3, j]
        ax.plot(steps, s.get("lr", np.full_like(g, np.nan)), color="C0")
        ax.set_ylabel("lr")
        ax.set_xlabel("optimizer step")
        # Shade contiguous bursts (gap > 200 steps starts a new one); isolated single spikes are not bursts.
        clusters = np.split(burst, np.flatnonzero(np.diff(burst) > 200) + 1) if burst.size else []
        for c in clusters:
            if c.size >= 5:
                for r in range(4):
                    axes[r, j].axvspan(c[0], c[-1] + 1, color="red", alpha=0.12)
        bursts = [f"{c[0]}-{c[-1]}({c.size})" for c in clusters if c.size >= 5]
        w = s.get("w_eff")
        w = f" w_eff {np.nanmean(w):.1f}" if w is not None and np.isfinite(np.nanmean(w)) else ""
        print(
            f"{name:10s}{w} steps>1: {burst.size:4d} bursts {bursts or '-'} max {np.nanmax(g):6.2f} "
            f"| val@end cos {vcos[-1]:.3f} flow {vflow[-1]:.4f} | val best-cos step {vs[int(np.argmax(vcos))]}"
        )
    fig.suptitle(
        "Git-10M alignment runs: train/val curves from stats.npz + val.jsonl (red = grad-norm burst > 1)",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("->", a.out)


if __name__ == "__main__":
    main()
