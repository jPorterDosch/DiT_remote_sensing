"""Gradient geometry of the two objectives (2026-10-28): results/figs/grad_geometry/*.

1. heatmaps (stats.npz only, no GPU): cos(g_FM, g_SEM) per block x noise level at the measurement
   snapshots (init, every 2000 steps, final; 8 fixed batches, aggregated per block = cosine of the
   per-block SUMS over the batches, `meas/<tag>/t<t>/agg_cos`), one panel per snapshot, one figure
   per run (proj / naive / fm seed 42), plus a through-training line plot per t of the block-group
   means (blocks 0-18 double, 19-56 single) and the global row.
2. gram (grad_geometry_s<step>.npz from tasks/train_diffusion._grad_geometry): per block and
   pooled, (a) the cross-t / cross-objective cosine matrix of the per-t MEAN gradients (fm@t_i,
   sem@t_j), (b) the per-batch coherence within (objective, t), (c) a PCA of the per-batch gradient
   cloud from the Gram matrix (kernel PCA on the centred Gram), scatter coloured by objective and
   shaded by t. Panels: block 27 (the last block before the tap), blocks 5-10 POOLED (sum of the per-block Gram
   matrices, i.e. the cosine of the 6-block concatenated gradient; NOT the mean of per-block cosines used by
   the 8-batch through-training 'blocks 5-10' group),
   the pooled Gram. Writes grad_geometry_summary.json with every number plotted.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "results/figs/grad_geometry"
os.makedirs(OUT, exist_ok=True)
RUNS = {  # seed-42 arms of the main comparison (phase 2)
    "proj": "git10m_flux_ca8e6ce7+42+proj-ffcd7963",
    "naive": "git10m_flux_ca8e6ce7+42+naive-11e56b1e",
    "fm": "git10m_flux_ca8e6ce7+42+fm-8c098506",
}
N_DOUBLE = 19
N_SEM = 28  # the semantic loss reads the tokens ENTERING block 28: LoRA blocks 28-56 get no semantic gradient
SUMMARY: dict = {}


def snapshot_tags(s) -> list[str]:
    tags = sorted({k.split("/")[1] for k in s.files if k.startswith("meas/")})

    def key(t):
        return -1 if t == "init" else 10**9 if t == "final" else int(t[4:])

    return sorted(tags, key=key)


def heatmaps() -> None:
    for arm, run in RUNS.items():
        p = f"models/git10m/{run}/stats.npz"
        if not os.path.exists(p):
            print(f"{arm}: {p} missing, skipped")
            continue
        s = np.load(p, mmap_mode="r", allow_pickle=True)
        ts = [int(t) for t in s["meas_t"]]
        tags = snapshot_tags(s)
        blocks = [int(b) for b in s["blocks"]][:-1]  # last row = global
        if len(blocks) != 57:
            raise SystemExit(f"{arm}: {len(blocks)} blocks in stats, expected 57")
        cube = np.full((len(tags), len(ts), len(blocks) + 1), np.nan)
        for i, tag in enumerate(tags):
            for j, t in enumerate(ts):
                cube[i, j] = np.asarray(s[f"meas/{tag}/t{t}/agg_cos"])
        show = [tags[0], tags[len(tags) // 2], tags[-1]]
        fig, axes = plt.subplots(
            1, len(show), figsize=(4.2 * len(show), 6.5), sharey=True, constrained_layout=True
        )
        for ax, tag in zip(np.atleast_1d(axes), show):
            m = cube[tags.index(tag), :, :N_SEM].T  # blocks x t; blocks >= 28 get no semantic gradient (NaN)
            im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6, origin="lower")
            ax.set_xticks(range(len(ts)))
            ax.set_xticklabels(ts)
            ax.set_xlabel("noise level t")
            ax.axhline(N_DOUBLE - 0.5, color="k", lw=0.6, ls="--")
            ax.set_title(f"{arm} s42, {tag}")
        np.atleast_1d(axes)[0].set_ylabel(
            "block (0-18 double, 19-27 single; 28-56 receive no semantic gradient)"
        )
        fig.colorbar(im, ax=axes, shrink=0.6, label="cos(g_FM, g_SEM), 8-batch aggregate")
        fig.savefig(f"{OUT}/cos_block_t_{arm}.png", dpi=150)
        plt.close(fig)
        # through training: block-group means per t
        steps = [0 if t == "init" else 20000 if t == "final" else int(t[4:]) for t in tags]
        groups = {
            "double 0-18": slice(0, N_DOUBLE),
            "single 19-27": slice(N_DOUBLE, N_SEM),
            "blocks 5-10": slice(5, 11),
        }
        fig, axes = plt.subplots(
            1, len(groups) + 1, figsize=(4 * (len(groups) + 1), 3.4), sharey=True, constrained_layout=True
        )
        rec = {}
        for ax, (g, sl) in zip(axes, list(groups.items()) + [("global", None)]):
            for j, t in enumerate(ts):
                y = cube[:, j, -1] if sl is None else np.nanmean(cube[:, j, sl], axis=1)
                ax.plot(
                    steps, y, marker="o", ms=3, label=f"t={t}", color=plt.cm.viridis(j / max(len(ts) - 1, 1))
                )
                rec[f"{g}/t{t}"] = {str(st): float(v) for st, v in zip(steps, y)}
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(g)
            ax.set_xlabel("step")
        axes[0].set_ylabel("cos(g_FM, g_SEM)")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{arm} s42: gradient agreement through training")
        fig.savefig(f"{OUT}/cos_through_training_{arm}.png", dpi=150)
        plt.close(fig)
        SUMMARY[f"heatmap/{arm}"] = {
            "tags": tags,
            "t": ts,
            "block_x_t_at": {tag: cube[tags.index(tag), :, :N_SEM].T.round(4).tolist() for tag in show},
            "through_training": rec,
        }
        print(f"{arm}: heatmaps from {len(tags)} snapshots -> {OUT}/cos_block_t_{arm}.png")


def _mean_cos_matrix(
    G: np.ndarray, obj: np.ndarray, t: np.ndarray, ts: list[int]
) -> tuple[np.ndarray, list[str]]:
    """Cosines between the per-(objective, t) MEAN gradients, from the Gram matrix of the per-batch
    vectors: <mean_a, mean_b> = mean of the G block (a, b)."""
    labels, idx = [], []
    for o, name in ((0, "fm"), (1, "sem")):
        for tt in ts:
            labels.append(f"{name}@{tt}")
            idx.append(np.where((obj == o) & (t == tt))[0])
    n = len(labels)
    dots = np.array([[G[np.ix_(idx[a], idx[b])].mean() for b in range(n)] for a in range(n)])
    norms = np.sqrt(np.clip(np.diag(dots), 1e-300, None))
    return dots / np.outer(norms, norms), labels


def _coherence(G: np.ndarray, idx: np.ndarray) -> float:
    """Mean pairwise cosine of the per-batch gradients of one (objective, t) (direction only)."""
    sub = G[np.ix_(idx, idx)]
    d = np.sqrt(np.clip(np.diag(sub), 1e-300, None))
    c = sub / np.outer(d, d)
    m = len(idx)
    return float((c.sum() - np.trace(c)) / (m * (m - 1))) if m > 1 else float("nan")


def _kpca(G: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """2-d coordinates of the per-batch gradient vectors from their Gram matrix (double centring,
    top-k eigenvectors), plus the explained-variance fractions. Vectors are first scaled to unit
    norm so the picture is about direction, not the fm / sem magnitude ratio."""
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    C = G / np.outer(d, d)
    n = C.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    K = J @ C @ J
    w, v = np.linalg.eigh(K)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    coords = v[:, :k] * np.sqrt(np.clip(w[:k], 0, None))
    return coords, w[:k] / max(w[w > 0].sum(), 1e-300)


def gram() -> None:
    files = sorted(glob.glob("models/git10m/*/grad_geometry_s*.npz"))
    if not files:
        print("no grad_geometry_s*.npz yet; gram figures skipped")
        return
    for f in files:
        d = np.load(f, allow_pickle=True)
        run = f.split("/")[-2]
        arm, step = str(d["arm"]), int(d["step"])
        if not bool(d["gate"]):
            raise SystemExit(f"{f}: gate FAIL recorded, refusing")
        obj, t, ts = d["obj"], d["t"], [int(x) for x in d["meas_t"]]
        blocks = [int(b) for b in d["blocks"]]
        # blocks >= 28 carry no semantic gradient (the feature is the input of block 28): pool over 0-27 only
        sem_blocks = [blocks.index(b) for b in blocks if b < N_SEM]
        panels = {
            "block 27 (last before the tap)": d["gram"][blocks.index(27)],
            "blocks 5-10 (sum)": d["gram"][[blocks.index(b) for b in range(5, 11)]].sum(0),
            "blocks 0-27 (pooled)": d["gram"][sem_blocks].sum(0),
        }
        sem_norm_hi = np.sqrt(
            np.diag(d["gram"][[blocks.index(b) for b in blocks if b >= N_SEM]].sum(0))[obj == 1]
        ).max()
        if sem_norm_hi > 0:
            raise SystemExit(
                f"{f}: semantic gradient reaches blocks >= {N_SEM} (max norm {sem_norm_hi:.3g}); the tap moved?"
            )
        tag = f"{arm}_s{d['seed']}_step{step}"
        rec = {
            "file": f,
            "run": run,
            "arm": arm,
            "step": step,
            "n_batches": int(d["n_batches"]),
            "t": ts,
            "panels": {},
        }
        # (a) cosine matrices of the per-t means
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), constrained_layout=True)
        for ax, (name, G) in zip(axes, panels.items()):
            M, labels = _mean_cos_matrix(G, obj, t, ts)
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            nt = len(ts)
            ax.axhline(nt - 0.5, color="k", lw=0.6)
            ax.axvline(nt - 0.5, color="k", lw=0.6)
            ax.set_title(name, fontsize=10)
            coh = {
                f"{'fm' if o == 0 else 'sem'}@{tt}": _coherence(G, np.where((obj == o) & (t == tt))[0])
                for o in (0, 1)
                for tt in ts
            }
            rec["panels"][name] = {
                "mean_cos": M.round(4).tolist(),
                "labels": labels,
                "coherence": {k: round(v, 4) for k, v in coh.items()},
                "cos_fm_sem_same_t": {str(tt): float(M[i, nt + i]) for i, tt in enumerate(ts)},
            }
        fig.colorbar(im, ax=axes, shrink=0.7, label="cosine of per-t mean gradients")
        fig.suptitle(
            f"{arm} s{d['seed']} step {step}: FM vs semantic LoRA gradients across noise levels ({int(d['n_batches'])} batches)"
        )
        fig.savefig(f"{OUT}/cross_t_cos_{tag}.png", dpi=150)
        plt.close(fig)
        # (c) kernel PCA scatter of the per-batch cloud
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), constrained_layout=True)
        for ax, (name, G) in zip(axes, panels.items()):
            xy, ev = _kpca(G)
            for o, mk, lab in ((0, "o", "FM"), (1, "^", "semantic")):
                sel = obj == o
                sc = ax.scatter(
                    xy[sel, 0],
                    xy[sel, 1],
                    c=t[sel],
                    cmap="viridis",
                    marker=mk,
                    s=18,
                    alpha=0.8,
                    vmin=min(ts),
                    vmax=max(ts),
                    label=lab,
                    edgecolors="k" if o else "none",
                    linewidths=0.3,
                )
            ax.set_title(f"{name}  (PC1 {100 * ev[0]:.0f} %, PC2 {100 * ev[1]:.0f} %)", fontsize=10)
            ax.set_xlabel("PC1")
            ax.legend(fontsize=8)
            rec["panels"][name]["pca_explained"] = [float(x) for x in ev]
        axes[0].set_ylabel("PC2")
        fig.colorbar(sc, ax=axes, shrink=0.7, label="noise level t")
        fig.suptitle(
            f"{arm} s{d['seed']} step {step}: per-batch unit gradients, kernel PCA of the Gram matrix"
        )
        fig.savefig(f"{OUT}/pca_{tag}.png", dpi=150)
        plt.close(fig)
        SUMMARY[f"gram/{tag}"] = rec
        p = rec["panels"]["blocks 0-27 (pooled)"]
        print(
            f"{tag}: cos(fm,sem) same t {p['cos_fm_sem_same_t']}; coherence {p['coherence']} -> {OUT}/cross_t_cos_{tag}.png, pca_{tag}.png"
        )


if __name__ == "__main__":
    heatmaps()
    gram()
    json.dump(SUMMARY, open(f"{OUT}/grad_geometry_summary.json", "w"), indent=1)
    print("GRAD_GEOMETRY_FIGS_OK", sorted(os.listdir(OUT)))
