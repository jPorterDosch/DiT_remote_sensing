"""Mentor-meeting figures from artifacts already on disk (no compute: numpy + matplotlib, seconds).

    python experiments/figs/mentor_figs.py            # -> results/figs/*.png

Inputs: results/matrix/cells/<model>/<dataset>/f<frac>_s<seed>.npz (key score); models/git10m/<run>/
stats.npz + meta.json (gradient-alignment statistics logged every optimizer step by tasks/align.py);
results/eval/sweep_mb/sweep-select_*tsw-reg*.npz (trained slot weights); logs/git10m-gen-eval-*.out
([gen-eval] FID lines). Every number on a figure is read from one of these, never typed in.
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "results/figs"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["m_eurosat", "m_brick_kiln", "m_pv4ger", "m_so2sat", "m_forestnet", "m_bigearthnet"]
MODELS = {  # matrix model -> (label, color, linestyle)
    "dinov3-web": ("DINOv3-web 7B (reference)", "0.6", ":"),
    "dinov3-sat": ("DINOv3-sat 7B (teacher)", "k", "--"),
    "flux-pretrained": ("frozen FLUX, own best block", "tab:gray", "-"),
    "flux-pretrained-k28": ("frozen FLUX, k=28", "tab:blue", "-"),
    "flux-fm": ("fm (LoRA, FM loss only)", "tab:orange", "-"),
    "flux-naive": ("naive (FM + semantic)", "tab:green", "-"),
    "flux-proj": ("proj (FM + projected semantic)", "tab:red", "-"),
    "flux-proj-shufbatch": ("shuffled-teacher control", "tab:purple", "-."),
}


# (model, dataset) cells that must not be plotted: the shuffle control's pv4ger cache holds one
# train image with NaN features at t=260, so its cells crash whenever that image is sampled and
# the surviving cells are an incomplete seed set (audit 2026-10-21, logs/nan_audit-6448007.out).
EXCLUDE = {("flux-proj-shufbatch", "m_pv4ger")}


N_SEEDS = 3  # the matrix protocol: seeds 0/1/2 below 100 %, one fit at 100 %


def savefig(fig, name, **kw):
    """NOTITLE=1: hide the suptitle (paper caption carries it) and write <name>_notitle.png."""
    if os.environ.get("NOTITLE"):
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        name = f"{name}_notitle"
    fig.savefig(f"{OUT}/{name}.png", **kw)


def cells(model: str, dataset: str) -> dict[float, list[float]]:
    """Per-fraction test scores (x100) of one (model, dataset) from the matrix cells. Review
    2026-10-27: a partially finished arm (fewer seeds than the protocol) is refused, not averaged
    and titled '3 seeds'; an arm with no cells at all is skipped (returns {}), as before."""
    out: dict[float, list[float]] = {}
    if (model, dataset) in EXCLUDE:
        return out
    for p in glob.glob(f"results/matrix/cells/{model}/{dataset}/f*_s*.npz"):
        f = float(re.search(r"f([\d.]+)_s", p).group(1))
        out.setdefault(f, []).append(float(np.load(p, allow_pickle=True)["score"]) * 100)
    for f, v in out.items():
        want = 1 if f == 1.0 else N_SEEDS
        if len(v) != want:
            raise SystemExit(
                f"{model} {dataset} f={f:g}: {len(v)} cells, protocol has {want} -- arm incomplete, not plotted"
            )
    return out


def fig_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, ds in zip(axes.ravel(), DATASETS):
        for m, (lab, col, ls) in MODELS.items():
            c = cells(m, ds)
            if not c:
                continue
            fr = sorted(c)
            mu = [np.mean(c[f]) for f in fr]
            sd = [
                np.std(c[f], ddof=1) if len(c[f]) > 1 else 0.0 for f in fr
            ]  # sample std, as eval/matrix.py summarize
            ax.errorbar(fr, mu, yerr=sd, label=lab, color=col, ls=ls, marker="o", ms=3, capsize=2)
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.05, 0.1, 0.25, 1.0])
        ax.set_xticklabels(["1%", "5%", "10%", "25%", "100%"])
        ax.set_title(ds.replace("m_", ""))
        ax.set_ylabel("micro-F1 (%)" if ds == "m_bigearthnet" else "test top-1 (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Label efficiency, GEO-Bench test, mean +- std over 3 seeds (100 %: 1 seed)")
    fig.tight_layout()
    savefig(fig, "label_efficiency", dpi=150)


def fig_gap_closed():
    arms = ["flux-fm", "flux-naive", "flux-proj"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, which in zip(axes, ["low-label mean (1/5/10/25 %)", "100 % labels"]):
        w = 0.25
        for i, arm in enumerate(arms):
            vals = []
            for ds in DATASETS:

                def stat(m):
                    c = cells(m, ds)
                    if which.startswith("100"):
                        return np.mean(c[1.0]) if 1.0 in c else np.nan
                    fr = [f for f in c if f < 1.0]
                    return np.mean([np.mean(c[f]) for f in fr]) if fr else np.nan

                a, fz, t = stat(arm), stat("flux-pretrained-k28"), stat("dinov3-sat")
                # the ratio is meaningless when teacher and frozen are within a point of each other
                vals.append(
                    (a - fz) / (t - fz) if np.isfinite([a, fz, t]).all() and abs(t - fz) >= 1.0 else np.nan
                )
            ax.bar(
                np.arange(len(DATASETS)) + (i - 1) * w, vals, w, label=MODELS[arm][0], color=MODELS[arm][1]
            )
            for j, v in enumerate(vals):
                if not np.isfinite(v) and i == 1:
                    ax.text(
                        j,
                        0.05,
                        "n/a: teacher\nwithin 1 pt\nof frozen",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="0.4",
                    )
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([d.replace("m_", "") for d in DATASETS], rotation=20)
        ax.set_title(f"teacher gap closed, {which}")
        ax.set_ylabel("(arm - frozen k28) / (teacher - frozen k28)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "gap_closed", dpi=150)


def run_label(run: str) -> tuple[str, str]:
    """(label, group) from the run's meta.json align block, never from the hash."""
    al = json.load(open(f"models/git10m/{run}/meta.json"))["align"]
    seed = re.search(r"\+(\d+)\+", run).group(1)
    tags = [al["arm"]]
    if al.get("target_shuffle"):
        tags.append("shuffled-teacher")
    if al.get("feat") != "ditf":
        tags.append(f"feat={al['feat']}")
    if not al.get("discard_channels"):
        tags.append("cd0")
    tgt = al.get("target", {}).get("preset", "")
    if tgt and not tgt.endswith("_sat"):
        tags.append(tgt)
    group = "shuffle" if al.get("target_shuffle") else al["arm"]
    return " ".join(tags) + f" s{seed}", group


GROUP_COLOR = {
    "proj": "tab:red",
    "naive": "tab:green",
    "fm": "tab:orange",
    "shuffle": "tab:purple",
    "sem": "tab:blue",
}


def load_runs():
    runs = sorted(
        os.path.basename(p)
        for p in glob.glob("models/git10m/git10m_flux_*")
        if os.path.isfile(p + "/stats.npz")
    )
    return [(r, *run_label(r), np.load(f"models/git10m/{r}/stats.npz")) for r in runs]


def ema(x, n=200):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    acc = np.nan
    a = 2 / (n + 1)
    for i, v in enumerate(x):
        if np.isfinite(v):
            acc = v if not np.isfinite(acc) else (1 - a) * acc + a * v
        out[i] = acc
    return out


def fig_alignment_vs_step(runs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for r, lab, grp, s in runs:
        g = s["step/cos_pre"][:, -1]
        cf = s["step/conflict"][:, -1].astype(float)
        lw = 1.8 if ("s42" in lab and lab.count(" ") == 1) or grp == "shuffle" else 0.9
        axes[0].plot(ema(g), label=lab, color=GROUP_COLOR[grp], lw=lw, alpha=0.9 if lw > 1 else 0.6)
        axes[1].plot(ema(cf, 500), label=lab, color=GROUP_COLOR[grp], lw=lw, alpha=0.9 if lw > 1 else 0.6)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("cos(g_FM, g_semantic), global over LoRA, EMA 200 steps")
    axes[1].set_ylabel("fraction of steps with cos < 0, EMA 500 steps")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Flow-matching vs semantic gradient alignment during training (stats.npz step/cos_pre, step/conflict)"
    )
    fig.tight_layout()
    savefig(fig, "grad_alignment_vs_step", dpi=150)


def fig_alignment_per_block(runs):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for r, lab, grp, s in runs:
        cp = s["step/cos_pre"][:, :-1]
        m = np.nanmean(cp, axis=0)
        ok = np.isfinite(m)
        lw = 1.8 if ("s42" in lab and lab.count(" ") == 1) or grp == "shuffle" else 0.9
        ax.plot(
            np.flatnonzero(ok),
            m[ok],
            marker="o",
            ms=3,
            label=lab,
            color=GROUP_COLOR[grp],
            lw=lw,
            alpha=0.9 if lw > 1 else 0.6,
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("FLUX block (LoRA blocks the k=28 alignment loss reaches)")
    ax.set_ylabel("mean cos(g_FM, g_semantic) over 20000 steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Per-block gradient alignment")
    fig.tight_layout()
    savefig(fig, "grad_alignment_per_block", dpi=150)


def fig_alignment_per_t(runs):
    pick = [x for x in runs if x[1] in ("proj s42", "naive s42", "fm s42") or x[2] == "shuffle"]
    fig, axes = plt.subplots(1, len(pick), figsize=(4 * len(pick), 4.2), squeeze=False)
    vmax = 0.4
    for ax, (r, lab, grp, s) in zip(axes[0], pick):
        tags = []
        for k in s.files:
            if k.startswith("meas/") and k.split("/")[1] not in tags:
                tags.append(k.split("/")[1])
        ts = [int(t) for t in s["meas_t"]]
        M = np.array([[float(s[f"meas/{tag}/t{t}/agg_cos"][-1]) for t in ts] for tag in tags])
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"t{t}" for t in ts], fontsize=7)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels([t.replace("step", "") for t in tags], fontsize=7)
        ax.set_title(lab, fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.8, label="global cos(g_FM, g_semantic), 8 measurement batches"
    )
    fig.suptitle("Gradient alignment by noise level over training (rows: measurement checkpoint)")
    savefig(fig, "grad_alignment_per_t", dpi=150, bbox_inches="tight")


def fig_slot_weights():
    arms = ["flux-fm", "flux-naive", "flux-proj"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, ds in zip(axes.ravel(), DATASETS):
        w = 0.27
        for i, arm in enumerate(arms):
            ps = glob.glob(f"results/eval/sweep_mb/sweep-select_{ds}_{arm}-k28-tsw-reg_*.npz")
            if not ps:
                continue
            info = json.loads(str(np.load(ps[0], allow_pickle=True)["info"]))["selected"]
            slots, wts = info["slots"], info["slot_weights"]
            ax.bar(
                np.arange(len(slots)) + (i - 1) * w,
                wts,
                w,
                label=f"{MODELS[arm][0]} (lr={info['lr']}, wd={info['wd']})",
                color=MODELS[arm][1],
            )
            ax.set_xticks(range(len(slots)))
            ax.set_xticklabels(slots, fontsize=7)
        ax.set_title(ds.replace("m_", ""))
        ax.set_ylabel("learned softmax weight")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("tsweight head: learned timestep weights at block 28 (val-selected lr / wd / epoch)")
    fig.tight_layout()
    savefig(fig, "tsweight_slot_weights", dpi=150)


GEN_EVAL_ROOT = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/models/git10m_gen_eval/n5000_h100_g1.0"


def fid_table() -> dict[str, tuple[float, float]]:
    """FID / KID per run against the OVERLAPPING reference (the gen-eval log's own number: every
    100th row of index_retained, a subset of the training rows). Labelled as such in the figure and
    in fid_table.json next to the geographically held-out score (fid_table_geoheldout)."""
    out = {}
    for f in sorted(glob.glob("logs/git10m-gen-eval-*.out")):
        txt = open(f, errors="replace").read()
        m = re.search(r"\[gen-eval\] n=5000 FID=([\d.]+) KID=([\d.]+)", txt)
        run = re.search(r"git10m_flux_[0-9a-f]{8}\+\d+\+[a-z]+-[0-9a-f]{8}", txt)
        if m:
            key = run.group(0) if run else "pretrained"
            out[key] = (float(m.group(1)), float(m.group(2)))
    return out


def fid_table_geoheldout(keys) -> dict[str, tuple[float, float]]:
    """The same runs scored against the geographically held-out reference (rescore_fid.py's per-run
    `gen_metrics_geoheldout.json`, keyed by the lora checkpoint's run name). Missing for a run ->
    refused: the two columns of fid_table.json must cover the same runs."""
    by_ck = {}
    for p in glob.glob(f"{GEN_EVAL_ROOT}/*/gen_metrics_geoheldout.json"):
        j = json.load(open(p))
        ck = (j.get("checkpoint") or {}).get("lora_checkpoint") or "pretrained"
        m = re.search(r"git10m_flux_[0-9a-f]{8}\+\d+\+[a-z]+-[0-9a-f]{8}", ck)
        by_ck[m.group(0) if m else "pretrained"] = (float(j["fid"]), float(j["kid_mean"]))
    missing = [k for k in keys if k not in by_ck]
    if missing:
        raise SystemExit(
            f"no gen_metrics_geoheldout.json for {missing} under {GEN_EVAL_ROOT} -- run rescore_fid.py first"
        )
    return {k: by_ck[k] for k in keys}


def fig_fid_vs_acc(runs):
    fids = fid_table()
    model_of = {"pretrained": "flux-pretrained-k28"}
    for r, lab, grp, s in runs:
        if lab == "fm s42":
            model_of[r] = "flux-fm"
        elif lab == "naive s42":
            model_of[r] = "flux-naive"
        elif lab == "proj s42":
            model_of[r] = "flux-proj"
        elif grp == "shuffle":
            model_of[r] = "flux-proj-shufbatch"
    ds5 = [d for d in DATASETS if d != "m_bigearthnet"]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = {r: lab for r, lab, _, _ in runs}
    for key, (fid, kid) in fids.items():
        m = model_of.get(key)
        acc = [np.mean(cells(m, d)[1.0]) for d in ds5 if m and 1.0 in cells(m, d)] if m else []
        lab = labels.get(key, key if key == "pretrained" else key.split("+")[-1])
        x = 100.0 / fid  # 100/FID: right = better generation
        if m and len(acc) == len(ds5):
            ax.scatter(x, np.mean(acc), s=60, color=MODELS[m][1], zorder=3)
            ax.annotate(
                f"{lab} (FID {fid:.1f})",
                (x, np.mean(acc)),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        else:
            ax.axvline(x, color="0.8", lw=0.8)
            ax.annotate(
                lab + f" (FID {fid:.1f})",
                (x, ax.get_ylim()[0]),
                rotation=90,
                fontsize=6,
                color="0.5",
                xytext=(2, 2),
                textcoords="offset points",
            )
    ax.set_xlabel("100 / FID, 5000 samples, g=1.0, OVERLAPPING reference (right = better generation)")
    ax.set_ylabel("mean 100 %-label test accuracy over 5 single-label datasets")
    ax.set_title("Generation kept vs representation gained (seed 42)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, "fid_vs_accuracy", dpi=150)
    geo = fid_table_geoheldout(list(fids))
    with open(f"{OUT}/fid_table.json", "w") as fh:
        json.dump(
            {
                "_reference": {
                    "overlap_ref": "every 100th row of index_retained (training rows), the gen-eval log number",
                    "geoheldout_ref": "excluded-city rows, disjoint from training (rescore_fid.py)",
                },
                "overlap_ref": {labels.get(k, k): v for k, v in fids.items()},
                "geoheldout_ref": {labels.get(k, k): v for k, v in geo.items()},
            },
            fh,
            indent=1,
        )


MAIN = "/nfs/home/jdosch1/COSC424/DiT_remote_sensing"  # frozen multi-block sweep lives in the main checkout


def best_val_per_candidate(path: str) -> dict[str, float]:
    """val accuracy of the best C per candidate (t<T> singles + concat9t) from one sweep-block file."""
    z = np.load(path, allow_pickle=True)
    out: dict[str, float] = {}
    for r in json.loads(str(z["val_table"])):
        out[r["candidate"]] = max(out.get(r["candidate"], -1), r["val_acc"])
    return out


def fig_per_timestep():
    """Per-timestep linear-probe VAL accuracy at block 28, wide C grid, 100 % labels: frozen FLUX
    (main checkout sweep, flux-oneshot-ens8-mb) vs the adapted arms (flux-<arm>-k28-lrw)."""
    srcs = {
        "flux-pretrained-k28": f"{MAIN}/results/eval/sweep_mb/sweep-block_{{ds}}_flux-oneshot-ens8-mb-k28_*.npz",
        "flux-fm": "results/eval/sweep_mb/sweep-block_{ds}_flux-fm-k28-lrw-k28_*.npz",
        "flux-naive": "results/eval/sweep_mb/sweep-block_{ds}_flux-naive-k28-lrw-k28_*.npz",
        "flux-proj": "results/eval/sweep_mb/sweep-block_{ds}_flux-proj-k28-lrw-k28_*.npz",
    }
    dss = [d for d in DATASETS if d != "m_bigearthnet"]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 4))
    for ax, ds in zip(axes, dss):
        for m, pat in srcs.items():
            ps = glob.glob(pat.format(ds=ds))
            if not ps:
                continue
            v = best_val_per_candidate(ps[0])
            ts = sorted(int(k[1:]) for k in v if k.startswith("t"))
            lab, col, _ = MODELS[m]
            ax.plot(ts, [v[f"t{t}"] * 100 for t in ts], marker="o", ms=3, color=col, label=lab)
            ax.axhline(v["concat9t"] * 100, color=col, ls=":", lw=0.9)
        ax.set_title(ds.replace("m_", ""))
        ax.set_xlabel("timestep t (higher = more noise)")
        ax.set_ylabel("val top-1 (%), best C per cell")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.suptitle(
        "Per-timestep linear probe at block 28 (dotted: 9-t concat), official val split, wide C grid"
    )
    fig.tight_layout()
    savefig(fig, "per_timestep_probe", dpi=150)


if __name__ == "__main__":
    runs = load_runs()
    print("runs:", [(r, lab) for r, lab, _, _ in runs])
    fig_curves()
    fig_gap_closed()
    fig_alignment_vs_step(runs)
    fig_alignment_per_block(runs)
    fig_alignment_per_t(runs)
    fig_slot_weights()
    fig_fid_vs_acc(runs)
    fig_per_timestep()
    print("FIGS_OK", sorted(os.listdir(OUT)))
