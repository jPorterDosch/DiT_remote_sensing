"""Paper figures from artifacts on disk (numpy + matplotlib + PIL, seconds; run as a short CPU job).
    python experiments/figs/paper_figs.py            # -> results/figs/{seg_curves,fid_vs_accuracy_heldout,gen_grid_column}.{png,pdf}
    python experiments/figs/paper_figs.py null trade # -> results/figs/{gen_grid_null_3x4,fid_vs_so2sat,fid_vs_so2sat_mean5}.{png,pdf,json}
    python experiments/figs/paper_figs.py rs         # -> results/figs/gen_grid_rs_prompts.{png,pdf,svg,json}
Inputs (every number is read, never typed):
  $STORE/models/prompt_gen_rs/<tag>_<list>_n<N>_<px>px_g<G>_s<steps>/   prompt_gen.py --prompts runs (rs)
  results/seg/phase3_lr/summary.json                       segmentation label-efficiency (mIoU)
  results/matrix/cells/<model>/<dataset>/f1.0_s0.npz       100 %-label test accuracy (mentor_figs.cells)
  results/figs/fid_table.json  (geoheldout_ref)            FID of the FLUX arms vs the held-out reference
  $STORE/baselines/dsat_samples/null_n5000/gen_metrics_geoheldout.json   DiffusionSat FID, same reference
  SatDiFuser Table 2 (arXiv 2503.07890 v2; their protocol, 100 % labels, trained fuser + linear head)
  models/git10m_gen_eval/n5000_h100_g1.0/<hash>/samples/NNNNN.png        fixed-seed samples per arm
"""

from __future__ import annotations
import glob
import hashlib
import json
import os
import sys
import textwrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mentor_figs import DATASETS, MODELS, cells  # noqa: E402  (chdir's to the repo root)

OUT = "results/figs"
STORE = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing"
GEN_EVAL_ROOT = f"{STORE}/models/git10m_gen_eval/n5000_h100_g1.0"
DSAT_NULL = "/lustre/isaac24/scratch/jdosch1/baselines/dsat_samples/null_n5000"
REAL_REF = f"{STORE}/git10m/fid_ref_geoheldout_n5000"
# SatDiFuser Table 2, classification, top-1 / micro-F1 (%), columns ben / bk / es / fn / pv / s2s; their
# protocol: full labelled train split, trained fuser + linear head, flip / jitter augmentation.
# Quoted as a reference point only (memory satdifuser-reference, verified from the PDF 2026-10-14).
SATDIFUSER_T2 = {
    "gw": {
        "m_bigearthnet": 66.2,
        "m_brick_kiln": 98.1,
        "m_eurosat": 97.7,
        "m_forestnet": 57.1,
        "m_pv4ger": 97.4,
        "m_so2sat": 59.3,
    },
    "lw": {
        "m_bigearthnet": 64.8,
        "m_brick_kiln": 97.7,
        "m_eurosat": 96.8,
        "m_forestnet": 56.2,
        "m_pv4ger": 97.7,
        "m_so2sat": 58.9,
    },
    "moe": {
        "m_bigearthnet": 65.8,
        "m_brick_kiln": 98.3,
        "m_eurosat": 97.3,
        "m_forestnet": 58.4,
        "m_pv4ger": 97.8,
        "m_so2sat": 58.8,
    },
}
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
    }
)


def save(fig, name, svg=False, dpi=300):
    # NOTITLE=1: drop the figure-level suptitle (the paper caption carries it) and write *_notitle files;
    # per-panel titles stay (they label panels, not the figure).
    # svg=True: also write an SVG (image grids: tiles embed as base64 PNG at their own pixel size).
    if os.environ.get("NOTITLE"):
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        name = f"{name}_notitle"
    fig.savefig(f"{OUT}/{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    exts = "png,pdf"
    if svg:
        fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
        exts += ",svg"
    print(f"-> {OUT}/{name}.{{{exts}}}")


# ---------------------------------------------------------------- 1. segmentation curves
SEG_TASKS = [
    ("m_pv4ger_seg", "m-pv4ger-seg (2 classes)"),
    ("m_chesapeake_landcover", "m-chesapeake (7 classes)"),
    ("m_SA_crop_type", "m-SA-crop-type (9 classes)"),
]
SEG_MODELS = {  # summary.json row -> (label, color, linestyle); frozen = block-28 row (matched to the adapted arms)
    "dinov3-web": MODELS["dinov3-web"],
    "dinov3-sat": MODELS["dinov3-sat"],
    "flux-pretrained-k28": MODELS["flux-pretrained-k28"],
    "flux-naive": MODELS["flux-naive"],
    "flux-proj": MODELS["flux-proj"],
}


def fig_seg_curves(summary="results/seg/phase3_lr/summary.json"):
    d = json.load(open(summary))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, (task, title) in zip(axes, SEG_TASKS):
        aulc = []
        for m, (lab, col, ls) in SEG_MODELS.items():
            row = d.get(f"{m}|{task}")
            if row is None:
                raise SystemExit(f"{summary}: no row {m}|{task}")
            pf = row["per_fraction"]
            fr = sorted(float(f) for f in pf)
            mu = [100 * pf[_key(pf, f)]["mean"] for f in fr]
            sd = [100 * (pf[_key(pf, f)].get("std") or 0.0) for f in fr]
            ax.errorbar(fr, mu, yerr=sd, label=lab, color=col, ls=ls, marker="o", ms=2.5, capsize=1.5, lw=1)
            aulc.append(
                (lab.split(" ")[0] if m.startswith("dinov3") else lab.split(" (")[0], 100 * row["aulc"])
            )
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.05, 0.1, 0.25, 1.0])
        ax.set_xticklabels(["1", "5", "10", "25", "100"], fontsize=5.5)
        ax.set_xticks([], minor=True)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlabel("labelled fraction of the train split (%)")
        short = {
            "DINOv3-web": "web",
            "DINOv3-sat": "sat",
            "frozen FLUX, k=28": "frozen",
            "naive": "naive",
            "proj": "proj",
        }
        ax.text(
            0.98,
            0.03,
            "AULC\n" + "\n".join(f"{short.get(n, n)} {v:.1f}" for n, v in aulc),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5,
            linespacing=1.1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
        )
    axes[0].set_ylabel("test mIoU (%)")
    h, labs = axes[0].get_legend_handles_labels()
    fig.legend(h, labs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(
        "Segmentation label efficiency, GEO-Bench test, linear per-token head, mean $\\pm$ std over 3 label-subset seeds (100 %: 1 fit)",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "seg_curves")


def _key(pf: dict, f: float) -> str:
    for k in pf:
        if abs(float(k) - f) < 1e-9:
            return k
    raise KeyError(f)


# ---------------------------------------------------------------- 2. FID (held-out) vs accuracy, with DiffusionSat
SINGLE5 = [d for d in DATASETS if d != "m_bigearthnet"]
ARMS = {  # label -> (matrix model, label, color, marker, training-run name whose gen-eval dir holds the FID)
    "pretrained": ("flux-pretrained-k28", "FLUX pretrained", MODELS["flux-pretrained-k28"][1], "o", ""),
    "fm s42": ("flux-fm", "fm s42", MODELS["flux-fm"][1], "o", "git10m_flux_ca8e6ce7+42+fm-8c098506"),
    "naive s42": (
        "flux-naive",
        "naive s42",
        MODELS["flux-naive"][1],
        "o",
        "git10m_flux_ca8e6ce7+42+naive-11e56b1e",
    ),
    "proj s42": (
        "flux-proj",
        "proj s42",
        MODELS["flux-proj"][1],
        "o",
        "git10m_flux_ca8e6ce7+42+proj-ffcd7963",
    ),
    "fm s43": ("flux-fm-s43", "fm s43", MODELS["flux-fm"][1], "^", "git10m_flux_ca8e6ce7+43+fm-c0aa0eab"),
    "proj s43": (
        "flux-proj-s43",
        "proj s43",
        MODELS["flux-proj"][1],
        "^",
        "git10m_flux_ca8e6ce7+43+proj-ffcd7963",
    ),
    "naive s43": (
        "flux-naive-s43",
        "naive s43",
        MODELS["flux-naive"][1],
        "^",
        "git10m_flux_ca8e6ce7+43+naive-11e56b1e",
    ),
}


def fid_heldout(run_name: str) -> float:
    """FID vs the geo-held-out reference from the run's gen-eval dir (rescore_fid.py output); refused if
    the dir is missing or ambiguous, so an arm is never silently dropped."""
    return float(json.load(open(f"{gen_run_dir(run_name)}/gen_metrics_geoheldout.json"))["fid"])


REFERENCE_IDENTITY = ("real", "extractor", "n")


def assert_same_reference(name: str, other: dict, held: dict) -> None:
    """GATE (review 2026-11-12, item 11): a baseline's FID may share an axis or a column with the FLUX arms
    only if it was scored against the SAME reference images (count, size, sha1), with the same extractor, on
    the same number of samples of the same size. `other` / `held`: the baseline's and one FLUX arm's
    gen_metrics_geoheldout.json. It does NOT compare how the samples were drawn (steps, guidance, native
    resolution): a baseline has its own sampler, and the caption must state it. Stops the figure."""
    bad = {
        f: (other.get(f), held.get(f))
        for f in REFERENCE_IDENTITY
        if other.get(f) is None or other.get(f) != held.get(f)
    }
    hw = ((other.get("fake") or {}).get("hw"), (held.get("fake") or {}).get("hw"))
    if hw[0] is None or hw[0] != hw[1]:
        bad["fake.hw"] = hw
    if bad:
        raise SystemExit(
            f"GATE reference identity FAIL: {name}: {bad}; do NOT plot this FID beside the FLUX arms"
        )
    print(
        f"GATE reference identity PASS: {name}: n={other['n']}, {hw[0]} px, reference sha1 {other['real']['sha1'][:12]}, {other['extractor']}"
    )


def dsat_null_heldout() -> dict:
    """DiffusionSat null-prompt metrics vs the geo-held-out reference, gated against the seed-42 joint arm."""
    dsat = json.load(open(f"{DSAT_NULL}/gen_metrics_geoheldout.json"))
    held = json.load(
        open(f"{gen_run_dir('git10m_flux_ca8e6ce7+42+proj-ffcd7963')}/gen_metrics_geoheldout.json")
    )
    assert_same_reference("DiffusionSat (null prompt)", dsat, held)
    return dsat


OFFSETS = {  # label offset (points) and alignment per point, so the top-right cluster stays legible
    "pretrained": ((4, 3), "left"),
    "fm s42": ((4, 3), "left"),
    "naive s42": ((-4, 5), "right"),
    "proj s42": ((4, -9), "left"),
    "fm s43": ((4, -3), "left"),
    "proj s43": ((-4, -9), "right"),
    "naive s43": ((4, -9), "left"),
}


def mean_acc100(model: str) -> float | None:
    """mean 100 %-label test accuracy over the 5 single-label datasets, or None if the arm is incomplete."""
    acc = []
    for ds in SINGLE5:
        try:
            c = cells(model, ds)
        except SystemExit as e:  # partially finished arm (phase 6 still landing): not plotted
            print(f"   skip {model}: {e}")
            return None
        if 1.0 not in c:
            return None
        acc.append(np.mean(c[1.0]))
    return float(np.mean(acc))


def fig_fid_vs_acc_heldout():
    """Teaser: FID (held-out reference, log axis, lower = right) vs mean 100 %-label accuracy. No in-plot
    text: arms by colour, seed 43 hollow, DiffusionSat one hollow square (gw / MoE differ by 0.2)."""
    from matplotlib.lines import Line2D

    dsat = dsat_null_heldout()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pts, handles = {}, []
    seen = set()
    for key, (model, lab, col, _mk, run) in ARMS.items():
        y = mean_acc100(model)
        if y is None:
            raise SystemExit(f"{key}: 100 %-label cells incomplete; the teaser must show every arm")
        fid = fid_heldout(run)
        s43 = key.endswith("s43")
        ax.scatter(
            fid, y, s=26, marker="o", facecolor="none" if s43 else col, edgecolor=col, lw=1.0, zorder=3
        )
        pts[lab] = {"fid_heldout": fid, "mean_acc100_single5": y}
        arm = lab.split(" ")[0] if key != "pretrained" else "FLUX pretrained"
        if arm not in seen:
            seen.add(arm)
            handles.append(Line2D([], [], marker="o", color=col, ls="", markersize=4.5, label=arm))
    # DiffusionSat: generation measured here (null prompt, g = 1.0, same reference); representation from
    # SatDiFuser Table 2 (their protocol): gw 81.9 / MoE 82.1 mean over the 5 tasks, plotted as one point.
    fid = float(dsat["fid"])
    ys = {cfg: float(np.mean([SATDIFUSER_T2[cfg][d] for d in SINGLE5])) for cfg in ("gw", "moe")}
    y = float(np.mean(list(ys.values())))
    ax.scatter(fid, y, s=34, marker="s", facecolor="none", edgecolor="tab:brown", lw=1.0, zorder=3)
    pts["DiffusionSat x SatDiFuser"] = {
        "fid_heldout": fid,
        "mean_acc100_single5": y,
        "per_fuser": ys,
        "accuracy_source": "SatDiFuser Table 2, their protocol",
    }
    handles.append(
        Line2D([], [], marker="s", color="tab:brown", mfc="none", ls="", markersize=5, label="DiffusionSat*")
    )
    for m, (lab, col, ls) in (("dinov3-sat", MODELS["dinov3-sat"]), ("dinov3-web", MODELS["dinov3-web"])):
        ym = mean_acc100(m)
        if ym is not None:
            ax.axhline(ym, color=col, ls=ls, lw=0.8, zorder=1)
            handles.append(Line2D([], [], color=col, ls=ls, lw=0.8, label=lab.replace(" 7B", "")))
            pts[lab] = {"mean_acc100_single5": ym}
    handles.append(
        Line2D([], [], marker="o", color="0.3", mfc="none", ls="", markersize=4.5, label="hollow = seed 43")
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ticks = [150, 100, 75, 50, 40]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("FID vs geo-held-out Git-10M (lower is better)")
    ax.set_ylabel("mean test acc. (%)\n5 single-label tasks, 100 % labels")
    ax.set_ylim(74, 83.6)
    ax.grid(alpha=0.3)
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=5.5,
        frameon=True,
        framealpha=0.9,
        handletextpad=0.4,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.6,
    )
    fig.tight_layout()
    save(fig, "fid_vs_accuracy_heldout")
    json.dump(pts, open(f"{OUT}/fid_vs_accuracy_heldout.json", "w"), indent=1)


# ---------------------------------------------------------------- 2b. block x t heatmaps, single column
TAG_LABEL = {"init": "init", "step10000": "10k", "final": "20k"}


def fig_block_t_column(
    panels=(("proj", "init"), ("proj", "step10000"), ("proj", "final")),
    name="cos_block_t_proj_col",
    figsize=(3.4, 2.4),
):
    """cos(g_FM, g_SEM) per (block, timestep index tau) from grad_geometry_summary.json: the 8-fixed-batch
    snapshots of the TRAINING RUN (stats.npz; init, every 2000 steps, final; per block the cosine of the
    batch-summed gradients), NOT the 32-batch checkpoint study behind fig_sem_coherence. figsize (3.4, 2.4)
    = one column; (7.0, 2.6) = figure* full width (name *_wide)."""
    s = json.load(open(f"{OUT}/grad_geometry/grad_geometry_summary.json"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (arm, tag) in zip(axes, panels):
        h = s[f"heatmap/{arm}"]
        M = np.asarray(h["block_x_t_at"][tag], dtype=float)
        t = h["t"]
        im = ax.imshow(
            M, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6, interpolation="nearest"
        )
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([str(x) for x in t], fontsize=5, rotation=90)
        ax.axhline(18.5, color="k", lw=0.5, ls="--")
        ax.set_title(f"{arm} {TAG_LABEL[tag]}", fontsize=6.5, pad=3)
        ax.tick_params(length=2)
    axes[0].set_yticks([0, 5, 10, 15, 20, 25])
    axes[0].set_ylabel("block (0-18 double, 19-27 single)")
    fig.text(0.5, 0.01, "timestep index \u03c4 (t = \u03c4 / 1000)", ha="center", fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.2, top=0.9, wspace=0.12)
    cax = fig.add_axes([0.88, 0.2, 0.025, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cos(g_FM, g_SEM)", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2)
    save(fig, name)


# ---------------------------------------------------------------- 2c. semantic-gradient coherence through training
def fig_sem_coherence(name="sem_coherence_vs_step", stacked=False):
    """Block-27 batch coherence of the semantic gradient (mean pairwise cosine of the 32 per-batch unit
    gradients at one t) vs training step, three arms, from the gram entries of grad_geometry_summary.json.
    The quantity that separates fm (never applies g_SEM) from the joint arms; cos(g_FM, g_SEM) does not."""
    s = json.load(open(f"{OUT}/grad_geometry/grad_geometry_summary.json"))
    panel = "block 27 (last before the tap)"
    arms = {"fm": MODELS["flux-fm"][1], "naive": MODELS["flux-naive"][1], "proj": MODELS["flux-proj"][1]}
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    out = {}
    for ax, (t, ttl) in zip(axes, ((100, "\u03c4 = 100 (low noise)"), (740, "\u03c4 = 740 (high noise)"))):
        for arm, col in arms.items():
            pts = sorted(
                (g["step"], g["panels"][panel]["coherence"][f"sem@{t}"])
                for k, g in s.items()
                if k.startswith("gram/") and g["arm"] == arm
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=1, color=col, label=arm)
            out[f"{arm}/sem@{t}"] = pts
        ax.set_title(ttl, fontsize=7)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.set_xticks([0, 5000, 10000, 15000, 20000])
        ax.set_xticklabels(["0", "5k", "10k", "15k", "20k"])
    if stacked:
        for ax in axes:
            ax.set_ylabel("semantic-gradient\ncoherence, block 27")
        axes[0].set_xlabel("")
    else:
        axes[0].set_ylabel("semantic-gradient\nbatch coherence, block 27")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, name)
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)


# ---------------------------------------------------------------- 3. generation grid, one column
def gen_run_dir(run_substr: str) -> str:
    """gen-eval dir whose gen_metrics.json names this lora checkpoint run ('' = pretrained)."""
    hits = []
    for p in glob.glob(f"{GEN_EVAL_ROOT}/*/gen_metrics.json"):
        ck = (json.load(open(p)).get("checkpoint") or {}).get("lora_checkpoint", "")
        if (run_substr == "" and ck == "") or (run_substr and run_substr in ck):
            hits.append(os.path.dirname(p))
    if len(hits) != 1:
        raise SystemExit(f"gen-eval dir for {run_substr!r}: {hits}")
    return hits[0]


def fig_gen_grid(idx=(0, 1, 2, 3, 4, 5)):
    rows = [
        ("Git-10M held-out (real)", REAL_REF, None, None),
        ("FLUX pretrained", gen_run_dir("") + "/samples", "pretrained", None),
        ("fm (s42)", gen_run_dir("ca8e6ce7+42+fm-8c098506") + "/samples", "fm s42", None),
        ("naive (s42)", gen_run_dir("ca8e6ce7+42+naive-11e56b1e") + "/samples", "naive s42", None),
        ("proj (s42)", gen_run_dir("ca8e6ce7+42+proj-ffcd7963") + "/samples", "proj s42", None),
        (
            "DiffusionSat (null prompt)",
            f"{DSAT_NULL}/samples224",
            None,
            f"{DSAT_NULL}/gen_metrics_geoheldout.json",
        ),
    ]
    dsat_null_heldout()  # GATE: the DiffusionSat FID below is against the same reference as the FLUX rows
    ft = json.load(open(f"{OUT}/fid_table.json"))["geoheldout_ref"]
    n = len(idx)
    tile = 0.5
    fig, axes = plt.subplots(
        len(rows),
        n,
        figsize=(n * tile + 0.05, len(rows) * tile),
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )
    for r, (lab, d, fkey, fjson) in enumerate(rows):
        fid = ft[fkey][0] if fkey else (json.load(open(fjson))["fid"] if fjson else None)
        for c, i in enumerate(idx):
            p = f"{d}/{i:05d}.png"
            im = Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)
            # interpolation="none": PDF / SVG embed the 224 px tile unresampled (zoomable), PNG resamples at save dpi
            ax = axes[r, c]
            ax.imshow(np.asarray(im), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(
            lab + (f"\nFID {fid:.0f}" if fid is not None else ""),
            rotation=0,
            ha="right",
            va="center",
            fontsize=5.5,
            labelpad=4,
        )
    # caption text, not in the figure: FLUX rows share the latent seed per column (seeds 1000-1005), 224 px,
    # g = 1.0, 28 steps; DiffusionSat 512 px resized to 224; FID vs the geo-held-out reference.
    save(
        fig, "gen_grid_column", svg=True, dpi=600
    )  # 600 dpi: ~300 px per 0.5 in tile, above the 224 px source


# ---------------------------------------------------------------- 4. null-prompt grid + FID vs So2Sat (trade-off)
# Reviewer figures: the same fixed-seed null-prompt samples scored for generation (FID) against the
# recognition score of the same checkpoint. Arms carry the names used in the response text.
TRADE_STYLE = {  # arm -> (label, colour, marker); colours are the paper's arm colours, the marker shape repeats the
    # identity (naive green / proj red are a red-green pair). sem has no MODELS entry: its own colour.
    "pretrained": ("Pretrained", MODELS["flux-pretrained-k28"][1], "s"),
    "fm": ("FM-only", MODELS["flux-fm"][1], "^"),
    "sem": ("Semantic-only", "tab:pink", "D"),
    "naive": ("Joint", MODELS["flux-naive"][1], "o"),
    "proj": ("Projected joint", MODELS["flux-proj"][1], "*"),
}
TRADE_ARMS = {  # key -> (matrix model, arm, adaptation seed, training-run substring of the gen-eval checkpoint)
    "pretrained": ("flux-pretrained-k28", "pretrained", None, ""),
    "fm s42": ("flux-fm", "fm", 42, "git10m_flux_ca8e6ce7+42+fm-8c098506"),
    "fm s43": ("flux-fm-s43", "fm", 43, "git10m_flux_ca8e6ce7+43+fm-c0aa0eab"),
    "fm s44": ("flux-fm-s44", "fm", 44, "git10m_flux_ca8e6ce7+44+fm-c0aa0eab"),
    "sem s42": ("flux-sem-s42", "sem", 42, "git10m_flux_ca8e6ce7+42+sem-55ae3c4c"),
    "sem s43": ("flux-sem-s43", "sem", 43, "git10m_flux_ca8e6ce7+43+sem-55ae3c4c"),
    "naive s42": ("flux-naive", "naive", 42, "git10m_flux_ca8e6ce7+42+naive-11e56b1e"),
    "naive s43": ("flux-naive-s43", "naive", 43, "git10m_flux_ca8e6ce7+43+naive-11e56b1e"),
    "proj s42": ("flux-proj", "proj", 42, "git10m_flux_ca8e6ce7+42+proj-ffcd7963"),
    "proj s43": ("flux-proj-s43", "proj", 43, "git10m_flux_ca8e6ce7+43+proj-ffcd7963"),
}
# The seed-43 FM-only checkpoint is the damaged run of OUTLINE section 4.5 (So2Sat collapses); seed 44 is its
# rerun. Both are plotted: 43 because the figure must not hide it, 44 so the reader sees it is one run, not the arm.
UNSTABLE = {"fm s43": "FM-only seed 43\n(unstable checkpoint)"}
SEED_FILL = {None: "full", 42: "full", 43: "none", 44: "left"}
SEED_NOTE = "Filled: seed 42.  Hollow: seed 43.  Half-filled: seed 44 (FM-only rerun)."


def sampling_identity(gen: dict, held: dict) -> dict:
    """Everything that must be EQUAL across arms for their samples to be comparable, from one arm's
    gen_metrics.json (`gen`) and gen_metrics_geoheldout.json (`held`). The initial noise is
    torch.randn(generator seeded with seeds[i // gen_n]) at a shape fixed by (gen_n, img_size)
    (train_diffusion._sample), so equal `seeds` + `sampler` is equal noise; the tensor itself is not
    stored (`latent_sha1` hashes the FINAL latent and differs per arm by design)."""
    sm = gen["sampler"]
    want = [sm["gen_seed"] + 1000 + i for i in range(-(-gen["n"] // sm["gen_n"]))]
    if gen["seeds"] != want:
        raise SystemExit(
            f"sampling identity: seeds are not gen_seed + 1000 + batch index ({gen['seeds'][:3]}...)"
        )
    if (sm["gen_steps"], sm["guidance"]) != (gen["gen_steps"], gen["guidance"]) or list(
        sm["img_size"]
    ) != list(gen["fake"]["hw"]):
        raise SystemExit(
            f"sampling identity: sampler block {sm} disagrees with the recorded steps / guidance / sample size"
        )
    if held["fake"] != gen["fake"]:
        raise SystemExit(
            "sampling identity: the held-out FID was computed on different samples than gen_metrics.json"
        )
    if gen["extractor"] != held["extractor"]:
        raise SystemExit("sampling identity: the two FIDs of one arm used different extractors")
    return {
        "sampler": sm,
        "seeds": gen["seeds"],
        "n": gen["n"],
        "sample_hw": gen["fake"]["hw"],
        "extractor": held["extractor"],
        "heldout_ref": held["real"],
        "overlap_ref": gen["real"],
    }


def assert_same_sampling(idents: dict[str, dict]) -> dict:
    """GATE: every arm was sampled with the same noise seeds, steps, guidance, resolution and count, and
    scored with the same extractor against the same reference images. A mismatch stops the figure."""
    keys = list(idents)
    ref = idents[keys[0]]
    for k in keys[1:]:
        bad = [f for f in ref if idents[k][f] != ref[f]]
        if bad:
            raise SystemExit(
                f"GATE sampling identity FAIL: {k!r} differs from {keys[0]!r} in {bad}; do NOT compare these arms"
            )
    sm = ref["sampler"]
    print(
        f"GATE sampling identity PASS: {len(keys)} arms, n={ref['n']}, seeds {ref['seeds'][0]}..{ref['seeds'][-1]} x {sm['gen_n']}/batch, "
        f"{sm['gen_steps']} steps, guidance {sm['guidance']}, {ref['sample_hw']} px, held-out ref sha1 {ref['heldout_ref']['sha1'][:12]}"
    )
    return ref


def _drawn_texts(fig) -> list:
    """Every text the saved figure shows: figure texts, axes titles / axis labels / annotations, the tick
    labels of ticks inside the view (matplotlib keeps label objects for ticks it does not draw) and legend
    entries. Call after a draw."""
    out, legends = [], list(fig.legends)

    def add(t):
        if t is not None and t.get_visible() and t.get_text().strip():
            out.append(t)

    for t in fig.texts:
        add(t)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts):
            add(t)
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if lo - 1e-6 * (hi - lo) <= tick.get_loc() <= hi + 1e-6 * (hi - lo):
                    add(tick.label1)
                    add(tick.label2)
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for lg in legends:
        if lg.get_visible():
            add(lg.get_title())
            for t in lg.get_texts():
                add(t)
    return out


def assert_labels_fit(fig, name: str, tol: float = 1.0) -> int:
    """GATE: no label is truncated. Every drawn text lies inside the figure canvas (so the file does not
    depend on bbox_inches='tight' growing the page to rescue it) and no two texts overlap by more than
    `tol` px in both directions (a label drawn over another is cut just the same). Stops the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    boxes = [(t.get_text().replace("\n", " / "), t.get_window_extent(r)) for t in _drawn_texts(fig)]
    if not boxes:
        raise SystemExit(
            f"GATE labels fit FAIL: {name}: no text found, the check is not looking at the figure"
        )
    for txt, b in boxes:
        if b.x0 < -tol or b.y0 < -tol or b.x1 > w + tol or b.y1 > h + tol:
            raise SystemExit(
                f"GATE labels fit FAIL: {name}: {txt!r} is outside the canvas "
                f"(x {b.x0:.0f}..{b.x1:.0f} of {w}, y {b.y0:.0f}..{b.y1:.0f} of {h} px)"
            )
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1 :]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol:
                raise SystemExit(f"GATE labels fit FAIL: {name}: {ta!r} and {tb!r} overlap")
    print(f"GATE labels fit PASS: {name}: {len(boxes)} texts inside the {w}x{h} px canvas, none overlapping")
    return len(boxes)


def trade_arm(key: str) -> dict:
    model, arm, seed, run = TRADE_ARMS[key]
    d = gen_run_dir(run)
    gen, held = json.load(open(f"{d}/gen_metrics.json")), json.load(open(f"{d}/gen_metrics_geoheldout.json"))
    if abs(held["fid_train_overlap_ref"] - gen["fid"]) > 1e-9:
        raise SystemExit(
            f"{key}: gen_metrics_geoheldout.json carries a different overlap-reference FID than gen_metrics.json"
        )
    return {
        "key": key,
        "arm": arm,
        "seed": seed,
        "matrix_model": model,
        "gen_eval_dir": d,
        "identity": sampling_identity(gen, held),
        "fid_heldout": float(held["fid"]),
        "fid_train_overlap_ref": float(gen["fid"]),
    }


def fig_null_grid(
    keys=("pretrained", "fm s42", "sem s42", "naive s42"), idx=(0, 4, 8), name="gen_grid_null_3x4"
):
    """Null-prompt samples, rows = initial-noise draw, columns = arm: every tile of a row starts from the
    same noise. idx are sample file indices; file i is position i % gen_n of the batch seeded seeds[i // gen_n],
    so the default (0, 4, 8) is the first image of the first three seeds (fixed, not picked by looking)."""
    arms = [trade_arm(k) for k in keys]
    ref = assert_same_sampling({a["key"]: a["identity"] for a in arms})
    gen_n, hw = ref["sampler"]["gen_n"], tuple(ref["sample_hw"])
    tile = 0.85
    fig, axes = plt.subplots(
        len(idx),
        len(arms),
        figsize=(len(arms) * tile + 0.15, len(idx) * tile + 0.3),
        squeeze=False,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )
    for c, a in enumerate(arms):
        sub = (
            f"s{a['seed']}, " if a["seed"] else ""
        )  # s = adaptation seed (rows are the sampling-noise seeds)
        axes[0, c].set_title(
            f"{TRADE_STYLE[a['arm']][0]}\n{sub}FID {a['fid_heldout']:.1f}",
            fontsize=5.5,
            pad=2,
            linespacing=1.15,
        )
        for r, i in enumerate(idx):
            im = Image.open(f"{a['gen_eval_dir']}/samples/{i:05d}.png").convert("RGB")
            if im.size != (hw[1], hw[0]):
                raise SystemExit(f"{a['key']} sample {i}: {im.size} px, the sampler recorded {hw}")
            ax = axes[r, c]
            ax.imshow(np.asarray(im), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    rows = [
        {"file_index": i, "noise_seed": ref["seeds"][i // gen_n], "batch_position": i % gen_n} for i in idx
    ]
    for r, row in enumerate(rows):
        axes[r, 0].set_ylabel(f"noise seed {row['noise_seed']}", fontsize=5.5, labelpad=2)
    assert_labels_fit(fig, name)
    save(fig, name, svg=True, dpi=600)
    json.dump(
        {
            "prompt": "null (train_diffusion._sample: _expand_null_embeddings)",
            "fid": "geo-held-out Git-10M reference, N = 5000",
            "sampling": ref,
            "rows": rows,
            "columns": [
                {
                    k: a[k]
                    for k in ("key", "arm", "seed", "gen_eval_dir", "fid_heldout", "fid_train_overlap_ref")
                }
                for a in arms
            ],
        },
        open(f"{OUT}/{name}.json", "w"),
        indent=1,
    )


# ---------------------------------------------------------------- 5. prompted grid (remote-sensing prompt list)
# Rows = prompts of a fixed list (experiments/isaac/git10m/rs_prompts.json, written before sampling), columns =
# arms, every tile of a row from the same noise draw. Samples: prompt_gen.py --prompts (no paired real images,
# so no FID); the per-prompt CLIP score of each sample is the only number, and it goes to the JSON, not the figure.
PROMPT_GEN_RS = f"{STORE}/models/prompt_gen_rs"
PROMPT_IDENTITY = ("captions_sha1", "n", "img_size", "steps", "guidance", "seed", "noise")


def prompt_arm(key: str, root: str | None = None, run: str = "") -> dict:
    """One arm of a prompt_gen.py --prompts run: the single finished run dir <root>/<tag>_*/ (dir name containing
    `run`) whose manifest names this arm's checkpoint (TRADE_ARMS training-run substring, '' = pretrained)."""
    root = root or PROMPT_GEN_RS
    _, arm, seed, ckpt_run = TRADE_ARMS[key]
    tag = key.replace(" ", "-")
    hits = sorted(
        p for p in glob.glob(f"{root}/{tag}_*/manifest.json") if run in os.path.basename(os.path.dirname(p))
    )
    if len(hits) != 1:
        raise SystemExit(
            f"GATE prompt arm FAIL: {key!r}: need exactly one finished run {root}/{tag}_*{run}*/, found {hits}"
        )
    d = os.path.dirname(hits[0])
    man = json.load(open(hits[0]))
    if man["tag"] != tag or (man["ckpt"] == "") != (ckpt_run == "") or ckpt_run not in man["ckpt"]:
        raise SystemExit(
            f"GATE prompt arm FAIL: {d} is tag {man['tag']!r} checkpoint {man['ckpt']!r}, {key!r} needs {ckpt_run or 'pretrained'!r}"
        )
    if not man.get("prompts_file"):
        raise SystemExit(f"GATE prompt arm FAIL: {d} is not a prompt_gen.py --prompts run")
    caps = [p["caption"] for p in man["prompts"]]
    if (
        hashlib.sha1(json.dumps(caps, ensure_ascii=False).encode()).hexdigest() != man["captions_sha1"]
        or len(caps) != man["n"]
    ):
        raise SystemExit(
            f"GATE prompt arm FAIL: {d}: the stored prompts do not hash to the stamped captions_sha1 / n"
        )
    clip = json.load(open(f"{d}/clip_per_pair.json"))["fake"]
    if len(clip) != man["n"]:
        raise SystemExit(f"GATE prompt arm FAIL: {d}: {len(clip)} CLIP scores for {man['n']} prompts")
    return {
        "key": key,
        "arm": arm,
        "seed": seed,
        "dir": d,
        "ckpt": man["ckpt"],
        "prompts_file": man["prompts_file"],
        "identity": {f: man[f] for f in PROMPT_IDENTITY},
        "prompts": man["prompts"],
        "clip": clip,
    }


def assert_same_prompt_sampling(idents: dict[str, dict]) -> dict:
    """GATE: every arm sampled the same ordered caption list with the same noise rule and seed, steps, guidance
    and resolution. A mismatch stops the figure."""
    keys = list(idents)
    ref = idents[keys[0]]
    for k in keys[1:]:
        bad = [f for f in PROMPT_IDENTITY if idents[k][f] != ref[f]]
        if bad:
            raise SystemExit(
                f"GATE prompt sampling identity FAIL: {k!r} differs from {keys[0]!r} in {bad}; do NOT compare these arms"
            )
    print(
        f"GATE prompt sampling identity PASS: {len(keys)} arms, {ref['n']} prompts (sha1 {ref['captions_sha1'][:12]}), seed {ref['seed']}, "
        f"{ref['steps']} steps, guidance {ref['guidance']}, {ref['img_size']} px"
    )
    return ref


def fig_prompt_grid(
    keys=("pretrained", "fm s42", "sem s42", "naive s42"),
    idx=None,
    name="gen_grid_rs_prompts",
    root: str | None = None,
    run: str = "",
    wrap: int = 72,
    n_boot: int = 10000,
):
    """idx: prompt indices to show; default = the entries flagged "figure": true in the prompt list (fixed in
    the list before sampling, not picked by looking); "all" = every prompt (supplement)."""
    arms = [prompt_arm(k, root, run) for k in keys]
    ref = assert_same_prompt_sampling({a["key"]: a["identity"] for a in arms})
    prompts, px = arms[0]["prompts"], ref["img_size"]
    if idx is None:
        idx = tuple(i for i, p in enumerate(prompts) if p.get("figure"))
        if not idx:
            raise SystemExit(
                f'{arms[0]["prompts_file"]}: no entry is flagged "figure": true; pass the indices'
            )
    elif idx == "all":
        idx = tuple(range(len(prompts)))
    if not all(0 <= i < len(prompts) for i in idx) or len(set(idx)) != len(idx):
        raise SystemExit(f"prompt indices {idx} are not distinct indices into the {len(prompts)} prompts")
    lines = [textwrap.wrap(prompts[i]["caption"], wrap) for i in idx]
    tile, head_h = 0.85, 0.14
    heights = [head_h] + [h for ln in lines for h in (0.085 * len(ln) + 0.05, tile)]
    fig = plt.figure(figsize=(len(arms) * tile + 0.1, sum(heights) + 0.03 * len(heights)))
    gs = fig.add_gridspec(
        len(heights),
        len(arms),
        height_ratios=heights,
        wspace=0.03,
        hspace=0.06,
        left=0.01,
        right=0.99,
        top=0.995,
        bottom=0.005,
    )
    for c, a in enumerate(arms):
        ax = fig.add_subplot(gs[0, c])
        ax.axis("off")
        ax.text(
            0.5, 0.5, TRADE_STYLE[a["arm"]][0], ha="center", va="center", fontsize=6, transform=ax.transAxes
        )
    for r, i in enumerate(idx):
        ax = fig.add_subplot(gs[1 + 2 * r, :])
        ax.axis("off")
        ax.text(
            0.0,
            0.5,
            "\n".join(lines[r]),
            ha="left",
            va="center",
            fontsize=5.5,
            linespacing=1.1,
            transform=ax.transAxes,
        )
        for c, a in enumerate(arms):
            im = Image.open(f"{a['dir']}/{i:05d}.png").convert("RGB")
            if im.size != (px, px):
                raise SystemExit(f"{a['key']} sample {i}: {im.size} px, the manifest recorded {px}")
            ax = fig.add_subplot(gs[2 + 2 * r, c])
            ax.imshow(np.asarray(im), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    assert_labels_fit(fig, name)
    save(fig, name, svg=True, dpi=600)
    # CLIP score per arm over ALL prompts of the list (not the shown rows); difference to the first column paired
    # per prompt, bootstrap over prompts (one image per prompt = the sampling unit).
    rng = np.random.default_rng(0)
    base = np.asarray(arms[0]["clip"], float)
    boot = rng.integers(0, len(base), size=(n_boot, len(base)))
    cols = []
    for a in arms:
        s = np.asarray(a["clip"], float)
        d = s - base
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        cols.append(
            {
                "key": a["key"],
                "arm": a["arm"],
                "seed": a["seed"],
                "dir": a["dir"],
                "ckpt": a["ckpt"],
                "clip_mean_all": float(s.mean()),
                "clip_mean_shown": float(s[list(idx)].mean()),
                f"clip_minus_{arms[0]['key']}": {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]},
            }
        )
        print(
            f"  {a['key']:<12} CLIP {s.mean():6.2f} (all {len(s)} prompts)  {s[list(idx)].mean():6.2f} (shown)  "
            f"minus {arms[0]['key']} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}]"
        )
    json.dump(
        {
            "prompt_list": arms[0]["prompts_file"],
            "sampling": ref,
            "clip": "100 x cosine(CLIP ViT-L/14 image, caption) of each sample (prompt_gen.clip_score); "
            f"ci95 = percentile bootstrap over prompts, {n_boot} draws, numpy default_rng(0)",
            "rows": [
                {
                    "prompt_index": i,
                    **{k: v for k, v in prompts[i].items()},
                    "noise_seed": ref["seed"] * 1_000_003 + i,
                }
                for i in idx
            ],
            "columns": cols,
        },
        open(f"{OUT}/{name}.json", "w"),
        indent=1,
    )


def task_acc100(model: str, dataset: str) -> float:
    c = cells(model, dataset)
    if 1.0 not in c:
        raise SystemExit(f"{model} {dataset}: no 100 %-label cell")
    return float(np.mean(c[1.0]))


def fig_fid_vs_so2sat(name="fid_vs_so2sat"):
    """Generation vs recognition per checkpoint: x = FID of the null-prompt samples vs the geo-held-out
    Git-10M reference (log, lower = LEFT), y = test accuracy of the linear probe at 100 % labels. Two files:
    <name>_mean5 (mean of the 5 single-label tasks, the headline panel) and <name> (m-So2Sat alone).
    Desirable corner: upper left. Filled = adaptation seed 42, hollow = 43, half = 44 (FM-only rerun of the
    unstable seed-43 checkpoint, which stays in the plot with its annotation). Drawn at CVPR column width
    (3.25 in), so the font sizes are the print sizes. The arm legend sits in the empty lower-right of the
    plotting area; the seed markers are explained by ONE line under the axis (SEED_NOTE)."""
    from matplotlib.lines import Line2D

    arms = [trade_arm(k) for k in TRADE_ARMS]
    ref = assert_same_sampling({a["key"]: a["identity"] for a in arms})
    for a in arms:
        a["so2sat_acc100"] = task_acc100(a["matrix_model"], "m_so2sat")
        a["mean_acc100_single5"] = mean_acc100(a["matrix_model"])
        if a["mean_acc100_single5"] is None:
            raise SystemExit(f"{a['key']}: 100 %-label cells incomplete on the 5 single-label tasks")
        print(
            f"   {a['key']:11s} FID held-out {a['fid_heldout']:7.2f} (overlap ref {a['fid_train_overlap_ref']:7.2f})  "
            f"so2sat {a['so2sat_acc100']:.2f}  mean5 {a['mean_acc100_single5']:.2f}"
        )
    variants = (
        (f"{name}_mean5", "mean_acc100_single5", "Mean accuracy across five\nsingle-label tasks (%) \u2191"),
        (name, "so2sat_acc100", "m-So2Sat accuracy (%) \u2191"),
    )
    for fname, ykey, ylabel in variants:
        fig, ax = plt.subplots(figsize=(3.25, 2.75))
        for arm, (lab, col, mk) in TRADE_STYLE.items():
            pts = sorted(
                (a for a in arms if a["arm"] == arm and a["seed"] in (42, 43)), key=lambda a: a["seed"]
            )
            if len(pts) == 2:  # the two adaptation seeds of one arm, joined: the seed spread
                ax.plot(
                    [p["fid_heldout"] for p in pts],
                    [p[ykey] for p in pts],
                    color=col,
                    lw=0.6,
                    alpha=0.5,
                    zorder=2,
                )
        for a in arms:
            lab, col, mk = TRADE_STYLE[a["arm"]]
            ax.plot(
                [a["fid_heldout"]],
                [a[ykey]],
                ls="",
                marker=mk,
                ms=7.5 if mk == "*" else 5,
                color=col,
                mec=col,
                mew=0.9,
                fillstyle=SEED_FILL[a["seed"]],
                markerfacecoloralt="white",
                zorder=3,
            )
            if a["key"] in UNSTABLE:
                ax.annotate(
                    UNSTABLE[a["key"]],
                    (a["fid_heldout"], a[ykey]),
                    xytext=(9, 7),
                    textcoords="offset points",
                    fontsize=5.5,
                    color="0.15",
                    ha="left",
                    va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkB=3),
                )
        ax.set_xscale("log")
        lo, hi = min(a["fid_heldout"] for a in arms), max(a["fid_heldout"] for a in arms)
        ax.set_xlim(lo / 1.15, hi * 1.15)
        ticks = [t for t in (20, 25, 30, 40, 50, 75, 100, 150, 200) if lo / 1.15 <= t <= hi * 1.15]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Git-10M FID \u2193")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.margins(y=0.08)
        handles = [
            Line2D([], [], ls="", marker=mk, color=col, ms=6.5 if mk == "*" else 4.5, label=lab)
            for lab, col, mk in TRADE_STYLE.values()
        ]
        ax.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(1.0, 0.45),
            fontsize=5.5,
            frameon=True,
            framealpha=0.9,
            edgecolor="0.8",
            handletextpad=0.3,
            labelspacing=0.35,
            borderpad=0.4,
        )
        fig.text(0.5, 0.012, SEED_NOTE, ha="center", va="bottom", fontsize=5.5, color="0.25")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        assert_labels_fit(fig, fname)
        save(fig, fname)
    json.dump(
        {
            "x": "fid_heldout: FID vs the geo-held-out Git-10M reference, N = 5000 null-prompt samples (log axis)",
            "y": {
                "so2sat_acc100": "m_so2sat matrix cell f1.0, test score x100",
                "mean_acc100_single5": f"mean over {SINGLE5}",
            },
            "seed_note": SEED_NOTE,
            "sampling": ref,
            "unstable": UNSTABLE,
            "points": [{k: v for k, v in a.items() if k != "identity"} for a in arms],
        },
        open(f"{OUT}/{name}.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    which = sys.argv[1:] or ["seg", "fid", "gen"]
    if "seg" in which:
        fig_seg_curves()
    if "fid" in which:
        fig_fid_vs_acc_heldout()
    if "coh" in which:
        fig_sem_coherence()
        fig_sem_coherence("sem_coherence_vs_step_stacked", stacked=True)
    if "col" in which:
        fig_block_t_column()
        fig_block_t_column(name="cos_block_t_proj_wide", figsize=(7.0, 2.6))
        fig_block_t_column(
            (("proj", "init"), ("proj", "step10000"), ("proj", "final"), ("fm", "final")),
            "cos_block_t_proj_fm_col",
        )
    if "gen" in which:
        fig_gen_grid()
    if (
        "null" in which
    ):  # NULL_ARMS="pretrained,fm s42,naive s42,proj s42" NULL_IDX=0,4,8 NULL_NAME=... override the layout
        kw = {}
        if os.environ.get("NULL_ARMS"):
            kw["keys"] = tuple(k.strip() for k in os.environ["NULL_ARMS"].split(","))
        if os.environ.get("NULL_IDX"):
            kw["idx"] = tuple(int(i) for i in os.environ["NULL_IDX"].split(","))
        if os.environ.get("NULL_NAME"):
            kw["name"] = os.environ["NULL_NAME"]
        fig_null_grid(**kw)
    if "trade" in which:
        fig_fid_vs_so2sat()
    if (
        "rs" in which
    ):  # RS_ARMS="pretrained,fm s42,sem s42,naive s42,proj s42" RS_IDX=0,7,13 | all  RS_NAME=...  RS_ROOT=<dir>  RS_RUN=<dir-name substring>
        kw = {}
        if os.environ.get("RS_ARMS"):
            kw["keys"] = tuple(k.strip() for k in os.environ["RS_ARMS"].split(","))
        if os.environ.get("RS_IDX"):
            kw["idx"] = (
                "all"
                if os.environ["RS_IDX"] == "all"
                else tuple(int(i) for i in os.environ["RS_IDX"].split(","))
            )
        for env, arg in (("RS_NAME", "name"), ("RS_ROOT", "root"), ("RS_RUN", "run")):
            if os.environ.get(env):
                kw[arg] = os.environ[env]
        fig_prompt_grid(**kw)
    print("PAPER_FIGS_OK", which)
