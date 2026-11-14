"""P1 mechanism analyses from logs only (no GPU): results/figs/mechanism/*.

1. timestep: per-timestep probe gain of each adapted arm over frozen FLUX (block 28,
   test accuracy at the val-selected C per candidate, results/eval/sweep_mb) against (a) the measured cos(g_FM, g_SEM) at that t and
   (b) the split-eval backbone share at that t. The gradient cosine is a SNAPSHOT series: measured at
   init, every 2000 steps and at the end on 8 fixed batches at t in {100, 260, 420, 580, 740, 900};
   this script uses the init, step-10000 and final snapshots and says which. Correlations are
   Spearman over (dataset x t) points, n small, reported with n.
2. contributions: from the per-step log (accumulated gradients of every optimizer step):
   magnitude ratio w|g_SEM| / |g_FM|; the first-order cross term c = w <g_FM, g_SEM> / |g_FM|^2 (the
   relative change of FM's own first-order descent caused by adding the semantic gradient: c < 0
   hurts, c < -1 means the summed step raises the FM loss to first order); how much the one-sided
   projection changes the update (fraction of (step, block) pairs projected, the removed fraction of
   |g_SEM| = |cos| there, the relative change of the combined update); cos(update, g_FM) after AdamW.
   The fm arm applies w = 0: its ratio / cross term use the CONFIG weight and are marked so.
3. teachers: the same gradient geometry for the DINOv3-web runs against the DINOv3-sat runs.
Writes mechanism_summary.json with every number plotted.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "figs"))
import mentor_figs as MF  # noqa: E402  (run_label, ema, GROUP_COLOR, DATASETS, best_val_per_candidate)

OUT = "results/figs/mechanism"
os.makedirs(OUT, exist_ok=True)
MAIN = MF.MAIN
ARMS = {"flux-fm": "fm s42", "flux-naive": "naive s42", "flux-proj": "proj s42"}
PROBE_T = [100, 180, 260, 340, 420, 500, 580, 740, 900]


def runs_by_label() -> dict[str, tuple[str, str, np.lib.npyio.NpzFile]]:
    """mentor_figs.run_label does not encode the align weight, k or rank: the sweep runs would all
    read "proj s42". Suffix the non-default ones (w2 / w4 / k56 / r16) so no run shadows another."""
    out = {}
    for r, lab, grp, s in MF.load_runs():
        al = json.load(open(f"models/git10m/{r}/meta.json"))
        cfg = al.get("cfg", al)
        tags = []
        # meta["align"]["weight"] is the EFFECTIVE weight (0 for fm): the config weight lives in cfg.
        wc = float(cfg["align"]["weight"])
        if wc != 0.5:
            tags.append(f"w{wc:g}")
        if int(s["k"]) != 28:
            tags.append(f"k{int(s['k'])}")
        if int(cfg.get("lora_rank", 4)) != 4:
            tags.append(f"r{cfg['lora_rank']}")
        ss = int(cfg["align"].get("stop_step", 0) or 0)
        if ss:
            tags.append(f"stop{ss // 1000}k")
        if cfg["align"].get("sem_t_weights"):
            tags.append("semt" + "".join("1" if w > 0 else "0" for w in cfg["align"]["sem_t_weights"]))
        key = lab if not tags else lab.replace(" s", " " + " ".join(tags) + " s")
        if key in out:
            raise SystemExit(f"two runs share the label {key!r}: {out[key][0]} and {r}")
        out[key] = (r, grp, s)
    return out


def meas_cos(s, tag: str, t: int, k: int) -> tuple[float, float]:
    """(global aggregated cos, mean over blocks < k) at one snapshot and t."""
    a = s[f"meas/{tag}/t{t}/agg_cos"]
    r = a[:k]
    r = r[np.isfinite(r)]
    return float(a[-1]), float(r.mean()) if r.size else float("nan")


def probe_gain(arm_model: str, ds: str) -> dict[int, float] | None:
    fz = glob.glob(f"{MAIN}/results/eval/sweep_mb/sweep-block_{ds}_flux-oneshot-ens8-mb-k28_*.npz")
    ad = glob.glob(f"results/eval/sweep_mb/sweep-block_{ds}_{arm_model}-k28-lrw-k28_*.npz")
    if not fz or not ad:
        return None
    if len(fz) > 1 or len(ad) > 1:
        raise SystemExit(f"probe_gain {arm_model} {ds}: ambiguous sweep files {fz} / {ad}")
    z0, z1 = np.load(fz[0], allow_pickle=True), np.load(ad[0], allow_pickle=True)
    # rule 6: pair on the test-image identity, not on the label vector alone
    for key in ("test_paths", "test_labels", "candidates", "C_grid"):
        if not np.array_equal(z0[key], z1[key]):
            raise SystemExit(f"probe_gain {arm_model} {ds}: {key} differs between {fz[0]} and {ad[0]}")
    # rule 8: the two sweeps must be the same instrument -- same extraction pins apart from the
    # weights (frozen vs +lora), neither a smoke, the adapted one the default lr head (the frozen
    # file predates the `head` key, i.e. lr)
    p0, p1 = (json.loads(str(z["pins"])) for z in (z0, z1))
    p0.pop("weights", None), p1.pop("weights", None)
    if p0 != p1:
        raise SystemExit(f"probe_gain {arm_model} {ds}: extraction pins differ {p0} vs {p1}")
    if bool(z0["smoke"]) or bool(z1["smoke"]):
        raise SystemExit(f"probe_gain {arm_model} {ds}: a smoke sweep file ({fz[0]} / {ad[0]})")
    head = str(z1["head"]) if "head" in z1.files else "lr"
    if head != "lr" or ("head" in z0.files and str(z0["head"]) != "lr"):
        raise SystemExit(f"probe_gain {arm_model} {ds}: head {head!r} is not the lr probe")
    v0, v1 = test_at_val_c(fz[0]), test_at_val_c(ad[0])
    return {t: (v1[f"t{t}"] - v0[f"t{t}"]) * 100 for t in PROBE_T}


def test_at_val_c(path: str) -> dict[str, float]:
    """Per-candidate TEST accuracy at the C selected on val (first maximum in stored order) from one
    sweep-block file: the sealed per-image test vector of exactly that (candidate, C) is opened.
    (Review 2026-10-26: the previous max-over-C VAL accuracy selected and scored on the same split.)"""
    z = np.load(path, allow_pickle=True)
    best: dict[str, tuple[float, float]] = {}
    for r in json.loads(str(z["val_table"])):
        if r["candidate"] not in best or r["val_acc"] > best[r["candidate"]][0]:
            best[r["candidate"]] = (r["val_acc"], r["C"])
    y = z["test_labels"]
    if y.ndim != 1:
        raise SystemExit(
            f"{path}: multi-label test labels {y.shape}; test_at_val_c scores single-label tasks only"
        )
    out = {}
    for cand, (_, C) in best.items():
        v = z[f"sealed__{cand}__C{C}"]  # sealed per-image test PREDICTIONS (eval.sweep sealed_key)
        if v.shape != y.shape:
            raise SystemExit(f"{path}: sealed__{cand}__C{C} shape {v.shape} != test_labels {y.shape}")
        out[cand] = float((v == y).mean())
    return out


def sp(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    r = spearmanr(x[ok], y[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def analysis_timestep(R) -> dict:
    dss = [d for d in MF.DATASETS if d != "m_bigearthnet"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    summ = {}
    for row, (model, lab) in enumerate(ARMS.items()):
        run, grp, s = R[lab]
        k = int(s["k"])
        meas_t = [int(t) for t in s["meas_t"]]
        sp_path = f"models/git10m/{run}/split_eval.json"
        se = json.load(open(sp_path))["per_t"] if os.path.exists(sp_path) else None
        if se is None:
            print(f"WARNING {lab}: no split_eval.json yet ({run}); backbone share omitted", flush=True)
        gains = {ds: probe_gain(model, ds) for ds in dss}
        gains = {d: g for d, g in gains.items() if g}
        mean_gain = {t: float(np.mean([g[t] for g in gains.values()])) for t in PROBE_T}
        cos = {tag: {t: meas_cos(s, tag, t, k) for t in meas_t} for tag in ("init", "step10000", "final")}
        nan = float("nan")
        share = {t: (se[str(t)]["cos_on"] - se[str(t)]["cos_off"]) if se else nan for t in PROBE_T}
        total = {t: (se[str(t)]["cos_on"] - se[str(t)]["cos_init"]) if se else nan for t in PROBE_T}
        col = MF.GROUP_COLOR[grp]
        ax = axes[row, 0]
        for ds, g in gains.items():
            ax.plot(
                PROBE_T,
                [g[t] for t in PROBE_T],
                marker="o",
                ms=3,
                lw=0.8,
                alpha=0.6,
                label=ds.replace("m_", ""),
            )
        ax.plot(PROBE_T, [mean_gain[t] for t in PROBE_T], color="k", lw=2, label="mean")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{lab}: probe gain over frozen FLUX at block {k} (test, C selected on val)")
        ax.set_ylabel("accuracy points")
        ax.legend(fontsize=6)
        ax = axes[row, 1]
        for tag, ls in (("init", ":"), ("step10000", "--"), ("final", "-")):
            ax.plot(
                meas_t,
                [cos[tag][t][0] for t in meas_t],
                ls=ls,
                marker="s",
                ms=3,
                color=col,
                label=f"global cos, {tag}",
            )
        ax2 = ax.twinx()
        ax2.plot(
            PROBE_T,
            [share[t] for t in PROBE_T],
            color="0.3",
            marker="^",
            ms=3,
            label="backbone share (split-eval)",
        )
        ax2.set_ylabel("backbone share of teacher cosine")
        ax.set_ylabel("cos(g_FM, g_SEM)")
        ax.set_title("gradient alignment (8 fixed batches) and backbone share by t")
        ax.legend(fontsize=6, loc="upper left")
        ax2.legend(fontsize=6, loc="lower right")
        ax = axes[row, 2]
        xs, ys, xs2, ys2 = [], [], [], []
        for ds, g in gains.items():
            for t in meas_t:
                xs.append(cos["final"][t][0])
                ys.append(g[t])
            for t in PROBE_T:
                xs2.append(share[t])
                ys2.append(g[t])
        ax.scatter(xs, ys, color=col, s=18, label="vs final global cos (t in meas grid)")
        ax.scatter(xs2, ys2, color="0.3", s=12, marker="^", label="vs backbone share (9 t)")
        r1, r2 = sp(xs, ys), sp(xs2, ys2)

        def fmt(r):
            return "n/a" if r["rho"] is None else f"{r['rho']:.2f}"

        # the share / cosine series are ONE per run (repeated per dataset): the per-t statistic on the
        # dataset-mean gain (n = 9 / 6 independent x) is the honest one; the pooled points are shown only
        rm0 = sp([cos["final"][t][0] for t in meas_t], [mean_gain[t] for t in meas_t])
        rs0 = sp([share[t] for t in PROBE_T], [mean_gain[t] for t in PROBE_T])
        ax.set_title(
            f"mean gain vs cos: rho={fmt(rm0)} (n={rm0['n']}); vs share: rho={fmt(rs0)} (n={rs0['n']}); pooled points rho {fmt(r1)}/{fmt(r2)} (clustered)",
            fontsize=8,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("probe gain (points)")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=6)
        rm = sp([cos["final"][t][0] for t in meas_t], [mean_gain[t] for t in meas_t])
        rs = sp([share[t] for t in PROBE_T], [mean_gain[t] for t in PROBE_T])
        summ[lab] = {
            "run": run,
            "k": k,
            "datasets": list(gains),
            "probe_t": PROBE_T,
            "meas_t": meas_t,
            "mean_gain_by_t": mean_gain,
            "gain_by_dataset": gains,
            "cos_global_by_snapshot": {tag: {t: cos[tag][t][0] for t in meas_t} for tag in cos},
            "cos_blocks_lt_k_by_snapshot": {tag: {t: cos[tag][t][1] for t in meas_t} for tag in cos},
            "backbone_share_by_t": share,
            "total_cos_gain_by_t": total,
            "spearman_gain_vs_final_cos_points": r1,
            "spearman_gain_vs_share_points": r2,
            "spearman_meangain_vs_final_cos": rm,
            "spearman_meangain_vs_share": rs,
        }
    for ax in axes[:, :2].ravel():
        ax.set_xlabel("timestep t (higher = more noise)")
        ax.grid(alpha=0.3)
    fig.suptitle("Timestep-resolved probe gain vs gradient alignment vs backbone share (seed 42 runs)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/timestep_gain_vs_cos.png", dpi=150)
    return summ


def _w(s, lab: str, run: str) -> tuple[np.ndarray, str]:
    """Per-step effective weight; fm applies 0 so its ratio uses the CONFIG weight (marked), read from
    meta.json cfg (stats.npz align_weight and meta["align"]["weight"] are the effective one)."""
    wc = float(json.load(open(f"models/git10m/{run}/meta.json"))["cfg"]["align"]["weight"])
    if "step/w_eff" in s.files:
        w = s["step/w_eff"].astype(float)
    else:
        w = np.full(s["step/norm_fm"].shape[0], wc)
    if str(s["arm"]) == "fm":
        return np.full_like(w, wc), "config weight (not applied: fm)"
    return w, "effective weight"


def analysis_contributions(R) -> dict:
    # the sem arm applies g_FM with weight 0: its "summed step" does not exist (review 2026-10-26)
    labs = [lab for lab in R if R[lab][1] not in ("shuffle", "sem")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    summ = {}
    for lab in labs:
        run, grp, s = R[lab]
        k = int(s["k"])
        nf, ns, dot = (
            s["step/norm_fm"].astype(float),
            s["step/norm_sem"].astype(float),
            s["step/dot"].astype(float),
        )
        cp, cpost = s["step/cos_pre"].astype(float), s["step/cos_post"].astype(float)
        n = int(np.flatnonzero(nf[:, -1] > 0).max()) + 1 if (nf[:, -1] > 0).any() else 0
        nf, ns, dot, cp, cpost = nf[:n], ns[:n], dot[:n], cp[:n], cpost[:n]
        w, wnote = _w(s, lab, run)
        w = w[:n]
        ratio = w * ns[:, -1] / nf[:, -1]
        c = w * dot[:, -1] / nf[:, -1] ** 2
        last = slice(max(0, n - 5000), n)
        # steps with no semantic gradient at all (sem_t_weights window empty in every microbatch of
        # the step: ~0.75^8 of steps for a 25 % window) carry no conflict / projection information;
        # they are excluded from those fractions and their rate is reported (review 2026-10-26).
        nonempty = ns[:, -1] > 0
        rec = {
            "frac_steps_empty_sem": float(1.0 - nonempty.mean()),
            "run": run,
            "steps": n,
            "weight_note": wnote,
            "k": k,
            "ratio_median_all": float(np.median(ratio)),
            "ratio_iqr_all": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
            "ratio_median_last5k": float(np.median(ratio[last])),
            "cross_median_all": float(np.median(c)),
            "cross_median_last5k": float(np.median(c[last])),
            "frac_steps_cross_neg": float((c < 0).mean()),
            "frac_steps_cross_lt_minus1": float((c < -1).mean()),
            "cos_pre_global_mean": float(np.nanmean(cp[:, -1])),
            "cos_pre_global_mean_last5k": float(np.nanmean(cp[last, -1])),
            "frac_steps_conflict_global": float((dot[nonempty, -1] < 0).mean())
            if nonempty.any()
            else float("nan"),
            "cos_post_global_mean": float(np.nanmean(cpost[:, -1])),
        }
        if "step/cos_update" in s.files:
            cu = s["step/cos_update"][:n].astype(float)
            # step/cos_update is cos(delta theta, g_FM): negative = descent along FM. Report the
            # readable sign too: cos(delta theta, -g_FM) > 0 means the AdamW step descends the FM loss.
            rec["cos_update_vs_gfm_mean"] = float(np.nanmean(cu))
            rec["cos_update_vs_gfm_mean_last5k"] = float(np.nanmean(cu[last]))
            rec["cos_update_vs_minus_gfm_mean"] = -rec["cos_update_vs_gfm_mean"]
        blk = slice(0, k)
        rb = w[:, None] * ns[:, blk] / nf[:, blk]
        rec["ratio_median_by_block_lt_k"] = [float(x) for x in np.nanmedian(rb, axis=0)]
        if str(s["arm"]) == "proj":
            pj = s["step/projected"][:n, blk]
            cpb = cp[:, blk]
            rec["frac_step_block_projected"] = float(pj[nonempty].mean()) if nonempty.any() else float("nan")
            rec["frac_steps_any_block_projected"] = (
                float(pj[nonempty].any(axis=1).mean()) if nonempty.any() else float("nan")
            )
            rec["removed_frac_of_gsem_where_projected_mean"] = (
                float(np.nanmean(np.abs(cpb[pj]))) if pj.any() else 0.0
            )
            # removed vector per block: w |g_SEM| |cos| along g_FM; combined norm from the identity
            removed = np.where(pj, w[:, None] * ns[:, blk] * np.abs(cpb), 0.0)
            comb = np.sqrt(
                np.maximum(nf[:, blk] ** 2 + (w[:, None] * ns[:, blk]) ** 2 + 2 * w[:, None] * dot[:, blk], 0)
            )
            rel = np.sqrt(np.nansum(removed**2, axis=1)) / np.sqrt(np.nansum(comb**2, axis=1))
            rec["projection_relative_change_of_update_median"] = float(np.nanmedian(rel))
            rec["projection_relative_change_of_update_p90"] = float(np.nanpercentile(rel, 90))
            rec["projection_relative_change_of_update_max"] = float(np.nanmax(rel))
        summ[lab] = rec
        col = MF.GROUP_COLOR[grp]
        core = lab.count(" ") == 1 and "s42" in lab
        lw, al = (1.8, 0.95) if core else (0.8, 0.5)
        axes[0, 0].plot(
            MF.ema(ratio),
            color=col,
            lw=lw,
            alpha=al,
            label=lab + (" (config w)" if "config" in wnote else ""),
        )
        axes[0, 1].plot(MF.ema(c), color=col, lw=lw, alpha=al, label=lab)
        axes[1, 0].hist(
            np.clip(c[last], -3, 3),
            bins=60,
            histtype="step",
            color=col,
            lw=lw,
            alpha=al,
            label=lab,
            density=True,
        )
        if core or "web" in lab:
            axes[1, 1].plot(
                range(k),
                rec["ratio_median_by_block_lt_k"],
                marker="o",
                ms=3,
                color=col,
                lw=lw,
                alpha=al,
                label=lab,
            )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("w |g_SEM| / |g_FM| (global), EMA 200")
    axes[0, 0].set_title("semantic / FM gradient magnitude")
    axes[0, 1].axhline(0, color="k", lw=0.6)
    axes[0, 1].axhline(-1, color="k", lw=0.6, ls="--")
    axes[0, 1].set_ylabel("c = w <g_FM,g_SEM> / |g_FM|^2, EMA 200")
    axes[0, 1].set_title("first-order cross term (c<0: semantic part slows FM descent; c<-1: raises FM loss)")
    axes[1, 0].axvline(0, color="k", lw=0.6)
    axes[1, 0].axvline(-1, color="k", lw=0.6, ls="--")
    axes[1, 0].set_xlabel("c, last 5000 steps (clipped to [-3, 3])")
    axes[1, 0].set_title("distribution of the cross term")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("block")
    axes[1, 1].set_ylabel("median w |g_SEM| / |g_FM|")
    axes[1, 1].set_title("magnitude ratio per block (< k)")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes[0]:
        ax.set_xlabel("optimizer step")
    fig.suptitle("Gradient contributions from the per-step log (accumulated gradients, every optimizer step)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/gradient_contributions.png", dpi=150)
    return summ


def analysis_teachers(R) -> dict:
    pick = [lab for lab in R if lab in ("proj s42", "naive s42") or "web" in lab]
    if not any("web" in lab for lab in pick):
        return {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    summ = {}
    for lab in pick:
        run, grp, s = R[lab]
        k = int(s["k"])
        cp = s["step/cos_pre"].astype(float)
        n = int(np.flatnonzero(np.isfinite(cp[:, -1])).max()) + 1
        cp = cp[:n]
        last = slice(max(0, n - 5000), n)
        meas_t = [int(t) for t in s["meas_t"]]
        fin = {t: meas_cos(s, "final", t, k)[0] for t in meas_t}
        ini = {t: meas_cos(s, "init", t, k)[0] for t in meas_t}
        summ[lab] = {
            "run": run,
            "teacher": "web" if "web" in lab else "sat",
            "steps": n,
            "cos_global_mean": float(np.nanmean(cp[:, -1])),
            "cos_global_mean_last5k": float(np.nanmean(cp[last, -1])),
            "frac_conflict_global": float((s["step/dot"][:n, -1] < 0).mean()),
            "cos_by_block_mean": [float(x) for x in np.nanmean(cp[:, :k], axis=0)],
            "final_agg_cos_by_t": fin,
            "init_agg_cos_by_t": ini,
        }
        col = MF.GROUP_COLOR[grp]
        ls = "--" if "web" in lab else "-"
        axes[0].plot(MF.ema(cp[:, -1]), color=col, ls=ls, lw=1.2, label=lab)
        axes[1].plot(range(k), summ[lab]["cos_by_block_mean"], color=col, ls=ls, marker="o", ms=3, label=lab)
        axes[2].plot(
            meas_t, [fin[t] for t in meas_t], color=col, ls=ls, marker="s", ms=3, label=f"{lab} final"
        )
        axes[2].plot(
            meas_t,
            [ini[t] for t in meas_t],
            color=col,
            ls=ls,
            marker="x",
            ms=3,
            lw=0.6,
            alpha=0.6,
            label=f"{lab} init",
        )
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("global cos(g_FM, g_SEM), EMA 200")
    axes[0].set_title("alignment during training")
    axes[1].set_xlabel("block")
    axes[1].set_ylabel("mean cos over training")
    axes[1].set_title("per block")
    axes[2].set_xlabel("timestep t")
    axes[2].set_ylabel("aggregated global cos, 8 batches")
    axes[2].set_title("by noise level, init vs final snapshot")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Gradient geometry: DINOv3-sat (solid) vs DINOv3-web (dashed) teacher, seed 42")
    fig.tight_layout()
    fig.savefig(f"{OUT}/teacher_geometry.png", dpi=150)
    return summ


if __name__ == "__main__":
    R = runs_by_label()
    print("runs:", sorted(R))
    out = {
        "timestep": analysis_timestep(R),
        "contributions": analysis_contributions(R),
        "teachers": analysis_teachers(R),
    }
    with open(f"{OUT}/mechanism_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    for lab, r in out["timestep"].items():
        print(
            f"TIMESTEP {lab}: mean gain by t {[round(r['mean_gain_by_t'][t], 2) for t in PROBE_T]}; final cos by t "
            f"{[round(v, 3) for v in r['cos_global_by_snapshot']['final'].values()]}; share by t {[round(r['backbone_share_by_t'][t], 3) for t in PROBE_T]}; "
            f"rho(gain,cos)={r['spearman_gain_vs_final_cos_points']['rho']} n={r['spearman_gain_vs_final_cos_points']['n']}; "
            f"rho(gain,share)={r['spearman_gain_vs_share_points']['rho']} n={r['spearman_gain_vs_share_points']['n']}"
        )
    for lab, r in out["contributions"].items():
        extra = (
            f" projected {r['frac_step_block_projected']:.3f} removed|cos| {r['removed_frac_of_gsem_where_projected_mean']:.3f} rel-change med {r['projection_relative_change_of_update_median']:.2e} p90 {r['projection_relative_change_of_update_p90']:.2e}"
            if "frac_step_block_projected" in r
            else ""
        )
        print(
            f"CONTRIB {lab}: ratio med {r['ratio_median_all']:.3f} (last5k {r['ratio_median_last5k']:.3f}) [{r['weight_note']}]; c med {r['cross_median_all']:+.3f}; "
            f"frac c<0 {r['frac_steps_cross_neg']:.3f} c<-1 {r['frac_steps_cross_lt_minus1']:.3f}; cos_pre {r['cos_pre_global_mean']:+.4f}; cos(step,-g_FM) {r.get('cos_update_vs_minus_gfm_mean', float('nan')):+.3f}{extra}"
        )
    for lab, r in out["teachers"].items():
        print(
            f"TEACHER {lab}: cos mean {r['cos_global_mean']:+.4f} last5k {r['cos_global_mean_last5k']:+.4f} conflict {r['frac_conflict_global']:.3f} final by t {[round(v, 3) for v in r['final_agg_cos_by_t'].values()]}"
        )
    print("MECHANISM_OK")
