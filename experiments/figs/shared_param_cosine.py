"""Gradient agreement restricted to the parameters both losses actually touch.

The alignment loss is attached at block k (28), so the semantic gradient is identically zero on
the LoRA parameters of blocks >= k. The "global" cosine logged in stats.npz (column -1 of
step/cos_pre) is taken over ALL LoRA parameters, so those zero-semantic blocks inflate |g_FM| and
shrink the cosine. This script recomputes, from the per-block dot products and norms already in
stats.npz, the cosine over the shared subset (blocks < k) together with the norm ratios, both for
the per-step training series and for the fixed-t measurement batches. It changes no number used
elsewhere; it is the companion statistic the methods section quotes next to the global one.

Output: results/figs/mechanism/shared_param_cosine.{json,md}. Prints SHARED_COS_OK.
"""

import argparse
import glob
import json
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def restricted(dot, nf, ns, sub):
    d = dot[:, sub].sum(1)
    a = np.sqrt((nf[:, sub] ** 2).sum(1))
    b = np.sqrt((ns[:, sub] ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / (a * b), a, b


def one_run(run):
    s = np.load(os.path.join(run, "stats.npz"), allow_pickle=True)
    k, blocks = int(s["k"]), np.asarray(s["blocks"])
    w = float(s["align_weight"])
    sub, allb = (blocks >= 0) & (blocks < k), blocks >= 0
    dot, nf, ns = (s[f"step/{n}"].astype(float) for n in ("dot", "norm_fm", "norm_sem"))
    cp = s["step/cos_pre"].astype(float)
    if not np.allclose(np.nan_to_num(ns[:, ~sub & allb]), 0):
        raise SystemExit(
            f"{run}: semantic gradient is not zero on blocks >= {k}; the premise of this script fails"
        )
    glob_re = dot[:, allb].sum(1) / (np.sqrt((nf[:, allb] ** 2).sum(1)) * np.sqrt((ns[:, allb] ** 2).sum(1)))
    if not np.allclose(np.nan_to_num(glob_re), np.nan_to_num(cp[:, -1]), atol=1e-3):
        raise SystemExit(
            f"{run}: logged global cosine != cosine recomputed from per-block sums (projector included?)"
        )
    cr, nfr, nsr = restricted(dot, nf, ns, sub)
    nfa = nf[:, -1]
    valid = np.isfinite(cr) & (nsr > 0)
    last = np.zeros_like(valid)
    last[-5000:] = True
    out = {
        "k": k,
        "align_weight": w,
        "n_steps": int(valid.sum()),
        "cos_global_mean": float(np.nanmean(cp[valid, -1])),
        "cos_shared_mean": float(np.nanmean(cr[valid])),
        "cos_global_mean_last5k": float(np.nanmean(cp[valid & last, -1])),
        "cos_shared_mean_last5k": float(np.nanmean(cr[valid & last])),
        "frac_conflict_global": float(np.mean(cp[valid, -1] < 0)),
        "frac_conflict_shared": float(np.mean(cr[valid] < 0)),
        "norm_fm_shared_over_all_mean": float(np.nanmean(nfr[valid] / nfa[valid])),
        "norm_sem_over_fm_shared_raw_mean": float(np.nanmean(nsr[valid] / nfr[valid])),
        "norm_sem_over_fm_shared_applied_mean": float(w * np.nanmean(nsr[valid] / nfr[valid])),
        "norm_sem_over_fm_all_raw_mean": float(np.nanmean(nsr[valid] / nfa[valid])),
    }
    # fixed-t measurement batches: cos/norm per block -> dot per block
    meas = {}
    for key in s.files:
        m = re.match(r"meas/(\w+)/t(\d+)/cos$", key)
        if not m:
            continue
        tag, t = m.group(1), int(m.group(2))
        c = s[key].astype(float)
        a = s[f"meas/{tag}/t{t}/norm_fm"].astype(float)
        b = s[f"meas/{tag}/t{t}/norm_sem"].astype(float)
        d = c * a * b
        cr_t, _, _ = restricted(d, a, b, sub)
        meas.setdefault(tag, {})[t] = {
            "cos_global_mean": float(np.nanmean(s[f"meas/{tag}/t{t}/agg_cos"][-1]))
            if s[f"meas/{tag}/t{t}/agg_cos"].ndim == 1
            else None,
            "cos_shared_mean": float(np.nanmean(cr_t)),
        }
    out["meas"] = {tag: {str(t): v for t, v in sorted(d.items())} for tag, d in meas.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["models/git10m/git10m_flux_ca8e6ce7+4[23]+*"])
    ap.add_argument("--out", default="results/figs/mechanism/shared_param_cosine")
    a = ap.parse_args()
    os.chdir(ROOT)
    runs = sorted(r for pat in a.runs for r in glob.glob(pat) if os.path.exists(os.path.join(r, "stats.npz")))
    if not runs:
        raise SystemExit("no runs matched")
    res = {os.path.basename(r): one_run(r) for r in runs}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"runs": res, "note": __doc__}, open(a.out + ".json", "w"), indent=1)
    lines = [
        "| run | cos global | cos shared (blocks<k) | conflict frac global / shared | |g_FM,<k| / |g_FM,all| | |g_SEM| / |g_FM,<k| raw (x w) |",
        "|---|---|---|---|---|---|",
    ]
    for n, r in res.items():
        lines.append(
            f"| {n} | {r['cos_global_mean']:+.4f} | {r['cos_shared_mean']:+.4f} | {r['frac_conflict_global']:.3f} / {r['frac_conflict_shared']:.3f} | {r['norm_fm_shared_over_all_mean']:.3f} | {r['norm_sem_over_fm_shared_raw_mean']:.2f} ({r['norm_sem_over_fm_shared_applied_mean']:.2f}) |"
        )
    lines.append("")
    lines.append(
        "Means over all 20,000 optimizer steps (semantic gradient before the alignment weight; 'x w' = after). "
        "Per-step cos_pre column -1 of stats.npz is the global cosine; the shared cosine uses the per-block dot/norm sums over blocks < k."
    )
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("SHARED_COS_OK")


if __name__ == "__main__":
    main()
