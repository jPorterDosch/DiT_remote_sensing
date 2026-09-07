"""The eta-collapse test: is an arm's t-dependence entirely input-SNR-driven?

Averaging M eps draws scales effective noise by 1/sqrt(M): eta_eff = eta / sqrt(M) with
eta = t/(1-t). If accuracy is a function of input SNR alone, the ens1 and ens8 curves must
collapse onto a single curve when plotted against eta_eff -- with ZERO free parameters, since
sqrt(8) = 2.828 is known in advance. Residual gap after rescaling = what the arm contributes
beyond input SNR at matched effective noise.

This is the only summary statistic the 2026-08-19 audit did not sink: it is WITHIN-arm, at
constant dimensionality and constant C, so neither the capacity confound (F2) nor the
operating-point sensitivity (F1) applies. The raw arm is the positive control for the test
itself: its t-dependence is input-SNR by construction, so it MUST collapse -- if it does not,
the test is broken.

Overlap is thin (rescaled ens8 points mostly exceed the measured ens1 eta range), so the
comparison interpolates ens1's accuracy-vs-eta curve at each rescaled ens8 point and reports
only in-range points.

FINDINGS (2026-08-22, RESEARCH_NOTES 6e). The raw arm COLLAPSES (mean residuals -0.0045
EuroSAT / -0.0010 RESISC45) -- the test's own positive control passes. The DiT arm does NOT:
residuals trend with t (+0.019 at t=260 to -0.049 at t=580 on RESISC45). Resolved by the
fixed-conditioning control (6f): correct t-conditioning RESCUES high-noise inputs, so the
deficit at matched eta_eff is input structure, not conditioning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
SQRT8 = 8**0.5

RAW = {  # dataset -> (full-cache template, pooling grid, frozen C from protocol_shape)
    "eurosat": ("models/raw_xt_n5000/eurosat_rawxt_ens{m}_full.npz", 1, 1000.0),
    "resisc45": ("models/raw_xt_n5000/resisc45_rawxt_ens{m}_full.npz", 4, 0.01),
}
DIT = {  # dataset -> {ens: cache}, frozen C
    "eurosat": (
        {
            1: "models/n5000_eurosat_oneshot_ens1/eurosat_flux_0b91191d+42/multistep_train_feats_oneshot_g1.0.npz",
            8: "models/n5000_eurosat_oneshot_ens8/eurosat_flux_23ee82c7+42/multistep_train_feats_oneshot_g1.0.npz",
        },
        0.01,
    ),
    "resisc45": (
        {
            1: "models/n5000_resisc45_oneshot_ens1/resisc45_flux_19044381+42/multistep_train_feats_oneshot_g1.0.npz",
            8: "models/n5000_resisc45_oneshot_ens8/resisc45_flux_4118f153+42/multistep_train_feats_oneshot_g1.0.npz",
        },
        0.1,
    ),
}


def repool(path, g):
    d = np.load(path)
    f = d["feats"]
    n, k, dim = f.shape
    hw = int(round((dim // 16) ** 0.5))
    c = dim // (hw * hw)
    x = torch.from_numpy(f).reshape(n, k, c, hw, hw)
    p = F.adaptive_avg_pool2d(x.reshape(n * k, c, hw, hw), (g, g))
    return p.reshape(n, k, c * g * g).numpy(), d["labels"], [int(t) for t in d["timesteps"]]


def _fold(x, y, tr, va, c):
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[tr]), y[tr])
    return float((m.predict(sc.transform(x[va])) == y[va]).mean())


def acc(x, y, c, n_jobs=5):
    jobs = [
        (tr, va) for s in SEEDS for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(x, y)
    ]
    return float(np.mean(Parallel(n_jobs=n_jobs)(delayed(_fold)(x, y, tr, va, c) for tr, va in jobs)))


def collapse(name, feats1, feats8, y, ts, c):
    eta = np.array([(t / 1000) / (1 - t / 1000) for t in ts])
    a1 = np.array([acc(feats1[:, i, :], y, c) for i in range(len(ts))])
    a8 = np.array([acc(feats8[:, i, :], y, c) for i in range(len(ts))])
    print(f"\n{name}  (C={c})", flush=True)
    print(f"  {'t':>5} {'eta':>6} {'ens1':>8} {'ens8':>8}", flush=True)
    for i, t in enumerate(ts):
        print(f"  {t:>5} {eta[i]:>6.2f} {a1[i]:>8.4f} {a8[i]:>8.4f}", flush=True)
    # Interpolate ens1's accuracy-vs-eta at ens8's rescaled etas; report in-range points.
    print(f"  {'ens8@t':>7} {'eta_eff':>8} {'ens8 acc':>9} {'ens1 interp':>12} {'residual':>10}", flush=True)
    resid = []
    for i, t in enumerate(ts):
        ee = eta[i] / SQRT8
        if ee < eta.min() or ee > eta.max():
            print(f"  {t:>7} {ee:>8.3f}  (outside measured ens1 range -- skipped)", flush=True)
            continue
        interp = float(np.interp(ee, eta, a1))
        r = a8[i] - interp
        resid.append(r)
        print(f"  {t:>7} {ee:>8.3f} {a8[i]:>9.4f} {interp:>12.4f} {r:>+10.4f}", flush=True)
    if resid:
        print(
            f"  mean residual over in-range points: {np.mean(resid):+.4f} "
            f"(0 = full collapse = pure input-SNR)",
            flush=True,
        )


def main():
    for ds in ("eurosat", "resisc45"):
        tmpl, g, c_raw = RAW[ds]
        f1, y, ts = repool(tmpl.format(m=1), g)
        f8, y8, _ = repool(tmpl.format(m=8), g)
        if not (np.array_equal(y, y8)):
            raise RuntimeError("assertion failed: np.array_equal(y, y8)")
        collapse(f"{ds} RAW x_t ({g}x{g}) -- positive control, MUST collapse", f1, f8, y, ts, c_raw)

        paths, c_dit = DIT[ds]
        d1, d8 = np.load(paths[1]), np.load(paths[8])
        if not (np.array_equal(d1["labels"], d8["labels"])):
            raise RuntimeError("assertion failed: np.array_equal(d1['labels'], d8['labels'])")
        collapse(
            f"{ds} DiT one-shot",
            d1["feats"],
            d8["feats"],
            d1["labels"],
            [int(t) for t in d1["timesteps"]],
            c_dit,
        )


if __name__ == "__main__":
    main()
