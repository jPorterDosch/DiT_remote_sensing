"""Bigearthnet threshold / F1-variant diagnostic (2026-10-31). Refits the SELECTED cell of three
arms (same view, C, scaler, solver as the reported number) and scores alternative decision
rules. Reported number = micro-F1 at sigmoid > 0.5. Val-tuned rules are legitimate (val is
disjoint from test); test-tuned ("oracle") rows are ceilings only, never reportable."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from eval import protocols as P
from eval import matrix as M

OUT = os.environ["OUT"]
DS = "m_bigearthnet"
ARMS = {
    "dinov3-sat": (
        {
            "name": "dinov3-sat",
            "kind": "dino",
            "features": "models/dino_feats/dinov3_vit7b16_sat_{dataset}_{split}.npz",
            "view": "cls4mp",
            "teacher": None,
        },
        "cls4mp",
        1e-3,
        0.6786,
    ),
    "dinov3-web": (
        {
            "name": "dinov3-web",
            "kind": "dino",
            "features": "models/dino_feats/dinov3_vit7b16_web_{dataset}_{split}.npz",
            "view": "cls4mp",
            "teacher": None,
        },
        "cls4mp",
        1e-3,
        0.6866,
    ),
    "flux-proj": (
        {
            "name": "flux-proj",
            "kind": "flux",
            "teacher": "dinov3-sat",
            "view": None,
            "features": "models/git10m_eval/git10m_flux_ca8e6ce7+42+proj-ffcd7963/{dataset}/*/multistep_{split}_feats_oneshot_g1.0_lora-proj_k*.npz",
            "expect": {
                "extraction_mode": "ONESHOT",
                "ensemble_size": 8,
                "weights": "flux-dev+lora:proj",
                "k": 28,
            },
        },
        "concat9t",
        1e-3,
        0.618,
    ),
}
GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)


def micro(y, p):
    return f1_score(y, p, average="micro", zero_division=0)


def tune_global(y, prob):
    scores = [(micro(y, (prob > t).astype(int)), t) for t in GRID]
    return max(scores)[1]


def tune_perclass(y, prob):
    """Coordinate ascent on micro-F1 (3 sweeps), start at 0.5; labels absent in y keep 0.5."""
    th = np.full(y.shape[1], 0.5)
    for _ in range(3):
        for j in range(y.shape[1]):
            if y[:, j].sum() == 0:
                continue
            best = (micro(y, (prob > th).astype(int)), th[j])
            for t in GRID:
                th2 = th.copy()
                th2[j] = t
                s = micro(y, (prob > th2).astype(int))
                if s > best[0]:
                    best = (s, t)
            th[j] = best[1]
    return th


def report(name, y, prob, th, note):
    pred = (prob > th).astype(int)
    r = {
        "arm": name,
        "rule": note,
        "micro_f1": micro(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y, pred, average="samples", zero_division=0),
    }
    print(json.dumps(r), flush=True)
    return r


rows = []
arm = sys.argv[1]
model, view, C, expect = ARMS[arm]
t0 = time.time()
cands, ytr, yva, yte, paths, inputs = M.load_cands(model, DS)
if view not in cands:
    raise SystemExit(f"{arm}: view {view} not among {list(cands)}")
Atr, Ava, Ate = cands[view]
print(f"{arm}: loaded {view} {Atr.shape} in {time.time() - t0:.0f}s", flush=True)
sc = StandardScaler().fit(Atr)
n_jobs = P.budget_ovr_jobs({view: cands[view]}, ytr)
with threadpool_limits(limits=M.blas_threads(len(Atr)), user_api="blas"):
    with P._shared_for_workers(lambda: sc.transform(Atr), ytr) as A:
        clf = P.official_lr(C, ytr).fit(A, ytr)
        pv, pt = clf.predict_proba(sc.transform(Ava)), clf.predict_proba(sc.transform(Ate))
        del A
print(f"{arm}: fit done in {time.time() - t0:.0f}s, ovr_n_jobs={n_jobs}", flush=True)
np.savez(
    os.path.join(OUT, f"{arm}_probs.npz"), prob_val=pv, prob_test=pt, y_val=yva, y_test=yte, view=view, C=C
)
r = report(arm, yte, pt, 0.5, "reported: fixed 0.5")
assert abs(r["micro_f1"] - expect) < 0.003, f"{arm}: refit {r['micro_f1']:.4f} != reported {expect}"
report(arm, yva, pv, 0.5, "val, fixed 0.5")
tg = tune_global(yva, pv)
report(arm, yte, pt, tg, f"val-tuned global threshold {tg}")
tc = tune_perclass(yva, pv)
report(arm, yte, pt, tc, "val-tuned per-class thresholds")
to = tune_global(yte, pt)
report(arm, yte, pt, to, f"ORACLE test-tuned global {to} (ceiling, not reportable)")
# macro-F1 with per-class thresholds tuned on val for MACRO (what a macro-reporting paper would do)
th = np.full(yva.shape[1], 0.5)
for j in range(yva.shape[1]):
    if yva[:, j].sum() == 0:
        continue
    th[j] = max((f1_score(yva[:, j], (pv[:, j] > t).astype(int), zero_division=0), t) for t in GRID)[1]
report(arm, yte, pt, th, "val-tuned per-class thresholds for macro-F1")
print("THRESH_DIAG_OK", arm, flush=True)
