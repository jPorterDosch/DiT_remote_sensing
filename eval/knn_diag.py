"""Representation diagnostics without a fitted head (P2, 2026-10-24): does adaptation move the
student's neighbourhood structure toward the teacher's?

For every (model, dataset, candidate) on the official TEST split, with the features of the matrix
protocol (eval.matrix.load_cands: block-k pooled tokens per timestep + the 9-t concat for FLUX
arms, the DINO views for the teachers), z-scored per dimension on the split itself (label-free) and
compared by cosine:
  purity@k      leave-one-out fraction of the k nearest neighbours sharing the image's label
                (multi-label: mean label-Jaccard to the neighbours);
  knn_acc@k     leave-one-out majority-vote accuracy (single-label datasets only);
  agree@k       mean Jaccard overlap between the image's k-NN set in the model's space and in the
                teacher's space (DINOv3-sat and DINOv3-web, first official view), chance = k/(N-1);
  rank_corr     Spearman correlation between the model's and the teacher's cosine-similarity rows
                (full N x N structure, not just the top-k), mean over images.
  pr            participation ratio of the z-scored features, (sum lambda)^2 / sum lambda^2 of the
                covariance eigenvalues: the effective dimensionality the probe sees;
  dim90         number of principal components holding 90 % of the variance.
k is fixed in advance (10 and 20, both reported; no selection), so nothing is tuned on the data it
scores. Writes results/knn/knn_diag.json and knn_diag.md (one row per model x dataset x candidate).

Cross-arm test (the "missing test", 2026-10-24): for every adapted model with a 100 %-label matrix
cell (results/matrix/cells/<model>/<dataset>/f1.0_s0.npz) the change of test score over frozen FLUX
is set against the change of (i) agreement with the model's OWN teacher, (ii) purity, (iii) pr, at
the t100 and concat candidates; Spearman over models per dataset and pooled over datasets (z-scored
per dataset). Probe gain tracking teacher agreement across arms = the gain is the teacher's
similarity structure; tracking pr = a geometric (dimensionality) mechanism. Correlational across
arms; the causal evidence is the sem / stop runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from eval import features as F  # noqa: E402
from eval import matrix as M  # noqa: E402

KS = (10, 20)
DEFAULT_MODELS = {
    "eval/matrix_specs/phase2.json": [
        "dinov3-sat",
        "flux-pretrained-k28",
        "flux-fm",
        "flux-naive",
        "flux-proj",
    ],
    "eval/matrix_specs/phase1.json": ["dinov3-web"],
    "eval/matrix_specs/phase3_ablations.json": [
        "flux-proj-s43",
        "flux-fm-s43",
        "flux-proj-cd0",
        "flux-proj-raw",
        "flux-proj-ln",
        "flux-proj-shufbatch",
    ],
}
TEACHERS = ("dinov3-sat", "dinov3-web")
FROZEN = "flux-pretrained-k28"
SPEC_OF: dict[str, str] = {}  # model name -> the matrix spec it was loaded from (load_models)
CROSS_CANDS = ("t100", "concat9t")
# the shuffle control's pv4ger cache holds a NaN image (audit 2026-10-21): no cells, no diagnostics
EXCLUDE = {("flux-proj-shufbatch", "m_pv4ger")}


def participation(X: np.ndarray) -> tuple[float, int]:
    Z = X.astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    # eigenvalues of the covariance via the Gram matrix (N << D for the concat candidate)
    g = Z @ Z.T / (len(Z) - 1)
    lam = np.clip(np.linalg.eigvalsh(g), 0, None)[::-1]
    pr = float(lam.sum() ** 2 / max((lam**2).sum(), 1e-30))
    c = np.cumsum(lam) / max(lam.sum(), 1e-30)
    return pr, int(np.searchsorted(c, 0.9) + 1)


def matrix_score(model: dict, dataset: str, spec_path: str) -> float | None:
    """The 100 % test score of one matrix cell, validated as `eval.matrix.collect` validates it
    (review 2026-10-27): the cell's protocol config must be the spec's current one and its stored
    feature sha1s must match the feature files on disk; a stale cell is refused, not read. Absent
    cell -> None (the point is dropped, said aloud by the caller)."""
    spec = M.load_spec(spec_path)
    c_grid = tuple(float(c) for c in spec.get("c_grid", M.DEFAULT_C_GRID))
    seed0 = int(spec.get("seeds", M.DEFAULT_SEEDS)[0])
    p = M.cell_path("results/matrix", model["name"], dataset, 1.0, seed0)
    if not os.path.exists(p):
        return None
    c = M._load_cell(p)
    cfg = json.loads(json.dumps(M.cell_config(model, dataset, 1.0, seed0, c_grid)))
    if not M.config_current(c["config"], cfg, c["compute"], c["labels"].ndim == 2):
        raise SystemExit(
            f"{p}: stale cell (protocol/config differs from {spec_path}); not used for the cross-arm test"
        )
    present, _ = M.features_present(model, dataset)
    if present and c["inputs"] != M.inputs_sha(M.input_files(model, dataset)):
        raise SystemExit(
            f"{p}: stale cell (feature files changed since it was computed); not used for the cross-arm test"
        )
    return c["score"]


def cross_arm(rows: list[dict], models: list[dict]) -> dict:
    """Delta over frozen FLUX per (model, dataset): matrix 100 % score vs the diagnostics."""
    from scipy.stats import spearmanr

    teacher_of = {m["name"]: m.get("teacher") for m in models}
    model_of = {m["name"]: m for m in models}
    by = {(r["model"], r["dataset"], r["candidate"]): r for r in rows}
    pts = []
    for (mn, ds, c), r in by.items():
        if c not in CROSS_CANDS or mn == FROZEN or not teacher_of.get(mn) or (mn, ds) in EXCLUDE:
            continue
        f = by.get((FROZEN, ds, c))
        sm, sf = (
            matrix_score(model_of[mn], ds, SPEC_OF[mn]),
            matrix_score(model_of[FROZEN], ds, SPEC_OF[FROZEN]),
        )
        if f is None or sf is None:
            raise SystemExit(
                f"cross_arm: frozen reference missing for {ds} / {c} (diag row {f is not None}, 100 % cell {sf is not None})"
            )
        if sm is None:
            print(
                f"WARNING cross_arm: no 100 % matrix cell for {mn} / {ds} (results/matrix/cells/{mn}/{ds}/f1_s0.npz); point dropped from the cross-arm correlation",
                flush=True,
            )
            continue
        t = teacher_of[mn]
        pts.append(
            {
                "model": mn,
                "dataset": ds,
                "candidate": c,
                "teacher": t,
                "d_score": sm - sf,
                "d_agree_teacher@10": r[f"agree_{t}@10"] - f[f"agree_{t}@10"],
                "d_rank_corr_teacher": r[f"rank_corr_{t}"] - f[f"rank_corr_{t}"],
                "d_purity@10": r["purity@10"] - f["purity@10"],
                "d_pr": r["pr"] - f["pr"],
                "d_dim90": r["dim90"] - f["dim90"],
            }
        )
    keys = ["d_agree_teacher@10", "d_rank_corr_teacher", "d_purity@10", "d_pr", "d_dim90"]
    out = {"points": pts, "per_dataset": {}, "pooled": {}}

    def corr(sub, key):
        x = np.array([p[key] for p in sub])
        y = np.array([p["d_score"] for p in sub])
        if len(sub) < 4 or np.std(x) == 0 or np.std(y) == 0:
            return {"rho": None, "p": None, "n": len(sub)}
        r = spearmanr(x, y)
        return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(sub)}

    for c in CROSS_CANDS:
        for ds in sorted({p["dataset"] for p in pts}):
            sub = [p for p in pts if p["dataset"] == ds and p["candidate"] == c]
            out["per_dataset"][f"{ds}|{c}"] = {k: corr(sub, k) for k in keys}
        # pooled: z-score every delta within its dataset so no dataset's scale dominates
        sub = [dict(p) for p in pts if p["candidate"] == c]
        for ds in {p["dataset"] for p in sub}:
            idx = [i for i, p in enumerate(sub) if p["dataset"] == ds]
            for k in keys + ["d_score"]:
                v = np.array([sub[i][k] for i in idx])
                sd = v.std() if v.std() > 0 else 1.0
                for i, z in zip(idx, (v - v.mean()) / sd):
                    sub[i][k] = float(z)
        out["pooled"][c] = {k: corr(sub, k) for k in keys}
        # model-level statistic (rule 4: points sharing a model are not independent): the per-model
        # mean of the within-dataset z-scored deltas, Spearman over models, plus the leave-one-model-out
        # range of the pooled rho (the shuffle control is an outlier that can carry the pooled value)
        mods = sorted({p["model"] for p in sub})
        agg = [
            {
                "model": m,
                **{k: float(np.mean([p[k] for p in sub if p["model"] == m])) for k in keys + ["d_score"]},
            }
            for m in mods
        ]
        out.setdefault("model_level", {})[c] = {k: corr(agg, k) for k in keys}
        loo = {}
        for k in keys:
            rs = [corr([p for p in sub if p["model"] != m], k)["rho"] for m in mods]
            rs = [r for r in rs if r is not None]
            loo[k] = {"min": min(rs), "max": max(rs), "n_models": len(mods)} if rs else None
        out.setdefault("leave_one_model_out", {})[c] = loo
    return out


def zscore_cos(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)  # leave-one-out
    return S


def knn_sets(S: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(-S, k, axis=1)[:, :k]


def purity(nn: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    if y.ndim == 1:
        same = (y[nn] == y[:, None]).mean(1)
        votes = np.array([np.bincount(y[row], minlength=int(y.max()) + 1).argmax() for row in nn])
        return float(same.mean()), float((votes == y).mean())
    yb = y.astype(bool)
    inter = (yb[nn] & yb[:, None, :]).sum(-1)
    union = (yb[nn] | yb[:, None, :]).sum(-1)
    return float(np.where(union > 0, inter / np.maximum(union, 1), 1.0).mean()), None


def jaccard_sets(a: np.ndarray, b: np.ndarray) -> float:
    out = []
    for ra, rb in zip(a, b):
        sa, sb = set(ra.tolist()), set(rb.tolist())
        out.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(out))


def row_rank_corr(Sa: np.ndarray, Sb: np.ndarray) -> float:
    """Mean over images of Spearman(rho) between the two similarity rows (diagonal excluded)."""
    n = Sa.shape[0]
    mask = ~np.eye(n, dtype=bool)
    A = Sa[mask].reshape(n, n - 1)
    B = Sb[mask].reshape(n, n - 1)
    ra = A.argsort(1).argsort(1).astype(np.float64)
    rb = B.argsort(1).argsort(1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return float(((ra * rb).sum(1) / np.sqrt((ra**2).sum(1) * (rb**2).sum(1))).mean())


def load_models(specs: dict[str, list[str]]) -> list[dict]:
    out, seen = [], set()
    for path, names in specs.items():
        spec = M.load_spec(path)
        by = {m["name"]: m for m in spec["models"]}
        for n in names:
            if n not in by:
                raise SystemExit(f"{path}: no model {n!r}")
            if n in seen:
                raise SystemExit(f"model {n!r} listed twice")
            seen.add(n)
            SPEC_OF[n] = path  # the spec it came from: matrix_score re-validates the cell against it
            out.append(by[n])
    return out


def test_features(model: dict, dataset: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    cands, _, _, yte, paths, _ = M.load_cands(model, dataset)
    te = {c: v[2] for c, v in cands.items()}
    if model["kind"] == "dino":
        first = F.DINO_OFFICIAL_VIEWS.split(",")[0]
        te = {first: te[first]}
    return te, np.asarray(yte), [str(p) for p in paths]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["m_eurosat", "m_brick_kiln", "m_pv4ger", "m_so2sat", "m_forestnet", "m_bigearthnet"],
    )
    ap.add_argument(
        "--spec-models",
        nargs="*",
        default=None,
        help="spec.json:model1,model2 (default: the phase-2 arms, both teachers, the seed-43 repeats)",
    )
    ap.add_argument("--out-dir", default="results/knn")
    ap.add_argument(
        "--merge",
        nargs="*",
        default=None,
        help="per-dataset knn_diag.json files: pool their rows, redo the cross-arm test, write to --out-dir (no features loaded)",
    )
    a = ap.parse_args(argv)
    if a.merge is not None:
        rows = [r for f in a.merge for r in json.load(open(f))["rows"]]
        models = load_models(
            DEFAULT_MODELS
            if not a.spec_models
            else {s.split(":")[0]: s.split(":")[1].split(",") for s in a.spec_models}
        )
        os.makedirs(a.out_dir, exist_ok=True)
        write_outputs(a.out_dir, rows, models)
        print("KNN_DIAG_OK", len(rows), "rows (merged)")
        return 0
    specs = (
        DEFAULT_MODELS
        if not a.spec_models
        else {s.split(":")[0]: s.split(":")[1].split(",") for s in a.spec_models}
    )
    models = load_models(specs)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for ds in a.datasets:
        feats, labels, paths = {}, None, None
        for m in models:
            ok, why = M.features_present(m, ds)
            if not ok:
                print(f"skip {m['name']} on {ds}: {why}", flush=True)
                continue
            te, y, p = test_features(m, ds)
            if paths is None:
                labels, paths = y, p
            elif p != paths:
                raise SystemExit(
                    f"{m['name']} on {ds}: test image list differs from {models[0]['name']} (rule 6)"
                )
            feats[m["name"]] = te
            print(f"loaded {m['name']} {ds}: {sorted(te)} N={len(p)}", flush=True)
        if not any(t in feats for t in TEACHERS):
            raise SystemExit(f"{ds}: no teacher features loaded")
        n = len(paths)
        S = {(mn, c): zscore_cos(X) for mn, cs in feats.items() for c, X in cs.items()}
        teacher_S = {t: next(iter(v for (mn, _), v in S.items() if mn == t)) for t in TEACHERS if t in feats}
        teacher_nn = {(t, k): knn_sets(Sx, k) for t, Sx in teacher_S.items() for k in KS}
        for (mn, c), Sx in S.items():
            if (mn, ds) in EXCLUDE:
                continue
            rec = {"model": mn, "dataset": ds, "candidate": c, "n": n, "multilabel": labels.ndim == 2}
            rec["pr"], rec["dim90"] = participation(feats[mn][c])
            for k in KS:
                nn = knn_sets(Sx, k)
                pur, acc = purity(nn, labels)
                rec[f"purity@{k}"] = pur
                rec[f"knn_acc@{k}"] = acc
                rec[f"chance_agree@{k}"] = k / (n - 1)
                for t in teacher_S:
                    rec[f"agree_{t}@{k}"] = 1.0 if mn == t else jaccard_sets(nn, teacher_nn[(t, k)])
            for t, St in teacher_S.items():
                rec[f"rank_corr_{t}"] = 1.0 if mn == t else row_rank_corr(Sx, St)
            rows.append(rec)
            print(
                f"{ds:16s} {mn:22s} {c:9s} pr {rec['pr']:6.1f} dim90 {rec['dim90']:4d} purity@10 {rec['purity@10']:.3f} acc@10 {rec['knn_acc@10'] if rec['knn_acc@10'] is not None else float('nan'):.3f} "
                + " ".join(
                    f"agree_{t.split('-')[1]}@10 {rec[f'agree_{t}@10']:.3f} rc_{t.split('-')[1]} {rec[f'rank_corr_{t}']:.3f}"
                    for t in teacher_S
                ),
                flush=True,
            )
    write_outputs(a.out_dir, rows, models)
    print("KNN_DIAG_OK", len(rows), "rows")
    return 0


def write_outputs(out_dir: str, rows: list[dict], models: list[dict]) -> None:
    xa = cross_arm(rows, models)
    with open(os.path.join(out_dir, "knn_diag.json"), "w") as f:
        json.dump(
            {
                "ks": KS,
                "split": "test",
                "standardization": "z-score per dim on the split, cosine",
                "rows": rows,
                "cross_arm": xa,
            },
            f,
            indent=1,
        )
    md = [
        "| dataset | model | candidate | pr | dim90 | purity@10 | knn_acc@10 | agree sat@10 | agree web@10 | chance@10 | rank corr sat | rank corr web |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:

        def g(key, r=r):
            return "—" if r.get(key) is None else f"{r[key]:.3f}"

        md.append(
            f"| {r['dataset']} | {r['model']} | {r['candidate']} | {r['pr']:.1f} | {r['dim90']} | {g('purity@10')} | {g('knn_acc@10')} | {g('agree_dinov3-sat@10')} | {g('agree_dinov3-web@10')} | {g('chance_agree@10')} | {g('rank_corr_dinov3-sat')} | {g('rank_corr_dinov3-web')} |"
        )
    md += [
        "",
        "### Cross-arm: delta over frozen FLUX, Spearman(delta test score, delta diagnostic)",
        "",
        "| scope | candidate | agree teacher | rank corr teacher | purity | pr | dim90 | n |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(c):
        return "—" if c["rho"] is None else f"{c['rho']:+.2f} (p {c['p']:.2f})"

    for key, d in (
        list(xa["per_dataset"].items())
        + [(f"pooled|{c}", v) for c, v in xa["pooled"].items()]
        + [(f"model-level|{c}", v) for c, v in xa.get("model_level", {}).items()]
    ):
        scope, c = key.split("|")
        md.append(
            f"| {scope} | {c} | {fmt(d['d_agree_teacher@10'])} | {fmt(d['d_rank_corr_teacher'])} | {fmt(d['d_purity@10'])} | {fmt(d['d_pr'])} | {fmt(d['d_dim90'])} | {d['d_pr']['n']} |"
        )
        print(
            f"CROSS {scope:16s} {c:9s} agree {fmt(d['d_agree_teacher@10'])} rankcorr {fmt(d['d_rank_corr_teacher'])} purity {fmt(d['d_purity@10'])} pr {fmt(d['d_pr'])} n={d['d_pr']['n']}",
            flush=True,
        )

    def fl(v):
        return "—" if not v else f"{v['min']:+.2f} .. {v['max']:+.2f}"

    for c, loo in xa.get("leave_one_model_out", {}).items():
        md.append(
            f"| pooled, leave-one-model-out range | {c} | {fl(loo['d_agree_teacher@10'])} | {fl(loo['d_rank_corr_teacher'])} | {fl(loo['d_purity@10'])} | {fl(loo['d_pr'])} | {fl(loo['d_dim90'])} | {(loo['d_pr'] or {}).get('n_models', '—')} models |"
        )
        print(
            f"CROSS LOO {c:9s} agree {fl(loo['d_agree_teacher@10'])} rankcorr {fl(loo['d_rank_corr_teacher'])} purity {fl(loo['d_purity@10'])}",
            flush=True,
        )
    md.append("")
    md.append(
        "pooled = all (model, dataset) points, within-dataset z-scored (points sharing a model are NOT independent, rule 4); model-level = Spearman over per-model means (independent units); the leave-one-model-out range shows how far one model (the shuffle control) carries the pooled value."
    )
    with open(os.path.join(out_dir, "knn_diag.md"), "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
