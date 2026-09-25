"""Probe one feature arm under one protocol; cache per-image vectors; log to W&B.

    python -m eval.probe --protocol cv --dataset resisc45 --arm dinov2-clsmp \\
        --kind dino --features results/eval_feats/dinov2_vitl14_resisc45_n5000.npz --view clsmp
    python -m eval.probe --protocol cv --dataset resisc45 --arm flux-inv-sec13 --kind flux \\
        --features models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz \\
        --view sec13:1
    python -m eval.probe --protocol official --dataset m_eurosat --arm flux-oneshot-ens8 --kind flux \\
        --features 'models/m_eurosat_oneshot_ens8/*/multistep_{split}_feats_oneshot_g1.0.npz'

Protocols (eval/protocols.py): cv | budget | mlp (paired on the dataset's CV identity
list) and official (val-selected, one test eval). Output: results/eval/<run name>.npz with
one or more CELLS (cv: "cv"; budget: "b{k}_s{seed}"; mlp: "s{seed}"; official: "test"),
each a per-image correctness vector `correct__<cell>` over indices `ev__<cell>` into
`paths`. eval.compare pairs two such files cell by cell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # noqa: E402

from eval import features as F  # noqa: E402
from eval import protocols as P  # noqa: E402
from eval import wb  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", required=True, choices=["cv", "budget", "mlp", "official"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--arm", required=True, help="short arm label, part of the run name (e.g. dinov2-clsmp)")
    p.add_argument("--kind", required=True, choices=["flux", "dino", "vae"])
    p.add_argument("--features", required=True, help="feature .npz (official: pattern with {split})")
    p.add_argument("--view", default=None, help="see eval/features.py; official: comma-separated candidates")
    p.add_argument(
        "--identity", default=None, help="CV identity cache (default: features.CV_IDENTITY[dataset])"
    )
    p.add_argument("--pca", type=int, default=None, help="cv only: in-fold PCA width (diagnostic arms)")
    p.add_argument("--nested-c", type=float, nargs="+", default=None, help="cv only: in-fold nested C grid")
    p.add_argument("--budgets", type=int, nargs="+", default=list(P.BUDGETS))
    p.add_argument("--mlp-epochs", type=int, default=P.MLP_EPOCHS, help="smoke only; changes the run hash")
    p.add_argument("--max-images", type=int, default=None, help="SMOKE: strided cap; results not quotable")
    p.add_argument("--n-jobs", type=int, default=7)
    p.add_argument("--out-dir", default="results/eval")
    return p


# ----------------------------------------------------------------- paired protocols
def _paired_inputs(args):
    ident = args.identity or F.CV_IDENTITY.get(args.dataset)
    if ident is None:
        raise SystemExit(f"no CV identity registered for {args.dataset}; pass --identity")
    id_paths, y = F.load_identity(ident)
    d = F.load(args.features)
    F.check_identity(d, id_paths, args.features)
    mode, X = F.view(args.kind, d, args.view)
    keep = None
    if args.max_images:
        keep = np.linspace(0, len(y) - 1, args.max_images).astype(int)
        y = y[keep]
        id_paths = [id_paths[i] for i in keep]
        X = tuple(a[keep] for a in X) if mode == "sec13" else X[keep]
    return ident, id_paths, y, mode, X


def run_cv(args, y, mode, X):
    if mode == "sec13":
        if args.pca or args.nested_c:
            raise SystemExit("--pca/--nested-c apply to plain views only (sec13 has its own PCA-512 base)")
        cv, n_warn, picks = P.run_cv(y, P.fold_sec13, (*X, y), args.n_jobs)
    else:
        cv, n_warn, picks = P.run_cv(y, P.fold_plain, (X, y, P.C, args.pca, args.nested_c), args.n_jobs)
    extra = "" if len(set(picks)) == 1 else f"  C picked: {sorted(set(picks))}"
    flag = "CONVERGED" if n_warn == 0 else f"NOT-CONVERGED x{n_warn}/15"
    print(f"  {args.arm:<28s} acc {cv.mean():.4f}   [{flag}]{extra}", flush=True)
    return {"cv": (cv, np.arange(len(y)))}, {"conv_warnings": n_warn, "C_picks": sorted(set(picks))}


def _budget_cell(mode, X, y, b, s):
    tr, ev = P.budget_draw(y, b, s)
    c, nw = P.budget_fit_sec13(*X, y, tr, ev) if mode == "sec13" else P.budget_fit_plain(X, y, tr, ev)
    return b, s, c, ev, nw


def run_budget(args, y, mode, X):
    from joblib import Parallel, delayed

    # Cells run in joblib workers exactly as label_budget_curves did: loky caps BLAS threads
    # per worker, and a main-process fit (all BLAS threads) flips ~1/500 borderline
    # predictions at 100 labels/class -- measured by the budget gate, 2026-09-24.
    jobs = [(b, s) for b in args.budgets for s in P.SEEDS]
    out = Parallel(n_jobs=args.n_jobs)(delayed(_budget_cell)(mode, X, y, b, s) for b, s in jobs)
    cells, warn, rows = {}, 0, []
    for b, s, c, ev, nw in out:
        cells[f"b{b}_s{s}"] = (c, ev)
        warn += nw
        rows.append({"labels_per_class": b, "seed": s, "acc": float(c.mean()), "conv_warnings": nw})
        print(f"  budget {b:>3}/class seed {s}: acc {c.mean():.4f}" + ("" if nw == 0 else f" [NOT-CONVERGED x{nw}]"))
    wb.log_table("budget", rows)
    return cells, {"conv_warnings": warn}


def run_mlp(args, y, mode, X):
    import torch

    if mode != "plain":
        raise SystemExit("mlp protocol takes a plain view (e.g. flux concat, dino clsmp)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_eval = P.MLP_N_EVAL if not args.max_images else max(len(np.unique(y)) * 5, len(y) // 4)
    cells, lrs = {}, {}
    for s in P.SEEDS:
        c, ev, lr = P.run_mlp_seed(X, y, s, device, epochs=args.mlp_epochs, n_eval=n_eval)
        cells[f"s{s}"] = (c, ev)
        lrs[f"s{s}"] = lr
        print(f"  seed {s} lr*={lr} acc {c.mean():.4f}", flush=True)
    return cells, {"lr_star": lrs}


# ---------------------------------------------------------------- official protocol
def official_flux(args):
    spec = F.OFFICIAL[args.dataset]
    loaded = {
        s: F.load_flux_split(args.features, args.dataset, s, n, smoke=bool(args.max_images))
        for s, n in spec["sizes"].items()
    }
    (Xtr, ytr, ts, m_tr, _, h_tr), (Xva, yva, ts_v, m_va, _, h_va), (Xte, yte, ts_t, m_te, p_te, h_te) = (
        loaded["train"],
        loaded["val"],
        loaded["test"],
    )
    if not (ts == ts_v == ts_t):
        raise SystemExit(f"timestep mismatch across splits: {ts} {ts_v} {ts_t}")
    for key in (
        "ensemble_size",
        "extraction_mode",
        "img_size",
        "num_inversion_steps",
        "pooling",
        "degrade_to",
    ):
        if len({json.dumps(m.get(key)) for m in (m_tr, m_va, m_te)}) != 1:
            raise SystemExit(f"meta mismatch across splits on {key}")
    if m_tr.get("extraction_mode") == "ONESHOT":
        seeds = {m_tr.get("eps_seed"), m_va.get("eps_seed"), m_te.get("eps_seed")}
        if len(seeds) != 3:
            raise SystemExit(f"eps seeds not pairwise distinct across splits: {seeds} (6t audit F1)")

    def make(X, ti):
        return X.reshape(X.shape[0], -1) if ti is None else X[:, ti, :]

    names = [(f"t{t}", i) for i, t in enumerate(ts)] + [(f"concat{len(ts)}t", None)]
    if args.view and args.view != "all":
        want = args.view.split(",")
        names = [n for n in names if n[0] in want]
    cands = {n: (make(Xtr, i), make(Xva, i), make(Xte, i)) for n, i in names}
    inputs = h_tr + h_va + h_te
    return cands, ytr, yva, yte, p_te, inputs, {"timesteps": ts, "meta_train": m_tr}


def official_dino(args):
    data = {}
    for s in F.OFFICIAL[args.dataset]["sizes"]:
        path = args.features.format(split=s)
        d = F.load(path)
        files, y = F.list_official_split(args.dataset, s)
        F.check_identity(d, files, path)
        data[s] = (d, y, files, path)
    names = (args.view or "cls,clsmp").split(",")
    cands = {n: tuple(F.view("dino", data[s][0], n)[1] for s in ("train", "val", "test")) for n in names}
    return (
        cands,
        data["train"][1],
        data["val"][1],
        data["test"][1],
        data["test"][2],
        [data[s][3] for s in data],
        {},
    )


def run_official(args):
    from sklearn.metrics import f1_score

    if args.dataset not in F.OFFICIAL:
        raise SystemExit(f"{args.dataset}: no OFFICIAL entry (count the shipped partition first, rule 16)")
    loader = {"flux": official_flux, "dino": official_dino}.get(args.kind)
    if loader is None:
        raise SystemExit(f"official protocol supports kinds flux, dino (got {args.kind})")
    cands, ytr, yva, yte, test_paths, inputs, extra = loader(args)
    for p in inputs:
        wb.use_features(p)
    correct, pred, sel, table = P.run_official(cands, ytr, yva, yte)
    for r in table:
        print(
            f"  val {r['candidate']:>9} C={r['C']:<5} acc={r['val_acc']:.4f}"
            + ("" if r["conv_warnings"] == 0 else f"  [NOT CONVERGED x{r['conv_warnings']}]")
        )
    n_cls = len(F.official_classes(args.dataset))
    f1 = float(f1_score(yte, pred, average="macro", labels=np.arange(n_cls)))
    top1 = float(correct.mean())
    se = (top1 * (1 - top1) / len(yte)) ** 0.5
    print(f"  SELECTED on val: {sel['candidate']}, C={sel['C']} (val {sel['val_acc']:.4f})")
    print(
        f"  TEST: top1={top1:.4f} ±{1.96 * se:.4f}  macro_f1={f1:.4f}"
        + ("" if sel["test_warn"] == 0 else f"  [NOT CONVERGED x{sel['test_warn']}]")
    )
    wb.log_table("val_selection", table)
    info = {"selected": sel, "macro_f1": f1, "conv_warnings": sel["test_warn"], **extra}
    return {"test": (correct, np.arange(len(yte)))}, info, test_paths, yte


# ------------------------------------------------------------------------------ main
PROTOCOL_CONSTANTS = {
    "cv": {"C": P.C, "seeds": P.SEEDS, "folds": 5},
    "budget": {"C": P.C, "seeds": P.SEEDS},
    "mlp": {
        "hidden": P.MLP_HIDDEN,
        "batch": P.MLP_BATCH,
        "wd": P.MLP_WD,
        "lr_grid": P.MLP_LR_GRID,
        "n_eval": P.MLP_N_EVAL,
    },
    "official": {"C_grid": P.C_GRID, "max_iter": P.OFFICIAL_MAX_ITER},
}


def main(argv=None) -> str:
    args = _parser().parse_args(argv)
    if args.protocol != "official" and args.view is None:
        raise SystemExit("--view is required for cv/budget/mlp (see eval/features.py)")
    feat_ids = (
        [F.OFFICIAL.get(args.dataset, {}).get("root"), args.features]
        if args.protocol == "official"
        else wb.file_identity(args.features)
    )
    config = {
        "protocol": args.protocol,
        "protocol_constants": PROTOCOL_CONSTANTS[args.protocol],
        "kind": args.kind,
        "view": args.view,
        "features": feat_ids,
        "identity": args.identity or F.CV_IDENTITY.get(args.dataset),
        "pca": args.pca,
        "nested_c": args.nested_c,
        "budgets": args.budgets if args.protocol == "budget" else None,
        "mlp_epochs": args.mlp_epochs if args.protocol == "mlp" else None,
        "max_images": args.max_images,
    }
    exp = f"probe-{args.protocol}" + ("-smoke" if args.max_images else "")
    _, name = wb.init(exp, args.dataset, args.arm, config, job_type="probe")
    print(f"== {name}")

    if args.protocol == "official":
        cells, info, paths, labels = run_official(args)
    else:
        _, paths, labels, mode, X = _paired_inputs(args)
        wb.use_features(args.features)
        runner = {"cv": run_cv, "budget": run_budget, "mlp": run_mlp}[args.protocol]
        cells, info = runner(args, labels, mode, X)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    np.savez(
        out,
        cells=np.array(list(cells)),
        **{f"correct__{c}": v for c, (v, _) in cells.items()},
        **{f"ev__{c}": e for c, (_, e) in cells.items()},
        paths=np.array(paths),
        labels=labels,
        protocol=np.array(args.protocol),
        arm=np.array(args.arm),
        dataset=np.array(args.dataset),
        run_name=np.array(name),
        config=np.array(json.dumps(config, default=str)),
        info=np.array(json.dumps(info, default=str)),
        smoke=np.array(bool(args.max_images)),
    )
    print(f"per-image vectors cached to {out}")
    if wandb.run is not None:
        for c, (v, _) in cells.items():
            wandb.run.summary[f"acc/{c}"] = float(v.mean())
        wandb.run.summary["acc/mean_over_cells"] = float(np.mean([v.mean() for v, _ in cells.values()]))
        wandb.run.summary["n_images"] = len(labels)
        for k, v in info.items():
            wandb.run.summary[k] = v if isinstance(v, (int, float, str)) else json.dumps(v, default=str)
        wb.log_results(out)
        wandb.finish()
    return out


if __name__ == "__main__":
    main()
