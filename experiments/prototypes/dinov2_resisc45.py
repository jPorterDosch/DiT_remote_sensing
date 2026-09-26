"""6ac: DINOv2 ViT-L/14 vs frozen FLUX on RESISC45, paired on the IDENTICAL 5,000 images.

Falsification test for 6ab finding 1 ("web-scale pretraining, not generative modelling,
separates the top of SatDiFuser's table"): m-eurosat was saturated and could not rank
extractors, RESISC45 has ~4 pts of headroom under this instrument, so it can.

Identity: the images are the exact file paths stored in the section-13 inversion cache, in
cache order, so every arm consumes the same rows in the same order and StratifiedKFold
(which depends only on n, y and random_state) yields byte-identical folds -> per-image
PAIRED deltas. Guards for all of that run before any probe. See RESEARCH_NOTES 6ac.

Probe = the section-13 instrument: 3 seeds x 5-fold, in-fold StandardScaler,
LogisticRegression(C=0.1, max_iter=2000). Convergence is counted and printed per arm
(CLAUDE.md rule 8). Per-image correctness vectors are cached (rule 12).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from experiments.prototypes._absorption_harness import (  # noqa: E402
    C as HARNESS_C,
    RESISC45_BASE,
    RESISC45_TBEST_IDX,
    SEEDS,
    ci,
    ditf,
)
from data.resisc45_dataset import RESISC45_CLASSES, RESISC45Dataset  # noqa: E402
from tasks.extraction import _stratified_indices  # noqa: E402

ENS8_CACHE = "models/n5000_resisc45_oneshot_ens8/resisc45_flux_4118f153+42/multistep_train_feats_oneshot_g1.0.npz"
ENS8_T260_IDX = 2  # timesteps [100,180,260,...]
C_GRID = (0.01, 0.1, 1.0)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- extraction
@torch.no_grad()
def extract(model, paths, device, batch=64, tag=""):
    cls_all, mp_all = [], []
    for i in range(0, len(paths), batch):
        imgs = []
        for f in paths[i : i + batch]:
            im = Image.open(f).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        x = ((torch.stack(imgs) - IMAGENET_MEAN) / IMAGENET_STD).to(device)
        out = model.forward_features(x)
        cls_all.append(out["x_norm_clstoken"].float().cpu().numpy())
        mp_all.append(out["x_norm_patchtokens"].mean(dim=1).float().cpu().numpy())
        if i % (batch * 20) == 0:
            print(f"  {tag}{i}/{len(paths)}", flush=True)
    return np.concatenate(cls_all), np.concatenate(mp_all)


def cached_extract(cache, paths, tag=""):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        assert list(d["paths"]) == list(paths), f"{cache}: cached paths differ from requested"
        print(f"  reusing {cache}")
        return d["cls"], d["mp"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False).to(device).eval()
    cls_f, mp_f = extract(model, paths, device, tag=tag)
    del model
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, cls=cls_f, mp=mp_f, paths=np.array(paths))
    return cls_f, mp_f


# ---------------------------------------------------------------- probes
def _fold_plain(X, y, C, pca, nested, tr, va):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(X[tr])
        a, b = sc.transform(X[tr]), sc.transform(X[va])
        if pca:
            k = min(pca, a.shape[1], len(tr) - 1)
            if k < a.shape[1]:
                p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
                a, b = p.transform(a), p.transform(b)
        if nested:
            # Selection touches ONLY the training rows of this outer fold (rule 1).
            best = None
            for Ci in nested:
                accs = []
                for itr, iva in StratifiedKFold(3, shuffle=True, random_state=0).split(a, y[tr]):
                    mi = LogisticRegression(C=Ci, max_iter=2000).fit(a[itr], y[tr][itr])
                    accs.append(float((mi.predict(a[iva]) == y[tr][iva]).mean()))
                s = float(np.mean(accs))
                if best is None or s > best[0]:
                    best = (s, Ci)
            C = best[1]
        m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
        pred = m.predict(b)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return va, pred, C, n_warn


def _fold_flux(base, extra, y, tr, va):
    """Section-13 base treatment: PCA-512-protected t-best + others appended raw."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(base[tr])
        a, b = sc.transform(base[tr]), sc.transform(base[va])
        k = min(512, a.shape[1], len(tr) - 1)
        if k < a.shape[1]:
            p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
            a, b = p.transform(a), p.transform(b)
        if extra is not None:
            s2 = StandardScaler().fit(extra[tr])
            a = np.hstack([a, s2.transform(extra[tr])])
            b = np.hstack([b, s2.transform(extra[va])])
        m = LogisticRegression(C=HARNESS_C, max_iter=2000).fit(a, y[tr])
        pred = m.predict(b)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return va, pred, HARNESS_C, n_warn


def run_arm(name, y, fold_fn, fold_args, n_jobs=7):
    out = np.zeros(len(y))
    warn_tot, picks, fold_sets = 0, [], []
    for s in SEEDS:
        jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros((len(y), 1)), y))
        fold_sets.append([tuple(va) for _, va in jobs])
        for va, pred, C, nw in Parallel(n_jobs=n_jobs)(
                delayed(fold_fn)(*fold_args, tr, va) for tr, va in jobs):
            out[va] += pred == y[va]
            warn_tot += nw
            picks.append(C)
    acc = out.mean() / len(SEEDS)
    cv = out / len(SEEDS)
    flag = "CONVERGED" if warn_tot == 0 else f"NOT-CONVERGED x{warn_tot}/15"
    extra = "" if len(set(picks)) == 1 else f"  C picked: {sorted(set(picks))}"
    print(f"  {name:<34s} acc {acc:.4f}   [{flag}]{extra}", flush=True)
    return cv, fold_sets, warn_tot


def paired(name, a, b):
    """b - a per image, image-level bootstrap (rule 4)."""
    d = b - a
    lo, hi = ci(d)
    verdict = "PARITY (CI spans 0)" if lo <= 0 <= hi else ("B > A" if lo > 0 else "A > B")
    print(f"  {name}: delta {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  -> {verdict}")
    return {"delta": float(d.mean()), "ci": (lo, hi), "verdict": verdict}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--max-images", type=int, default=None, help="smoke cap (strided)")
    p.add_argument("--full-split", action="store_true", help="secondary reference run")
    p.add_argument("--n-jobs", type=int, default=7)
    args = p.parse_args()

    # ---- identity guards (must be able to fail; rule 6) -------------------
    inv = np.load(RESISC45_BASE)
    paths = [str(x) for x in inv["paths"]]
    y = inv["labels"].astype(np.int64)
    idx = inv["subset_indices"]
    train_ds = RESISC45Dataset(args.path, split="train", img_size=256)
    all_labels = np.array([c for _, c, _ in train_ds.samples])
    recomputed = _stratified_indices(all_labels, 5000, 42)
    assert np.array_equal(recomputed, idx), "subset_indices do not reproduce from _stratified_indices(42)"
    assert [train_ds.samples[i][0] for i in idx] == paths, "cached paths != train-split paths at those indices"
    dir_labels = np.array([RESISC45_CLASSES.index(os.path.basename(os.path.dirname(f))) for f in paths])
    assert np.array_equal(dir_labels, y), "labels derived from directories != cached labels"
    assert all(os.path.exists(f) for f in paths), "missing image file"
    ens8 = np.load(ENS8_CACHE)
    assert np.array_equal(ens8["labels"], y) and [str(x) for x in ens8["paths"]] == paths, "ens8 cache identity"
    print(f"IDENTITY OK: 5,000 paths reproduce, labels agree 3 ways, ens8 cache aligned "
          f"({len(np.unique(y))} classes)")

    keep = None
    if args.max_images:
        keep = np.linspace(0, len(paths) - 1, args.max_images).astype(int)
        paths = [paths[i] for i in keep]
        y = y[keep]
        print(f"SMOKE: {len(paths)} images — accuracies are meaningless at this scale")

    # ---- features --------------------------------------------------------
    suffix = f"_smoke{args.max_images}" if args.max_images else ""
    cls_f, mp_f = cached_extract(f"results/dinov2_resisc45_feats_n5000{suffix}.npz", paths)
    dino = {"cls": cls_f, "clsmp": np.concatenate([cls_f, mp_f], axis=1)}
    fi = ditf(inv["feats"], inv["mods"])
    base = fi[:, RESISC45_TBEST_IDX, :]
    others = np.concatenate([fi[:, i, :] for i in range(fi.shape[1]) if i != RESISC45_TBEST_IDX], axis=1)
    ens8_t260 = ditf(ens8["feats"], ens8["mods"])[:, ENS8_T260_IDX, :]
    if keep is not None:
        base, others, ens8_t260 = base[keep], others[keep], ens8_t260[keep]
    print(f"widths: FLUX base {base.shape[1]}(->PCA512) + others {others.shape[1]} | "
          f"ens8-t260 {ens8_t260.shape[1]} | DINOv2 cls {dino['cls'].shape[1]} clsmp {dino['clsmp'].shape[1]}")

    # ---- arms ------------------------------------------------------------
    print("\nARMS (3 seeds x 5-fold, in-fold scaler, LR C=0.1 unless noted):")
    A1, folds_a1, _ = run_arm("A1 FLUX inversion base (sec-13)", y,
                              _fold_flux, (base, others, y), args.n_jobs)
    A2, _, _ = run_arm("A2 FLUX ens8 t260 (3072d, no PCA)", y,
                       _fold_plain, (ens8_t260, y, HARNESS_C, None, None), args.n_jobs)
    B1, folds_b1, _ = run_arm("B1 DINOv2 clsmp 2048d  [PRIMARY]", y,
                              _fold_plain, (dino["clsmp"], y, HARNESS_C, None, None), args.n_jobs)
    B2, _, _ = run_arm("B2 DINOv2 cls 1024d", y,
                       _fold_plain, (dino["cls"], y, HARNESS_C, None, None), args.n_jobs)
    B3, _, _ = run_arm("B3 DINOv2 clsmp nested-C", y,
                       _fold_plain, (dino["clsmp"], y, None, None, C_GRID), args.n_jobs)
    B1p, _, _ = run_arm("B1p DINOv2 clsmp +PCA512 (diag)", y,
                        _fold_plain, (dino["clsmp"], y, HARNESS_C, 512, None), args.n_jobs)
    assert folds_a1 == folds_b1, "FOLDS DIFFER between arms — the paired statistic is invalid"
    # The pairing rests on the claim "StratifiedKFold.split ignores X". Test that claim on
    # the actual matrices, and FIRE-TEST the test by feeding it the mismatch it exists to
    # catch (a permuted label vector must change the folds), per CLAUDE.md rule 6.
    def _folds(X, yy, seed=0):
        return [tuple(va) for _, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy)]
    f_flux, f_dino = _folds(base, y), _folds(dino["clsmp"], y)
    f_perm = _folds(base, y[np.random.default_rng(0).permutation(len(y))])
    assert f_flux == f_dino, "folds depend on X — paired statistic invalid"
    assert f_flux != f_perm, "fold-identity guard cannot fail (it did not notice permuted labels)"
    print("  fold-identity guard: PASS on real matrices, and FIRED on permuted labels")

    # ---- paired statistics ----------------------------------------------
    print("\nPAIRED per-image deltas (image-level bootstrap, 10k, seed 0):")
    stats = {
        "primary_B1_minus_A1": paired("PRIMARY  B1 DINOv2clsmp - A1 FLUXinv ", A1, B1),
        "B3_minus_A1": paired("         B3 DINOv2nested  - A1 FLUXinv ", A1, B3),
        "B1_minus_A2": paired("         B1 DINOv2clsmp  - A2 FLUXens8 ", A2, B1),
        "A1_minus_A2": paired("         A1 FLUXinv      - A2 FLUXens8 ", A2, A1),
    }
    out = "results/dinov2_resisc45_paired%s.npz" % suffix
    np.savez(out, A1=A1, A2=A2, B1=B1, B2=B2, B3=B3, B1p=B1p, labels=y,
             paths=np.array(paths), primary=np.array(str(stats["primary_B1_minus_A1"])),
             protocol=np.array("6ac: sec-13 instrument, 3 seeds x 5-fold, C=0.1, paired on identical 5000 images"))
    print(f"cached to {out}")

    # ---- secondary: full-split reference --------------------------------
    if args.full_split:
        print("\nSECONDARY (reference, NOT paired): full-split RESISC45, one test eval")
        tr_ds = RESISC45Dataset(args.path, split="train", img_size=256)
        te_ds = RESISC45Dataset(args.path, split="test", img_size=256)
        trp = [s[0] for s in tr_ds.samples]
        tep = [s[0] for s in te_ds.samples]
        ytr = np.array([s[1] for s in tr_ds.samples])
        yte = np.array([s[1] for s in te_ds.samples])
        assert not (set(trp) & set(tep)), "train/test overlap"
        print(f"  train {len(trp)} / test {len(tep)}")
        c1, m1 = cached_extract("results/dinov2_resisc45_feats_train.npz", trp, tag="train ")
        c2, m2 = cached_extract("results/dinov2_resisc45_feats_test.npz", tep, tag="test ")
        Xtr = np.concatenate([c1, m1], axis=1)
        Xte = np.concatenate([c2, m2], axis=1)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always", ConvergenceWarning)
            sc = StandardScaler().fit(Xtr)
            mdl = LogisticRegression(C=HARNESS_C, max_iter=2000).fit(sc.transform(Xtr), ytr)
            pred = mdl.predict(sc.transform(Xte))
            nw = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
        corr = (pred == yte).astype(np.int8)
        acc = float(corr.mean())
        se = (acc * (1 - acc) / len(yte)) ** 0.5
        print(f"  DINOv2 clsmp C=0.1 full-split test top-1 = {acc:.4f} +-{1.96 * se:.4f} "
              + ("[CONVERGED]" if nw == 0 else f"[NOT-CONVERGED x{nw}]"))
        np.savez("results/dinov2_resisc45_fullsplit.npz", correct=corr, labels=yte,
                 acc=np.array(acc), protocol=np.array("DINOv2 clsmp C=0.1, repo 80/20 per-class split, one eval"))


if __name__ == "__main__":
    main()
