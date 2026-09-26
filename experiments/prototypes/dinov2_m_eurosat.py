"""DINOv2 ViT-L/14 frozen linear probe on the exact GEO-Bench m-eurosat partition.

Fills the cell missing from SatDiFuser's Table 2 and the wider literature: a GENERIC
WEB-SCALE model (no EO pretraining of any kind) under the identical protocol our FLUX
numbers use. See RESEARCH_NOTES 6ab pre-registration.

Same images as the FLUX runs (data/m_eurosat_rgb 64px PNGs), bicubic to 224, ImageNet
normalization. Candidates: CLS (1024) and CLS||mean-patch (2048); (variant x C) selected
on the OFFICIAL val split; exactly ONE test evaluation. Deterministic end to end.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))
from data.eurosat_dataset import EUROSAT_CLASSES  # noqa: E402

C_GRID = (0.01, 0.1, 1.0)
MAX_ITER = 3000
EXPECT = {"train": 16200, "val": 996, "test": 996}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def list_split(root: str, split: str):
    files, labels = [], []
    for ci, cls in enumerate(EUROSAT_CLASSES):
        for f in sorted(glob.glob(os.path.join(root, split, cls, "*.png"))):
            files.append(f)
            labels.append(ci)
    assert len(files) == EXPECT[split], (split, len(files))
    return files, np.array(labels, dtype=np.int64)


@torch.no_grad()
def extract(model, files, device, batch=64):
    cls_all, mp_all = [], []
    for i in range(0, len(files), batch):
        imgs = []
        for f in files[i : i + batch]:
            im = Image.open(f).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        x = torch.stack(imgs)
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)
        out = model.forward_features(x)
        cls_all.append(out["x_norm_clstoken"].float().cpu().numpy())
        mp_all.append(out["x_norm_patchtokens"].mean(dim=1).float().cpu().numpy())
        if i % (batch * 20) == 0:
            print(f"  {i}/{len(files)}", flush=True)
    return np.concatenate(cls_all), np.concatenate(mp_all)


def fit_eval(Xtr, ytr, Xev, C):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Xtr, ytr)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return clf.predict(Xev), n_warn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/m_eurosat_rgb")
    p.add_argument("--out", default="results/dinov2_m_eurosat.npz")
    p.add_argument("--max-images", type=int, default=None, help="smoke cap per split")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()

    feats = {}
    for split in ("train", "val", "test"):
        files, y = list_split(args.root, split)
        if args.max_images:
            keep = np.linspace(0, len(files) - 1, args.max_images).astype(int)
            files = [files[i] for i in keep]
            y = y[keep]
        print(f"{split}: {len(files)} images")
        cls_f, mp_f = extract(model, files, device)
        feats[split] = {"cls": cls_f, "clsmp": np.concatenate([cls_f, mp_f], axis=1), "y": y}

    ytr, yva, yte = (feats[s]["y"] for s in ("train", "val", "test"))
    best = None
    for variant in ("cls", "clsmp"):
        sc = StandardScaler().fit(feats["train"][variant])
        A = sc.transform(feats["train"][variant])
        V = sc.transform(feats["val"][variant])
        for C in C_GRID:
            pred, n_warn = fit_eval(A, ytr, V, C)
            acc = float((pred == yva).mean())
            print(f"val {variant:>6} C={C:<5} acc={acc:.4f}" + ("" if n_warn == 0 else f" [NOT CONVERGED x{n_warn}]"))
            if best is None or acc > best[0]:
                best = (acc, variant, C)
    val_acc, variant, C = best
    print(f"SELECTED on val: {variant}, C={C} (val {val_acc:.4f})")

    sc = StandardScaler().fit(feats["train"][variant])
    pred, n_warn = fit_eval(sc.transform(feats["train"][variant]), ytr, sc.transform(feats["test"][variant]), C)
    correct = (pred == yte).astype(np.int8)
    top1 = float(correct.mean())
    f1 = float(f1_score(yte, pred, average="macro", labels=np.arange(len(EUROSAT_CLASSES))))
    se = (top1 * (1 - top1) / len(yte)) ** 0.5
    print(f"TEST ({variant}, C={C}): top1={top1:.4f} ±{1.96 * se:.4f}  macro_f1={f1:.4f}  "
          + ("[CONVERGED]" if n_warn == 0 else f"[NOT CONVERGED x{n_warn}]"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, correct_test=correct, labels_test=yte,
             selected=np.array(f"{variant},C={C},val={val_acc:.6f},test={top1:.6f},f1={f1:.6f}"),
             protocol=np.array("DINOv2 ViT-L/14 frozen; GEO-Bench default partition; val-selected (variant x C); one test eval"))
    print(f"cached to {args.out}")


if __name__ == "__main__":
    main()
