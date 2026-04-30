from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from registry import register_task
from utils import seed_worker


class _LinearProbe(nn.Module):
    """Single linear layer trained on top of frozen DiT features."""

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def _extract_features(cfg, model, dataloader, split_name: str):
    """Extract and return (features, labels) for all images in dataloader."""
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    device = torch.device(cfg.device)

    print("saving %s images' features..." % split_name)
    for batch in tqdm(dataloader):
        img = batch["img"].to(device)  # B, 3, H, W
        label = batch["label"]  # B

        for single_img, single_label in zip(img, label, strict=True):
            # TODO: if GPU can tolerate higher batch sizes, we can extract features for the whole batch at once instead of looping through images one by one.
            feat = model.extract(
                single_img,
                timestep=cfg.t,
                block_idx=cfg.k,
                ensemble_size=cfg.model.ensemble_size,
            )  # 1, C, H, W

            feat_vec = feat.mean(dim=[2, 3])  # 1, C  — global average pool
            feat_vec = F.normalize(feat_vec, dim=1)

            all_feats.append(feat_vec.cpu())
            all_labels.append(single_label.cpu())

    feats = torch.cat(all_feats, dim=0).numpy()  # N, C
    labels = torch.cat(all_labels, dim=0).numpy()  # N
    return feats, labels


def _subsample_by_fraction(feats, labels, fraction: float, seed: int, num_classes: int):
    # class-balanced subsample: take `fraction` percent of each class independently
    rng = np.random.default_rng(seed)
    keep_idx: list[int] = []
    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        n_keep = max(1, int(len(cls_idx) * fraction / 100.0))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx = np.array(keep_idx)
    return feats[keep_idx], labels[keep_idx]


def _train_linear_probe(train_feats, train_labels, num_epochs, lr, batch_size, device, num_classes: int):
    X = torch.from_numpy(train_feats).float().to(device)
    y = torch.from_numpy(train_labels).long().to(device)

    ds = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    probe = _LinearProbe(X.shape[1], num_classes).to(device)
    # TODO: make optimizer and loss configurable (e.g. SGD, label smoothing)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    t0 = time.perf_counter()
    probe.train()
    for epoch in range(num_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            total_steps += 1

    return probe, total_steps, time.perf_counter() - t0


@torch.no_grad()
def _evaluate_probe(probe, test_feats, test_labels, device):
    probe.eval()
    X = torch.from_numpy(test_feats).float().to(device)
    preds = probe(X).cpu().numpy().argmax(axis=1)
    top1 = (preds == test_labels).mean() * 100.0
    macro_f1 = f1_score(test_labels, preds, average="macro") * 100.0
    weighted_f1 = f1_score(test_labels, preds, average="weighted") * 100.0
    per_class_f1 = f1_score(test_labels, preds, average=None) * 100.0
    return top1, macro_f1, weighted_f1, per_class_f1


@register_task("classification")
class ClassificationTask:
    def run(self, cfg, model, dataset) -> dict:
        device = torch.device(cfg.device)

        loaders = dataset.get_data(cfg)
        train_loader = loaders["train"]
        test_loader = loaders["test"]

        os.makedirs(cfg.save_dir, exist_ok=True)
        train_feat_path = os.path.join(cfg.save_dir, "train_feats.npz")
        test_feat_path = os.path.join(cfg.save_dir, "test_feats.npz")

        # load cached features if available, otherwise extract and save
        if os.path.exists(train_feat_path) and not cfg.overwrite_features:
            print("loading cached train features from %s" % train_feat_path)
            d = np.load(train_feat_path)
            train_feats, train_labels = d["feats"], d["labels"]
        else:
            train_feats, train_labels = _extract_features(cfg, model, train_loader, "train")
            np.savez(train_feat_path, feats=train_feats, labels=train_labels)

        if os.path.exists(test_feat_path) and not cfg.overwrite_features:
            print("loading cached test features from %s" % test_feat_path)
            d = np.load(test_feat_path)
            test_feats, test_labels = d["feats"], d["labels"]
        else:
            test_feats, test_labels = _extract_features(cfg, model, test_loader, "test")
            np.savez(test_feat_path, feats=test_feats, labels=test_labels)

        result: dict = {}
        class_names = getattr(dataset, "class_names", dataset.category_list)
        num_classes = len(class_names)

        print("Label fractions: %s" % cfg.label_fractions)
        for frac in cfg.label_fractions:
            sub_feats, sub_labels = _subsample_by_fraction(
                train_feats,
                train_labels,
                fraction=frac,
                seed=cfg.seed,
                num_classes=num_classes,
            )
            probe, steps, elapsed = _train_linear_probe(
                sub_feats,
                sub_labels,
                num_epochs=cfg.clf_epochs,
                lr=cfg.clf_lr,
                batch_size=cfg.clf_batch_size,
                device=torch.device(cfg.device),
                num_classes=num_classes,
            )
            top1, macro_f1, weighted_f1, per_class_f1 = _evaluate_probe(
                probe, test_feats, test_labels, torch.device(cfg.device)
            )

            # per-class accuracy and F1 breakdown
            probe.eval()
            with torch.no_grad():
                X = torch.from_numpy(test_feats).float().to(device)
                preds = probe(X).cpu().numpy().argmax(axis=1)
            per_class_acc: dict[str, float] = {}
            per_class_f1_dict: dict[str, float] = {}
            for cls_idx, cls_name in enumerate(class_names):
                mask = test_labels == cls_idx
                cls_acc = (preds[mask] == test_labels[mask]).mean() * 100.0
                per_class_acc[cls_name] = round(float(cls_acc), 2)
                per_class_f1_dict[cls_name] = round(float(per_class_f1[cls_idx]), 2)

            result[frac] = {
                "label_fraction_pct": frac,
                "n_train_samples": int(len(sub_labels)),
                "top1_accuracy": round(float(top1), 2),
                "macro_f1": round(float(macro_f1), 2),
                "weighted_f1": round(float(weighted_f1), 2),
                "training_steps": steps,
                "wall_clock_seconds": round(elapsed, 2),
                "per_class_accuracy": per_class_acc,
                "per_class_f1": per_class_f1_dict,
            }

            print(
                "%s%% labels  top1: %.2f  macro-f1: %.2f  weighted-f1: %.2f  n=%d  steps=%d  time=%.1fs"
                % (frac, top1, macro_f1, weighted_f1, len(sub_labels), steps, elapsed)
            )

            torch.cuda.empty_cache()

        out_path = os.path.join(
            cfg.save_dir,
            "t%s_b%s_e%s_seed%s.json" % (cfg.t, cfg.k, cfg.model.ensemble_size, cfg.seed),
        )
        with open(out_path, "w+") as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

        return result
