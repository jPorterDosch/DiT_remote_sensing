from __future__ import annotations

import json
import os

import numpy as np
import torch

from registry import register_task

from .utils import (
    evaluate_probe,
    extract_features,
    train_probe,
)


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
            train_feats, train_labels = extract_features(cfg, model, train_loader, "train")
            np.savez(train_feat_path, feats=train_feats, labels=train_labels)

        if os.path.exists(test_feat_path) and not cfg.overwrite_features:
            print("loading cached test features from %s" % test_feat_path)
            d = np.load(test_feat_path)
            test_feats, test_labels = d["feats"], d["labels"]
        else:
            test_feats, test_labels = extract_features(cfg, model, test_loader, "test")
            np.savez(test_feat_path, feats=test_feats, labels=test_labels)

        result: dict = {}
        class_names = getattr(dataset, "class_names", dataset.category_list)
        num_classes = len(class_names)

        print("Label fraction: %s%%" % (cfg.label_fraction * 100))
        frac = cfg.label_fraction
        probe, steps, elapsed = train_probe(
            cfg.probe_type,
            train_feats,
            train_labels,
            num_epochs=cfg.clf_epochs,
            lr=cfg.clf_lr,
            batch_size=cfg.clf_batch_size,
            device=torch.device(cfg.device),
            num_classes=num_classes,
            grid_size=cfg.grid_size,
            polynomial_order=cfg.polynomial_order,
        )
        top1, macro_f1, weighted_f1, per_class_f1 = evaluate_probe(
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
            "top1_accuracy": round(float(top1), 2),
            "macro_f1": round(float(macro_f1), 2),
            "weighted_f1": round(float(weighted_f1), 2),
            "training_steps": steps,
            "wall_clock_seconds": round(elapsed, 2),
            "per_class_accuracy": per_class_acc,
            "per_class_f1": per_class_f1_dict,
        }

        print(
            "%s%% labels  top1: %.2f  macro-f1: %.2f  weighted-f1: %.2f  steps=%d  time=%.1fs"
            % (frac, top1, macro_f1, weighted_f1, steps, elapsed)
        )

        torch.cuda.empty_cache()

        out_path = os.path.join(
            cfg.save_dir,
            "t%s_b%s_e%s_seed%s.json" % (cfg.t, cfg.k, cfg.model.ensemble_size, cfg.seed),
        )
        with open(out_path, "w+") as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

        return result
