from __future__ import annotations

import json
import os

import numpy as np
import wandb
from torch.utils.data import DataLoader, Subset

from config_types import ExtractionMode
from registry import register_task
from utils import seed_worker

from .utils import extract_features

POOLING = "spatial_mean"  # global average pool over the feature map, pre-normalization


def _stratified_indices(labels: np.ndarray, subset_size: int, seed: int) -> np.ndarray:
    """Class-stratified sample of `subset_size` indices, split as evenly as possible
    across classes (first `subset_size % num_classes` classes get one extra).

    Deterministic given (labels order, subset_size, seed): np.random.default_rng(seed)
    plus per-class sorted-filename dataset order. All extraction runs that share these
    inputs select the IDENTICAL image subset — paired caches rely on this.
    """
    classes = np.unique(labels)
    base, rem = divmod(subset_size, len(classes))
    rng = np.random.default_rng(seed)

    selected: list[np.ndarray] = []
    for i, cls in enumerate(classes):
        n_cls = base + (1 if i < rem else 0)
        cls_indices = np.flatnonzero(labels == cls)
        if len(cls_indices) < n_cls:
            raise ValueError(
                f"class {cls} has only {len(cls_indices)} train images, need {n_cls} for a "
                f"stratified subset of {subset_size}"
            )
        selected.append(rng.choice(cls_indices, size=n_cls, replace=False))

    # Sort so extraction order (and hence the per-image eps stream) follows dataset order.
    return np.sort(np.concatenate(selected))


@register_task("extract")
class ExtractionTask:
    """Multi-timestep feature extraction only — no probe training.

    Modes (cfg.extraction_mode):
      ONESHOT:   block cfg.k hidden states at each timestep in cfg.t via one-shot
                 noising per t, identical eps per image across all K forward passes.
      INVERSION: one RF-Solver reverse-ODE chain per image (each state depends on the
                 previous one); features cached at the requested timesteps, which must
                 lie on the cfg.num_inversion_steps integration grid.

    Extracts a class-stratified subset of the train split and caches features pooled
    but pre-normalization as (N, K, C). Cache filenames are mode- and guidance-tagged
    (and step-count-tagged for inversion) so caches can never collide or be confused,
    and a meta.json with the exact extraction settings is written beside each cache.
    """

    def run(self, cfg, model, dataset) -> dict:
        if not isinstance(cfg.t, list):
            raise ValueError(
                "task='extract' requires a list of timesteps, e.g. --t 100 180 260 340 420 500 580; "
                f"got t={cfg.t}"
            )
        if cfg.label_fraction != 1.0:
            raise ValueError("task='extract' selects its own subset; run with label_fraction=1.0")

        inversion = cfg.extraction_mode == ExtractionMode.INVERSION
        # e.g. inversion_g1.0_n50 / oneshot_g1.0 / oneshot_g3.5 — collision-proof cache tag.
        cache_tag = f"{cfg.extraction_mode.value.lower()}_g{cfg.guidance_scale}"
        if inversion:
            cache_tag += f"_n{cfg.num_inversion_steps}"

        train_ds = dataset.get_data(cfg)["train"].dataset
        if not hasattr(train_ds, "samples"):
            raise ValueError(f"dataset {type(train_ds).__name__} has no .samples; cannot stratify by label")
        all_labels = np.array([class_idx for _, class_idx, _ in train_ds.samples])

        if cfg.subset_size is not None:
            indices = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)
            extract_ds = Subset(train_ds, indices.tolist())
        else:
            indices = np.arange(len(train_ds))
            extract_ds = train_ds

        loader = DataLoader(
            extract_ds,
            batch_size=cfg.batch_size,
            shuffle=False,  # deterministic order, the eps stream depends on it
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        num_timesteps = len(cfg.t)
        # eps stream / ensemble apply to the oneshot noising only; the inversion chain
        # draws no eps of its own (ae.encode still samples the VAE posterior).
        eps_seed = None if inversion else (cfg.eps_seed if cfg.eps_seed is not None else cfg.seed)
        extraction_config = {
            "extraction/mode": cfg.extraction_mode.value,
            "extraction/guidance_scale": cfg.guidance_scale,
            "extraction/num_timesteps": num_timesteps,
            "extraction/timesteps": list(cfg.t),
            "extraction/block_idx": cfg.k,
            "extraction/num_inversion_steps": cfg.num_inversion_steps if inversion else None,
            "extraction/eps_seed": eps_seed,
            "extraction/subset_size": len(indices),
            "extraction/subset_seed": cfg.subset_seed,
            "extraction/ensemble_size": None if inversion else cfg.model.ensemble_size,
            "extraction/pooling": POOLING,
        }
        wandb.config.update(extraction_config, allow_val_change=True)
        for key, value in extraction_config.items():
            print(f"{key}: {value}")

        class_names = getattr(dataset, "class_names", dataset.category_list)
        counts = np.bincount(all_labels[indices], minlength=len(class_names))
        print("per-class counts:", dict(zip(class_names, counts.tolist(), strict=True)))

        feats, labels, mods = extract_features(cfg, model, loader, "train_subset")

        expected_shape = (len(indices), num_timesteps, feats.shape[-1])
        if feats.ndim != 3 or feats.shape != expected_shape:
            raise RuntimeError(f"expected features of shape {expected_shape}, got {feats.shape}")
        if not np.array_equal(labels, all_labels[indices]):
            raise RuntimeError(
                "extracted labels do not match the selected subset — dataloader order changed?"
            )

        mode_extra = (
            {"num_inversion_steps": np.array(cfg.num_inversion_steps)}
            if inversion
            else {"eps_seed": np.array(eps_seed), "ensemble_size": np.array(cfg.model.ensemble_size)}
        )
        out_path = os.path.join(cfg.save_dir, f"multistep_train_feats_{cache_tag}.npz")
        np.savez(
            out_path,
            feats=feats,  # N, K, C — pooled, pre-normalization
            labels=labels,  # N
            mods=mods,  # K, 3, C — adaLN [shift, scale, gate] per timestep, for offline DiTF norm
            timesteps=np.array(cfg.t),
            subset_indices=indices,  # into the train split
            paths=np.array([train_ds.samples[i][0] for i in indices]),
            block_idx=np.array(cfg.k),
            subset_seed=np.array(cfg.subset_seed),
            extraction_mode=np.array(cfg.extraction_mode.value),
            guidance_scale=np.array(cfg.guidance_scale),
            pooling=np.array(POOLING),
            **mode_extra,
        )
        consistency = "chain state" if inversion else "eps-consistency assertion passed"
        print(f"cached {feats.shape} features ({consistency}) to {out_path}")

        # meta.json beside the cache: trustworthy provenance, only settings actually applied.
        meta = {
            "extraction_mode": cfg.extraction_mode.value,
            "guidance_scale": cfg.guidance_scale,
            "t": list(cfg.t),
            "k": cfg.k,
            "num_inversion_steps": cfg.num_inversion_steps if inversion else None,
            "eps_seed": eps_seed,
            "ensemble_size": None if inversion else cfg.model.ensemble_size,
            "subset_size": len(indices),
            "subset_seed": cfg.subset_seed,
            "img_size": cfg.img_size,
            "dataset": cfg.dataset.name,
            "pooling": POOLING,
            "feats_shape": list(feats.shape),
        }
        meta_path = os.path.join(cfg.save_dir, f"multistep_train_feats_{cache_tag}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        wandb.log({"extraction/num_images": feats.shape[0], "extraction/feat_dim": feats.shape[-1]})
        return {"feats_path": out_path, "meta_path": meta_path, "shape": list(feats.shape)}
