from __future__ import annotations

import json
import os

import numpy as np
import wandb
from torch.utils.data import DataLoader, Subset

from config_types import ExtractionMode
from eval.wb import log_features
from registry import register_task
from utils import env_int, env_value, seed_worker

from .utils import extract_features

POOLING = "spatial_mean"  # global average pool over the feature map, pre-normalization


# Datasets whose __getitem__ implements the DEGRADE_TO resolution-degradation hook. The env
# var silently no-ops elsewhere, so stamping _DEG provenance for any other dataset would make
# the provenance LIE (native-resolution features labelled as degraded).
_DEGRADE_AWARE_DATASETS = {"resisc45", "eurosat"}


ALL_CONTROL_FLAGS = ("FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO")


def env_provenance(cfg, honors: tuple[str, ...] = ALL_CONTROL_FLAGS):
    """Read the three extraction-control env vars, validate them against the config, and
    return (cache_tag_suffix, meta_fields, parts). Shared by ExtractionTask,
    TokenExtractionTask and the standalone extractors in experiments/, so a control cache can
    NEVER carry an unsuffixed name (the poisoning both task docstrings warn about). Uniform
    semantics: unset, "" and "0" all mean OFF for every flag.

    `honors` declares which flags the CALLER'S code path actually implements. A flag that is
    set but not honored raises here, because the alternative is worse than a no-op: the cache
    gets the control's suffix and meta while holding vanilla features, so the control-vs-
    vanilla comparison silently compares vanilla against itself. Three standalone extractors
    hand-copied this stamp without the contract and had exactly that defect (2026-09-06
    pre-merge review, findings 1-3): the two chain scripts stamped _FIXEDCOND on a path where
    invert_chain never reads it, and raw_xt_baseline -- which loads only the VAE -- stamped
    _RANDINIT over trained-VAE features. Pass the narrowest honors tuple that is true of your
    path; do not re-implement this function.
    """
    randinit = env_value("FLUX_RANDOM_INIT")
    fixedcond = env_value("FIXED_COND_T")
    degrade = env_value("DEGRADE_TO")
    # Fail on typos (FIXED_COND_T=100.0, DEGRADE_TO=abc) here, with the var named, rather
    # than as a bare ValueError during run-name generation (PR #5 review).
    fixedcond_i = env_int("FIXED_COND_T", 1, 1000)
    degrade_i = env_int("DEGRADE_TO", 1)

    unknown = [f for f in honors if f not in ALL_CONTROL_FLAGS]
    if unknown:
        raise ValueError(f"unknown control flag(s) in honors={honors}: {unknown}")
    for flag, value in (("FLUX_RANDOM_INIT", randinit), ("FIXED_COND_T", fixedcond), ("DEGRADE_TO", degrade)):
        if value and flag not in honors:
            raise ValueError(
                f"{flag} is set but this path does not honor it consistently "
                f"(honors={list(honors)}). The features would be VANILLA while the cache name "
                f"and meta claimed the control -- the comparison would pit vanilla against "
                f"itself. Unset {flag} for this script."
            )

    if fixedcond and cfg.extraction_mode == ExtractionMode.INVERSION:
        # feat_flux reads FIXED_COND_T only in the one-shot forward; invert_chain never sees
        # it. Stamping the tag anyway would label an ORDINARY inversion cache as a
        # fixed-conditioning control and the analysis would compare vanilla against itself.
        raise ValueError(
            "FIXED_COND_T is only implemented for the one-shot path; unset it for "
            "extraction_mode=inversion (the chain would silently ignore it and the cache "
            "would be mislabelled as a control)."
        )
    if degrade and cfg.dataset.name not in _DEGRADE_AWARE_DATASETS:
        raise ValueError(
            f"DEGRADE_TO is implemented only for {sorted(_DEGRADE_AWARE_DATASETS)}; dataset "
            f"'{cfg.dataset.name}' would silently ignore it while the cache claims degraded "
            "provenance."
        )

    # SINGLE SOURCE OF TRUTH for control provenance parts. Both consumers derive their
    # suffix from this list -- the cache tag joins with '_' UPPER, the run name (run.py)
    # joins with '+' lower -- so a new control flag added here propagates to both formats
    # and cannot be added to one and forgotten in the other (verified hazard, 2026-09-03
    # review: the two formats were previously maintained by hand in different modules).
    parts = []
    if randinit:
        parts.append("randinit")
    if fixedcond:
        parts.append(f"fixedcond{fixedcond_i}")
    if degrade:
        parts.append(f"deg{degrade_i}")
    suffix = "".join("_" + p.upper() for p in parts)
    meta = {
        "weights": "random_init" if randinit else "flux-dev",
        "fixed_cond_t": fixedcond_i,
        "degrade_to": degrade_i,
    }
    return suffix, meta, parts


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
                f"class {cls} has only {len(cls_indices)} images in this split, need {n_cls} for a "
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

    Extracts the cfg.extract_split split (class-stratified subset when subset_size is
    set, else the full split) and caches features pooled
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
        # RANDOM-WEIGHT PROVENANCE. FLUX_RANDOM_INIT builds the architecture WITHOUT
        # loading the checkpoint (see flux/util.load_flow_model), which changes the
        # features completely while changing no field of RunConfig -- so the config hash,
        # the run directory and this filename would all be identical to a real-weights
        # run. Every downstream sweep locates caches by GLOB, so an untrained cache
        # sitting at the expected path would be probed as if it were FLUX. Mark it in the
        # filename, where a `*_oneshot_g1.0.npz` glob cannot reach it.
        prov_suffix, prov_meta, _ = env_provenance(cfg)
        cache_tag = f"{cfg.extraction_mode.value.lower()}_g{cfg.guidance_scale}" + prov_suffix
        if inversion:
            cache_tag += f"_n{cfg.num_inversion_steps}"

        # cfg.extract_split is "train" for every historical cache; "test" extracts the
        # official test split for full-split probe evaluation. The split is stamped into
        # the cache FILENAME (multistep_{split}_feats_*) and meta.json, so a test cache
        # can never be globbed as a train cache.
        extract_ds_full = dataset.get_data(cfg)[cfg.extract_split].dataset
        if not hasattr(extract_ds_full, "samples"):
            raise ValueError(
                f"dataset {type(extract_ds_full).__name__} has no .samples; cannot stratify by label"
            )
        all_labels = np.array([class_idx for _, class_idx, _ in extract_ds_full.samples])

        if cfg.subset_size is not None:
            indices = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)
        else:
            indices = np.arange(len(extract_ds_full))
        if cfg.num_shards > 1:
            # Contiguous range shard for SLURM array jobs. subset_indices in the cache
            # records exactly which images this shard holds; the merge step verifies
            # the shards are disjoint and complete before any probe sees them.
            indices = np.array_split(indices, cfg.num_shards)[cfg.shard_index]
            print(f"shard {cfg.shard_index}/{cfg.num_shards}: {len(indices)} images")
        extract_ds = Subset(extract_ds_full, indices.tolist())

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
            "extraction/split": cfg.extract_split,
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

        feats, labels, mods = extract_features(cfg, model, loader, f"{cfg.extract_split}_subset")

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
        out_path = os.path.join(cfg.save_dir, f"multistep_{cfg.extract_split}_feats_{cache_tag}.npz")
        np.savez(
            out_path,
            feats=feats,  # N, K, C — pooled, pre-normalization
            labels=labels,  # N
            mods=mods,  # K, 3, C — adaLN [shift, scale, gate] per timestep, for offline DiTF norm
            timesteps=np.array(cfg.t),
            subset_indices=indices,  # into cfg.extract_split's split
            paths=np.array([extract_ds_full.samples[i][0] for i in indices]),
            block_idx=np.array(cfg.k),
            subset_seed=np.array(cfg.subset_seed),
            extraction_mode=np.array(cfg.extraction_mode.value),
            split=np.array(cfg.extract_split),  # in-file, so split identity survives a rename/copy
            guidance_scale=np.array(cfg.guidance_scale),
            pooling=np.array(POOLING),
            weights=np.array(prov_meta["weights"]),
            degrade_to=np.array(prov_meta["degrade_to"] if prov_meta["degrade_to"] else -1),
            fixed_cond_t=np.array(prov_meta["fixed_cond_t"] if prov_meta["fixed_cond_t"] else -1),
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
            "seed": cfg.seed,  # global RNG seed (VAE posterior stream) — audited by probes
            "ensemble_size": None if inversion else cfg.model.ensemble_size,
            "subset_size": len(indices),
            "subset_seed": cfg.subset_seed,
            "split": cfg.extract_split,
            "num_shards": cfg.num_shards,
            "shard_index": cfg.shard_index,
            "img_size": cfg.img_size,
            "dataset": cfg.dataset.name,
            "pooling": POOLING,
            "feats_shape": list(feats.shape),
            # "random_init" = untrained control, NOT FLUX weights. Recorded here as well as
            # in the filename so provenance survives a rename or a copy.
            **prov_meta,
        }
        meta_path = os.path.join(cfg.save_dir, f"multistep_{cfg.extract_split}_feats_{cache_tag}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        wandb.log({"extraction/num_images": feats.shape[0], "extraction/feat_dim": feats.shape[-1]})
        # Reference artifact (path + checksum, no upload): eval.probe declares the same
        # reference as its input, giving the extract -> probe lineage edge.
        log_features(out_path)
        return {"feats_path": out_path, "meta_path": meta_path, "shape": list(feats.shape)}
