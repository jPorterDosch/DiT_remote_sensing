from __future__ import annotations

import json
import os

import numpy as np
import wandb
from torch.utils.data import DataLoader, Subset

from config_types import ExtractionMode
from registry import register_task
from utils import seed_worker

from .extraction import POOLING, _stratified_indices, env_provenance
from .utils import extract_features


@register_task("extract_tokens")
class TokenExtractionTask:
    """Pooled extraction PLUS a pre-pool per-token rider at a subset of timesteps.

    Identical to task="extract" in every respect that touches the pooled cache — same
    subset selection, same dataloader order, same eps stream, same cache tag — and it
    writes that pooled (N, K, C) cache unchanged. The addition is a second artifact:
    PRE-POOL token features (N, S, L, C) at cfg.per_token_t, i.e. the tensor the pooled
    cache is a spatial mean OVER.

    This exists because every aggregation in the pipeline pools tokens before the time
    axis is introduced, so any joint space x time structure ("which regions resolve
    when") is destroyed before a probe can see it. The rider keeps that structure.

    Token order is row-major over the (h, w) feature grid and is by construction the
    same index set at every timestep. Whether index i actually tracks the same patch
    ACROSS timesteps is a property of the model and the chain, not of this layout — it
    is exactly what the alignment check tests, and it is deliberately not assumed here.
    """

    def run(self, cfg, model, dataset) -> dict:
        if not isinstance(cfg.t, list):
            raise ValueError(
                "task='extract_tokens' requires a list of timesteps, e.g. --t 260 420 580; "
                f"got t={cfg.t}"
            )
        if cfg.subset_size is None:
            raise ValueError(
                "task='extract_tokens' requires subset_size: the per-token rider is L~=196-256"
                " times the pooled cache, and an unbounded run (~25k images) needs ~240 GB host"
                " RAM that OOMs only AFTER the full GPU extraction."
            )
        if not cfg.per_token_t:
            raise ValueError(
                "task='extract_tokens' with empty per_token_t would run the full extraction and"
                " then fail the rider-count assertion -- hours of GPU work for a condition"
                " detectable now. Use task='extract' if you only want the pooled cache."
            )
        if cfg.label_fraction != 1.0:
            raise ValueError("task='extract_tokens' selects its own subset; run with label_fraction=1.0")

        missing = [t for t in cfg.per_token_t if t not in cfg.t]
        if missing:
            raise ValueError(
                f"per_token_t values {missing} are not in t={list(cfg.t)} — the rider can only "
                "cache timesteps the extraction actually visits."
            )
        # Canonicalize to cfg.t order. extract_features fills the rider's S axis by walking
        # cfg.t, so if per_token_t were given in a different order (e.g. --per_token_t 580
        # 260) the saved `timesteps` array would mislabel that axis. Sorting here makes the
        # metadata true by construction rather than relying on the caller's ordering.
        per_token_t = [t for t in cfg.t if t in set(cfg.per_token_t)]

        inversion = cfg.extraction_mode == ExtractionMode.INVERSION
        # Provenance suffix shared with ExtractionTask: without it a FLUX_RANDOM_INIT /
        # DEGRADE_TO / FIXED_COND_T run writes a cache byte-identically NAMED to a
        # real-weights one, and every downstream glob probes poisoned features as real.
        prov_suffix, prov_meta = env_provenance(cfg)
        cache_tag = f"{cfg.extraction_mode.value.lower()}_g{cfg.guidance_scale}" + prov_suffix
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
        eps_seed = None if inversion else (cfg.eps_seed if cfg.eps_seed is not None else cfg.seed)
        extraction_config = {
            "extraction/mode": cfg.extraction_mode.value,
            "extraction/guidance_scale": cfg.guidance_scale,
            "extraction/num_timesteps": num_timesteps,
            "extraction/timesteps": list(cfg.t),
            "extraction/per_token_timesteps": per_token_t,
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

        want_vel = bool(getattr(cfg, "want_velocity", False))
        if want_vel and not inversion:
            raise ValueError(
                "want_velocity requires extraction_mode=inversion — the one-shot path has no "
                "chain, so its velocity would be evaluated at an independently-noised point "
                "rather than along a trajectory."
            )
        token_sink: dict = {"timesteps": per_token_t, "feats": []}
        vel_sink: list | None = [] if want_vel else None
        feats, labels, mods = extract_features(
            cfg, model, loader, "train_subset", token_sink=token_sink, vel_sink=vel_sink
        )

        expected_shape = (len(indices), num_timesteps, feats.shape[-1])
        if feats.ndim != 3 or feats.shape != expected_shape:
            raise RuntimeError(f"expected pooled features of shape {expected_shape}, got {feats.shape}")
        if not np.array_equal(labels, all_labels[indices]):
            raise RuntimeError(
                "extracted labels do not match the selected subset — dataloader order changed?"
            )

        mode_extra = (
            {"num_inversion_steps": np.array(cfg.num_inversion_steps)}
            if inversion
            else {"eps_seed": np.array(eps_seed), "ensemble_size": np.array(cfg.model.ensemble_size)}
        )
        paths = np.array([train_ds.samples[i][0] for i in indices])

        out_path = os.path.join(cfg.save_dir, f"multistep_train_feats_{cache_tag}.npz")
        np.savez(
            out_path,
            feats=feats,
            labels=labels,
            mods=mods,
            timesteps=np.array(cfg.t),
            subset_indices=indices,
            paths=paths,
            block_idx=np.array(cfg.k),
            subset_seed=np.array(cfg.subset_seed),
            extraction_mode=np.array(cfg.extraction_mode.value),
            guidance_scale=np.array(cfg.guidance_scale),
            pooling=np.array(POOLING),
            weights=np.array(prov_meta["weights"]),
            degrade_to=np.array(prov_meta["degrade_to"] if prov_meta["degrade_to"] else -1),
            fixed_cond_t=np.array(prov_meta["fixed_cond_t"] if prov_meta["fixed_cond_t"] else -1),
            **mode_extra,
        )
        print(f"cached pooled {feats.shape} to {out_path}")

        # Rider assembly runs AFTER the pooled cache is on disk: a rider-only failure
        # must not discard hours of successfully computed pooled extraction.
        # --- assemble + assert the per-token rider
        if len(token_sink["feats"]) != len(indices):
            raise RuntimeError(
                f"token rider collected {len(token_sink['feats'])} images, expected {len(indices)}"
            )
        import torch

        tokens = torch.cat(token_sink["feats"], dim=0).numpy()  # N, S, L, C
        h, w = token_sink["hw"]
        expected_tok = (len(indices), len(per_token_t), h * w, feats.shape[-1])
        if tokens.shape != expected_tok:
            raise RuntimeError(f"expected token features of shape {expected_tok}, got {tokens.shape}")
        # The pooled cache must be the spatial mean of the rider at the shared timesteps —
        # if this fails the two artifacts describe different computations and no downstream
        # comparison between them is meaningful.
        shared = [cfg.t.index(t) for t in per_token_t]
        pooled_from_tokens = tokens.mean(axis=2)  # N, S, C
        ref = feats[:, shared, :]
        # Features are extracted in bf16 (eps ~ 2^-7), and the two means accumulate in a
        # different order and precision, so bit-level equality is not available at any
        # sane tolerance. What must hold is STRUCTURAL: the rider's spatial mean and the
        # pooled cache must be the same vector up to dtype noise. Cosine catches a real
        # mismatch (wrong axis, wrong token order, stale buffer -> cosine well below 1)
        # while being insensitive to bf16 rounding.
        a = pooled_from_tokens.reshape(-1, pooled_from_tokens.shape[-1]).astype(np.float64)
        b = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        rel = np.abs(a - b).sum(1) / (np.abs(b).sum(1) + 1e-12)
        if cos.min() < 0.999 or rel.max() > 0.02:
            raise RuntimeError(
                f"token rider does not spatially average to the pooled cache\n"
                f"  tokens {tokens.shape} -> mean(axis=2) {pooled_from_tokens.shape}\n"
                f"  pooled ref {ref.shape}  (grid {h}x{w}, L={h * w})\n"
                f"  min cosine {cos.min():.6f}   max rel-L1 {rel.max():.4f}\n"
                f"  tokens-mean[0,0,:6] = {np.round(pooled_from_tokens[0, 0, :6], 4)}\n"
                f"  pooled    [0,0,:6] = {np.round(ref[0, 0, :6], 4)}"
            )
        print(
            f"rider consistency OK: min cosine {cos.min():.6f}, max rel-L1 {rel.max():.4f} "
            f"(bf16 noise) over {len(cos)} (image, timestep) pairs"
        )


        # per_token_t is deliberately excluded from the config hash (it cannot change the
        # pooled cache), so the rider's timesteps go in the FILENAME instead — otherwise
        # two runs differing only in per_token_t would resolve to the same path.
        tok_tag = f"{cache_tag}_t{'-'.join(str(t) for t in per_token_t)}"
        tok_path = os.path.join(cfg.save_dir, f"multistep_train_tokens_{tok_tag}.npz")
        np.savez(
            tok_path,
            tokens=tokens,  # N, S, L, C — PRE-POOL, pre-normalization
            labels=labels,
            timesteps=np.array(per_token_t),
            grid_hw=np.array([h, w]),
            subset_indices=indices,
            paths=paths,
            block_idx=np.array(cfg.k),
            subset_seed=np.array(cfg.subset_seed),
            extraction_mode=np.array(cfg.extraction_mode.value),
            guidance_scale=np.array(cfg.guidance_scale),
            **mode_extra,
        )
        gb = tokens.nbytes / 1024**3
        print(f"cached tokens {tokens.shape} (grid {h}x{w}, {gb:.2f} GB) to {tok_path}")

        vel_path = None
        if want_vel:
            if vel_sink is None or len(vel_sink) != len(indices):
                raise RuntimeError(
                    f"velocity sink collected {0 if vel_sink is None else len(vel_sink)} images, "
                    f"expected {len(indices)}"
                )
            vels = torch.cat(vel_sink, dim=0).numpy()  # N, K, T, d
            if vels.shape[:2] != (len(indices), num_timesteps):
                raise RuntimeError(
                    f"expected velocities (N={len(indices)}, K={num_timesteps}, T, d), got {vels.shape}"
                )
            if vels.shape[2] != h * w:
                # Velocity is packed latent; its token axis must match the feature grid or
                # the two cannot be indexed against each other per patch.
                raise RuntimeError(
                    f"velocity token axis {vels.shape[2]} != feature grid {h}x{w}={h * w}"
                )
            vel_path = os.path.join(cfg.save_dir, f"multistep_train_vels_{cache_tag}.npz")
            np.savez(
                vel_path,
                vels=vels,  # N, K, T, d — packed-latent model velocity, all cached timesteps
                labels=labels,
                timesteps=np.array(cfg.t),
                grid_hw=np.array([h, w]),
                subset_indices=indices,
                paths=paths,
                block_idx=np.array(cfg.k),
                subset_seed=np.array(cfg.subset_seed),
                extraction_mode=np.array(cfg.extraction_mode.value),
                guidance_scale=np.array(cfg.guidance_scale),
                **mode_extra,
            )
            print(
                f"cached velocities {vels.shape} ({vels.nbytes / 1024**3:.2f} GB) to {vel_path}"
            )

        meta = {
            **prov_meta,
            "extraction_mode": cfg.extraction_mode.value,
            "guidance_scale": cfg.guidance_scale,
            "t": list(cfg.t),
            "per_token_t": per_token_t,
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
            "tokens_shape": list(tokens.shape),
            "grid_hw": [h, w],
            "want_velocity": want_vel,
            "vels_path": vel_path,
            "image_ids": [os.path.basename(str(p)) for p in paths.tolist()],
        }
        meta_path = os.path.join(cfg.save_dir, f"multistep_train_tokens_{tok_tag}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        wandb.log(
            {
                "extraction/num_images": feats.shape[0],
                "extraction/feat_dim": feats.shape[-1],
                "extraction/num_tokens": h * w,
                "extraction/tokens_gb": gb,
            }
        )
        return {
            "feats_path": out_path,
            "tokens_path": tok_path,
            "vels_path": vel_path,
            "meta_path": meta_path,
            "shape": list(feats.shape),
            "tokens_shape": list(tokens.shape),
        }
