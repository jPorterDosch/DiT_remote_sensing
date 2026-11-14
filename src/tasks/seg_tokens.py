"""Per-token FLUX features of one official SEGMENTATION split: `run.py --task extract_seg_tokens`.

The classification caches pool every token; a per-pixel head needs the tokens. This task is the
one-shot multi-block extraction of tasks/extraction.py (same Featurizer4Eval.forward, same eps
stream, same ensemble averaging, same adaLN mods, same provenance stamps) with the spatial mean
REMOVED: for every (timestep, block) it writes one float16 memmap of shape (N, L, C) -- image
order = eval.features.list_official_seg_split order, L = (img/16)^2 tokens row-major over the
feature grid, C = the backbone's hidden width (FLUX 3072, GeoCore 4096), pre-normalization (DiTF is applied per token by eval/seg_probe.py with
the mods saved beside the tokens). Written image by image straight into the .npy on disk, so
host RAM does not scale with N (token_extraction's rider keeps everything in RAM and is capped
for that reason).

Files under cfg.save_dir:
  seg_tokens_{split}_t{t}_k{k}_{tag}.npy      (N, L, C) float16, one per (t, k)
  seg_tokens_{split}_{tag}_meta.npz / .json    mods (K, B, 3, C), timesteps, blocks, paths,
                                               grid_hw, eps_seed, ensemble_size, img_size,
                                               weights (+ lora fields), guidance, double-block stamp
tag = oneshot_g<guidance><provenance suffix>_ens<E>_img<S>, so an adapted (lora) or control cache
can never be globbed as a frozen one (tasks/extraction.py's rule).

cfg: --t <list>, --k <list or int>, --img-size S S, --extract-split {train,val,test},
--model.ensemble-size E, --eps-seed (per split: 42 train / 43 test / 44 val, the classification
convention), --lora-checkpoint for an adapted arm, --max-images N for a strided SMOKE (stamped,
never quotable). Prints SEG_TOKENS_OK <split>.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_types import ExtractionMode
from registry import register_task
from utils import seed_worker

from eval.features import DISCARD

from .extraction import block_stamp_for, model_provenance


def token_path(save_dir: str, split: str, t: int, k: int, tag: str) -> str:
    return os.path.join(save_dir, f"seg_tokens_{split}_t{t}_k{k}_{tag}.npy")


def meta_path(save_dir: str, split: str, tag: str, ext: str = "npz") -> str:
    return os.path.join(save_dir, f"seg_tokens_{split}_{tag}_meta.{ext}")


@register_task("extract_seg_tokens")
class SegTokenExtractionTask:
    def run(self, cfg, model, dataset) -> dict:
        if not isinstance(cfg.t, list) or not cfg.t:
            raise ValueError(
                f"task='extract_seg_tokens' needs a list of timesteps (--t 100 260 420), got t={cfg.t}"
            )
        if cfg.extraction_mode != ExtractionMode.ONESHOT:
            raise ValueError(
                "task='extract_seg_tokens' is one-shot only (the inversion chain caches one block)"
            )
        if cfg.label_fraction != 1.0:
            raise ValueError("task='extract_seg_tokens' extracts the full split; run with label_fraction=1.0")
        blocks = [cfg.k] if isinstance(cfg.k, int) else list(cfg.k)
        if any(a >= b for a, b in zip(blocks, blocks[1:])):
            raise ValueError(f"blocks must be strictly increasing, got {blocks}")
        if cfg.img_size[0] != cfg.img_size[1] or cfg.img_size[0] % 16:
            raise ValueError(f"img_size must be square and a multiple of 16, got {cfg.img_size}")
        ens = int(cfg.model.ensemble_size)
        split = cfg.extract_split
        device = torch.device(cfg.device)

        # One provenance path for every backbone (tasks/extraction.model_provenance): FLUX keeps the
        # control-env + LoRA stamps byte-identical; GeoCore stamps its own weights tag and widths.
        prov_suffix, prov_meta = model_provenance(model, cfg)
        tag = f"oneshot_g{cfg.guidance_scale}{prov_suffix}_ens{ens}_img{cfg.img_size[0]}"
        # Channels the fp16 overflow check below may ignore: FLUX's DiTF massive channels; another
        # backbone's list only if the adapter carries one (GeoCore's is stamped post hoc from the
        # classification train cache, so at extraction time nothing is ignored: any overflow raises).
        is_flux = getattr(model, "weights_tag", "flux-dev") == "flux-dev"
        overflow_ok = list(DISCARD) if is_flux else list(getattr(model, "discard_channels", None) or [])
        max_images = getattr(cfg, "max_images", None)
        if max_images:
            tag += f"_smoke{max_images}"

        ds_full = dataset.get_data(cfg)[split].dataset
        if not hasattr(ds_full, "samples") or not hasattr(ds_full, "masks"):
            raise ValueError(f"dataset {type(ds_full).__name__} is not an OfficialSegSplit")
        n_full = len(ds_full)
        indices = np.arange(n_full)
        if max_images:
            indices = np.linspace(0, n_full - 1, int(max_images)).astype(int)
        if cfg.num_shards > 1:
            raise ValueError(
                "extract_seg_tokens writes one memmap per split; shard by split (one job per split), not by index"
            )
        ds = Subset(ds_full, indices.tolist())
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True, worker_init_fn=seed_worker)  # fmt: skip
        paths = [ds_full.samples[i][0] for i in indices]
        masks = [ds_full.masks[i] for i in indices]
        eps_seed = cfg.eps_seed if cfg.eps_seed is not None else cfg.seed
        generator = torch.Generator(device=device).manual_seed(eps_seed)
        model.eval()
        os.makedirs(cfg.save_dir, exist_ok=True)
        for t in cfg.t:
            for k in blocks:
                if os.path.exists(token_path(cfg.save_dir, split, t, k, tag)):
                    raise FileExistsError(
                        f"{token_path(cfg.save_dir, split, t, k, tag)} exists -- never overwrite a token cache"
                    )
        print(f"[seg_tokens] {cfg.dataset.name} {split}: {len(indices)} images, t={list(cfg.t)}, k={blocks}, ens={ens}, "
              f"img={cfg.img_size[0]}, eps_seed={eps_seed}, weights={prov_meta['weights']} -> {cfg.save_dir} ({tag})", flush=True)  # fmt: skip

        mm: dict[tuple[int, int], np.memmap] = {}
        mods: dict[int, torch.Tensor] = {}
        grid = None
        n_done, t0 = 0, time.time()
        with torch.inference_mode():
            for batch in loader:
                for img in batch["img"]:
                    img = img.to(device)
                    latents = noise = None
                    for t in cfg.t:
                        feat_raw, ada, latents_used, noise_used = model.extract_raw(
                            img,
                            timestep=t,
                            block_idx=blocks if len(blocks) > 1 else blocks[0],
                            ensemble_size=ens,
                            guidance=cfg.guidance_scale,
                            latents=latents,
                            noise=noise,
                            generator=generator,
                        )  # feat_raw (B, C, h, w), ada (B, 3, C); single block: (1, C, h, w), (1, 3, C)
                        if latents is None:
                            latents, noise = latents_used, noise_used
                        else:
                            if not torch.equal(noise_used, noise):
                                raise RuntimeError(
                                    f"eps-consistency violation at t={t}: the eps differs from t={cfg.t[0]}'s"
                                )
                            if not torch.equal(latents_used, latents):
                                raise RuntimeError(f"latent-consistency violation at t={t}")
                        if feat_raw.shape[0] != len(blocks):
                            raise RuntimeError(f"expected {len(blocks)} block features, got {feat_raw.shape}")
                        _, C, h, w = feat_raw.shape
                        if grid is None:
                            grid = (h, w)
                            for tt in cfg.t:
                                for kk in blocks:
                                    mm[(tt, kk)] = np.lib.format.open_memmap(
                                        token_path(cfg.save_dir, split, tt, kk, tag),
                                        mode="w+",
                                        dtype=np.float16,
                                        shape=(len(indices), h * w, C),
                                    )
                        elif grid != (h, w):
                            raise RuntimeError(f"token grid changed mid-extraction: {grid} -> {(h, w)}")
                        toks = feat_raw.reshape(len(blocks), C, h * w).transpose(1, 2)  # B, L, C
                        toks = toks.to(torch.float16)
                        # fp16 tops out at 65504; the massive-activation channels DiTF zeroes may
                        # overflow harmlessly, any other non-finite value would turn into NaN rows
                        bad = ~torch.isfinite(toks)
                        bad[..., [c for c in overflow_ok if c < C]] = False
                        if bad.any():
                            raise RuntimeError(
                                f"t={t}: {int(bad.sum())} non-finite fp16 token values outside the DiTF-discarded "
                                f"channels (image {n_done}, max |x| {float(feat_raw.float().abs().max()):.1f})"
                            )
                        toks = toks.cpu().numpy()
                        for j, k in enumerate(blocks):
                            mm[(t, k)][n_done] = toks[j]
                        ada = ada.detach().float().cpu()  # B, 3, C
                        if t not in mods:
                            mods[t] = ada
                        elif not torch.equal(mods[t], ada):
                            # adaLN depends on (t, guidance, null-prompt pooled embedding) only; one
                            # mod row per (t, block) is written, so it must hold for every image
                            raise RuntimeError(f"adaLN mod at t={t} differs between images (image {n_done})")
                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"  {split} {n_done}/{len(indices)} ({time.time() - t0:.0f}s)", flush=True)
        if n_done != len(indices):
            raise RuntimeError(f"{split}: extracted {n_done} of {len(indices)} images")
        for m in mm.values():
            m.flush()
            del m
        mods_arr = torch.stack([mods[t] for t in cfg.t], dim=0).numpy()  # K, B, 3, C
        h, w = grid
        stamp = block_stamp_for(model, blocks)  # against the MODEL's double-block count
        np.savez(
            meta_path(cfg.save_dir, split, tag),
            mods=mods_arr,
            timesteps=np.array(list(cfg.t)),
            blocks=np.array(blocks),
            paths=np.array(paths),
            masks=np.array(masks),
            subset_indices=indices,
            grid_hw=np.array([h, w]),
            eps_seed=np.array(eps_seed),
            ensemble_size=np.array(ens),
            img_size=np.array(cfg.img_size[0]),
            guidance_scale=np.array(cfg.guidance_scale),
            extraction_mode=np.array("ONESHOT"),
            split=np.array(split),
            weights=np.array(prov_meta["weights"]),
        )
        meta = {
            **prov_meta,
            **stamp,
            "task": "extract_seg_tokens",
            "dataset": cfg.dataset.name,
            "split": split,
            "n": int(n_done),
            "max_images": max_images,
            "t": list(cfg.t),
            "k": blocks,
            "extraction_mode": "ONESHOT",
            "guidance_scale": cfg.guidance_scale,
            "ensemble_size": ens,
            "eps_seed": eps_seed,
            "seed": cfg.seed,
            "img_size": list(cfg.img_size),
            "grid_hw": [h, w],
            "feat_dim": int(C),
            "dtype": "float16",
            "layout": "N,L,C per (t, k) file",
            "tag": tag,
            "files": {
                f"t{t}_k{k}": os.path.basename(token_path(cfg.save_dir, split, t, k, tag))
                for t in cfg.t
                for k in blocks
            },
            "images_sha1": hashlib.sha1("\n".join(paths).encode()).hexdigest()[:16],
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(meta_path(cfg.save_dir, split, tag, "json"), "w") as f:
            json.dump(meta, f, indent=1)
        gb = len(indices) * h * w * C * 2 * len(cfg.t) * len(blocks) / 1024**3
        print(f"[seg_tokens] {split}: {n_done} images x {h * w} tokens x {C}, {len(cfg.t) * len(blocks)} files, "
              f"{gb:.1f} GB, {meta['elapsed_s']}s", flush=True)  # fmt: skip
        print(f"SEG_TOKENS_OK {split}", flush=True)
        return meta
