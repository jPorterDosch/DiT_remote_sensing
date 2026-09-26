"""6w-D extraction: chain contraction (Lyapunov-style) on RESISC45, n=1500 paired subset.

Per image: encode the clean latent ONCE (posterior sample from the seeded global RNG),
then run TWO 50-step RF-Solver inversion chains — from z0 and from z0 + delta — via
invert_chain(latents=...), so both chains share the identical posterior sample and the
only difference is the perturbation. delta = fixed rademacher direction (drawn once,
seed 0, shared across images) scaled to 1e-2 x std(z0) per image.

Block (14 d): at each of the 7 cache timesteps, log10(||z_t' - z_t||_F / ||delta||_F)
(perturbation growth) and cos(flat(z_t' - z_t), flat(delta_packed)) (memory of the
perturbation direction). States are the packed (1, T, d) tensors from want_states=True.

n=1500: stratified subsample OF THE n5000 SUBSET (seed 7), so every image pairs with a
base-cache row via subset_indices. Cost: 2 chains x ~14 s = ~28 s/img -> ~12 h.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

import datasets  # noqa: F401,E402
from registry import DATASETS  # noqa: E402
from config_types import ExtractionMode  # noqa: E402
from run import DatasetConfig, ModelConfig, RunConfig  # noqa: E402
from tasks.extraction import _stratified_indices, env_provenance  # noqa: E402
from utils import seed_all, seed_worker  # noqa: E402

from models.flux.feat_flux import Featurizer4Eval, prepare  # noqa: E402

from torch.utils.data import DataLoader, Subset  # noqa: E402

T_LIST = [100, 180, 260, 340, 420, 500, 580]
N_SUB = 1500
REL_DELTA = 1e-2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default="results/lyapunov_chains_resisc45.npz")
    args = p.parse_args()

    device = torch.device("cuda")
    cfg = RunConfig(
        task="extract",
        model=ModelConfig(name="flux", ensemble_size=1),
        dataset=DatasetConfig(name="resisc45", path=args.path),
        img_size=[256, 256],
        t=list(T_LIST),
        extraction_mode=ExtractionMode.INVERSION,
        num_inversion_steps=50,
        subset_size=5000,
        subset_seed=42,
        batch_size=1,
        num_workers=4,
        label_fraction=1.0,
    )
    seed_all(cfg.seed)
    prov_suffix, prov_meta, _ = env_provenance(cfg, honors=("FLUX_RANDOM_INIT", "DEGRADE_TO"))

    dataset = DATASETS[cfg.dataset.name](cfg)
    train_ds = dataset.get_data(cfg)["train"].dataset
    all_labels = np.array([ci for _, ci, _ in train_ds.samples])
    idx5000 = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)

    # Stratified n=1500 SUBSAMPLE OF THE 5000 (seed 7) — every selected index exists in
    # the base cache's subset_indices, so per-image pairing survives.
    y5000 = all_labels[idx5000]
    rng = np.random.default_rng(7)
    keep = []
    per_class = N_SUB // len(np.unique(y5000))
    for cls in np.unique(y5000):
        pos = np.flatnonzero(y5000 == cls)
        keep.append(rng.choice(pos, size=min(per_class, len(pos)), replace=False))
    keep = np.sort(np.concatenate(keep))
    indices = idx5000[keep]
    if args.max_images is not None:
        keep, indices = keep[: args.max_images], indices[: args.max_images]
    print(f"n={len(indices)} (subsample of the n5000 subset; positions recorded)")

    loader = DataLoader(
        Subset(train_ds, indices.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    fz = Featurizer4Eval(cat_list=list(dataset.category_list), ensemble_size=1)
    ae = fz.ae

    delta_dir = None  # fixed rademacher direction, materialized at first latent
    dgen = torch.Generator(device="cpu").manual_seed(0)
    rows, labels = [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)
            labels.append(int(batch["label"][0]))
            z0 = ae.encode(img).to(torch.bfloat16)  # ONE posterior sample, shared by both chains
            if delta_dir is None:
                delta_dir = (torch.randint(0, 2, z0.shape, generator=dgen).float() * 2 - 1).to(device)
            scale = REL_DELTA * float(z0.float().std())
            delta = (delta_dir * scale).to(z0.dtype)
            res_a = fz.invert_chain(None, cache_timesteps=T_LIST, num_inversion_steps=50,
                                    block_idx=28, guidance=1.0, want_states=True, latents=z0)
            res_b = fz.invert_chain(None, cache_timesteps=T_LIST, num_inversion_steps=50,
                                    block_idx=28, guidance=1.0, want_states=True,
                                    latents=(z0.float() + delta.float()).to(z0.dtype))
            dpk, _ = prepare(img=delta)
            dflat = dpk.float().flatten()
            dnorm = float(dflat.norm())
            feats_row = []
            for t_nom in T_LIST:
                dz = (res_b["states"][t_nom].float() - res_a["states"][t_nom].float()).flatten()
                feats_row.append(float(np.log10(max(float(dz.norm()) / max(dnorm, 1e-12), 1e-12))))
                feats_row.append(float(torch.nn.functional.cosine_similarity(dz, dflat, dim=0)))
            rows.append(feats_row)
            if n % 25 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    block = np.array(rows, dtype=np.float32)  # (N, 14)
    out = args.out.replace(".npz", prov_suffix + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, block=block, labels=y, subset_indices=indices, keep_positions=keep,
             t=np.array(T_LIST), rel_delta=np.array(REL_DELTA))
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({**prov_meta, "t": T_LIST, "n": int(len(indices)), "rel_delta": REL_DELTA,
                   "num_inversion_steps": 50, "block_dims": int(block.shape[1]),
                   "note": "6w-D perturbed-pair chains; shared posterior sample; delta dir seed 0"}, f, indent=2)
    print(f"wrote {out}  block {block.shape}")


if __name__ == "__main__":
    main()
