"""Raw-state baseline for the INVERSION arm: probe the ODE states z_t, no block features.

The one-shot counterpart (raw_xt_baseline.py) is free because x_t = t*eps + (1-t)*x0 is a
closed form. The inversion state has NO closed form -- it is the RF-Solver trajectory -- so
this costs a full chain per image. Capturing the state itself is free once the chain runs
(invert_chain(want_states=True) clones the tensor already being fed to the model).

WHY IT IS THE SHARPER HALF OF THE COMPARISON. The inversion chain draws no eps at all, so
z_t is a DETERMINISTIC function of x0. The one-shot baseline decays with t because eps
swamps the (1-t)*x0 term at relative noise eta = t/(1-t). If the inversion raw-state probe
stays flat in t where the one-shot raw probe decays, the entire Tier-1 inversion advantage is
reproduced with no reference to learned representations at all -- it would be a property of
the input the network is handed, not of what the network computes from it.

Pooling matches raw_xt_baseline.py exactly. States are packed (1, T, d); they are unpacked
back to (1, 16, h, w) -- the inverse of feat_flux.prepare -- so `mean`/`q2`/`full` mean the
same operation on both arms and the dimensionalities line up.

FINDINGS. n=500 raw-state caches behind RESEARCH_NOTES 6 (inv-state column) and the
path-geometry probe (6g). Superseded for curvature work by solver_curvature.py, which
captures states AND velocities in one pass at n=5000.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from einops import rearrange

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

import datasets  # noqa: F401,E402
from registry import DATASETS  # noqa: E402
from run import DatasetConfig, ModelConfig, RunConfig  # noqa: E402
from tasks.extraction import _stratified_indices  # noqa: E402
from utils import seed_all, seed_worker  # noqa: E402

from models.flux.feat_flux import Featurizer4Eval  # noqa: E402

from torch.utils.data import DataLoader, Subset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_xt_baseline import POOLINGS, pool  # noqa: E402


def unpack(z: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(1, T, d) packed -> (1, 16, h, w). Inverse of feat_flux.prepare's rearrange."""
    return rearrange(z, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h // 2, w=w // 2, ph=2, pw=2)


def _env_suffix_and_meta():
    """Provenance stamp for the extraction-control env vars, mirroring
    tasks.extraction.env_provenance: these standalone extractors honor FLUX_RANDOM_INIT
    (via load_flow_model) and DEGRADE_TO (via the dataset hook) but previously wrote
    UN-suffixed cache names -- a stale export would poison the exact filenames the
    downstream probes glob (2026-09-04 review)."""
    from utils import env_int, env_value

    fixedcond = env_int("FIXED_COND_T", 1, 1000)
    degrade = env_int("DEGRADE_TO")
    parts = []
    if env_value("FLUX_RANDOM_INIT"):
        parts.append("RANDINIT")
    if fixedcond:
        parts.append(f"FIXEDCOND{fixedcond}")
    if degrade:
        parts.append(f"DEG{degrade}")
    suffix = "".join("_" + p for p in parts)
    meta = {
        "weights": "random_init" if env_value("FLUX_RANDOM_INIT") else "flux-dev",
        "degrade_to": degrade,
        "fixed_cond_t": fixedcond,
    }
    return suffix, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["eurosat", "resisc45"])
    p.add_argument("--path", required=True)
    p.add_argument("--img-size", type=int, nargs=2, required=True)
    p.add_argument("--t", type=int, nargs="+", default=[100, 180, 260, 340, 420, 500, 580])
    p.add_argument("--num-inversion-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--k", type=int, default=28)
    p.add_argument("--subset-size", type=int, default=500)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out-dir", default="models/raw_xt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RunConfig(
        task="extract",
        model=ModelConfig(name="flux", ensemble_size=1),
        dataset=DatasetConfig(name=args.dataset, path=args.path),
        img_size=list(args.img_size),
        t=list(args.t),
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        batch_size=1,
        num_workers=4,
        label_fraction=1.0,
    )
    seed_all(cfg.seed)

    dataset = DATASETS[cfg.dataset.name](cfg)
    train_ds = dataset.get_data(cfg)["train"].dataset
    all_labels = np.array([ci for _, ci, _ in train_ds.samples])
    indices = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)
    if args.max_images is not None:
        indices = indices[: args.max_images]
    loader = DataLoader(
        Subset(train_ds, indices.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print("loading FLUX (the chain needs the DiT)")
    fz = Featurizer4Eval(cat_list=list(dataset.category_list), ensemble_size=1)

    out: dict = {h: [] for h in POOLINGS}
    labels: list[int] = []
    hw = None
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)[0]  # C, H, W — invert_chain unsqueezes
            labels.append(int(batch["label"][0]))
            res = fz.invert_chain(
                img,
                cache_timesteps=list(args.t),
                num_inversion_steps=args.num_inversion_steps,
                block_idx=args.k,
                guidance=args.guidance,
                want_states=True,
            )
            missing = [t for t in args.t if t not in res["states"]]
            if missing:
                raise RuntimeError(f"states missing at {missing}")
            _, c, h, w = res["latents_clean"].shape
            if hw is None:
                hw = (c, h, w)
            elif hw != (c, h, w):
                raise RuntimeError(f"latent grid changed mid-extraction: {hw} -> {(c, h, w)}")
            zs = [unpack(res["states"][t], h, w).float() for t in args.t]  # each 1,16,h,w
            for how in POOLINGS:
                out[how].append(torch.stack([pool(z, how) for z in zs]).cpu().numpy())
            if n % 25 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    os.makedirs(args.out_dir, exist_ok=True)
    for how in POOLINGS:
        feats = np.stack(out[how]).astype(np.float32)
        base = f"{args.dataset}_invstate_n{args.num_inversion_steps}_{how}" + _env_suffix_and_meta()[0]
        np.savez(
            os.path.join(args.out_dir, base + ".npz"),
            feats=feats,
            labels=y,
            timesteps=np.array(args.t),
            subset_indices=indices,
            subset_seed=np.array(cfg.subset_seed),
        )
        with open(os.path.join(args.out_dir, base + "_meta.json"), "w") as f:
            json.dump(
                {
                    **_env_suffix_and_meta()[1],
                    "dataset": args.dataset,
                    "img_size": list(args.img_size),
                    "t": list(args.t),
                    "pooling": how,
                    "latent_chw": list(hw),
                    "num_inversion_steps": args.num_inversion_steps,
                    "guidance": args.guidance,
                    "k": args.k,
                    "subset_size": int(len(indices)),
                    "subset_seed": cfg.subset_seed,
                    "feats_shape": list(feats.shape),
                    "extraction_mode": "INVERSION_RAW_STATE_NO_BLOCK_FEATS",
                },
                f,
                indent=2,
            )
        print(f"wrote {base}.npz  {feats.shape}")


if __name__ == "__main__":
    main()
