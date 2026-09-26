"""6z extraction: FM loss t-profile (I) and explicit time-partial dv/dt at fixed x (II).

One pass, RESISC45 n=5000, ens1, eps stream 42 (one eps per image, reused across all t —
the repo's consistency convention). Per image and per t in the standing 7-t grid:
  v  = v_theta(x_t, t)          -> L(t) = mean((v - u)^2), u = eps - x0        (I)
  vd = v_theta(x_t, t + DELTA)  -> dt-partial = (vd - v)/DELTA (same x_t!)     (II)
Blocks: I = 7 d; II per t = [pooled 64; ||.||_F; cos(., v)] -> 462 d.
See RESEARCH_NOTES 6z for the pre-registration.
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
from run import DatasetConfig, ModelConfig, RunConfig  # noqa: E402
from tasks.extraction import _stratified_indices, env_provenance  # noqa: E402
from utils import seed_all, seed_worker  # noqa: E402

from models.flux.feat_flux import Featurizer4Eval, prepare  # noqa: E402

from torch.utils.data import DataLoader, Subset  # noqa: E402

T_LIST = [100, 180, 260, 340, 420, 500, 580]
DELTA = 0.02  # a-priori; grid spacing is 0.08


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--subset-size", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default="results/loss_profile_resisc45.npz")
    args = p.parse_args()

    device = torch.device("cuda")
    cfg = RunConfig(
        task="extract",
        model=ModelConfig(name="flux", ensemble_size=1),
        dataset=DatasetConfig(name="resisc45", path=args.path),
        img_size=[256, 256],
        t=list(T_LIST),
        subset_size=args.subset_size,
        subset_seed=42,
        eps_seed=42,
        batch_size=1,
        num_workers=4,
        label_fraction=1.0,
    )
    seed_all(cfg.seed)
    prov_suffix, prov_meta, _ = env_provenance(cfg, honors=("FLUX_RANDOM_INIT", "DEGRADE_TO"))

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

    fz = Featurizer4Eval(cat_list=list(dataset.category_list), ensemble_size=1)
    flux, ae = fz.model, fz.ae
    txt = fz.null_prompt_embeds.to(device=device, dtype=torch.bfloat16)
    txt_ids, vec = fz.text_ids.to(device), fz.vec.to(device=device, dtype=torch.bfloat16)

    def velocity(xt_tok, img_ids, t_val, ref_dtype):
        tvec = torch.full((1,), t_val, device=device, dtype=ref_dtype)
        gvec = torch.full((1,), 1.0, device=device, dtype=ref_dtype)
        pred, _ = flux.forward_velocity_feat(
            img=xt_tok, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
            timesteps=tvec, ft_indices=[28], guidance=gvec,
        )
        return pred.float()

    eps_gen = torch.Generator(device=device).manual_seed(cfg.eps_seed)
    I_rows, II_rows, labels = [], [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)
            labels.append(int(batch["label"][0]))
            latents = ae.encode(img).to(torch.bfloat16)
            noise = torch.randn(latents.shape, generator=eps_gen, device=device, dtype=latents.dtype)
            u_tok, _ = prepare(img=(noise.float() - latents.float()).to(latents.dtype))
            u = u_tok.float()
            L_row, II_row = [], []
            for t_nom in T_LIST:
                t = t_nom / 1000.0
                x_t = (t * noise.float() + (1.0 - t) * latents.float()).to(latents.dtype)
                xt_tok, img_ids = prepare(img=x_t)
                v = velocity(xt_tok, img_ids, t, latents.dtype)
                vd = velocity(xt_tok, img_ids, t + DELTA, latents.dtype)  # SAME x_t
                L_row.append(float(((v - u) ** 2).mean()))
                dpart = (vd - v) / DELTA  # (1, T, 64)
                II_row += list(dpart.mean(dim=1)[0].cpu().numpy())
                II_row.append(float(dpart.norm()))
                II_row.append(float(torch.nn.functional.cosine_similarity(
                    dpart.flatten(), v.flatten(), dim=0)))
            I_rows.append(L_row)
            II_rows.append(II_row)
            if n % 50 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    I_block = np.array(I_rows, dtype=np.float32)  # (N, 7)
    II_block = np.array(II_rows, dtype=np.float32)  # (N, 462)
    out = args.out.replace(".npz", prov_suffix + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, I_block=I_block, II_block=II_block, labels=y, subset_indices=indices,
             t=np.array(T_LIST), delta=np.array(DELTA))
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({**prov_meta, "t": T_LIST, "n": int(len(indices)), "eps_seed": cfg.eps_seed,
                   "delta": DELTA, "I_dims": int(I_block.shape[1]), "II_dims": int(II_block.shape[1]),
                   "note": "6z: FM loss profile + explicit dt-partial at fixed x"}, f, indent=2)
    print(f"wrote {out}  I {I_block.shape}  II {II_block.shape}")


if __name__ == "__main__":
    main()
