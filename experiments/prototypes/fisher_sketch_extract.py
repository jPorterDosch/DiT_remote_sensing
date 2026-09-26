"""6y-G extraction: partial-Fisher gradient sketch (RESISC45, t=260, eps stream 42).

Per image: one full forward of the flow-matching loss L = mean((v_theta(x_t) - u)^2) and
one backward restricted to the parameters of the k=28 single-stream block (the SAME site
the features are read from — the gradient there is activation x backprop-delta, and the
delta is the component activations cannot contain). The ~10^8-dim gradient is sketched to
4096 d by a fixed seeded COUNT-SKETCH (signed feature hashing; JL family — implementation
of 6y-G's 'seeded random projection', chosen because a dense rademacher matrix at this
width is not materializable).

Cost: forward+backward ~= 3 forward-equivalents ~= 0.3 s/img -> ~30 min at n=5000.
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

T_NOM = 260
SKETCH_D = 4096
K_GLOBAL = 28  # single_blocks index = 28 - 19 = 9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--subset-size", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default="results/fisher_sketch_resisc45.npz")
    args = p.parse_args()

    device = torch.device("cuda")
    cfg = RunConfig(
        task="extract",
        model=ModelConfig(name="flux", ensemble_size=1),
        dataset=DatasetConfig(name="resisc45", path=args.path),
        img_size=[256, 256],
        t=[T_NOM],
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

    # Gradients only for the k=28 single block; everything else frozen.
    for prm in flux.parameters():
        prm.requires_grad_(False)
    block = flux.single_blocks[K_GLOBAL - 19]
    params = [prm for prm in block.parameters()]
    for prm in params:
        prm.requires_grad_(True)
    n_grad = sum(prm.numel() for prm in params)
    print(f"grad params in single block {K_GLOBAL - 19}: {n_grad / 1e6:.1f} M")

    # Fixed count-sketch: hash index + sign per gradient coordinate, drawn once (seed 0).
    gen = torch.Generator(device="cpu").manual_seed(0)
    hidx, sign = [], []
    for prm in params:
        n_i = prm.numel()
        hidx.append(torch.randint(0, SKETCH_D, (n_i,), generator=gen))
        sign.append((torch.randint(0, 2, (n_i,), generator=gen).float() * 2 - 1))
    hidx = [h.to(device) for h in hidx]
    sign = [s.to(device) for s in sign]

    t = T_NOM / 1000.0
    eps_gen = torch.Generator(device=device).manual_seed(cfg.eps_seed)
    rows, loss_vals, labels = [], [], []
    for n, batch in enumerate(loader):
        img = batch["img"].to(device)
        labels.append(int(batch["label"][0]))
        with torch.no_grad():
            latents = ae.encode(img).to(torch.bfloat16)
            noise = torch.randn(latents.shape, generator=eps_gen, device=device, dtype=latents.dtype)
            x_t = (t * noise.float() + (1.0 - t) * latents.float()).to(latents.dtype)
            xt_tok, img_ids = prepare(img=x_t)
            u_tok, _ = prepare(img=(noise.float() - latents.float()).to(latents.dtype))
        tvec = torch.full((1,), t, device=device, dtype=latents.dtype)
        gvec = torch.full((1,), 1.0, device=device, dtype=latents.dtype)
        pred, _ = flux.forward_velocity_feat(
            img=xt_tok, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
            timesteps=tvec, ft_indices=[K_GLOBAL], guidance=gvec,
        )
        loss = ((pred.float() - u_tok.float()) ** 2).mean()
        for prm in params:
            prm.grad = None
        loss.backward()
        sk = torch.zeros(SKETCH_D, device=device, dtype=torch.float32)
        for prm, h, s in zip(params, hidx, sign):
            g = prm.grad.detach().float().flatten()
            sk.scatter_add_(0, h, g * s)
        rows.append(sk.cpu().numpy())
        loss_vals.append(float(loss.detach()))
        if n % 50 == 0:
            print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    block_arr = np.stack(rows).astype(np.float32)
    out = args.out.replace(".npz", prov_suffix + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, block=block_arr, labels=y, subset_indices=indices,
             t=np.array(T_NOM), loss=np.array(loss_vals, dtype=np.float32))
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({**prov_meta, "t": T_NOM, "n": int(len(indices)), "eps_seed": cfg.eps_seed,
                   "sketch_d": SKETCH_D, "grad_params": int(n_grad), "block": K_GLOBAL,
                   "note": "6y-G partial-Fisher count-sketch, single block 9 params"}, f, indent=2)
    print(f"wrote {out}  block {block_arr.shape}")


if __name__ == "__main__":
    main()
