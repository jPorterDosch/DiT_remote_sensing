"""6y-H extraction: rotation/flip equivariance defects (RESISC45, t=260, eps stream 42).

Per image and g in {rot90, rot180, rot270, hflip}: the group acts in PIXEL space on the
image and — crucially — the SAME eps is transported by the group action in latent space,
so the noise correspondence is preserved and the defect isolates the MODEL's response,
not a fresh noise draw. Block per g = [pooled-velocity delta (64); ||pooled block-28
feature delta||; cos(feat_g, feat)] -> 4 x 66 = 264 d. A perfectly rotation-equivariant
model has zero pooled-velocity defect (spatial mean is invariant under the spatial
action); the defect vector is an orientation-anisotropy fingerprint a single unrotated
forward provably never sees.

Cost: 5 VAE encodes + 5 forwards per image ~= 0.5 s -> ~45 min at n=5000.
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
GROUP = ("rot90", "rot180", "rot270", "hflip")


def act(x: torch.Tensor, g: str) -> torch.Tensor:
    """Group action on a (B, C, H, W) tensor's spatial dims."""
    if g == "rot90":
        return torch.rot90(x, 1, (-2, -1))
    if g == "rot180":
        return torch.rot90(x, 2, (-2, -1))
    if g == "rot270":
        return torch.rot90(x, 3, (-2, -1))
    if g == "hflip":
        return torch.flip(x, (-1,))
    raise ValueError(g)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--subset-size", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default="results/equivariance_defect_resisc45.npz")
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
    # Posterior-MEAN encoding: ae.encode otherwise SAMPLES per call, so the five encodes
    # per image would each draw fresh VAE noise and the "defect" would partly measure
    # posterior sampling variance instead of the model's rotation response.
    ae.reg.sample = False
    txt = fz.null_prompt_embeds.to(device=device, dtype=torch.bfloat16)
    txt_ids, vec = fz.text_ids.to(device), fz.vec.to(device=device, dtype=torch.bfloat16)
    t = T_NOM / 1000.0

    def vel_feat(img_px, eps_lat):
        latents = ae.encode(img_px).to(torch.bfloat16)
        x_t = (t * eps_lat.float() + (1.0 - t) * latents.float()).to(latents.dtype)
        xt_tok, img_ids = prepare(img=x_t)
        tvec = torch.full((1,), t, device=device, dtype=latents.dtype)
        gvec = torch.full((1,), 1.0, device=device, dtype=latents.dtype)
        pred, up_ft = flux.forward_velocity_feat(
            img=xt_tok, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
            timesteps=tvec, ft_indices=[28], guidance=gvec,
        )
        return pred.float().mean(dim=1)[0], up_ft[0].float().mean(dim=1)[0]  # (64,), (3072,)

    eps_gen = torch.Generator(device=device).manual_seed(cfg.eps_seed)
    rows, labels = [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)
            labels.append(int(batch["label"][0]))
            eps = torch.randn((1, 16, img.shape[-2] // 8, img.shape[-1] // 8),
                              generator=eps_gen, device=device, dtype=torch.bfloat16)
            v0, f0 = vel_feat(img, eps)
            feats_row = []
            for g in GROUP:
                vg, fg = vel_feat(act(img, g), act(eps, g))
                dv = (vg - v0).cpu().numpy()
                feats_row += list(dv)
                feats_row.append(float((fg - f0).norm()))
                feats_row.append(float(torch.nn.functional.cosine_similarity(fg, f0, dim=0)))
            rows.append(feats_row)
            if n % 50 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    block = np.array(rows, dtype=np.float32)  # (N, 264)
    out = args.out.replace(".npz", prov_suffix + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, block=block, labels=y, subset_indices=indices, t=np.array(T_NOM),
             group=np.array(GROUP))
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({**prov_meta, "t": T_NOM, "n": int(len(indices)), "eps_seed": cfg.eps_seed,
                   "group": list(GROUP), "block_dims": int(block.shape[1]),
                   "note": "6y-H equivariance defects; eps transported by the group action"}, f, indent=2)
    print(f"wrote {out}  block {block.shape}")


if __name__ == "__main__":
    main()
