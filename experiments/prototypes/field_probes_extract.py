"""6w-C extraction: local velocity-field geometry via finite-difference probes (RESISC45).

Per image and t in {100, 260, 580}: v0 = v(x_t), then for 4 FIXED rademacher directions
u_j (drawn once, seed 0, shared across ALL images so the feature means the same thing
everywhere): v± = v(x_t ± h·u_j) with h = 1e-2 · std(x_t). J u_j ≈ (v+ − v−) / (2h).

Stats per (t, j):  hutch_j = <u_j, J u_j>  (the Hutchinson quadratic — 4 of these
average to a divergence estimate up to dimensional scaling);  ||J u_j||_F  (field
stiffness along u_j);  cos(J u_j, v0)  (whether the response aligns with the flow).
Block = 3t x 4j x 3 stats + 3 per-t hutch means = 39 d (RESEARCH_NOTES 6w-C).
NOTE: local one-shot density-rate geometry, NOT integrated likelihood.

Cost: 9 forwards per (image, t) x 3 t = 27/img; ~2.5 s/img -> ~3.5 h at n=5000.
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

T_LIST = [100, 260, 580]
N_PROBES = 4
REL_H = 1e-2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--subset-size", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default="results/field_probes_resisc45.npz")
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
    txt, txt_ids, vec = fz.null_prompt_embeds, fz.text_ids, fz.vec
    txt = txt.to(device=device, dtype=torch.bfloat16)
    vec = vec.to(device=device, dtype=torch.bfloat16)

    # Fixed probe directions, drawn once and shared across every image (packed shape is
    # constant at 256px: T=256 tokens x 64 channels).
    probe_gen = torch.Generator(device="cpu").manual_seed(0)
    U = None  # materialized on the first image once the packed shape is known

    def velocity(xt_tok, img_ids, tvec, gvec):
        pred, _ = flux.forward_velocity_feat(
            img=xt_tok, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
            timesteps=tvec, ft_indices=[28], guidance=gvec,
        )
        return pred.float()

    eps_gen = torch.Generator(device=device).manual_seed(cfg.eps_seed)
    rows, labels = [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)
            labels.append(int(batch["label"][0]))
            latents = ae.encode(img).to(torch.bfloat16)
            noise = torch.randn(latents.shape, generator=eps_gen, device=device, dtype=latents.dtype)
            feats_row = []
            for t_nom in T_LIST:
                t = t_nom / 1000.0
                x_t = (t * noise.float() + (1.0 - t) * latents.float()).to(latents.dtype)
                xt_tok, img_ids = prepare(img=x_t)
                if U is None:
                    U = (torch.randint(0, 2, (N_PROBES, *xt_tok.shape[1:]), generator=probe_gen).float() * 2 - 1).to(device)
                tvec = torch.full((1,), t, device=device, dtype=latents.dtype)
                gvec = torch.full((1,), 1.0, device=device, dtype=latents.dtype)
                v0 = velocity(xt_tok, img_ids, tvec, gvec)
                v0f = v0.flatten()
                h = REL_H * float(xt_tok.float().std())
                hutches = []
                for j in range(N_PROBES):
                    uj = U[j : j + 1]
                    vp = velocity((xt_tok.float() + h * uj).to(xt_tok.dtype), img_ids, tvec, gvec)
                    vm = velocity((xt_tok.float() - h * uj).to(xt_tok.dtype), img_ids, tvec, gvec)
                    Ju = ((vp - vm) / (2 * h)).flatten()
                    hutch = float((uj.flatten() * Ju).sum())
                    hutches.append(hutch)
                    feats_row += [hutch, float(Ju.norm()),
                                  float(torch.nn.functional.cosine_similarity(Ju, v0f, dim=0))]
                feats_row.append(float(np.mean(hutches)))
            rows.append(feats_row)
            if n % 50 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    block = np.array(rows, dtype=np.float32)  # (N, 39)
    out = args.out.replace(".npz", prov_suffix + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, block=block, labels=y, subset_indices=indices, t=np.array(T_LIST),
             n_probes=np.array(N_PROBES), rel_h=np.array(REL_H))
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({**prov_meta, "t": T_LIST, "n": int(len(indices)), "eps_seed": cfg.eps_seed,
                   "n_probes": N_PROBES, "rel_h": REL_H, "block_dims": int(block.shape[1]),
                   "note": "6w-C finite-difference field probes; fixed rademacher dirs seed 0"}, f, indent=2)
    print(f"wrote {out}  block {block.shape}")


if __name__ == "__main__":
    main()
