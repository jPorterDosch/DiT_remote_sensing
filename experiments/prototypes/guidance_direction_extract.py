"""6w-B extraction: class-conditional guidance directions on RESISC45 (see RESEARCH_NOTES
6w-B pre-registration).

Per image, one-shot x_t at the a-priori t=260 (ens1, dedicated eps stream seed 42), then
46 velocity evaluations: the empty prompt + the 45 class prompts encoded by
experiments/encode_resisc45_prompts.py (→ experiments/prototypes/) (ALL 46 rows from that one file — including the
null arm, so every velocity in this experiment shares one text-encoding convention; the
historical null_embeddings.pt differs in the padding region, see that script's check).

Block (136 d): [ L_p = mean((v_p - u)^2) for p in null+45 (46);  ||pooled(v_c - v_null)||
(45);  cos(pooled delta_c, pooled v_null) (45) ]  with u = eps - x0 (the rectified-flow
target, matching train_diffusion's v_target = noise - latents) and pooled = token mean of
the packed (1, T, 64) prediction.

Side product (descriptive only): zero-shot accuracy of argmin_c L_c.

Cost: 46 forwards/image; ~4-6 s/img on an A6000 -> ~6-8 h at n=5000.
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

T_NOM = 260  # a-priori best t for the RESISC45 one-shot arm (RESEARCH_NOTES 6w-B)
PROMPT_FILE = "models/prompts/resisc45_prompt_embeds.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="data/resisc45/NWPU-RESISC45")
    p.add_argument("--subset-size", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None, help="smoke cap")
    p.add_argument("--out", default="results/guidance_direction_resisc45.npz")
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

    pf = torch.load(PROMPT_FILE, weights_only=False)
    prompt_embeds = pf["prompt_embeds"].to(device)  # (46, 512, 4096) bf16
    vecs = pf["vec"].to(device)  # (46, 768) bf16
    n_prompts = prompt_embeds.shape[0]
    assert n_prompts == 46, n_prompts
    txt_ids = torch.zeros(1, prompt_embeds.shape[1], 3, device=device)

    dataset = DATASETS[cfg.dataset.name](cfg)
    train_ds = dataset.get_data(cfg)["train"].dataset
    all_labels = np.array([ci for _, ci, _ in train_ds.samples])
    indices = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)
    if args.max_images is not None:
        indices = indices[: args.max_images]
    loader = DataLoader(
        Subset(train_ds, indices.tolist()),
        batch_size=1,
        shuffle=False,  # eps stream depends on deterministic order
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    fz = Featurizer4Eval(cat_list=list(dataset.category_list), ensemble_size=1)
    flux = fz.model
    ae = fz.ae
    t = T_NOM / 1000.0

    eps_gen = torch.Generator(device=device).manual_seed(cfg.eps_seed)
    L_all, dnorm_all, dcos_all, labels = [], [], [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)
            labels.append(int(batch["label"][0]))
            latents = ae.encode(img).to(torch.bfloat16)
            noise = torch.randn(latents.shape, generator=eps_gen, device=device, dtype=latents.dtype)
            # fp32 mixing, cast back (bf16 t rounding was a past bug — same regime as training)
            x_t = (t * noise.float() + (1.0 - t) * latents.float()).to(latents.dtype)
            xt_tok, img_ids = prepare(img=x_t)
            u_tok, _ = prepare(img=(noise.float() - latents.float()).to(latents.dtype))  # RF target
            u = u_tok.float()
            tvec = torch.full((1,), t, device=device, dtype=latents.dtype)
            gvec = torch.full((1,), 1.0, device=device, dtype=latents.dtype)

            preds_pooled = []
            L = np.zeros(n_prompts, dtype=np.float64)
            for pi in range(n_prompts):
                pred, _ = flux.forward_velocity_feat(
                    img=xt_tok,
                    img_ids=img_ids,
                    txt=prompt_embeds[pi : pi + 1],
                    txt_ids=txt_ids,
                    y=vecs[pi : pi + 1],
                    timesteps=tvec,
                    ft_indices=[28],
                    guidance=gvec,
                )
                pf32 = pred.float()
                L[pi] = float(((pf32 - u) ** 2).mean())
                preds_pooled.append(pf32.mean(dim=1)[0])  # (64,)
            pooled = torch.stack(preds_pooled)  # (46, 64)
            delta = pooled[1:] - pooled[0:1]  # (45, 64)
            dn = delta.norm(dim=1)
            dcos = torch.nn.functional.cosine_similarity(delta, pooled[0:1].expand_as(delta), dim=1)
            L_all.append(L)
            dnorm_all.append(dn.cpu().numpy())
            dcos_all.append(dcos.cpu().numpy())
            if n % 50 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    L_all = np.stack(L_all)  # (N, 46)
    block = np.concatenate([L_all, np.stack(dnorm_all), np.stack(dcos_all)], axis=1).astype(np.float32)
    zs_pred = L_all[:, 1:].argmin(axis=1)
    zs_acc = float((zs_pred == y).mean())
    print(f"zero-shot argmin-L accuracy (descriptive): {zs_acc:.4f} (chance {1 / 45:.4f})")

    out = args.out.replace(".npz", prov_suffix + ".npz")  # control runs can never claim the vanilla name
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        block=block,
        L=L_all.astype(np.float32),
        labels=y,
        subset_indices=indices,
        t=np.array(T_NOM),
        zero_shot_acc=np.array(zs_acc),
    )
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({**prov_meta, "t": T_NOM, "n": int(len(indices)), "eps_seed": cfg.eps_seed,
                   "prompts_file": PROMPT_FILE, "block_dims": int(block.shape[1]),
                   "note": "6w-B guidance directions; null arm = fresh empty-prompt encoding"}, f, indent=2)
    print(f"wrote {out}  block {block.shape}")


if __name__ == "__main__":
    main()
