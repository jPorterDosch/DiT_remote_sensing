"""Frozen DINO-family feature extraction (CLS + mean-patch) for the eval pipeline.

    python -m eval.extract_dino --preset dinov2_vitl14 --dataset resisc45          # CV identity set
    python -m eval.extract_dino --preset dinov2_vitl14 --dataset m_eurosat         # official splits
    python -m eval.extract_dino --preset dinov3_vit7b16_sat --dataset m_forestnet  # weights: see PRESETS

resisc45 (any CV dataset): the images are the CV identity cache's paths in cache order
(eval/features.CV_IDENTITY), so the output pairs with every FLUX arm.
-> results/eval_feats/<preset>_<dataset>_n<N>.npz
Official datasets: every image of every official split, class-list order.
-> results/eval_feats/<preset>_<dataset>_<split>.npz

Model loading and the forward pass are dino_family_resisc45's (which generalized
dinov2_resisc45/dinov2_m_eurosat). DINOv3 presets load a local gated checkpoint (Meta's
download form -> ditf_models/dinov3/, ISAAC scratch) and refuse a file whose name lacks the
preset's hash: web and sat weights need different normalization, so a swapped file would
otherwise run silently mislabelled.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # noqa: E402

from eval import features as F  # noqa: E402
from eval import wb  # noqa: E402

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# dinov3 README "Image transforms", SAT-493M weights (read from the shipped repo 2026-09-29)
SAT493M = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))


def _dinov3(entry: str, data: str, hash_: str, norm: tuple, dtype, batch: int) -> dict:
    ckpt = f"ditf_models/dinov3/{entry}_pretrain_{data}-{hash_}.pth"
    return dict(
        repo="facebookresearch/dinov3",
        entry=entry,
        weights=ckpt,
        hash=hash_,
        norm=norm,
        dtype=dtype,
        batch=batch,
    )


# Web (LVD-1689M) vs satellite (SAT-493M) pretraining at matched architecture (ViT-7B/16, the
# pre-registered 6ad pair and Meta's Table-18 comparison); hashes from dinov3/hub/backbones.py.
PRESETS = {
    "dinov2_vitl14": dict(
        repo="facebookresearch/dinov2",
        entry="dinov2_vitl14",
        weights=None,
        norm=IMAGENET,
        dtype=torch.float32,
        batch=64,
    ),
    "dinov3_vit7b16_web": _dinov3("dinov3_vit7b16", "lvd1689m", "a955f4ea", IMAGENET, torch.bfloat16, 8),
    "dinov3_vit7b16_sat": _dinov3("dinov3_vit7b16", "sat493m", "a6675841", SAT493M, torch.bfloat16, 8),
}


def load_model(preset, weights, device):
    """Source: dino_family_resisc45.load_model."""
    spec = PRESETS[preset]
    kwargs = {}
    if weights and not spec["weights"]:
        raise SystemExit(f"{preset} loads its hub weights; --weights would be ignored")
    if spec["weights"]:
        weights = weights or spec["weights"]
        if spec["hash"] not in os.path.basename(weights):
            raise SystemExit(f"{preset}: checkpoint {weights} lacks hash {spec['hash']} -- wrong weights")
        if not os.path.exists(weights):
            raise SystemExit(
                f"{preset}: {weights} missing (gated: ai.meta.com/resources/models-and-libraries/dinov3-downloads)"
            )
        kwargs["weights"] = weights
    model = torch.hub.load(spec["repo"], spec["entry"], **kwargs)
    model = model.to(device=device, dtype=spec["dtype"]).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{preset}: {n_par / 1e9:.2f} B params, dtype {spec['dtype']}, batch {spec['batch']}")
    return model, spec


@torch.no_grad()
def extract(model, spec, paths, device, mean, std):
    """CLS + mean-patch of the last block (cls, mp: source dino_family_resisc45.extract) and
    the CLS tokens of the last 4 blocks concatenated (cls4) -- the inputs of DINOv2/v3's own
    linear eval, create_linear_input(n_last_blocks in {1, 4}, use_avgpool) in
    dinov3/eval/linear.py. One pass via get_intermediate_layers(norm=True); batch 0 is
    checked against forward_features so cls/mp keep their old definition (incl. DINOv3's
    untied cls norm)."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    batch = spec["batch"]
    cls_all, mp_all, cls4_all = [], [], []
    for i in range(0, len(paths), batch):
        imgs = []
        for f in paths[i : i + batch]:
            im = Image.open(f).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        x = ((torch.stack(imgs) - mean_t) / std_t).to(device=device, dtype=spec["dtype"])
        blocks = model.get_intermediate_layers(x, n=4, return_class_token=True, norm=True)
        patch, cls = blocks[-1]
        if i == 0:
            ref = model.forward_features(x)
            for got, want in ((cls, ref["x_norm_clstoken"]), (patch, ref["x_norm_patchtokens"])):
                if not torch.allclose(got.float(), want.float(), atol=1e-3, rtol=1e-3):
                    raise SystemExit("get_intermediate_layers' last block != forward_features -- refusing")
        cls_all.append(cls.float().cpu().numpy())
        mp_all.append(patch.mean(dim=1).float().cpu().numpy())
        cls4_all.append(torch.cat([c for _, c in blocks], dim=-1).float().cpu().numpy())
        if i % (batch * 20) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.concatenate(cls_all), np.concatenate(mp_all), np.concatenate(cls4_all)


def main(argv=None) -> list[str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument(
        "--dataset", required=True, help="a CV_IDENTITY key (resisc45) or an OFFICIAL key (m_eurosat)"
    )
    p.add_argument("--weights", default=None, help="override the preset's checkpoint path (DINOv3)")
    p.add_argument("--max-images", type=int, default=None, help="SMOKE: strided cap per split")
    p.add_argument("--out-dir", default="results/eval_feats")
    args = p.parse_args(argv)

    spec = PRESETS[args.preset]
    mean, std = spec["norm"]

    if args.dataset in F.CV_IDENTITY:
        paths, y = F.load_identity(F.CV_IDENTITY[args.dataset])
        jobs = {f"n{len(paths)}": (paths, y)}
    elif args.dataset in F.OFFICIAL:
        jobs = {s: F.list_official_split(args.dataset, s) for s in F.official_spec(args.dataset)["sizes"]}
    else:
        raise SystemExit(f"{args.dataset}: neither a CV_IDENTITY nor an OFFICIAL dataset")
    if args.max_images:
        jobs = {
            k: ([ps[i] for i in keep], ys[keep])
            for k, (ps, ys) in jobs.items()
            for keep in [np.linspace(0, len(ps) - 1, args.max_images).astype(int)]
        }
    tag = "_smoke" if args.max_images else ""

    config = {
        "preset": args.preset,
        "weights": args.weights or spec["weights"],
        "mean": mean,
        "std": std,
        "img": 224,
        "pool": "cls+meanpatch+cls4",
        "max_images": args.max_images,
        # which images, in which order (rule 11): the run name changes with the image set
        "images": {
            k: hashlib.sha1("\n".join(ps).encode()).hexdigest()[:16] + f"/{len(ps)}"
            for k, (ps, _) in jobs.items()
        },
    }
    _, name = wb.init(
        "extract-dino" + ("-smoke" if args.max_images else ""),
        args.dataset,
        args.preset,
        config,
        job_type="extract",
    )
    print(f"== {name}  normalization mean {mean} std {std}")
    device = "cuda"
    model, spec = load_model(args.preset, args.weights, device)
    os.makedirs(args.out_dir, exist_ok=True)
    outs = []
    for split, (paths, y) in jobs.items():
        print(f"{split}: {len(paths)} images")
        cls_f, mp_f, cls4_f = extract(model, spec, paths, device, mean, std)
        out = os.path.join(args.out_dir, f"{args.preset}_{args.dataset}_{split}{tag}.npz")
        np.savez(
            out,
            cls=cls_f,
            mp=mp_f,
            cls4=cls4_f,
            paths=np.array(paths),
            labels=y,
            mean=np.array(mean),
            std=np.array(std),
            preset=np.array(args.preset),
        )
        print(f"cached cls {cls_f.shape} + mp {mp_f.shape} + cls4 {cls4_f.shape} to {out}")
        wb.log_features(out)
        outs.append(out)
    if wandb.run is not None:
        wandb.finish()
    return outs


if __name__ == "__main__":
    main()
