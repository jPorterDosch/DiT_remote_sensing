"""Frozen DINO-family feature extraction (CLS + mean-patch) for the eval pipeline.

    python -m eval.extract_dino --preset dinov2_vitl14 --dataset resisc45          # CV identity set
    python -m eval.extract_dino --preset dinov2_vitl14 --dataset m_eurosat         # official splits
    python -m eval.extract_dino --preset dinov3_sat7b --weights <pth> --mean M M M --std S S S \\
        --dataset resisc45

resisc45 (any CV dataset): the images are the CV identity cache's paths in cache order
(eval/features.CV_IDENTITY), so the output pairs with every FLUX arm.
-> results/eval_feats/<preset>_<dataset>_n<N>.npz
Official datasets: every image of every official split, class-list order.
-> results/eval_feats/<preset>_<dataset>_<split>.npz

Model loading and the forward pass are dino_family_resisc45's (which generalized
dinov2_resisc45/dinov2_m_eurosat); the sat preset REFUSES ImageNet normalization.
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
PRESETS = {
    "dinov2_vitl14": dict(
        repo="facebookresearch/dinov2",
        entry="dinov2_vitl14",
        needs_weights=False,
        dtype=torch.float32,
        batch=64,
    ),
    "dinov3_web7b": dict(
        repo="facebookresearch/dinov3",
        entry="dinov3_vit7b16",
        needs_weights=True,
        dtype=torch.bfloat16,
        batch=8,
    ),
    "dinov3_sat7b": dict(
        repo="facebookresearch/dinov3",
        entry="dinov3_vit7b16",
        needs_weights=True,
        dtype=torch.bfloat16,
        batch=8,
        needs_explicit_norm=True,
    ),
}


def load_model(preset, weights, device):
    """Source: dino_family_resisc45.load_model."""
    spec = PRESETS[preset]
    kwargs = {}
    if spec["needs_weights"]:
        if not (weights and os.path.exists(weights)):
            raise SystemExit(f"--weights required and must exist for {preset}")
        kwargs["weights"] = weights
    model = torch.hub.load(spec["repo"], spec["entry"], **kwargs)
    model = model.to(device=device, dtype=spec["dtype"]).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{preset}: {n_par / 1e9:.2f} B params, dtype {spec['dtype']}, batch {spec['batch']}")
    return model, spec


@torch.no_grad()
def extract(model, spec, paths, device, mean, std):
    """Source: dino_family_resisc45.extract (verbatim)."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    batch = spec["batch"]
    cls_all, mp_all = [], []
    for i in range(0, len(paths), batch):
        imgs = []
        for f in paths[i : i + batch]:
            im = Image.open(f).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        x = ((torch.stack(imgs) - mean_t) / std_t).to(device=device, dtype=spec["dtype"])
        out = model.forward_features(x)
        if not (isinstance(out, dict) and "x_norm_clstoken" in out and "x_norm_patchtokens" in out):
            raise SystemExit(f"unexpected forward_features output: {type(out)}")
        cls_all.append(out["x_norm_clstoken"].float().cpu().numpy())
        mp_all.append(out["x_norm_patchtokens"].mean(dim=1).float().cpu().numpy())
        if i % (batch * 20) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.concatenate(cls_all), np.concatenate(mp_all)


def main(argv=None) -> list[str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument(
        "--dataset", required=True, help="a CV_IDENTITY key (resisc45) or an OFFICIAL key (m_eurosat)"
    )
    p.add_argument("--weights", default=None, help="local checkpoint path (gated presets)")
    p.add_argument("--mean", type=float, nargs=3, default=None)
    p.add_argument("--std", type=float, nargs=3, default=None)
    p.add_argument("--max-images", type=int, default=None, help="SMOKE: strided cap per split")
    p.add_argument("--out-dir", default="results/eval_feats")
    args = p.parse_args(argv)

    spec = PRESETS[args.preset]
    if spec.get("needs_explicit_norm") and (args.mean is None or args.std is None):
        raise SystemExit(
            f"{args.preset}: pass --mean/--std copied from the SHIPPED dinov3 repo README "
            f"({torch.hub.get_dir()}/facebookresearch_dinov3_main/README* after first hub load). "
            "Sat-493M does not use ImageNet statistics (rule 16: verify the artifact)."
        )
    mean, std = (args.mean or IMAGENET[0]), (args.std or IMAGENET[1])

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
        "weights": args.weights,
        "mean": mean,
        "std": std,
        "img": 224,
        "pool": "cls+meanpatch",
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
        cls_f, mp_f = extract(model, spec, paths, device, mean, std)
        out = os.path.join(args.out_dir, f"{args.preset}_{args.dataset}_{split}{tag}.npz")
        np.savez(
            out,
            cls=cls_f,
            mp=mp_f,
            paths=np.array(paths),
            labels=y,
            mean=np.array(mean),
            std=np.array(std),
            preset=np.array(args.preset),
        )
        print(f"cached {cls_f.shape} + {mp_f.shape} to {out}")
        wb.log_features(out)
        outs.append(out)
    if wandb.run is not None:
        wandb.finish()
    return outs


if __name__ == "__main__":
    main()
