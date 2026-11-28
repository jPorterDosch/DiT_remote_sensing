"""Frozen DINO-family feature extraction (CLS + mean-patch) for the eval pipeline.

    python -m eval.extract_dino --preset dinov2_vitl14 --dataset resisc45          # CV identity set
    python -m eval.extract_dino --preset dinov2_vitl14 --dataset m_eurosat         # official splits
    python -m eval.extract_dino --preset dinov3_vit7b16_sat --dataset m_forestnet  # weights: see PRESETS

resisc45 (any CV dataset): the images are the CV identity cache's paths in cache order
(eval/features.CV_IDENTITY), so the output pairs with every FLUX arm.
-> results/eval_feats/<preset>_<dataset>_n<N>.npz
Official datasets: every image of every official split, class-list order.
-> results/eval_feats/<preset>_<dataset>_<split>.npz
Segmentation (eval/features.OFFICIAL_SEG), `--tokens --img 512`: the last block's normed PATCH
tokens of every image, (N, L, D) float16 memmap written image-batch by image-batch (L = (img/16)^2,
row-major over the patch grid), for the per-token head of eval/seg_probe.py.
-> <out-dir>/<preset>_<dataset>_<split>_tokens<img>.npy + _meta.json

Model loading and the forward pass are dino_family_resisc45's (which generalized
dinov2_resisc45/dinov2_m_eurosat). DINOv3 presets load a local gated checkpoint (Meta's
download form -> ditf_models/dinov3/, ISAAC scratch) and refuse a file whose name lacks the
preset's hash: web and sat weights need different normalization, so a swapped file would
otherwise run silently mislabelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    # Weak-teacher control (2026-11-07): ViT-L/16 web weights, same family / patch size / loader as
    # the 7B teachers so only teacher capacity and domain change (0.3 B vs 7 B, LVD-1689M).
    "dinov3_vitl16_web": _dinov3("dinov3_vitl16", "lvd1689m", "8aa4cbdd", IMAGENET, torch.bfloat16, 32),
}


DTYPES = {"float32": torch.float32}  # --dtype overrides of a preset's dtype


def resolve_dtype(preset: str, device: str, dtype: str):
    """The torch dtype of a run. 'preset' keeps the preset's dtype (bf16 for DINOv3, chosen so the
    7B models fit one GPU). bf16 on a CPU is refused: it is orders of magnitude slower there and
    the caller must say that the features are float32, not get them silently."""
    dt = PRESETS[preset]["dtype"] if dtype == "preset" else DTYPES[dtype]
    if device == "cpu" and dt != torch.float32:
        raise SystemExit(f"{preset} on --device cpu needs --dtype float32 (the preset dtype is {dt})")
    return dt


def load_model(preset, weights, device, dtype=None):
    """Source: dino_family_resisc45.load_model. dtype: None = the preset's."""
    spec = dict(PRESETS[preset])
    if dtype is not None:
        spec["dtype"] = dtype
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
def extract_tokens(model, spec, paths, device, mean, std, img: int, out_path: str) -> tuple[int, int]:
    """Last-block normed patch tokens (get_intermediate_layers(n=1, norm=True), the tensor whose
    spatial mean is `mp` in extract()) at img x img, written straight into an (N, L, D) float16
    .npy memmap. Returns the patch grid (h, w). Batch 0 is checked against forward_features."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    batch = max(1, spec["batch"] * (224 * 224) // (img * img))
    mm, grid = None, None
    for i in range(0, len(paths), batch):
        imgs = []
        for f in paths[i : i + batch]:
            im = Image.open(f).convert("RGB").resize((img, img), Image.Resampling.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        x = ((torch.stack(imgs) - mean_t) / std_t).to(device=device, dtype=spec["dtype"])
        ((patch, cls),) = model.get_intermediate_layers(x, n=1, return_class_token=True, norm=True)
        if i == 0:
            ref = model.forward_features(x)
            if not torch.allclose(patch.float(), ref["x_norm_patchtokens"].float(), atol=1e-3, rtol=1e-3):
                raise SystemExit("get_intermediate_layers' last block != forward_features -- refusing")
            n_tok = patch.shape[1]
            side = int(round(n_tok**0.5))
            if side * side != n_tok or side != img // model.patch_size:
                raise SystemExit(
                    f"{n_tok} patch tokens is not the {img // model.patch_size}^2 grid of img {img}"
                )
            grid = (side, side)
            mm = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float16, shape=(len(paths), n_tok, patch.shape[-1])
            )
        mm[i : i + len(imgs)] = patch.to(torch.float16).cpu().numpy()
        if i % (batch * 20) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    mm.flush()
    del mm
    return grid


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
    p.add_argument(
        "--tokens", action="store_true", help="segmentation: per-patch tokens of an OFFICIAL_SEG dataset"
    )
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument(
        "--dtype",
        default="preset",
        choices=("preset", *DTYPES),
        help="'preset' = the preset's dtype (bf16 for DINOv3); float32 is required on --device cpu. "
        "Recorded in the cache (dtype, device).",
    )
    p.add_argument(
        "--img", type=int, default=224, help="--tokens: input side (512 -> 32 x 32 patches of ViT/16)"
    )
    args = p.parse_args(argv)

    spec = PRESETS[args.preset]
    mean, std = spec["norm"]
    dtype = resolve_dtype(args.preset, args.device, args.dtype)  # before any I/O: a bad pair fails here

    if args.tokens:
        if args.device != "cuda" or args.dtype != "preset":
            raise SystemExit("--tokens runs at the preset dtype on cuda only (the token caches' definition)")
        return _tokens_main(args, spec, mean, std)
    if args.img != 224:
        raise SystemExit("--img is for --tokens; the pooled caches are 224 px")
    if args.dataset in F.CV_IDENTITY:
        paths, y = F.load_identity(F.CV_IDENTITY[args.dataset])
        jobs = {f"n{len(paths)}": (paths, y)}
    elif args.dataset in F.OFFICIAL or args.dataset in F.OFFICIAL_CXR:
        jobs = {s: F.list_official_split(args.dataset, s) for s in F.official_spec(args.dataset)["sizes"]}
    else:
        raise SystemExit(f"{args.dataset}: neither a CV_IDENTITY nor an OFFICIAL / OFFICIAL_CXR dataset")
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
    if (args.device, args.dtype) != ("cuda", "preset"):  # default runs keep their config (and run name)
        config |= {"device": args.device, "dtype": str(dtype)}
    _, name = wb.init(
        "extract-dino" + ("-smoke" if args.max_images else ""),
        args.dataset,
        args.preset,
        config,
        job_type="extract",
    )
    print(f"== {name}  normalization mean {mean} std {std}")
    device = args.device
    model, spec = load_model(args.preset, args.weights, device, dtype)
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
            dtype=np.array(str(dtype)),  # caches written before 2026-11-12 lack both: preset dtype, cuda
            device=np.array(device),
        )
        print(f"cached cls {cls_f.shape} + mp {mp_f.shape} + cls4 {cls4_f.shape} to {out}")
        wb.log_features(out)
        outs.append(out)
    if wandb.run is not None:
        wandb.finish()
    return outs


def _tokens_main(args, spec, mean, std) -> list[str]:
    if args.dataset not in F.OFFICIAL_SEG:
        raise SystemExit(
            f"--tokens is for the OFFICIAL_SEG datasets {sorted(F.OFFICIAL_SEG)}, got {args.dataset!r}"
        )
    if args.img % 16 or args.img <= 0:
        raise SystemExit(f"--img must be a positive multiple of 16, got {args.img}")
    device = "cuda"
    model, spec = load_model(args.preset, args.weights, device)
    os.makedirs(args.out_dir, exist_ok=True)
    outs = []
    for split in F.official_seg_spec(args.dataset)["sizes"]:
        paths, masks = F.list_official_seg_split(args.dataset, split)
        if args.max_images:
            keep = np.linspace(0, len(paths) - 1, args.max_images).astype(int)
            paths, masks = [paths[i] for i in keep], [masks[i] for i in keep]
        tag = f"_smoke{args.max_images}" if args.max_images else ""
        base = os.path.join(args.out_dir, f"{args.preset}_{args.dataset}_{split}_tokens{args.img}{tag}")
        if os.path.exists(base + ".npy"):
            raise SystemExit(f"{base}.npy exists -- never overwrite a token cache")
        print(f"{split}: {len(paths)} images at {args.img} px", flush=True)
        grid = extract_tokens(model, spec, paths, device, mean, std, args.img, base + ".npy")
        meta = {
            "task": "extract_dino --tokens",
            "dataset": args.dataset,
            "split": split,
            "n": len(paths),
            "max_images": args.max_images,
            "preset": args.preset,
            "weights": args.weights or spec["weights"],
            "weights_hash": spec["hash"],
            "mean": list(mean),
            "std": list(std),
            "img": args.img,
            "grid_hw": list(grid),
            "feat_dim": int(np.load(base + ".npy", mmap_mode="r").shape[-1]),
            "dtype": "float16",
            "tokens": "last block, norm=True patch tokens (get_intermediate_layers)",
            "file": os.path.basename(base + ".npy"),
            "paths": paths,
            "masks": masks,
            "images_sha1": hashlib.sha1("\n".join(paths).encode()).hexdigest()[:16],
        }
        with open(base + "_meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        print(f"cached tokens {np.load(base + '.npy', mmap_mode='r').shape} to {base}.npy", flush=True)
        outs.append(base + ".npy")
    print("DINO_TOKENS_OK", flush=True)
    return outs


if __name__ == "__main__":
    main()
