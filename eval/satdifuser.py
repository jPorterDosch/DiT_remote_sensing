"""SatDiFuser (Jia et al., ICCV 2025) frozen DiffusionSat features under OUR protocol.

Their Table 2 trains a fuser + linear head jointly on 100 % of the labels (their protocol);
our matrix probes frozen pooled features with a logistic regression, val-selected, at five
label fractions. To put them in the same table their FEATURES have to go through our
protocol, so this module dumps them in our cache format:

  dump     (their conda env: diffusers 0.17, torch 2.2; experiments/isaac/satdifuser/dump.sbatch)
           Their LDMExtractor VERBATIM (imported from the clone): DiffusionSat UNet at 256 px,
           DDIM inversion of the clean VAE latent, features hooked at every up_blocks attn1 and
           conv2 output at their save_timesteps [47, 45, 40] (10 UNet passes per image), the
           prompt "A satellite image", no guidance. What we add is a spatial mean over each
           pyramid scale, exactly the mean their classifier head applies AFTER the fuser
           (archs/classifiers.py AdaptiveAvgPool2d(1)) -- here applied before any trained
           layer. One file per official split, images in eval.features.list_official_split
           order (the DINO caches' convention), from our exported RGB trees, so the split
           identity is byte-identical to every other arm's. The VAE posterior is SAMPLED with a
           seeded generator, as in their run.py (latent_dist.sample(generator=g)); the seed is
           per split (train 42 / test 43 / val 44, the flux caches' decoupling) and recorded.
  select   (DiTF env) the "block" of a SatDiFuser arm is the pyramid scale (s32 / s16 / s8 / s4
           at 256 px, plus `all` = every scale concatenated). Like eval.sweep select for the
           FLUX blocks: every (scale x {t47, t45, t40, concat3t} x C) cell at 100 % labels,
           scored on the official val split, the best scale per dataset printed as the
           matrix's per_dataset expect.scale. Val only (rule 1); test never read here.
  compare  the PNG-quantization check: a dump from the exported 8-bit PNGs against one from
           the raw GEO-Bench hdf5 (their own loader's /4095 float path, --raw-dir) on the same
           images; per-(image, t, scale) cosine must be ~1.

Pins the matrix checks on every cache (eval.probe.official_satdifuser): dataset, weights
(DiffusionSat checkpoint name), img_size, num_timesteps, save_timesteps, diffusion_mode, prompt,
pooling. CLAUDE.md rule 16: nothing here reads a paper table; split sizes are
eval.features.OFFICIAL's counted ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import features as F  # noqa: E402

WEIGHTS = "diffusionsat-sd21-256/checkpoint-150000"  # the `weights` pin of every cache
POOLING = "mean_hw"  # spatial mean of each pyramid scale (their head's AdaptiveAvgPool2d(1))
SCALE_PREFIX = "feats_s"  # npz member per scale: feats_s32 (N, T, C32), ...
SPLIT_SEED = {"train": 42, "test": 43, "val": 44}  # VAE-posterior sampling seed per split
PIN_KEYS = (
    "dataset",
    "weights",
    "img_size",
    "num_timesteps",
    "save_timesteps",
    "diffusion_mode",
    "prompt",
    "pooling",
)


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------ dump helpers
def pool_pyramid(feats: dict, timesteps: list[int]) -> tuple[list[int], list[np.ndarray]]:
    """feats = their LDMExtractor.feats: {save_timestep: [tensor (B, C_w, w, w) per scale, widths
    descending]}. Returns (widths, [array (B, T, C_w) per scale]) with T in `timesteps` order
    (descending save index, the order their fusers iterate: sorted(keys, reverse=True))."""
    import torch

    missing = [t for t in timesteps if t not in feats]
    if missing:
        raise RuntimeError(f"extractor saved no features at save_timesteps {missing}; got {sorted(feats)}")
    widths = None
    per_t = []
    for t in timesteps:
        tensors = feats[t]
        ws = [int(x.shape[-1]) for x in tensors]
        if any(x.ndim != 4 or x.shape[-1] != x.shape[-2] for x in tensors):
            raise RuntimeError(
                f"t={t}: expected square (B, C, w, w) pyramid tensors, got {[tuple(x.shape) for x in tensors]}"
            )
        if ws != sorted(ws, reverse=True) or len(set(ws)) != len(ws):
            raise RuntimeError(f"t={t}: pyramid widths {ws} are not strictly descending")
        if widths is None:
            widths = ws
        elif ws != widths:
            raise RuntimeError(f"pyramid widths differ across timesteps: {widths} vs {ws} at t={t}")
        per_t.append([x.float().mean(dim=(2, 3)) for x in tensors])  # (B, C_w) each
    out = []
    for s in range(len(widths)):
        out.append(
            torch.stack([per_t[i][s] for i in range(len(timesteps))], dim=1).cpu().numpy()
        )  # (B, T, C_w)
    return widths, out


def load_png(path: str) -> np.ndarray:
    """uint8 (H, W, 3) of an exported PNG."""
    from PIL import Image

    px = np.asarray(Image.open(path).convert("RGB"))
    if px.dtype != np.uint8 or px.ndim != 3:
        raise SystemExit(f"{path}: {px.dtype} {px.shape}, expected uint8 RGB")
    return px


def to_model_input(img01: np.ndarray, img_size: int):
    """Their datasets/m_eurosat.py transform on a float [0, 1] (H, W, 3) image: to (3, H, W),
    Resize(img_size, bilinear, antialias=True), Normalize(0.5, 0.5) -> [-1, 1]."""
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    x = torch.from_numpy(np.ascontiguousarray(img01.transpose(2, 0, 1))).float()
    x = TF.resize(x, [img_size, img_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return (x - 0.5) / 0.5


class _PngSplit:
    """Our exported tree -> their input tensor. uint8/255 is the exporter's clip(raw/scale) to
    8 bits; `compare` measures what that quantization does to the features."""

    def __init__(self, files: list[str], img_size: int):
        self.files, self.img_size = files, img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return to_model_input(load_png(self.files[i]).astype(np.float32) / 255.0, self.img_size)


class _RawSplit:
    """Their loader's path on the raw GEO-Bench hdf5 (pack_to_3d(rgb bands) / scale, clip),
    looked up by sample name so the image ORDER is still ours."""

    def __init__(self, task: str, raw_dir: str, split: str, files: list[str], img_size: int):
        import geobench

        from eval.export_geobench import SPECS

        self.spec, self.img_size = SPECS[task], img_size
        gb_split = {"train": "train", "val": "valid", "test": "test"}[split]
        ds = geobench.GeobenchDataset(raw_dir, split=gb_split, partition_name="default")
        self.ds = ds
        by_name = {
            ds[i].sample_name: i for i in range(len(ds))
        }  # loads every sample once; compare runs are small
        want = [os.path.splitext(os.path.basename(f))[0] for f in files]
        missing = [n for n in want if n not in by_name]
        if missing:
            raise SystemExit(
                f"{split}: {len(missing)} exported images not in the raw partition, e.g. {missing[:3]}"
            )
        self.index = [by_name[n] for n in want]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sample = self.ds[self.index[i]]
        image, _ = sample.pack_to_3d(band_names=self.spec.rgb_bands)
        img01 = np.clip(image.astype(np.float32) / self.spec.scale, 0.0, 1.0)
        return to_model_input(img01, self.img_size)


def _git_commit(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def load_their_pipeline(repo: str, model_root: str, device: str):
    """run.py's model construction, verbatim: SatUNet (no metadata) + DiffusionSatPipeline + VAE."""
    import torch
    from diffusers import AutoencoderKL

    if repo not in sys.path:
        sys.path.append(repo)  # APPEND: their top-level `utils` / `datasets` must not shadow ours (src/)
    from diffusionsat import DiffusionSatPipeline, SatUNet  # noqa: E402  (their package)

    unet = SatUNet.from_pretrained(
        model_root,
        subfolder="checkpoint-150000/unet",
        revision=None,
        num_metadata=0,
        use_metadata=False,
        low_cpu_mem_usage=False,
    )
    unet.requires_grad_(False)
    unet.to(device)
    pipe = DiffusionSatPipeline.from_pretrained(model_root, unet=unet)
    pipe.to(device)
    vae = AutoencoderKL.from_pretrained(model_root, subfolder="vae", revision=None)
    vae.requires_grad_(False)
    vae.to(device)
    return pipe, vae, torch


def extractor_config(repo: str, img_size: int, batch_size: int, prompt: str):
    """configs/_base_/satdifuser.yaml (their extraction settings) + the task fields the
    extractor reads (configs/tasks/meurosat.yaml: img_size, prompt, resize_outputs=-1)."""
    from omegaconf import OmegaConf

    base = os.path.join(repo, "configs", "_base_", "satdifuser.yaml")
    cfg = OmegaConf.load(base)
    for k in ("layer_idxs", "save_timesteps", "num_timesteps", "diffusion_mode"):
        if k not in cfg:
            raise SystemExit(f"{base}: no {k!r} -- their extraction settings moved; refusing to guess")
    if cfg.diffusion_mode != "inversion":
        raise SystemExit(f"{base}: diffusion_mode={cfg.diffusion_mode!r}, the paper's protocol is inversion")
    cfg.img_size = img_size
    cfg.batch_size = batch_size
    cfg.prompt = prompt
    cfg.resize_outputs = -1  # pyramid output, as every classification task config
    return cfg, file_sha1(base)


def dump(args) -> list[str]:
    import torch
    from omegaconf import OmegaConf

    if args.repo not in sys.path:
        sys.path.append(args.repo)  # APPEND: see load_their_pipeline
    from archs.ldm_extractor import LDMExtractor  # noqa: E402  (their extractor, untouched)

    dataset = args.dataset
    spec = F.official_spec(dataset)
    task = dataset.replace("_", "-")
    device = "cuda"
    cfg, cfg_sha = extractor_config(args.repo, args.img_size, args.batch_size, args.prompt)
    save_ts = sorted(int(t) for t in cfg.save_timesteps)[::-1]  # descending: their fusers' order
    pipe, vae, _ = load_their_pipeline(args.repo, args.model_root, device)
    ext = LDMExtractor(cfg, pipe)
    seq = torch.flip(ext.scheduler.timesteps, dims=(0,)).tolist()  # ascending, as generalized_steps
    seq_next = [-1] + seq[:-1]
    # inversion step i evaluates the UNet at t = seq_next[i] and saves at cur_t = len(seq) - i - 1
    t_ddpm = [int(seq_next[len(seq) - c - 1]) for c in save_ts]
    unet_passes = int(ext.max_i)
    print(f"[satdifuser] save_timesteps {save_ts} = DDPM t {t_ddpm}; {unet_passes} UNet passes per image; "
          f"layers {len(ext.collected_layers)}; dims {dict(ext.collected_dims)}", flush=True)  # fmt: skip

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "_smoke" if args.max_images else ""
    if args.raw_dir:
        tag += "_raw"
    outs = []
    for split in spec["sizes"]:
        if args.splits and split not in args.splits:
            continue
        files, y = F.list_official_split(dataset, split, args.data_root)
        if args.max_images:
            keep = np.linspace(0, len(files) - 1, args.max_images).astype(int)
            files, y = [files[i] for i in keep], y[keep]
        ds = (
            _RawSplit(task, args.raw_dir, split, files, args.img_size)
            if args.raw_dir
            else _PngSplit(files, args.img_size)
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        g = torch.Generator(device=device).manual_seed(SPLIT_SEED[split])
        widths, chunks, n_done, t0 = None, [], 0, time.time()
        layer_widths = None
        for images in loader:
            images = images.to(device)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample(generator=g) * 0.18215
                with torch.inference_mode():
                    feats, _ = ext.forward(latents)
            w, pooled = pool_pyramid(feats, save_ts)
            if widths is None:
                widths = w
                # attn outputs are (B, L, C) with L = w*w; conv outputs (B, C, w, w) (their collect_feats)
                layer_widths = [
                    int(round(m.feats.shape[1] ** 0.5)) if m.feats.ndim == 3 else int(m.feats.shape[-1])
                    for m in ext.collected_layers
                ]
            elif w != widths:
                raise RuntimeError(f"pyramid widths changed mid-split: {widths} -> {w}")
            chunks.append(pooled)
            n_done += images.shape[0]
            if len(chunks) % 20 == 0:
                print(f"  {split} {n_done}/{len(ds)} ({time.time() - t0:.0f}s)", flush=True)
        if n_done != len(files):
            raise RuntimeError(f"{split}: extracted {n_done} of {len(files)} images")
        arrays = {
            f"{SCALE_PREFIX}{w}": np.concatenate([c[s] for c in chunks]).astype(np.float16)
            for s, w in enumerate(widths)
        }
        dims = {f"s{w}": int(a.shape[-1]) for w, a in zip(widths, arrays.values())}
        out = os.path.join(args.out_dir, f"satdifuser_{dataset}_{split}{tag}.npz")
        np.savez(
            out,
            **arrays,
            timesteps=np.array(save_ts),
            t_ddpm=np.array(t_ddpm),
            scales=np.array(widths),
            paths=np.array(files),
            labels=y,
            weights=np.array(WEIGHTS),
            pooling=np.array(POOLING),
        )
        # per-scale channel layout: the collected layers at that width, in their collection order
        layers_by_scale = {
            f"s{w}": [n for n, lw in zip(ext.collect_layer_names, layer_widths) if lw == w] for w in widths
        }
        meta = {
            "dataset": dataset,
            "split": split,
            "n": int(n_done),
            "weights": WEIGHTS,
            "model_root": os.path.abspath(args.model_root),
            "unet_config_sha1": file_sha1(
                os.path.join(args.model_root, "checkpoint-150000", "unet", "config.json")
            ),
            "repo": os.path.abspath(args.repo),
            "repo_commit": _git_commit(args.repo),
            "extractor_yaml_sha1": cfg_sha,
            "img_size": args.img_size,
            "prompt": args.prompt,
            "num_timesteps": int(cfg.num_timesteps),
            "save_timesteps": save_ts,
            "t_ddpm": t_ddpm,
            "unet_passes_per_image": unet_passes,
            "diffusion_mode": str(cfg.diffusion_mode),
            "layer_idxs": OmegaConf.to_container(cfg.layer_idxs),
            "layers_by_scale": layers_by_scale,
            "dims": dims,
            "scales": widths,
            "pooling": POOLING,
            "vae_scaling": 0.18215,
            "vae_sample": "latent_dist.sample(generator)",
            "vae_seed": SPLIT_SEED[split],
            "autocast": "cuda fp16 (their LDMExtractor.forward)",
            "feats_dtype": "float16",
            "source": "raw geobench hdf5 (their loader path)"
            if args.raw_dir
            else "exported RGB PNG tree (uint8/255)",
            "data_root": os.path.abspath(args.data_root or spec["root"]),
            "max_images": args.max_images,
            "batch_size": args.batch_size,
            "elapsed_s": round(time.time() - t0, 1),
            "images_sha1": hashlib.sha1("\n".join(files).encode()).hexdigest()[:16],
        }
        with open(out.replace(".npz", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(
            f"[satdifuser] {split}: {n_done} images, dims {dims}, {meta['elapsed_s']}s -> {out}", flush=True
        )
        outs.append(out)
    print("SATDIFUSER_DUMP_OK", flush=True)
    return outs


# ------------------------------------------------------------------------- loading (DiTF env)
def load_split(path: str) -> tuple[dict, dict]:
    d = F.load(path)
    meta = F.cache_meta(path)
    if not meta:
        raise SystemExit(f"{path}: no _meta.json beside the cache -- cannot verify its identity, refusing")
    for k in ("timesteps", "scales", "paths", "labels", "weights"):
        if k not in d.files:
            raise SystemExit(f"{path}: no {k!r} member")
    return d, meta


def scale_names(d) -> list[str]:
    return [f"s{int(w)}" for w in d["scales"]]


def candidates(d, scale: str, ts: list[int]) -> dict[str, np.ndarray]:
    """{t47: (N, C), t45, t40, concat3t: (N, 3C)} at one scale; `all` = scales concatenated."""
    names = scale_names(d)
    if scale == "all":
        X = np.concatenate([d[f"{SCALE_PREFIX}{int(w)}"] for w in d["scales"]], axis=-1)
    elif scale in names:
        X = d[f"{SCALE_PREFIX}{scale[1:]}"]
    else:
        raise SystemExit(f"scale {scale!r} not among {names + ['all']}")
    X = X.astype(np.float64)
    if X.ndim != 3 or X.shape[1] != len(ts):
        raise SystemExit(f"feats {X.shape} do not match (N, T={len(ts)}, C)")
    out = {f"t{t}": np.ascontiguousarray(X[:, i, :]) for i, t in enumerate(ts)}
    out[f"concat{len(ts)}t"] = np.ascontiguousarray(X.reshape(len(X), -1))
    return out


def select(args) -> dict:
    """Best (scale, candidate, C) on the official val split at 100 % labels, per dataset."""
    from threadpoolctl import threadpool_limits

    from eval import protocols as P
    from eval.sweep import SWEEP_FIT_THREADS

    c_grid = tuple(float(c) for c in args.c_grid)
    results = {}
    for dataset in args.datasets:
        splits = {}
        for s in ("train", "val"):  # selection is val-only: the test caches are not opened here
            path = args.features.replace("{dataset}", dataset).format(split=s)
            d, meta = load_split(path)
            F.check_pins(meta, {"dataset": dataset, "split": s, "weights": WEIGHTS, "pooling": POOLING}, path)
            if meta.get("max_images"):
                raise SystemExit(f"{path}: a --max-images smoke cache, never quotable")
            files, y = F.list_official_split(dataset, s)
            F.check_identity(d, files, path)
            if not np.array_equal(d["labels"], y):
                raise SystemExit(f"{path}: labels differ from the official split's")
            splits[s] = (d, meta, path)
        metas = [m for _, m, _ in splits.values()]
        for k in PIN_KEYS:
            if len({json.dumps(m.get(k)) for m in metas}) != 1:
                raise SystemExit(f"{dataset}: caches disagree on {k}: {[m.get(k) for m in metas]}")
        if len({m.get("vae_seed") for m in metas}) != len(metas):
            raise SystemExit(
                f"{dataset}: the splits must have distinct vae_seed, got {[m.get('vae_seed') for m in metas]}"
            )
        ts = [int(t) for t in splits["train"][0]["timesteps"]]
        scales = scale_names(splits["train"][0]) + ["all"]
        ytr, yva = splits["train"][0]["labels"], splits["val"][0]["labels"]
        rows, best = [], None
        for scale in scales:
            c = {s: candidates(splits[s][0], scale, ts) for s in ("train", "val")}
            # official_all_cells also predicts a "test" slot; val goes there, so no test row is
            # ever read by selection, and those predictions are discarded
            cands = {n: (c["train"][n], c["val"][n], c["val"][n]) for n in c["train"]}
            with threadpool_limits(limits=SWEEP_FIT_THREADS, user_api="blas"):
                table, _ = P.official_all_cells(cands, ytr, yva, yva, c_grid)
            for r in table:
                r = {"scale": scale, **r}
                rows.append(r)
                if best is None or r["val_acc"] > best["val_acc"]:
                    best = r
            print(
                f"{dataset} {scale}: best val {max(r['val_acc'] for r in table):.4f}", flush=True
            )  # table = this scale only
        results[dataset] = {
            "selected": best,
            "rows": rows,
            "timesteps": ts,
            "scales": scales,
            "c_grid": list(c_grid),
        }
        print(
            f"{dataset}: SELECT scale={best['scale']} candidate={best['candidate']} C={best['C']} val={best['val_acc']:.4f}",
            flush=True,
        )
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "select.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("per_dataset for eval/matrix_specs (kind satdifuser):")
    print(
        json.dumps({d: {"expect": {"scale": r["selected"]["scale"]}} for d, r in results.items()}, indent=1)
    )
    print(f"-> {out}")
    return results


def compare(args) -> float:
    """Per-(image, t, scale) cosine between two dumps of the same images (PNG vs raw)."""
    a, ma = load_split(args.a)
    b, mb = load_split(args.b)
    if [str(x) for x in a["paths"]] != [str(x) for x in b["paths"]]:
        raise SystemExit("the two caches hold different images")
    worst, lines = 1.0, []
    for w in a["scales"]:
        x, y = (
            a[f"{SCALE_PREFIX}{int(w)}"].astype(np.float64),
            b[f"{SCALE_PREFIX}{int(w)}"].astype(np.float64),
        )
        cos = (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-12)
        rel = np.abs(x - y).sum(-1) / (np.abs(y).sum(-1) + 1e-12)
        worst = min(worst, float(cos.min()))
        lines.append(
            f"s{int(w)}: cosine min {cos.min():.5f} mean {cos.mean():.5f}; rel-L1 max {rel.max():.4f}"
        )
    print("\n".join(lines))
    print(f"sources: {ma.get('source')} vs {mb.get('source')}")
    verdict = "PASS" if worst >= args.min_cosine else "FAIL"
    print(f"COMPARE {verdict}: min cosine {worst:.5f} (threshold {args.min_cosine})")
    if verdict == "FAIL":
        raise SystemExit(1)
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="their extractor -> our pooled caches (their conda env, GPU)")
    d.add_argument("--repo", required=True, help="SatDiFuser clone (archs/, diffusionsat/, configs/)")
    d.add_argument(
        "--model-root",
        required=True,
        help="DiffusionSat resolution256 dir (model_index.json, checkpoint-150000/unet, vae)",
    )
    d.add_argument("--dataset", required=True, choices=sorted(F.OFFICIAL))
    d.add_argument("--out-dir", required=True)
    d.add_argument("--data-root", default=None, help="exported RGB tree (default: OFFICIAL root)")
    d.add_argument(
        "--raw-dir",
        default=None,
        help="raw GEO-Bench task dir: their loader path instead of the PNGs (compare)",
    )
    d.add_argument("--splits", nargs="*", default=None)
    d.add_argument("--img-size", type=int, default=256)
    d.add_argument("--prompt", default="A satellite image")
    d.add_argument("--batch-size", type=int, default=32)
    d.add_argument("--num-workers", type=int, default=4)
    d.add_argument("--max-images", type=int, default=None, help="SMOKE / compare: strided cap per split")
    s = sub.add_parser("select", help="best scale per dataset on val at 100 %% labels (DiTF env)")
    s.add_argument("--features", required=True, help="pattern with {dataset} and {split}")
    s.add_argument("--datasets", nargs="+", required=True)
    s.add_argument("--out-dir", default="results/satdifuser_select")
    s.add_argument(
        "--c-grid", nargs="+", default=["1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1", "10", "100"]
    )
    c = sub.add_parser("compare", help="cosine between two dumps of the same images")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--min-cosine", type=float, default=0.99)
    args = p.parse_args(argv)
    return {"dump": dump, "select": select, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
