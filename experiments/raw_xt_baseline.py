"""Raw x_t baseline: probe the VAE latents directly, with NO DiT forward.

WHY. The one-shot arm noises in closed form, x_t = t*eps + (1-t)*x0 (feat_flux.
Featurizer4Eval.forward). The clean image therefore survives INSIDE the network input as a
literal linear term, so a linear probe on x_t itself can ride (1-t)*x0 for free. Any accuracy
the DiT features achieve is only evidence about the REPRESENTATION to the extent it exceeds
what this costless baseline already gets. Without it, part of the one-shot accuracy-vs-t curve
could be pure protocol artifact.

WHAT IT PREDICTS. eps is drawn per image (tasks/utils.extract_features seeds one generator
OUTSIDE the image loop and draws sequentially), so it is i.i.d. across images rather than a
constant offset a probe bias could absorb. Dividing by (1-t) -- which a linear probe with
StandardScaler is invariant to -- the probe sees x0 corrupted by white noise of relative std

    eta = t / (1 - t)      0.11  0.22  0.35  0.52  0.72  1.00  1.38   at t = 100..580

so raw-x_t accuracy must DECAY monotonically in t, steeply past t=500 where eta > 1. It also
predicts the ensemble effect analytically: averaging M draws scales eta by 1/sqrt(M), so the
ens=8 baseline should sit at the accuracy the ens=1 baseline reaches at eta/2.83.

POOLING. The pooled DiT feature is a spatial mean over the token grid (C=3072). The identical
operation on a 16-channel VAE latent leaves only 16 numbers, which would understate the
baseline for reasons of dimensionality rather than information -- the exact confound that
invalidated the first token-geometry run. So three poolings are written:

    mean  16 dims          protocol-matched: what spatial_mean does to a latent
    q2    64 dims          2x2 quadrant means, the section-2 partition
    full  16*h*w dims      no pooling

MEASURED (2026-08-16) -- `full` is NOT an upper bound, contrary to the obvious expectation.
Spatial mean-pooling AVERAGES eps over h*w latent positions, denoising by ~sqrt(h*w) (~28x at
28x28), while `full` keeps every position's noise and hands 12,544 dims to 500 samples. At
t=100 on EuroSAT: mean 0.666 > q2 0.594 > full 0.478 -- monotonically WORSE with more
dimensions, the opposite of the section-2 confound. The baseline's strength is therefore the
MAX over poolings, and which pooling wins shifts with t.


Cost: VAE encode only (load_ae, no DiT / T5 / CLIP), so this is minutes, not hours.

NOT COVERED. The INVERSION arm's states are not constructible in closed form -- they are the
ODE trajectory and cost a full chain per image. This script covers the one-shot arm only; the
inversion-state analogue is a separate, expensive run.

FINDINGS, updated (RESEARCH_NOTES 6). n=5000: EuroSAT raw 0.737 at t=100 (74% of the DiT
arm's above-chance) with the ens8-ens1 control monotone and significant; RESISC45 0.316
(36%). TERMINOLOGY CORRECTION (audit F3): this baseline uses the TRAINED FLUX VAE encoder --
it is not model-free. Genuinely model-free pooled pixels reach 0.472 on EuroSAT (43%); the
VAE contributes more than the entire transformer stack adds on top of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

import datasets  # noqa: F401,E402  — triggers @register_dataset
from registry import DATASETS  # noqa: E402
from run import DatasetConfig, ModelConfig, RunConfig  # noqa: E402
from tasks.extraction import _stratified_indices  # noqa: E402
from utils import seed_all, seed_worker  # noqa: E402

from models.flux.util import load_ae  # noqa: E402

from torch.utils.data import DataLoader, Subset  # noqa: E402

POOLINGS = ("mean", "q2", "full")


def pool(x: torch.Tensor, how: str) -> torch.Tensor:
    """x: (1, C, h, w) float32 -> (D,)."""
    if how == "mean":
        return x.mean(dim=[2, 3]).flatten()
    if how == "q2":
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten()
    if how == "full":
        return x.flatten()
    raise ValueError(how)


def _env_suffix_and_meta():
    """Provenance stamp for the extraction-control env vars, mirroring
    tasks.extraction.env_provenance: these standalone extractors honor FLUX_RANDOM_INIT
    (via load_flow_model) and DEGRADE_TO (via the dataset hook) but previously wrote
    UN-suffixed cache names -- a stale export would poison the exact filenames the
    downstream probes glob (2026-09-04 review)."""
    from utils import env_value

    parts = []
    if env_value("FLUX_RANDOM_INIT"):
        parts.append("RANDINIT")
    if env_value("FIXED_COND_T"):
        parts.append(f"FIXEDCOND{env_value('FIXED_COND_T')}")
    if env_value("DEGRADE_TO"):
        parts.append(f"DEG{env_value('DEGRADE_TO')}")
    suffix = "".join("_" + p for p in parts)
    meta = {
        "weights": "random_init" if env_value("FLUX_RANDOM_INIT") else "flux-dev",
        "degrade_to": int(env_value("DEGRADE_TO")) if env_value("DEGRADE_TO") else None,
        "fixed_cond_t": int(env_value("FIXED_COND_T")) if env_value("FIXED_COND_T") else None,
    }
    return suffix, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["eurosat", "resisc45"])
    p.add_argument("--path", required=True)
    p.add_argument("--img-size", type=int, nargs=2, required=True)
    p.add_argument("--t", type=int, nargs="+", default=[100, 180, 260, 340, 420, 500, 580])
    p.add_argument("--subset-size", type=int, default=500)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--eps-seed", type=int, default=42)
    p.add_argument("--ensemble-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--max-images", type=int, default=None, help="smoke-test cap")
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
        eps_seed=args.eps_seed,
        batch_size=1,
        num_workers=4,
        label_fraction=1.0,
    )
    seed_all(cfg.seed)

    dataset = DATASETS[cfg.dataset.name](cfg)
    train_ds = dataset.get_data(cfg)["train"].dataset
    all_labels = np.array([class_idx for _, class_idx, _ in train_ds.samples])
    indices = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)
    if args.max_images is not None:
        indices = indices[: args.max_images]
    loader = DataLoader(
        Subset(train_ds, indices.tolist()),
        batch_size=1,
        shuffle=False,  # the eps stream depends on deterministic order
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(f"loading VAE only (no DiT / T5 / CLIP) on {device}")
    ae = load_ae("flux-dev", device=device)

    ts = [t / 1000 for t in args.t]
    # One generator PER ensemble config, each seeded identically, so each config sees the
    # same per-image draw pattern a dedicated run would (draws are consumed sequentially
    # across images inside the loop, exactly as tasks/utils does).
    gens = {m: torch.Generator(device=device).manual_seed(args.eps_seed) for m in args.ensemble_sizes}
    out: dict = {(m, h): [] for m in args.ensemble_sizes for h in POOLINGS}
    labels: list[int] = []

    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to(device)  # 1, 3, H, W
            labels.append(int(batch["label"][0]))
            for m in args.ensemble_sizes:
                acc = None
                for _ in range(m):
                    # ae.encode samples the VAE posterior, so members differ here too --
                    # mirroring the real path, which re-encodes per ensemble member.
                    lat = ae.encode(img).to(torch.bfloat16)
                    eps = torch.randn(lat.shape, generator=gens[m], device=device, dtype=lat.dtype)
                    # x_t for every timestep from ONE (lat, eps) pair — the real path also
                    # reuses both across all K timesteps and asserts they do not change.
                    xt = torch.stack([t * eps + (1.0 - t) * lat for t in ts])  # K,1,C,h,w
                    acc = xt.float() if acc is None else acc + xt.float()
                acc = acc / m
                for how in POOLINGS:
                    out[(m, how)].append(
                        torch.stack([pool(acc[i], how) for i in range(len(ts))]).cpu().numpy()
                    )
            if n % 50 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    y = np.array(labels, dtype=np.int64)
    os.makedirs(args.out_dir, exist_ok=True)
    for (m, how), rows in out.items():
        feats = np.stack(rows).astype(np.float32)  # N, K, D
        base = f"{args.dataset}_rawxt_ens{m}_{how}" + _env_suffix_and_meta()[0]
        np.savez(
            os.path.join(args.out_dir, base + ".npz"),
            feats=feats,
            labels=y,
            timesteps=np.array(args.t),
            subset_indices=indices,
            subset_seed=np.array(cfg.subset_seed),
        )
        meta = {
            **_env_suffix_and_meta()[1],
            "dataset": args.dataset,
            "img_size": list(args.img_size),
            "t": list(args.t),
            "ensemble_size": m,
            "pooling": how,
            "eps_seed": args.eps_seed,
            "subset_size": int(len(indices)),
            "subset_seed": cfg.subset_seed,
            "feats_shape": list(feats.shape),
            "extraction_mode": "RAW_XT_NO_DIT",
            "noising": "x_t = (t/1000)*eps + (1-t/1000)*x0, bf16, eps per image",
        }
        with open(os.path.join(args.out_dir, base + "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {base}.npz  {feats.shape}")


if __name__ == "__main__":
    main()
