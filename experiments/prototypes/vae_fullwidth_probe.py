"""Q1 (9.3): full-width VAE-latent control on the 6ac 5,000 RESISC45 images.

Closes the capacity confound in the C2 source ladder: the 0.32 "VAE latent" rung was a
16-256 d pooled probe vs 3072-18944 d DiT arms. Here the CLEAN latent x0 (posterior mean,
no noise, no DiT) is probed UNPOOLED (32x32x16 = 16,384 d) and at 2x2/4x4/1x1 poolings,
under the section-13 instrument, paired against the cached 6ac arms (identical folds,
identical images). See RESEARCH_NOTES 9.3 Q1 for the pre-registered read.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

from experiments.prototypes._absorption_harness import C as HARNESS_C  # noqa: E402
from experiments.prototypes.dinov2_resisc45 import _fold_plain, paired, run_arm  # noqa: E402
from PIL import Image  # noqa: E402

LAT_CACHE = "results/vae_latents_resisc45_n5000.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=7)
    args = p.parse_args()

    r = np.load("results/dinov2_resisc45_paired.npz", allow_pickle=True)
    paths = [str(x) for x in r["paths"]]
    y = r["labels"]
    if args.max_images:
        keep = np.linspace(0, len(paths) - 1, args.max_images).astype(int)
        paths = [paths[i] for i in keep]
        y = y[keep]

    if os.path.exists(LAT_CACHE) and not args.max_images:
        d = np.load(LAT_CACHE, allow_pickle=True)
        assert list(d["paths"]) == paths, "cached latent paths differ"
        lat = d["lat"]
        print(f"reusing {LAT_CACHE} {lat.shape}")
    else:
        device = torch.device("cuda")
        from models.flux.util import load_ae  # AE only — the 12B transformer is not needed
        ae = load_ae("flux-dev", device=device)
        ae.reg.sample = False  # posterior mean: deterministic, no VAE sampling noise
        ae.eval()
        rows = []
        with torch.no_grad():
            for n, f in enumerate(paths):
                im = Image.open(f).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
                x = (torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0 - 0.5) * 2
                z = ae.encode(x[None].to(device))  # (1,16,32,32)
                rows.append(z.float().cpu().numpy()[0])
                if n % 250 == 0:
                    print(f"  {n}/{len(paths)}", flush=True)
        lat = np.stack(rows).astype(np.float32)
        if not args.max_images:
            np.savez(LAT_CACHE, lat=lat, paths=np.array(paths))
        del ae
        torch.cuda.empty_cache()

    N, C_, H, W = lat.shape
    variants = {
        "full 32x32 (16384d)": lat.reshape(N, -1),
        "pool 4x4 (256d)": lat.reshape(N, C_, 8, 4, 8, 4).mean(axis=(3, 5)).reshape(N, -1),
        "pool 2x2 (64d)": lat.reshape(N, C_, 2, 16, 2, 16).mean(axis=(3, 5)).reshape(N, -1),
        "pool 1x1 (16d)": lat.mean(axis=(2, 3)),
    }

    print("\nVAE-latent arms (section-13 instrument, 3 seeds x 5-fold, C=0.1):")
    A1, B1 = r["A1"], r["B1"]
    if args.max_images:
        A1, B1 = None, None  # cached vectors are full-5000; no pairing at smoke scale
    out = {}
    for name, X in variants.items():
        cv, _, _ = run_arm(f"VAE {name}", y, _fold_plain,
                           (np.ascontiguousarray(X), y, HARNESS_C, None, None), args.n_jobs)
        out[name] = cv
    if A1 is not None:
        print("\nPAIRED deltas (identical folds/images as 6ac):")
        best = max(out, key=lambda k: out[k].mean())
        paired(f"VAE best ({best}) - A1 FLUXinv ", A1, out[best])
        paired(f"VAE best ({best}) - B1 DINOv2  ", B1, out[best])
        np.savez("results/vae_fullwidth_probe_resisc45.npz", labels=y,
                 **{f"vae_{i}": v for i, v in enumerate(out.values())},
                 names=np.array(list(out.keys())))
        print("cached to results/vae_fullwidth_probe_resisc45.npz")


if __name__ == "__main__":
    main()
