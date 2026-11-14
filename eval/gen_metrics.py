"""Distributional generation metrics for the adaptation arms (spec s8): FID and KID between a set
of generated images and a FIXED real-image reference set, on Inception-v3 pool features.

Feature extractor: torchvision's Inception-v3 (IMAGENET1K_V1 weights, fc removed -> 2048-d
pool features), inputs resized to 299 x 299 (bilinear, antialiased) and ImageNet-normalized.
These are NOT the TF-FID Inception weights (pytorch-fid / torch-fidelity), so the absolute values
are comparable WITHIN this project (same extractor, same reference set, same n) and not to
published FID tables. KID (Binkowski et al. 2018: unbiased MMD^2 with the cubic polynomial
kernel, averaged over random subsets) is reported next to FID because FID is biased upward at
small n; with n ~ 1,000 read KID first and FID as the familiar number.

Reference set (tasks/train_diffusion.py): the first `fid_n` strided rows of the run's validation
source -- for git10m the retained holdout shared by every index, unflipped, decoded at the
training resolution -- so every arm scores against the identical images; the reference identity
(row count, image size, sha1 of the pixel bytes) is written next to the numbers and must agree
across arms. Generated set: `fid_n` fixed-seed samples (seed gen_seed + batch index) with the
same sampler settings as the fixed-seed grids (align.gen_steps, cfg.guidance_scale, null
prompt).

CLI (recompute from PNG folders, e.g. two runs' gen dirs):
    python -m eval.gen_metrics --real <dir-of-pngs> --fake <dir-of-pngs> [--n N]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTRACTOR = "torchvision/inception_v3/IMAGENET1K_V1/pool2048/299bilinear"


class InceptionFeatures:
    """Inception-v3 pool features (N, 2048) of images given as float NCHW in [-1, 1] (any size)
    or uint8 NHWC. `net` may be injected (tests) and must map (N, 3, 299, 299) ImageNet-normalized
    floats to (N, D)."""

    def __init__(self, device, net: torch.nn.Module | None = None, batch: int = 64):
        self.device = torch.device(device)
        if net is None:
            import torchvision

            w = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            net = torchvision.models.inception_v3(weights=w, aux_logits=True)
            net.fc = torch.nn.Identity()
            self.name = EXTRACTOR
        else:
            self.name = f"injected/{type(net).__name__}"
        self.net = net.to(self.device).eval()
        self.batch = batch
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def to_float01(x) -> torch.Tensor:
        """uint8 NHWC (numpy or tensor) or float NCHW in [-1, 1] -> float NCHW in [0, 1]."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dtype == torch.uint8:
            if x.ndim != 4 or x.shape[-1] != 3:
                raise ValueError(f"uint8 images must be NHWC with 3 channels, got {tuple(x.shape)}")
            return x.permute(0, 3, 1, 2).float() / 255.0
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"float images must be NCHW with 3 channels, got {tuple(x.shape)}")
        lo, hi = float(x.min()), float(x.max())
        if lo < -1.001 or hi > 1.001:
            raise ValueError(f"float images must be in [-1, 1], got [{lo:.3f}, {hi:.3f}]")
        return (x.float() + 1) / 2

    @torch.no_grad()
    def __call__(self, imgs) -> np.ndarray:
        x = self.to_float01(imgs)
        out = []
        for i in range(0, x.shape[0], self.batch):
            b = x[i : i + self.batch].to(self.device)
            b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            b = (b - self.mean) / self.std
            f = self.net(b)
            if isinstance(
                f, (tuple, list)
            ):  # torchvision InceptionOutputs in train mode; eval returns a tensor
                f = f[0]
            out.append(f.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)


def fid(fa: np.ndarray, fb: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (fp64; the standard
    sqrtm with the imaginary-part guard of Heusel et al.'s reference code)."""
    from scipy import linalg

    fa, fb = np.asarray(fa, np.float64), np.asarray(fb, np.float64)
    if fa.ndim != 2 or fb.ndim != 2 or fa.shape[1] != fb.shape[1]:
        raise ValueError(f"feature sets must be (n, d) with equal d, got {fa.shape} and {fb.shape}")
    if min(len(fa), len(fb)) < 2:
        raise ValueError("FID needs at least 2 samples per set")
    mu1, mu2 = fa.mean(0), fb.mean(0)
    s1, s2 = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"sqrtm imaginary component {np.max(np.abs(covmean.imag)):.3e}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid(fa: np.ndarray, fb: np.ndarray, subsets: int = 100, subset_size: int | None = None, seed: int = 0):
    """Kernel Inception Distance: unbiased MMD^2 with k(x, y) = (x.y / d + 1)^3, averaged over
    `subsets` random subsets of `subset_size` (default min(1000, n // 2): a subset equal to the
    whole set would make every draw identical and the std meaningless). Returns (mean, std)."""
    fa, fb = np.asarray(fa, np.float64), np.asarray(fb, np.float64)
    if fa.ndim != 2 or fb.ndim != 2 or fa.shape[1] != fb.shape[1]:
        raise ValueError(f"feature sets must be (n, d) with equal d, got {fa.shape} and {fb.shape}")
    n = min(len(fa), len(fb))
    m = min(subset_size or min(1000, max(2, n // 2)), n)
    if m < 2:
        raise ValueError("KID needs at least 2 samples per set")
    d = fa.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(subsets):
        x = fa[rng.choice(len(fa), m, replace=False)]
        y = fb[rng.choice(len(fb), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        vals.append((kxx.sum() + kyy.sum()) / (m * (m - 1)) - 2 * kxy.mean())
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std())


def image_identity(imgs_uint8: np.ndarray) -> dict:
    """What fixes a reference set: count, size, sha1 of the pixel bytes in order."""
    a = np.ascontiguousarray(imgs_uint8)
    if a.dtype != np.uint8 or a.ndim != 4:
        raise ValueError(f"expected uint8 NHWC, got {a.dtype} {a.shape}")
    return {
        "n": int(a.shape[0]),
        "hw": [int(a.shape[1]), int(a.shape[2])],
        "sha1": hashlib.sha1(a.tobytes()).hexdigest(),
    }


def compare(
    real_uint8: np.ndarray, fake_uint8: np.ndarray, extractor: InceptionFeatures, subsets: int = 100
) -> dict:
    """FID + KID of fake vs real (uint8 NHWC each) with the sets' identities. Refuses unequal n:
    FID's small-sample bias depends on n, so arms are only comparable at one n."""
    if real_uint8.shape[0] != fake_uint8.shape[0]:
        raise ValueError(
            f"real ({real_uint8.shape[0]}) and fake ({fake_uint8.shape[0]}) must have the same n"
        )
    fr, ff = extractor(real_uint8), extractor(fake_uint8)
    km, ks = kid(fr, ff, subsets=subsets)
    return {
        "n": int(real_uint8.shape[0]),
        "fid": fid(fr, ff),
        "kid_mean": km,
        "kid_std": ks,
        "kid_subsets": int(subsets),
        "extractor": extractor.name,
        "real": image_identity(real_uint8),
        "fake": image_identity(fake_uint8),
    }


def _load_pngs(d: str, n: int | None) -> np.ndarray:
    from PIL import Image

    files = sorted(glob.glob(os.path.join(d, "*.png")))
    files = [f for f in files if not f.endswith("_grid.png")]
    if n is not None:
        files = files[:n]
    if not files:
        raise SystemExit(f"no PNGs in {d}")
    arrs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    if len({a.shape for a in arrs}) != 1:
        raise SystemExit(f"{d}: PNGs differ in size: {sorted({a.shape for a in arrs})}")
    return np.stack(arrs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="directory of real PNGs (the reference set)")
    ap.add_argument("--fake", required=True, help="directory of generated PNGs")
    ap.add_argument("--n", type=int, default=None, help="use the first n of each (sorted by name)")
    ap.add_argument("--subsets", type=int, default=100)
    ap.add_argument("--out", default=None, help="write the JSON here as well")
    a = ap.parse_args(argv)
    real, fake = _load_pngs(a.real, a.n), _load_pngs(a.fake, a.n)
    if len(real) != len(fake):  # FID is biased in n: never score a silently shortened set
        raise SystemExit(
            f"--real has {len(real)} PNGs and --fake {len(fake)}"
            + (f" (asked for --n {a.n})" if a.n is not None else "; pass --n to score the first n of each")
        )
    n = len(real)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = compare(real[:n], fake[:n], InceptionFeatures(dev), subsets=a.subsets)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
