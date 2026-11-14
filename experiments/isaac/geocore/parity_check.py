"""GPU parity gate for the GeoCore adapter (smoke.sbatch step 3).

Loads GeoCore-9B through the adapter, takes image 0 of a smoke cache's subset, rebuilds z_t for
ensemble member 0 at the cache's first timestep, and checks the adapter's k=8 feature (the input
of the first single block) against the double-block-7 OUTPUT captured by wrapping
`double_blocks[7].forward_kv_extract` during a plain vendored `Flux2.forward` on the same z_t.
Also checks that the pooled feature in the cache equals the mean over the ensemble members
re-derived with the cache's eps_seed stream. Prints GEOCORE_PARITY_OK.

    python experiments/isaac/geocore/parity_check.py --cache <train cache .npz> --geocore-ckpt ... --geocore-vae ... --dataset-path ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--geocore-ckpt", required=True)
    ap.add_argument("--geocore-vae", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--atol", type=float, default=2e-2)
    args = ap.parse_args()

    import models  # noqa: F401  (registers adapters)
    from models.geocore.adapter import GeoCoreModel, _latent_ids
    from PIL import Image

    meta = json.load(open(args.cache[: -len(".npz")] + "_meta.json"))
    d = np.load(args.cache)
    ks, ts = list(meta["k"]), list(meta["t"])
    assert 8 in ks and meta["layout"] == "N,K,B,C", (ks, meta.get("layout"))
    cfg = types.SimpleNamespace(
        lora_checkpoint="", extraction_mode="ONESHOT", guidance_scale=0.0, img_size=[args.img_size] * 2, k=ks,
        geocore_ckpt=args.geocore_ckpt, geocore_vae=args.geocore_vae, device="cuda",
    )  # fmt: skip
    # Mirror run.py: seed_all also sets cudnn.deterministic=True / benchmark=False, which pins
    # the (bf16) VAE conv algorithms. Without it cuDNN may pick another algorithm and z0 differs
    # by rounding, which the 32-block trunk amplifies (smoke 6532226: 0.07 at k=4 -> 3.8 at k=31).
    from utils import seed_all

    seed_all(int(meta["seed"]))
    m = GeoCoreModel(cfg, ["x"])
    dev = m.device

    # Image 0 of the subset through the dataset's own transform, EXPRESSION-IDENTICAL to
    # data/eurosat_dataset.EuroSATDataset.__getitem__ (bicubic to img_size, then
    # (x / 255 - 0.5) * 2): a mathematically equal but differently rounded float32 input
    # ((x / 255) * 2 - 1) costs ~1 bf16 ulp on the massive channels (3.8 on |x| ~ 1000, smoke
    # 6531445), which is the whole budget of this gate.
    from torchvision.transforms import PILToTensor

    path = str(d["paths"][0])
    im = Image.open(path).convert("RGB").resize((args.img_size, args.img_size), Image.Resampling.BICUBIC)
    img = (PILToTensor()(im) / 255.0 - 0.5) * 2

    ens = int(meta["ensemble_size"])
    g = torch.Generator(device=dev).manual_seed(int(meta["eps_seed"]))
    t0 = ts[0]
    feat_raw, ada, lat, eps = m.extract_raw(img, timestep=t0, block_idx=ks, ensemble_size=ens, generator=g)
    # Pool EXACTLY as tasks/utils.extract_features does: the spatial mean runs in the feature's
    # own dtype (bf16 here) and is cast to float32 only afterwards. A float32 mean differs by the
    # bf16 rounding of the result: 3.796 on channels of |x| ~ 1e3 at k=31, 0.066 at k=4 (smokes
    # 6531445 / 6532226 / 6532381, identical to the last digit, within-process repeat exactly 0).
    pooled = feat_raw.mean(dim=[2, 3]).float().cpu().numpy()  # (B, C)
    cached = d["feats"][0, 0]  # (B, C) image 0, timestep 0
    # Within-process repeatability on the SAME latents/eps: nonzero here means kernel
    # nondeterminism, which no reconstruction can beat and must be reported, not tuned away.
    feat_rep, _, _, _ = m.extract_raw(
        img, timestep=t0, block_idx=ks, ensemble_size=ens, latents=lat, noise=eps
    )
    err_rep = float(np.abs(feat_rep.mean(dim=[2, 3]).float().cpu().numpy() - pooled).max())
    print(f"within-process repeat (same latents/eps) max |diff| = {err_rep:.3e}")
    err_pool = float(np.abs(pooled - cached).max())
    scale = float(np.abs(cached).max())
    print(f"pooled-feature max |adapter - cache| = {err_pool:.3e} (image 0, t={t0}; max |cache| = {scale:.3e}, "
          f"per-block max |diff| = {np.abs(pooled - cached).max(1).round(3).tolist()})")  # fmt: skip

    # Member 0 again, plain vendored forward with the method wrapped.
    assert 8 in ks, f"block 8 not among the extracted blocks {ks}"
    f_member0, _, _, _ = m.extract_raw(
        img, timestep=t0, block_idx=[8], ensemble_size=1, latents=lat[:1], noise=eps[:1]
    )
    t = t0 / 1000.0
    zt = (t * eps[:1] + (1 - t) * lat[:1]).to(m.dtype)
    _, _, h, w = zt.shape
    captured = {}
    blk = m.model.double_blocks[7]
    orig = blk.forward_kv_extract

    def wrapped(*a, **k):
        out = orig(*a, **k)
        captured["img_out"] = out[0].clone()
        return out

    blk.forward_kv_extract = wrapped
    null = torch.full((1,), -999.0, device=dev, dtype=m.dtype)
    with torch.no_grad():
        m.model.forward(zt, _latent_ids(h, w, 1, dev, m.dtype), torch.full((1,), t, device=dev, dtype=m.dtype),
                        m.ctx, m.ctx_ids, m.y, null, null, null)  # fmt: skip
    blk.forward_kv_extract = orig
    ref = captured["img_out"].float().reshape(1, h, w, -1).permute(0, 3, 1, 2)
    err_tap = float((f_member0.float() - ref).abs().max())
    print(f"k=8 block-input vs double_blocks[7] output max |diff| = {err_tap:.3e} (bf16 trunk)")
    ok = err_tap <= args.atol and err_pool <= args.atol
    print("GEOCORE_PARITY_OK" if ok else f"GEOCORE_PARITY_FAIL atol={args.atol}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
