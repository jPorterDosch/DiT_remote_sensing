# ruff: noqa: E402  — sys.path and FLUX_DEV/AE env must be set before flux imports
"""Round-trip validation of the RF-Solver inversion chain (extraction_mode=inversion).

For a handful of EuroSAT images: invert the clean image to its t=1 latent along the
reverse generative ODE, regenerate from that latent with the same solver/conditioning,
and report per-image reconstruction error. The VAE encode->decode round trip (no
inversion) is reported alongside as the error floor — the inversion chain cannot beat
the autoencoder itself.

Feature caching at --timesteps is exercised during inversion (same code path as
extraction), and cached shapes are asserted, so a passing run validates the extractor
plumbing as well as the ODE round trip.

Usage (defaults reproduce the REPORT.md numbers):
    python validate_inversion.py
    python validate_inversion.py --order 1              # naive Euler ablation
    python validate_inversion.py --guidance 3.5         # extraction-default guidance
    python validate_inversion.py --num-inversion-steps 100
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))  # flux.* internal imports

# Local FLUX weights fallback: util.py resolves FLUX_DEV/AE env vars at import time,
# defaulting to ISAAC paths that don't exist on workstations. Point them at the local
# copy when present so the script runs on both machines unchanged.
_local_flux = os.path.join(_root, "ditf_models", "FLUX.1-dev")
# not os.environ.get(...) — treat an empty (or unrelated-empty) var as unset, matching util.py.
if not os.environ.get("FLUX_DEV") and os.path.exists(os.path.join(_local_flux, "flux1-dev.safetensors")):
    os.environ["FLUX_DEV"] = os.path.join(_local_flux, "flux1-dev.safetensors")
if not os.environ.get("AE") and os.path.exists(os.path.join(_local_flux, "ae.safetensors")):
    os.environ["AE"] = os.path.join(_local_flux, "ae.safetensors")

import torch
import tyro

from data.eurosat_dataset import EUROSAT_CLASSES, EuroSATDataset
from flux.feat_flux import Featurizer4Eval
from flux.sampling import unpack


@dataclass
class ValidationConfig:
    # Reference block for the timestep-isolation sweep — override per experiment
    # (e.g. 29 is EuroSAT-optimal). Exercised here only to validate feature caching.
    block_idx: int = 28
    # Nominal timesteps ([1, 1000]) at which block features are cached along the chain.
    # Must lie on the num_inversion_steps integration grid.
    timesteps: list[int] = field(default_factory=lambda: [100, 180, 260, 340, 420, 500, 580])
    # Integration granularity of the reverse ODE — how finely the chain is discretized.
    # A separate knob from --timesteps (which are a subsample of this grid).
    num_inversion_steps: int = 50
    # Guidance strength for every velocity evaluation (same value is used for the
    # inversion and the regeneration directions). The one-shot extractor default is 3.5.
    guidance: float = 1.0
    # 2 = RF-Solver second-order step (default), 1 = naive Euler (ablation baseline).
    order: int = 2
    # Number of validation images (one per EuroSAT class, deterministic order).
    num_images: int = 10
    dataset_path: str = "data/eurosat/EuroSAT_RGB"
    img_size: int = 224
    # Where to save original|reconstruction side-by-side images (empty = don't save).
    out_dir: str = "inversion_validation"
    seed: int = 42


def pick_one_per_class(ds: EuroSATDataset, n: int) -> list[int]:
    """Deterministic: the first test-split sample of each of the first n classes."""
    chosen: dict[int, int] = {}
    for idx, (_, class_idx, _) in enumerate(ds.samples):
        if class_idx not in chosen:
            chosen[class_idx] = idx
        if len(chosen) == len(EUROSAT_CLASSES):
            break
    return [chosen[c] for c in sorted(chosen)][:n]


@torch.no_grad()
def main(cfg: ValidationConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")

    ds = EuroSATDataset(cfg.dataset_path, split="test", img_size=cfg.img_size)
    indices = pick_one_per_class(ds, cfg.num_images)
    print(f"validating on {len(indices)} images (one per class), img_size={ds.img_size}")
    print(
        f"solver order={cfg.order} ({'RF-Solver 2nd-order' if cfg.order == 2 else 'naive Euler'}), "
        f"num_inversion_steps={cfg.num_inversion_steps}, guidance={cfg.guidance}, "
        f"block_idx={cfg.block_idx}, cache timesteps={cfg.timesteps}"
    )

    feat = Featurizer4Eval(cat_list=[])

    try:
        import lpips as lpips_lib

        lpips_model = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    except ImportError:
        lpips_model = None
        print("lpips not installed — reporting pixel MSE/PSNR only")

    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)

    K = len(cfg.timesteps)
    rows = []
    feat_shape_checked = False
    header = f"{'class':<22} {'vae_floor_mse':>13} {'inv_mse':>10} {'psnr_db':>8}" + (
        f" {'lpips':>7}" if lpips_model else ""
    )
    print("\n" + header)
    print("-" * len(header))

    for idx in indices:
        sample = ds[idx]
        img = sample["img"].to(device)  # C, H, W in [-1, 1]
        class_name = sample["class_name"]

        # --- VAE-only round trip: the reconstruction error floor.
        z0 = feat.ae.encode(img.unsqueeze(0))
        vae_recon = feat.ae.decode(z0.float()).clamp(-1, 1)
        vae_mse = ((vae_recon - img.unsqueeze(0)) / 2).pow(2).mean().item()  # [0,1] scale

        # --- Invert to t=1, caching features along the way (exercises extraction path).
        out = feat.invert_chain(
            img,
            cache_timesteps=cfg.timesteps,
            num_inversion_steps=cfg.num_inversion_steps,
            block_idx=cfg.block_idx,
            guidance=cfg.guidance,
            order=cfg.order,
            t_stop=1000,
        )
        if not feat_shape_checked:
            feats_gap = torch.cat([out["feats"][t].mean(dim=[2, 3]) for t in cfg.timesteps], dim=0)  # K, C
            if feats_gap.shape != (K, 3072):
                raise RuntimeError(f"feature shape {tuple(feats_gap.shape)} != ({K}, 3072)")
            mods0 = out["mods"][cfg.timesteps[0]]
            if mods0.shape[1:] != (3, 3072):
                raise RuntimeError(f"mod shape {tuple(mods0.shape)}")
            print(f"[feature check] cached (K={K}, 3072) pre-norm features + (3, 3072) mods OK")
            feat_shape_checked = True

        # --- Regenerate from the inverted latent with the same solver settings.
        z_rec = feat.generate_chain(
            out["z_final"],
            out["img_ids"],
            t_start=1000,
            num_inversion_steps=cfg.num_inversion_steps,
            guidance=cfg.guidance,
            order=cfg.order,
        )
        latents_rec = unpack(z_rec, cfg.img_size, cfg.img_size)
        recon = feat.ae.decode(latents_rec.float()).clamp(-1, 1)

        mse = ((recon - img.unsqueeze(0)) / 2).pow(2).mean().item()  # [0,1] scale
        psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
        row = {"class": class_name, "vae_mse": vae_mse, "mse": mse, "psnr": psnr}

        if lpips_model is not None:
            row["lpips"] = lpips_model(
                recon.float(), img.unsqueeze(0).float(), normalize=False
            ).item()  # inputs already in [-1, 1]

        rows.append(row)
        line = f"{class_name:<22} {vae_mse:>13.5f} {mse:>10.5f} {psnr:>8.2f}"
        if lpips_model is not None:
            line += f" {row['lpips']:>7.4f}"
        print(line)

        if cfg.out_dir:
            from torchvision.utils import save_image

            pair = torch.cat([img.unsqueeze(0), recon.float()], dim=0)  # 2, C, H, W
            save_image(
                (pair + 1) / 2,
                os.path.join(cfg.out_dir, f"{class_name}_o{cfg.order}_n{cfg.num_inversion_steps}.png"),
                nrow=2,
            )

    def mean_std(key: str) -> tuple[float, float]:
        vals = torch.tensor([r[key] for r in rows])
        return vals.mean().item(), vals.std().item()

    m_vae, s_vae = mean_std("vae_mse")
    m_mse, s_mse = mean_std("mse")
    m_psnr, s_psnr = mean_std("psnr")
    print("-" * len(header))
    print(f"mean vae floor mse : {m_vae:.5f} ± {s_vae:.5f}")
    print(f"mean inversion mse : {m_mse:.5f} ± {s_mse:.5f}")
    print(f"mean psnr          : {m_psnr:.2f} ± {s_psnr:.2f} dB")
    if lpips_model is not None:
        m_lp, s_lp = mean_std("lpips")
        print(f"mean lpips         : {m_lp:.4f} ± {s_lp:.4f}")
    if cfg.out_dir:
        print(f"side-by-side reconstructions saved under {cfg.out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(ValidationConfig))
