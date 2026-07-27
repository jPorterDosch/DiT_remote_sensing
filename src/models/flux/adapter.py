from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from registry import register_model

from .feat_flux import Featurizer4Eval


@register_model("flux")
class FluxModel:
    def __init__(self, cfg, category_list: list[str]) -> None:
        self._cd = cfg.cd
        self._discard_channels = list(cfg.discard_channels)
        self._pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)
        self._inner = Featurizer4Eval(
            cat_list=list(category_list),
            ensemble_size=cfg.model.ensemble_size,
        )

        if getattr(cfg, "lora_checkpoint", ""):
            from models.lora import lora_wrap_flux

            flux = self._inner.model

            lora_wrap_flux(
                flux,
                cfg.k,
                cfg.lora_rank,
                cfg.lora_alpha,
                cfg.lora_dropout,
                wrap_o=cfg.wrap_output,
            )

            # load all tensors onto cpu first regardless of where they were saved from
            ckpt = torch.load(cfg.lora_checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = flux.load_state_dict(ckpt["lora_state_dict"], strict=False)

            print(f"Loaded LoRA checkpoint: {cfg.lora_checkpoint}")

            if missing:
                print(f"\tMissing keys: {missing}")
            if unexpected:
                print(f"\tUnexpected keys: {unexpected}")

    @torch.no_grad()
    def extract_raw(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: int,
        ensemble_size: int,
        caption: str = "a photo of a image",
        category: str = "image",
        guidance: float = 3.5,
        latents: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-shot raw features, PRE-normalization (no channel discard / LayerNorm /
        adaLN / L2) — the multi-timestep extraction path applies those offline.

        Returns (feat_raw (1, C, h, w), ada (1, 3, C), latents, noise) where latents/
        noise are the tensors actually mixed into the noisy input (see Featurizer4Eval.
        forward), so callers can assert eps-constancy across timesteps.

        NOTE: called by tasks/utils.extract_features but missing from every branch —
        reconstructed as a thin delegation to Featurizer4Eval.forward, which already
        returns exactly this tuple.
        """
        return self._inner.forward(
            None,
            img,
            caption=caption,
            category=category,
            timestep=timestep,
            block_idx=block_idx,
            ensemble_size=ensemble_size,
            guidance=guidance,
            latents=latents,
            noise=noise,
            generator=generator,
        )

    @torch.no_grad()
    def extract_inversion(
        self,
        img: torch.Tensor,
        timesteps: list[int],
        num_inversion_steps: int,
        block_idx: int,
        guidance: float = 3.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inversion-chain features: one RF-Solver reverse-ODE trajectory per image,
        caching pre-normalization block hidden states at each requested timestep.

        The chain draws no eps of its own (no ensemble), but ae.encode samples the
        VAE posterior from the global RNG, so features are reproducible only under
        identical RNG state (see Featurizer4Eval.invert_chain). Returns
        (feats (K, C, h, w), mods (K, 3, C)) in the order of `timesteps`.
        """
        out = self._inner.invert_chain(
            img,
            cache_timesteps=list(timesteps),
            num_inversion_steps=num_inversion_steps,
            block_idx=block_idx,
            guidance=guidance,
        )
        feats = torch.cat([out["feats"][t] for t in timesteps], dim=0)  # K, C, h, w
        mods = torch.cat([out["mods"][t] for t in timesteps], dim=0)  # K, 3, C
        return feats, mods

    @torch.no_grad()
    def extract(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: int,
        ensemble_size: int,
        caption: str = "a photo of a image",
        category: str = "image",
        guidance: float = 3.5,
    ) -> torch.Tensor:
        # Featurizer4Eval.forward returns (feat, mod, latents, noise); the trailing
        # latents/noise are only needed by the multi-timestep eps-constancy checks.
        feat_raw, ada, _latents, _noise = self._inner.forward(
            None,
            img,
            caption=caption,
            category=category,
            timestep=timestep,
            block_idx=block_idx,
            ensemble_size=ensemble_size,
            guidance=guidance,
        )

        B, C, H, W = feat_raw.shape

        if self._cd:
            for ch in self._discard_channels:
                feat_raw[:, ch, :, :] = 0.0

        feat = rearrange(feat_raw, "b c h w -> b (h w) c")
        feat = self._pre_norm(feat)
        feat = rearrange(feat, "b (h w) c -> b c h w", h=H, w=W)

        shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return (1 + scale) * feat + shift  # B, C, H, W
