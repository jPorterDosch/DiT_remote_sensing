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
    def extract(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: int,
        ensemble_size: int,
        caption: str = "a photo of a image",
        category: str = "image",
    ) -> torch.Tensor:
        feat_raw, ada = self._inner.forward(
            None,
            img,
            caption=caption,
            category=category,
            timestep=timestep,
            block_idx=block_idx,
            ensemble_size=ensemble_size,
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
