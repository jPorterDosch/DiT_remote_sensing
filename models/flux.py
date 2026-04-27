from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange
from src.flux.feat_flux import Featurizer4Eval
from registry import register_model


@register_model("flux")
class FluxModel:
    def __init__(self, cfg, category_list: list[str]) -> None:
        self._cd       = cfg.cd
        self._pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)
        self._inner    = Featurizer4Eval(
            cat_list=list(category_list),
            ensemble_size=cfg.model.ensemble_size,
        )

    @torch.no_grad()
    def extract(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: list[int],
        ensemble_size: int,
        caption:  str = "a photo of a image",
        category: str = "image",
    ) -> torch.Tensor:
        feat_raw, ada = self._inner.forward(
            None, img,
            caption=caption,
            category=category,
            timestep=timestep,
            block_idx=block_idx,
            ensemble_size=ensemble_size,
        )

        B, C, H, W = feat_raw.shape

        if self._cd:
            feat_raw[:, 154,  :, :] = 0.0
            feat_raw[:, 1446, :, :] = 0.0

        feat = rearrange(feat_raw, "b c h w -> b (h w) c")
        feat = self._pre_norm(feat)
        feat = rearrange(feat, "b (h w) c -> b c h w", h=H, w=W)

        shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return (1 + scale) * feat + shift  # B, C, H, W
