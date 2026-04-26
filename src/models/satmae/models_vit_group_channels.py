# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
from collections.abc import Sequence
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import VisionTransformer

from .utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed


class GroupChannelsVisionTransformer(VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        global_pool: bool = False,
        channel_embed: int = 256,
        channel_groups: Sequence[Sequence[int]] = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        img_size = kwargs["img_size"]
        patch_size = kwargs["patch_size"]
        # in_c = kwargs["in_chans"]
        embed_dim = kwargs["embed_dim"]

        self.channel_groups = channel_groups

        self.patch_embed = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, len(group), embed_dim) for group in channel_groups]
        )
        first_patch_embed = cast(PatchEmbed, self.patch_embed[0])
        num_patches = int(first_patch_embed.num_patches)

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], torch.arange(num_groups).numpy()
        )
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # Extra embedding for cls to fill embed_dim
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

        self.use_global_pool = global_pool
        if self.use_global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, list(group), :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, _, _, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)
        x = x.view(b, -1, D)  # (N, G*L, D)

        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.use_global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    # Need to override forward to prevent timm inherited forward from pooling, when we already handle global pool in forward_features.
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)

        if hasattr(self, "head_drop"):
            x = self.head_drop(x)

        x = self.head(x)
        return x


def vit_large_patch16(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
