from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )
        # print(params.depth)
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def _forward_trunk(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        ft_indices=None,
        early_exit: bool = False,
    ) -> tuple[Tensor | None, list]:
        """Single shared trunk behind forward / forward_feat / forward_velocity_feat.

        The inversion-chain extractor mixes velocity-only and velocity+feature calls
        inside one ODE trajectory, so all entry points MUST compute identical
        velocities — one trunk enforces that instead of hand-synced copies.

        ft_indices: block indices to cache hidden states from (None/empty = none).
        up_ft convention: [feat] per double block, [feat, mod] per single block,
        in ft_indices order. early_exit=True stops after max(ft_indices) and skips
        the final layer; the returned pred is then None.

        Returns (pred, up_ft).
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        ft_indices = list(ft_indices) if ft_indices else []
        max_ft = max(ft_indices) if ft_indices else -1

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec_t = vec.clone().detach()
        vec_y = self.vector_in(y)
        vec = vec + vec_y
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        up_ft = []

        n_double = len(self.double_blocks)
        n_single = len(self.single_blocks)
        if early_exit:
            n_double = min(n_double, max_ft + 1)
            n_single = min(n_single, max(0, max_ft + 1 - len(self.double_blocks)))

        for i in range(n_double):
            img, txt, img_feat = self.double_blocks[i].forward_feat(
                img=img, txt=txt, vec=vec, pe=pe, return_feat=i in ft_indices
            )
            if i in ft_indices:
                up_ft.append(img_feat.clone().detach())

        if n_single > 0:
            img = torch.cat((txt, img), 1)
            offset = len(self.double_blocks)
            for i in range(n_single):
                img, img_feat, mod = self.single_blocks[i].forward_feat(
                    img, vec=(vec, vec_t, vec_y), pe=pe, return_feat=(offset + i) in ft_indices
                )
                if (offset + i) in ft_indices:
                    up_ft.append(img_feat[:, txt.shape[1] :, ...].clone().detach())
                    up_ft.append(mod)

        if early_exit:
            return None, up_ft

        img = img[:, txt.shape[1] :, ...]
        pred = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return pred, up_ft

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        pred, _ = self._forward_trunk(img, img_ids, txt, txt_ids, timesteps, y, guidance)
        return pred

    def forward_feat(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        ft_indices,
        cat=None,
        guidance: Tensor | None = None,
    ):
        """Block features only; early-exits after max(ft_indices), velocity not computed."""
        _, up_ft = self._forward_trunk(
            img, img_ids, txt, txt_ids, timesteps, y, guidance, ft_indices=ft_indices, early_exit=True
        )
        return up_ft

    def forward_velocity_feat(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        ft_indices,
        guidance: Tensor | None = None,
    ):
        """Full forward pass returning BOTH the velocity prediction and block features.

        Unlike forward_feat (which early-exits after max(ft_indices) and discards the
        velocity), this runs every block so the returned prediction is the model's
        velocity field v(x_t, t) — needed by the inversion-chain extractor, which must
        step the reverse ODE AND cache features from the same forward pass.

        Returns (pred, up_ft): pred is (N, T, in_channels) packed-latent velocity;
        up_ft matches forward_feat's convention ([feat] for double blocks,
        [feat, mod] for single blocks, in ft_indices order).
        """
        return self._forward_trunk(
            img, img_ids, txt, txt_ids, timesteps, y, guidance, ft_indices=ft_indices, early_exit=False
        )
