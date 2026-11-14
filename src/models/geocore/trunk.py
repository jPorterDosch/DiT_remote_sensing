"""Block-input feature tap for the vendored GeoCore `Flux2` trunk.

Mirrors `Flux2.forward` (flux2.py) step for step -- `make_vec`, the three global modulations,
`img_in` / `txt_in`, RoPE on both id sets, the double blocks, the `[txt, img]` concatenation and
the single blocks -- but stops after the deepest requested block and returns, for every
requested block k, the image tokens ENTERING block k plus the (shift, scale, gate) triple that
block applies. That is the FLUX harness's feature definition (config_types.DOUBLE_BLOCK_FEATURE,
"block input + first adaLN triple"), so GeoCore caches are drop-in for every FLUX consumer.

GeoCore's modulation is global: one triple shared by all double blocks (the first of the two
`double_stream_modulation_img` triples, the attention path), one shared by all single blocks.
The block arithmetic itself is the vendored `forward_kv_extract` with `num_ref_tokens=0`,
exactly as `Flux2.forward` calls it, so features are the model's own.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .flux2 import Flux2, GeoCore9BParams


def tiny_params(
    hidden: int = 64,
    heads: int = 2,
    depth: int = 2,
    depth_single: int = 3,
    mlp_ratio: float = 2.0,
    context_in_dim: int = 16,
    y_in_dim: int = 8,
    z_out_dim: int = 16,
    in_channels: int = 8,
    axes_dim: list[int] | None = None,
    theta: int = 2000,
) -> GeoCore9BParams:
    """A CPU-sized `GeoCore9BParams` for tests and smokes (sum(axes_dim) == hidden // heads)."""
    return GeoCore9BParams(
        in_channels=in_channels,
        context_in_dim=context_in_dim,
        hidden_size=hidden,
        num_heads=heads,
        depth=depth,
        depth_repa=depth,
        depth_single_blocks=depth_single,
        axes_dim=list(axes_dim) if axes_dim is not None else [8, 12, 12],
        theta=theta,
        mlp_ratio=mlp_ratio,
        y_in_dim=y_in_dim,
        z_out_dim=z_out_dim,
    )


def _check_indices(ft_indices, n_blocks: int) -> list[int]:
    ks = list(ft_indices)
    if not ks:
        raise ValueError("ft_indices must name at least one block")
    if any(not isinstance(k, int) for k in ks):
        raise ValueError(f"ft_indices must be ints, got {ks}")
    if any(a >= b for a, b in zip(ks, ks[1:])):
        raise ValueError(f"ft_indices must be strictly increasing, got {ks}")
    if ks[0] < 0 or ks[-1] >= n_blocks:
        raise ValueError(f"ft_indices must lie in [0, {n_blocks - 1}], got {ks}")
    return ks


def _stack_mod(mod: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """(B, 3, C) from GeoCore's (B, 1, C) modulation chunks (Modulation.forward adds the token axis)."""
    shift, scale, gate = (m.reshape(m.shape[0], -1) for m in mod)
    return torch.stack([shift, scale, gate], dim=1)


def randomize_zero_init(model: Flux2, generator: torch.Generator | None = None, std: float = 0.1) -> Flux2:
    """TEST HELPER. GeoCore zero-initialises its adaLN modulation linears and its null geo
    embeddings (DiT convention), so a freshly constructed tiny model has all-zero (shift, scale,
    gate) and every block adds nothing: block inputs would not depend on the trunk at all. Tests
    that check tapping / sharing / conditioning need non-degenerate modulation, so this gives
    those parameters random values. Never used on loaded weights."""
    with torch.no_grad():
        for mod in (
            model.double_stream_modulation_img,
            model.double_stream_modulation_txt,
            model.single_stream_modulation,
        ):
            mod.lin.weight.normal_(0.0, std, generator=generator)
            if mod.lin.bias is not None:
                mod.lin.bias.normal_(0.0, std, generator=generator)
        for emb in (model.null_res_emb, model.null_lon_emb, model.null_lat_emb):
            emb.normal_(0.0, 1.0, generator=generator)
        # The conditioning MLPs' output layers are zero-initialised too (vec would ignore t / geo).
        for embedder in (model.time_in, model.res_in, model.lon_in, model.lat_in):
            embedder.out_layer.weight.normal_(0.0, std, generator=generator)
            if embedder.out_layer.bias is not None:
                embedder.out_layer.bias.normal_(0.0, std, generator=generator)
    return model


@torch.no_grad()
def forward_feat(
    model: Flux2,
    x: Tensor,
    x_ids: Tensor,
    timesteps: Tensor,
    ctx: Tensor,
    ctx_ids: Tensor,
    y: Tensor,
    res: Tensor,
    lon: Tensor,
    lat: Tensor,
    ft_indices,
) -> list[Tensor]:
    """[feat_0, mod_0, feat_1, mod_1, ...] in ascending `ft_indices`: feat = image tokens entering
    block k, (B, L_img, C); mod = that block family's (shift, scale, gate), (B, 3, C). Early-exits
    after max(ft_indices); the final layer never runs. Arguments as `Flux2.forward` (x is the
    (B, C_in, H, W) latent; timesteps the fraction in [0, 1])."""
    n_double = len(model.double_blocks)
    n_blocks = n_double + len(model.single_blocks)
    ks = _check_indices(ft_indices, n_blocks)
    max_ft = ks[-1]
    wanted = set(ks)

    if x.ndim != 4:
        raise ValueError(f"x must be (B, C, H, W), got {tuple(x.shape)}")
    x = x.flatten(2).transpose(1, 2)
    num_txt_tokens = ctx.shape[1]

    vec = model.make_vec(timesteps, res, lon, lat, y)
    double_block_mod_img = model.double_stream_modulation_img(vec)
    double_block_mod_txt = model.double_stream_modulation_txt(vec)
    single_block_mod, _ = model.single_stream_modulation(vec)

    img = model.img_in(x)
    txt = model.txt_in(ctx)
    pe_x = model.pe_embedder(x_ids)
    pe_ctx = model.pe_embedder(ctx_ids)

    out: list[Tensor] = []
    for i, block in enumerate(model.double_blocks):
        if i in wanted:
            out.append(img.clone())
            out.append(_stack_mod(double_block_mod_img[0]))
        if i == max_ft:
            return out
        img, txt, _ = block.forward_kv_extract(
            img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, num_ref_tokens=0
        )

    img = torch.cat((txt, img), dim=1)
    pe = torch.cat((pe_ctx, pe_x), dim=2)
    for j, block in enumerate(model.single_blocks):
        k = n_double + j
        if k in wanted:
            out.append(img[:, num_txt_tokens:].clone())
            out.append(_stack_mod(single_block_mod))
        if k == max_ft:
            return out
        img, _ = block.forward_kv_extract(img, pe, single_block_mod, num_txt_tokens, num_ref_tokens=0)
    raise AssertionError("unreachable: max_ft is a valid block index")
