from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer


# Used for learning rate scaling and gradient clipping with mixed precision training. Returns grad norm for logging.
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, device: torch.device) -> None:
        self._scaler = torch.amp.GradScaler(device.type)

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: Optimizer,
        clip_grad: float | None = None,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> torch.Tensor | None:
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._scaler.unscale_(optimizer)

        if not update_grad:
            return None

        if clip_grad is not None:
            if parameters is None:
                raise ValueError("parameters must be provided when clip_grad is not None")
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            if parameters is None:
                raise ValueError("parameters must be provided to compute grad norm")
            norm = get_grad_norm(parameters)

        self._scaler.step(optimizer)
        self._scaler.update()

        return norm

    def state_dict(self) -> dict[str, Any]:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler.load_state_dict(state_dict)


def get_grad_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)

    if len(grads) == 0:
        return torch.tensor(0.0)

    device = grads[0].device

    if norm_type == float("inf"):
        return max(g.detach().abs().max().to(device) for g in grads)
    else:
        return torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type,
        )


# ==============================================================
# 2D sine-cosine positional embedding for vision transformers, adapted from repo
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]

        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# ==============================================================
# Layer-wise learning rate decay for vision transformers, adapted from repo
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=None, layer_decay=0.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    if no_weight_decay_list is None:
        no_weight_decay_list = []

    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers
