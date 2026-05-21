import math

import torch


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        """Frozen base layer + trainable low-rank (A,B)."""
        super().__init__()

        if not isinstance(base, torch.nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        dtype = base.weight.dtype
        device = base.weight.device

        self.in_features = base.in_features
        self.out_features = base.out_features

        # Frozen copy of original weight/bias.
        self.weight = torch.nn.Parameter(
            base.weight.detach().clone().to(dtype=dtype, device=device),
            requires_grad=False,
        )

        self.bias = None
        if base.bias is not None:
            self.bias = torch.nn.Parameter(
                base.bias.detach().clone().to(dtype=dtype, device=device),
                requires_grad=False,
            )

        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        # A: down projection, B: up projection.
        self.A = torch.nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.B = torch.nn.Parameter(torch.empty(self.out_features, rank, device=device, dtype=dtype))

        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same as linear(x, weight, bias), but with LoRA delta added.
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = torch.nn.functional.linear(
            torch.nn.functional.linear(self.drop(x), self.A),
            self.B,
        )
        return base_out + self.scaling * lora_out


def lora_wrap_flux(
    model: torch.nn.Module,
    block_idx: int | list[int],
    rank: int,
    alpha: float,
    dropout: float,
    wrap_o: bool = False,
):
    """
    Wrap selected BFL FLUX projections with LoRA.

    Global block indexing:
      double_blocks: 0 ... len(double_blocks)-1
      single_blocks: len(double_blocks) ... len(double_blocks)+len(single_blocks)-1

    Double blocks:
      wraps img_attn.qkv
      optionally wraps img_attn.proj

    Single blocks:
      wraps linear1 = qkv + mlp_in
      optionally wraps linear2 = attn_proj + mlp_out

    Args:
        model: The model containing the attention module to wrap.
        block_idx: The index (or indices) of the attention block to wrap.
        rank: The LoRA rank (number of parameters in the low-rank update).
        alpha: The LoRA scaling factor.
        dropout: The LoRA dropout rate.
        wrap_o: If True, also wrap img_attn.proj for double blocks and linear2 for single blocks.
    """
    if not hasattr(model, "double_blocks") or not hasattr(model, "single_blocks"):
        raise ValueError("Model must have double_blocks and single_blocks attributes.")

    block_indices = [block_idx] if isinstance(block_idx, int) else block_idx

    if len(block_indices) != len(set(block_indices)):
        raise ValueError("block_idx contains duplicate indices, which is not allowed.")

    n_double = len(model.double_blocks)  # Avoid hardcoding FLUX-dev's 19 double blocks.

    wrapped = 0

    for i, block in enumerate(model.double_blocks):
        if not hasattr(block, "img_attn"):
            raise ValueError(f"double_blocks[{i}] missing img_attn: {type(block)}")
        if not hasattr(block.img_attn, "qkv") or not hasattr(block.img_attn, "proj"):
            raise ValueError(f"double_blocks[{i}].img_attn missing qkv/proj: {type(block.img_attn)}")

        if i in block_indices:
            block.img_attn.qkv = LoRALinear(block.img_attn.qkv, rank, alpha, dropout)

            if wrap_o:
                block.img_attn.proj = LoRALinear(block.img_attn.proj, rank, alpha, dropout)

            wrapped += 1

    for i, block in enumerate(model.single_blocks):
        if not hasattr(block, "linear1") or not hasattr(block, "linear2"):
            raise ValueError(f"Expected block to have linear1 and linear2 attributes, but got {type(block)}")
        if (n_double + i) in block_indices:
            block.linear1 = LoRALinear(block.linear1, rank, alpha, dropout)

            if wrap_o:
                block.linear2 = LoRALinear(block.linear2, rank, alpha, dropout)

            wrapped += 1

    if wrapped == 0:
        raise ValueError(
            f"No blocks matched block_idx={sorted(block_indices)}. "
            f"Valid global indices are 0...{n_double + len(model.single_blocks) - 1}."
        )

    print(
        f"Wrapped {wrapped} attention blocks with LoRA (rank={rank}, alpha={alpha}, dropout={dropout}, wrap_o={wrap_o})."
    )
