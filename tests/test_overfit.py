# ruff: noqa: E402 — sys.path must be mutated before any local imports
from __future__ import annotations

import os
import sys


def _find_project_root(start: str) -> str:
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"Could not locate project root (src/) from {start}")
        d = parent


_root = _find_project_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)                              # registry.py lives here
sys.path.insert(0, os.path.join(_root, "src"))         # datasets, tasks, utils, …
sys.path.insert(0, os.path.join(_root, "src", "models"))  # flux.* internal imports

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless SLURM nodes
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models.flux.feat_flux import Featurizer4Eval, prepare
from models.lora import lora_wrap_flux

# ---------------------------------------------------------------------------
# Helpers inlined from src/tasks/train_diffusion.py to avoid import chain
# (importing tasks/ pulls in torchvision, tensorboard, etc.).
# ---------------------------------------------------------------------------


class MIMDecoder(torch.nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeatureCapture:
    def __init__(self):
        self.features: torch.Tensor | None = None
        self._handle = None

    def register(self, block: torch.nn.Module, txt_len: int) -> None:
        def _hook(module, inp, output):
            if isinstance(output, tuple):
                self.features = output[0]               # DoubleStreamBlock: (img, txt)
            else:
                self.features = output[:, txt_len:, :]  # SingleStreamBlock: slice off txt prefix

        self._handle = block.register_forward_hook(_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _random_masking(
    x: torch.Tensor, mask_token: torch.nn.Parameter, mask_ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones(B, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask_expanded = mask.unsqueeze(-1).to(dtype=x.dtype)
    x_masked = x * (1 - mask_expanded) + mask_token.to(dtype=x.dtype) * mask_expanded
    return x_masked, mask


# ---------------------------------------------------------------------------
# T3: Single-batch overfit test (two-head: flow + MIM)
# ---------------------------------------------------------------------------


def test_overfit(
    block_idx_global: int = 28,
    lora_rank: int = 4,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    mask_ratio: float = 0.75,
    mim_loss_weight: float = 1.0,
    lr: float = 1e-2,
    warmup_steps: int = 20,
    n_steps: int = 100,
    guidance_scale: float = 3.5,
    timestep: int = 260,
    min_reduction: float = 0.5,   # require >=50% combined loss drop
    out_path: str | None = None,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[overfit] device={device}")

    print("[overfit] loading Flux + VAE ...")
    featurizer = Featurizer4Eval(flux_id="flux-dev", cat_list=[])
    flux = featurizer.model
    vae = featurizer.ae
    flux.to(device)
    vae.to(device)

    # Fixed synthetic image in [-1, 1] — no dataset needed.
    torch.manual_seed(42)
    imgs = torch.rand(1, 3, 224, 224, device=device) * 2 - 1

    with torch.no_grad():
        latents = vae.encode(imgs).to(torch.bfloat16)

    # mask_token lives in feature space (hidden_size), not token space (in_channels).
    mask_token = torch.nn.Parameter(torch.zeros(1, 1, flux.hidden_size, device=device))
    decoder = MIMDecoder(flux.hidden_size, flux.in_channels).to(device=device, dtype=torch.bfloat16)

    n_double = len(flux.double_blocks)
    hooked_block = (
        flux.double_blocks[block_idx_global]
        if block_idx_global < n_double
        else flux.single_blocks[block_idx_global - n_double]
    )
    txt_len = featurizer.null_prompt_embeds.shape[1]
    capture = _FeatureCapture()
    capture.register(hooked_block, txt_len)

    # Freeze base weights, then insert LoRA at the target block.
    for param in flux.parameters():
        param.requires_grad = False
    lora_wrap_flux(flux, block_idx_global, lora_rank, lora_alpha, lora_dropout, wrap_o=False)
    for name, param in flux.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            param.requires_grad = True

    trainable_params = [p for p in flux.parameters() if p.requires_grad]
    trainable_params.append(mask_token)
    trainable_params.extend(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(1, warmup_steps)
    )

    print(
        f"[overfit] trainable params: {sum(p.numel() for p in trainable_params):,} "
        f"({len(trainable_params)} tensors)"
    )

    # Precompute fixed inputs — noisy latent and targets are constant across steps.
    t = timestep / 1000.0
    noise = torch.randn_like(latents)
    latents_noisy = t * noise + (1.0 - t) * latents
    img_tokens, img_ids = prepare(img=latents_noisy)
    img_tokens = img_tokens.to(device, dtype=latents.dtype)
    img_ids = img_ids.to(device)

    txt = featurizer.null_prompt_embeds.to(device=device, dtype=latents.dtype)
    txt_ids = featurizer.text_ids.to(device=device)
    y = featurizer.vec.to(device=device, dtype=latents.dtype)
    guidance_vec = torch.full((1,), guidance_scale, device=device, dtype=latents.dtype)
    timesteps_vec = torch.full((1,), t, device=device, dtype=latents.dtype)

    # Flow target: velocity = noise - clean_latent (flow matching).
    v_target, _ = prepare(noise - latents)
    v_target = v_target.to(device=device, dtype=latents.dtype)

    # MIM target: clean latent tokens
    target_clean, _ = prepare(latents)
    target_clean = target_clean.to(device=device, dtype=torch.float32)

    flux.train()
    vae.eval()

    total_losses: list[float] = []
    flow_losses: list[float] = []
    mim_losses: list[float] = []

    print(
        f"[overfit] running {n_steps} steps  block={block_idx_global}  "
        f"lr={lr}  mask_ratio={mask_ratio}  mim_loss_weight={mim_loss_weight}"
    )

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        # Reset so a hook failure is caught loudly rather than reusing stale features.
        capture.features = None

        # Flow head: Flux sees clean unmasked tokens.
        v_pred = flux(
            img=img_tokens,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps_vec,
            guidance=guidance_vec,
        )

        if capture.features is None:
            raise RuntimeError(
                f"Feature hook did not fire. Check that block_idx_global={block_idx_global} "
                f"is valid (n_double={n_double}, n_single={len(flux.single_blocks)})."
            )

        flow_loss = F.mse_loss(v_pred.float(), v_target.float())

        # MIM head: mask captured features, decode, MSE vs clean latents at masked positions.
        features_masked, mask = _random_masking(capture.features, mask_token, mask_ratio)
        pred_clean = decoder(features_masked)
        per_token = ((pred_clean.float() - target_clean) ** 2).mean(dim=-1)
        mim_loss = (per_token * mask).sum() / mask.sum().clamp(min=1)

        total_loss = flow_loss + mim_loss_weight * mim_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_losses.append(float(total_loss.detach().cpu()))
        flow_losses.append(float(flow_loss.detach().cpu()))
        mim_losses.append(float(mim_loss.detach().cpu()))

        if step % 10 == 0 or step == n_steps - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  step {step:3d}: total={total_losses[-1]:.6f}  "
                f"flow={flow_losses[-1]:.6f}  mim={mim_losses[-1]:.6f}  "
                f"lr={current_lr:.2e}"
            )

    capture.remove()

    # --- Loss curve ---
    if out_path is None:
        out_path = os.path.join(_root, "overfit_loss_curve.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(total_losses, linewidth=1.2, label="total")
    ax.plot(flow_losses, linewidth=1.0, linestyle="--", label="flow")
    ax.plot(mim_losses, linewidth=1.0, linestyle=":", label="mim")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Two-head overfit — block {block_idx_global}, lr={lr}, "
        f"mask_ratio={mask_ratio}, mim_w={mim_loss_weight}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[overfit] loss curve saved → {out_path}")

    # Assert combined loss drops by at least min_reduction.
    window = min(5, n_steps // 10)
    initial_loss = sum(total_losses[:window]) / window
    final_loss = sum(total_losses[-window:]) / window
    reduction = (initial_loss - final_loss) / (initial_loss + 1e-12)
    print(
        f"[overfit] initial(avg {window})={initial_loss:.6f}  "
        f"final(avg {window})={final_loss:.6f}  "
        f"reduction={reduction*100:.1f}%"
    )

    assert reduction >= min_reduction, (
        f"Combined loss did not decrease by >={min_reduction*100:.0f}% "
        f"(got {reduction*100:.1f}%). "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
    print(f"[overfit] PASS — combined loss reduced by {reduction*100:.1f}% (>={min_reduction*100:.0f}% required)")
    return total_losses


if __name__ == "__main__":
    test_overfit()
