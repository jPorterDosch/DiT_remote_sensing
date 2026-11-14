from __future__ import annotations

import hashlib
import glob
import math
import re
from collections import OrderedDict
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from einops import repeat
import wandb

from models.flux.adapter import FluxModel
from models.flux.feat_flux import Featurizer4Eval, prepare
from models.flux.sampling import denoise, get_schedule, unpack
from config_types import NUM_BLOCKS, double_block_stamp
from models.lora import LORA_SCOPE, lora_block_indices, lora_wrap_flux
from datasets.utils import _strided_indices
from registry import register_task
from utils import to_jsonable

from utils import seed_worker
from sklearn.metrics import f1_score

from models import gradproj as GP

from . import align as AL
from .utils import _flatten_scalars_into


def _expand_null_embeddings(featurizer: Featurizer4Eval, batch_size: int, device, dtype):
    txt, txt_ids, y = (
        featurizer.null_prompt_embeds,
        featurizer.text_ids,
        featurizer.vec,
    )

    if any(v is None for v in [txt, txt_ids, y]):
        raise ValueError("Null prompt embeddings must be provided in featurizer.")

    if txt.shape[0] == 1 and batch_size > 1:
        txt = repeat(txt, "1 ... -> b ...", b=batch_size)
    if txt_ids.shape[0] == 1 and batch_size > 1:
        txt_ids = repeat(txt_ids, "1 ... -> b ...", b=batch_size)
    if y.shape[0] == 1 and batch_size > 1:
        y = repeat(y, "1 ... -> b ...", b=batch_size)

    txt = txt.to(device=device, dtype=dtype)
    txt_ids = txt_ids.to(device=device)
    y = y.to(device=device, dtype=dtype)

    return txt, txt_ids, y


def _get_lora_state_dict(model: torch.nn.Module) -> dict:
    """
    Extracts the state dict of only the LoRA adapter parameters from the given model. This is useful for saving checkpoints that only include the trainable LoRA parameters,
        which can be much smaller than the full model weights.

    The function iterates through all named parameters in the model and selects those whose names end with ".A" or ".B", which are the convention for LoRA adapter weights.
        It returns a dictionary containing only these parameters and their values.

    Args:
        model: The PyTorch model containing LoRA adapters.

    Returns:
        A dictionary containing only the LoRA adapter parameters from the model's state dict.
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            lora_state_dict[name] = param.detach().cpu()
    return lora_state_dict


def _save_lora_checkpoint(
    cfg,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    train_metrics: dict,
    val_metrics: dict,
    name: str,
    mask_token: torch.nn.Parameter | None = None,
    decoder: MIMDecoder | None = None,
    probe_head: torch.nn.Module | None = None,
    extra: dict | None = None,
) -> None:
    checkpoint_dir = Path(cfg.save_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Zero-padded so a lexicographic glob orders step900 BEFORE step1000 (a plain
    # f"{global_step}" does not, and "lora_best_*" is globbed by downstream tooling).
    checkpoint_path = checkpoint_dir / f"{name}_step{global_step:08d}.pt"

    ckpt = {
        "global_step": global_step,
        "lora_state_dict": _get_lora_state_dict(model),
        "mask_token": mask_token.detach().cpu() if mask_token is not None else None,
        "decoder_state_dict": decoder.state_dict() if decoder is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "probe_head_state_dict": probe_head.state_dict() if probe_head is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "cfg": to_jsonable(cfg),
        "lora_scope": LORA_SCOPE,
        # up_ft[0] at a double block changed definition on 2026-10-05 (config_types)
        **double_block_stamp(cfg.k),
        # Alignment runs add "align" (arm, target, weight, projector state, ...) so an extraction
        # from this checkpoint can stamp weights="flux-dev+lora:<arm>" (tasks/extraction.py).
        **(extra or {}),
    }

    torch.save(ckpt, checkpoint_path)


def _flow_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict:
    """
    Helper to compute shared flow metrics for train/val loops. Computes MSE, RMSE,
    mean norms of pred/target/residual, relative flow error, and cosine similarity between pred and target.

    All metrics are computed on unmasked regions.

    Interpretation:
        - MSE/RMSE: overall error magnitude. MSE is optimization objective, RMSE is more interpretable in flow units.
        - Norms: average magnitude of predicted flow, true flow, and error, which can help identify
            if model is under/over-shooting on average.
        - Relative flow error: error normalized by true flow magnitude, which can help identify if model is performing worse on smaller or larger flows.
        - Cosine similarity: directional alignment between predicted and true flow, independent of magnitude.
    """
    visible = (1 - mask).bool()

    # Edge case check for degenerate mask that hides all positions
    if visible.sum() == 0:
        return {
            f"{prefix}/flow_mse_loss": float("nan"),
            f"{prefix}/flow_rmse": float("nan"),
            f"{prefix}/pred_norm_mean": float("nan"),
            f"{prefix}/target_norm_mean": float("nan"),
            f"{prefix}/residual_norm_mean": float("nan"),
            f"{prefix}/relative_flow_error": float("nan"),
            f"{prefix}/cosine_pred_target": float("nan"),
        }

    pred = pred[visible]
    target = target[visible]
    pred_f = pred.float()
    target_f = target.float()
    residual_f = pred_f - target_f

    mse = F.mse_loss(pred_f, target_f)
    target_norm = target_f.norm(dim=1)
    residual_norm = residual_f.norm(dim=1)

    return {
        f"{prefix}/flow_mse_loss": float(mse.detach().cpu()),
        f"{prefix}/flow_rmse": float(torch.sqrt(mse).detach().cpu()),
        f"{prefix}/pred_norm_mean": float(pred_f.norm(dim=1).mean().detach().cpu()),
        f"{prefix}/target_norm_mean": float(target_norm.mean().detach().cpu()),
        f"{prefix}/residual_norm_mean": float(residual_f.norm(dim=1).mean().detach().cpu()),
        f"{prefix}/relative_flow_error": float((residual_norm / (target_norm + 1e-8)).mean().detach().cpu()),
        f"{prefix}/cosine_pred_target": float(
            F.cosine_similarity(pred_f, target_f, dim=1).mean().detach().cpu()
        ),
    }


def _mim_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict:
    """
    pred/target: (B, L, D) in clean latent space.
    mask: (B, L), 1 = masked position.
    Logs masked loss (optimization objective) and unmasked loss (diagnostic).
    """
    pred_f = pred.float()
    target_f = target.float()
    mask_f = mask.float()

    per_token = ((pred_f - target_f) ** 2).mean(dim=-1)
    n_masked = mask_f.sum().clamp(min=1)
    n_unmasked = (1 - mask_f).sum().clamp(min=1)

    masked_loss = (per_token * mask_f).sum() / n_masked
    unmasked_loss = (per_token * (1 - mask_f)).sum() / n_unmasked

    return {
        f"{prefix}/mim_loss": float(masked_loss.detach().cpu()),
        f"{prefix}/mim_rmse": float(masked_loss.sqrt().detach().cpu()),
        f"{prefix}/mim_unmasked_loss": float(unmasked_loss.detach().cpu()),
        f"{prefix}/mask_ratio": float(mask_f.mean().detach().cpu()),
    }


class MIMDecoder(torch.nn.Module):
    """
    Projects intermediate Flux block features to clean latent token space.
    Training-only: not used during inference / feature extraction.
    hidden_size -> in_channels (e.g. 3072 -> 64 for flux-dev).
    """

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
    """
    Captures intermediate block output during the forward pass via a registered hook.
    DoubleStreamBlock returns (img, txt); SingleStreamBlock returns concatenated (txt || img).
    txt_len is needed to slice off text tokens in the single-stream case.

    WARNING (2026-09-10): NON-FUNCTIONAL against this repo's Flux. _forward_trunk invokes
    blocks via .forward_feat() (a plain method call), so register_forward_hook never fires
    and .features stays None. Kept only as documentation of the old MIM feature path; use
    _forward_trunk's ft_indices to get block features instead.
    """

    def __init__(self):
        self.features: torch.Tensor | None = None
        self._handle = None

    def register(self, block: torch.nn.Module, txt_len: int) -> None:
        def _hook(module, inp, output):
            if isinstance(output, tuple):
                self.features = output[0]  # DoubleStreamBlock: (img, txt)
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
    """
    Replace a random subset of tokens with mask_token. Similar to MAE but tokens are replaced
    (not dropped) so length is correct for positional encoding.

    x: (B, L, D)
    Returns: x_masked (B, L, D), mask (B, L) with 1 = masked position.
    """
    B, L, _ = x.shape
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


def _no_mask(x: torch.Tensor) -> torch.Tensor:
    """Return a mask tensor where every token is visible."""
    return torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)


def _cycle_loader(dataloader: torch.utils.data.DataLoader):
    """Yield batches from dataloader indefinitely, re-shuffling each epoch."""
    while True:
        for batch in dataloader:
            yield batch


def _add_batch_metrics(accum_metrics: dict, batch_metrics: dict) -> dict:
    """Append scalar values from batch_metrics into per-key lists in accum_metrics."""
    for key, value in batch_metrics.items():
        if key not in accum_metrics:
            accum_metrics[key] = []
        accum_metrics[key].append(value)
    return accum_metrics


def _average_batch_metrics(accum_metrics: dict) -> dict:
    """Average the per-key lists produced by _add_batch_metrics into scalar floats."""
    return {k: float(np.mean(np.array(v))) for k, v in accum_metrics.items()}


# ---------------------------------------------------------------------------------------
# PINNED TRAINING CONSTANTS (2026-09-08). Both are enforced in FinetuneDiffusionTask.run.
#
# PINNED_GUIDANCE: cfg.guidance_scale is ONE field feeding BOTH this training path and the
# extraction path (tasks/utils.py). Every feature cache in the project was extracted at
# g=1.0, so training at run.py's old 3.5 default would have adapted the model at a guidance
# it is never probed at, confounding every frozen-vs-adapted comparison with a conditioning
# shift. Pinned to the value the caches use. To sweep guidance, change this constant AND
# re-extract -- the extraction cache tag carries g{value}, so the two cannot silently diverge.
#
# PINNED_MIM_LOSS_WEIGHT: the MIM auxiliary head is a dropped research angle. It decoded from
# the SAME block-28 features the probes read, so any nonzero weight makes an adapted-model
# result partly about the auxiliary objective rather than about flow adaptation.
PINNED_GUIDANCE = 1.0
PINNED_MIM_LOSS_WEIGHT = 0.0

# Validation must be DETERMINISTIC across evaluations or best_val_loss selects on sampling
# noise. With multi-timestep training both t and eps vary per batch, so val fixes both: t by
# round-robin over the grid (below) and eps from this fixed seed.
VAL_NOISE_SEED = 12345

# Validation evaluates every image at every grid timestep (see _validate_diffusion), so its
# cost is n_images * len(grid) forwards. The cap is applied as a STRIDED subset of the test
# split in run() -- a prefix of the class-directory-major split would be single-class
# (2026-09-11 review: at 512 it was 512x AnnualCrop, so best-checkpoint selection and every
# per-t diagnostic described ONE class -- the same prefix defect make_loaders' max_samples
# fix removed, reintroduced here). 128 images x 7 t ~ 900 forwards per call keeps validation
# affordable at realistic log_val_steps; the subset is fixed across evaluations.
VAL_MAX_IMAGES = 128


def _normalize_timestep_grid(t) -> list[int]:
    """cfg.t is int | list[int] (validated in run.py). Training accepts both: a scalar is the
    one-timestep grid, preserving the previous single-t behaviour exactly."""
    return [int(t)] if isinstance(t, int) else [int(x) for x in t]


def _sem_t_weights_for(
    t: torch.Tensor, grid: list[int], weights: list[float] | None, t_weights: list[float] | None = None
) -> torch.Tensor | None:
    """Per-image alignment-loss weights (align.sem_t_weights) for the sampled t (B,) in [0, 1]:
    the weight of the grid point each t came from, divided by the EXPECTED weight under the t
    sampling distribution (t_weights, uniform when None) so that E[w] = 1: a window holding 25 %
    of the sampling mass gets weight 4 inside it and 0 outside, and the semantic gradient mass
    per optimizer step matches the unrestricted run in expectation (review 2026-10-26, rule 8:
    without this the restricted arm would run at align.weight x P(window)). All-ones weights are
    unchanged. None when the feature is off. Raises if a t is not on the grid (the training t are
    always grid points; a mismatch means a bug) or if the window has no sampling mass."""
    if weights is None:
        return None
    p = np.ones(len(grid)) if t_weights is None else np.asarray(t_weights, dtype=float)
    if p.shape != (len(grid),) or p.sum() <= 0:
        raise ValueError(f"sem_t_weights: t_weights must give one weight per grid point, got {t_weights}")
    mass = float((p / p.sum() * np.asarray(weights, dtype=float)).sum())
    if mass <= 0:
        raise ValueError(f"sem_t_weights {weights} put no mass on sampled timesteps (t_weights {t_weights})")
    lut = {int(g): float(w) / mass for g, w in zip(grid, weights)}
    out = []
    for ti in t.detach().float().cpu().tolist():
        key = int(round(ti * 1000))
        if key not in lut:
            raise ValueError(f"sem_t_weights: sampled t {ti:.4f} is not on the grid {grid}")
        out.append(lut[key])
    return torch.tensor(out, device=t.device, dtype=torch.float32)


def _sample_timesteps(
    grid: list[int],
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Per-EXAMPLE timestep drawn uniformly from `grid`, returned as t in [0,1], shape (B,).

    Flow matching trains over the whole noise schedule; this harness previously trained at a
    single fixed t, so adapting at one t and then probing the 7-timestep grid left 6 of 7
    probe points off-distribution for the adapter. Sampling is over the PROBE grid rather
    than continuous U(0,1) so that every timestep the probes read is trained on, and the
    train/probe correspondence stays exact. Per-example (not per-batch) so a single optimizer
    step averages gradients across the schedule instead of chasing one t at a time.

    Returned in FLOAT32, not the latent dtype: bf16 rounds t=0.100 to 0.1001, and the
    previous code did the noising arithmetic with a Python float. Callers do the x_t math in
    fp32 and cast only the model's `timesteps` argument, so precision matches the old path.
    """
    if weights is not None:
        # Non-uniform grid sampling (align.t_weights): the high-noise timesteps the matrix
        # probes get some training without the semantic ones' share.
        if len(weights) != len(grid) or any(not (w > 0) for w in weights):
            raise ValueError(f"t weights {weights} must be one positive weight per grid point {grid}")
        w = torch.tensor(list(weights), device=device, dtype=torch.float32)
        idx = torch.multinomial(w, batch_size, replacement=True, generator=generator)
    else:
        idx = torch.randint(len(grid), (batch_size,), device=device, generator=generator)
    table = torch.tensor([x / 1000.0 for x in grid], device=device, dtype=torch.float32)
    return table[idx]


def _fine_tune_diffusion_microbatch(
    cfg,
    featurizer: Featurizer4Eval,
    timestep_grid: list[int],
    batch: dict,
    device: torch.device,
    grad_accum_steps: int,
    mask_token: torch.nn.Parameter,
    mask_ratio: float,
    decoder: MIMDecoder,
    capture: _FeatureCapture | None,
    probe_head: torch.nn.Module | None = None,
    supervised: bool = False,
    freeze_backbone: bool = False,
    align: AL.AlignState | None = None,
    rng: dict | None = None,
) -> dict:
    """
    One MIM training microbatch. Caller handles gradient accumulation, optimizer step,
    logging, and checkpointing. mask_token is applied to img tokens before
    the Flux forward so attention can propagate context into masked positions.
    Flow loss is computed only at unmasked positions.

    Returns batch_metrics dict (train/mim_loss and diagnostics).
    Input-masked MIM microbatch.

    align (AlignConfig.arm set): the block-k tokens stay attached, the semantic loss is computed
    against the frozen target and BOTH gradients go to align.accumulate() instead of .backward();
    the caller combines them at the accumulation boundary (align.apply). rng: explicit
    generators {"t", "eps", "ae"} so every arm and a resumed run draw the same timesteps, noise
    and VAE posterior samples (None = the global RNG, the pre-alignment behaviour).
    probe_head None (a label-free corpus such as git10m): no probe loss at all.
    """
    model = featurizer.model
    vae = featurizer.ae
    model.train()
    vae.eval()

    imgs = batch["img"].to(device)
    latents = _encode_latents(vae, imgs, rng)

    # Per-example t over the probe grid; (B,) for the model, (B,1,1,1) to broadcast over CHW.
    t = _sample_timesteps(
        timestep_grid,
        imgs.shape[0],
        device,
        generator=rng["t"] if rng else None,
        weights=cfg.align.t_weights if align is not None else None,
    )
    t_b = t.view(-1, *([1] * (latents.dim() - 1)))

    noise = (
        torch.randn(latents.shape, generator=rng["eps"], device=device, dtype=latents.dtype)
        if rng
        else torch.randn_like(latents).to(device)
    )
    latents_noisy = (t_b * noise.float() + (1.0 - t_b) * latents.float()).to(latents.dtype)

    img, img_ids = prepare(img=latents_noisy)
    img = img.to(device, dtype=latents.dtype)
    img_ids = img_ids.to(device)

    txt, txt_ids, y = _expand_null_embeddings(
        featurizer, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
    )
    guidance_vec = torch.full((imgs.shape[0],), cfg.guidance_scale, device=device, dtype=latents.dtype)

    # Mask input tokens before the Flux forward.
    if mask_ratio > 0:
        img_masked, mask = _random_masking(img, mask_token, mask_ratio)
    else:
        img_masked, mask = img, _no_mask(img)

    # One trunk pass returns the velocity AND block-k features. detach_features governs
    # whether probe CE can reach the backbone: True (default) keeps adaptation LABEL-FREE
    # (the head is a passive readout); False is the explicit supervised_finetune arm.
    v_pred, up_ft = model.forward_velocity_feat(
        img=img_masked,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=y,
        timesteps=t.to(latents.dtype),
        ft_indices=[cfg.k],
        guidance=guidance_vec,
        detach_features=not (supervised or align is not None),
    )
    # (B, L, C) block input; up_ft[1] is the block's adaLN mod for double AND single blocks
    # (double blocks changed definition 2026-10-05: block input + img_mod1, see layers.py)
    feat_tokens = up_ft[0]
    feat_pooled = feat_tokens.float().mean(dim=1)  # (B, C) spatial mean, matches POOLING

    v_target, _ = prepare(noise - latents)
    v_target = v_target.to(device=device, dtype=v_pred.dtype)

    # Flow loss: unmasked positions only — masked positions have no reliable target.
    per_token_flow = ((v_pred.float() - v_target.float()) ** 2).mean(dim=-1)
    flow_loss = (per_token_flow * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)

    # Concurrent linear probe (2026-09-13 protocol realignment): trained on the SAME
    # batches, at each batch's sampled t, Scale-MAE-style (probe on full train split,
    # eval on the official test split -- no complement machinery).
    if probe_head is not None:
        labels = batch["label"].to(device)
        # Align mode keeps feat_tokens attached for the SEM loss, and AlignState.two_grads frees
        # that graph; the head-only CE (M1 rule) must therefore read a detached copy or its
        # backward below walks the freed graph ("backward through the graph a second time").
        logits = probe_head(feat_pooled.detach() if align is not None else feat_pooled)
        probe_ce = F.cross_entropy(logits, labels)
        probe_acc = float((logits.argmax(dim=1) == labels).float().mean().cpu())
    else:
        probe_ce, probe_acc = None, float("nan")

    mim_metrics: dict = {}
    if cfg.mim_loss_weight != 0:
        # MIM head: decode directly from captured features. Skipped entirely at weight 0 --
        # multiplying by zero would still run the decoder forward and backward every
        # microbatch for a term that cannot affect the update.
        if capture is None or capture.features is None:
            raise RuntimeError(
                "MIM is enabled but no feature capture exists -- forward hooks never fire "
                "on this Flux (blocks are invoked via .forward_feat, not __call__); wire "
                "features through _forward_trunk ft_indices, non-detached."
            )
        pred_clean = decoder(capture.features)
        target_clean, _ = prepare(latents)
        target_clean = target_clean.to(device=device, dtype=pred_clean.dtype)
        per_token_mim = ((pred_clean - target_clean) ** 2).mean(dim=-1)
        mim_loss = (per_token_mim * mask).sum() / mask.sum().clamp(min=1)
        total_loss = flow_loss + cfg.mim_loss_weight * mim_loss
        mim_metrics = _mim_metrics(pred_clean, target_clean, mask, prefix="train")
    else:
        total_loss = flow_loss

    align_metrics: dict = {}
    # CE is scaled by 1/grad_accum_steps in EVERY arm and the head is stepped by the
    # caller at the accumulation boundary in EVERY arm, so the head's update rule is
    # identical across the three modes (rule 8; 2026-09-16 review, M1).
    if align is not None:
        # Semantic alignment: the two LoRA gradients are kept apart (AlignState.accumulate)
        # and combined by the arm's rule at the boundary; nothing touches .grad here.
        if supervised or freeze_backbone:
            raise ValueError("align.arm is incompatible with supervised_finetune / freeze_backbone")
        align_loss, align_loss_unw = align.align_loss(
            feat_tokens,
            imgs,
            up_ft[1],
            w_img=_sem_t_weights_for(t, timestep_grid, cfg.align.sem_t_weights, cfg.align.t_weights),
            return_unweighted=True,
        )
        align.accumulate(total_loss / grad_accum_steps, align_loss / grad_accum_steps)
        # train/align_loss = the UNWEIGHTED -cos (stats.npz step/align_loss, the [train] print, the
        # curves): comparable across arms and with val/align_loss_mean. The sem_t_weights-weighted
        # loss that is actually applied is logged beside it (identical when the feature is off).
        align_metrics["train/align_loss"] = float(align_loss_unw.detach().cpu())
        align_metrics["train/align_loss_applied"] = float(align_loss.detach().cpu())
    elif supervised:
        # Single graph: CE reaches the LoRA through non-detached features.
        ((total_loss + probe_ce) / grad_accum_steps).backward()
    elif freeze_backbone:
        # Probe-only arm: no backbone gradients at all (the flow loss is logged as a
        # diagnostic but never backpropagated).
        (probe_ce / grad_accum_steps).backward()
    else:
        # Label-free adaptation (default): flow gradient into the LoRA; CE touches ONLY
        # the head (features are detached).
        (total_loss / grad_accum_steps).backward()
    if probe_ce is not None and not supervised and not freeze_backbone:
        (probe_ce / grad_accum_steps).backward()

    return {
        **_flow_metrics(v_pred, v_target, mask=mask, prefix="train"),
        **mim_metrics,
        **align_metrics,
        "train/total_loss": float(total_loss.detach().cpu()),
        "train/probe_ce": float(probe_ce.detach().cpu()) if probe_ce is not None else float("nan"),
        "train/probe_acc": probe_acc,
        # Loss magnitude varies strongly with t, so the mean t of the batch is needed to read
        # the training curve at all once t is sampled rather than fixed.
        "train/t_mean": float(t.float().mean().detach().cpu()),
        # Per-row t and index position for the per-microbatch log (AlignStats.log_micro); popped by
        # the loop before the scalar metrics are averaged.
        "_t": t.detach().float().cpu().numpy(),
        "_index": np.asarray(batch["index"]) if "index" in batch else None,
    }


@torch.no_grad()
def _validate_diffusion(
    cfg,
    featurizer: Featurizer4Eval,
    timestep_grid: list[int],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_token: torch.nn.Parameter,
    mask_ratio: float,
    decoder: MIMDecoder,
    capture: _FeatureCapture | None,
    probe_head: torch.nn.Module | None = None,
    align: AL.AlignState | None = None,
) -> dict:
    model = featurizer.model
    vae = featurizer.ae
    model.eval()
    vae.eval()
    if probe_head is not None:
        probe_head.eval()
    if align is not None:
        align.projector.eval()

    val_metrics: dict = {}
    per_t_flow: dict[int, list[float]] = {t_nom: [] for t_nom in timestep_grid}
    per_t_probe: dict[int, list[float]] = {t_nom: [] for t_nom in timestep_grid}
    per_t_align: dict[int, list[float]] = {t_nom: [] for t_nom in timestep_grid}
    # The seeded eps below is NOT enough for deterministic validation: ae.encode samples the
    # VAE posterior from the GLOBAL rng (DiagonalGaussian, sample=True), whose state depends
    # on how many draws training consumed. Validate on the posterior MEAN so val_loss -- and
    # therefore best-checkpoint selection -- is a function of the weights alone. Training
    # keeps sampling; restored in the finally below.
    reg_sample_prev = vae.reg.sample
    vae.reg.sample = False

    # Deterministic across evaluations: eps from a fixed seed, consumed in a fixed order
    # (unshuffled loader x fixed grid order). Sampling t or eps freshly here would make
    # val_loss -- and therefore best-checkpoint selection -- a function of the draw.
    try:
        noise_gen = torch.Generator(device=device).manual_seed(VAL_NOISE_SEED)
        for batch in dataloader:
            imgs = batch["img"].to(device)
            latents = vae.encode(imgs).to(torch.bfloat16)

            # EVERY image is evaluated at EVERY grid timestep. Assigning one t per batch
            # round-robin would tie each val image permanently to one timestep (the loader
            # order is fixed), so the per-t diagnostics would compare DISJOINT image subsets
            # -- timestep effect confounded with subset difficulty -- and val/total_loss
            # would weight timesteps unevenly whenever len(loader) % len(grid) != 0. The
            # len(grid)-fold cost is bounded by VAL_MAX_IMAGES.
            for t_nom in timestep_grid:
                t = torch.full((imgs.shape[0],), t_nom / 1000.0, device=device, dtype=torch.float32)
                t_b = t.view(-1, *([1] * (latents.dim() - 1)))

                noise = torch.randn(latents.shape, generator=noise_gen, device=device, dtype=latents.dtype)
                latents_noisy = (t_b * noise.float() + (1.0 - t_b) * latents.float()).to(latents.dtype)

                img, img_ids = prepare(img=latents_noisy)
                img = img.to(device, dtype=latents.dtype)
                img_ids = img_ids.to(device)

                txt, txt_ids, y = _expand_null_embeddings(
                    featurizer, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
                )
                guidance_vec = torch.full(
                    (imgs.shape[0],), cfg.guidance_scale, device=device, dtype=latents.dtype
                )

                if mask_ratio > 0:
                    img_masked, mask = _random_masking(img, mask_token, mask_ratio)
                else:
                    img_masked, mask = img, _no_mask(img)

                v_pred, up_ft = model.forward_velocity_feat(
                    img=img_masked,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=y,
                    timesteps=t.to(latents.dtype),
                    ft_indices=[cfg.k],
                    guidance=guidance_vec,
                )
                if probe_head is not None:
                    feat_pooled = up_ft[0].float().mean(dim=1)
                    preds = probe_head(feat_pooled).argmax(dim=1)
                    # Per-image, not per-batch means: a batch size that does not divide
                    # the val subset would otherwise overweight the last partial batch.
                    per_t_probe[t_nom].extend((preds == batch["label"].to(device)).float().cpu().tolist())
                if align is not None:
                    # (batch mean, batch size): aggregated per IMAGE below, so a val subset that
                    # is not a multiple of the batch size does not overweight the last batch.
                    per_t_align[t_nom].append(
                        (float(align.align_loss(up_ft[0], imgs, up_ft[1]).detach().cpu()), int(imgs.shape[0]))
                    )

                v_target, _ = prepare(noise - latents)
                v_target = v_target.to(device=device, dtype=v_pred.dtype)

                per_token_flow = ((v_pred.float() - v_target.float()) ** 2).mean(dim=-1)
                flow_loss = (per_token_flow * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)
                per_t_flow[t_nom].append(float(flow_loss.detach().cpu()))

                mim_metrics: dict = {}
                if cfg.mim_loss_weight != 0:
                    if capture is None or capture.features is None:
                        raise RuntimeError("MIM enabled but no feature capture -- see microbatch note.")
                    pred_clean = decoder(capture.features)
                    target_clean, _ = prepare(latents)
                    target_clean = target_clean.to(device=device, dtype=pred_clean.dtype)
                    per_token_mim = ((pred_clean - target_clean) ** 2).mean(dim=-1)
                    mim_loss = (per_token_mim * mask).sum() / mask.sum().clamp(min=1)
                    total = flow_loss + cfg.mim_loss_weight * mim_loss
                    mim_metrics = _mim_metrics(pred_clean, target_clean, mask, prefix="val")
                else:
                    total = flow_loss

                batch_metrics = {
                    **_flow_metrics(v_pred, v_target, mask=mask, prefix="val"),
                    **mim_metrics,
                    "val/total_loss": float(total.detach().cpu()),
                }
                for k, v in batch_metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)

        for k in val_metrics:
            val_metrics[k] = float(np.mean(np.array(val_metrics[k])))

        # Per-timestep val flow loss: with one t the curve is a scalar, but across the grid it is
        # the diagnostic that shows WHERE on the schedule adaptation is happening.
        for t_nom, vals in per_t_flow.items():
            if vals:
                val_metrics[f"val/flow_loss_t{t_nom}"] = float(np.mean(np.array(vals)))
        # Per-timestep probe accuracy on the strided val subset: the live per-t curve that
        # replaces the old post-training extraction table.
        if probe_head is not None:
            accs = []
            for t_nom, vals in per_t_probe.items():
                if vals:
                    a = float(np.mean(np.array(vals)))
                    val_metrics[f"val/probe_acc_t{t_nom}"] = a
                    accs.append(a)
            if accs:
                val_metrics["val/probe_acc_mean"] = float(np.mean(accs))
        if align is not None:
            # Held-out semantic loss (= -cos to the frozen target) per t, next to the FM loss.
            for t_nom, vals in per_t_align.items():
                n_img = sum(n for _, n in vals)
                val_metrics[f"val/align_loss_t{t_nom}"] = (
                    float(sum(v * n for v, n in vals) / n_img) if n_img else float("nan")
                )
            val_metrics["val/align_loss_mean"] = float(
                np.mean([val_metrics[f"val/align_loss_t{t}"] for t in timestep_grid])
            )

        return val_metrics
    finally:
        vae.reg.sample = reg_sample_prev


# ---------------------------------------------------------------------------------------------
# Semantic-alignment helpers (AlignConfig.arm set). The forward pieces live here, next to the
# microbatch they mirror; the gradient bookkeeping is tasks/align.py.
def _encode_latents(vae, imgs: torch.Tensor, rng: dict | None) -> torch.Tensor:
    """VAE posterior sample as bf16 latents. With rng["ae"] (a CUDA RNG state) the posterior
    noise comes from that stream and the stream is advanced, so a resumed run or any extra
    extraction cannot desynchronise the training latents between arms."""
    with torch.no_grad():
        if not rng or "ae" not in rng:
            return vae.encode(imgs).to(torch.bfloat16)
        dev = imgs.device
        saved = torch.cuda.get_rng_state(dev)
        torch.cuda.set_rng_state(rng["ae"], dev)
        try:
            return vae.encode(imgs).to(torch.bfloat16)
        finally:
            rng["ae"] = torch.cuda.get_rng_state(dev)
            torch.cuda.set_rng_state(saved, dev)


def _align_losses(cfg, featurizer: Featurizer4Eval, align: AL.AlignState, imgs, latents, t, noise):
    """(flow_loss, align_loss) with gradients, the microbatch's forward without masking / probe:
    used by the measurement passes. t: (B,) fp32 in [0, 1]; noise like latents."""
    device = imgs.device
    t_b = t.view(-1, *([1] * (latents.dim() - 1)))
    x_t = (t_b * noise.float() + (1.0 - t_b) * latents.float()).to(latents.dtype)
    img, img_ids = prepare(img=x_t)
    img, img_ids = img.to(device, dtype=latents.dtype), img_ids.to(device)
    txt, txt_ids, y = _expand_null_embeddings(
        featurizer, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
    )
    guidance = torch.full((imgs.shape[0],), cfg.guidance_scale, device=device, dtype=latents.dtype)
    v_pred, up_ft = featurizer.model.forward_velocity_feat(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=y,
        timesteps=t.to(latents.dtype),
        ft_indices=[cfg.k],
        guidance=guidance,
        detach_features=False,
    )
    v_target, _ = prepare(noise - latents)
    v_target = v_target.to(device=device, dtype=v_pred.dtype)
    flow_loss = ((v_pred.float() - v_target.float()) ** 2).mean()
    return flow_loss, align.align_loss(up_ft[0], imgs, up_ft[1])


@torch.no_grad()
def _feat_only(cfg, featurizer: Featurizer4Eval, latents, t, noise):
    """(block-k input tokens, the block's modulation) only (early exit after block k, no velocity):
    the projector warmup."""
    device = latents.device
    t_b = t.view(-1, *([1] * (latents.dim() - 1)))
    x_t = (t_b * noise.float() + (1.0 - t_b) * latents.float()).to(latents.dtype)
    img, img_ids = prepare(img=x_t)
    img, img_ids = img.to(device, dtype=latents.dtype), img_ids.to(device)
    txt, txt_ids, y = _expand_null_embeddings(
        featurizer, batch_size=latents.shape[0], device=device, dtype=latents.dtype
    )
    guidance = torch.full((latents.shape[0],), cfg.guidance_scale, device=device, dtype=latents.dtype)
    up_ft = featurizer.model.forward_feat(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=y,
        timesteps=t.to(latents.dtype),
        ft_indices=[cfg.k],
        guidance=guidance,
    )
    return up_ft[0], up_ft[1]


@torch.no_grad()
def _feat_stats(
    cfg, featurizer: Featurizer4Eval, flux, batches, meas_t, seed: int, discard, tag: str
) -> dict:
    """Per-block feature statistics at the measurement timesteps (align.feat_stats, logging only,
    2026-10-23): for every block b (all 57) the block-input image tokens of the fixed measurement
    batches (posterior-mean latents, own seeded eps: no training RNG is consumed), with the LoRA ON
    and OFF (every LoRALinear.scaling = 0 = the pretrained model on the same input). Returns
    {t: {feat_norm (57,): mean token L2 norm (LoRA on, raw), feat_mc (57, len(discard)): mean |x| of
    the massive channels, feat_pr (57,): participation ratio of the centred tokens of one batch
    (massive channels zeroed; mean over batches; bounded by the batch's token count),
    feat_drift_cos (57,): mean per-token cosine LoRA-on vs LoRA-off (massive channels zeroed),
    feat_drift_rel (57,): ||on - off||_F / ||off||_F (same)}}."""
    from models.lora import LoRALinear

    t0 = time.time()
    loras = [m for m in flux.modules() if isinstance(m, LoRALinear)]
    blocks = list(range(NUM_BLOCKS))
    dz = torch.tensor(list(discard), dtype=torch.long) if len(discard) else None
    out = {}

    def feats(latents, t, noise):
        t_b = t.view(-1, *([1] * (latents.dim() - 1)))
        x_t = (t_b * noise.float() + (1.0 - t_b) * latents.float()).to(latents.dtype)
        img, img_ids = prepare(img=x_t)
        img, img_ids = img.to(latents.device, dtype=latents.dtype), img_ids.to(latents.device)
        txt, txt_ids, y = _expand_null_embeddings(
            featurizer, batch_size=latents.shape[0], device=latents.device, dtype=latents.dtype
        )
        guidance = torch.full(
            (latents.shape[0],), cfg.guidance_scale, device=latents.device, dtype=latents.dtype
        )
        up = featurizer.model.forward_feat(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=t.to(latents.dtype),
            ft_indices=blocks,
            guidance=guidance,
        )
        if len(up) != 2 * len(blocks):
            raise RuntimeError(
                f"_feat_stats: forward_feat returned {len(up)} tensors, expected {2 * len(blocks)}"
            )
        return up[0::2]

    for t_nom in meas_t:
        g = torch.Generator(device=batches[0][1].device).manual_seed(seed * 1000 + t_nom + 500_000)
        acc = {k: torch.zeros(len(blocks), dtype=torch.float64) for k in ("norm", "pr", "cos", "d2", "o2")}
        acc_mc = torch.zeros(len(blocks), len(discard), dtype=torch.float64)
        n_tok, n_b = 0, 0
        for imgs, latents in batches:
            t = torch.full((latents.shape[0],), t_nom / 1000.0, device=latents.device, dtype=torch.float32)
            noise = torch.randn(latents.shape, generator=g, device=latents.device, dtype=latents.dtype)
            on = feats(latents, t, noise)
            saved = [m.scaling for m in loras]
            for m in loras:
                m.scaling = 0.0
            try:
                off = feats(latents, t, noise)
            finally:
                for m, sc in zip(loras, saved):
                    m.scaling = sc
            for i, (xa, xp) in enumerate(zip(on, off)):
                xa, xp = xa.float().flatten(0, 1), xp.float().flatten(0, 1)  # (B*L, C)
                acc["norm"][i] += float(xa.norm(dim=-1).sum())
                if dz is not None:
                    acc_mc[i] += xa[:, dz.to(xa.device)].abs().sum(0).double().cpu()
                    xa, xp = (
                        xa.index_fill(-1, dz.to(xa.device), 0.0),
                        xp.index_fill(-1, dz.to(xa.device), 0.0),
                    )
                xc = xa - xa.mean(0, keepdim=True)
                G = xc @ xc.t()
                acc["pr"][i] += float(torch.diagonal(G).sum() ** 2 / (G * G).sum().clamp_min(1e-30))
                acc["cos"][i] += float(F.cosine_similarity(xa, xp, dim=-1).sum())
                acc["d2"][i] += float(((xa - xp) ** 2).sum())
                acc["o2"][i] += float((xp**2).sum())
            n_tok += on[0].shape[0] * on[0].shape[1]
            n_b += 1
        out[t_nom] = {
            "feat_norm": (acc["norm"] / n_tok).float().numpy(),
            "feat_mc": (acc_mc / n_tok).float().numpy(),
            "feat_pr": (acc["pr"] / n_b).float().numpy(),
            "feat_drift_cos": (acc["cos"] / n_tok).float().numpy(),
            "feat_drift_rel": (acc["d2"].sqrt() / acc["o2"].sqrt().clamp_min(1e-30)).float().numpy(),
        }
    d = out[meas_t[0]]
    print(
        f"[feat-stats:{tag}] t={meas_t[0]} block 28: norm={d['feat_norm'][28]:.2f} pr={d['feat_pr'][28]:.1f} "
        f"drift_cos={d['feat_drift_cos'][28]:.4f} drift_rel={d['feat_drift_rel'][28]:.4f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return out


@torch.no_grad()
def _sample(cfg, featurizer: Featurizer4Eval, n: int, seed: int, device) -> tuple[np.ndarray, str]:
    """n fixed-seed samples from the current weights (null prompt, cfg.guidance_scale, align.gen_steps)
    as uint8 NHWC, plus the sha1 of the final packed latent (fp32)."""
    a = cfg.align
    flux, ae = featurizer.model, featurizer.ae
    flux.eval()
    H, W = cfg.img_size
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((n, 16, H // 8, W // 8), generator=gen, device=device, dtype=torch.bfloat16)
    img, img_ids = prepare(z)
    img, img_ids = img.to(device), img_ids.to(device)
    txt, txt_ids, y = _expand_null_embeddings(featurizer, batch_size=n, device=device, dtype=img.dtype)
    ts = get_schedule(a.gen_steps, img.shape[1], shift=True)
    x = denoise(flux, img, img_ids, txt, txt_ids, y, timesteps=ts, guidance=cfg.guidance_scale)
    sha = hashlib.sha1(x.float().cpu().numpy().tobytes()).hexdigest()[:12]
    x = unpack(x.float(), H, W)
    pix = ae.decode(x.to(next(ae.parameters()).dtype)).float().clamp(-1, 1)
    arr = ((pix + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return arr, sha


@torch.no_grad()
def _generate(cfg, featurizer: Featurizer4Eval, out_dir: str, tag: str, device) -> str:
    """Fixed-seed samples (align.gen_n, seed align.gen_seed) saved as PNGs + a grid. Returns the
    latent sha1: identical across arms at step 0 by construction (an executed cross-arm
    determinism check), different after."""
    a = cfg.align
    arr, sha = _sample(cfg, featurizer, a.gen_n, a.gen_seed, device)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(os.path.join(out_dir, f"{tag}_{i}.png"))
    Image.fromarray(np.concatenate(list(arr), axis=1)).save(os.path.join(out_dir, f"{tag}_grid.png"))
    print(
        f"[gen:{tag}] {a.gen_n} images, {a.gen_steps} steps, g={cfg.guidance_scale}, latent sha1={sha}",
        flush=True,
    )
    return sha


def _reference_images(ds, n: int, num_workers: int) -> np.ndarray:
    """The first n strided rows of the validation source as uint8 NHWC: the FID/KID reference,
    identical in every arm (the git10m val view is unflipped; the loader's transform is the
    training one, so real and generated images meet at the same resolution)."""
    if len(ds) < n:
        raise ValueError(f"align.fid_n={n} but the validation source has only {len(ds)} rows")
    loader = DataLoader(
        Subset(ds, _strided_indices(len(ds), n)), batch_size=32, shuffle=False, num_workers=num_workers
    )
    out = []
    for b in loader:
        x = b["img"].float().clamp(-1, 1)
        out.append(((x + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).numpy())
    arr = np.concatenate(out, axis=0)
    if arr.shape[0] != n:
        raise RuntimeError(f"reference set has {arr.shape[0]} rows, expected {n}")
    return arr


@torch.no_grad()
def _gen_metrics(
    cfg,
    featurizer: Featurizer4Eval,
    ref: np.ndarray,
    extractor,
    device,
    tag: str,
    save_dir: str | None = None,
) -> dict:
    """FID + KID (eval/gen_metrics.compare) of fid_n fixed-seed samples against `ref`. The samples
    (and, once, the reference) are saved as PNGs under save_dir so the metric can be recomputed
    offline (`python -m eval.gen_metrics --real <ref dir> --fake <dir>`); a failure inside the
    metric itself (sqrtm, OOM in the extractor) is RECORDED under "error" with the samples kept,
    never raised -- the training deliverables are already on disk at this point."""
    from eval import gen_metrics as GM

    a = cfg.align
    t0 = time.time()
    fake, shas = [], []
    seeds = []
    for i in range(0, a.fid_n, a.gen_n):
        n = min(a.gen_n, a.fid_n - i)
        seed = a.gen_seed + 1000 + i // a.gen_n
        arr, sha = _sample(cfg, featurizer, n, seed, device)
        fake.append(arr)
        shas.append(sha)
        seeds.append(seed)
    fake = np.concatenate(fake, axis=0)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(fake.shape[0]):
            Image.fromarray(fake[i]).save(os.path.join(save_dir, f"{i:05d}.png"))
        ref_dir = os.path.join(os.path.dirname(save_dir), "fid_ref")
        if not os.path.isdir(ref_dir):
            os.makedirs(ref_dir, exist_ok=True)
            for i in range(ref.shape[0]):
                Image.fromarray(ref[i]).save(os.path.join(ref_dir, f"{i:05d}.png"))
    base = {"tag": tag, "n": int(fake.shape[0]), "gen_steps": a.gen_steps, "guidance": cfg.guidance_scale,
            "seeds": seeds, "latent_sha1": hashlib.sha1("".join(shas).encode()).hexdigest()[:12],
            "real": GM.image_identity(ref), "fake": GM.image_identity(fake), "samples_dir": save_dir}  # fmt: skip
    try:
        res = GM.compare(ref, fake, extractor, subsets=a.fid_subsets)
    except Exception as e:  # noqa: BLE001 -- recorded, the samples are on disk for a rerun
        res = {"fid": None, "kid_mean": None, "kid_std": None, "error": f"{type(e).__name__}: {e}"}
        print(f"[gen-metrics:{tag}] FAILED (samples kept in {save_dir}): {res['error']}", flush=True)
    res = {**base, **res, "seconds": round(time.time() - t0, 1)}
    if res.get("fid") is not None:
        print(
            f"[gen-metrics:{tag}] n={res['n']} FID={res['fid']:.2f} KID={res['kid_mean']:.4f}+-{res['kid_std']:.4f} "
            f"ref sha1={res['real']['sha1'][:12]} in {res['seconds']:.0f}s",
            flush=True,
        )
    return res


def _save_resumable(path: str, rec: dict) -> None:
    """ckpt.pt written atomically (tmp + os.replace): present == complete."""
    tmp = path + ".tmp"
    torch.save(rec, tmp)
    os.replace(tmp, path)
    print(f"[ckpt] step {rec['global_step']} -> {path}", flush=True)


@register_task("finetune-diffusion")
class FinetuneDiffusionTask:
    def run(self, cfg, model: FluxModel, dataset) -> dict:
        # cfg.t may be an int (single-t, the previous behaviour) or the probe grid.
        timestep_grid = _normalize_timestep_grid(cfg.t)

        # --- pinned training constants; see the module-level block for the reasoning ---
        if cfg.guidance_scale != PINNED_GUIDANCE:
            raise ValueError(
                f"guidance_scale is pinned to {PINNED_GUIDANCE} for finetune-diffusion, got "
                f"{cfg.guidance_scale}. cfg.guidance_scale feeds BOTH training and extraction, "
                f"and every existing feature cache is g={PINNED_GUIDANCE}; training at another "
                "value adapts the model at a guidance it is never probed at. To sweep "
                "guidance, change PINNED_GUIDANCE in tasks/train_diffusion.py and re-extract."
            )
        if cfg.mim_loss_weight != PINNED_MIM_LOSS_WEIGHT:
            raise ValueError(
                f"mim_loss_weight is pinned to {PINNED_MIM_LOSS_WEIGHT} (the MIM auxiliary "
                f"objective is a dropped angle), got {cfg.mim_loss_weight}. Its decoder reads "
                "the same block features the probes read, so a nonzero weight makes the "
                "adapted-model result partly about the auxiliary head rather than about flow "
                "adaptation."
            )
        # 2026-09-13 protocol realignment: probe trains CONCURRENTLY on the full train
        # split, eval on the official test split (Scale-MAE/SatMAE style) -- the complement
        # split and post-training extraction stage are gone with the CV-inside-train-subset
        # protocol that required them. cfg.model.ensemble_size is unused by this task now
        # (no extract_features probe stage), so the old ens1 gate is gone too.
        if cfg.label_fraction != 1.0:
            # The dataset wrappers subsample the TRAIN SPLIT ITSELF under label_fraction,
            # so a "label-scarce" run would also shrink the label-free flow-adaptation
            # data, confounding label scarcity with data scarcity (2026-09-16 review,
            # M2 -- this gate existed before the protocol realignment and was lost).
            raise ValueError(
                f"label_fraction={cfg.label_fraction} is not supported by finetune-diffusion: "
                "the wrappers subsample the train split itself, which would shrink the "
                "flow-adaptation data along with the probe labels. Label-fraction sweeps "
                "need CE-label masking, which is not implemented."
            )
        if cfg.lora_checkpoint:
            # The adapter already wrapped LoRA when loading the checkpoint; this task
            # wraps again and LoRALinear raises a cryptic "base must be nn.Linear".
            # Resuming adaptation is not supported -- fail with the real reason.
            raise ValueError(
                "finetune-diffusion cannot resume from --lora-checkpoint (the model is "
                "already LoRA-wrapped at load; a second wrap is invalid). Checkpoints are "
                "for extraction/eval tasks; start adaptation runs from scratch."
            )
        # Semantic alignment (AlignConfig.arm): fm / naive / proj arms, label-free corpora.
        use_align = bool(cfg.align.arm)
        has_labels = bool(getattr(dataset, "category_list", []))
        if use_align and (cfg.freeze_backbone or cfg.supervised_finetune):
            raise ValueError("align.arm is incompatible with freeze_backbone / supervised_finetune")
        if not has_labels and not use_align:
            raise ValueError(
                f"dataset {cfg.dataset.name!r} has no labels: finetune-diffusion on a label-free corpus "
                "needs --align.arm fm|naive|proj (the FM-only arm is align.arm=fm)"
            )
        if cfg.align.resume and not use_align:
            raise ValueError("align.resume needs align.arm")
        if cfg.align.split_eval and (not use_align or cfg.align.resume):
            raise ValueError("align.split_eval needs align.arm and excludes align.resume")
        if cfg.align.virtual_steps and (not use_align or cfg.align.resume or cfg.align.split_eval):
            raise ValueError(
                "align.virtual_steps needs align.arm and excludes align.resume and align.split_eval"
            )
        if cfg.align.grad_geometry and (
            not use_align or cfg.align.resume or cfg.align.split_eval or cfg.align.virtual_steps
        ):
            raise ValueError(
                "align.grad_geometry needs align.arm and excludes align.resume, align.split_eval and align.virtual_steps"
            )
        if cfg.align.vs_grad_batches <= 0:
            raise ValueError("align.vs_grad_batches must be positive")
        use_mim = cfg.mim_loss_weight != 0
        if use_mim and cfg.freeze_backbone:
            # Unreachable while PINNED_MIM_LOSS_WEIGHT == 0, kept so unpinning cannot revive
            # it: the frozen arm builds no optimizer, so the MIM decoder / mask_token would
            # sit at random init while train/mim_loss is logged as if it were optimized.
            raise ValueError("freeze_backbone with mim_loss_weight != 0: the MIM decoder would never train")
        if not use_mim and cfg.mask_ratio > 0:
            raise ValueError(
                f"mask_ratio={cfg.mask_ratio} with mim_loss_weight=0 is incoherent: input "
                "masking exists to serve the MIM objective, and with MIM off it only hides "
                f"{cfg.mask_ratio:.0%} of the flow target behind a mask token that now "
                "receives no gradient. Set mask_ratio=0 for a pure flow-matching finetune."
            )
        mode = (
            "FROZEN backbone (probe-only)"
            if cfg.freeze_backbone
            else (
                "SUPERVISED finetune (CE -> LoRA)"
                if cfg.supervised_finetune
                else (
                    f"label-free flow adaptation + semantic alignment [{cfg.align.arm}, w={cfg.align.weight}, "
                    f"target={cfg.align.target}]"
                    if use_align
                    else "label-free flow adaptation"
                )
            )
        )
        print(
            f"finetune-diffusion [{mode}]: timesteps={timestep_grid}  "
            f"guidance={cfg.guidance_scale}  mim_loss_weight={cfg.mim_loss_weight}  "
            f"mask_ratio={cfg.mask_ratio}",
            flush=True,
        )
        # Unwrap registered adapter to get the raw Featurizer4Eval.
        featurizer_model: Featurizer4Eval = model._inner
        device = torch.device(cfg.device)

        best_val_loss = float("inf")

        # Move Flux and VAE to GPU; VAE stays in eval and its weights stay frozen.
        flux = featurizer_model.model
        ae = featurizer_model.ae
        flux.to(device)
        ae.to(device)

        # MIM components: mask_token replaces FLUX input latent tokens
        # before the FLUX forward, so trainable attention cannot directly see masked tokens.
        # MIM components exist only when the MIM objective runs; with it pinned off they
        # would be dead parameters carried through every checkpoint.
        if use_mim:
            mask_token = torch.nn.Parameter(torch.zeros(1, 1, flux.in_channels, device=device))
            decoder = MIMDecoder(flux.hidden_size, flux.in_channels).to(device=device, dtype=torch.bfloat16)
        else:
            mask_token, decoder = None, None

        # NO feature hook. _forward_trunk invokes blocks via .forward_feat(), a plain
        # method call, so register_forward_hook NEVER fires on this Flux implementation --
        # the smoke test died on "Feature capture returned None" at the first microbatch
        # (2026-09-10). The training loop only ever needed block features for the MIM
        # decoder, which is pinned off; the post-training probe extracts features through
        # extract_features (ft_indices path), which needs no hook. If MIM is re-enabled,
        # take features from _forward_trunk's ft_indices NON-detached (up_ft is detached
        # today, which would also silently cut the LoRA gradient MIM relies on).
        capture = None

        # Freeze base weights; insert LoRA adapters only when the backbone trains.
        for param in flux.parameters():
            param.requires_grad = False

        if not cfg.freeze_backbone:
            lora_wrap_flux(
                flux,
                lora_block_indices(flux),
                cfg.lora_rank,
                cfg.lora_alpha,
                cfg.lora_dropout,
                wrap_o=cfg.wrap_output,
                # Alignment: fp32 LoRA (tasks/align.py docstring); otherwise the base dtype as before.
                param_dtype=torch.float32 if use_align else None,
            )
            # Only the LoRA A/B matrices should be trainable.
            for name, param in flux.named_parameters():
                if name.endswith(".A") or name.endswith(".B"):
                    param.requires_grad = True

        trainable_params = [p for p in flux.parameters() if p.requires_grad]
        align: AL.AlignState | None = None
        if use_align:
            # --no-cd keeps the massive channels in the aligned features too (FluxModel.extract
            # honours cfg.cd the same way; review 2026-10-15). The queued arms pass --cd.
            align = AL.AlignState(
                flux, cfg.align, cfg.k, device, discard=list(cfg.discard_channels) if cfg.cd else []
            )
            trainable_params = list(align.params)  # block order == the order the fp32 gate checks
        if use_mim:
            trainable_params.append(mask_token)
            trainable_params.extend(decoder.parameters())
        # With MIM off, mask_token and the decoder receive no gradient; handing them to the
        # optimizer anyway would leave dead parameters in the checkpoint's param groups.

        if not cfg.freeze_backbone and len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found in FLUX after LoRA wrapping. "
                "Please check configuration and LoRA wrapping logic."
            )

        # Concurrent linear probe head (float32; features are cast in the microbatch).
        # The head's optimization regime is IDENTICAL across the three arms (rule 8): its
        # own AdamW at clf_lr with EXPLICIT weight_decay=0.0, CE scaled by
        # 1/grad_accum_steps every microbatch, stepped at the accumulation boundary, no
        # warmup, no grad clipping. Folding the head into the main optimizer in the
        # supervised arm (the previous code) gave it finetune_lr + lora_wd + warmup +
        # clipping there but clf_lr + AdamW-default wd + per-microbatch steps elsewhere,
        # so supervised-vs-label-free deltas partly measured head-optimizer differences
        # rather than gradient routing (2026-09-16 review, M1). In supervised mode the CE
        # still reaches the LoRA through the shared backward; only the head's own UPDATE
        # rule is pinned equal.
        if has_labels:
            num_classes = len(dataset.category_list)
            probe_head = torch.nn.Linear(flux.hidden_size, num_classes).to(device=device, dtype=torch.float32)
            probe_optimizer = torch.optim.AdamW(probe_head.parameters(), lr=cfg.clf_lr, weight_decay=0.0)
        else:  # label-free corpus (git10m): no probe at all; the downstream probe is run.py --task extract
            num_classes, probe_head, probe_optimizer = 0, None, None

        if cfg.freeze_backbone:
            optimizer = None
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=cfg.finetune_lr, weight_decay=cfg.lora_wd)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=max(1, cfg.warmup_steps),
            )

        loaders = dataset.get_data(cfg)
        train_loader = loaders["train"]
        test_loader = loaders.get("test")  # absent for a label-free corpus

        # Strided validation subset (see VAL_MAX_IMAGES). Deterministic, class-covering,
        # identical at every evaluation. Drawn from the dataset's REAL val split when the
        # wrapper provides one (m_eurosat does; GEO-Bench publishes it) so even diagnostic
        # validation never reads the test split; falls back to the test split for wrappers
        # without one (eurosat/resisc45 -- the pre-2026-09-17 behaviour, unchanged).
        _val_source = loaders.get("val", test_loader)
        if _val_source is None:
            raise ValueError(f"dataset {cfg.dataset.name!r} provides neither a val nor a test loader")
        _test_ds = _val_source.dataset
        val_loader = DataLoader(
            torch.utils.data.Subset(_test_ds, _strided_indices(len(_test_ds), VAL_MAX_IMAGES)),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # (The old complement split lived here. With the 2026-09-13 protocol -- concurrent
        # probe on the full train split, evaluation on the OFFICIAL test split -- the
        # train/probe separation is the standard train/test boundary and needs no subset
        # machinery. The n5000 subset extraction remains, unchanged, for the mechanism
        # experiments in experiments/.)

        history = []
        global_step = 0
        micro_step = 0

        grad_accum_steps = cfg.gradient_accumulation_steps if cfg.use_gradient_accumulation else 1
        log_train_steps = cfg.log_train_steps
        log_val_steps = cfg.log_val_steps

        accum_metrics: dict = {}
        last_train_metrics: dict = {}
        last_val_metrics: dict = {}

        # ---- semantic alignment: explicit RNG streams, statistics, measurement batches, step-0
        # passes, or the resume of all of it from ckpt.pt (tasks/align.py).
        S = cfg.max_train_steps
        rng: dict | None = None
        stats: AL.AlignStats | None = None
        meas: dict = {}
        val_rows: list[dict] = []
        sha_init = None
        gen_metrics: dict = {}
        fid_ref, fid_extractor = None, None
        gate_path = os.path.join(cfg.save_dir, "conflict_gate.txt")
        ckpt_path = os.path.join(cfg.save_dir, "ckpt.pt")
        gen_dir = os.path.join(cfg.save_dir, "gen")
        train_data = getattr(dataset, "describe", None)  # git10m: index provenance (manifest sha256, counts)

        def _fail(msg: str):
            with open(gate_path, "w") as f:
                f.write(f"projection verifier / fp32 gate / run: FAIL -- {msg}\n")
            raise RuntimeError(msg)

        if use_align:
            GP.check_fp32_params(align.params, optimizer, where="init")
            rng = {
                "t": torch.Generator(device=device).manual_seed(cfg.seed + 21),
                "eps": torch.Generator(device=device).manual_seed(cfg.seed + 22),
                # The VAE posterior noise gets its own stream (saved in the checkpoint).
                "ae": torch.Generator(device=device).manual_seed(cfg.seed + 23).get_state(),
            }
            stats = AL.AlignStats(S, align.n_rows, grad_accum_steps, cfg.batch_size)
            # Fixed measurement batches: unflipped rows, posterior-MEAN latents, so every pass
            # (init, final, every meas_every steps) and every arm differentiates the same inputs.
            # split_eval reads none of the measurement / FID machinery: skip the VAE encodes below
            # and the Inception load (review 2026-10-24; meas_n = 0 -> empty batch list).
            meas_src = getattr(dataset, "meas_src", train_loader.dataset)
            # virtual_steps differentiates on vs_grad_batches of these fixed training batches.
            meas_n = (
                0
                if cfg.align.split_eval
                else cfg.align.vs_grad_batches * cfg.batch_size
                if (cfg.align.virtual_steps or cfg.align.grad_geometry)
                else cfg.align.meas_batches * cfg.batch_size
            )
            meas_loader = DataLoader(
                Subset(meas_src, _strided_indices(len(meas_src), meas_n) if meas_n else []),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
            reg_prev = ae.reg.sample
            ae.reg.sample = False
            try:
                with torch.no_grad():
                    meas_batches = [
                        (b["img"].to(device), ae.encode(b["img"].to(device)).to(torch.bfloat16))
                        for b in meas_loader
                    ]
            finally:
                ae.reg.sample = reg_prev

            def measure(tag: str) -> dict:
                flux.eval()
                align.projector.eval()
                out = align.measure(
                    meas_batches,
                    lambda imgs, latents, t, noise: _align_losses(
                        cfg, featurizer_model, align, imgs, latents, t, noise
                    ),
                    cfg.align.meas_t,
                    cfg.seed,
                    tag,
                )
                if cfg.align.feat_stats:
                    fs = _feat_stats(
                        cfg,
                        featurizer_model,
                        flux,
                        meas_batches,
                        cfg.align.meas_t,
                        cfg.seed,
                        align.discard,
                        tag,
                    )
                    for t_nom, d in fs.items():
                        out[t_nom].update(d)
                flux.train()
                align.projector.train()
                return out

            def resumable_record() -> dict:
                return {
                    "global_step": global_step,
                    "identity": AL.resume_identity(cfg, train_data),
                    "cfg": to_jsonable(cfg),
                    "lora": [p.detach().cpu() for p in align.params],
                    "align": align.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "probe_head": probe_head.state_dict() if probe_head is not None else None,
                    "probe_optimizer": probe_optimizer.state_dict() if probe_optimizer is not None else None,
                    "rng": {
                        "t": rng["t"].get_state().cpu(),
                        "eps": rng["eps"].get_state().cpu(),
                        "ae": rng["ae"].cpu(),
                    },
                    "stats": stats.state(),
                    "meas": meas,
                    "val_rows": val_rows,
                    "history": history,
                    "sha_init": sha_init,
                    "gen_metrics": gen_metrics,
                    "best_val_loss": best_val_loss,
                }

            print(f"[align] {json.dumps(align.describe())}", flush=True)
            if train_data is not None:
                print(f"[data] {json.dumps(train_data)}", flush=True)
            if cfg.align.fid_n and not (
                cfg.align.split_eval or cfg.align.virtual_steps or cfg.align.grad_geometry
            ):
                # Reference set + Inception extractor up front: a missing weight file or a short
                # validation source fails here, not after the training budget.
                from eval import gen_metrics as GM

                fid_ref = _reference_images(_test_ds, cfg.align.fid_n, cfg.num_workers)
                fid_extractor = GM.InceptionFeatures(device)
                print(
                    f"[gen-metrics] reference {GM.image_identity(fid_ref)} extractor {fid_extractor.name}",
                    flush=True,
                )

        if cfg.align.resume:
            # Everything the interrupted run had: fp32 LoRA, projector + its optimizer, LoRA
            # optimizer + scheduler, the per-step statistics, measurement / validation history and
            # the t / eps / VAE-posterior generators, so the continued run draws what the original
            # would have; the sampler is re-positioned at the consumed microbatch count.
            ck = torch.load(cfg.align.resume, map_location="cpu", weights_only=False)
            want, have = AL.resume_identity(cfg, train_data), ck.get("identity")
            if have != want:
                diff = {
                    k: (have.get(k) if have else None, want[k])
                    for k in want
                    if not have or have.get(k) != want[k]
                }
                raise SystemExit(
                    f"{cfg.align.resume}: resume identity differs (checkpoint, this run): {diff}"
                )
            if not hasattr(dataset, "train_loader"):
                raise SystemExit(
                    f"resume needs a dataset with a positioned train_loader (git10m), not {cfg.dataset.name}"
                )
            with torch.no_grad():
                for p, v in zip(align.params, ck["lora"]):
                    p.copy_(v.to(p.device))
            align.load_state_dict(ck["align"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            if probe_head is not None:
                probe_head.load_state_dict(ck["probe_head"])
                probe_optimizer.load_state_dict(ck["probe_optimizer"])
            rng["t"].set_state(ck["rng"]["t"])
            rng["eps"].set_state(ck["rng"]["eps"])
            rng["ae"] = ck["rng"]["ae"]
            stats = AL.AlignStats.from_state(ck["stats"], S, align.n_rows, grad_accum_steps, cfg.batch_size)
            meas, val_rows, history, sha_init = ck["meas"], ck["val_rows"], ck["history"], ck["sha_init"]
            gen_metrics = ck.get("gen_metrics", {})
            global_step, best_val_loss = int(ck["global_step"]), float(ck["best_val_loss"])
            GP.check_fp32_params(align.params, optimizer, where="resume")
            train_loader = dataset.train_loader(cfg, global_step * grad_accum_steps)
            print(f"[resume] {cfg.align.resume}: continuing at step {global_step}/{S}", flush=True)
        elif use_align and not (cfg.align.split_eval or cfg.align.virtual_steps or cfg.align.grad_geometry):
            if not cfg.align.no_gen:
                sha_init = _generate(cfg, featurizer_model, gen_dir, "step0", device)
            if cfg.align.fid_n and cfg.align.fid_step0:
                gen_metrics["step0"] = _gen_metrics(cfg, featurizer_model, fid_ref, fid_extractor, device, "step0",
                                                    save_dir=os.path.join(gen_dir, "fid_step0"))  # fmt: skip
            if cfg.align.proj_warmup_steps > 0:
                # Projector-only phase, LoRA frozen, identical in every arm (own seeded loader and
                # t / eps streams; the VAE stream is the training one, advanced equally in every arm).
                warm_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(cfg.seed + 1),
                    drop_last=True,
                )
                tg = torch.Generator(device=device).manual_seed(cfg.seed + 11)
                eg = torch.Generator(device=device).manual_seed(cfg.seed + 12)

                def warm_batches():
                    it = _cycle_loader(warm_loader)
                    for _ in range(cfg.align.proj_warmup_steps):
                        imgs = next(it)["img"].to(device)
                        latents = _encode_latents(ae, imgs, rng)
                        t = _sample_timesteps(timestep_grid, imgs.shape[0], device, tg, cfg.align.t_weights)
                        noise = torch.randn(latents.shape, generator=eg, device=device, dtype=latents.dtype)
                        feat, mod = _feat_only(cfg, featurizer_model, latents, t, noise)
                        yield feat, imgs, mod

                flux.eval()
                align.warmup(warm_batches())
                flux.train()
            meas["init"] = measure("init")
            val0 = _validate_diffusion(
                cfg,
                featurizer=featurizer_model,
                timestep_grid=timestep_grid,
                dataloader=val_loader,
                device=device,
                mask_token=mask_token,
                mask_ratio=cfg.mask_ratio,
                decoder=decoder,
                capture=capture,
                probe_head=probe_head,
                align=align,
            )
            val_rows.append({"step": 0, **val0})
            wandb.log(val0, step=0)

        if cfg.align.split_eval:
            # Diagnostic only: load, validate twice, write split_eval.json, leave the run dir otherwise
            # untouched (no ckpt.pt / val.jsonl / meta.json / generation).
            _split_eval(
                cfg,
                featurizer_model,
                flux,
                align,
                timestep_grid,
                val_loader,
                device,
                mask_token,
                decoder,
                capture,
                probe_head,
            )
            return
        if cfg.align.virtual_steps:
            # Diagnostic only (see AlignConfig.virtual_steps): writes <run dir>/virtual_steps.json.
            _virtual_steps(
                cfg,
                featurizer_model,
                flux,
                align,
                timestep_grid,
                val_loader,
                meas_batches,
                device,
                mask_token,
                decoder,
                capture,
                probe_head,
            )
            return
        if cfg.align.grad_geometry:
            # Diagnostic only (see AlignConfig.grad_geometry): writes <run dir>/grad_geometry_s<step>.npz.
            _grad_geometry(cfg, featurizer_model, flux, align, meas_batches, device)
            return

        train_iter = _cycle_loader(train_loader)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if align is not None:
            flux.train()
            align.projector.train()
        t_step = time.time()

        while global_step < cfg.max_train_steps:
            batch = next(train_iter)
            if t_step is None:  # the step clock starts at its first microbatch (see step_time below)
                t_step = time.time()

            batch_metrics = _fine_tune_diffusion_microbatch(
                cfg=cfg,
                featurizer=featurizer_model,
                timestep_grid=timestep_grid,
                batch=batch,
                device=device,
                grad_accum_steps=grad_accum_steps,
                mask_token=mask_token,
                mask_ratio=cfg.mask_ratio,
                decoder=decoder,
                capture=capture,
                probe_head=probe_head,
                supervised=cfg.supervised_finetune,
                freeze_backbone=cfg.freeze_backbone,
                align=align,
                rng=rng,
            )

            mb_t, mb_rows = batch_metrics.pop("_t", None), batch_metrics.pop("_index", None)
            if align is not None and stats is not None:
                stats.log_micro(
                    global_step,
                    micro_step % grad_accum_steps,
                    batch_metrics["train/flow_mse_loss"],
                    batch_metrics.get("train/align_loss", float("nan")),
                    mb_t,
                    mb_rows if mb_rows is not None else np.full(len(mb_t), -1),
                )
            accum_metrics = _add_batch_metrics(accum_metrics, batch_metrics)
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            gn = None
            lr_used = float("nan")
            summary: dict = {}
            if optimizer is not None:
                if align is not None:
                    # Combine the two accumulated gradients by the arm's rule (one-sided surgery in
                    # the proj arm), verify, write the fp32 result into .grad, gate the dtypes.
                    try:
                        summary = align.apply(stats, global_step)
                        align.pre_step_gate(optimizer, global_step)
                    except RuntimeError as e:
                        _fail(str(e))
                    before = [p.detach().clone() for p in align.params]
                gn = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                if align is not None and not torch.isfinite(gn):
                    _fail(f"non-finite gradient norm at step {global_step + 1}")
                lr_used = scheduler.get_last_lr()[
                    0
                ]  # this step's LR; after scheduler.step() it is the next one's
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if align is not None:
                    try:
                        align.after_step(stats, global_step, before, optimizer)
                    except RuntimeError as e:
                        _fail(str(e))
                    del before
            # Head steps at the boundary in every arm, unclipped and unscheduled (M1).
            if probe_optimizer is not None:
                probe_optimizer.step()
                probe_optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_train_metrics = _average_batch_metrics(accum_metrics)
            accum_metrics = {}
            if align is not None:
                i = global_step - 1
                if not np.isfinite(step_train_metrics["train/total_loss"]) or not np.isfinite(
                    step_train_metrics["train/align_loss"]
                ):
                    _fail(f"non-finite loss at step {global_step}: {step_train_metrics}")
                sc = stats.sc
                sc["flow_loss"][i] = step_train_metrics["train/flow_mse_loss"]
                sc["align_loss"][i] = step_train_metrics["train/align_loss"]
                sc["t_mean"][i] = step_train_metrics["train/t_mean"]
                sc["grad_norm"][i] = float(gn)
                sc["lr"][i] = lr_used
                sc["step_time"][i] = time.time() - t_step
                # Restarted at the next microbatch, so the validation / measurement / checkpoint
                # blocks of this boundary are not charged to the next step (meta.json's
                # mean_step_time_s / img_per_s are training throughput; review 2026-10-15).
                t_step = None
                step_train_metrics.update({f"train/{k}": v for k, v in summary.items()})
                step_train_metrics["train/grad_norm"] = float(gn)
                if global_step % 10 == 0 or global_step == S:
                    print(
                        f"[train] step={global_step}/{S} flow={sc['flow_loss'][i]:.5f} align={sc['align_loss'][i]:.4f} "
                        f"cos<k mean={summary['cos_lt_k_mean']:.3f} frac<0={summary['frac_neg']:.2f} "
                        f"global_cos={summary['global_cos']:.3f} nproj={summary['nproj']} gnorm={float(gn):.3f} "
                        f"{sc['step_time'][i]:.1f}s {cfg.batch_size * grad_accum_steps / sc['step_time'][i]:.2f} img/s",
                        flush=True,
                    )
            last_train_metrics = step_train_metrics

            if global_step % log_train_steps == 0:
                wandb.log(last_train_metrics, step=global_step)

            if global_step % log_val_steps == 0 or global_step == cfg.max_train_steps:
                val_metrics = _validate_diffusion(
                    cfg,
                    featurizer=featurizer_model,
                    timestep_grid=timestep_grid,
                    dataloader=val_loader,
                    device=device,
                    mask_token=mask_token,
                    mask_ratio=cfg.mask_ratio,
                    decoder=decoder,
                    capture=capture,
                    probe_head=probe_head,
                    align=align,
                )
                if probe_head is not None:
                    probe_head.train()
                if align is not None:
                    align.projector.train()
                    val_rows.append({"step": global_step, **val_metrics})
                last_val_metrics = val_metrics

                wandb.log(val_metrics, step=global_step)

                history.append(
                    {
                        "global_step": global_step,
                        "train_metrics": step_train_metrics,
                        "val_metrics": val_metrics,
                    }
                )

                # train/mim_loss exists only when the MIM head runs (weight != 0).
                mim_part = (
                    f"train_mim={step_train_metrics['train/mim_loss']:.6f} "
                    if "train/mim_loss" in step_train_metrics
                    else ""
                )
                print(
                    f"[finetune-diffusion] step={global_step} "
                    f"train_total={step_train_metrics['train/total_loss']:.6f} "
                    f"train_flow={step_train_metrics['train/flow_mse_loss']:.6f} "
                    f"{mim_part}"
                    f"val_total={val_metrics['val/total_loss']:.6f}"
                    + (f" val_align={val_metrics['val/align_loss_mean']:.4f}" if align is not None else "")
                )

                val_loss = val_metrics["val/total_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_lora_checkpoint(
                        cfg=cfg,
                        model=flux,
                        optimizer=optimizer,
                        global_step=global_step,
                        train_metrics=step_train_metrics,
                        val_metrics=val_metrics,
                        name="lora_best",
                        mask_token=mask_token,
                        decoder=decoder,
                        probe_head=probe_head,
                        extra=_align_ckpt_fields(cfg, align, global_step),
                    )
            if align is not None and global_step < S:
                if cfg.align.meas_every and global_step % cfg.align.meas_every == 0:
                    meas[f"step{global_step}"] = measure(f"step{global_step}")
                if cfg.align.snapshot_every and global_step % cfg.align.snapshot_every == 0:
                    # Logging only: LoRA + projector at a regular cadence (lora_best_* is irregular).
                    _save_lora_checkpoint(
                        cfg=cfg,
                        model=flux,
                        optimizer=optimizer,
                        global_step=global_step,
                        train_metrics=step_train_metrics,
                        val_metrics=last_val_metrics,
                        name="lora_snap",
                        mask_token=mask_token,
                        decoder=decoder,
                        probe_head=probe_head,
                        extra=_align_ckpt_fields(cfg, align, global_step),
                    )
                if cfg.align.ckpt_every and global_step % cfg.align.ckpt_every == 0:
                    _save_resumable(
                        ckpt_path, resumable_record()
                    )  # after val/meas: a resume never skips them
                    if cfg.align.keep_ckpt_every and global_step % cfg.align.keep_ckpt_every == 0:
                        # Exact branch point: optimizer, scheduler, RNG streams and stats included.
                        shutil.copyfile(
                            ckpt_path, os.path.join(cfg.save_dir, f"ckpt_step{global_step:08d}.pt")
                        )

        align_meta: dict = {}
        if align is not None:
            # Deliverable order (review 2026-10-14): measurement pass -> gate file, stats.npz,
            # val.jsonl, meta.json, lora_last -> ONLY THEN the final generation and the FID/KID
            # pass, so a failure in the generation block cannot discard the run's outputs
            # (meta.json is rewritten with the generation fields afterwards).
            meas["final"] = measure("final")
            sha_final = None
            # Reached only if no step failed (_fail writes the FAIL file and raises).
            gate = "PASS" if cfg.align.arm == "proj" else "N/A (no projection in this arm)"
            fp32_gate = f"PASS (init + pre/post every optimizer step, {S} steps)"
            with open(gate_path, "w") as f:
                f.write(f"projection verifier: {gate}\nfp32 LoRA params: {fp32_gate}\n")
            flat_meas = {
                f"meas/{tag}/t{t_nom}/{k}": v
                for tag, per_t in meas.items()
                for t_nom, d in per_t.items()
                for k, v in d.items()
            }
            np.savez(
                os.path.join(cfg.save_dir, "stats.npz"),
                blocks=np.array(align.blocks + [-1]),
                k=np.array(cfg.k),
                arm=np.array(cfg.align.arm),
                align_weight=np.array(align.w),
                fm_weight=np.array(align.w_fm),
                stop_step=np.array(cfg.align.stop_step),
                meas_t=np.array(cfg.align.meas_t),
                **stats.npz_fields(),
                **flat_meas,
            )
            with open(os.path.join(cfg.save_dir, "val.jsonl"), "w") as f:
                for r in val_rows:
                    f.write(json.dumps(r) + "\n")
            n_train = len(train_loader.dataset)
            align_meta = {
                "align": align.describe(),
                "gate": gate,
                "fp32_gate": fp32_gate,
                "gen_sha_step0": sha_init,
                "gen_sha_final": sha_final,  # filled in after the final generation below
                "gen_metrics": {
                    k: {f: v.get(f) for f in ("n", "fid", "kid_mean", "kid_std", "error")}
                    for k, v in gen_metrics.items()
                },
                "gen_metrics_ref_sha1": gen_metrics["step0"]["real"]["sha1"]
                if "step0" in gen_metrics
                else None,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
                "mean_step_time_s": float(np.nanmean(stats.sc["step_time"])),
                "img_per_s": float(cfg.batch_size * grad_accum_steps / np.nanmean(stats.sc["step_time"])),
                # No epochs: quote unique exposures (CLAUDE.md: budget by unique samples).
                "samples_seen": int(S * grad_accum_steps * cfg.batch_size),
                "train_size": int(n_train),
                "unique_fraction_seen": float(min(1.0, S * grad_accum_steps * cfg.batch_size / n_train)),
                "train_data": train_data,
                "resumed_from": cfg.align.resume or None,
            }

            def _write_meta():
                with open(os.path.join(cfg.save_dir, "meta.json"), "w") as f:
                    json.dump(
                        {"cfg": to_jsonable(cfg), "run_name": cfg.make_run_name(), **align_meta}, f, indent=2
                    )

            _write_meta()
            print(
                f"[align] done: gate={gate} peak_mem={align_meta['peak_mem_gb']} GB -> {cfg.save_dir}",
                flush=True,
            )

        if history:
            _save_lora_checkpoint(
                cfg=cfg,
                model=flux,
                optimizer=optimizer,
                global_step=global_step,
                train_metrics=last_train_metrics,
                val_metrics=last_val_metrics,
                name="lora_last",
                mask_token=mask_token,
                decoder=decoder,
                probe_head=probe_head,
                extra=_align_ckpt_fields(cfg, align, global_step),
            )

        if align is not None and not cfg.align.no_gen:
            # Final generation + distributional metric, after every deliverable is on disk.
            sha_final = _generate(cfg, featurizer_model, gen_dir, "final", device)
            align_meta["gen_sha_final"] = sha_final
            if cfg.align.fid_n:
                gen_metrics["final"] = _gen_metrics(cfg, featurizer_model, fid_ref, fid_extractor, device, "final",
                                                    save_dir=os.path.join(gen_dir, "fid_final"))  # fmt: skip
                with open(os.path.join(cfg.save_dir, "gen_metrics.json"), "w") as f:
                    json.dump(gen_metrics, f, indent=1)
                align_meta["gen_metrics"] = {
                    k: {f: v.get(f) for f in ("n", "fid", "kid_mean", "kid_std", "error")}
                    for k, v in gen_metrics.items()
                }
                align_meta["gen_metrics_ref_sha1"] = gen_metrics["final"]["real"]["sha1"]
            _write_meta()

        # ------------------------------------------------------------------
        # FINAL EVALUATION (2026-09-13 protocol): the concurrently-trained head, applied to
        # the OFFICIAL test split once per grid timestep. Replaces the old post-training
        # extraction + sklearn probe stage entirely: the head was trained during the run, so
        # the only remaining cost is the test-split forwards (len(test) x len(grid)).
        # Deterministic: posterior-mean latents, seeded eps, unshuffled loader -- same
        # regime as validation. Evaluates the FINAL weights (fixed training budget on every
        # arm; see 6s item 6 -- reloading lora_best would make the number depend on the
        # val-selection rule).
        # ------------------------------------------------------------------
        for param in flux.parameters():
            param.requires_grad = False

        flux.eval()
        if probe_head is None or test_loader is None:
            # Label-free corpus: the downstream evaluation is run.py --task extract --lora-checkpoint
            # <this run>/checkpoints/lora_last_*.pt per GEO-Bench task, then eval.matrix.
            return {
                "history": history,
                "probe_results": {
                    "protocol": "none (label-free corpus); evaluate via task=extract + eval.matrix"
                },
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "num_trainable_params_backbone": sum(p.numel() for p in trainable_params),
                "num_trainable_params_head": 0,
                **align_meta,
            }
        probe_head.eval()
        reg_prev = featurizer_model.ae.reg.sample
        featurizer_model.ae.reg.sample = False
        per_t_top1: dict[int, float] = {}
        per_t_f1: dict[int, float] = {}
        try:
            with torch.no_grad():
                test_gen = torch.Generator(device=device).manual_seed(VAL_NOISE_SEED)
                all_preds: dict[int, list] = {t_nom: [] for t_nom in timestep_grid}
                all_labels_l: list = []
                for batch in test_loader:
                    imgs = batch["img"].to(device)
                    all_labels_l.append(batch["label"].numpy())
                    latents = featurizer_model.ae.encode(imgs).to(torch.bfloat16)
                    for t_nom in timestep_grid:
                        t = torch.full((imgs.shape[0],), t_nom / 1000.0, device=device, dtype=torch.float32)
                        t_b = t.view(-1, *([1] * (latents.dim() - 1)))
                        noise = torch.randn(
                            latents.shape, generator=test_gen, device=device, dtype=latents.dtype
                        )
                        latents_noisy = (t_b * noise.float() + (1.0 - t_b) * latents.float()).to(
                            latents.dtype
                        )
                        img, img_ids = prepare(img=latents_noisy)
                        img = img.to(device, dtype=latents.dtype)
                        img_ids = img_ids.to(device)
                        txt, txt_ids, yv = _expand_null_embeddings(
                            featurizer_model, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
                        )
                        guidance_vec = torch.full(
                            (imgs.shape[0],), cfg.guidance_scale, device=device, dtype=latents.dtype
                        )
                        _, up_ft = flux.forward_velocity_feat(
                            img=img,
                            img_ids=img_ids,
                            txt=txt,
                            txt_ids=txt_ids,
                            y=yv,
                            timesteps=t.to(latents.dtype),
                            ft_indices=[cfg.k],
                            guidance=guidance_vec,
                        )
                        feat_pooled = up_ft[0].float().mean(dim=1)
                        all_preds[t_nom].append(probe_head(feat_pooled).argmax(dim=1).cpu().numpy())
        finally:
            featurizer_model.ae.reg.sample = reg_prev

        y_true = np.concatenate(all_labels_l)
        for t_nom in timestep_grid:
            y_hat = np.concatenate(all_preds[t_nom])
            per_t_top1[t_nom] = round(float((y_hat == y_true).mean() * 100.0), 2)
            # labels= so an absent class counts as 0 instead of shrinking the macro average
            # (same statistic as eval.probe / eval.sweep, which pass labels=arange(n_cls)).
            per_t_f1[t_nom] = round(
                float(f1_score(y_true, y_hat, average="macro", labels=np.arange(num_classes)) * 100.0), 2
            )
            print(
                f"[probe] test split t={t_nom}: top1={per_t_top1[t_nom]:.2f} macro_f1={per_t_f1[t_nom]:.2f}",
                flush=True,
            )

        probe_results = {
            "protocol": "concurrent head, full train split; eval = official test split",
            "mode": mode,
            "per_timestep_top1": {f"t{k}": v for k, v in per_t_top1.items()},
            "per_timestep_macro_f1": {f"t{k}": v for k, v in per_t_f1.items()},
            "mean_top1_over_timesteps": round(float(np.mean(list(per_t_top1.values()))), 2),
            # eps for the test eval is consumed batch-major from a fixed seed, so per-
            # (image, t) noise matches across arms ONLY at equal batch_size; recorded so
            # a mismatch is visible in the results.
            "eval_batch_size": cfg.batch_size,
        }
        flat = {}
        _flatten_scalars_into("probe", probe_results, flat)
        wandb.log(flat)

        return {
            "history": history,
            "probe_results": probe_results,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            # Backbone and head counted separately: the head is trainable in every
            # mode, the backbone only outside freeze mode.
            "num_trainable_params_backbone": sum(p.numel() for p in trainable_params),
            "num_trainable_params_head": sum(p.numel() for p in probe_head.parameters()),
            **align_meta,
        }


def _load_lora_ckpt(cfg, flux, align, path: str) -> int:
    """Load the LoRA tensors and the projector of a lora_*.pt of THIS run into the live model after
    checking the checkpoint belongs to this configuration (arm, feat, target, k, rank, alpha, run
    dir). Returns the checkpoint's global step. Shared by the split-eval and virtual-steps
    diagnostics (moved out of _split_eval verbatim, 2026-10-24)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    extra = ck.get("align") or {}
    if "projector_state_dict" not in extra or "lora_state_dict" not in ck:
        raise SystemExit(
            f"{path}: not an alignment-run lora checkpoint (needs lora_state_dict + align.projector_state_dict)"
        )
    ck_cfg = ck.get("cfg") or {}
    want = {
        "arm": cfg.align.arm,
        "feat": cfg.align.feat,
        "target": cfg.align.target,
        "k": cfg.k,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "run": os.path.basename(cfg.save_dir),
    }
    have = {
        "arm": extra.get("arm"),
        "feat": extra.get("feat"),
        "target": extra.get("target"),
        "k": ck_cfg.get("k"),
        "lora_rank": ck_cfg.get("lora_rank"),
        "lora_alpha": ck_cfg.get("lora_alpha"),
        "run": os.path.basename(str(ck_cfg.get("save_dir", ""))),
    }
    if have != want:
        raise SystemExit(
            f"{path}: checkpoint / this run differ: { ({k: (have[k], want[k]) for k in want if have[k] != want[k]}) }"
        )
    sd = ck["lora_state_dict"]
    params = dict(flux.named_parameters())
    lora_names = sorted(n for n in params if n.endswith(".A") or n.endswith(".B"))
    if sorted(sd) != lora_names:
        raise SystemExit(
            f"{path}: LoRA tensor names differ from the wrapped model ({len(sd)} vs {len(lora_names)})"
        )
    with torch.no_grad():
        for n, v in sd.items():
            params[n].copy_(v.to(device=params[n].device, dtype=params[n].dtype))
    align.projector.load_state_dict(extra["projector_state_dict"])
    return int(ck.get("global_step", -1))


def _split_eval(
    cfg, featurizer, flux, align, timestep_grid, val_loader, device, mask_token, decoder, capture, probe_head
) -> None:
    """Backbone / projector split of the teacher cosine (align.split_eval, mentor request 2026-10-22).

    Loads the LoRA + projector of a finished run's lora_*.pt, then runs the training validation pass
    (the same strided val subset and t grid as val.jsonl) twice: LoRA ON (the extracted weights) and
    LoRA OFF (every LoRALinear.scaling = 0: the pretrained backbone under the trained projector).
      cos_on   = teacher cosine of adapted backbone + trained projector  (must equal val.jsonl's row
                 at the checkpoint step: gate 1)
      cos_off  = pretrained backbone + trained projector
      cos_init = val.jsonl step 0: pretrained backbone + warmed-up projector (LoRA B = 0 at step 0)
      backbone_share = cos_on - cos_off; projector_share = cos_off - cos_init.
    Gate 2: the LoRA-OFF flow loss must equal the step-0 flow loss (proves scaling = 0 removes the
    adapter). Writes <run dir>/split_eval.json; a failed gate exits non-zero (rule 13).
    Caveat recorded in the json: the projector was trained on ADAPTED features, so cos_off is a
    slight under-estimate of a projector trained on frozen features (the fm arm is that estimate)."""
    from models.lora import LoRALinear

    path = cfg.align.split_eval
    step = _load_lora_ckpt(cfg, flux, align, path)

    def run_val() -> dict:
        return _validate_diffusion(
            cfg,
            featurizer=featurizer,
            timestep_grid=timestep_grid,
            dataloader=val_loader,
            device=device,
            mask_token=mask_token,
            mask_ratio=cfg.mask_ratio,
            decoder=decoder,
            capture=capture,
            probe_head=probe_head,
            align=align,
        )

    on = run_val()
    loras = [m for m in flux.modules() if isinstance(m, LoRALinear)]
    if not loras:
        raise SystemExit("split_eval: no LoRALinear modules in the model")
    saved = [m.scaling for m in loras]
    for m in loras:
        m.scaling = 0.0
    try:
        off = run_val()
    finally:
        for m, sc in zip(loras, saved):
            m.scaling = sc

    rows = [json.loads(line) for line in open(os.path.join(cfg.save_dir, "val.jsonl"))]
    ref = next((r for r in rows if r["step"] == step), None)
    row0 = next((r for r in rows if r["step"] == 0), None)
    if ref is None or row0 is None:
        raise SystemExit(f"split_eval: val.jsonl lacks a row at step {step} or step 0")
    key, fkey = "val/align_loss_mean", "val/flow_mse_loss"
    cos_on, cos_off, cos_ref, cos_init = -on[key], -off[key], -ref[key], -row0[key]
    tol = 2e-3
    gate1 = abs(cos_on - cos_ref) <= tol
    gate2 = abs(off[fkey] - row0[fkey]) <= tol
    per_t = {
        t: {
            "cos_on": -on[f"val/align_loss_t{t}"],
            "cos_off": -off[f"val/align_loss_t{t}"],
            "cos_init": -row0[f"val/align_loss_t{t}"],
        }
        for t in timestep_grid
    }
    out = {
        "checkpoint": path,
        "step": step,
        "arm": cfg.align.arm,
        "weight_config": cfg.align.weight,
        # sem: g_FM never applied (fm_weight 0); stop_step: the semantic weight was 0 from that step on.
        "fm_weight": align.w_fm,
        "stop_step": cfg.align.stop_step,
        "weight_effective_at_checkpoint": 0.0
        if (cfg.align.stop_step and step > cfg.align.stop_step)
        else align.w,
        "k": cfg.k,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "n_lora_modules": len(loras),
        "val_images": len(val_loader.dataset),
        "t_grid": list(timestep_grid),
        "cos_on": cos_on,
        "cos_off": cos_off,
        "cos_init": cos_init,
        "cos_ref_val_jsonl": cos_ref,
        "backbone_share": cos_on - cos_off,
        "projector_share": cos_off - cos_init,
        "total_gain": cos_on - cos_init,
        "flow_on": on[fkey],
        "flow_off": off[fkey],
        "flow_init_val_jsonl": row0[fkey],
        "per_t": per_t,
        "tol": tol,
        "gate_cos_matches_val_jsonl": gate1,
        "gate_lora_off_is_pretrained": gate2,
        "caveat": "projector trained on adapted features; cos_off under-estimates a frozen-feature projector (fm arm = that estimate)",
    }
    with open(os.path.join(cfg.save_dir, "split_eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"SPLIT_EVAL {cfg.align.arm} step={step} cos_init={cos_init:.4f} cos_off={cos_off:.4f} cos_on={cos_on:.4f} "
        f"(val.jsonl {cos_ref:.4f}) backbone_share={cos_on - cos_off:+.4f} projector_share={cos_off - cos_init:+.4f} "
        f"flow_off={off[fkey]:.4f} (step0 {row0[fkey]:.4f}) gate1={'PASS' if gate1 else 'FAIL'} gate2={'PASS' if gate2 else 'FAIL'}",
        flush=True,
    )
    if not (gate1 and gate2):
        raise SystemExit("SPLIT_EVAL gate FAIL (see line above)")
    print("SPLIT_EVAL_OK", flush=True)


# Virtual step size as a fraction of ||theta_LoRA|| (AlignConfig.virtual_steps). The LoRA A/B are
# fp32 but the forward casts them to bf16 (0.4 % relative resolution), so a step below ~1 % of the
# parameter norm is partly rounded away: no smaller eta, and the first-order term is isolated from
# the curvature by stepping both ways (see _virtual_steps) rather than by shrinking the step.
def _grad_geometry(cfg, featurizer, flux, align, meas_batches, device) -> None:
    """Gradient geometry at saved checkpoints of a finished run (AlignConfig.grad_geometry, 2026-10-28):
    the per-batch LoRA gradients of the FM loss and of the semantic loss on the fixed training
    measurement batches (posterior-mean latents) at every t in meas_t, with measure()'s noise stream
    (seed * 1000 + t, the stream _virtual_steps uses), as ONE Gram matrix per block over all
    2 x |meas_t| x n_batches vectors (rows ordered objective-major: fm@t0 b0.., fm@t1 .., sem@t0 ..).
    Every cosine of the analysis (cross-t, cross-objective, per-batch coherence, PCA of the cloud) is
    a function of the Gram matrix, so no gradient vector is stored. The pooled (all-block) Gram is the
    sum over blocks. Writes <run dir>/grad_geometry_s<step>.npz per checkpoint (refuses to
    overwrite), prints GRAD_GEOMETRY lines and GRAD_GEOMETRY_OK. GATE: the per-t mean cos(fm, sem)
    recomputed from the Gram matrix equals the one computed from the vectors (1e-6)."""
    paths = sorted(
        {p for pat in cfg.align.grad_geometry.split(",") if pat.strip() for p in glob.glob(pat.strip())}
    )
    if not paths:
        raise SystemExit(f"align.grad_geometry: no checkpoint matches {cfg.align.grad_geometry!r}")
    if not meas_batches:
        raise SystemExit("grad_geometry: no training measurement batches (vs_grad_batches x batch_size = 0?)")
    n_b, ts = len(meas_batches), [int(t) for t in cfg.align.meas_t]
    n_vec = 2 * len(ts) * n_b
    obj = np.repeat([0, 1], len(ts) * n_b)  # 0 = fm, 1 = sem
    t_of = np.tile(np.repeat(ts, n_b), 2)
    b_of = np.tile(np.arange(n_b), 2 * len(ts))
    dims = {b: sum(p.numel() for _, p in align.groups[b]) for b in align.blocks}
    out_paths = []
    for path in paths:
        step = _load_lora_ckpt(cfg, flux, align, path)
        out_path = os.path.join(cfg.save_dir, f"grad_geometry_s{step}.npz")
        if os.path.exists(out_path):
            raise SystemExit(f"grad_geometry: {out_path} exists; remove it first")
        flux.eval()
        align.projector.eval()
        t0 = time.time()
        vecs = {
            b: torch.empty(n_vec, dims[b], dtype=torch.float32) for b in align.blocks
        }  # CPU, ~12 GB at 32 batches
        for ti, t_nom in enumerate(ts):
            e_gen = torch.Generator(device=device).manual_seed(cfg.seed * 1000 + t_nom)
            for bi, (imgs, latents) in enumerate(meas_batches):
                t = torch.full((imgs.shape[0],), t_nom / 1000.0, device=device, dtype=torch.float32)
                noise = torch.randn(latents.shape, generator=e_gen, device=device, dtype=latents.dtype)
                fl, al = _align_losses(cfg, featurizer, align, imgs, latents, t, noise)
                d_fm, d_al, _ = align.two_grads(fl, al)
                f_fm, f_al = GP.flatten(align.groups, d_fm), GP.flatten(align.groups, d_al)
                r_fm, r_al = ti * n_b + bi, len(ts) * n_b + ti * n_b + bi
                for b in align.blocks:
                    vecs[b][r_fm] = f_fm[b]
                    vecs[b][r_al] = f_al[b]
        gram = np.empty((len(align.blocks), n_vec, n_vec), dtype=np.float64)
        for i, b in enumerate(align.blocks):
            v = vecs[b].to(device=device, dtype=torch.float64)
            gram[i] = (v @ v.T).cpu().numpy()
            del v
        pooled = gram.sum(axis=0)
        # GATE: per-t mean cos(fm, sem) from the Gram matrix vs from the mean vectors themselves
        mean_cos_gram, mean_cos_vec = [], []
        for ti, t_nom in enumerate(ts):
            rf = slice(ti * n_b, (ti + 1) * n_b)
            rs = slice(len(ts) * n_b + ti * n_b, len(ts) * n_b + (ti + 1) * n_b)
            dot = pooled[rf, rs].sum() / (n_b * n_b)
            nf = math.sqrt(pooled[rf, rf].sum()) / n_b
            ns = math.sqrt(pooled[rs, rs].sum()) / n_b
            mean_cos_gram.append(dot / max(nf * ns, 1e-30))
            mf = torch.cat([vecs[b][rf].mean(0) for b in align.blocks]).double()
            ms = torch.cat([vecs[b][rs].mean(0) for b in align.blocks]).double()
            mean_cos_vec.append(float((mf @ ms) / max(float(mf.norm() * ms.norm()), 1e-30)))
        gate = max(abs(a - c) for a, c in zip(mean_cos_gram, mean_cos_vec)) < 1e-6
        if not gate:
            raise SystemExit(
                f"GRAD_GEOMETRY gate FAIL at {path}: gram {mean_cos_gram} vs vectors {mean_cos_vec}"
            )
        del vecs
        np.savez(
            out_path,
            gram=gram,
            pooled=pooled,
            obj=obj,
            t=t_of,
            batch=b_of,
            blocks=np.array(align.blocks),
            block_dim=np.array([dims[b] for b in align.blocks]),
            meas_t=np.array(ts),
            n_batches=np.array(n_b),
            n_images=np.array(sum(int(b[0].shape[0]) for b in meas_batches)),
            step=np.array(step),
            checkpoint=np.array(path),
            arm=np.array(cfg.align.arm),
            seed=np.array(cfg.seed),
            mean_cos_fm_sem=np.array(mean_cos_gram),
            gate=np.array(gate),
        )
        out_paths.append(out_path)
        print(f"GRAD_GEOMETRY {cfg.align.arm} step={step} vectors={n_vec} ({len(ts)} t x {n_b} batches x 2) blocks={len(align.blocks)} "
              f"cos(fm,sem) per t {[round(c, 3) for c in mean_cos_gram]} (t = {ts}) gate={'PASS' if gate else 'FAIL'} ({time.time() - t0:.0f}s) -> {out_path}", flush=True)  # fmt: skip
    flux.train()
    align.projector.train()
    print(f"GRAD_GEOMETRY_OK {' '.join(out_paths)}", flush=True)


VS_ETAS = (0.01, 0.03)


def _val_grads(cfg, featurizer, align, timestep_grid, val_loader, device):
    """Mean FM and semantic LoRA gradients over the validation pass's inputs (every held-out image x
    every grid t, the VAL_NOISE_SEED noise stream, posterior-mean latents), i.e. the gradients of
    the two quantities the virtual steps are scored on. Returns (g_fm, g_sem) as per-block flat
    fp32 CPU vectors, the number of (image-batch, t) terms averaged, and the same two gradients per
    grid t ({t: (g_fm_t, g_sem_t)}, each a mean over the image batches at that t)."""
    vae = featurizer.ae
    reg_prev = vae.reg.sample
    vae.reg.sample = False
    acc_fm = acc_al = None
    per_t = {}
    n = 0
    try:
        noise_gen = torch.Generator(device=device).manual_seed(VAL_NOISE_SEED)
        for batch in val_loader:
            imgs = batch["img"].to(device)
            with torch.no_grad():
                latents = vae.encode(imgs).to(torch.bfloat16)
            for t_nom in timestep_grid:
                t = torch.full((imgs.shape[0],), t_nom / 1000.0, device=device, dtype=torch.float32)
                noise = torch.randn(latents.shape, generator=noise_gen, device=device, dtype=latents.dtype)
                fl, al = _align_losses(cfg, featurizer, align, imgs, latents, t, noise)
                d_fm, d_al, _ = align.two_grads(fl, al)
                f_fm, f_al = GP.flatten(align.groups, d_fm), GP.flatten(align.groups, d_al)
                if t_nom not in per_t:
                    per_t[t_nom] = [
                        OrderedDict((b, f_fm[b].clone()) for b in align.blocks),
                        OrderedDict((b, f_al[b].clone()) for b in align.blocks),
                        0,
                    ]
                else:
                    for b in align.blocks:
                        per_t[t_nom][0][b] += f_fm[b]
                        per_t[t_nom][1][b] += f_al[b]
                per_t[t_nom][2] += 1
                if acc_fm is None:
                    acc_fm, acc_al = f_fm, f_al
                else:
                    for b in align.blocks:
                        acc_fm[b] += f_fm[b]
                        acc_al[b] += f_al[b]
                n += 1
    finally:
        vae.reg.sample = reg_prev
    for b in align.blocks:
        acc_fm[b] /= n
        acc_al[b] /= n
    for t_nom, (g_fm, g_al, m) in per_t.items():
        for b in align.blocks:
            g_fm[b] /= m
            g_al[b] /= m
    return acc_fm, acc_al, n, {t: (g[0], g[1]) for t, g in per_t.items()}


def _virtual_steps(
    cfg,
    featurizer,
    flux,
    align,
    timestep_grid,
    val_loader,
    meas_batches,
    device,
    mask_token,
    decoder,
    capture,
    probe_head,
) -> None:
    """Held-out virtual steps (P1, plan 2026-10-23): what one SGD step along each gradient does to the
    held-out FM loss and teacher cosine, at saved checkpoints of a finished run.

    Per checkpoint (AlignConfig.virtual_steps, a comma-separated list of paths / globs; with
    AlignConfig.vs_per_t also the single-noise-level directions g_FM@t / g_SEM@t, t in meas_t, and
    always the per-t gradient statistics: within-objective coherence = mean pairwise cosine of the
    per-batch gradients at one t, estimated as (|sum g|^2 - sum |g|^2) / (n (n-1)) over the mean
    |g|^2; the cross-t cosines of the per-t mean gradients; and the first-order predictions of every
    stepped direction's effect at every grid t from the per-t validation gradients):
      1. load LoRA + projector (_load_lora_ckpt; same identity checks as split-eval);
      2. g_FM, g_SEM = mean LoRA gradients over vs_grad_batches fixed TRAINING batches (the
         measurement batches: posterior-mean latents) x the meas_t grid with measure()'s noise stream,
         plus P(g_SEM) (one-sided projection, align.proj_scope) and the summed directions
         g_FM + w g_SEM and g_FM + w P(g_SEM), w = the run's semantic weight (fm: its CONFIG weight,
         marked, since fm applies 0);
      3. the same two gradients on the validation inputs (_val_grads): cos(train, val) per gradient
         says how well each direction generalises, cos(g_FM_val, g_SEM_val) is the held-out gradient
         alignment, and <g_val, delta> is the first-order prediction of every measured change;
      4. baseline = the validation pass (eta = 0). GATE 1: it reproduces val.jsonl's flow / align
         rows at the checkpoint step (tol 2e-3) when such a row exists;
      5. for each direction d in {fm, sem, sum, proj} and eta in VS_ETAS, BOTH ways: theta <- theta0 -+
         eta ||theta0|| d/||d|| (descent, then ascent), validation pass each, theta <- theta0. The
         antisymmetric half (descent - ascent) / 2 is the first-order change (compared with the
         <g_val, delta> prediction), the symmetric half (descent + ascent) / 2 the curvature along
         d. GATE 2: every LoRA tensor equals theta0 bit for bit afterwards.
    Nothing here is an optimizer step of the run: AdamW's state is untouched and no file of the run
    changes except the new <run dir>/virtual_steps_<steps>[_pert].json (refuses to overwrite). Prints VIRTUAL_STEPS lines, VIRTUAL_STEPS_OK."""
    paths = sorted(
        {p for pat in cfg.align.virtual_steps.split(",") if pat.strip() for p in glob.glob(pat.strip())}
    )
    if not paths:
        raise SystemExit(f"align.virtual_steps: no checkpoint matches {cfg.align.virtual_steps!r}")
    if not meas_batches:
        raise SystemExit("virtual_steps: no training measurement batches (vs_grad_batches x batch_size = 0?)")
    val_rows = [json.loads(line) for line in open(os.path.join(cfg.save_dir, "val.jsonl"))]
    # one output file per checkpoint list, never overwritten (a later list on the same run dir
    # must not erase an earlier job's rows)
    step_tags = ["s%d" % int(re.search(r"step(\d+)", os.path.basename(p)).group(1)) for p in paths]
    out_path = os.path.join(
        cfg.save_dir,
        "virtual_steps_" + "_".join(step_tags) + ("_pert" if cfg.align.vs_per_t else "") + ".json",
    )
    if os.path.exists(out_path):
        raise SystemExit(f"virtual_steps: {out_path} exists; remove it or pick a different checkpoint list")
    fkey, akey = "val/flow_mse_loss", "val/align_loss_mean"
    w = align.w if align.w > 0 else cfg.align.weight
    w_note = "effective weight" if align.w > 0 else "config weight (fm applies 0)"
    tol = 2e-3

    def run_val() -> dict:
        return _validate_diffusion(
            cfg,
            featurizer=featurizer,
            timestep_grid=timestep_grid,
            dataloader=val_loader,
            device=device,
            mask_token=mask_token,
            mask_ratio=cfg.mask_ratio,
            decoder=decoder,
            capture=capture,
            probe_head=probe_head,
            align=align,
        )

    def flat_norm(g) -> float:
        return float(torch.sqrt(sum((g[b].double() ** 2).sum() for b in align.blocks)))

    def flat_dot(a, b_) -> float:
        return float(sum((a[b].double() * b_[b].double()).sum() for b in align.blocks))

    def cosine(a, b_) -> float:
        return flat_dot(a, b_) / max(flat_norm(a) * flat_norm(b_), 1e-30)

    out = {
        "etas": list(VS_ETAS),
        "grad_batches": len(meas_batches),
        "grad_images": sum(int(b[0].shape[0]) for b in meas_batches),
        "grad_t": list(cfg.align.meas_t),
        "val_images": len(val_loader.dataset),
        "t_grid": list(timestep_grid),
        "weight_used": w,
        "weight_note": w_note,
        "arm": cfg.align.arm,
        "fm_weight": align.w_fm,
        "tol": tol,
        "checkpoints": [],
    }
    for path in paths:
        t0 = time.time()
        step = _load_lora_ckpt(cfg, flux, align, path)
        flux.eval()
        align.projector.eval()
        # 2. training-batch gradients, measure()'s streams
        acc_fm = acc_al = None
        t_fm, t_al = OrderedDict(), OrderedDict()  # per-t sums of the per-batch gradients
        sq_fm, sq_al = OrderedDict(), OrderedDict()  # per-t sums of the per-batch squared norms
        for t_nom in cfg.align.meas_t:
            e_gen = torch.Generator(device=device).manual_seed(cfg.seed * 1000 + t_nom)
            for imgs, latents in meas_batches:
                t = torch.full((imgs.shape[0],), t_nom / 1000.0, device=device, dtype=torch.float32)
                noise = torch.randn(latents.shape, generator=e_gen, device=device, dtype=latents.dtype)
                fl, al = _align_losses(cfg, featurizer, align, imgs, latents, t, noise)
                d_fm, d_al, _ = align.two_grads(fl, al)
                f_fm, f_al = GP.flatten(align.groups, d_fm), GP.flatten(align.groups, d_al)
                if t_nom not in t_fm:
                    t_fm[t_nom] = OrderedDict((b, f_fm[b].clone()) for b in align.blocks)
                    t_al[t_nom] = OrderedDict((b, f_al[b].clone()) for b in align.blocks)
                    sq_fm[t_nom] = sq_al[t_nom] = 0.0
                else:
                    for b in align.blocks:
                        t_fm[t_nom][b] += f_fm[b]
                        t_al[t_nom][b] += f_al[b]
                sq_fm[t_nom] += flat_norm(f_fm) ** 2
                sq_al[t_nom] += flat_norm(f_al) ** 2
                if acc_fm is None:
                    acc_fm, acc_al = f_fm, f_al
                else:
                    for b in align.blocks:
                        acc_fm[b] += f_fm[b]
                        acc_al[b] += f_al[b]
        n_b = len(meas_batches)
        n_terms = len(cfg.align.meas_t) * n_b
        for b in align.blocks:
            acc_fm[b] /= n_terms
            acc_al[b] /= n_terms
            for t_nom in cfg.align.meas_t:
                t_fm[t_nom][b] /= n_b
                t_al[t_nom][b] /= n_b

        def coherence(sum_g, sq, m) -> float:
            # mean pairwise <g_i, g_j> over the m(m-1) ordered pairs, relative to the mean |g_i|^2:
            # 1 = all per-batch gradients identical in direction and size, 0 = uncorrelated.
            if m < 2:
                return float("nan")
            s2 = flat_norm(sum_g) ** 2  # sum_g is the per-t MEAN: |m g_mean|^2 = |sum g_i|^2
            return ((s2 * m * m - sq) / (m * (m - 1))) / max(sq / m, 1e-30)

        per_t_stats = {
            "n_batches": n_b,
            "norm_fm": {t: flat_norm(t_fm[t]) for t in cfg.align.meas_t},
            "norm_sem": {t: flat_norm(t_al[t]) for t in cfg.align.meas_t},
            "coherence_fm": {t: coherence(t_fm[t], sq_fm[t], n_b) for t in cfg.align.meas_t},
            "coherence_sem": {t: coherence(t_al[t], sq_al[t], n_b) for t in cfg.align.meas_t},
            "coherence_fm_all_t": coherence(acc_fm, sum(sq_fm.values()), n_terms),
            "coherence_sem_all_t": coherence(acc_al, sum(sq_al.values()), n_terms),
            "cos_fm_sem": {t: cosine(t_fm[t], t_al[t]) for t in cfg.align.meas_t},
            "cross_t_cos_fm": {
                ti: {tj: cosine(t_fm[ti], t_fm[tj]) for tj in cfg.align.meas_t} for ti in cfg.align.meas_t
            },
            "cross_t_cos_sem": {
                ti: {tj: cosine(t_al[ti], t_al[tj]) for tj in cfg.align.meas_t} for ti in cfg.align.meas_t
            },
            "cross_t_cos_fm_sem": {
                ti: {tj: cosine(t_fm[ti], t_al[tj]) for tj in cfg.align.meas_t} for ti in cfg.align.meas_t
            },
        }
        g_proj, hits = GP.project_one_sided(acc_al, acc_fm, scope=cfg.align.proj_scope)
        dirs = OrderedDict(
            fm=acc_fm,
            sem=acc_al,
            sum=OrderedDict((b, acc_fm[b] + w * acc_al[b]) for b in align.blocks),
            proj=OrderedDict((b, acc_fm[b] + w * g_proj[b]) for b in align.blocks),
        )
        if cfg.align.vs_per_t:
            for t_nom in cfg.align.meas_t:
                dirs[f"fm@t{t_nom}"] = t_fm[t_nom]
            for t_nom in cfg.align.meas_t:
                dirs[f"sem@t{t_nom}"] = t_al[t_nom]
        # 3. validation-input gradients (pooled and per grid t)
        v_fm, v_al, n_val_terms, v_per_t = _val_grads(
            cfg, featurizer, align, timestep_grid, val_loader, device
        )
        # first-order prediction of every candidate direction's effect at every grid t, per unit
        # ||delta theta|| (the t_grad x t_eval matrix, before any step is taken)
        pred_matrix = {}
        for name, d in (
            list(dirs.items())
            + [(f"fm@t{t}", t_fm[t]) for t in cfg.align.meas_t if not cfg.align.vs_per_t]
            + [(f"sem@t{t}", t_al[t]) for t in cfg.align.meas_t if not cfg.align.vs_per_t]
        ):
            dn = flat_norm(d)
            unit = OrderedDict((b, d[b] / dn) for b in align.blocks)
            pred_matrix[name] = {
                "flow": {t: -flat_dot(v_per_t[t][0], unit) for t in timestep_grid},
                "cos": {t: flat_dot(v_per_t[t][1], unit) for t in timestep_grid},
            }
        theta0 = [p.detach().clone() for p in align.params]
        theta_norm = float(torch.sqrt(sum((p.double() ** 2).sum() for p in theta0)))
        # 4. baseline and gate 1
        base = run_val()
        ref = next((r for r in val_rows if r["step"] == step), None)
        gate1 = None
        if ref is not None:
            gate1 = abs(base[fkey] - ref[fkey]) <= tol and abs(base[akey] - ref[akey]) <= tol
            if not gate1:
                raise SystemExit(
                    f"VIRTUAL_STEPS gate 1 FAIL at {path}: baseline flow {base[fkey]:.5f} / align {base[akey]:.5f} vs val.jsonl {ref[fkey]:.5f} / {ref[akey]:.5f}"
                )
        rows = []
        for name, d in dirs.items():
            dn = flat_norm(d)
            unit = OrderedDict((b, d[b] / dn) for b in align.blocks)
            upd = GP.unflatten(align.groups, unit)
            pred_flow = -flat_dot(v_fm, unit)  # d loss / d eta' per unit of ||delta theta||
            pred_cos = flat_dot(v_al, unit)  # cos = -align loss
            for eta in VS_ETAS:
                scale = eta * theta_norm
                m = {}
                for sign, way in ((1.0, "descent"), (-1.0, "ascent")):
                    with torch.no_grad():
                        for n, p, p0 in zip(align.names, align.params, theta0):
                            p.copy_(p0 - sign * scale * upd[n].to(device=p.device, dtype=p.dtype))
                    m[way] = run_val()
                    with torch.no_grad():
                        for p, p0 in zip(align.params, theta0):
                            p.copy_(p0)
                d = {
                    way: {"flow": m[way][fkey] - base[fkey], "cos": -(m[way][akey] - base[akey])} for way in m
                }
                row = {
                    "direction": name,
                    "eta": eta,
                    "step_norm": scale,
                    "d_flow_descent": d["descent"]["flow"],
                    "d_flow_ascent": d["ascent"]["flow"],
                    "d_cos_descent": d["descent"]["cos"],
                    "d_cos_ascent": d["ascent"]["cos"],
                    "first_order_flow": (d["descent"]["flow"] - d["ascent"]["flow"]) / 2,
                    "curvature_flow": (d["descent"]["flow"] + d["ascent"]["flow"]) / 2,
                    "first_order_cos": (d["descent"]["cos"] - d["ascent"]["cos"]) / 2,
                    "curvature_cos": (d["descent"]["cos"] + d["ascent"]["cos"]) / 2,
                    "pred_first_order_flow": pred_flow * scale,
                    "pred_first_order_cos": pred_cos * scale,
                    "d_flow_per_t_descent": {
                        t: m["descent"][f"val/flow_loss_t{t}"] - base[f"val/flow_loss_t{t}"]
                        for t in timestep_grid
                    },
                    "d_cos_per_t_descent": {
                        t: -(m["descent"][f"val/align_loss_t{t}"] - base[f"val/align_loss_t{t}"])
                        for t in timestep_grid
                    },
                    "first_order_flow_per_t": {
                        t: (m["descent"][f"val/flow_loss_t{t}"] - m["ascent"][f"val/flow_loss_t{t}"]) / 2
                        for t in timestep_grid
                    },
                    "first_order_cos_per_t": {
                        t: -(m["descent"][f"val/align_loss_t{t}"] - m["ascent"][f"val/align_loss_t{t}"]) / 2
                        for t in timestep_grid
                    },
                    "pred_first_order_flow_per_t": {
                        t: pred_matrix[name]["flow"][t] * scale for t in timestep_grid
                    },
                    "pred_first_order_cos_per_t": {
                        t: pred_matrix[name]["cos"][t] * scale for t in timestep_grid
                    },
                }
                rows.append(row)
                print(
                    f"VIRTUAL_STEP step={step} dir={name:4s} eta={eta:g} flow: descent {row['d_flow_descent']:+.5f} ascent {row['d_flow_ascent']:+.5f} "
                    f"first-order {row['first_order_flow']:+.5f} (pred {row['pred_first_order_flow']:+.5f}) curvature {row['curvature_flow']:+.5f} | "
                    f"cos: first-order {row['first_order_cos']:+.5f} (pred {row['pred_first_order_cos']:+.5f}) curvature {row['curvature_cos']:+.5f}",
                    flush=True,
                )
        # gate 2: restored bit for bit
        gate2 = all(torch.equal(p.detach(), p0) for p, p0 in zip(align.params, theta0))
        if not gate2:
            raise SystemExit(f"VIRTUAL_STEPS gate 2 FAIL at {path}: LoRA parameters not restored")
        rec = {
            "checkpoint": path,
            "step": step,
            "theta_norm": theta_norm,
            "n_grad_terms": n_terms,
            "n_val_terms": n_val_terms,
            "norm_fm_train": flat_norm(acc_fm),
            "norm_sem_train": flat_norm(acc_al),
            "norm_fm_val": flat_norm(v_fm),
            "norm_sem_val": flat_norm(v_al),
            "cos_fm_sem_train": cosine(acc_fm, acc_al),
            "cos_fm_sem_val": cosine(v_fm, v_al),
            "cos_fm_train_val": cosine(acc_fm, v_fm),
            "cos_sem_train_val": cosine(acc_al, v_al),
            "projected_units": int(hits.sum()),
            "cos_proj_vs_sem": cosine(g_proj, acc_al),
            "per_t_grad": per_t_stats,
            "pred_matrix_per_unit_step": pred_matrix,
            "stepped_per_t_directions": bool(cfg.align.vs_per_t),
            "baseline": {
                "flow": base[fkey],
                "cos": -base[akey],
                "flow_per_t": {t: base[f"val/flow_loss_t{t}"] for t in timestep_grid},
                "cos_per_t": {t: -base[f"val/align_loss_t{t}"] for t in timestep_grid},
            },
            "gate1_baseline_matches_val_jsonl": gate1,
            "gate2_params_restored": gate2,
            "rows": rows,
            "seconds": time.time() - t0,
        }
        out["checkpoints"].append(rec)
        print(
            f"VIRTUAL_STEPS {cfg.align.arm} step={step} |theta|={theta_norm:.3f} cos(fm,sem) train={rec['cos_fm_sem_train']:+.4f} val={rec['cos_fm_sem_val']:+.4f} "
            f"cos(train,val) fm={rec['cos_fm_train_val']:.3f} sem={rec['cos_sem_train_val']:.3f} base flow={base[fkey]:.4f} cos={-base[akey]:.4f} "
            f"gate1={'PASS' if gate1 else ('n/a' if gate1 is None else 'FAIL')} gate2={'PASS' if gate2 else 'FAIL'} ({rec['seconds']:.0f}s)",
            flush=True,
        )
        print(
            f"VIRTUAL_STEPS_PER_T step={step} coherence fm {[round(per_t_stats['coherence_fm'][t], 3) for t in cfg.align.meas_t]} "
            f"sem {[round(per_t_stats['coherence_sem'][t], 3) for t in cfg.align.meas_t]} "
            f"cos(fm,sem) {[round(per_t_stats['cos_fm_sem'][t], 3) for t in cfg.align.meas_t]} (t = {list(cfg.align.meas_t)})",
            flush=True,
        )
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    print(f"VIRTUAL_STEPS_OK {out_path}", flush=True)


def _align_ckpt_fields(cfg, align: AL.AlignState | None, global_step: int) -> dict | None:
    """Extra checkpoint fields of an alignment run: what tasks/extraction.py stamps into the
    provenance of features extracted from it, plus the projector so the run can be inspected."""
    if align is None:
        return None
    return {
        "align": {
            **cfg.align.identity(),
            # identity() carries the CONFIG weight; the fm arm trains with 0 whatever it says, so
            # the block's `weight` is the EFFECTIVE one and the config value moves to weight_config
            # (review 2026-10-15; tasks/extraction.py copies this block into every adapted cache).
            "weight_config": cfg.align.weight,
            "weight": align.w,
            "target_info": align.target.describe(),
            "w_eff": align.w,
            "fm_weight": align.w_fm,
            "stop_step": cfg.align.stop_step,
            "w_eff_at_checkpoint": 0.0
            if (cfg.align.stop_step and global_step > cfg.align.stop_step)
            else align.w,
            "global_step": global_step,
            "projector_state_dict": align.projector.state_dict(),
            "git10m": cfg.git10m.identity(),
        }
    }
