from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import repeat
import wandb

from models.flux.adapter import FluxModel
from models.flux.feat_flux import Featurizer4Eval, prepare
from models.lora import lora_wrap_flux
from registry import register_task
from utils import to_jsonable

from utils import seed_worker
from .utils import _flatten_scalars_into, evaluate_probe, extract_features, train_probe


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
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "cfg": to_jsonable(cfg),
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


# Channels zeroed before offline DiTF normalization, matching every offline probe in
# experiments/ (traj_readout.DISCARD_CHANNELS et al.): the two massive-activation channels.
PROBE_DISCARD_CHANNELS = [154, 1446]


def _apply_ditf_normalization(feats: np.ndarray, mods: np.ndarray, discard: list[int]) -> np.ndarray:
    """Offline DiTF normalization on pooled multi-timestep features. feats (N, K, C),
    mods (K, 3, C). Mirrors experiments/traj_readout.apply_ditf_normalization -- a copy, not
    an import, because src/tasks cannot depend on experiments/; change both together.
    Sequence: zero discard channels, LayerNorm over C, adaLN (1+scale)*x+shift, L2 norm.
    Makes the per-timestep probe features directly comparable to the single-timestep path,
    which applies the same treatment inside model.extract.
    """
    x = feats.astype(np.float64).copy()
    if discard:
        bad = [ch for ch in discard if ch < 0 or ch >= x.shape[-1]]
        if bad:
            raise ValueError(f"discard channels {bad} out of range for feature dim {x.shape[-1]}")
        x[:, :, discard] = 0.0
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased variance, matches nn.LayerNorm
    x = (x - mu) / np.sqrt(var + 1e-6)
    x = (1.0 + mods[None, :, 1, :].astype(np.float64)) * x + mods[None, :, 0, :].astype(np.float64)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return x.astype(np.float32)


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


def _sample_timesteps(
    grid: list[int],
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
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
) -> dict:
    """
    One MIM training microbatch. Caller handles gradient accumulation, optimizer step,
    logging, and checkpointing. mask_token is applied to img tokens before
    the Flux forward so attention can propagate context into masked positions.
    Flow loss is computed only at unmasked positions.

    Returns batch_metrics dict (train/mim_loss and diagnostics).
    Input-masked MIM microbatch.
    """
    model = featurizer.model
    vae = featurizer.ae
    model.train()
    vae.eval()

    imgs = batch["img"].to(device)
    with torch.no_grad():
        latents = vae.encode(imgs).to(torch.bfloat16)

    # Per-example t over the probe grid; (B,) for the model, (B,1,1,1) to broadcast over CHW.
    t = _sample_timesteps(timestep_grid, imgs.shape[0], device)
    t_b = t.view(-1, *([1] * (latents.dim() - 1)))

    noise = torch.randn_like(latents).to(device)
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

    v_pred = model(
        img=img_masked,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=y,
        timesteps=t.to(latents.dtype),
        guidance=guidance_vec,
    )

    v_target, _ = prepare(noise - latents)
    v_target = v_target.to(device=device, dtype=v_pred.dtype)

    # Flow loss: unmasked positions only — masked positions have no reliable target.
    per_token_flow = ((v_pred.float() - v_target.float()) ** 2).mean(dim=-1)
    flow_loss = (per_token_flow * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)

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

    (total_loss / grad_accum_steps).backward()

    return {
        **_flow_metrics(v_pred, v_target, mask=mask, prefix="train"),
        **mim_metrics,
        "train/total_loss": float(total_loss.detach().cpu()),
        # Loss magnitude varies strongly with t, so the mean t of the batch is needed to read
        # the training curve at all once t is sampled rather than fixed.
        "train/t_mean": float(t.float().mean().detach().cpu()),
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
) -> dict:
    model = featurizer.model
    vae = featurizer.ae
    model.eval()
    vae.eval()

    val_metrics: dict = {}
    per_t_flow: dict[int, list[float]] = {t_nom: [] for t_nom in timestep_grid}
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

                v_pred = model(
                    img=img_masked,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=y,
                    timesteps=t.to(latents.dtype),
                    guidance=guidance_vec,
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

        return val_metrics
    finally:
        vae.reg.sample = reg_sample_prev


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
        # Entry-time (before any model/dataset load): both caps re-index the train split,
        # so exclude_probe_indices' stored indices would refer to DIFFERENT images.
        if cfg.model.ensemble_size != 1:
            # Entry-time (was in the probe section AFTER training -- a non-sweep launch at
            # the ensemble_size=8 default would train for days and THEN die, 2026-09-11).
            raise ValueError(
                f"finetune-diffusion requires model.ensemble_size=1 (got "
                f"{cfg.model.ensemble_size}): every frozen comparison cache is ens1, and "
                f"ens8 makes the post-training probe 8x the forwards. Pass "
                f"--model.ensemble-size 1."
            )
        if cfg.exclude_probe_indices and cfg.max_samples is not None:
            raise ValueError(
                "exclude_probe_indices cannot be combined with max_samples: the smoke cap "
                "re-indexes the train split. Smoke-test WITHOUT the exclusion (a warning "
                "is printed); real runs use the exclusion without max_samples."
            )
        if cfg.exclude_probe_indices and cfg.label_fraction < 1.0:
            raise ValueError(
                "exclude_probe_indices cannot be combined with label_fraction < 1: the "
                "label-fraction subsample re-indexes the train split. Apply label scarcity "
                "at the PROBE -- the flow objective is label-free, so subsampling adapter "
                "training by label fraction has no meaning."
            )
        use_mim = cfg.mim_loss_weight != 0
        if not use_mim and cfg.mask_ratio > 0:
            raise ValueError(
                f"mask_ratio={cfg.mask_ratio} with mim_loss_weight=0 is incoherent: input "
                "masking exists to serve the MIM objective, and with MIM off it only hides "
                f"{cfg.mask_ratio:.0%} of the flow target behind a mask token that now "
                "receives no gradient. Set mask_ratio=0 for a pure flow-matching finetune."
            )
        print(
            f"finetune-diffusion: timesteps={timestep_grid}  guidance={cfg.guidance_scale}  "
            f"mim_loss_weight={cfg.mim_loss_weight}  mask_ratio={cfg.mask_ratio}",
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

        # Freeze base weights, then insert LoRA adapters at the target block.
        for param in flux.parameters():
            param.requires_grad = False

        lora_wrap_flux(
            flux,
            cfg.k,
            cfg.lora_rank,
            cfg.lora_alpha,
            cfg.lora_dropout,
            wrap_o=cfg.wrap_output,
        )

        # Only the LoRA A/B matrices should be trainable.
        for name, param in flux.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                param.requires_grad = True

        trainable_params = [p for p in flux.parameters() if p.requires_grad]
        if use_mim:
            trainable_params.append(mask_token)
            trainable_params.extend(decoder.parameters())
        # With MIM off, mask_token and the decoder receive no gradient; handing them to the
        # optimizer anyway would leave dead parameters in the checkpoint's param groups.

        if len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found in FLUX after LoRA wrapping. "
                "Please check configuration and LoRA wrapping logic."
            )

        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.finetune_lr, weight_decay=cfg.lora_wd)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, cfg.warmup_steps),
        )

        loaders = dataset.get_data(cfg)
        train_loader = loaders["train"]
        test_loader = loaders["test"]

        # Strided validation subset (see VAL_MAX_IMAGES). Deterministic, class-covering,
        # identical at every evaluation.
        from datasets.utils import _strided_indices

        _test_ds = test_loader.dataset
        val_loader = DataLoader(
            torch.utils.data.Subset(_test_ds, _strided_indices(len(_test_ds), VAL_MAX_IMAGES)),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # COMPLEMENT SPLIT (blocker 4, RESEARCH_NOTES 6n): exclude the probe subset from
        # training, keyed on the STORED subset_indices of the given cache -- never
        # re-derived, so it is exact even if the sampling code changes. Without this the
        # adapter trains on the very images the probes evaluate.
        if cfg.exclude_probe_indices:
            _cache = np.load(cfg.exclude_probe_indices)
            excl = _cache["subset_indices"]
            base_ds = train_loader.dataset
            n_total = len(base_ds)
            if excl.max() >= n_total:
                raise ValueError(
                    f"exclude_probe_indices max index {excl.max()} >= train split size "
                    f"{n_total} -- wrong dataset or wrong cache? "
                    f"({cfg.exclude_probe_indices})"
                )
            # A wrong-dataset cache whose indices happen to FIT would pass the size check
            # and silently exclude 5000 WRONG images while the real probe subset trains
            # into the adapter (2026-09-11 review, finding 1). The cache stores the labels
            # of its subset; they must match this dataset's labels at those indices.
            if "labels" in _cache.files:
                ds_labels = np.array([c for _, c, _ in base_ds.samples])[excl]
                if not np.array_equal(ds_labels, _cache["labels"]):
                    raise ValueError(
                        "exclude_probe_indices cache labels do not match this dataset's "
                        "labels at those indices -- the cache is from a DIFFERENT dataset "
                        f"or split ({cfg.exclude_probe_indices}). Excluding it would leave "
                        "the real probe subset in training under a clean +excl run name."
                    )
            keep = np.setdiff1d(np.arange(n_total), excl)
            if len(keep) + len(excl) != n_total:
                raise ValueError("exclusion arithmetic failed -- duplicate indices in cache?")
            train_loader = DataLoader(
                torch.utils.data.Subset(base_ds, keep.tolist()),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
            print(
                f"complement split: training on {len(keep)} of {n_total} train images "
                f"({len(excl)} probe images EXCLUDED, from {cfg.exclude_probe_indices})",
                flush=True,
            )
        else:
            print(
                "WARNING: no --exclude-probe-indices set -- the adapter will train on the "
                "probe subset. Fine for smoke tests; NOT fine for any frozen-vs-adapted "
                "comparison (blocker 4).",
                flush=True,
            )

        history = []
        global_step = 0
        micro_step = 0
        train_iter = _cycle_loader(train_loader)

        grad_accum_steps = cfg.gradient_accumulation_steps if cfg.use_gradient_accumulation else 1
        log_train_steps = cfg.log_train_steps
        log_val_steps = cfg.log_val_steps

        accum_metrics: dict = {}
        last_train_metrics: dict = {}
        last_val_metrics: dict = {}

        optimizer.zero_grad(set_to_none=True)

        while global_step < cfg.max_train_steps:
            batch = next(train_iter)

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
            )

            accum_metrics = _add_batch_metrics(accum_metrics, batch_metrics)
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_train_metrics = _average_batch_metrics(accum_metrics)
            accum_metrics = {}
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
                )
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
            )

        # NOTE (2026-09-10 review, finding 6): the probe below evaluates the FINAL weights
        # at step max_train_steps, NOT the best-val checkpoint -- even though the val
        # determinism work (VAL_NOISE_SEED, posterior-mean encoding, every image x every t)
        # exists to make best_val_loss meaningful. That is deliberate for now: reloading
        # lora_best would make the reported number depend on the val-selection rule, and the
        # frozen-vs-adapted comparison wants a fixed training budget on both sides. If you
        # change this, change it on BOTH arms and re-run everything.
        #
        # After model tuning, evaluate frozen model with small classifier head to evaluate feature quality.
        # Freeze current model weights (including LoRA adapters) and train a small classifier head on top of captured features
        # for classification. This probes whether the adapted features are more linearly separable for the downstream task.
        for param in flux.parameters():
            param.requires_grad = False

        # Use the same train/test loader to prevent data leakage.
        if not hasattr(dataset, "category_list"):
            raise ValueError("Dataset must have category_list attribute for classification probe evaluation.")
        class_names = dataset.category_list
        num_classes = len(class_names)

        # PROBE EXTRACTION LOADERS -- deliberately NOT the training loaders.
        # (a) make_loaders sets shuffle=True on train, but multi-t extract_features draws
        #     per-image eps sequentially from a seeded generator and REQUIRES a deterministic
        #     order (its docstring; ExtractionTask builds its own unshuffled loader for the
        #     same reason). With the training loader the order also depends on how much
        #     global RNG training consumed, so features would not be reproducible or
        #     eps-paired with any frozen re-extraction.
        # (b) ensemble_size defaults to 8; at 7 timesteps over a full split that is ~8x the
        #     forwards AND protocol-incomparable with the frozen ens1 caches.
        # Both were live before the 2026-09-10 review (findings 3 and 4).
        probe_train_loader = DataLoader(
            train_loader.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        probe_test_loader = test_loader  # already shuffle=False

        multi_t = isinstance(cfg.t, list)
        if multi_t:
            # Multi-timestep extract_features returns PRE-norm (N, K, C) + adaLN mods; the
            # single-timestep path returns post-norm (N, C) from model.extract. Apply DiTF
            # offline so the per-timestep probe curve is readable -- with training sampling t
            # over the grid, that curve is the primary readout of WHERE adaptation moved the
            # representation.
            #
            # NOT identical to the single-t path, in two ways (2026-09-10 review, finding 5):
            #   (i)  model.extract LayerNorms PER TOKEN then pools; this pools then
            #        LayerNorms. That is the documented pooled-vs-per-token approximation
            #        every offline probe in experiments/ uses, so the multi-t numbers are
            #        comparable to THOSE, not to the single-t branch below.
            #   (ii) channel discard follows cfg.cd, matching model.extract's behaviour,
            #        instead of being unconditional.
            # Never compare a multi_t cell against a single-t cell directly.
            train_raw, train_labels, train_mods = extract_features(cfg, model, probe_train_loader, "train")
            test_raw, test_labels, test_mods = extract_features(cfg, model, probe_test_loader, "test")
            if not np.allclose(train_mods, test_mods, atol=1e-5):
                raise RuntimeError(
                    "adaLN mods differ between train and test extraction; they depend only "
                    "on t and the null embeds, so this indicates a timestep mismatch."
                )
            discard = PROBE_DISCARD_CHANNELS if getattr(cfg, "cd", False) else []
            print(
                f"[probe] offline DiTF normalization, discard_channels={discard} (cfg.cd={getattr(cfg, 'cd', False)})"
            )
            # Persist PRE-norm features + mods BEFORE any probing: extraction over the
            # complement+test splits is tens of GPU-hours, and without this file a crash
            # (or SLURM expiry) anywhere downstream forces a full re-extraction
            # (2026-09-11 review, finding 4). Re-analysis is offline from here.
            feats_path = os.path.join(cfg.save_dir, "adapted_probe_feats_prenorm.npz")
            os.makedirs(cfg.save_dir, exist_ok=True)
            np.savez(
                feats_path,
                train_feats=train_raw,
                train_labels=train_labels,
                test_feats=test_raw,
                test_labels=test_labels,
                mods=train_mods,
                timesteps=np.array(cfg.t),
                cd=np.array(bool(getattr(cfg, "cd", False))),
            )
            print(f"[probe] wrote pre-norm features: {feats_path}", flush=True)
            train_by_t = _apply_ditf_normalization(train_raw, train_mods, discard)
            test_by_t = _apply_ditf_normalization(test_raw, test_mods, discard)
            probe_slices = [
                (f"t{t_nom}", train_by_t[:, i_t, :], test_by_t[:, i_t, :]) for i_t, t_nom in enumerate(cfg.t)
            ]
        else:
            train_feats, train_labels = extract_features(cfg, model, probe_train_loader, "train")
            test_feats, test_labels = extract_features(cfg, model, probe_test_loader, "test")
            probe_slices = [("t{}".format(cfg.t), train_feats, test_feats)]

        per_t_results: dict[str, dict] = {}
        best = None  # (top1, tag, preds)
        for tag, tr_x, te_x in probe_slices:
            probe, steps, elapsed = train_probe(
                cfg.probe_type,
                tr_x,
                train_labels,
                num_epochs=cfg.clf_epochs,
                lr=cfg.clf_lr,
                batch_size=cfg.clf_batch_size,
                device=torch.device(cfg.device),
                num_classes=num_classes,
                grid_size=cfg.grid_size,
                polynomial_order=cfg.polynomial_order,
            )
            top1, macro_f1, weighted_f1, per_class_f1 = evaluate_probe(
                probe, te_x, test_labels, torch.device(cfg.device), num_classes=num_classes
            )

            # per-class accuracy and F1 breakdown
            probe.eval()
            with torch.no_grad():
                X = torch.from_numpy(te_x).float().to(device)
                preds = probe(X).cpu().numpy().argmax(axis=1)
            per_class_acc: dict[str, float] = {}
            per_class_f1_dict: dict[str, float] = {}
            for cls_idx, cls_name in enumerate(class_names):
                cls_mask = test_labels == cls_idx
                if not cls_mask.any():
                    # Class absent from this test split: accuracy is undefined (an empty
                    # mean is nan + a warning) and f1 would read as a real 0.0. Report None.
                    per_class_acc[cls_name] = None
                    per_class_f1_dict[cls_name] = None
                    continue
                cls_acc = (preds[cls_mask] == test_labels[cls_mask]).mean() * 100.0
                per_class_acc[cls_name] = round(float(cls_acc), 2)
                per_class_f1_dict[cls_name] = round(float(per_class_f1[cls_idx]), 2)

            per_t_results[tag] = {
                "top1_accuracy": round(float(top1), 2),
                "macro_f1": round(float(macro_f1), 2),
                "weighted_f1": round(float(weighted_f1), 2),
                "training_steps": steps,
                "wall_clock_seconds": round(elapsed, 2),
                "per_class_accuracy": per_class_acc,
                "per_class_f1": per_class_f1_dict,
            }
            print(f"[probe] {tag}: top1={top1:.2f} macro_f1={macro_f1:.2f}")
            if best is None or top1 > best[0]:
                best = (top1, tag, preds)

        best_top1, best_tag, best_preds = best
        # PROTOCOL STATUS (2026-09-11 review, finding 6, DECIDED): this in-task probe is a
        # DIAGNOSTIC, never the frozen-vs-adapted headline. It trains on the 16.6k-image
        # complement and evaluates on the test split -- a protocol no frozen arm exists
        # under (the frozen numbers are 5-fold CV inside the n5000 subset). The reportable
        # frozen-vs-adapted delta comes from the OFFLINE probe battery on the n5000 probe
        # subset, re-extracted identically on both arms (task=extract, with/without
        # --lora-checkpoint). Comparing this table against any frozen number conflates
        # probe-train size and eval population with adaptation.
        # NOT a headline number: best_tag is an argmax over len(grid) probes all scored on
        # the SAME test split, so top1_at_best_timestep carries a winner's curse over
        # correlated probes (the concat_trajectory defect class, RESEARCH_NOTES 6m item 1).
        # The reportable quantities are the per-timestep table and, for frozen-vs-adapted,
        # the per-timestep DELTA -- never this max. Named to make misuse obvious, and the
        # mean over the grid is provided as a selection-free scalar.
        mean_top1 = float(np.mean([r["top1_accuracy"] for r in per_t_results.values()]))
        probe_results = {
            "label_fraction_pct": cfg.label_fraction,
            "argmax_timestep_SELECTED_ON_TEST": best_tag,
            "top1_at_best_timestep_BIASED": round(float(best_top1), 2),
            "mean_top1_over_timesteps": round(mean_top1, 2),
            "per_timestep": per_t_results,
        }

        flat = {}
        _flatten_scalars_into("probe", probe_results, flat)
        # One confusion matrix (the best timestep's) -- one per t would be 7 wandb media
        # panels of marginal value.
        flat["probe/confusion_matrix"] = wandb.plot.confusion_matrix(
            y_true=test_labels.tolist(),
            preds=best_preds.tolist(),
            class_names=class_names,
        )
        wandb.log(flat)

        return {
            "history": history,
            "probe_results": probe_results,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "num_trainable_params": sum(p.numel() for p in trainable_params),
        }
