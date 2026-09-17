from __future__ import annotations

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
from datasets.utils import _strided_indices
from registry import register_task
from utils import to_jsonable

from utils import seed_worker
from sklearn.metrics import f1_score

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
    probe_head: torch.nn.Module = None,
    supervised: bool = False,
    freeze_backbone: bool = False,
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
        detach_features=not supervised,
    )
    feat_tokens = up_ft[0]  # (B, L, C); single blocks also append mod at up_ft[1]
    feat_pooled = feat_tokens.float().mean(dim=1)  # (B, C) spatial mean, matches POOLING

    v_target, _ = prepare(noise - latents)
    v_target = v_target.to(device=device, dtype=v_pred.dtype)

    # Flow loss: unmasked positions only — masked positions have no reliable target.
    per_token_flow = ((v_pred.float() - v_target.float()) ** 2).mean(dim=-1)
    flow_loss = (per_token_flow * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)

    # Concurrent linear probe (2026-09-13 protocol realignment): trained on the SAME
    # batches, at each batch's sampled t, Scale-MAE-style (probe on full train split,
    # eval on the official test split -- no complement machinery).
    labels = batch["label"].to(device)
    logits = probe_head(feat_pooled)
    probe_ce = F.cross_entropy(logits, labels)
    probe_acc = float((logits.argmax(dim=1) == labels).float().mean().cpu())

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

    # CE is scaled by 1/grad_accum_steps in EVERY arm and the head is stepped by the
    # caller at the accumulation boundary in EVERY arm, so the head's update rule is
    # identical across the three modes (rule 8; 2026-09-16 review, M1).
    if supervised:
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
        (probe_ce / grad_accum_steps).backward()

    return {
        **_flow_metrics(v_pred, v_target, mask=mask, prefix="train"),
        **mim_metrics,
        "train/total_loss": float(total_loss.detach().cpu()),
        "train/probe_ce": float(probe_ce.detach().cpu()),
        "train/probe_acc": probe_acc,
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
    probe_head: torch.nn.Module = None,
) -> dict:
    model = featurizer.model
    vae = featurizer.ae
    model.eval()
    vae.eval()
    if probe_head is not None:
        probe_head.eval()

    val_metrics: dict = {}
    per_t_flow: dict[int, list[float]] = {t_nom: [] for t_nom in timestep_grid}
    per_t_probe: dict[int, list[float]] = {t_nom: [] for t_nom in timestep_grid}
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
        use_mim = cfg.mim_loss_weight != 0
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
                else "label-free flow adaptation"
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
        num_classes = len(dataset.category_list)
        probe_head = torch.nn.Linear(flux.hidden_size, num_classes).to(device=device, dtype=torch.float32)
        probe_optimizer = torch.optim.AdamW(probe_head.parameters(), lr=cfg.clf_lr, weight_decay=0.0)

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
        test_loader = loaders["test"]

        # Strided validation subset (see VAL_MAX_IMAGES). Deterministic, class-covering,
        # identical at every evaluation.
        _test_ds = test_loader.dataset
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
        train_iter = _cycle_loader(train_loader)

        grad_accum_steps = cfg.gradient_accumulation_steps if cfg.use_gradient_accumulation else 1
        log_train_steps = cfg.log_train_steps
        log_val_steps = cfg.log_val_steps

        accum_metrics: dict = {}
        last_train_metrics: dict = {}
        last_val_metrics: dict = {}

        if optimizer is not None:
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
                probe_head=probe_head,
                supervised=cfg.supervised_finetune,
                freeze_backbone=cfg.freeze_backbone,
            )

            accum_metrics = _add_batch_metrics(accum_metrics, batch_metrics)
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            # Head steps at the boundary in every arm, unclipped and unscheduled (M1).
            probe_optimizer.step()
            probe_optimizer.zero_grad(set_to_none=True)
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
                    probe_head=probe_head,
                )
                probe_head.train()
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
                        probe_head=probe_head,
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
            )

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
            per_t_f1[t_nom] = round(float(f1_score(y_true, y_hat, average="macro") * 100.0), 2)
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
        }
