from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch.utils.tensorboard import SummaryWriter

from models.flux.adapter import FluxModel
from models.flux.feat_flux import Featurizer4Eval, prepare
from models.lora import lora_wrap_flux
from registry import register_task
from utils import to_jsonable

from .utils import evaluate_probe, extract_features, log_scalars_recursive, train_probe


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

    checkpoint_path = checkpoint_dir / f"{name}_step{global_step}.pt"

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


def _fine_tune_diffusion_microbatch(
    cfg,
    featurizer: Featurizer4Eval,
    timestep: int,
    batch: dict,
    device: torch.device,
    grad_accum_steps: int,
    mask_token: torch.nn.Parameter,
    mask_ratio: float,
    decoder: MIMDecoder,
    capture: _FeatureCapture,
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

    t = timestep / 1000.0

    imgs = batch["img"].to(device)
    with torch.no_grad():
        latents = vae.encode(imgs).to(torch.bfloat16)

    noise = torch.randn_like(latents).to(device)
    latents_noisy = t * noise + (1.0 - t) * latents

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

    capture.features = None

    v_pred = model(
        img=img_masked,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=y,
        timesteps=torch.full((imgs.shape[0],), t, device=device, dtype=latents.dtype),
        guidance=guidance_vec,
    )

    if capture.features is None:
        raise RuntimeError(
            f"Feature capture returned None — check that block_idx={cfg.k} is valid "
            "and the hook was registered before model.forward() was called."
        )

    v_target, _ = prepare(noise - latents)
    v_target = v_target.to(device=device, dtype=v_pred.dtype)

    # Flow loss: unmasked positions only — masked positions have no reliable target.
    per_token_flow = ((v_pred.float() - v_target.float()) ** 2).mean(dim=-1)
    flow_loss = (per_token_flow * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)

    # MIM head: decode directly from captured features.
    # Features at masked positions carry context from Flux attention — no second masking needed.
    pred_clean = decoder(capture.features)
    target_clean, _ = prepare(latents)
    target_clean = target_clean.to(device=device, dtype=pred_clean.dtype)
    per_token_mim = ((pred_clean - target_clean) ** 2).mean(dim=-1)
    mim_loss = (per_token_mim * mask).sum() / mask.sum().clamp(min=1)

    total_loss = flow_loss + cfg.mim_loss_weight * mim_loss
    (total_loss / grad_accum_steps).backward()

    return {
        **_flow_metrics(v_pred, v_target, mask=mask, prefix="train"),
        **_mim_metrics(pred_clean, target_clean, mask, prefix="train"),
        "train/total_loss": float(total_loss.detach().cpu()),
    }


@torch.no_grad()
def _validate_diffusion(
    cfg,
    featurizer: Featurizer4Eval,
    timestep: int,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_token: torch.nn.Parameter,
    mask_ratio: float,
    decoder: MIMDecoder,
    capture: _FeatureCapture,
) -> dict:
    model = featurizer.model
    vae = featurizer.ae
    model.eval()
    vae.eval()

    t = timestep / 1000.0
    val_metrics: dict = {}

    for batch in dataloader:
        imgs = batch["img"].to(device)
        latents = vae.encode(imgs).to(torch.bfloat16)

        noise = torch.randn_like(latents).to(device)
        latents_noisy = t * noise + (1.0 - t) * latents

        img, img_ids = prepare(img=latents_noisy)
        img = img.to(device, dtype=latents.dtype)
        img_ids = img_ids.to(device)

        txt, txt_ids, y = _expand_null_embeddings(
            featurizer, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
        )
        guidance_vec = torch.full((imgs.shape[0],), cfg.guidance_scale, device=device, dtype=latents.dtype)

        if mask_ratio > 0:
            img_masked, mask = _random_masking(img, mask_token, mask_ratio)
        else:
            img_masked, mask = img, _no_mask(img)

        capture.features = None

        v_pred = model(
            img=img_masked,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=torch.full((imgs.shape[0],), t, device=device, dtype=latents.dtype),
            guidance=guidance_vec,
        )

        if capture.features is None:
            raise RuntimeError(
                f"Feature capture returned None — check that block_idx={cfg.k} is valid "
                "and the hook was registered before model.forward() was called."
            )

        v_target, _ = prepare(noise - latents)
        v_target = v_target.to(device=device, dtype=v_pred.dtype)

        per_token_flow = ((v_pred.float() - v_target.float()) ** 2).mean(dim=-1)
        flow_loss = (per_token_flow * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)

        pred_clean = decoder(capture.features)
        target_clean, _ = prepare(latents)
        target_clean = target_clean.to(device=device, dtype=pred_clean.dtype)
        per_token_mim = ((pred_clean - target_clean) ** 2).mean(dim=-1)
        mim_loss = (per_token_mim * mask).sum() / mask.sum().clamp(min=1)

        batch_metrics = {
            **_flow_metrics(v_pred, v_target, mask=mask, prefix="val"),
            **_mim_metrics(pred_clean, target_clean, mask, prefix="val"),
            "val/total_loss": float((flow_loss + cfg.mim_loss_weight * mim_loss).detach().cpu()),
        }
        for k, v in batch_metrics.items():
            if k not in val_metrics:
                val_metrics[k] = []
            val_metrics[k].append(v)

    for k in val_metrics:
        val_metrics[k] = float(np.mean(np.array(val_metrics[k])))

    return val_metrics


@register_task("finetune-diffusion")
class FinetuneDiffusionTask:
    def run(self, cfg, model: FluxModel, dataset) -> dict:
        # Unwrap registered adapter to get the raw Featurizer4Eval.
        featurizer_model: Featurizer4Eval = model._inner
        device = torch.device(cfg.device)

        tb_dir = os.path.join(cfg.save_dir, "tensorboard_logs")
        writer = SummaryWriter(log_dir=tb_dir)

        best_val_loss = float("inf")

        # Move Flux and VAE to GPU; VAE stays in eval and its weights stay frozen.
        flux = featurizer_model.model
        ae = featurizer_model.ae
        flux.to(device)
        ae.to(device)

        # MIM components: mask_token replaces FLUX input latent tokens
        # before the FLUX forward, so trainable attention cannot directly see masked tokens.
        mask_token = torch.nn.Parameter(torch.zeros(1, 1, flux.in_channels, device=device))
        decoder = MIMDecoder(flux.hidden_size, flux.in_channels).to(device=device, dtype=torch.bfloat16)

        # Register feature hook on the target block.
        n_double = len(flux.double_blocks)
        block_idx_int = cfg.k if isinstance(cfg.k, int) else cfg.k[0]
        hooked_block = (
            flux.double_blocks[block_idx_int]
            if block_idx_int < n_double
            else flux.single_blocks[block_idx_int - n_double]
        )
        txt_len = featurizer_model.null_prompt_embeds.shape[1]
        capture = _FeatureCapture()
        capture.register(hooked_block, txt_len)

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
        trainable_params.append(mask_token)
        trainable_params.extend(decoder.parameters())

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
                timestep=cfg.t,
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
                for key, value in last_train_metrics.items():
                    writer.add_scalar(key, value, global_step)

            if global_step % log_val_steps == 0 or global_step == cfg.max_train_steps:
                val_metrics = _validate_diffusion(
                    cfg,
                    featurizer=featurizer_model,
                    timestep=cfg.t,
                    dataloader=test_loader,
                    device=device,
                    mask_token=mask_token,
                    mask_ratio=cfg.mask_ratio,
                    decoder=decoder,
                    capture=capture,
                )
                last_val_metrics = val_metrics

                for key, value in val_metrics.items():
                    writer.add_scalar(key, value, global_step)

                history.append(
                    {
                        "global_step": global_step,
                        "train_metrics": step_train_metrics,
                        "val_metrics": val_metrics,
                    }
                )

                print(
                    f"[finetune-diffusion] step={global_step} "
                    f"train_total={step_train_metrics['train/total_loss']:.6f} "
                    f"train_flow={step_train_metrics['train/flow_mse_loss']:.6f} "
                    f"train_mim={step_train_metrics['train/mim_loss']:.6f} "
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

        # After model tuning, evaluate frozen model with small classifier head to evaluate feature quality.
        # Freeze current model weights (including LoRA adapters) and train a small classifier head on top of captured features
        # for classification. This probes whether the adapted features are more linearly separable for the downstream task.
        for param in flux.parameters():
            param.requires_grad = False

        # Use the same train/test loader to prevent data leakage.
        train_feats, train_labels = extract_features(cfg, model, train_loader, "train")
        test_feats, test_labels = extract_features(cfg, model, test_loader, "test")

        if not hasattr(dataset, "category_list"):
            raise ValueError("Dataset must have category_list attribute for classification probe evaluation.")

        class_names = dataset.category_list
        num_classes = len(class_names)

        probe, steps, elapsed = train_probe(
            cfg.probe_type,
            train_feats,
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
            probe, test_feats, test_labels, torch.device(cfg.device)
        )

        # per-class accuracy and F1 breakdown
        probe.eval()
        with torch.no_grad():
            X = torch.from_numpy(test_feats).float().to(device)
            preds = probe(X).cpu().numpy().argmax(axis=1)
        per_class_acc: dict[str, float] = {}
        per_class_f1_dict: dict[str, float] = {}
        for cls_idx, cls_name in enumerate(class_names):
            mask = test_labels == cls_idx
            cls_acc = (preds[mask] == test_labels[mask]).mean() * 100.0
            per_class_acc[cls_name] = round(float(cls_acc), 2)
            per_class_f1_dict[cls_name] = round(float(per_class_f1[cls_idx]), 2)

        probe_results = {
            "label_fraction_pct": cfg.label_fraction,
            "top1_accuracy": round(float(top1), 2),
            "macro_f1": round(float(macro_f1), 2),
            "weighted_f1": round(float(weighted_f1), 2),
            "training_steps": steps,
            "wall_clock_seconds": round(elapsed, 2),
            "per_class_accuracy": per_class_acc,
            "per_class_f1": per_class_f1_dict,
        }

        # Log to TensorBoard under "probe/" prefix for easy comparison across runs with different label fractions.
        # NOTE: these will only have 1 timestep.
        log_scalars_recursive(writer, "probe", probe_results, step=0)

        capture.remove()
        writer.close()

        return {
            "history": history,
            "probe_results": probe_results,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "num_trainable_params": sum(p.numel() for p in trainable_params),
        }
