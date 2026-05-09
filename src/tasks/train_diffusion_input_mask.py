"""
Input-masked MIM fine-tuning of Flux-dev on EuroSAT.

Difference from train_diffusion.py (post-forward feature masking):
  - mask_token lives in input space (in_channels=64), not feature space.
  - Masking is applied to img tokens before the Flux forward pass.
  - Flow loss is restricted to unmasked positions only
  - Features at masked positions contain context propagated through
    Flux's attention rather than a constant mask_token substituted after
    the forward
"""
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

from .train_diffusion import (
    MIMDecoder,
    _FeatureCapture,
    _add_batch_metrics,
    _average_batch_metrics,
    _cycle_loader,
    _expand_null_embeddings,
    _flow_metrics,
    _get_lora_state_dict,
    _mim_metrics,
    _random_masking,
    _save_lora_checkpoint,
)


def _fine_tune_diffusion_input_mask_microbatch(
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
    Input-masked MIM microbatch. mask_token is applied to img tokens before
    the Flux forward so attention can propagate context into masked positions.
    Flow loss is computed only at unmasked positions.
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
    img_masked, mask = _random_masking(img, mask_token, mask_ratio)

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
        **_flow_metrics(v_pred, v_target, prefix="train"),
        **_mim_metrics(pred_clean, target_clean, mask, prefix="train"),
        "train/total_loss": float(total_loss.detach().cpu()),
    }


@torch.no_grad()
def _validate_diffusion_input_mask(
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

        img_masked, mask = _random_masking(img, mask_token, mask_ratio)

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
            **_flow_metrics(v_pred, v_target, prefix="val"),
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


@register_task("finetune-diffusion-input-mask")
class FinetuneDiffusionInputMaskTask:
    def run(self, cfg, model: FluxModel, dataset) -> dict:
        featurizer_model: Featurizer4Eval = model._inner
        device = torch.device(cfg.device)

        tb_dir = os.path.join(cfg.save_dir, "tensorboard_logs")
        writer = SummaryWriter(log_dir=tb_dir)

        best_val_loss = float("inf")

        flux = featurizer_model.model
        ae = featurizer_model.ae
        flux.to(device)
        ae.to(device)

        # mask_token lives in input token space (in_channels), not feature space (hidden_size).
        mask_token = torch.nn.Parameter(torch.zeros(1, 1, flux.in_channels, device=device))
        decoder = MIMDecoder(flux.hidden_size, flux.in_channels).to(device=device, dtype=torch.bfloat16)

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

        if len(cfg.label_fractions) != 1:
            raise ValueError(
                "Currently, only a single label fraction is supported for FinetuneDiffusionInputMaskTask. "
                "Received: %s" % cfg.label_fractions
            )

        for param in flux.parameters():
            param.requires_grad = False

        lora_wrap_flux(flux, cfg.k, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, wrap_o=cfg.wrap_output)

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
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(1, cfg.warmup_steps)
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

            batch_metrics = _fine_tune_diffusion_input_mask_microbatch(
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
                val_metrics = _validate_diffusion_input_mask(
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

                history.append({
                    "global_step": global_step,
                    "train_metrics": step_train_metrics,
                    "val_metrics": val_metrics,
                })

                print(
                    f"[finetune-diffusion-input-mask] step={global_step} "
                    f"train_total={step_train_metrics['train/total_loss']:.6f} "
                    f"train_flow={step_train_metrics['train/flow_mse_loss']:.6f} "
                    f"train_mim={step_train_metrics['train/mim_loss']:.6f} "
                    f"val_total={val_metrics['val/total_loss']:.6f} "
                    f"val_flow={val_metrics['val/flow_mse_loss']:.6f} "
                    f"val_mim={val_metrics['val/mim_loss']:.6f}"
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

        capture.remove()
        writer.close()

        return {
            "history": history,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "num_trainable_params": sum(p.numel() for p in trainable_params),
        }
