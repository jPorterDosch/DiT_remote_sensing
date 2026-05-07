import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch.utils.tensorboard import SummaryWriter

from models.flux.feat_flux import Featurizer4Eval, prepare
from models.flux.adapter import FluxModel
from models.lora import lora_wrap_flux
from registry import register_task
from utils import to_jsonable


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
) -> None:
    checkpoint_dir = Path(cfg.save_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{name}_step{global_step}.pt"

    ckpt = {
        "global_step": global_step,
        "lora_state_dict": _get_lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "cfg": to_jsonable(cfg),
    }

    torch.save(ckpt, checkpoint_path)


def _flow_metrics(pred: torch.Tensor, target: torch.Tensor, prefix: str) -> dict:
    """
    Helper to compute shared flow metrics for train/val loops. Computes MSE, RMSE,
    mean norms of pred/target/residual, relative flow error, and cosine similarity between pred and target.

    Interpretation:
        - MSE/RMSE: overall error magnitude. MSE is optimization objective, RMSE is more interpretable in flow units.
        - Norms: average magnitude of predicted flow, true flow, and error, which can help identify
            if model is under/over-shooting on average.
        - Relative flow error: error normalized by true flow magnitude, which can help identify if model is performing worse on smaller or larger flows.
        - Cosine similarity: directional alignment between predicted and true flow, independent of magnitude.
    """
    pred_f = pred.float()
    target_f = target.float()
    residual_f = pred_f - target_f

    pred_flat = pred_f.flatten(1)
    target_flat = target_f.flatten(1)
    residual_flat = residual_f.flatten(1)

    mse = F.mse_loss(pred_f, target_f)

    target_norm = target_flat.norm(dim=1)
    residual_norm = residual_flat.norm(dim=1)

    metrics = {
        f"{prefix}/flow_mse_loss": float(mse.detach().cpu()),
        f"{prefix}/flow_rmse": float(torch.sqrt(mse).detach().cpu()),
        f"{prefix}/pred_norm_mean": float(pred_flat.norm(dim=1).mean().detach().cpu()),
        f"{prefix}/target_norm_mean": float(target_norm.mean().detach().cpu()),
        f"{prefix}/residual_norm_mean": float(residual_norm.mean().detach().cpu()),
        f"{prefix}/relative_flow_error": float((residual_norm / (target_norm + 1e-8)).mean().detach().cpu()),
        f"{prefix}/cosine_pred_target": float(
            F.cosine_similarity(pred_flat, target_flat, dim=1).mean().detach().cpu()
        ),
    }

    return metrics


def _cycle_loader(dataloader: torch.utils.data.DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def _add_batch_metrics(accum_metrics: dict, batch_metrics: dict) -> dict:
    for key, value in batch_metrics.items():
        if key not in accum_metrics:
            accum_metrics[key] = []
        accum_metrics[key].append(value)
    return accum_metrics


def _average_batch_metrics(accum_metrics: dict) -> dict:
    return {k: float(np.mean(np.array(v))) for k, v in accum_metrics.items()}


def _fine_tune_diffusion_microbatch(
    cfg,
    featurizer: Featurizer4Eval,
    timestep: int,
    batch: dict,
    device: torch.device,
    grad_accum_steps: int,
) -> tuple[dict, int]:
    """
    Runs one training microbatch. Caller handles gradient accumulation,
        optimizer.step(), logging, validation, and checkpointing.
        - encodes images into VAE latent space
        - creates noisy/interpolated latents based on given timestep
        - computes target flow as straight-line derivative
        - optimizes MSE between FLUX predictions and target flow.
    Computes various flow metrics for analysis.

    Returns:
        - batch_metrics: dict of averaged flow metrics for this batch
    """
    model = featurizer.model
    vae = featurizer.ae
    model.train()
    vae.eval()

    guidance_scale = cfg.guidance_scale

    # Normalize timestep
    t = timestep / 1000.0  # assuming 1000 total diffusion steps (same as feat_flux.py)

    # Encode images into VAE latent space
    imgs = batch["img"].to(device)

    with torch.no_grad():
        latents = vae.encode(imgs)
        latents = latents.to(
            torch.bfloat16
        )  # FLUX is trained in bfloat16, so we convert the latents to bfloat16 before feeding into FLUX for fine-tuning.
        # Future work could explore whether training in full fp32 or using mixed precision with gradient scaling would improve performance at the cost of increased VRAM usage.
    # Create noisy/interpolated latent from given timestep

    noise = torch.randn_like(latents).to(device)

    latents_noisy = t * noise + (1.0 - t) * latents

    # Straight-line derivative wrt t
    target_flow = noise - latents

    # Patchify latents for FLUX transformer
    img, img_ids = prepare(img=latents_noisy)
    img = img.to(device, dtype=latents.dtype)
    img_ids = img_ids.to(device)

    txt, txt_ids, y = _expand_null_embeddings(
        featurizer, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
    )

    guidance_vec = torch.full((imgs.shape[0],), guidance_scale, device=device, dtype=latents.dtype)

    pred = model(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=y,
        timesteps=torch.full((imgs.shape[0],), t, device=device, dtype=latents.dtype),
        guidance=guidance_vec,
    )
    # Patchify target to match FLUX output shape.
    target, _ = prepare(target_flow)
    target = target.to(device=device, dtype=pred.dtype)

    raw_loss = F.mse_loss(pred, target)
    loss = raw_loss / grad_accum_steps  # Normalize loss for gradient accumulation
    loss.backward()

    return _flow_metrics(pred, target, prefix="train")


@torch.no_grad()
def _validate_diffusion(
    cfg,
    featurizer: Featurizer4Eval,
    timestep: int,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model = featurizer.model
    vae = featurizer.ae
    model.eval()
    vae.eval()

    guidance_scale = cfg.guidance_scale

    val_metrics: dict = {}

    # Normalize timestep
    t = timestep / 1000.0  # assuming 1000 total diffusion steps (same as feat_flux.py)

    for batch in dataloader:
        # Encode images into VAE latent space
        imgs = batch["img"].to(device)
        latents = vae.encode(imgs)
        latents = latents.to(
            torch.bfloat16
        )  # FLUX is trained in bfloat16, so we convert the latents to bfloat16 before feeding into FLUX for fine-tuning.
        # Future work could explore whether training in full fp32 or using mixed precision with gradient scaling would improve performance at the cost of increased VRAM usage.
        # Create noisy/interpolated latent from given timestep

        noise = torch.randn_like(latents).to(device)

        latents_noisy = t * noise + (1.0 - t) * latents

        # Straight-line derivative wrt t
        target_flow = noise - latents

        # Patchify latents for FLUX transformer
        img, img_ids = prepare(img=latents_noisy)
        img = img.to(device, dtype=latents.dtype)
        img_ids = img_ids.to(device)

        txt, txt_ids, y = _expand_null_embeddings(
            featurizer, batch_size=imgs.shape[0], device=device, dtype=latents.dtype
        )

        guidance_vec = torch.full((imgs.shape[0],), guidance_scale, device=device, dtype=latents.dtype)

        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=torch.full((imgs.shape[0],), t, device=device, dtype=latents.dtype),
            guidance=guidance_vec,
        )
        # Patchify target to match FLUX output shape.
        target, _ = prepare(target_flow)
        target = target.to(device=device, dtype=pred.dtype)

        batch_metrics = _flow_metrics(pred, target, prefix="val")
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
        featurizer_model: Featurizer4Eval = model._inner
        device = torch.device(cfg.device)

        tb_dir = os.path.join(cfg.save_dir, "tensorboard_logs")
        writer = SummaryWriter(log_dir=tb_dir)

        best_val_loss = float("inf")

        # Ensure model and VAE are on correct device before training.
        flux = featurizer_model.model
        ae = featurizer_model.ae
        flux.to(device)
        ae.to(device)

        # Validate label fraction -- currently, due to length of fine-tuning, only length 1 is supported.
        if len(cfg.label_fractions) != 1:
            raise ValueError(
                "Currently, only a single label fraction is supported for FinetuneDiffusionTask. Received: %s"
                % cfg.label_fractions
            )

        # Freeze base FLUX weights
        for param in flux.parameters():
            param.requires_grad = False

        # Insert LoRA adapters into selected blocks
        block_idx = cfg.k
        rank = cfg.lora_rank
        alpha = cfg.lora_alpha
        dropout = cfg.lora_dropout

        lora_wrap_flux(flux, block_idx, rank, alpha, dropout, wrap_o=cfg.wrap_output)

        # Explicitly set lora parameters to be trainable and add to/create optimizer
        for name, param in flux.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                param.requires_grad = True

        trainable_params = [param for param in flux.parameters() if param.requires_grad]

        # Sanity check
        if len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found in FLUX after LoRA wrapping. Please check configuration and LoRA wrapping logic."
            )

        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.finetune_lr, weight_decay=cfg.lora_wd)

        # Create loaders
        loaders = dataset.get_data(cfg)
        train_loader = loaders["train"]
        test_loader = loaders["test"]

        history = []
        global_step = 0
        micro_step = 0

        train_iter = _cycle_loader(train_loader)

        grad_accum_steps = cfg.gradient_accumulation_steps
        log_train_steps = cfg.log_train_steps
        log_val_steps = cfg.log_val_steps

        accum_metrics: dict = {}
        last_train_metrics: dict = {}
        last_val_metrics: dict = {}

        while global_step < cfg.max_train_steps:
            batch = next(train_iter)

            batch_metrics = _fine_tune_diffusion_microbatch(
                cfg=cfg,
                featurizer=featurizer_model,
                timestep=cfg.t,
                batch=batch,
                device=device,
                grad_accum_steps=grad_accum_steps,
            )

            accum_metrics = _add_batch_metrics(accum_metrics, batch_metrics)
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            # Optimizer step after gradient accumulation
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_train_metrics = _average_batch_metrics(accum_metrics)
            accum_metrics = {}
            last_train_metrics = step_train_metrics

            # Log training metrics at specified intervals
            if global_step % log_train_steps == 0:
                for key, value in last_train_metrics.items():
                    writer.add_scalar(key, value, global_step)

            # Run validation and log metrics at specified intervals
            if global_step % log_val_steps == 0 or global_step == cfg.max_train_steps:
                val_metrics = _validate_diffusion(
                    cfg,
                    featurizer=featurizer_model,
                    timestep=cfg.t,
                    dataloader=test_loader,
                    device=device,
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
                    f"train_loss={step_train_metrics['train/flow_mse_loss']:.6f} "
                    f"val_loss={val_metrics['val/flow_mse_loss']:.6f}"
                )

                val_loss = val_metrics["val/flow_mse_loss"]
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
                    )

        # Save final checkpoint at end of training as well
        # Guard against epoch, train_metrics, or val_metrics being unbound.
        if history:
            _save_lora_checkpoint(
                cfg=cfg,
                model=flux,
                optimizer=optimizer,
                global_step=global_step,
                train_metrics=last_train_metrics,
                val_metrics=last_val_metrics,
                name="lora_last",
            )

        writer.close()

        return {
            "history": history,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "num_trainable_params": sum(p.numel() for p in trainable_params),
        }
