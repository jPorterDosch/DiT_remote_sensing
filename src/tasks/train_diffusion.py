import torch
import torch.nn.functional as F
from registry import register_task

from ..models.flux.feat_flux import Featurizer4Eval, prepare
from ..models.lora import lora_wrap_flux


def _fine_tune_diffusion_one_epoch(
    cfg,
    model: torch.nn.Module,
    vae: torch.nn.Module,
    timestep: int,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    vae.eval()

    total_loss = 0.0
    n_batches = 0

    guidance_scale = cfg.guidance_scale

    # Normalize timestep
    t = timestep / 1000.0  # assuming 1000 total diffusion steps (same as feat_flux.py)

    for batch in dataloader:
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

        # TODO: add null embeddings here once other PR merged in (or just pull in changes directly if not too large)
        # txt, txt_ids, y = null_embedding_load()
        txt, txt_ids, y = None, None, None  # temp placeholders

        guidance_vec = torch.full(
            (imgs.shape[0],), guidance_scale, device=device, dtype=latents.dtype
        )

        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=torch.full(
                (imgs.shape[0],), t, device=device, dtype=latents.dtype
            ),
            guidance=guidance_vec,
        )
        # Patchify target to match FLUX output shape.
        target, _ = prepare(target_flow)
        target = target.to(device=device, dtype=pred.dtype)

        loss = F.mse_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
    }


def _validate_diffusion(
    cfg,
    model: torch.nn.Module,
    vae: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    # 1. Encode validation images into VAE latent space
    # 2. Sample timestep/noise and create noisy/interpolated latent
    # 3. Run FLUX forward
    # 4. Predict the flow/velocity target
    # 5. Compute validation loss and other metrics (e.g. FID between predicted and ground-truth noise)
    raise NotImplementedError(
        "Validation for FinetuneDiffusionTask is not implemented yet."
    )


@register_task("finetune-diffusion")
class FinetuneDiffusionTask:
    def run(self, cfg, model: Featurizer4Eval, dataset) -> dict:
        flux = model.model
        ae = model.ae
        device = torch.device(cfg.device)

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

        optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.lora_lr, weight_decay=cfg.lora_wd
        )

        # Create loaders
        loaders = dataset.get_data(cfg)
        train_loader = loaders["train"]
        test_loader = loaders["test"]

        history = []

        for epoch in range(cfg.finetune_epochs):
            train_metrics = _fine_tune_diffusion_one_epoch(
                cfg,
                flux,
                vae=ae,
                timestep=cfg.t,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            val_metrics = _validate_diffusion(
                cfg, flux, vae=ae, dataloader=test_loader, device=device
            )

            # Save model and log metrics periodically during training
