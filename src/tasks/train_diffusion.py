from registry import register_task


@register_task("train-diffusion")
class TrainDiffusionTask:
    def run(self, cfg, model, dataset) -> dict:
        t = cfg.t
        raise NotImplementedError("TrainDiffusionTask is not implemented yet.")
        # TODO: 1. Freeze base FLUX weights
        # 2. Insert LoRA adapters into selected blocks
        # 3. Encode images into VAE latent space
        # 4. Sample timestep/noise and create noisy/interpolated latent
        # 5. Run FLUX forward
        # 6. Predict the flow/velocity target
        # 7. Backprop diffusion/flow-matching loss into LoRA only
        # 8. Save LoRA weights
        # 9. Evaluate representations with label sweep
