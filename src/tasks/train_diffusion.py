from registry import register_task


@register_task("train-diffusion")
class TrainDiffusionTask:
    def run(self, cfg, model, dataset) -> dict:
        t = cfg.t
        raise NotImplementedError("TrainDiffusionTask is not implemented yet.")
