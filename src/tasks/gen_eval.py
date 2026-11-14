"""Post-hoc generation metrics of a (LoRA) checkpoint: `run.py --task gen-eval`.

The training task scores FID/KID on align.fid_n = 600 samples (the val holdout has 656 rows).
That is too few for a quotable distributional metric, so this task re-scores ANY checkpoint
-- the pretrained model when --lora-checkpoint is empty, else the adapter it names -- on a
larger sample, with the training task's own sampler, seeds and reference construction:

  samples   batches of align.gen_n at seeds align.gen_seed + 1000 + batch (train_diffusion
            _gen_metrics), align.gen_steps sampler steps, null prompt, cfg.guidance_scale;
            so --align.fid-n 600 on a run's lora_last reproduces its meta.json "final" number
            (an executed consistency check between this task and the training-time metric)
  reference the first align.fid_n strided rows of the dataset's val view, unflipped
            (train_diffusion._reference_images). For n > 656 give the git10m dataset a larger
            holdout: --git10m.index <index> --git10m.holdout-every 100 (--git10m.val-index unset,
            so the holdout is drawn from that index). The reference identity (sha1 of the
            uint8 array) is recorded; every arm must be scored against the SAME one.

Writes under the run dir: samples/*.png, fid_ref/*.png, a 4 x 4 grid, gen_metrics.json
(FID, KID mean/std, n, seeds, latent sha1, reference / sample identities, checkpoint
provenance: path, sha1, align arm, global_step). Prints GEN_EVAL_OK on success. Rule 10:
run.py _task_honors lists FLUX_RANDOM_INIT (model load) and DEGRADE_TO (dataset hook) for this
task and NOT FIXED_COND_T: denoise() never goes through Featurizer4Eval.forward, its only reader.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from models.flux.adapter import FluxModel
from registry import register_task
from utils import to_jsonable

from .train_diffusion import _gen_metrics, _reference_images


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt_provenance(path: str) -> dict:
    """What the checkpoint says about itself (the adapter already loaded its weights)."""
    if not path:
        return {"lora_checkpoint": "", "weights": "flux-dev (pretrained, no adapter)"}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    al = ck.get("align") or {}
    return {
        "lora_checkpoint": os.path.abspath(path),
        "lora_checkpoint_sha1": _file_sha1(path),
        "global_step": ck.get("global_step"),
        "arm": al.get("arm"),
        "feat": al.get("feat"),
        "target": (al.get("target_info") or {}).get("name")
        if isinstance(al.get("target_info"), dict)
        else al.get("target"),
        "target_shuffle": al.get("target_shuffle", ""),
        "weight": al.get("weight"),
        "lora_rank": (ck.get("cfg") or {}).get("lora_rank"),
        "weights": f"flux-dev+lora:{al.get('arm')}",
    }


@register_task("gen-eval")
class GenEvalTask:
    def run(self, cfg, model: FluxModel, dataset) -> dict:
        from eval import gen_metrics as GM

        a = cfg.align
        if not a.fid_n:
            raise ValueError("gen-eval needs --align.fid-n > 0 (the number of samples and reference rows)")
        if a.no_gen:
            raise ValueError("gen-eval is generation: drop --align.no-gen")
        if a.arm:
            raise ValueError("gen-eval reads the arm from the checkpoint: do not pass --align.arm")
        device = torch.device(cfg.device)
        featurizer = model._inner
        featurizer.model.eval()

        loaders = dataset.get_data(cfg)
        src = loaders.get("val") or loaders.get("test")
        if src is None:
            raise ValueError(
                f"dataset {cfg.dataset.name!r} provides neither a val nor a test loader for the reference"
            )
        ref = _reference_images(src.dataset, a.fid_n, cfg.num_workers)
        extractor = GM.InceptionFeatures(device)
        print(f"[gen-eval] reference {GM.image_identity(ref)} extractor {extractor.name}", flush=True)
        prov = _ckpt_provenance(cfg.lora_checkpoint)
        print(f"[gen-eval] checkpoint {json.dumps(to_jsonable(prov))}", flush=True)

        # The VAE decode is deterministic in the training task's final generation too.
        reg_prev = featurizer.ae.reg.sample
        featurizer.ae.reg.sample = False
        t0 = time.time()
        try:
            res = _gen_metrics(
                cfg,
                featurizer,
                ref,
                extractor,
                device,
                "gen-eval",
                save_dir=os.path.join(cfg.save_dir, "samples"),
            )
        finally:
            featurizer.ae.reg.sample = reg_prev
        if res.get("error"):
            raise RuntimeError(f"gen-eval metric failed (samples kept): {res['error']}")

        # A 4 x 4 grid of the first samples for the figure.
        files = sorted(f for f in os.listdir(os.path.join(cfg.save_dir, "samples")) if f.endswith(".png"))[
            :16
        ]
        if len(files) == 16:
            tiles = [
                np.asarray(Image.open(os.path.join(cfg.save_dir, "samples", f)).convert("RGB")) for f in files
            ]
            rows = [np.concatenate(tiles[i : i + 4], axis=1) for i in range(0, 16, 4)]
            Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(cfg.save_dir, "grid_4x4.png"))

        out = {
            **res,
            "checkpoint": prov,
            "sampler": {
                "gen_steps": a.gen_steps,
                "gen_n": a.gen_n,
                "gen_seed": a.gen_seed,
                "guidance": cfg.guidance_scale,
                "img_size": list(cfg.img_size),
            },  # fmt: skip
            "reference": {
                "dataset": cfg.dataset.name,
                "git10m": cfg.git10m.identity() if cfg.dataset.name == "git10m" else None,
                "n": int(ref.shape[0]),
            },  # fmt: skip
            "fid_subsets": a.fid_subsets,
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(cfg.save_dir, "gen_metrics.json"), "w") as f:
            json.dump(to_jsonable(out), f, indent=1)
        print(f"[gen-eval] n={res['n']} FID={res['fid']:.2f} KID={res['kid_mean']:.4f}+-{res['kid_std']:.4f} "
              f"ref sha1={res['real']['sha1'][:12]} in {out['elapsed_s']}s -> {cfg.save_dir}/gen_metrics.json", flush=True)  # fmt: skip
        print("GEN_EVAL_OK", flush=True)
        return out
