---
name: smoke-finetune
description: Run the canonical three-mode GPU smoke of the finetune-diffusion task and verify the log signatures before any real training is launched.
---

Run all three arms end-to-end (~5-10 min each) and verify signatures. Never skip modes —
seven crash bugs were caught one stage at a time by exactly this.

## Commands
```bash
export AE=ditf_models/FLUX.1-dev/ae.safetensors FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors WANDB_MODE=offline
rm -rf models/smoke_finetune
COMMON="--task finetune-diffusion --dataset.name eurosat --dataset.path data/eurosat/EuroSAT_RGB \
  --model.name flux --img-size 224 224 --t 100 260 580 --k 28 \
  --guidance-scale 1.0 --mim-loss-weight 0.0 --mask-ratio 0.0 --model.ensemble-size 1 \
  --max-train-steps 10 --log-train-steps 5 --log-val-steps 5 --max-samples 32 \
  --batch-size 1 --num-workers 4 --seed 42 --warmup-steps 3 --save-dir models/smoke_finetune"
python run.py $COMMON                        # label-free flow adaptation (default arm)
python run.py $COMMON --freeze-backbone      # frozen arm
python run.py $COMMON --supervised-finetune  # supervised arm
```
(Pins must be EXPLICIT — run.py defaults are the cache-identity values, not the pins.
`--exclude-*`-style flags must never be combined with `--max-samples`.)

## Verify in each log (a run that exits 0 is NOT verified)
- Pin banner: `guidance=1.0 mim_loss_weight=0.0 mask_ratio=0.0`, correct mode name.
- Step prints have NO `train_mim=` field; `train_total == train_flow`.
- FROZEN arm: `val_total` bit-identical across evaluations (backbone provably static).
- DEFAULT arm: flow loss and val_total move.
- Per-t probe table printed for every grid t at the end; all classes appear in per-class
  output (strided max_samples — a single-class table means the striding broke).
- Checkpoints exist with zero-padded step names.

Report the three exit codes and the signature checklist. Any failure: STOP, diagnose,
re-run only the failed mode after the fix; do not proceed to real training.
