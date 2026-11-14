# SatDiFuser matched comparison

Goal: SatDiFuser's (Jia et al., ICCV 2025) frozen DiffusionSat features under OUR label-efficiency
protocol (eval/matrix.py), so one table holds them beside DINOv3-sat, frozen FLUX and the adapted
arms. Their Table 2 numbers come from a trained fuser + linear head at 100 % labels and stay a
separate, footnoted column.

Everything of theirs lives on scratch, outside the repo (`$SDF`, default
`/lustre/isaac24/scratch/jdosch1/satdifuser`): `repo/` (clone, commit 51148cb), the conda env
(`.conda/envs/satdifuser`: their requirements.txt pins), `ckpt/resolution256/` (DiffusionSat
256 px, zenodo 13756199, `checkpoint-150000/unet` + `vae` + `model_index.json`), `geobench/`
(raw m-eurosat for the gate and the PNG-vs-raw check), `experiments/` (gate runs). The one-off
setup jobs are `$SDF/jobs/{env,ckpt,geobench_eurosat}.sbatch` (short partition).

Order, every step gated:

1. `gate_eurosat.sbatch` — their code, unmodified, their config, m-eurosat. `EPOCHS=1` first
   (timing + env check), then the full 40-epoch run. Pass = "Final test metric" within ~0.5 of
   their gw 97.7. Fail = stop and diagnose; nothing of theirs is dumped until this passes
   (CLAUDE.md rules 2, 16). Executed, not prose: `gate_check.py verdict` parses run.py's output,
   prints `GATE satdifuser-eurosat PASS|FAIL`, writes `$SDF/gate/eurosat_gw_<jobid>.json` and exits
   1 on FAIL; `dump.sbatch` starts with `gate_check.py require` and stops without a PASS verdict.
   Status 2026-11-12: the gate has never been run, and no SatDiFuser feature has been dumped.
2. `dump.sbatch --export=ALL,SMOKE=1 --array=0` — 64 m-eurosat images per split from our PNG tree
   AND from the raw hdf5 through their loader path, then `eval.satdifuser compare`: per-image
   cosine must be ~1 (PNG quantization is below the VAE sampling noise). Also the end-to-end check
   of eval/satdifuser.py in their env.
3. `dump.sbatch --array=0-5` — all six tasks, three splits each. ~10 UNet passes per image
   (inversion up to their deepest saved timestep), A40.
4. `select.sbatch --array=0-5` — DiTF env; the pyramid scale per dataset on val at 100 %
   (s32 / s16 / s8 / s4 / all). Fill `eval/matrix_specs/phase3_satdifuser.json` per_dataset.
5. `experiments/isaac/eval/matrix.sbatch` with `SPEC=eval/matrix_specs/phase3_satdifuser.json`
   (teacher + frozen FLUX cells are reused from results/matrix; only the satdifuser cells run),
   then `python -m eval.matrix report --spec eval/matrix_specs/phase3_satdifuser.json`.

What "their features" means here (eval/satdifuser.py docstring): their `LDMExtractor` imported from
the clone and run as their run.py runs it (VAE posterior sampled with a seeded generator, x 0.18215,
DDIM inversion, hooks on every up_blocks attn1 / conv2 output, save_timesteps [47, 45, 40], prompt
"A satellite image", no guidance, fp16 autocast), then a spatial mean per pyramid scale -- the mean
their classifier head takes after the fuser, here taken before any trained layer. Our caches record
the DDPM t of each saved index (`t_ddpm`), the collected layer names per scale and the checkpoint
identity; eval/probe.py official_satdifuser refuses a cache whose pins, split identity or VAE seeds
are off.

Differences from their protocol that remain, by design: no fuser (frozen features, linear probe),
no augmentation, label fractions, val-only selection of (scale, t, C), image bootstrap CIs.
