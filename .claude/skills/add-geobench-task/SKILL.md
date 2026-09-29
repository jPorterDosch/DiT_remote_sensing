---
name: add-geobench-task
description: Register, export and verify a GEO-Bench classification task (TaskSpec, OFFICIAL entry, manifest, DINOv3 control, FLUX smoke) before any ISAAC sweep is submitted.
---

Every value comes from the SHIPPED artifact (CLAUDE.md rule 16). Stop at the first FAIL and
diagnose the instrument (rule 2) -- never adjust a threshold to make a gate pass.

## 1. Read the shipped files (metadata only, a few MB)
Find the v1.0 Zenodo record (the one with `default_partition.json`):
`curl -s "https://zenodo.org/api/records?q=%22<task>%22%20geobench"`. Fetch task_specs.pkl,
default_partition.json and band_stats.json (NOT label_stats.json: 100s of MB) into
`data/<key>_meta/`, then print split counts, `label_type` (class_names / class_name for
multi-label), band names and band_stats ranges. If a reference loader exists (SatDiFuser
`datasets/<task>.py`), read its band choice and scale.

## 2. Register (two edits)
- `eval/export_geobench.py` SPECS: zenodo id, rgb_bands, scale (S2 DN /4095; S2 reflectance
  /0.4095; 8-bit /255), img_hw from a SAMPLE (task_specs patch_size can be wrong), verified=.
- `eval/features.py` OFFICIAL: `_geobench("<key>", train, val, test[, multilabel=True])`.

## 3. Export + manifest (workstation; transient disk ~2x the zip)
```bash
python -m eval.export_geobench --task <task> --dataset-dir data/<key>_meta \
    --out data/<key>_rgb --download --rm-raw --write-manifest
```
The export refuses a wrong scale (clip guard), shape, label kind or count. Commit
`eval/manifests/<key>.json`; ISAAC's `experiments/isaac/geobench/prepare.sh` verifies against it.

## 4. Look at it
Contact sheet (a few train images per class): natural colours, class names match content.
Record anything odd (masks, repeated sites, split overlap by location) in RESEARCH_NOTES.

## 5. Gates
```bash
pytest tests/test_geobench.py -q                                         # registry + guards
```
DINOv3 control (ISAAC only -- gated weights on scratch; bump dino.sbatch's --array to the OFFICIAL count):
`sbatch --export=ALL,PRESET=dinov3_vitl16_{web,sat} experiments/isaac/geobench/dino.sbatch`.
PASS: clsmp beats the shuffled-label null by >= 0.10 and the null is not above the majority
rate + 0.05; the job produces the DINO result only after its gate passes. Then the FLUX smoke (~5 min, caches deleted after):
```bash
WANDB_MODE=disabled AE=ditf_models/FLUX.1-dev/ae.safetensors FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors \
DATASET=<key> ENS=2 SUBSET_ARGS="--subset-size 24" SMOKE_SIZES=24,24,24 SWEEP_ROOT=models/smoke_gb \
OUT_DIR=results/eval/smoke_gb bash experiments/isaac/sweep/sweep_block.sh 33      # expect BLOCK OK
rm -rf models/smoke_gb results/eval/smoke_gb
```

## 6. Handoff
RESEARCH_NOTES entry (counts, scale choice, control numbers). `git ls-files --error-unmatch`
on every new file (rule 17). On ISAAC: `prepare.sh <task>`, then
`bash experiments/isaac/sweep/submit.sh <key>` (sizes --time, refuses > 72 h).
