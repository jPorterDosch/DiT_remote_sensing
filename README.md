### Overview
This project measures what frozen and LoRA-adapted FLUX.1-dev (a transformer trained with rectified flow, i.e. flow matching, not a diffusion/score objective) features contain for remote-sensing scene classification, against frozen DINO-family baselines under one paired protocol.

The original code which we fork off of is from the paper "Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
", published in NeurIPS 2025. Their Git repository is located at the following link: https://github.com/ganchaofan0000/DiTF/tree/main.

DiTF studies feature extraction from diffusion-transformer (DiT) architectures. FLUX shares that transformer family but is a flow-matching model; we reuse DiTF's feature normalization (massive-activation channel discard + adaLN modulation) on FLUX's block features (block 28: inherited, then validated by this project's block sweep as within the k=19-33 plateau, RESEARCH_NOTES 6f), and evaluate them, frozen and after label-free LoRA adaptation, on remote-sensing benchmarks.

### Project Setup
#### Download conda (if not already installed)
```
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh
source ~/.bashrc
```
#### Navigate back to copied directory
```
conda env create -f environment.yml
conda activate DiTF
pip install -e ".[all]"
```

### Evaluation pipeline
Every stage logs to one W&B project (`eval/wb.py`), as `{exp}_{dataset}_{arm}_{hash}` grouped by stage; the hash covers the protocol constants and input identities.

| Stage | Command | Output |
|---|---|---|
| FLUX features (frozen / adapted) | `python run.py --task extract ...` | `models/<save-dir>/<run>/multistep_<split>_feats_*.npz` |
| LoRA adaptation (all 57 blocks) | `python run.py --task finetune-diffusion ...` | `models/<save-dir>/<run>/checkpoints/` |
| GEO-Bench export | `python -m eval.export_geobench --task m-eurosat --dataset-dir <raw> --out data/m_eurosat_rgb` | `data/<task>_rgb/{train,val,test}/<Class>/*.png` |
| DINO features | `python -m eval.extract_dino --preset dinov2_vitl14 --dataset {resisc45,m_eurosat}` | `results/eval_feats/` |
| Probe one arm | `python -m eval.probe --protocol {cv,budget,mlp,official} --dataset D --arm NAME --kind {flux,dino,vae} --features F --view V` | `results/eval/<run>.npz` (per-image correctness) |
| Paired comparison | `python -m eval.compare A.npz B.npz` | B−A per cell, image-level bootstrap CI |

**Protocols** (`eval/protocols.py`): `cv` = 3 seeds × 5-fold, LR C=0.1 on the dataset's paired identity set (RESISC45: the 5,000-image section-13 subset); `budget` = 10/25/50/100 labels per class; `mlp` = matched MLP head on 1,000-image holdouts; `official` = select (candidate × C) on the official val split, then one test evaluation. Nothing reported is selected on the data that scores it.

**Views** (`eval/features.py`): FLUX `sec13:<t-idx>` (DiTF, PCA-protected best-t + other timesteps raw), `t:<idx>`, `concat`; DINO `cls`, `mp`, `clsmp`; VAE `full`, `pool4`, `pool2`, `pool1`.

Example, the headline RESISC45 comparison:
```
python -m eval.probe --protocol cv --dataset resisc45 --arm flux-inv-sec13 --kind flux --view sec13:1 \
    --features models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz \
    --expect extraction_mode=INVERSION num_inversion_steps=50
python -m eval.probe --protocol cv --dataset resisc45 --arm dinov2-clsmp --kind dino --view clsmp \
    --features results/eval_feats/dinov2_vitl14_resisc45_n5000.npz
python -m eval.compare <A.npz> <B.npz>   # the two "per-image vectors cached to ..." paths printed above
```
FLUX arms must pin the cache they expect (`--expect extraction_mode=ONESHOT ensemble_size=8`, or `extraction_mode=INVERSION num_inversion_steps=50`); a cache whose `_meta.json` disagrees is refused. Run names hash the protocol constants plus the SHA-1 of every input file, so a re-extracted cache gets a new result file instead of overwriting the old one.

**Gates.** `pytest tests/test_eval_gates.py -v -rs` (add `--gpu` for the extraction/MLP gates) re-runs every entry point against the banked per-image vectors of the prototype it replaced and reports PASS/FAIL/SKIP (a SKIP names the inputs missing on this machine). Run it after any change under `eval/`. Before any finetune launch, run the `smoke-finetune` skill, which includes the LoRA gradient-reach gate (`tests/check_lora_grad_reach.py`).

**ISAAC / offline.** Every ISAAC job sources `experiments/isaac/scratch_env.sh`: caches (HF, torch hub, wandb, TMPDIR) go to `/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing`, and the repo's `models/`, `data/`, `logs/`, `ditf_models/` become symlinks there (a populated home copy is refused with the one-time migration command; `results/` stays in home). Compute nodes run with `WANDB_MODE=offline`, so offline runs land in `$STORE/wandb/` for `wandb sync`. Offline runs cannot declare artifact inputs, so they record them; after `wandb sync`, run `python -m eval.wb link` on the login node to attach the extract → probe → compare lineage.

`experiments/prototypes/` holds the exploratory scripts the pipeline was distilled from (RESEARCH_NOTES cites them). They are kept for provenance, do not log to W&B, and are not maintained.
