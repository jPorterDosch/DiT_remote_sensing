### Overview
This project measures what frozen and LoRA-adapted FLUX.1-dev (a rectified-flow diffusion transformer) features contain for remote-sensing scene classification, against frozen DINO-family baselines under one paired protocol.

The original code which we fork off of is from the paper "Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
", published in NeurIPS 2025. Their Git repository is located at the following link: https://github.com/ganchaofan0000/DiTF/tree/main.

We take their findings on the specific properties required to extract features from diffusion models (block-28 features, DiTF normalization), and evaluate them, frozen and after label-free LoRA adaptation, on remote-sensing benchmarks.

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
    --features models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz
python -m eval.probe --protocol cv --dataset resisc45 --arm dinov2-clsmp --kind dino --view clsmp \
    --features results/eval_feats/dinov2_vitl14_resisc45_n5000.npz
python -m eval.compare results/eval/probe-cv_resisc45_flux-inv-sec13_*.npz results/eval/probe-cv_resisc45_dinov2-clsmp_*.npz
```

**Gates.** `python -m eval.gates` (add `--gpu` for the extraction/MLP gates) re-runs every entry point against the banked per-image vectors of the prototype it replaced and prints PASS/FAIL/SKIP. Run it after any change under `eval/`. Before any finetune launch, run the `smoke-finetune` skill, which includes the LoRA gradient-reach gate (`tests/check_lora_grad_reach.py`).

**ISAAC / offline.** Compute nodes run with `WANDB_MODE=offline`. Offline runs cannot declare artifact inputs, so they record them; after `wandb sync`, run `python -m eval.wb link` on the login node to attach the extract → probe → compare lineage.

`experiments/prototypes/` holds the exploratory scripts the pipeline was distilled from (RESEARCH_NOTES cites them). They are kept for provenance, do not log to W&B, and are not maintained.
