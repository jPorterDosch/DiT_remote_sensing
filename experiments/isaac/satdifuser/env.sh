# SatDiFuser's own environment (their requirements.txt pins: torch 2.2.2, diffusers 0.17.0,
# transformers 4.31.0; built by jobs/env.sbatch under $SDF, see README.md). Sourced by the
# satdifuser sbatch scripts; never by the DiTF ones.
SDF="${SDF:-/lustre/isaac24/scratch/jdosch1/satdifuser}"
SDF_ENV="${SDF_ENV:-/lustre/isaac24/scratch/jdosch1/.conda/envs/satdifuser}"
SDF_REPO="${SDF_REPO:-$SDF/repo}"
SDF_CKPT="${SDF_CKPT:-$SDF/ckpt}"            # <SDF_CKPT>/resolution256/{model_index.json, checkpoint-150000/unet, vae, ...}
export GEO_BENCH_DIR="${GEO_BENCH_DIR:-$SDF/geobench}"  # raw GEO-Bench root (classification_v1.0/<task>) for their loaders
export PATH="$SDF_ENV/bin:$PATH"
python3 -c "import sys, torch, diffusers; assert 'satdifuser' in sys.prefix, sys.prefix; assert diffusers.__version__ == '0.17.0', diffusers.__version__" \
    || { echo "FATAL: SatDiFuser env not ready at $SDF_ENV (jobs/env.sbatch)" >&2; exit 1; }
[ -f "$SDF_REPO/run.py" ] || { echo "FATAL: $SDF_REPO is not the SatDiFuser clone" >&2; exit 1; }
[ -f "$SDF_CKPT/resolution256/checkpoint-150000/unet/config.json" ] || { echo "FATAL: DiffusionSat 256 checkpoint missing under $SDF_CKPT/resolution256" >&2; exit 1; }
