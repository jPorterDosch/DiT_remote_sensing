#!/bin/bash
# Download the GeoCore-9B snapshot (JeonghyeokDo/GeoCore-9B: 4 bf16 shards 18.5 GB + FLUX.2 VAE)
# to $STORE/GeoCore-9B and verify every file against the upstream SHA256SUMS. Refuses to touch
# an existing directory (never overwrite weights). Prints GEOCORE_DOWNLOAD_OK.
#
#   sbatch experiments/isaac/geocore/download.sbatch
#
#SBATCH --job-name=geocore-download
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=2:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
set -euo pipefail

DEST="${DEST:-$STORE/GeoCore-9B}"
[ ! -e "$DEST" ] || { echo "FATAL: $DEST exists -- never overwrite weights; move it aside or set DEST" >&2; exit 1; }
echo "=== download JeonghyeokDo/GeoCore-9B -> $DEST $(date -Is)"
df -h "$STORE" | tail -1
python3 - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
p = snapshot_download(
    "JeonghyeokDo/GeoCore-9B",
    local_dir=dest,
    allow_patterns=["model-*.safetensors", "model.safetensors.index.json", "config.json",
                    "vae/*", "SHA256SUMS", "LICENSE-FLUX2-VAE.md", "README.md"],
    max_workers=4,
)
print("snapshot at", p)
PY
cd "$DEST"
ls -la . vae
# Every file named in SHA256SUMS must be present and match (the adapter re-verifies the pinned
# shard / index / VAE digests at every load; this is the first line of defence).
sha256sum -c SHA256SUMS
du -sh "$DEST"
echo "GEOCORE_DOWNLOAD_OK $DEST $(date -Is)"
