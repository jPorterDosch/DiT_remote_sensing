#!/bin/bash
# ============================================================================
# BigEarthNet-S2 v1.0 (Zenodo 12687186, ~66 GB) -> $STORE/geo_dedup/source_meta/bigearthnet/.
# Needed to locate m-bigearthnet: GEO-Bench's converter stored no georeferencing, so the
# samples are matched back to the original patches by pixel hash (eval/geo_dedup.py
# bigearthnet-index), whose *_labels_metadata.json carry the patch corners + projection.
# md5 from torchgeo.datasets.bigearthnet (BigEarthNet-S2-v1.0.tar.gz). Download only; the
# archive is streamed, never extracted (590k patch dirs would be ~7.7M Lustre files).
#
#   mkdir -p logs && sbatch experiments/isaac/geo_dedup/fetch_bigearthnet.sbatch
# ============================================================================
#SBATCH --job-name=fetch-bigearthnet
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
set -euo pipefail

URL="https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz"
MD5="5a64e9ce38deb036a435a7b59494924c"
DEST="$STORE/geo_dedup/source_meta/bigearthnet"
mkdir -p "$DEST"
[ -f "$DEST/BigEarthNet-S2-v1.0.tar.gz.md5ok" ] && { echo "already verified"; exit 0; }
aria2c -x 16 -s 16 -k 16M --file-allocation=none --continue=true -d "$DEST" -o BigEarthNet-S2-v1.0.tar.gz "$URL"
got=$(md5sum "$DEST/BigEarthNet-S2-v1.0.tar.gz" | cut -d' ' -f1)
[ "$got" = "$MD5" ] || { echo "FATAL: md5 $got != $MD5" >&2; exit 1; }
echo "$got" > "$DEST/BigEarthNet-S2-v1.0.tar.gz.md5ok"
echo "FETCH_OK $(ls -l "$DEST/BigEarthNet-S2-v1.0.tar.gz")"
