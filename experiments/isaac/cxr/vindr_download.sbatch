#!/bin/bash
# ============================================================================
# Download the VinDr-CXR Kaggle competition archive (vinbigdata-chest-xray-abnormalities-detection)
# into the mode-700 scratch tree. The archive stays a zip and is never unpacked. The log carries
# sizes and member counts only: archive member names are image ids and are never printed.
#
# Needs, once, by the account owner: the competition rules accepted in a browser and an API token
# at ~/.kaggle/access_token. The Kaggle CLI reads the token; this script never does.
#
#   mkdir -p logs && sbatch experiments/isaac/cxr/vindr_download.sbatch
#
# A job killed at the wall clock leaves a partial zip plus the CLI's resume marker; submitting
# again resumes it. A finished download writes $RAW/.download_ok and later runs refuse to start.
# Success is VINDR_DOWNLOAD_OK in the log.
# ============================================================================
#SBATCH --job-name=cxr-vindr-download
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

COMPETITION=vinbigdata-chest-xray-abnormalities-detection
KAGGLE="${KAGGLE:-/lustre/isaac24/scratch/jdosch1/tools/kaggle_venv/bin/kaggle}"
D=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr
RAW="$D/vindr_cxr/raw"
ZIP="$RAW/$COMPETITION.zip"

[ "$(stat -c %a "$D")" = 700 ] || { echo "FATAL: $D is not mode 700" >&2; exit 1; }
[ -x "$KAGGLE" ] || { echo "FATAL: no Kaggle CLI at $KAGGLE" >&2; exit 1; }
[ ! -e "$RAW/.download_ok" ] || { echo "FATAL: $RAW/.download_ok exists; download already complete" >&2; exit 1; }
python3 -I experiments/isaac/cxr/quota_guard.py --min-free-gb 900
mkdir -p "$RAW"
chmod 700 "$RAW"

echo "VINDR_DOWNLOAD_START partial_bytes=$( [ -f "$ZIP" ] && stat -c %s "$ZIP" || echo 0 )"
"$KAGGLE" competitions download "$COMPETITION" -p "$RAW" -q

# Central directory only: member count, counts by allow-listed extension and by allow-listed top
# directory. No member name reaches the log.
python3 -I - "$ZIP" <<'PY'
import collections
import os
import sys
import zipfile

path = sys.argv[1]
ext_ok = {".dicom", ".dcm", ".csv"}
top_ok = {"train", "test"}
with zipfile.ZipFile(path) as z:
    infos = [i for i in z.infolist() if not i.is_dir()]
ext, top = collections.Counter(), collections.Counter()
for i in infos:
    e = os.path.splitext(i.filename)[1].lower()
    ext[e if e in ext_ok else "other"] += 1
    head = i.filename.split("/", 1)[0] if "/" in i.filename else "<root>"
    top[head if head in top_ok or head == "<root>" else "other"] += 1
print(
    f"VINDR_ZIP bytes={os.path.getsize(path)} members={len(infos)} "
    f"uncompressed_bytes={sum(i.file_size for i in infos)} "
    f"ext={dict(sorted(ext.items()))} top={dict(sorted(top.items()))}"
)
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "$RAW/.download_ok"
echo "VINDR_DOWNLOAD_OK"
