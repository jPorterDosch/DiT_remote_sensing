#!/usr/bin/env bash
# ============================================================================
# m-eurosat data prep for ISAAC — run ONCE on a login node (network + CPU only).
#   1. Downloads the official GEO-Bench m-eurosat release from Zenodo (~1.2 GB).
#   2. Assembles the geobench dataset dir (task_specs.pkl + partitions + samples).
#   3. Exports the RGB image tree data/m_eurosat_rgb/{train,val,test}/<Class>/*.png
#      via experiments/export_m_eurosat.py (SatDiFuser-identical loading semantics).
# The exporter's gates verify 16,200/996/996 and the 10 EuroSAT classes; a partial
# download fails loudly here, never downstream.
# Requires geobench, but INSTALL IT WITH --no-deps:
#     pip install --no-deps geobench && pip install h5py rasterio
#   geobench pins huggingface_hub<0.20, pandas<2.0 and seaborn<0.13. Installing it
#   normally DOWNGRADES those and hard-breaks the extraction stack (verified
#   2026-09-17): transformers/diffusers import split_torch_state_dict_into_shards
#   at module level (needs hf_hub>=0.23), and pandas 1.5.3 is binary-incompatible
#   with numpy 2.1.0. Those pins are over-declared for our use — geobench only
#   imports h5py/numpy/rasterio/scipy/tqdm on the load_task_specs/GeobenchDataset
#   path; pandas/hf_hub/seaborn appear only in plot_tools.py and geobench_download.py,
#   which __init__.py never imports and this script never calls (we curl Zenodo).
# ============================================================================
set -euo pipefail

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"

RAW="${RAW:-/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/m_eurosat_meta}"
mkdir -p "$RAW"
Z="https://zenodo.org/api/records/8276933/files"

for f in task_specs.pkl default_partition.json label_map.json band_stats.json; do
    [ -f "$RAW/$f" ] || curl -sL -o "$RAW/$f" "$Z/$f/content"
done
if [ ! -f "$RAW/.data_unzipped" ]; then
    [ -f "$RAW/data.zip" ] || curl -L -o "$RAW/data.zip" "$Z/data.zip/content"
    unzip -q -o "$RAW/data.zip" -d "$RAW"
    touch "$RAW/.data_unzipped"
fi

# The zip unpacks the sample .hdf5 files flat, next to the metadata (verified on the
# 2026-09-17 local run: 18,192 id_*.hdf5 at top level).
n_hdf5=$(ls "$RAW"/id_*.hdf5 2>/dev/null | wc -l)
[ "$n_hdf5" -eq 18192 ] || { echo "expected 18,192 sample files, found $n_hdf5 — incomplete unzip?" >&2; exit 1; }

# sanity: partition really is the 16,200/996/996 default (CLAUDE.md rule 16)
python3 - "$RAW" <<'PY'
import json, sys
p = json.load(open(sys.argv[1] + "/default_partition.json"))
sizes = {k: len(v) for k, v in p.items()}
assert sizes == {"train": 16200, "valid": 996, "test": 996}, sizes
print("partition verified:", sizes)
PY

python3 experiments/export_m_eurosat.py --dataset-dir "$RAW" --out data/m_eurosat_rgb
echo "prep complete: data/m_eurosat_rgb"