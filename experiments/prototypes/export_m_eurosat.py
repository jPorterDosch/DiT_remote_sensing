"""Export GEO-Bench m-eurosat to an RGB image tree for this repo's extraction pipeline.

    python experiments/prototypes/export_m_eurosat.py --dataset-dir <dir with task_specs.pkl,
        default_partition.json, label_map.json and the sample .hdf5 files> \
        --out data/m_eurosat_rgb

Loading semantics MATCH SatDiFuser's datasets/m_eurosat.py exactly so the images the two
pipelines consume are identical up to their respective resize/normalize conventions:
  - geobench load_task_specs(dir) -> task.get_dataset(split=...), i.e. the package
    DEFAULT partition (default_partition.json: 16,200/996/996 -- NOT the papers' "2,000",
    CLAUDE.md rule 16),
  - RGB = bands ("04","03","02") via sample.pack_to_3d,
  - scaled /4095, clipped to [0,1]  (their exact lines), then saved as uint8 PNG.
This repo's standard input transform (x/255 - 0.5)*2 then reproduces their [-1,1] range.

Output tree (consumed by src/datasets/m_eurosat.py):
    <out>/{train,val,test}/<ClassName>/<sample_name>.png

Gates (executed, not prose): split sizes must be exactly 16,200/996/996; class names
must be exactly the 10 EuroSAT classes; every sample must be 64x64; per-split class
histogram is printed for the notes.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))

EXPECT = {"train": 16200, "valid": 996, "test": 996}
OUT_SPLIT = {"train": "train", "valid": "val", "test": "test"}
RGB_BANDS = ("04", "03", "02")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out", default="data/m_eurosat_rgb")
    args = p.parse_args()

    import geobench  # deferred: heavy import, and the error should name the missing dep

    task = geobench.load_task_specs(args.dataset_dir)
    class_names = list(task.label_type.class_names)
    # GEO-Bench renamed the classes ("Annual Crop", "Sea and Lake", ...) and reordered
    # them relative to EuroSAT's alphabetical directory names — labels MUST be mapped
    # through this explicit table, never by sorting (verified 2026-09-17: geobench label
    # order starts at 'Industrial Buildings').
    NAME_MAP = {
        "Annual Crop": "AnnualCrop",
        "Forest": "Forest",
        "Herbaceous Vegetation": "HerbaceousVegetation",
        "Highway": "Highway",
        "Industrial Buildings": "Industrial",
        "Pasture": "Pasture",
        "Permanent Crop": "PermanentCrop",
        "Residential Buildings": "Residential",
        "River": "River",
        "Sea and Lake": "SeaLake",
    }
    from data.eurosat_dataset import EUROSAT_CLASSES

    unmapped = [c for c in class_names if c not in NAME_MAP]
    if unmapped:
        raise SystemExit(f"geobench class names not in NAME_MAP: {unmapped}")
    if sorted(NAME_MAP[c] for c in class_names) != sorted(EUROSAT_CLASSES):
        raise SystemExit("mapped class set != EuroSAT classes")

    total = 0
    for split, expect_n in EXPECT.items():
        # Same class + same default partition SatDiFuser's task.get_dataset uses, but
        # pointed at our dataset dir instead of $GEO_BENCH_DIR.
        ds = geobench.GeobenchDataset(args.dataset_dir, split=split, partition_name="default")
        if len(ds) != expect_n:
            raise SystemExit(f"{split}: got {len(ds)} samples, expected {expect_n} — wrong partition?")
        hist: Counter = Counter()
        for i in range(len(ds)):
            sample = ds[i]
            image, _ = sample.pack_to_3d(band_names=RGB_BANDS)  # H, W, 3 — SatDiFuser's call
            image = np.clip(image.astype(np.float32) / 4095.0, 0.0, 1.0)  # their exact scaling
            if image.shape[:2] != (64, 64):
                raise SystemExit(f"{split}[{i}] has shape {image.shape}, expected 64x64")
            label = int(sample.label)
            cls = NAME_MAP[class_names[label]]
            hist[cls] += 1
            d = os.path.join(args.out, OUT_SPLIT[split], cls)
            os.makedirs(d, exist_ok=True)
            Image.fromarray((image * 255.0).round().astype(np.uint8)).save(
                os.path.join(d, f"{sample.sample_name}.png")
            )
            if i % 1000 == 0:
                print(f"  {split} {i}/{len(ds)}", flush=True)
        total += len(ds)
        print(f"{split}: {len(ds)} images; per-class: {dict(sorted(hist.items()))}", flush=True)

    print(f"exported {total} images to {args.out}")
    if total != sum(EXPECT.values()):
        raise SystemExit("count mismatch after export")


if __name__ == "__main__":
    main()
