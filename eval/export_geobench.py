"""Export a GEO-Bench classification task to the RGB PNG tree the pipeline consumes:

    python -m eval.export_geobench --task m-eurosat --dataset-dir data/m_eurosat_meta --out data/m_eurosat_rgb

    <out>/{train,val,test}/<ClassName>/<sample_name>.png

A task is exportable only once it has a TaskSpec below, and every TaskSpec value must be
read from the SHIPPED artifact (default_partition.json counts, band names, dtype range,
class names) -- never from a paper table (CLAUDE.md rule 16). m-eurosat's spec is
export_m_eurosat.py's (SatDiFuser's loading semantics), gated byte-identical against the
existing data/m_eurosat_rgb export in tests/test_eval_gates.py. The R3 tasks (m-forestnet, m-so2sat,
m-brick-kiln; RESEARCH_NOTES 9.4 Q7) are listed but refuse to run until verified.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import features as F  # noqa: E402  (puts src/ on sys.path)

OUT_SPLIT = {"train": "train", "valid": "val", "test": "test"}


@dataclass(frozen=True)
class TaskSpec:
    sizes: dict  # geobench split name -> count in default_partition.json
    rgb_bands: tuple
    scale: float  # divide raw band values by this, then clip to [0,1]
    img_hw: tuple
    name_map: dict = field(default_factory=dict)  # geobench class name -> output dir name
    verified: str = ""  # where/when the values were read from the shipped artifact


SPECS: dict[str, TaskSpec | None] = {
    "m-eurosat": TaskSpec(
        sizes={"train": 16200, "valid": 996, "test": 996},
        rgb_bands=("04", "03", "02"),
        scale=4095.0,
        img_hw=(64, 64),
        name_map={
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
        },
        verified="2026-09-17, default_partition.json + SatDiFuser datasets/m_eurosat.py (export_m_eurosat.py)",
    ),
    # R3 (9.4 Q7): fill ONLY from the shipped files; each needs an OFFICIAL entry in
    # eval/features.py (split sizes + class list) before probes accept it.
    "m-forestnet": None,  # Landsat-8: band mapping + dtype range unverified
    "m-so2sat": None,  # Sentinel-2 subset of S1/S2 stack: band names unverified
    "m-brick-kiln": None,  # Sentinel-2: band names + range unverified
}


def export(task: str, dataset_dir: str, out: str) -> int:
    spec = SPECS.get(task)
    if spec is None:
        raise SystemExit(
            f"{task}: no verified TaskSpec. Read default_partition.json (count every split), the "
            "band names and value range from the shipped task files, add a TaskSpec + an "
            "eval/features.OFFICIAL entry, then re-run (rule 16)."
        )
    import geobench  # deferred: heavy import, and the error should name the missing dep

    task_specs = geobench.load_task_specs(dataset_dir)
    class_names = list(task_specs.label_type.class_names)
    unmapped = [c for c in class_names if c not in spec.name_map]
    if unmapped:
        raise SystemExit(f"geobench class names not in name_map: {unmapped}")
    ds_key = task.replace("-", "_")
    if ds_key in F.OFFICIAL and sorted(spec.name_map[c] for c in class_names) != sorted(
        F.official_classes(ds_key)
    ):
        raise SystemExit("mapped class set != the pipeline's class list")

    total = 0
    for split, expect_n in spec.sizes.items():
        ds = geobench.GeobenchDataset(dataset_dir, split=split, partition_name="default")
        if len(ds) != expect_n:
            raise SystemExit(f"{split}: got {len(ds)} samples, expected {expect_n} -- wrong partition?")
        hist: Counter = Counter()
        for i in range(len(ds)):
            sample = ds[i]
            image, _ = sample.pack_to_3d(band_names=spec.rgb_bands)
            image = np.clip(image.astype(np.float32) / spec.scale, 0.0, 1.0)
            if image.shape[:2] != spec.img_hw:
                raise SystemExit(f"{split}[{i}] has shape {image.shape}, expected {spec.img_hw}")
            cls = spec.name_map[class_names[int(sample.label)]]
            hist[cls] += 1
            d = os.path.join(out, OUT_SPLIT[split], cls)
            os.makedirs(d, exist_ok=True)
            Image.fromarray((image * 255.0).round().astype(np.uint8)).save(
                os.path.join(d, f"{sample.sample_name}.png")
            )
            if i % 1000 == 0:
                print(f"  {split} {i}/{len(ds)}", flush=True)
        total += len(ds)
        print(f"{split}: {len(ds)} images; per-class: {dict(sorted(hist.items()))}", flush=True)
    if total != sum(spec.sizes.values()):
        raise SystemExit("count mismatch after export")
    print(f"exported {total} images to {out}")
    return total


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(SPECS))
    p.add_argument(
        "--dataset-dir", required=True, help="dir with task_specs.pkl, default_partition.json, *.hdf5"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    export(args.task, args.dataset_dir, args.out)


if __name__ == "__main__":
    main()
