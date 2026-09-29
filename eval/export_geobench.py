"""Export a GEO-Bench classification task to the RGB PNG tree the pipeline consumes:

    python -m eval.export_geobench --task m-forestnet --dataset-dir data/m_forestnet_meta \\
        --out data/m_forestnet_rgb [--download] [--rm-raw] [--write-manifest]

    single-label: <out>/{train,val,test}/<ClassDir>/<sample_name>.png
    multi-label:  <out>/{train,val,test}/<sample_name>.png + <out>/<split>/labels.npz (names, y)

Every TaskSpec value is read from the SHIPPED artifact (default_partition.json counts,
band names, dtype range, sample shapes) or a reference loader's source -- never from a
paper table (CLAUDE.md rule 16). Split sizes come from eval/features.OFFICIAL.

REPLICATION: the export hashes the decoded pixels + labels + sample names per split (not
PNG bytes: zlib versions differ) and compares them to the committed eval/manifests/<key>.json
(written once on the workstation with --write-manifest). A mismatch refuses, so the ISAAC
tree is provably the one the workstation gates ran on. The manifest's class list IS the
registry's class list for the tasks registered after m-eurosat.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import features as F  # noqa: E402  (puts src/ on sys.path)

OUT_SPLIT = {"train": "train", "valid": "val", "test": "test"}
MAX_CLIPPED = 0.05  # per-channel fraction of pixels at 0 or 255; more = wrong scale (fire-tested)
ZENODO_FILES = ("task_specs.pkl", "default_partition.json", "band_stats.json", "label_map.json", "data.zip")


@dataclass(frozen=True)
class TaskSpec:
    zenodo: int  # GEO-Bench v1.0 record (the one shipping default_partition.json)
    rgb_bands: tuple
    scale: float  # divide raw band values by this, then clip to [0,1]
    img_hw: tuple  # checked on EVERY sample (m-forestnet/m-pv4ger task_specs overstate it)
    name_map: dict = field(default_factory=dict)  # geobench class -> dir name; empty = dir_name()
    verified: str = ""  # where/when the values were read from the shipped artifact


# Scale convention: SatDiFuser's /4095 on Sentinel-2 DN (datasets/m_eurosat.py,
# m_bigearthnet.py); m-so2sat ships S2 reflectance already /10000, so the same physical map
# is /0.4095; m-forestnet (Landsat) and m-pv4ger (aerial) ship 8-bit RGB -> /255.
SPECS: dict[str, TaskSpec] = {
    "m-eurosat": TaskSpec(
        zenodo=8276933,
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
    "m-forestnet": TaskSpec(
        zenodo=8277533,
        rgb_bands=("04", "03", "02"),
        scale=255.0,
        img_hw=(256, 256),
        verified="2026-09-28, task_specs.pkl bands + band_stats.json (uint8, max 255) + samples",
    ),
    "m-so2sat": TaskSpec(
        zenodo=8276567,
        rgb_bands=("04", "03", "02"),
        scale=0.4095,
        img_hw=(32, 32),
        verified="2026-09-28, task_specs.pkl bands + band_stats.json (float reflectance, p99 0.3-0.4)",
    ),
    "m-brick-kiln": TaskSpec(
        zenodo=8276815,
        rgb_bands=("04", "03", "02"),
        scale=4095.0,
        img_hw=(64, 64),
        verified="2026-09-28, task_specs.pkl bands + band_stats.json (S2 DN, p99 1.2-1.9k)",
    ),
    "m-pv4ger": TaskSpec(
        zenodo=8276975,
        rgb_bands=("Red", "Green", "Blue"),
        scale=255.0,
        img_hw=(256, 256),  # task_specs claims 320x320; every sample is 256x256 (shape guard)
        verified="2026-09-29, task_specs.pkl bands + band_stats.json (uint8, max 254) + samples",
    ),
    "m-bigearthnet": TaskSpec(
        zenodo=8277477,
        rgb_bands=("04", "03", "02"),
        scale=4095.0,
        img_hw=(120, 120),
        verified="2026-09-28, task_specs.pkl + SatDiFuser datasets/m_bigearthnet.py (/4095, bands 04/03/02)",
    ),
}


def dir_name(cls: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", cls).strip("_")


def manifest_path(task: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", f"{F.key(task)}.json")


def download(record: int, dest: str) -> None:
    """Fetch the task's Zenodo files into dest (md5-checked), unzip data.zip, delete the zip."""
    if os.path.exists(os.path.join(dest, ".data_unzipped")):
        return
    os.makedirs(dest, exist_ok=True)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as r:
        files = {f["key"]: f for f in json.load(r)["files"] if f["key"] in ZENODO_FILES}
    for key, f in files.items():
        path = os.path.join(dest, key)
        print(f"  {key} ({f['size'] / 1e9:.2f} GB)", flush=True)
        with urllib.request.urlopen(f["links"]["self"]) as r, open(path, "wb") as out:
            shutil.copyfileobj(r, out, 1 << 22)
        md5 = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                md5.update(chunk)
        if f"md5:{md5.hexdigest()}" != f["checksum"]:
            raise SystemExit(f"{path}: md5 {md5.hexdigest()} != Zenodo {f['checksum']}")
    with zipfile.ZipFile(os.path.join(dest, "data.zip")) as z:
        z.extractall(dest)
    os.remove(os.path.join(dest, "data.zip"))
    open(os.path.join(dest, ".data_unzipped"), "w").close()


def _class_names(label_type) -> list[str]:
    # geobench names the attribute class_names (Classification) / class_name (MultiLabel)
    return list(getattr(label_type, "class_names", None) or label_type.class_name)


def export(task: str, dataset_dir: str, out: str) -> dict:
    """Write the RGB tree; return its manifest (classes + per-split counts and pixel hash)."""
    import geobench  # deferred: heavy import, and the error should name the missing dep

    spec, ds_key = SPECS[task], F.key(task)
    sizes = F.official_spec(ds_key)["sizes"]
    label_type = geobench.load_task_specs(dataset_dir).label_type
    multilabel = type(label_type).__name__ == "MultiLabelClassification"
    if multilabel != F.official_spec(ds_key).get("multilabel", False):
        raise SystemExit(
            f"{task}: shipped label type {type(label_type).__name__} != OFFICIAL multilabel flag"
        )
    names = _class_names(label_type)
    dirs = [spec.name_map[c] if spec.name_map else dir_name(c) for c in names]
    if len(set(dirs)) != len(dirs):
        raise SystemExit(f"class dir names collide: {dirs}")
    # label index = position in the pipeline's class list (m-eurosat's differs from geobench's)
    classes = F.official_classes(ds_key) if ds_key == "m_eurosat" else dirs
    if sorted(classes) != sorted(dirs):
        raise SystemExit("mapped class set != the pipeline's class list")
    to_idx = np.array([classes.index(d) for d in dirs])

    manifest = {"classes": classes, "splits": {}}
    for split, out_split in OUT_SPLIT.items():
        ds = geobench.GeobenchDataset(dataset_dir, split=split, partition_name="default")
        if len(ds) != sizes[out_split]:
            raise SystemExit(
                f"{split}: got {len(ds)} samples, expected {sizes[out_split]} -- wrong partition?"
            )
        digest, counts = hashlib.sha256(), np.zeros(len(classes), np.int64)
        clipped, sample_names, ys = np.zeros(3), [], []
        for i in range(len(ds)):
            sample = ds[i]
            image, _ = sample.pack_to_3d(band_names=spec.rgb_bands)
            if image.shape != (*spec.img_hw, 3):
                raise SystemExit(f"{split}[{i}] has shape {image.shape}, expected {(*spec.img_hw, 3)}")
            px = (np.clip(image.astype(np.float32) / spec.scale, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            clipped += ((px == 0) | (px == 255)).mean((0, 1))
            if multilabel:
                y = np.zeros(len(classes), np.int8)
                y[to_idx[np.flatnonzero(sample.label)]] = 1
                counts += y
                d = os.path.join(out, out_split)
            else:
                y = np.array(to_idx[int(sample.label)])
                counts[y] += 1
                d = os.path.join(out, out_split, classes[int(y)])
            os.makedirs(d, exist_ok=True)
            Image.fromarray(px).save(os.path.join(d, f"{sample.sample_name}.png"))
            for part in (sample.sample_name.encode(), y.tobytes(), px.tobytes()):
                digest.update(part)
            sample_names.append(sample.sample_name)
            ys.append(y)
            if i % 2000 == 0:
                print(f"  {split} {i}/{len(ds)}", flush=True)
        frac = clipped / len(ds)
        if (frac > MAX_CLIPPED).any():
            raise SystemExit(
                f"{split}: {np.round(frac, 3).tolist()} of R/G/B pixels clip at 0/255 (> {MAX_CLIPPED}) "
                f"-- scale {spec.scale} is wrong for this task's value range"
            )
        if multilabel:
            np.savez(os.path.join(out, out_split, "labels.npz"), names=np.array(sample_names), y=np.stack(ys))
        manifest["splits"][out_split] = {
            "n": len(ds),
            "counts": counts.tolist(),
            "sha256": digest.hexdigest(),
        }
        print(f"{split}: {len(ds)} images; per-class: {dict(zip(classes, counts.tolist()))}", flush=True)
    return manifest


def check_manifest(task: str, manifest: dict, write: bool = False) -> None:
    path = manifest_path(task)
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"manifest written -> {path} (commit it)")
        return
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: create it on the workstation with --write-manifest, then commit")
    with open(path) as f:
        ref = json.load(f)
    if ref != manifest:
        bad = [s for s in ref["splits"] if ref["splits"][s] != manifest["splits"].get(s)]
        raise SystemExit(f"{task}: export differs from the committed manifest (classes or splits {bad})")
    print(f"export matches the committed manifest {path}")


def remove_raw(dataset_dir: str) -> None:
    """Delete the sample .hdf5 files, keeping the small metadata (task_specs, partitions)."""
    for f in glob.glob(os.path.join(dataset_dir, "*.hdf5")):
        os.remove(f)
    if os.path.exists(marker := os.path.join(dataset_dir, ".data_unzipped")):
        os.remove(marker)
    print(f"removed raw samples in {dataset_dir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(SPECS))
    p.add_argument(
        "--dataset-dir", required=True, help="dir with task_specs.pkl, default_partition.json, *.hdf5"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--download", action="store_true", help="fetch + md5-check the Zenodo record first")
    p.add_argument(
        "--rm-raw", action="store_true", help="delete the sample .hdf5 files after a verified export"
    )
    p.add_argument("--write-manifest", action="store_true", help="workstation only: (re)create the manifest")
    args = p.parse_args(argv)
    if args.download:
        download(SPECS[args.task].zenodo, args.dataset_dir)
    check_manifest(args.task, export(args.task, args.dataset_dir, args.out), write=args.write_manifest)
    if args.rm_raw:
        remove_raw(args.dataset_dir)


if __name__ == "__main__":
    main()
