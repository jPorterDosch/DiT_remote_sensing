"""Export a GEO-Bench classification task to the RGB PNG tree the pipeline consumes:

    python -m eval.export_geobench --task m-forestnet --dataset-dir data/m_forestnet_meta \\
        --out data/m_forestnet_rgb [--download] [--rm-raw] [--write-manifest]
    python -m eval.export_geobench --task m-forestnet --out data/m_forestnet_rgb   # verify only

    single-label: <out>/{train,val,test}/<ClassDir>/<sample_name>.png
    multi-label:  <out>/{train,val,test}/<sample_name>.png + <out>/<split>/labels.npz (names, y)
    segmentation: <out>/{train,val,test}/images/<sample_name>.png + masks/<sample_name>.png
                  (uint8 class index in eval/features.OFFICIAL_SEG class order, SEG_IGNORE = 255
                  for the task's ignore classes; SEG_SPECS)

Every TaskSpec value is read from the SHIPPED artifact (default_partition.json counts,
band names, dtype range, sample shapes) or a reference loader's source -- never from a
paper table (CLAUDE.md rule 16). Split sizes come from eval/features.OFFICIAL.

REPLICATION: tree_manifest() reads the written tree back and hashes every image's name,
label and DECODED pixels (not PNG bytes: zlib versions differ) per split; the export then
must equal the committed eval/manifests/<key>.json (written once on the workstation with
--write-manifest), so the ISAAC tree is provably the one the workstation gates ran on.
Omit --dataset-dir to re-verify an existing tree (no raw data needed). The manifest's class
list IS the registry's class list for the tasks registered after m-eurosat.
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


@dataclass(frozen=True)
class SegSpec:
    zenodo: int  # GEO-Bench v1.0 record (the one shipping default_partition.json)
    rgb_bands: tuple
    scale: float  # divide raw band values by this, then clip to [0,1]
    img_hw: tuple  # checked on EVERY sample
    classes: tuple  # the shipped label_type.class_names, in order; the mask index = position
    ignore: tuple = ()  # class names mapped to SEG_IGNORE (excluded from the metric)
    verified: str = ""


# Segmentation tasks (eval/features.OFFICIAL_SEG). Mask values: the shipped Band label data,
# uint8 class index into label_type.class_names; `ignore` classes become SEG_IGNORE.
SEG_SPECS: dict[str, SegSpec] = {
    "m-pv4ger-seg": SegSpec(
        zenodo=8277054,
        rgb_bands=("Red", "Green", "Blue"),
        scale=255.0,
        img_hw=(320, 320),
        classes=("no solar pv", "solar pv"),
        verified="2026-10-17, default_partition.json 3221/403/403, task_specs SegmentationClasses, band_stats uint8 max 255, samples 320x320 uint8",
    ),
    "m-chesapeake-landcover": SegSpec(
        zenodo=8276993,
        rgb_bands=("Red", "Green", "Blue"),
        scale=1.0,  # SatDiFuser datasets/m_chesapeake.py: clip(image, 0, 1), no division (float bands)
        img_hw=(256, 256),
        # The shipped task_specs carries n_classes=7 and NO class names (SegmentationClasses(...,
        # class_names=None)); GEO-Bench's converter (make_benchmark/dataset_converters/
        # chesapeake_land_cover.py) lists the names with "no-data" LAST but never attaches them,
        # and the masks come from torchgeo's ChesapeakeCVPR class_set=7, whose index 0 is no-data
        # and 1..6 water .. impervious-roads. The shipped label_stats.json agrees: class 0 occurs
        # in 4.4 % of samples (3.1 % of pixels, tile borders), class 6 in most (2.9 %, roads).
        # SatDiFuser's configs/tasks/mchesapeake.yaml copies the converter's (wrong) order.
        classes=(
            "no-data",
            "water",
            "tree-canopy-forest",
            "low-vegetation-field",
            "barren-land",
            "impervious-other",
            "impervious-roads",
        ),
        ignore=("no-data",),
        verified="2026-10-17, default_partition.json 3000/1000/1000, task_specs n_classes 7 (no names), band_stats float 0..1, label_stats.json class frequencies, converter + torchgeo class_set=7",
    ),
    "m-SA-crop-type": SegSpec(
        zenodo=8277134,
        rgb_bands=("04", "03", "02"),
        scale=255.0,  # shipped band_stats: every band uint8-range (max 255; p99 R/G/B 87/63/45), NOT S2 DN
        img_hw=(256, 256),
        # Converter make_benchmark/dataset_converters/crop_type_south_africa.py crop_labels (index =
        # position; the mask IS the competition label raster); shipped label_stats.json: class 0
        # ("No Data") covers 48 % of the pixels and occurs in every sample -> ignored.
        classes=(
            "No Data",
            "Lucerne/Medics",
            "Planted pastures (perennial)",
            "Fallow",
            "Wine grapes",
            "Weeds",
            "Small grain grazing",
            "Wheat",
            "Canola",
            "Rooibos",
        ),
        ignore=("No Data",),
        verified="2026-10-19, export_geobench --count (job 6439355): 3000/1000/1000, n_classes 10, bands max 255 / p99 45-145 (scale 4095 fired the clip guard: 17.8 % of blue at 0); converter crop_labels + label_stats.json 2026-10-17",
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
    """Write the RGB tree, then return tree_manifest() of what landed on disk."""
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

    for split, out_split in OUT_SPLIT.items():
        ds = geobench.GeobenchDataset(dataset_dir, split=split, partition_name="default")
        if len(ds) != sizes[out_split]:
            raise SystemExit(
                f"{split}: got {len(ds)} samples, expected {sizes[out_split]} -- wrong partition?"
            )
        clipped, sample_names, ys = np.zeros(3), [], []
        for i in range(len(ds)):
            sample = ds[i]
            image, _ = sample.pack_to_3d(band_names=spec.rgb_bands)
            if image.shape != (*spec.img_hw, 3):
                raise SystemExit(f"{split}[{i}] has shape {image.shape}, expected {(*spec.img_hw, 3)}")
            px = (np.clip(image.astype(np.float32) / spec.scale, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            clipped += ((px == 0) | (px == 255)).mean((0, 1))
            if multilabel:
                y = np.zeros(len(classes), np.int64)
                y[to_idx[np.flatnonzero(sample.label)]] = 1
                d = os.path.join(out, out_split)
            else:
                y = to_idx[int(sample.label)]
                d = os.path.join(out, out_split, classes[y])
            os.makedirs(d, exist_ok=True)
            Image.fromarray(px).save(os.path.join(d, f"{sample.sample_name}.png"))
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
    return tree_manifest(ds_key, out, classes)


def count_seg(task: str, dataset_dir: str) -> dict:
    """What the SHIPPED artifact says (rule 16): partition sizes, label type + class names, bands,
    patch size, band stats, one sample's dtype / range / mask values. Printed for the
    OFFICIAL_SEG / SEG_SPECS entries; never typed from a paper."""
    import geobench

    with open(os.path.join(dataset_dir, "default_partition.json")) as f:
        part = json.load(f)
    ts = geobench.load_task_specs(dataset_dir)
    lt = ts.label_type
    out = {
        "sizes": {OUT_SPLIT[k]: len(v) for k, v in part.items()},
        "label_type": type(lt).__name__,
        "classes": list(getattr(lt, "class_names", None) or []),
        "n_classes": getattr(lt, "n_classes", None),
        "bands": [b.name for b in ts.bands_info],
        "patch_size": list(ts.patch_size),
    }
    bs_path = os.path.join(dataset_dir, "band_stats.json")
    if os.path.exists(bs_path):
        with open(bs_path) as f:
            out["band_stats"] = {
                k: {s: v.get(s) for s in ("min", "max", "percentile_99")} for k, v in json.load(f).items()
            }
    ds = geobench.GeobenchDataset(dataset_dir, split="train", partition_name="default")
    s = ds[0]
    spec = SEG_SPECS.get(task)
    bands = spec.rgb_bands if spec else tuple(out["bands"][:3])
    img, _ = s.pack_to_3d(band_names=bands)
    out["sample"] = {"name": s.sample_name, "shape": list(img.shape), "dtype": str(img.dtype), "min": float(img.min()),
                     "max": float(img.max()), "mask_shape": list(s.label.data.shape), "mask_dtype": str(s.label.data.dtype),
                     "mask_values": np.unique(s.label.data).tolist()}  # fmt: skip
    print(json.dumps(out, indent=1))
    return out


def export_seg(task: str, dataset_dir: str, out: str) -> dict:
    """Write images/ + masks/ per split, then return seg_tree_manifest() of what landed."""
    import geobench

    spec, ds_key = SEG_SPECS[task], F.key(task)
    reg = F.official_seg_spec(ds_key)
    lt = geobench.load_task_specs(dataset_dir).label_type
    if type(lt).__name__ != "SegmentationClasses":
        raise SystemExit(f"{task}: shipped label type {type(lt).__name__} is not SegmentationClasses")
    shipped = tuple(lt.class_names) if lt.class_names else None
    if shipped is not None and shipped != spec.classes:
        raise SystemExit(f"{task}: shipped class names {shipped} != SEG_SPECS {spec.classes}")
    if lt.n_classes != len(spec.classes):
        raise SystemExit(f"{task}: shipped n_classes {lt.n_classes} != SEG_SPECS {len(spec.classes)} classes")
    kept = [c for c in spec.classes if c not in spec.ignore]
    if [dir_name(c).lower() for c in kept] != [dir_name(c).lower() for c in reg["classes"]]:
        raise SystemExit(
            f"{task}: kept classes {kept} != OFFICIAL_SEG classes {reg['classes']} (dir_name, case-insensitive)"
        )
    # shipped index -> our index (kept classes in order), ignore -> SEG_IGNORE
    lut = np.full(256, F.SEG_IGNORE, np.uint8)
    for i, c in enumerate(spec.classes):
        if c not in spec.ignore:
            lut[i] = kept.index(c)
    for split, out_split in OUT_SPLIT.items():
        ds = geobench.GeobenchDataset(dataset_dir, split=split, partition_name="default")
        if len(ds) != reg["sizes"][out_split]:
            raise SystemExit(
                f"{split}: got {len(ds)} samples, expected {reg['sizes'][out_split]} -- count the partition (OFFICIAL_SEG)"
            )
        clipped = np.zeros(3)
        os.makedirs(os.path.join(out, out_split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, out_split, "masks"), exist_ok=True)
        for i in range(len(ds)):
            sample = ds[i]
            image, _ = sample.pack_to_3d(band_names=spec.rgb_bands)
            if image.shape != (*spec.img_hw, 3):
                raise SystemExit(f"{split}[{i}] has shape {image.shape}, expected {(*spec.img_hw, 3)}")
            px = (np.clip(image.astype(np.float32) / spec.scale, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            clipped += ((px == 0) | (px == 255)).mean((0, 1))
            mask = np.asarray(sample.label.data)
            if mask.shape != tuple(reg["mask_hw"]) or mask.max() >= len(spec.classes):
                raise SystemExit(
                    f"{split}[{i}] mask {mask.shape} max {mask.max()}: expected {reg['mask_hw']} with < {len(spec.classes)} classes"
                )
            Image.fromarray(px).save(os.path.join(out, out_split, "images", f"{sample.sample_name}.png"))
            Image.fromarray(lut[mask.astype(np.int64)]).save(
                os.path.join(out, out_split, "masks", f"{sample.sample_name}.png")
            )
            if i % 1000 == 0:
                print(f"  {split} {i}/{len(ds)}", flush=True)
        frac = clipped / len(ds)
        if (frac > MAX_CLIPPED).any():
            raise SystemExit(
                f"{split}: {np.round(frac, 3).tolist()} of R/G/B pixels clip at 0/255 (> {MAX_CLIPPED}) -- scale {spec.scale} is wrong"
            )
    return seg_tree_manifest(ds_key, out)


def seg_tree_manifest(ds_key: str, root: str) -> dict:
    """Manifest of an exported segmentation tree, read back from disk: per split n, pixel counts
    per class (+ ignore) and one sha256 over every (name, image shape + pixels, mask pixels)."""
    reg = F.official_seg_spec(ds_key)
    n_cls = len(reg["classes"])
    out = {"classes": list(reg["classes"]), "ignore": F.SEG_IGNORE, "splits": {}}
    for split in reg["sizes"]:
        images, masks = F.list_official_seg_split(ds_key, split, root)
        on_disk = glob.glob(os.path.join(root, split, "**", "*.png"), recursive=True)
        if sorted(on_disk) != sorted(images + masks):
            raise SystemExit(
                f"{root}/{split}: {len(on_disk)} PNGs on disk, {len(images) + len(masks)} listed -- stray/misplaced"
            )
        digest, counts = hashlib.sha256(), np.zeros(n_cls + 1, np.int64)
        for f, m in zip(images, masks, strict=True):
            px, mk = np.asarray(Image.open(f)), np.asarray(Image.open(m))
            if px.dtype != np.uint8 or px.ndim != 3 or px.shape[2] != 3:
                raise SystemExit(f"{f}: {px.dtype} {px.shape}, expected uint8 RGB")
            if mk.dtype != np.uint8 or mk.ndim != 2 or mk.shape != px.shape[:2]:
                raise SystemExit(f"{m}: {mk.dtype} {mk.shape}, expected a uint8 mask of {px.shape[:2]}")
            vals = np.bincount(mk.ravel(), minlength=256)
            if vals[n_cls : F.SEG_IGNORE].any():
                raise SystemExit(f"{m}: mask values outside the {n_cls} classes + ignore")
            counts[:n_cls] += vals[:n_cls]
            counts[n_cls] += vals[F.SEG_IGNORE]
            for part in (os.path.basename(f).encode(), str(px.shape).encode(), px.tobytes(), mk.tobytes()):
                digest.update(part)
        out["splits"][split] = {
            "n": len(images),
            "pixel_counts": counts.tolist(),
            "sha256": digest.hexdigest(),
        }
        print(f"{split}: {len(images)} images; pixels per class (+ignore): {counts.tolist()}", flush=True)
    return out


def tree_manifest(ds_key: str, root: str, classes: list[str] | None = None) -> dict:
    """Manifest of an exported tree, READ BACK FROM DISK in the pipeline's own listing order
    (features.list_official_split, which also checks split sizes): per split, the per-class
    counts and one sha256 over every image's (file name, label, decoded shape + pixels).
    Covers the PNGs actually written, labels.npz and the class-dir layout, refuses stray
    PNGs, and needs no raw data -- so any tree can be re-verified on any machine."""
    classes = list(classes or F.official_classes(ds_key))
    out = {"classes": classes, "splits": {}}
    for split in F.official_spec(ds_key)["sizes"]:
        files, y = F.list_official_split(ds_key, split, root, classes)
        on_disk = glob.glob(os.path.join(root, split, "**", "*.png"), recursive=True)
        if sorted(on_disk) != sorted(files):
            raise SystemExit(
                f"{root}/{split}: {len(on_disk)} PNGs on disk, {len(files)} listed -- stray/misplaced"
            )
        digest = hashlib.sha256()
        for f, label in zip(files, y, strict=True):
            px = np.asarray(Image.open(f))
            if px.dtype != np.uint8 or px.ndim != 3 or px.shape[2] != 3:
                raise SystemExit(f"{f}: {px.dtype} {px.shape}, expected uint8 RGB")
            for part in (
                os.path.basename(f).encode(),
                str(px.shape).encode(),
                np.int64(label).tobytes(),
                px.tobytes(),
            ):
                digest.update(part)
        counts = y.sum(0) if y.ndim == 2 else np.bincount(y, minlength=len(classes))
        out["splits"][split] = {"n": len(files), "counts": counts.tolist(), "sha256": digest.hexdigest()}
        print(f"{split}: {len(files)} images; per-class: {dict(zip(classes, counts.tolist()))}", flush=True)
    return out


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
    p.add_argument("--task", required=True, choices=sorted(SPECS) + sorted(SEG_SPECS))
    p.add_argument(
        "--count",
        action="store_true",
        help="segmentation: print what the shipped artifact says, write nothing",
    )
    p.add_argument(
        "--out", required=True, help="the RGB tree (written, or verified if --dataset-dir is omitted)"
    )
    p.add_argument(
        "--dataset-dir", default=None, help="raw geobench dir (task_specs.pkl, partitions, *.hdf5): export"
    )
    p.add_argument("--download", action="store_true", help="fetch + md5-check the Zenodo record first")
    p.add_argument(
        "--rm-raw", action="store_true", help="delete the sample .hdf5 files after a verified export"
    )
    p.add_argument("--write-manifest", action="store_true", help="workstation only: (re)create the manifest")
    args = p.parse_args(argv)
    if args.task in SEG_SPECS:
        if args.download and args.dataset_dir:
            download(SEG_SPECS[args.task].zenodo, args.dataset_dir)
        if args.count:
            if not args.dataset_dir:
                raise SystemExit("--count needs --dataset-dir")
            count_seg(args.task, args.dataset_dir)
            return
        manifest = (
            seg_tree_manifest(F.key(args.task), args.out)
            if args.dataset_dir is None
            else export_seg(args.task, args.dataset_dir, args.out)
        )
        check_manifest(args.task, manifest, write=args.write_manifest)
        if args.rm_raw:
            remove_raw(args.dataset_dir)
        return
    if args.dataset_dir is None:
        if args.download or args.rm_raw:
            raise SystemExit("--download/--rm-raw need --dataset-dir")
        manifest = tree_manifest(F.key(args.task), args.out)  # verify an existing tree
    else:
        if args.download:
            download(SPECS[args.task].zenodo, args.dataset_dir)
        manifest = export(args.task, args.dataset_dir, args.out)
    check_manifest(args.task, manifest, write=args.write_manifest)
    if args.rm_raw:
        remove_raw(args.dataset_dir)


if __name__ == "__main__":
    main()
