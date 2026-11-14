"""Feature loading, views and identity guards shared by probe.py and extract_dino.py.

A probe reads ONE feature file (cv/budget/mlp) or one file per split (official) and a
VIEW of it. Views reproduce exactly what the validated prototypes fed their probes:

  flux  sec13:<i>   DiTF-normalized; base = t-index i (PCA-512-protected in the fold),
                    the other timesteps appended raw  (6ac arm A1, section 13)
        t:<i>       DiTF-normalized single timestep i  (6ac arm A2)
        concat      DiTF-normalized all-t concat  (Q2 flux_7t)
  dino  cls | mp | clsmp | cls4 | cls4mp  (cls4 = last-4-block CLS concat; see DINO_OFFICIAL_VIEWS)
  vae   full | pool8 | pool4 | pool2 | pool1  (Q1 poolings of the 16x32x32 clean latent;
        poolN = N x N grid x 16 ch: 1024 / 256 / 64 / 16-d. Q1's "pool 4x4 (256d)" arm is pool8)

OFFICIAL-protocol FLUX candidates are RAW (not DiTF) single-t slices + the raw concat, as
m_eurosat_probe used; that is a protocol property, not a view (see probe.official_flux).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

DISCARD = [154, 1446]  # DiTF massive-activation channels
# The DINO arm's official candidates: DINOv2/v3 linear eval's own grid, n_last_blocks in
# {1, 4} x use_avgpool (dinov3/eval/linear.py create_linear_input), selected on val with C.
DINO_OFFICIAL_VIEWS = "cls,clsmp,cls4,cls4mp"

# cv/budget/mlp identity: the ordered image set every paired arm must consume. The
# RESISC45 5,000 are the section-13 inversion cache's paths, in cache order (6ac).
CV_IDENTITY = {
    "resisc45": "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
}


def _eurosat_classes() -> list[str]:
    from data.eurosat_dataset import EUROSAT_CLASSES

    return list(EUROSAT_CLASSES)


def _geobench(name: str, train: int, val: int, test: int, **extra) -> dict:
    """An OFFICIAL entry whose class list (LABEL order) is the committed export manifest's."""

    def classes() -> list[str]:
        with open(os.path.join(REPO_ROOT, "eval", "manifests", f"{name}.json")) as f:
            return json.load(f)["classes"]

    sizes = {"train": train, "val": val, "test": test}
    return {"sizes": sizes, "root": f"data/{name}_rgb", "classes": classes, **extra}


# Official-protocol datasets -- the ONLY place a GEO-Bench task is registered for
# eval.probe / eval.extract_dino / eval.sweep / run.py (src/datasets/geobench.py). Adding a
# task = one entry here plus its TaskSpec in eval/export_geobench.py.
#   sizes:      per-split counts COUNTED from the shipped partition (rule 16), never a paper
#   root:       exported RGB tree, <root>/<split>/<Class>/*.png (multi-label: <root>/<split>/*.png
#               + labels.npz)
#   classes:    zero-arg callable -> class names in LABEL order (the exporter's order)
#   multilabel: labels are multi-hot rows; the metric is micro-F1 (SatDiFuser val_logger)
OFFICIAL = {
    "m_eurosat": {
        "sizes": {"train": 16200, "val": 996, "test": 996},
        "root": "data/m_eurosat_rgb",
        "classes": _eurosat_classes,
    },
    "m_forestnet": _geobench("m_forestnet", 6464, 989, 993),
    "m_so2sat": _geobench("m_so2sat", 19992, 986, 986),
    "m_brick_kiln": _geobench("m_brick_kiln", 15063, 999, 999),
    "m_pv4ger": _geobench("m_pv4ger", 11814, 999, 999),
    "m_bigearthnet": _geobench("m_bigearthnet", 20000, 1000, 1000, multilabel=True),
}


# Official-protocol SEGMENTATION tasks (eval/seg_probe.py, src/datasets/geobench_seg.py,
# src/tasks/seg_tokens.py, eval/extract_dino.py --tokens). Trees written by
# `python -m eval.export_geobench --task <task>`: <root>/<split>/images/<name>.png (RGB uint8)
# + <root>/<split>/masks/<name>.png (uint8 class index, IGNORE = SEG_IGNORE). Sizes COUNTED from
# the shipped default partition (rule 16; eval.export_geobench count), classes in label order
# from the shipped task_specs; the metric is mIoU over the classes listed (ignore pixels excluded).
SEG_IGNORE = 255
# classes = export_geobench.dir_name of the shipped class names, ignore classes removed
# (export_geobench.SEG_SPECS.ignore -> SEG_IGNORE pixels); keys = F.key(task).
OFFICIAL_SEG = {
    "m_pv4ger_seg": {  # counted 2026-10-17: default_partition.json 3221 / 403 / 403
        "sizes": {"train": 3221, "val": 403, "test": 403},
        "root": "data/m_pv4ger_seg",
        "classes": ["no_solar_pv", "solar_pv"],
        "mask_hw": (320, 320),
    },
    "m_chesapeake_landcover": {  # counted 2026-10-17: default_partition.json 3000 / 1000 / 1000; class 0 (no-data) ignored
        "sizes": {"train": 3000, "val": 1000, "test": 1000},
        "root": "data/m_chesapeake_landcover",
        "classes": [
            "water",
            "tree_canopy_forest",
            "low_vegetation_field",
            "barren_land",
            "impervious_other",
            "impervious_roads",
        ],
        "mask_hw": (256, 256),
    },
    "m_SA_crop_type": {  # counted 2026-10-19 (job 6439355): default_partition.json 3000 / 1000 / 1000; n_classes 10, index 0 "No Data" ignored
        "sizes": {"train": 3000, "val": 1000, "test": 1000},
        "root": "data/m_SA_crop_type",
        "classes": [
            "Lucerne_Medics",
            "Planted_pastures_perennial",
            "Fallow",
            "Wine_grapes",
            "Weeds",
            "Small_grain_grazing",
            "Wheat",
            "Canola",
            "Rooibos",
        ],
        "mask_hw": (256, 256),
    },
}


def official_seg_spec(dataset: str) -> dict:
    if dataset not in OFFICIAL_SEG:
        raise SystemExit(
            f"{dataset}: not registered in eval/features.OFFICIAL_SEG (count the shipped partition "
            f"first, rule 16). Registered: {sorted(OFFICIAL_SEG)}"
        )
    return OFFICIAL_SEG[dataset]


def list_official_seg_split(dataset: str, split: str, root: str | None = None) -> tuple[list[str], list[str]]:
    """(image files, mask files) of one official segmentation split, sorted by name; every
    image must have its mask and the count must be the registered one."""
    spec = official_seg_spec(dataset)
    root = root or spec["root"]
    images = sorted(glob.glob(os.path.join(root, split, "images", "*.png")))
    masks = [os.path.join(root, split, "masks", os.path.basename(f)) for f in images]
    if len(images) != spec["sizes"][split]:
        raise SystemExit(
            f"{dataset} {split}: {len(images)} images under {root}, expected {spec['sizes'][split]}"
        )
    missing = [m for m in masks if not os.path.exists(m)]
    if missing:
        raise SystemExit(f"{dataset} {split}: {len(missing)} masks missing, e.g. {missing[:2]}")
    return images, masks


def path_key(p: str) -> str:
    """An image's identity independent of where the tree lives: its last 3 path components
    (<tree>/<split>/<file> or <split>/<class>/<file>). Absolute vs relative --dataset.path
    (banked m-eurosat caches vs sweep caches) must not make the same image two images."""
    return "/".join(os.path.normpath(p).split(os.sep)[-3:])


def key(task: str) -> str:
    """GEO-Bench task name -> OFFICIAL key (m-brick-kiln -> m_brick_kiln)."""
    return task.replace("-", "_")


def official_spec(dataset: str) -> dict:
    if dataset not in OFFICIAL:
        raise SystemExit(
            f"{dataset}: not registered in eval/features.OFFICIAL (count the shipped partition "
            f"first, rule 16). Registered: {sorted(OFFICIAL)}"
        )
    return OFFICIAL[dataset]


def official_classes(dataset: str) -> list[str]:
    return list(official_spec(dataset)["classes"]())


def ditf(f, m, discard=DISCARD):
    """Offline DiTF normalization. Source: _absorption_harness.ditf (verbatim). `discard` = the
    massive-activation channels zeroed first: FLUX's [154, 1446] by default; another backbone's
    caches carry their own list in meta ("discard_channels", meta_discard_channels)."""
    x = f.astype(np.float64).copy()
    x[:, :, list(discard)] = 0.0
    mu = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    x = (x - mu) / np.sqrt(v + 1e-6)
    x = (1 + m[None, :, 1, :]) * x + m[None, :, 0, :]
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def load(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"missing features file {path}")
    return np.load(path, allow_pickle=True)


def load_identity(path: str) -> tuple[list[str], np.ndarray]:
    d = load(path)
    return [str(x) for x in d["paths"]], d["labels"].astype(np.int64)


def check_identity(d, id_paths: list[str], src: str) -> None:
    """Every paired arm must consume the identical ordered image list (rules 6/11)."""
    if "paths" not in d.files:
        raise SystemExit(f"{src}: no 'paths' field -- cannot verify image identity, refusing")
    got = [str(x) for x in d["paths"]]
    if got != id_paths:
        n_same = sum(a == b for a, b in zip(got, id_paths))
        raise SystemExit(
            f"{src}: paths differ from the identity list ({n_same}/{len(id_paths)} positions agree)"
        )


def view(kind: str, d, spec: str):
    """Returns ("plain", X) or ("sec13", (base, others))."""
    if kind == "flux":
        fi = ditf(d["feats"], d["mods"])
        if spec.startswith("sec13:"):
            i = int(spec.split(":")[1])
            base = fi[:, i, :]
            others = np.concatenate([fi[:, j, :] for j in range(fi.shape[1]) if j != i], axis=1)
            return "sec13", (base, others)
        if spec.startswith("t:"):
            return "plain", fi[:, int(spec.split(":")[1]), :]
        if spec == "concat":
            return "plain", np.ascontiguousarray(fi.reshape(len(fi), -1)).astype(np.float32)
    elif kind == "dino":
        fields = {
            "cls": ["cls"],
            "mp": ["mp"],
            "clsmp": ["cls", "mp"],
            "cls4": ["cls4"],
            "cls4mp": ["cls4", "mp"],
        }
        if spec in fields:  # read only the requested arrays: banked files have no cls4
            X = np.concatenate([d[f] for f in fields[spec]], axis=1)
            return "plain", np.ascontiguousarray(X).astype(np.float32)
    elif kind == "vae":
        lat = d["lat"]
        N, C_, H, W = lat.shape
        pools = {
            "full": lambda: lat.reshape(N, -1),
            # Names are the GRID size (poolN = N x N cells x 16 ch). The Q1 prototype labelled
            # its 8x8-grid arm "pool 4x4 (256d)" (4 = window, and 1024-d in fact); that arm
            # is `pool8` here so the gate stays bit-identical, and `pool4` is a true 4x4 grid.
            "pool8": lambda: lat.reshape(N, C_, 8, 4, 8, 4).mean(axis=(3, 5)).reshape(N, -1),
            "pool4": lambda: lat.reshape(N, C_, 4, 8, 4, 8).mean(axis=(3, 5)).reshape(N, -1),
            "pool2": lambda: lat.reshape(N, C_, 2, 16, 2, 16).mean(axis=(3, 5)).reshape(N, -1),
            "pool1": lambda: lat.mean(axis=(2, 3)),
        }
        if spec in pools:
            return "plain", np.ascontiguousarray(pools[spec]())
    raise SystemExit(f"unknown view {spec!r} for kind {kind!r} (see eval/features.py docstring)")


def cache_meta(path: str) -> dict:
    mp = path.replace(".npz", "_meta.json")
    return json.load(open(mp)) if os.path.exists(mp) else {}


# ------------------------------------------------------------------- FLUX cache pins
PINS_COMMON = {"guidance_scale": 1.0, "k": 28, "weights": "flux-dev"}
# Mode-specific pins every FLUX arm must state (rule 11: an ens1 or wrong-mode cache must
# not pass under an ens8 arm label). Source: m_eurosat_probe.ARMS pins.
REQUIRED_EXPECT = {"ONESHOT": ("ensemble_size",), "INVERSION": ("num_inversion_steps",)}


def parse_expect(items: list[str] | None) -> dict:
    """--expect KEY=VALUE ... -> dict; values parsed as JSON when possible (8 -> int)."""
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"--expect {it!r}: use KEY=VALUE")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def flux_pins(dataset: str, expect: dict) -> dict:
    mode = expect.get("extraction_mode")
    if mode not in REQUIRED_EXPECT:
        raise SystemExit("FLUX arms need --expect extraction_mode=ONESHOT|INVERSION (+ its mode pins)")
    missing = [k for k in REQUIRED_EXPECT[mode] if k not in expect]
    if missing:
        raise SystemExit(f"{mode} arms must also pin: " + " ".join(f"--expect {k}=..." for k in missing))
    return {**PINS_COMMON, "dataset": dataset, **expect}


def check_pins(meta: dict, pins: dict, src: str) -> None:
    if not meta:
        raise SystemExit(f"{src}: no _meta.json beside the cache -- cannot verify its identity, refusing")
    for k, want in pins.items():
        if meta.get(k) != want:
            raise SystemExit(f"{src}: meta {k}={meta.get(k)!r}, expected {want!r}")


# ------------------------------------------------------------ official-split loading


def npz_member_memmap(path: str, name: str) -> np.ndarray:
    """Read-only memmap of one array inside an UNCOMPRESSED .npz (np.savez stores members
    ZIP_STORED). np.load reads a member whole -- 31 GB for a multi-block m-bigearthnet train
    cache -- while a block slice needs ~1/B of it."""
    import struct
    import zipfile

    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit(f"{path}:{name} is compressed -- cannot memmap; use np.load")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise SystemExit(f"{path}:{name}: bad zip local header")
        fn_len, ex_len = struct.unpack("<HH", local[26:30])
        f.seek(info.header_offset + 30 + fn_len + ex_len)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


# Double-stream block feature definition: one source of truth in src/config_types.py (the
# extractors stamp it via double_block_stamp; every double-block load here requires it).
from config_types import DOUBLE_BLOCK_FEATURE, NUM_DOUBLE_BLOCKS  # noqa: E402


DOUBLE_BLOCK_RECORD = "double_block_verification.json"  # experiments/isaac/sweep/verify_double_block.py


def is_flux_meta(meta: dict) -> bool:
    """FLUX.1-dev caches (frozen, random_init or +lora) vs another backbone's (weights tag
    such as "geocore-9b"). Absent `weights` = a historical FLUX cache."""
    w = meta.get("weights", "flux-dev") or "flux-dev"
    return w.startswith("flux-dev") or w == "random_init"


def meta_num_double_blocks(meta: dict, src: str) -> int:
    """Double-block count the cache was extracted with: FLUX's constant for FLUX caches, the
    stamped `num_double_blocks` for any other backbone (missing -> refuse, never guess)."""
    if is_flux_meta(meta):
        return NUM_DOUBLE_BLOCKS
    n = meta.get("num_double_blocks")
    if not isinstance(n, int):
        raise SystemExit(
            f"{src}: weights={meta.get('weights')!r} cache has no num_double_blocks in meta -- re-extract"
        )
    return n


def meta_discard_channels(meta: dict, src: str) -> list[int]:
    """DiTF discard channels for a cache: FLUX's constant, else the stamped list (eval/discard_channels.py)."""
    if is_flux_meta(meta):
        return list(DISCARD)
    d = meta.get("discard_channels")
    if not isinstance(d, list):
        raise SystemExit(
            f"{src}: weights={meta.get('weights')!r} cache has no discard_channels in meta -- the DiTF view is undefined for it"
        )
    return [int(c) for c in d]


def check_double_block_stamp(meta: dict, k: int, src: str) -> None:
    if k >= meta_num_double_blocks(meta, src):
        return
    if meta.get("double_block_feature") != DOUBLE_BLOCK_FEATURE:
        raise SystemExit(
            f"{src}: double-stream block k={k} but meta double_block_feature="
            f"{meta.get('double_block_feature')!r} != {DOUBLE_BLOCK_FEATURE!r} -- extracted before the "
            "2026-10-05 fix (different feature definition) or unstamped; re-extract"
        )
    if "double_block_feature_note" in meta:
        # A hand-backfilled stamp (2026-10-08) is a claim, not evidence (rules 6/10): it counts only
        # when verify_double_block.py's PASS record vouches for this exact file (sha1).
        from eval import wb

        rec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(src))), DOUBLE_BLOCK_RECORD)
        try:
            with open(rec_path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = None
        real = os.path.realpath(src)  # models/ is a symlink: compare resolved paths
        entry = next((e for r, e in (rec or {}).get("caches", {}).items()
                      if os.path.realpath(r if os.path.isabs(r) else os.path.join(wb.REPO_ROOT, r)) == real), None)  # fmt: skip
        if not (rec and rec.get("pass") and rec.get("feature") == DOUBLE_BLOCK_FEATURE and entry
                and entry.get("sha1") == wb.file_identity(src)["sha1"]):  # fmt: skip
            raise SystemExit(
                f"{src}: backfilled double_block_feature stamp without a passing verification record for this "
                f"file in {rec_path} -- run experiments/isaac/sweep/verify_double_block.py on its cache root"
            )


def _block_slice(path: str, meta: dict, pins: dict) -> dict:
    """One block of a multi-block cache (layout N,K,B,C) as a single-block cache dict: feats
    (N, K, C), mods (K, 3, C). The pinned k must be one of the cache's recorded blocks; every
    other pin is still checked by the caller."""
    blocks = meta.get("k")
    if not isinstance(blocks, list) or pins.get("k") not in blocks:
        raise SystemExit(f"{path}: pinned k={pins.get('k')!r} not among the cache's blocks {blocks}")
    j = blocks.index(pins["k"])
    check_double_block_stamp(meta, pins["k"], path)
    feats = npz_member_memmap(path, "feats")
    if feats.ndim != 4 or feats.shape[2] != len(blocks):
        raise SystemExit(f"{path}: feats {feats.shape} do not match layout N,K,B,C with {len(blocks)} blocks")
    d = np.load(path)
    out = {f: d[f] for f in ("labels", "subset_indices", "paths", "timesteps")}
    out["feats"] = np.ascontiguousarray(feats[:, :, j, :])
    out["mods"] = d["mods"][:, j]
    return out


def _load_flux_one(path: str, pins: dict, split: str):
    """Source: m_eurosat_probe._load_one. A multi-block cache (meta layout "N,K,B,C") is
    served as the single-block slice at the pinned k, so every single-block consumer (eval.sweep
    block, eval.probe official) reads it unchanged."""
    meta = cache_meta(path)
    if meta.get("split", "train") != split:
        raise SystemExit(f"{path}: meta split={meta.get('split')!r} != {split!r}")
    if meta.get("layout") == "N,K,B,C":
        # meta-only checks first: a wrong cache is refused before the ~1/B strided read of feats
        check_pins({**meta, "k": pins["k"]}, pins, path)
        d = _block_slice(path, meta, pins)
        return d, {**meta, "k": pins["k"], "multiblock_k": meta["k"]}
    d = np.load(path)
    check_pins(meta, pins, path)
    if isinstance(pins.get("k"), int):
        check_double_block_stamp(meta, pins["k"], path)
    return d, meta


def load_flux_split(
    pattern: str, pins: dict, split: str, n_expect: int, smoke: bool = False, with_mods: bool = False
):
    """One split of a FLUX run.py cache, merging shards. Source: m_eurosat_probe.load_split
    (per-arm pins passed in via flux_pins). Returns feats in DATASET order. with_mods=True adds
    the block's adaLN modulation (T, 3, C) under meta["mods"] (identical across shards, checked)
    for the DiTF transform of probe.official_flux; off by default so meta stays JSON-serialisable
    for every other consumer."""
    hits = sorted(glob.glob(pattern.format(split=split)))
    if not hits:
        raise SystemExit(f"no caches for {pattern.format(split=split)}")
    parts = [_load_flux_one(h, pins, split) for h in hits]
    if len(parts) == 1 and parts[0][1].get("num_shards", 1) in (1, None):
        d, meta = parts[0]
        feats, labels, idx, paths = d["feats"], d["labels"], d["subset_indices"], d["paths"]
        meta = {**meta, "shard_seeds": [(meta.get("seed"), meta.get("eps_seed"))]}
        mods_parts = [d]
    else:
        unsharded = [h for h, (_, m) in zip(hits, parts) if m.get("num_shards", 1) in (1, None)]
        if unsharded:
            raise SystemExit(
                f"{split}: pattern matches {len(hits)} caches but these are unsharded: {unsharded} "
                "-- the glob spans more than one extraction; narrow --features to one run dir"
            )
        parts.sort(key=lambda p: p[1]["shard_index"])
        n_shards = parts[0][1]["num_shards"]
        got = [p[1]["shard_index"] for p in parts]
        if len(set(got)) != len(got):
            dup = sorted({g for g in got if got.count(g) > 1})
            raise SystemExit(
                f"{split}: duplicate shard indices {dup} across {len(hits)} caches -- the glob "
                "spans more than one extraction generation (e.g. banked shared-seed shards beside "
                "per-shard-seed ones); move one generation aside or narrow --features"
            )
        if got != list(range(n_shards)):
            raise SystemExit(f"{split}: have shards {got}, expected 0..{n_shards - 1}")
        idx = np.concatenate([p[0]["subset_indices"] for p in parts])
        if len(np.unique(idx)) != len(idx) or len(idx) != n_expect:
            raise SystemExit(f"{split}: merged shard indices are not disjoint / complete (n={len(idx)})")
        if not smoke and not np.array_equal(np.sort(idx), np.arange(n_expect)):
            raise SystemExit(f"{split}: merged shard indices are not a cover of arange({n_expect})")
        shard_seeds = [(p[1].get("seed"), p[1].get("eps_seed")) for p in parts]
        if len(set(shard_seeds)) < len(shard_seeds):
            # Not a refusal: banked m-eurosat inversion shards (pre-2026-09-26) share seed 42.
            print(
                f"WARNING {split}: shards share (seed, eps_seed) {shard_seeds} -- position-paired "
                "noise across shards (audit F1, train-only); new extractions use per-shard seeds"
            )
        feats = np.concatenate([p[0]["feats"] for p in parts])
        labels = np.concatenate([p[0]["labels"] for p in parts])
        paths = np.concatenate([p[0]["paths"] for p in parts])
        # Every shard's (seed, eps_seed), not just shard 0's: the cross-split seed guard in
        # probe.official_flux must see shard >= 1 too (a 42+IDX shard collides with test=43).
        meta = {**parts[0][1], "shard_seeds": shard_seeds}
        mods_parts = [p[0] for p in parts]
    if with_mods:  # only the DiTF transform needs the mods; synthetic / legacy caches may lack them
        for pd, h in zip(mods_parts, hits):
            keys = pd.files if hasattr(pd, "files") else pd.keys()  # NpzFile or a _block_slice dict
            if "mods" not in keys:
                raise SystemExit(
                    f"{split}: {h} has no 'mods' array; the DiTF transform needs the block's adaLN mods"
                )
        mods = mods_parts[0]["mods"]
        for h, pd in zip(hits[1:], mods_parts[1:]):
            if pd["mods"].shape != mods.shape or not np.allclose(pd["mods"], mods, rtol=1e-5, atol=1e-6):
                raise SystemExit(f"{split}: shard {h} carries different adaLN mods than shard 0")
        if mods.ndim != 3 or mods.shape[0] != feats.shape[1] or mods.shape[2] != feats.shape[2]:
            raise SystemExit(f"{split}: mods {mods.shape} do not match feats {feats.shape} as (T, 3, C)")
        meta = {**meta, "mods": np.asarray(mods)}
    if feats.shape[0] != n_expect:
        raise SystemExit(f"{split}: N={feats.shape[0]}, expected {n_expect}")
    order = np.argsort(idx)
    ts = [int(t) for t in parts[0][0]["timesteps"]]
    return feats[order].astype(np.float64), labels[order], ts, meta, [str(p) for p in paths[order]], hits


def list_official_split(dataset: str, split: str, root: str | None = None, classes: list | None = None):
    """Image files + labels of one official split, in class-list order then sorted names
    (multi-label: labels.npz order, y multi-hot). `classes` overrides the registry's list (the
    exporter, before a manifest exists). Source: dinov2_m_eurosat.list_split (size check
    against the OFFICIAL entry)."""
    spec = official_spec(dataset)
    root = root or spec["root"]
    if spec.get("multilabel"):
        d = np.load(os.path.join(root, split, "labels.npz"))
        files = [os.path.join(root, split, f"{n}.png") for n in d["names"]]
        labels = d["y"].astype(np.int64)
        if not all(map(os.path.exists, files)):
            raise SystemExit(f"{dataset} {split}: labels.npz lists images missing under {root}")
    else:
        files, labels = [], []
        for ci, cls in enumerate(classes or official_classes(dataset)):
            for f in sorted(glob.glob(os.path.join(root, split, cls, "*.png"))):
                files.append(f)
                labels.append(ci)
        labels = np.array(labels, dtype=np.int64)
    if len(files) != spec["sizes"][split]:
        raise SystemExit(
            f"{dataset} {split}: {len(files)} images under {root}, expected {spec['sizes'][split]}"
        )
    return files, labels
