"""Feature loading, views and identity guards shared by probe.py and extract_dino.py.

A probe reads ONE feature file (cv/budget/mlp) or one file per split (official) and a
VIEW of it. Views reproduce exactly what the validated prototypes fed their probes:

  flux  sec13:<i>   DiTF-normalized; base = t-index i (PCA-512-protected in the fold),
                    the other timesteps appended raw  (6ac arm A1, section 13)
        t:<i>       DiTF-normalized single timestep i  (6ac arm A2)
        concat      DiTF-normalized all-t concat  (Q2 flux_7t)
  dino  cls | mp | clsmp
  vae   full | pool4 | pool2 | pool1  (Q1 poolings of the 16x32x32 clean latent)

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

# cv/budget/mlp identity: the ordered image set every paired arm must consume. The
# RESISC45 5,000 are the section-13 inversion cache's paths, in cache order (6ac).
CV_IDENTITY = {
    "resisc45": "models/n5000_resisc45_inversion/resisc45_flux_f7718ddb+42/multistep_train_feats_inversion_g1.0_n50.npz",
}


def _eurosat_classes() -> list[str]:
    from data.eurosat_dataset import EUROSAT_CLASSES

    return list(EUROSAT_CLASSES)


# Official-protocol datasets -- the ONLY place a GEO-Bench task is registered for
# eval.probe / eval.extract_dino / eval.sweep. Adding a task = one entry here (plus its
# TaskSpec in eval/export_geobench.py and a run.py dataset wrapper for FLUX extraction).
#   sizes:   per-split counts COUNTED from the shipped partition (rule 16), never a paper
#   root:    exported RGB tree, <root>/<split>/<Class>/*.png
#   classes: zero-arg callable -> class names in LABEL order (the exporter's order)
OFFICIAL = {
    "m_eurosat": {
        "sizes": {"train": 16200, "val": 996, "test": 996},
        "root": "data/m_eurosat_rgb",
        "classes": _eurosat_classes,
    },
}


def official_spec(dataset: str) -> dict:
    if dataset not in OFFICIAL:
        raise SystemExit(
            f"{dataset}: not registered in eval/features.OFFICIAL (count the shipped partition "
            f"first, rule 16). Registered: {sorted(OFFICIAL)}"
        )
    return OFFICIAL[dataset]


def official_classes(dataset: str) -> list[str]:
    return list(official_spec(dataset)["classes"]())


def ditf(f, m):
    """Offline DiTF normalization. Source: _absorption_harness.ditf (verbatim)."""
    x = f.astype(np.float64).copy()
    x[:, :, DISCARD] = 0.0
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
        parts = {"cls": [d["cls"]], "mp": [d["mp"]], "clsmp": [d["cls"], d["mp"]]}
        if spec in parts:
            return "plain", np.ascontiguousarray(np.concatenate(parts[spec], axis=1)).astype(np.float32)
    elif kind == "vae":
        lat = d["lat"]
        N, C_, H, W = lat.shape
        pools = {
            "full": lambda: lat.reshape(N, -1),
            "pool4": lambda: lat.reshape(N, C_, 8, 4, 8, 4).mean(axis=(3, 5)).reshape(N, -1),
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


def _load_flux_one(path: str, pins: dict, split: str):
    """Source: m_eurosat_probe._load_one."""
    d = np.load(path)
    meta = cache_meta(path)
    if meta.get("split", "train") != split:
        raise SystemExit(f"{path}: meta split={meta.get('split')!r} != {split!r}")
    check_pins(meta, pins, path)
    return d, meta


def load_flux_split(pattern: str, pins: dict, split: str, n_expect: int, smoke: bool = False):
    """One split of a FLUX run.py cache, merging shards. Source: m_eurosat_probe.load_split
    (per-arm pins passed in via flux_pins). Returns feats in DATASET order."""
    hits = sorted(glob.glob(pattern.format(split=split)))
    if not hits:
        raise SystemExit(f"no caches for {pattern.format(split=split)}")
    parts = [_load_flux_one(h, pins, split) for h in hits]
    if len(parts) == 1 and parts[0][1].get("num_shards", 1) in (1, None):
        d, meta = parts[0]
        feats, labels, idx, paths = d["feats"], d["labels"], d["subset_indices"], d["paths"]
    else:
        parts.sort(key=lambda p: p[1]["shard_index"])
        n_shards = parts[0][1]["num_shards"]
        got = [p[1]["shard_index"] for p in parts]
        if got != list(range(n_shards)):
            raise SystemExit(f"{split}: have shards {got}, expected 0..{n_shards - 1}")
        idx = np.concatenate([p[0]["subset_indices"] for p in parts])
        if len(np.unique(idx)) != len(idx) or len(idx) != n_expect:
            raise SystemExit(f"{split}: merged shard indices are not disjoint / complete (n={len(idx)})")
        if not smoke and not np.array_equal(np.sort(idx), np.arange(n_expect)):
            raise SystemExit(f"{split}: merged shard indices are not a cover of arange({n_expect})")
        feats = np.concatenate([p[0]["feats"] for p in parts])
        labels = np.concatenate([p[0]["labels"] for p in parts])
        paths = np.concatenate([p[0]["paths"] for p in parts])
        meta = parts[0][1]
    if feats.shape[0] != n_expect:
        raise SystemExit(f"{split}: N={feats.shape[0]}, expected {n_expect}")
    order = np.argsort(idx)
    ts = [int(t) for t in parts[0][0]["timesteps"]]
    return feats[order].astype(np.float64), labels[order], ts, meta, [str(p) for p in paths[order]], hits


def list_official_split(dataset: str, split: str, root: str | None = None):
    """Image files + labels of one official split, in class-list order then sorted names.
    Source: dinov2_m_eurosat.list_split (size check against the OFFICIAL entry)."""
    spec = official_spec(dataset)
    root = root or spec["root"]
    files, labels = [], []
    for ci, cls in enumerate(official_classes(dataset)):
        for f in sorted(glob.glob(os.path.join(root, split, cls, "*.png"))):
            files.append(f)
            labels.append(ci)
    if len(files) != spec["sizes"][split]:
        raise SystemExit(
            f"{dataset} {split}: {len(files)} images under {root}, expected {spec['sizes'][split]}"
        )
    return files, np.array(labels, dtype=np.int64)
