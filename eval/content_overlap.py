"""Residual content overlap between Git-10M rows that CANNOT be located and GEO-Bench.

The geographic exclusion (eval/geo_dedup.py) says nothing about the 2,045,786 Git-10M rows
(1,706,416 unique images) whose sources publish no coordinates, nor about m-so2sat. This module
turns "how much residual overlap is there" into a measured number: every such image is embedded
with the same frozen DINOv3 backbone that produced the cached GEO-Bench features
(eval/extract_dino.py, models/dino_feats/<preset>_<task>_<split>.npz, CLS of the last block,
224 px bicubic, the preset's normalization), its nearest GEO-Bench sample over all six tasks and
all three splits is found by cosine similarity, and the count above a threshold is reported.

The threshold is NOT chosen by eye. It is a quantile of a NULL: Git-10M rows that were RETAINED
by the geographic pass (located, > buffer from every evaluation box, so known not to be the same
place) embedded through the identical path. Their max-similarity distribution is what "a
different place that may look alike" scores; tau = its (1 - fpr) quantile. Planted POSITIVES
(GEO-Bench test images pushed through the Git-10M path: identical, flipped, 80 % centre crop,
JPEG q60, half resolution) measure what the instrument catches at tau, and EXCLUDED rows (same
place as an evaluation sample by coordinates, usually another sensor and date) show what a true
same-location pair scores. Gates (executed, PASS/FAIL, rule 13): the identity positives must
reproduce their cached reference (cos >= 0.99 for >= 99 %) -- a preprocessing mismatch between
this path and the cache would otherwise silently read as "no overlap" -- and must be recalled at
tau in full.

    python -m eval.content_overlap embed --preset dinov3_vit7b16_sat --arrow-dir A \\
        --manifest D/dedup_v2_b1000/unassessable.csv --unique-names --part 0/8 --out OUT_p0of8.npz
    python -m eval.content_overlap embed ... --manifest D/dedup_v2_b1000/retained.csv --sample 20000 --seed 0 --out null.npz
    python -m eval.content_overlap embed ... --manifest D/dedup_v2_b1000/excluded.csv --sample 20000 --out excl.npz
    python -m eval.content_overlap controls --preset dinov3_vit7b16_sat --per-task 200 --out ctrl.npz
    python -m eval.content_overlap match --preset dinov3_vit7b16_sat --ref-dir models/dino_feats \\
        --null null.npz --controls ctrl.npz --excluded excl.npz --query OUT_p*.npz --fpr 1e-3 --out-dir REPORT
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import features as F  # noqa: E402

IMG = 224
VARIANTS = ("identity", "hflip", "crop80", "jpeg60", "half")
# Synthetic uniform images embedded with the controls. A uniform image (no-data fill: Git-10M
# holds all-white SkyScript / SSL4EO tiles, GEO-Bench m-pv4ger has all-white patches) has ONE CLS
# vector whatever its source, so such rows pair at cosine 1.0 across sets and saturated the null
# at fpr 1e-4 (2026-10-14: 21 of 200k null rows, 4 of 101k reference rows, all pure white).
# `match` drops every row within DEGENERATE_COS of any blank control from every set, and counts them.
BLANK_VARIANTS = {"blank_white": 255, "blank_black": 0, "blank_gray": 128}
DEGENERATE_COS = 0.999
DEFAULT_FPRS = (1e-2, 1e-3, 1e-4)


# ------------------------------------------------------------------------- rows to embed
def _git10m_mod():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from datasets import git10m

    return git10m


def rows_from_manifest(csv: str, unique_names: bool, drop_city_only: bool = False) -> pd.DataFrame:
    """shard, row, source, img_name of a dedup manifest (retained / excluded / unassessable.csv),
    sorted by (shard, row). unique_names keeps the first row of every img_name (Git-10M repeats
    public-dataset images on adjacent rows, one per caption, same bytes) BEFORE any sampling or
    partitioning, so parts never share an image. drop_city_only (excluded.csv of a city pass):
    drop rows whose every matched id is a `city:<name>` ROI -- those sit somewhere inside a
    ~50 km So2Sat city, not at the place of an evaluation sample, and would make the
    same-location reference read as "inside a city" (71 % of dedup_v3_b1000_cities/excluded.csv)."""
    cols = ["shard", "row", "source", "img_name"] + (["matched_ids"] if drop_city_only else [])
    df = pd.read_csv(csv, usecols=cols, dtype={"shard": np.int64, "row": np.int64})
    if drop_city_only:
        ids = df["matched_ids"].astype(str).str.split(";")
        city_only = ids.map(lambda L: all(t.startswith("city:") for t in L))
        df = df.loc[~city_only, cols[:4]].reset_index(drop=True)
        if len(df) == 0:
            raise SystemExit(f"{csv}: every row is excluded by a city ROI only; no same-location rows")
    df = df.sort_values(["shard", "row"], kind="stable").reset_index(drop=True)
    if unique_names:
        df = df[~df["img_name"].duplicated(keep="first")].reset_index(drop=True)
    return df


def select_rows(df: pd.DataFrame, sample: int | None, seed: int, part: str | None) -> pd.DataFrame:
    """Deterministic subset: a seeded random sample of `sample` rows, then contiguous part i of n."""
    if sample is not None:
        if sample > len(df):
            raise SystemExit(f"--sample {sample} > {len(df)} rows")
        idx = np.sort(np.random.default_rng(seed).choice(len(df), sample, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    if part:
        i, n = (int(v) for v in part.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--part {part}: want i/n with 0 <= i < n")
        edges = np.linspace(0, len(df), n + 1).astype(int)
        df = df.iloc[edges[i] : edges[i + 1]].reset_index(drop=True)
    return df


class _Rows:
    """(shard, row) -> decoded PIL image from the converted arrow shards (datasets.git10m
    layout). The shard's img_name must equal the manifest's: a misaddressed row is fatal."""

    def __init__(self, arrow_dir: str, df: pd.DataFrame):
        g = _git10m_mod()
        self.shard_name, self.batch_rows = g.shard_name, g.BATCH_ROWS
        self.arrow_dir = arrow_dir
        self.shard, self.row = df["shard"].to_numpy(np.int64), df["row"].to_numpy(np.int64)
        self.name = df["img_name"].to_numpy(str)
        self._readers: dict[int, ipc.RecordBatchFileReader] = {}
        for s in np.unique(self.shard):
            p = os.path.join(arrow_dir, self.shard_name(int(s)))
            if not os.path.exists(p):
                raise FileNotFoundError(p)

    def __len__(self):
        return len(self.shard)

    def raw(self, i: int) -> tuple[str, bytes]:
        s, r = int(self.shard[i]), int(self.row[i])
        rd = self._readers.get(s)
        if rd is None:
            rd = ipc.open_file(pa.memory_map(os.path.join(self.arrow_dir, self.shard_name(s))))
            self._readers[s] = rd
        b = rd.get_batch(r // self.batch_rows)
        j = r % self.batch_rows
        return b.column("img_name")[j].as_py(), b.column("image_bytes")[j].as_py()

    def __getitem__(self, i) -> Image.Image:
        name, raw = self.raw(i)
        if name != self.name[i]:
            raise SystemExit(
                f"shard {self.shard[i]} row {self.row[i]}: img_name {name!r} != manifest {self.name[i]!r}"
            )
        return Image.open(io.BytesIO(raw))


# ------------------------------------------------------------------------- embedding
def to_tensor(im: Image.Image, mean, std):
    import torch

    im = im.convert("RGB").resize((IMG, IMG), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def variant(im: Image.Image, kind: str) -> Image.Image:
    im = im.convert("RGB")
    if kind == "identity":
        return im
    if kind == "hflip":
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if kind == "crop80":
        w, h = im.size
        dw, dh = int(round(w * 0.1)), int(round(h * 0.1))
        return im.crop((dw, dh, w - dw, h - dh))
    if kind == "jpeg60":
        b = io.BytesIO()
        im.save(b, format="JPEG", quality=60)
        return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    if kind == "half":
        w, h = im.size
        return im.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BICUBIC)
    raise ValueError(kind)


def load_backbone(preset: str, device: str):
    """The eval/extract_dino model + its normalization; CLS of the last block (forward_features'
    x_norm_clstoken), which is what the cached GEO-Bench `cls` arrays hold."""
    from eval import extract_dino as E

    model, spec = E.load_model(preset, None, device)
    return model, spec


def embed_images(model, spec, images, device: str, batch: int, log_every: int = 50) -> np.ndarray:
    """images: iterable of PIL images (or callables returning one) -> CLS float32 [n, d]."""
    import torch

    mean, std = spec["norm"]
    out, buf = [], []

    def flush():
        x = torch.stack(buf).to(device=device, dtype=spec["dtype"])
        with torch.no_grad():
            cls = model.forward_features(x)["x_norm_clstoken"]
        out.append(cls.float().cpu().numpy())
        buf.clear()

    for k, im in enumerate(images):
        buf.append(to_tensor(im() if callable(im) else im, mean, std))
        if len(buf) == batch:
            flush()
            if (k + 1) % (batch * log_every) == 0:
                print(f"  {k + 1} embedded", flush=True)
    if buf:
        flush()
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def cmd_embed(a) -> None:
    import torch

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: refusing to overwrite")
    df = rows_from_manifest(a.manifest, a.unique_names, a.drop_city_only)
    n_all = len(df)
    df = select_rows(df, a.sample, a.seed, a.part)
    if a.max_rows:
        df = df.iloc[: a.max_rows].reset_index(drop=True)
    rows = _Rows(a.arrow_dir, df)
    print(
        f"{len(df)} rows to embed (of {n_all}; sample={a.sample} part={a.part} unique={a.unique_names})",
        flush=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, spec = load_backbone(a.preset, device)
    loader = torch.utils.data.DataLoader(rows, batch_size=None, num_workers=a.workers, collate_fn=lambda x: x)
    cls = embed_images(model, spec, loader, device, a.batch or spec["batch"] * 4)
    if cls.shape[0] != len(df):
        raise SystemExit(f"embedded {cls.shape[0]} != {len(df)} rows")
    np.savez(
        a.out + ".tmp.npz",
        shard=df["shard"].to_numpy(np.int16),
        row=df["row"].to_numpy(np.int32),
        source=df["source"].to_numpy(str),
        img_name=df["img_name"].to_numpy(str),
        cls=cls.astype(np.float16),
        preset=np.array(a.preset),
        origin=np.array(
            json.dumps(
                {
                    "manifest": os.path.abspath(a.manifest),
                    "manifest_sha256": _sha256(a.manifest),
                    "sample": a.sample,
                    "seed": a.seed,
                    "part": a.part,
                    "unique_names": a.unique_names,
                    "drop_city_only": a.drop_city_only,
                    "n_all": n_all,
                    "img": IMG,
                    "mean": spec["norm"][0],
                    "std": spec["norm"][1],
                }
            )
        ),
    )
    os.replace(a.out + ".tmp.npz", a.out)
    print(f"CONTENT_EMBED_OK {cls.shape} -> {a.out}")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_controls(a) -> None:
    """GEO-Bench test images (strided, every class present) through the Git-10M embedding path,
    in VARIANTS. The identity variant must reproduce the cached reference (gate in `match`)."""
    import torch

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: refusing to overwrite")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, spec = load_backbone(a.preset, device)
    tasks, paths, kinds, imgs = [], [], [], []
    for task in a.tasks:
        files, _ = F.list_official_split(task, a.split, root=a.root and os.path.join(a.root, f"{task}_rgb"))
        keep = np.linspace(0, len(files) - 1, min(a.per_task, len(files))).astype(int)
        for i in keep:
            for kind in VARIANTS:
                tasks.append(task)
                paths.append(files[i])
                kinds.append(kind)
                imgs.append((files[i], kind))
    for kind, fill in BLANK_VARIANTS.items():  # the degenerate-image detectors (see BLANK_VARIANTS)
        tasks.append("synthetic")
        paths.append(f"synthetic:{kind}")
        kinds.append(kind)
        imgs.append((None, kind))

    def _image(f, k):
        if f is None:
            v = BLANK_VARIANTS[k]
            return Image.new("RGB", (IMG, IMG), (v, v, v))
        return variant(Image.open(f), k)

    cls = embed_images(
        model,
        spec,
        (lambda f=f, k=k: _image(f, k) for f, k in imgs),
        device,
        a.batch or spec["batch"] * 4,
    )
    np.savez(
        a.out + ".tmp.npz",
        task=np.array(tasks),
        path=np.array(paths),
        variant=np.array(kinds),
        cls=cls.astype(np.float16),
        preset=np.array(a.preset),
        split=np.array(a.split),
    )
    os.replace(a.out + ".tmp.npz", a.out)
    print(
        f"CONTENT_CONTROLS_OK {cls.shape} ({len(a.tasks)} tasks x {a.per_task} x {len(VARIANTS)} "
        f"+ {len(BLANK_VARIANTS)} blank) -> {a.out}"
    )


# ------------------------------------------------------------------------- matching
def load_reference(ref_dir: str, preset: str, tasks: list[str], splits=("train", "val", "test")):
    """Cached GEO-Bench CLS features of `preset`, every task x split, L2-normalised, with
    identity (task, split, basename) and a sha1 over what was loaded."""
    feats, task_l, split_l, path_l, h = [], [], [], [], hashlib.sha1()
    for task in tasks:
        sizes = F.official_spec(task)["sizes"]
        for split in splits:
            p = os.path.join(ref_dir, f"{preset}_{task}_{split}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"{p} missing: extract it first (experiments/isaac/geobench/dino.sbatch)")
            z = np.load(p)
            if str(z["preset"]) != preset:
                raise SystemExit(f"{p}: preset {z['preset']} != {preset}")
            cls = z["cls"].astype(np.float32)
            if cls.shape[0] != sizes[split]:
                raise SystemExit(f"{p}: {cls.shape[0]} rows, OFFICIAL says {sizes[split]}")
            feats.append(cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12))
            task_l += [task] * len(cls)
            split_l += [split] * len(cls)
            names = [os.path.basename(str(q)) for q in z["paths"]]
            if len(set(names)) != len(names):
                raise SystemExit(f"{p}: basenames repeat; identity by basename is ambiguous")
            path_l += names
            h.update(f"{task}/{split}/{cls.shape}".encode())
            h.update(cls[:64].tobytes())
    ref = np.concatenate(feats)
    return ref, np.array(task_l), np.array(split_l), np.array(path_l), h.hexdigest()[:16]


def max_cosine(q: np.ndarray, ref: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Per query row: max cosine over ref rows and its argmax (ref is L2-normalised)."""
    q = q.astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    best, arg = np.empty(len(q), np.float32), np.empty(len(q), np.int64)
    for s in range(0, len(q), chunk):
        sim = q[s : s + chunk] @ ref.T
        arg[s : s + chunk] = sim.argmax(axis=1)
        best[s : s + chunk] = sim[np.arange(len(sim)), arg[s : s + chunk]]
    return best, arg


def blank_vectors(ctrl) -> np.ndarray:
    """The L2-normalised CLS of the synthetic uniform images in a controls file; refuses a controls
    file that predates them (no silent 'no degenerate rows')."""
    kinds = np.asarray(ctrl["variant"]).astype(str)
    have = [k for k in BLANK_VARIANTS if (kinds == k).any()]
    if set(have) != set(BLANK_VARIANTS):
        raise SystemExit(
            f"controls hold blank variants {have}, need {sorted(BLANK_VARIANTS)}: rerun `controls` "
            "(the degenerate-image filter needs them)"
        )
    v = np.asarray(ctrl["cls"])[np.isin(kinds, list(BLANK_VARIANTS))].astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def degenerate_mask(cls: np.ndarray, blank: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """True for rows within DEGENERATE_COS of any blank control: a uniform (no-data) image."""
    out = np.zeros(len(cls), dtype=bool)
    for s in range(0, len(cls), chunk):
        q = np.asarray(cls[s : s + chunk]).astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        out[s : s + chunk] = (q @ blank.T).max(axis=1) >= DEGENERATE_COS
    return out


def thresholds(null_max: np.ndarray, fprs) -> dict[str, float]:
    """tau per false-positive rate: the (1 - fpr) quantile of the null's max-similarity."""
    if len(null_max) < 1000:
        raise SystemExit(f"null has {len(null_max)} rows; need >= 1000 for a 1e-3 quantile to mean anything")
    return {f"{fpr:g}": float(np.quantile(null_max, 1 - fpr)) for fpr in fprs}


def identity_gate(ctrl, ref, ref_task, ref_split, ref_path, split: str) -> dict:
    """Each identity control vs ITS OWN cached reference row (same task, split, basename):
    cos >= 0.99 for >= 99 % of them, else the embedding path differs from the cache's."""
    key = {(t, s, p): i for i, (t, s, p) in enumerate(zip(ref_task, ref_split, ref_path))}
    ident = ctrl["variant"] == "identity"
    q = ctrl["cls"][ident].astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    cos, missing = [], 0
    for task, path, v in zip(ctrl["task"][ident], ctrl["path"][ident], q):
        j = key.get((task, split, os.path.basename(str(path))))
        if j is None:
            missing += 1
            continue
        cos.append(float(v @ ref[j]))
    cos = np.array(cos)
    frac = float((cos >= 0.99).mean()) if len(cos) else 0.0
    ok = missing == 0 and len(cos) > 0 and frac >= 0.99
    return {
        "pass": bool(ok),
        "n": int(len(cos)),
        "missing_reference": int(missing),
        "frac_cos_ge_0.99": frac,
        "min_cos": float(cos.min()) if len(cos) else None,
        "median_cos": float(np.median(cos)) if len(cos) else None,
    }


def cmd_match(a) -> None:
    from eval.features import OFFICIAL

    os.makedirs(a.out_dir, exist_ok=True)
    tasks = list(OFFICIAL)
    ref, ref_task, ref_split, ref_path, ref_sha = load_reference(a.ref_dir, a.preset, tasks)
    print(f"reference: {ref.shape} rows over {len(tasks)} tasks x 3 splits (sha {ref_sha})", flush=True)

    def load_q(path, role: str):
        z = np.load(path)
        if str(z["preset"]) != a.preset:
            raise SystemExit(f"{path}: preset {z['preset']} != {a.preset}")
        o = json.loads(str(z["origin"]))
        want = {"null": "retained.csv", "excluded": "excluded.csv", "query": "unassessable.csv"}[role]
        if os.path.basename(o["manifest"]) != want:
            raise SystemExit(f"{path}: {role} must be embedded from {want}, not {o['manifest']}")
        if role == "excluded" and not o.get("drop_city_only"):
            raise SystemExit(f"{path}: the same-location reference needs embed --drop-city-only")
        o["manifest_dir"] = os.path.dirname(o["manifest"])
        return z, o

    ctrl = np.load(a.controls)
    if str(ctrl["preset"]) != a.preset:
        raise SystemExit(f"{a.controls}: preset {ctrl['preset']} != {a.preset}")
    blank = blank_vectors(ctrl)
    degenerate: dict = {"cos": DEGENERATE_COS, "blank_variants": sorted(BLANK_VARIANTS)}
    # Uniform (no-data) images out of EVERY set before any statistic: they pair at cosine 1.0
    # across sources and would set tau and count as hits (BLANK_VARIANTS).
    keep = ~degenerate_mask(ref, blank)
    degenerate["reference"] = int((~keep).sum())
    degenerate["reference_rows"] = [
        f"{t}/{sp}/{p}" for t, sp, p in zip(ref_task[~keep], ref_split[~keep], ref_path[~keep])
    ]
    ref, ref_task, ref_split, ref_path = ref[keep], ref_task[keep], ref_split[keep], ref_path[keep]
    ctrl = {k: ctrl[k] for k in ctrl.files}
    c_keep = ~degenerate_mask(ctrl["cls"], blank) | np.isin(ctrl["variant"].astype(str), list(BLANK_VARIANTS))
    degenerate["controls"] = int((~c_keep).sum())
    for k in ("task", "path", "variant", "cls"):
        ctrl[k] = ctrl[k][c_keep]
    null, null_o = load_q(a.null, "null")
    n_keep = ~degenerate_mask(null["cls"], blank)
    degenerate["null"] = int((~n_keep).sum())
    degenerate["null_per_source"] = {
        s: int((~n_keep & (null["source"] == s)).sum()) for s in sorted(set(null["source"]))
    }
    null_max, _ = max_cosine(null["cls"][n_keep], ref)
    taus = thresholds(null_max, a.fprs)
    print(
        f"degenerate rows dropped: {json.dumps({k: v for k, v in degenerate.items() if k != 'reference_rows'})}",
        flush=True,
    )
    gate_id = identity_gate(ctrl, ref, ref_task, ref_split, ref_path, str(ctrl["split"]))
    ctrl_max, _ = max_cosine(ctrl["cls"], ref)
    recall = {
        kind: {f: float((ctrl_max[ctrl["variant"] == kind] >= t).mean()) for f, t in taus.items()}
        for kind in VARIANTS
    }
    gate_recall = {
        "pass": all(recall["identity"][f] == 1.0 for f in taus),
        "identity_recall": recall["identity"],
    }
    gate_null = {"pass": all(t < 0.999 for t in taus.values()), "taus": taus}
    gates = {
        "identity_reproduces_cache": gate_id,
        "identity_recalled_at_tau": gate_recall,
        "null_not_saturated": gate_null,
    }
    for name, g in gates.items():
        print(
            f"{'PASS' if g['pass'] else 'FAIL'} {name}: {json.dumps({k: v for k, v in g.items() if k != 'pass'})}",
            flush=True,
        )
    if not all(g["pass"] for g in gates.values()):
        with open(os.path.join(a.out_dir, "report.json"), "w") as f:
            json.dump({"gates": gates, "degenerate": degenerate, "status": "FAIL"}, f, indent=1)
        raise SystemExit("instrument gate FAILED: diagnose the embedding path before reading any count")

    excl = None
    if a.excluded:
        e, excl_o = load_q(a.excluded, "excluded")
        if excl_o["manifest_dir"] != null_o["manifest_dir"]:
            raise SystemExit(f"excluded from {excl_o['manifest_dir']} but null from {null_o['manifest_dir']}")
        e_keep = ~degenerate_mask(e["cls"], blank)
        degenerate["excluded"] = int((~e_keep).sum())
        excl_max, _ = max_cosine(e["cls"][e_keep], ref)
        excl = {f: float((excl_max >= t).mean()) for f, t in taus.items()}
        excl["n"] = int(len(excl_max))
        excl["median"] = float(np.median(excl_max))

    files = sorted(sum((glob.glob(p) for p in a.query), []))
    if not files:
        raise SystemExit(f"no query files match {a.query}")
    q_src, q_task, q_split, q_name, q_max, q_shard, q_row, q_arg = [], [], [], [], [], [], [], []
    parts, q_origin = {}, None
    degenerate["queries"], q_deg_src = 0, {}
    for p in files:
        z, o = load_q(p, "query")
        # every part from ONE manifest (same pass, same sha, same unique-names flag), and the parts
        # must tile it: a missing part or a part of another family would silently shrink the count
        key = {k: o[k] for k in ("manifest_sha256", "n_all", "unique_names", "sample", "seed")}
        if q_origin is None:
            q_origin = key
        elif key != q_origin:
            raise SystemExit(f"{p}: origin {key} differs from the first part's {q_origin}")
        if o["manifest_dir"] != null_o["manifest_dir"]:
            raise SystemExit(f"{p}: queries from {o['manifest_dir']} but null from {null_o['manifest_dir']}")
        i, n = (int(v) for v in (o["part"] or "0/1").split("/"))
        if i in parts:
            raise SystemExit(f"{p}: part {i}/{n} already loaded from {parts[i]}")
        parts[i] = (p, n)
        k = ~degenerate_mask(z["cls"], blank)
        degenerate["queries"] += int((~k).sum())
        for src in set(z["source"][~k].tolist()):
            q_deg_src[src] = q_deg_src.get(src, 0) + int(((~k) & (z["source"] == src)).sum())
        m, arg = max_cosine(z["cls"][k], ref)
        q_max.append(m)
        q_arg.append(arg)
        q_src.append(z["source"][k])
        q_name.append(z["img_name"][k])
        q_shard.append(z["shard"][k].astype(np.int64))
        q_row.append(z["row"][k].astype(np.int64))
        print(
            f"  {p}: {len(m)} queries ({int((~k).sum())} degenerate dropped), max {m.max():.3f}", flush=True
        )
    degenerate["queries_per_source"] = q_deg_src
    n_parts = {n for _, n in parts.values()}
    if len(n_parts) != 1 or set(parts) != set(range(next(iter(n_parts)))):
        raise SystemExit(f"query parts do not tile the manifest: have {sorted(parts)} of {sorted(n_parts)}")
    q_max, q_arg = np.concatenate(q_max), np.concatenate(q_arg)
    q_src, q_name = np.concatenate(q_src), np.concatenate(q_name)
    if q_origin["sample"] is None and len(q_max) + degenerate["queries"] != q_origin["n_all"]:
        raise SystemExit(
            f"{len(q_max)} queries + {degenerate['queries']} degenerate loaded but the manifest holds {q_origin['n_all']} rows"
        )
    q_shard, q_row = np.concatenate(q_shard), np.concatenate(q_row)
    if len(np.unique(q_name)) != len(q_name):
        raise SystemExit("query files repeat an img_name: parts overlap")
    q_task, q_split = ref_task[q_arg], ref_split[q_arg]

    per = {}
    for f, t in taus.items():
        hit = q_max >= t
        # at tau = the (1 - fpr) null quantile, fpr x n queries exceed it by chance alone; only
        # the excess over that is evidence of overlap. The excess is NOT a bound: the null's
        # source mix is not the queries', so its false-positive rate need not be theirs (a
        # negative excess means the null over-predicts, and the excess under-states overlap).
        # The bound on overlap detectable at this tau is the raw hit count (PROTOCOL.md section 8)
        expected_false = float(f) * len(q_max)
        per[f] = {
            "tau": t,
            "hits": int(hit.sum()),
            "fraction": float(hit.mean()),
            "expected_false_hits": expected_false,
            "excess_hits": int(hit.sum()) - expected_false,
            "null_rows_above_tau": int((null_max >= t).sum()),
            "per_source": {s: int((hit & (q_src == s)).sum()) for s in sorted(set(q_src))},
            "per_nearest_task": {s: int((hit & (q_task == s)).sum()) for s in tasks},
            "per_nearest_split": {s: int((hit & (q_split == s)).sum()) for s in ("train", "val", "test")},
        }
    rep = {
        "status": "OK",
        "preset": a.preset,
        "reference": {"rows": int(len(ref)), "sha": ref_sha, "dir": os.path.abspath(a.ref_dir)},
        "manifest_dir": null_o["manifest_dir"],
        "null": {
            "file": os.path.abspath(a.null),
            "origin": null_o,
            "n": int(len(null_max)),
            "median": float(np.median(null_max)),
            "quantiles": {q: float(np.quantile(null_max, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)},
        },
        "taus": taus,
        "controls": {
            "file": os.path.abspath(a.controls),
            "per_variant_recall": recall,
            "per_variant_median": {k: float(np.median(ctrl_max[ctrl["variant"] == k])) for k in VARIANTS},
        },
        "excluded_same_location": excl,
        "queries": {
            "files": [os.path.abspath(p) for p in files],
            "origin": q_origin,
            "n": int(len(q_max)),
            "median": float(np.median(q_max)),
            "per_source_n": {s: int((q_src == s).sum()) for s in sorted(set(q_src))},
        },
        "hits": per,
        "gates": gates,
        "degenerate": degenerate,
    }
    np.savez(
        os.path.join(a.out_dir, "query_max_sim.npz"),
        shard=q_shard.astype(np.int16),
        row=q_row.astype(np.int32),
        img_name=q_name,
        source=q_src,
        max_sim=q_max,
        nearest_task=q_task,
        nearest_split=q_split,
        nearest_path=ref_path[q_arg],
        null_max_sim=null_max,
        ref_sha=np.array(ref_sha),
    )
    loose = taus[max(taus, key=float)]  # the largest fpr = the loosest tau
    hit = q_max >= loose
    pd.DataFrame(
        {
            "shard": q_shard[hit],
            "row": q_row[hit],
            "img_name": q_name[hit],
            "source": q_src[hit],
            "max_sim": q_max[hit],
            "nearest_task": q_task[hit],
            "nearest_split": q_split[hit],
            "nearest_sample": ref_path[q_arg][hit],
        }
    ).sort_values("max_sim", ascending=False).to_csv(os.path.join(a.out_dir, "hits.csv"), index=False)
    with open(os.path.join(a.out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out_dir, "report.md"), "w") as f:
        f.write(report_markdown(rep))
    print(report_markdown(rep))
    print(f"CONTENT_MATCH_OK {len(q_max)} queries -> {a.out_dir}")


def report_markdown(rep: dict) -> str:
    L = [
        f"# Residual content overlap ({rep['preset']}, CLS cosine vs {rep['reference']['rows']} GEO-Bench samples)",
        "",
    ]
    L.append(
        f"Null (retained Git-10M rows, n={rep['null']['n']}): median max-sim {rep['null']['median']:.3f}; "
        + ", ".join(f"q{q}={v:.3f}" for q, v in rep["null"]["quantiles"].items())
    )
    L += [
        "",
        "| fpr | tau | null rows >= tau | hits | expected false | excess | fraction | "
        + " | ".join(f"recall {k}" for k in VARIANTS)
        + " | excluded (same place) |",
        "|---" * (8 + len(VARIANTS)) + "|",
    ]
    for f, t in rep["taus"].items():
        h = rep["hits"][f]
        rec = " | ".join(f"{rep['controls']['per_variant_recall'][k][f]:.3f}" for k in VARIANTS)
        ex = f"{rep['excluded_same_location'][f]:.3f}" if rep.get("excluded_same_location") else "--"
        L.append(
            f"| {f} | {t:.4f} | {h['null_rows_above_tau']} | {h['hits']} | {h['expected_false_hits']:.0f} | {h['excess_hits']:+.0f} | {h['fraction']:.5f} | {rec} | {ex} |"
        )
    L += [
        "",
        "`hits` counts queries at or above tau; `expected false` = fpr x queries (what the null predicts with zero overlap); "
        "`excess` = hits - expected false is evidence of overlap only when positive, and it is not a bound: the null's sources "
        "(Google tiles, SSL4EO-S12, SkyScript) are not the queries' (Million-AID, DIOR, AID, GeoPile, RSI-CB, VGoogle), so a "
        "negative excess says the null over-predicts the queries' false positives. The upper bound on overlap detectable at "
        "a given tau is the raw `hits` count (divide by the planted-positive recall for overlap of that kind). "
        "`null rows >= tau` is how many null rows set the quantile: a tau resting on < 20 rows is noisy.",
    ]
    f1 = list(rep["taus"])[len(rep["taus"]) // 2]
    h = rep["hits"][f1]
    L += [
        "",
        f"Hits at fpr {f1} by source: " + ", ".join(f"{k} {v}" for k, v in h["per_source"].items()),
        f"Hits at fpr {f1} by nearest task: "
        + ", ".join(f"{k} {v}" for k, v in h["per_nearest_task"].items()),
        f"Hits at fpr {f1} by nearest split: "
        + ", ".join(f"{k} {v}" for k, v in h["per_nearest_split"].items()),
        "",
        "Gates: " + ", ".join(f"{k} {'PASS' if g['pass'] else 'FAIL'}" for k, g in rep["gates"].items()),
        "",
        f"Degenerate (uniform, no-data) images dropped from every set before any statistic (cosine >= {rep['degenerate']['cos']} "
        f"to a synthetic blank control): reference {rep['degenerate']['reference']}, null {rep['degenerate']['null']}, "
        f"excluded {rep['degenerate'].get('excluded', 0)}, queries {rep['degenerate']['queries']} "
        f"({', '.join(f'{k} {v}' for k, v in rep['degenerate']['queries_per_source'].items()) or 'none'}), "
        f"controls {rep['degenerate']['controls']}.",
        "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="CLS embeddings of Git-10M rows (index bucket or manifest csv)")
    e.add_argument("--preset", required=True)
    e.add_argument("--arrow-dir", required=True)
    e.add_argument(
        "--manifest", required=True, help="dedup manifest csv: unassessable.csv | retained.csv | excluded.csv"
    )
    e.add_argument("--sample", type=int, default=None, help="seeded random subset of the rows")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--part", default=None, help="i/n: contiguous part i of n (after --sample)")
    e.add_argument("--unique-names", action="store_true", help="one row per img_name (first occurrence)")
    e.add_argument(
        "--drop-city-only",
        action="store_true",
        help="excluded.csv of a city pass: drop rows matched by city ROIs only (same-location reference)",
    )
    e.add_argument("--batch", type=int, default=None)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--max-rows", type=int, default=None, help="SMOKE: cap after selection")
    e.add_argument("--out", required=True)
    c = sub.add_parser("controls", help="GEO-Bench test images through the same path, in VARIANTS")
    c.add_argument("--preset", required=True)
    c.add_argument("--tasks", nargs="+", default=None)
    c.add_argument("--split", default="test")
    c.add_argument("--per-task", type=int, default=200)
    c.add_argument("--root", default=None, help="override the data root (<root>/<task>_rgb)")
    c.add_argument("--batch", type=int, default=None)
    c.add_argument("--out", required=True)
    m = sub.add_parser(
        "match", help="nearest GEO-Bench sample per query, null-calibrated thresholds, gates, report"
    )
    m.add_argument("--preset", required=True)
    m.add_argument("--ref-dir", required=True, help="models/dino_feats")
    m.add_argument("--null", required=True, help="embed output of RETAINED rows")
    m.add_argument("--controls", required=True)
    m.add_argument(
        "--excluded", default=None, help="embed output of EXCLUDED rows (same place by coordinates)"
    )
    m.add_argument("--query", nargs="+", required=True, help="embed outputs (globs) of the rows to assess")
    m.add_argument("--fprs", type=float, nargs="+", default=list(DEFAULT_FPRS))
    m.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    if a.cmd == "embed":
        cmd_embed(a)
    elif a.cmd == "controls":
        if a.tasks is None:
            a.tasks = list(F.OFFICIAL)
        cmd_controls(a)
    else:
        cmd_match(a)


if __name__ == "__main__":
    main()
