"""Linear per-token segmentation probe on cached tokens: model x dataset x label fraction x seed.

    python -m eval.seg_probe run    --spec eval/seg_specs/phase1.json [--models ...] [--datasets ...]
    python -m eval.seg_probe report --spec eval/seg_specs/phase1.json [--out-dir results/seg]

The segmentation analogue of eval.matrix (frozen features, linear head, val-only selection,
image bootstrap), on the token caches of tasks/seg_tokens.py (FLUX, pre-normalization, DiTF
applied here per token with the saved adaLN mods) and eval.extract_dino --tokens (DINOv3 patch
tokens, already normed). One cell = (model, dataset, fraction, seed):
  * head: per-token linear classifier on channel-standardized tokens (mean / std fitted on the
    probe-training tokens of the cell only), logits bilinearly upsampled from the token grid to
    the mask resolution, cross-entropy with the ignore index, Adam, a FIXED number of steps and
    batch of images for every fraction (matched optimization budget, rule 8);
  * candidates: every (timestep, block) file of the model x the learning-rate grid, selected on
    the FULL official val split by mIoU; one test evaluation of the selected candidate (rule 1):
    the test confusion of every candidate is computed in the same pass (one load per token file)
    and kept SEALED -- only the selected one is opened, exactly eval.sweep's rule;
  * fractions are nested per (dataset, seed): a seeded permutation of the train images, the
    first ceil(f * N) at fraction f (images are the labelled unit); seeds 0, 1, 2; the 100 % cell
    is seed-free and computed once;
  * metric: mIoU over the registered classes (ignore pixels excluded; a class absent from both
    prediction and reference in a set is skipped), from the summed confusion matrix; the
    per-image confusion matrices of the test split are stored (rule 12) and the 100 % cell's
    image bootstrap CI is computed from them.
Cells are cached under results/seg/cells/<model>/<dataset>/f<frac>_s<seed>.npz with the
selection, the val table, the sha1 of every token file read (by role), the meta of every split
and the protocol config; a cell whose inputs or config changed is refused unless --redo, and
the spec's pins are re-checked on the stored metas whenever a cell is tabulated.

Spec (JSON): {"fractions", "seeds", "lr_grid", "steps", "batch_images", "datasets",
  "models": [{"name", "kind": "flux"|"dino", "tokens": dir pattern with {dataset}, "tag": cache
              tag (flux) | "img" (dino), "expect": {pins checked on every meta}, "timesteps": [...]
              (flux), "blocks": [...] (flux), "teacher": name | null, "per_dataset": {...}}]}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval import features as F  # noqa: E402

PROTOCOL_VERSION = "seg-2026-10-18"
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_LR_GRID = (1e-3, 1e-2, 1e-1)
# Operating point of the fits, stamped in every cell (review 2026-10-27, CLAUDE.md rule 9: a default is
# not a pin). Train tokens are standardised then held in `train_dtype` on `device`; val / test are
# scored in fp32 on the host. Cells fitted at another operating point are refused by `collect`, never
# mixed into one table. Cells written before the stamp existed (all 24 probe.sbatch submissions
# 2026-10-17 .. 2026-10-27, none with --train-dtype / --device, sacct SubmitLine) ran at this one.
SEG_COMPUTE_LEGACY = {"train_dtype": "float16", "device": "cuda"}
DEFAULT_STEPS = 2000
DEFAULT_BATCH_IMAGES = 16
OUT_DIR = "results/seg"
BOOTSTRAP_N = 2000


# ----------------------------------------------------------------------------- protocol pieces
def image_subsets(n: int, fractions, seed: int) -> dict[float, np.ndarray]:
    """Nested image subsets: one seeded permutation, the first ceil(f * n) images at fraction f."""
    perm = np.random.default_rng(seed).permutation(n)
    out = {}
    for f in fractions:
        f = float(f)
        if not 0 < f <= 1:
            raise SystemExit(f"fraction {f} not in (0, 1]")
        out[f] = np.sort(perm[: math.ceil(f * n)]) if f < 1 else np.arange(n)
    return out


def confusion_per_image(
    pred: np.ndarray, ref: np.ndarray, n_cls: int, ignore: int = F.SEG_IGNORE
) -> np.ndarray:
    """(B, C, C) confusion matrices, rows = reference class, cols = predicted class."""
    b = pred.shape[0]
    out = np.zeros((b, n_cls, n_cls), np.int64)
    for i in range(b):
        keep = ref[i] != ignore
        r, p = ref[i][keep].astype(np.int64), pred[i][keep].astype(np.int64)
        if (r >= n_cls).any() or (p >= n_cls).any():
            raise SystemExit(
                f"class index out of range: ref max {r.max() if r.size else None}, pred max {p.max() if p.size else None}, n_cls {n_cls}"
            )
        out[i] = np.bincount(r * n_cls + p, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    return out


def miou(conf: np.ndarray) -> float:
    """mIoU of one summed (C, C) confusion; classes with an empty union are skipped."""
    conf = conf.astype(np.float64)
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    ok = union > 0
    if not ok.any():
        return float("nan")
    return float((inter[ok] / union[ok]).mean())


def bootstrap_miou(conf_img: np.ndarray, n: int = BOOTSTRAP_N, seed: int = 0) -> tuple[float, float]:
    """95 % image-bootstrap interval of mIoU from per-image (N, C, C) confusions (rule 4)."""
    N = conf_img.shape[0]
    flat = conf_img.reshape(N, -1).astype(np.float64)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
        vals.append(miou((w @ flat).reshape(conf_img.shape[1:])))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# ----------------------------------------------------------------------------- token caches
def _resolve(model: dict, dataset: str) -> dict:
    m = {k: v for k, v in model.items() if k != "per_dataset"}
    m.update((model.get("per_dataset") or {}).get(dataset, {}))
    if "tokens" not in m:
        raise SystemExit(f"model {model['name']}: no tokens pattern for {dataset}")
    m["tokens"] = m["tokens"].replace("{dataset}", dataset)
    return m


def check_meta(meta: dict, expect: dict, src: str) -> None:
    if not meta:
        raise SystemExit(f"{src}: no meta json -- refusing")
    for k, want in expect.items():
        if meta.get(k) != want:
            raise SystemExit(f"{src}: meta {k}={meta.get(k)!r}, expected {want!r}")
    if meta.get("max_images"):
        raise SystemExit(f"{src}: a --max-images smoke cache, never quotable")


def meta_name(m: dict, dataset: str, split: str) -> str:
    """Basename of one split's meta json (the extractor's naming: tasks/seg_tokens.py,
    eval/extract_dino.py --tokens). The ONE place the cache file names are spelled."""
    if m["kind"] == "flux":
        return f"seg_tokens_{split}_{m['tag']}_meta.json"
    if m["kind"] == "dino":
        return f"{m['preset']}_{dataset}_{split}_tokens{int(m['img'])}_meta.json"
    raise SystemExit(f"model {m['name']}: kind {m['kind']!r} not supported (flux, dino)")


def find_run_meta(tokens_dir: str, name: str) -> str:
    """The FLUX meta json of one split. run.py writes every extraction into its own run dir
    (save_dir/<run_name>, one per split since the seed differs), so the meta sits one level
    below the spec's tokens dir; exactly one match is required (a second run dir with the same
    tag is a stale or duplicate extraction and must be moved aside, never picked silently)."""
    hits = sorted(glob.glob(os.path.join(glob.escape(tokens_dir), "*", name)))
    if len(hits) != 1:
        raise SystemExit(
            f"{tokens_dir}/*/{name}: {len(hits)} matches {hits[:4]} -- refusing (need exactly one run dir per split)"
        )
    return hits[0]


class TokenSource:
    """One model's token files for one dataset: {candidate name: {split: path}} + meta per split."""

    def __init__(self, model: dict, dataset: str):
        self.m = _resolve(model, dataset)
        self.kind = self.m["kind"]
        self.dataset = dataset
        self.sizes = F.official_seg_spec(dataset)["sizes"]
        self.meta: dict[str, dict] = {}
        self.mods: dict[str, np.ndarray] = {}
        self.files: dict[str, dict[str, str]] = {}
        self.dirs: dict[str, str] = {}  # split -> the dir holding its files
        self.meta_npz: dict[str, str] = {}  # flux: split -> the meta npz (mods, paths)
        self._npz_paths: dict[str, list[str]] = {}
        if self.kind == "flux":
            for split in self.sizes:
                mp = find_run_meta(self.m["tokens"], meta_name(self.m, dataset, split))
                with open(mp) as fh:
                    meta = json.load(fh)
                check_meta(meta, {"dataset": dataset, "split": split, **dict(self.m.get("expect") or {})}, mp)
                self.meta[split] = meta
                self.dirs[split] = os.path.dirname(mp)
                self.meta_npz[split] = mp[: -len(".json")] + ".npz"
                with np.load(self.meta_npz[split]) as npz:
                    ts_npz, ks_npz = npz["timesteps"].tolist(), npz["blocks"].tolist()
                    # mod_for indexes mods by the json's t / k lists: they must be the npz's own
                    if ts_npz != list(meta["t"]) or ks_npz != list(meta["k"]):
                        raise SystemExit(
                            f"{mp}: t={meta['t']} k={meta['k']} differ from its npz's timesteps={ts_npz} blocks={ks_npz}"
                        )
                    self.mods[split] = npz["mods"]  # K, B, 3, C
                    self._npz_paths[split] = [str(q) for q in npz["paths"]]
            ts, ks = [int(t) for t in self.m["timesteps"]], [int(k) for k in self.m["blocks"]]
            for split, meta in self.meta.items():
                for t in ts:
                    if t not in meta["t"]:
                        raise SystemExit(f"{split}: timestep {t} not in the cache's {meta['t']}")
                for k in ks:
                    if k not in meta["k"]:
                        raise SystemExit(f"{split}: block {k} not in the cache's {meta['k']}")
            for t in ts:
                for k in ks:
                    self.files[f"t{t}_k{k}"] = {
                        s: os.path.join(self.dirs[s], self.meta[s]["files"][f"t{t}_k{k}"]) for s in self.sizes
                    }
        elif self.kind == "dino":
            img = int(self.m["img"])
            for split in self.sizes:
                mp = os.path.join(self.m["tokens"], meta_name(self.m, dataset, split))
                if os.path.exists(mp):
                    with open(mp) as fh:
                        meta = json.load(fh)
                else:
                    meta = {}
                check_meta(
                    meta,
                    {"dataset": dataset, "split": split, "img": img, **dict(self.m.get("expect") or {})},
                    mp,
                )
                self.meta[split] = meta
                self.dirs[split] = self.m["tokens"]
            self.files["patch"] = {s: os.path.join(self.dirs[s], self.meta[s]["file"]) for s in self.sizes}
        else:
            raise SystemExit(f"model {self.m['name']}: kind {self.kind!r} not supported (flux, dino)")
        for name, per_split in self.files.items():
            for s, p in per_split.items():
                if not os.path.exists(p):
                    raise SystemExit(f"{name} {s}: token file missing: {p}")
        # identity across splits: same weights / checkpoint / grid
        keys = (
            "weights",
            "lora_checkpoint_sha1",
            "img_size",
            "grid_hw",
            "ensemble_size",
            "guidance_scale",
            "img",
            "preset",
        )
        for key in keys:
            vals = {json.dumps(m.get(key)) for m in self.meta.values()}
            if len(vals) != 1:
                raise SystemExit(f"{self.m['name']} {dataset}: splits disagree on {key}: {vals}")
        if self.kind == "flux":
            seeds = [m.get("eps_seed") for m in self.meta.values()]
            if len(set(seeds)) != len(seeds):
                raise SystemExit(
                    f"{self.m['name']} {dataset}: eps_seed shared between splits {seeds} (audit F1)"
                )

    def grid(self) -> tuple[int, int]:
        g = self.meta["train"]["grid_hw"]
        return int(g[0]), int(g[1])

    def _paths(self, split: str) -> list[str]:
        """The cache's own ordered image list (flux: the meta npz; dino: the meta json)."""
        if self.kind == "flux":
            return list(self._npz_paths[split])
        return list(self.meta[split]["paths"])

    def discard_for(self, split: str) -> list[int] | None:
        """DiTF discard channels of one split's cache: FLUX's constant, another backbone's stamped
        list (eval.features.meta_discard_channels refuses a cache without one); None for dino."""
        if self.kind != "flux":
            return None
        return F.meta_discard_channels(self.meta[split], self.meta_npz[split])

    def mod_for(self, cand: str, split: str) -> np.ndarray | None:
        if self.kind != "flux":
            return None
        t, k = (int(x[1:]) for x in cand.split("_"))
        meta = self.meta[split]
        return self.mods[split][meta["t"].index(t), meta["k"].index(k)]  # (3, C)

    def input_roles(self) -> dict[str, str]:
        """{role: absolute path} of every file a cell's numbers depend on: "<candidate>/<split>"
        for the token memmaps and, for FLUX, "meta/<split>" for the meta npz (the adaLN mods every
        DiTF uses). Roles, not file names, are what the stored numbers are bound to."""
        roles = {
            f"{cand}/{s}": os.path.abspath(p) for cand, per in self.files.items() for s, p in per.items()
        }
        for s, q in self.meta_npz.items():
            roles[f"meta/{s}"] = os.path.abspath(q)
        return roles

    def input_files(self) -> list[str]:
        return sorted(set(self.input_roles().values()))

    def input_sha1s(self) -> dict[str, dict]:
        """{role: {"path", "sha1"}}, each file hashed once."""
        sha = {p: file_sha1(p) for p in self.input_files()}
        return {role: {"path": p, "sha1": sha[p]} for role, p in self.input_roles().items()}


# ----------------------------------------------------------------------------- the head
def ditf_tokens(x, mod, discard=F.DISCARD):
    """DiTF per token (eval.features.ditf on (N, K, C) with K = 1): massive channels zeroed,
    LayerNorm over C, the block's adaLN scale/shift, L2. x (n, C) float32 torch, mod (3, C)."""
    import torch

    x = x.clone()
    x[:, discard] = 0.0
    mu = x.mean(-1, keepdim=True)
    v = x.var(-1, keepdim=True, unbiased=False)
    x = (x - mu) / torch.sqrt(v + 1e-6)
    shift, scale = mod[0], mod[1]
    x = (1 + scale)[None] * x + shift[None]
    return x / x.norm(dim=-1, keepdim=True)


def _prep(tokens_np, mod, device, discard=F.DISCARD):
    """fp16 (B, L, C) numpy -> normalized float32 torch (B, L, C) on device."""
    import torch

    x = torch.from_numpy(np.array(tokens_np, dtype=np.float32)).to(
        device
    )  # copy: memmap slices are read-only
    if mod is not None:
        b, n_tok, c = x.shape
        x = ditf_tokens(x.reshape(b * n_tok, c), mod, discard=discard).reshape(b, n_tok, c)
    return x


PREP_CHUNK = 8  # images per DiTF chunk (8 x 1024 x 4096 fp32 = 134 MB plus the DiTF clone)


def _chunks(mm, idx, mod, device, discard=F.DISCARD):
    """The normalized fp32 (b, L, C) chunks of rows `idx` of a token memmap, in order."""
    import torch

    for i in range(0, len(idx), PREP_CHUNK):
        x = _prep(mm[idx[i : i + PREP_CHUNK]], mod, device, discard)
        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"non-finite normalized tokens in rows {idx[i : i + PREP_CHUNK].tolist()} (an inf / NaN in the cache)"
            )
        yield x


def channel_stats(chunks):
    """Per-channel mean and unbiased std (+1e-6) over every token of the (b, L, C) fp32 chunks
    iterated ONCE: float64 sums of x - c and (x - c)^2 with c = the first chunk's channel mean,
    so a near-constant channel with a large offset does not cancel (the plain sum-of-squares form
    loses 3e-4 relative at |mean| / std = 1e6). Equals tr.reshape(-1, C).mean(0) / .std(0) + 1e-6
    of the fp32 set (the old one-shot reduction) to float64 round-off, far below fp32."""
    import torch

    shift = s = ss = n = None
    for x in chunks:
        x = x.reshape(-1, x.shape[-1]).double()
        if shift is None:
            shift = x.mean(0)
            s, ss, n = torch.zeros_like(shift), torch.zeros_like(shift), 0
        d = x - shift
        s, ss, n = s + d.sum(0), ss + (d * d).sum(0), n + d.shape[0]
    var = (ss - s * s / n) / (n - 1)
    std = var.clamp_min(0).sqrt()
    return (shift + s / n).float(), std.float() + 1e-6


def prep_all(mm, idx, mod, device, out_device, dtype, mean=None, std=None, discard=F.DISCARD):
    """Rows `idx` of a token memmap, DiTF-normalized chunk by chunk on `device`, optionally
    channel-standardized with (mean, std), stored in `dtype` on `out_device`.
    Train tokens: standardized first, THEN float16 on the GPU -- the rounding the head sees is
    2^-11 in standardized units for every channel (fp16 before standardization would amplify it
    by |x_c| / std_c on near-constant channels). 100 % m_pv4ger_seg: 20 GB FLUX / 27 GB DINO on
    the 48 GB A40 instead of the 40 / 54 GB fp32 that overflowed it.
    Val / test tokens: fp32 on the host (their standardization is per cell), <= 34 GB."""
    import torch

    idx = np.asarray(idx)
    out = torch.empty((len(idx), *mm.shape[1:]), dtype=dtype, device=out_device)
    for i, x in zip(range(0, len(idx), PREP_CHUNK), _chunks(mm, idx, mod, device, discard)):
        if mean is not None:
            x = (x - mean) / std
        out[i : i + PREP_CHUNK] = x.to(dtype).to(out_device)
    return out


def load_masks(mask_files: list[str]) -> np.ndarray:
    from PIL import Image

    out = None
    for i, f in enumerate(mask_files):
        m = np.asarray(Image.open(f))
        if m.dtype != np.uint8 or m.ndim != 2:
            raise SystemExit(f"{f}: {m.dtype} {m.shape}, expected a uint8 class-index mask")
        if out is None:
            out = np.zeros((len(mask_files), *m.shape), np.uint8)
        out[i] = m
    return out


EVAL_CHUNK = 16  # images per val/test forward


def fit_eval_candidate(
    tr_x,
    tr_mask,
    ev_sets: dict,
    mean,
    std,
    n_cls: int,
    grid: tuple,
    lr_grid,
    steps: int,
    batch_images: int,
    seed: int,
    device,
) -> dict:
    """Train one linear head per lr on the probe-training tokens (on device, DiTF-normalized and
    channel-standardized with (mean, std) fitted on them, (n, L, C)), return
    {lr: {"val": mIoU, "test_conf": (N_test, C, C), "val_conf": ...}}.
    ev_sets: {split: (normalized fp32 tokens (N, L, C) torch on the host, masks (N, H, W))},
    standardized with the same (mean, std) per chunk; every lr head is scored in one pass."""
    import torch
    import torch.nn.functional as TF

    h, w = grid
    n, L, C = tr_x.shape
    tr_y = torch.from_numpy(tr_mask).to(device)  # uint8; int64 per batch only
    H, W = tr_y.shape[-2:]
    # a subset smaller than one batch would otherwise be re-drawn every step; the batch size
    # that was actually used is recorded in the result so the sample budget is auditable
    batch_images = min(int(batch_images), n)
    out, heads = {}, {}
    for lr in lr_grid:
        g = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)
        head = torch.nn.Linear(C, n_cls).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=float(lr))
        order = torch.randperm(n, generator=g)
        pos = 0
        for step in range(steps):
            if pos + batch_images > n:
                order, pos = torch.randperm(n, generator=g), 0
            idx = order[pos : pos + batch_images].to(device)
            pos += batch_images
            logits = head(tr_x[idx].float())  # b, L, n_cls
            logits = logits.transpose(1, 2).reshape(len(idx), n_cls, h, w)
            logits = TF.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            loss = TF.cross_entropy(logits, tr_y[idx].long(), ignore_index=F.SEG_IGNORE)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {step} (lr={lr}): an all-ignore batch or a diverged head"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        heads[float(lr)] = head
        out[float(lr)] = {
            "lr": float(lr),
            "train_loss_last": float(loss.item()),
            "batch_images": int(batch_images),
        }
    with torch.no_grad():
        for split, (tok, masks) in ev_sets.items():
            confs = {lr: [] for lr in heads}
            for i in range(0, tok.shape[0], EVAL_CHUNK):
                x = (tok[i : i + EVAL_CHUNK].to(device) - mean) / std
                for lr, head in heads.items():
                    logits = head(x).transpose(1, 2).reshape(x.shape[0], n_cls, h, w)
                    logits = TF.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                    pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    confs[lr].append(confusion_per_image(pred, masks[i : i + EVAL_CHUNK], n_cls))
            for lr in heads:
                conf = np.concatenate(confs[lr])
                out[lr][f"{split}_conf"] = conf
                out[lr][split] = miou(conf.sum(0))
    return out


# ----------------------------------------------------------------------------- cells
def cell_path(out_dir: str, model: str, dataset: str, frac: float, seed: int) -> str:
    return os.path.join(out_dir, "cells", model, dataset, f"f{frac:g}_s{seed}.npz")


_SHA1_CACHE: dict = {}


def file_sha1(path: str) -> str:
    """Full-content sha1, cached in <path>.seg_sha1.json keyed on (size, mtime_ns, inode) so a
    20 GB token file is read once per FILE, not per run. A head + tail window would miss a
    re-extraction that changed rows in the middle of the file and keep a stale cell. The inode is
    part of the key because Lustre stores mtime to the second and every file of one extraction
    is written within it: a rename-swap of two same-size files keeps (size, mtime) at each path
    but not the inode. (eval.wb.file_identity keys on (size, mtime_ns) only; it is shared with
    the running matrix jobs and is left as is.)"""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, st.st_ino)

    def same_file() -> bool:  # the path still holds the file `st` described
        now = os.stat(path)
        return (now.st_size, now.st_mtime_ns, now.st_ino) == key[1:]

    sidecar = path + ".seg_sha1.json"
    if key not in _SHA1_CACHE:
        try:
            with open(sidecar) as fh:
                rec = json.load(fh)
            if (rec.get("bytes"), rec.get("mtime_ns"), rec.get("ino")) == key[1:] and rec.get("sha1"):
                _SHA1_CACHE[key] = rec["sha1"]
        except (OSError, ValueError):
            pass
    if key not in _SHA1_CACHE:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(64 << 20), b""):
                h.update(block)
        if not same_file():
            raise SystemExit(f"{path}: rewritten while being hashed")
        _SHA1_CACHE[key] = h.hexdigest()
        try:  # atomic: a concurrent reader sees the old sidecar or the new one, never a prefix
            tmp = f"{sidecar}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(
                    {
                        "bytes": st.st_size,
                        "mtime_ns": st.st_mtime_ns,
                        "ino": st.st_ino,
                        "sha1": _SHA1_CACHE[key],
                    },
                    fh,
                )
            os.replace(tmp, sidecar)
        except OSError:
            pass  # an unwritable cache dir just skips the sidecar
    if not same_file():
        raise SystemExit(f"{path}: replaced while being identified")
    return _SHA1_CACHE[key]


# keys that do NOT touch a cell's numbers: labels, the cache location, and the pins -- which are
# enforced on every TokenSource build, on the stored meta of every split when the cache is gone
# (CellInputCheck) and on every tabulated cell (collect), both via check_cell_pins
MODEL_LABEL_KEYS = ("name", "tokens", "teacher", "expect")


def cell_config(model: dict, dataset: str, frac: float, seed: int, spec: dict) -> dict:
    m = _resolve(model, dataset)
    return {
        "model": {
            k: v for k, v in m.items() if k not in MODEL_LABEL_KEYS and not k.startswith("_")
        },  # any other key invalidates
        "dataset": dataset,
        "frac": frac,
        "seed": seed,
        "lr_grid": [float(x) for x in spec.get("lr_grid", DEFAULT_LR_GRID)],
        "steps": int(spec.get("steps", DEFAULT_STEPS)),
        "batch_images": int(spec.get("batch_images", DEFAULT_BATCH_IMAGES)),
        "selection": "full official val, mIoU",
        "protocol": PROTOCOL_VERSION,
    }


def cache_present(model: dict, dataset: str) -> bool:
    """Whether ANY split's token cache of this (model, dataset) exists under the spec's tokens dir."""
    m = _resolve(model, dataset)
    for split in F.official_seg_spec(dataset)["sizes"]:
        name = meta_name(m, dataset, split)
        pat = (
            os.path.join(glob.escape(m["tokens"]), "*", name)
            if m["kind"] == "flux"
            else os.path.join(m["tokens"], name)
        )
        if glob.glob(pat):
            return True
    return False


class CellInputCheck:
    """Decides, once per (model, dataset), whether existing cells may be reused. The question is
    always "do the token files the spec points at NOW carry the bytes the cell was computed
    from?", so a spec repointed to a re-extracted cache is refused even while the old files
    still exist (cell_config no longer stores the tokens dir).
      cache under the spec's tokens dir: its files' sha1s (hashed once), keyed by ROLE
        (candidate/split, meta/split) so each file keeps its binding, must equal the stored ones
        -> reused (same dir or relocated), else refused;
      no cache there, the stored files gone: reused -- the README deletes a dataset's token dir
        once its cells exist -- with the spec's pins re-checked on the cell's stored meta of
        every split (check_cell_pins), and said aloud;
      no cache there, stored files (partly) present elsewhere: refused (a mis-pointed spec)."""

    def __init__(self, model: dict, dataset: str):
        self.model, self.dataset = model, dataset
        self.m = _resolve(model, dataset)
        self.present = cache_present(model, dataset)
        self.src = TokenSource(model, dataset) if self.present else None
        self.inputs = self.src.input_sha1s() if self.present else None

    def check(self, p: str, old_inputs: dict, metas: dict) -> None:
        if not old_inputs or not all(
            isinstance(v, dict) and {"path", "sha1"} <= set(v) for v in old_inputs.values()
        ):
            raise SystemExit(f"{p}: old inputs format (not role-keyed) -- --redo to recompute")
        if self.present:
            if {r: v["sha1"] for r, v in old_inputs.items()} != {
                r: v["sha1"] for r, v in self.inputs.items()
            }:
                raise SystemExit(f"{p}: the cache under {self.m['tokens']} does not hold the bytes this cell was "
                                 f"computed from (a re-extraction) -- --redo to recompute")  # fmt: skip
            if {v["path"] for v in old_inputs.values()} != {v["path"] for v in self.inputs.values()}:
                print(f"{p}: kept (token files relocated, sha1s unchanged)")
            return
        present = [v["path"] for v in old_inputs.values() if os.path.exists(v["path"])]
        if present:
            raise SystemExit(f"{p}: no cache under the spec's tokens dir, but {len(present)} of its {len(old_inputs)} "
                             f"token files exist elsewhere -- point the spec at them or --redo")  # fmt: skip
        check_cell_pins(p, self.m, self.dataset, metas)
        print(
            f"{p}: kept (config matches, pins hold on its stored metas; its token files were deleted, no cache of that tag remains)"
        )


def check_cell_pins(p: str, m: dict, dataset: str, metas: dict) -> None:
    """The spec's pins (expect, dataset, split) against the cell's stored meta of EVERY split --
    what TokenSource enforces on a live cache, re-enforced on the record when the cache is gone
    (run) and whenever a cell is tabulated (collect)."""
    for split in F.official_seg_spec(dataset)["sizes"]:
        if split not in metas:
            raise SystemExit(
                f"{p}: no stored meta for split {split!r} -- an old cell format, --redo to recompute"
            )
        check_meta(
            metas[split],
            {"dataset": dataset, "split": split, **dict(m.get("expect") or {})},
            f"{p} (stored {split} meta)",
        )


def run(
    spec: dict, models, datasets, out_dir: str, redo: bool, device: str = "cuda", train_dtype: str = "float16"
) -> list[dict]:
    import torch

    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    seeds = [int(s) for s in spec.get("seeds", DEFAULT_SEEDS)]
    lr_grid = [float(x) for x in spec.get("lr_grid", DEFAULT_LR_GRID)]
    steps = int(spec.get("steps", DEFAULT_STEPS))
    batch_images = int(spec.get("batch_images", DEFAULT_BATCH_IMAGES))
    if steps < 1 or batch_images < 1 or not lr_grid:
        raise SystemExit(
            f"spec: steps={steps}, batch_images={batch_images}, lr_grid={lr_grid} -- each must be >= 1 / non-empty"
        )
    dev = torch.device(device)
    tr_dtype = {"float16": torch.float16, "float32": torch.float32}[train_dtype]
    compute = {"train_dtype": train_dtype, "device": dev.type}
    done = []
    for model in models:
        for dataset in datasets:
            spec_d = F.official_seg_spec(dataset)
            classes = list(spec_d["classes"])
            n_cls = len(classes)
            cells = [(f, s) for f in fractions for s in (seeds if f < 1 else [seeds[0]])]
            todo, inputs_check = [], None
            for f, s in cells:
                p = cell_path(out_dir, model["name"], dataset, f, s)
                cfg = cell_config(model, dataset, f, s, spec)
                if os.path.exists(p) and not redo:
                    stored = np.load(p, allow_pickle=True)
                    old = json.loads(str(stored["config"]))
                    if old != cfg:
                        raise SystemExit(f"{p}: stored config differs from the spec's (--redo to recompute)")
                    inputs_check = inputs_check or CellInputCheck(model, dataset)
                    inputs_check.check(p, json.loads(str(stored["inputs"])), json.loads(str(stored["metas"])))
                    continue
                todo.append((f, s, p, cfg))
            if not todo:
                print(f"{model['name']} x {dataset}: all {len(cells)} cells present")
                continue
            src = inputs_check.src if inputs_check and inputs_check.present else TokenSource(model, dataset)
            grid = src.grid()
            masks, paths = {}, {}
            for split in src.sizes:
                images, mask_files = F.list_official_seg_split(dataset, split)
                paths[split] = images
                if src._paths(split) != images:
                    raise SystemExit(
                        f"{model['name']} {dataset} {split}: token cache image list != the official listing"
                    )
                masks[split] = load_masks(mask_files)
                if masks[split].shape[1:] != tuple(spec_d["mask_hw"]):
                    raise SystemExit(
                        f"{dataset} {split}: masks {masks[split].shape[1:]} != registered {spec_d['mask_hw']}"
                    )
                bad = np.setdiff1d(np.unique(masks[split]), np.r_[np.arange(n_cls), F.SEG_IGNORE])
                if bad.size:
                    raise SystemExit(
                        f"{dataset} {split}: mask values {bad.tolist()} outside the {n_cls} classes + ignore"
                    )
            inputs = inputs_check.inputs if inputs_check and inputs_check.present else src.input_sha1s()
            subsets = {s: image_subsets(len(paths["train"]), fractions, s) for s in seeds}
            # one load per candidate file, every (frac, seed, lr) head trained from it
            results: dict[tuple, dict] = {}
            for cand, per_split in src.files.items():
                t0 = time.time()
                mods = {
                    s: (
                        None
                        if src.mod_for(cand, s) is None
                        else torch.from_numpy(src.mod_for(cand, s)).to(dev)
                    )
                    for s in src.sizes
                }
                discard = {
                    s: src.discard_for(s) for s in src.sizes
                }  # per split's meta (FLUX constant / stamped list)
                mm = {s_: np.load(per_split[s_], mmap_mode="r") for s_ in src.sizes}
                for s_, m_ in mm.items():
                    if m_.shape[0] != len(paths[s_]) or m_.shape[1] != grid[0] * grid[1]:
                        raise SystemExit(
                            f"{per_split[s_]}: {m_.shape} rows x tokens, the listing has {len(paths[s_])} images on a {grid} grid"
                        )
                # val / test read from Lustre and normalized ONCE per candidate (host, fp32)
                ev = {s_: (prep_all(mm[s_], np.arange(mm[s_].shape[0]), mods[s_], dev, "cpu", torch.float32, discard=discard[s_]), masks[s_])
                      for s_ in ("val", "test")}  # fmt: skip
                for f, s, p, cfg in todo:
                    idx = subsets[s][f]
                    # standardization fitted on the cell's probe-training tokens only (rule 3)
                    mean, std = channel_stats(_chunks(mm["train"], idx, mods["train"], dev, discard["train"]))
                    tr_tok = prep_all(
                        mm["train"],
                        idx,
                        mods["train"],
                        dev,
                        dev,
                        tr_dtype,
                        mean,
                        std,
                        discard=discard["train"],
                    )
                    r = fit_eval_candidate(tr_tok, masks["train"][idx], ev, mean, std, n_cls, grid, lr_grid, steps,
                                           batch_images, s, dev)  # fmt: skip
                    for lr, res in r.items():
                        results[(cand, lr, f, s)] = res
                    del tr_tok
                    torch.cuda.empty_cache() if dev.type == "cuda" else None
                del ev, mm
                print(
                    f"  {model['name']} {dataset} {cand}: {len(todo)} cells x {len(lr_grid)} lr in {time.time() - t0:.0f}s",
                    flush=True,
                )
            for f, s, p, cfg in todo:
                table = [{"candidate": c, "lr": lr, "val_miou": results[(c, lr, f, s)]["val"],
                          "batch_images": results[(c, lr, f, s)]["batch_images"], "train_loss_last": results[(c, lr, f, s)]["train_loss_last"]}
                         for c in src.files for lr in lr_grid]  # fmt: skip
                if any(math.isnan(r["val_miou"]) for r in table):
                    raise SystemExit(
                        f"{model['name']} {dataset} f={f:g} s={s}: NaN val mIoU in the table -- refusing to select"
                    )
                best = max(
                    table, key=lambda r: r["val_miou"]
                )  # first max wins (max keeps the first maximal row)
                res = results[(best["candidate"], best["lr"], f, s)]
                test_conf = res["test_conf"]  # the ONE opened test result
                lo, hi = bootstrap_miou(test_conf) if f == 1.0 else (float("nan"), float("nan"))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(
                    p,
                    test_miou=np.array(miou(test_conf.sum(0))),
                    test_conf_per_image=test_conf,
                    test_paths=np.array(paths["test"]),
                    val_miou=np.array(best["val_miou"]),
                    selected=np.array(
                        json.dumps(
                            {"candidate": best["candidate"], "lr": best["lr"], "val_miou": best["val_miou"]}
                        )
                    ),
                    val_table=np.array(json.dumps(table)),
                    train_indices=subsets[s][f],
                    classes=np.array(classes),
                    ci95=np.array([lo, hi]),
                    config=np.array(json.dumps(cfg)),
                    inputs=np.array(json.dumps(inputs)),
                    metas=np.array(json.dumps(src.meta)),
                    compute=np.array(json.dumps(compute)),
                )
                print(f"{model['name']} {dataset} f={f:g} s={s}: test mIoU {miou(test_conf.sum(0)):.4f} "
                      f"(val {best['val_miou']:.4f}, {best['candidate']} lr {best['lr']:g})", flush=True)  # fmt: skip
                done.append({"model": model["name"], "dataset": dataset, "frac": f, "seed": s, "path": p})
    return done


# ----------------------------------------------------------------------------- report
def collect(spec: dict, out_dir: str) -> tuple[dict, list[dict]]:
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    seeds = [int(s) for s in spec.get("seeds", DEFAULT_SEEDS)]
    cells, missing = {}, []
    compute_seen: dict[str, str] = {}
    for model in spec["models"]:
        for dataset in spec["datasets"]:
            inputs_check = None  # built once per (model, dataset): hashes the token files now on disk
            for f in fractions:
                for s in seeds if f < 1 else [seeds[0]]:
                    p = cell_path(out_dir, model["name"], dataset, f, s)
                    if not os.path.exists(p):
                        missing.append({"model": model["name"], "dataset": dataset, "frac": f, "seed": s})
                        continue
                    d = np.load(p, allow_pickle=True)
                    cfg = json.loads(str(d["config"]))
                    if cfg != cell_config(model, dataset, f, s, spec):
                        raise SystemExit(f"{p}: stale cell (config differs from the spec); rerun with --redo")
                    # Same gate as `run`: the stored input sha1s against the token files the spec points at
                    # NOW (a re-extracted cache is refused), pins on the stored metas when the cache is gone.
                    inputs_check = inputs_check or CellInputCheck(model, dataset)
                    inputs_check.check(p, json.loads(str(d["inputs"])), json.loads(str(d["metas"])))
                    comp = json.loads(str(d["compute"])) if "compute" in d.files else SEG_COMPUTE_LEGACY
                    comp_key = json.dumps(comp, sort_keys=True)
                    if compute_seen and comp_key not in compute_seen:
                        other = next(iter(compute_seen))
                        raise SystemExit(f"{p}: fitted at {comp}, but {compute_seen[other]} at {json.loads(other)} -- "
                                         f"one table never mixes operating points; --redo the odd cells")  # fmt: skip
                    compute_seen.setdefault(comp_key, p)
                    cells[(model["name"], dataset, f, s)] = {
                        "test": float(d["test_miou"]), "val": float(d["val_miou"]),
                        "ci95": d["ci95"].tolist(), "selected": json.loads(str(d["selected"])),
                    }  # fmt: skip
    return cells, missing


def summarize(spec: dict, cells: dict) -> dict:
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    out = {}
    for model in spec["models"]:
        for dataset in spec["datasets"]:
            per_f = {}
            for f in fractions:
                vals = [
                    v["test"]
                    for (m, d, ff, s), v in cells.items()
                    if m == model["name"] and d == dataset and ff == f
                ]
                if not vals:
                    continue
                per_f[f] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
                if f == 1.0:
                    per_f[f]["ci95"] = cells[
                        (model["name"], dataset, f, spec.get("seeds", DEFAULT_SEEDS)[0])
                    ]["ci95"]
                    per_f[f]["selected"] = cells[
                        (model["name"], dataset, f, spec.get("seeds", DEFAULT_SEEDS)[0])
                    ]["selected"]
            aulc = (
                float(np.mean([per_f[f]["mean"] for f in fractions]))
                if all(f in per_f for f in fractions)
                else None
            )
            out[(model["name"], dataset)] = {
                "per_fraction": per_f,
                "aulc": aulc,
                "teacher": model.get("teacher"),
            }
    return out


def tables_markdown(spec: dict, summary: dict, missing: list[dict]) -> str:
    fractions = [float(f) for f in spec.get("fractions", DEFAULT_FRACTIONS)]
    lines = [
        f"# Segmentation label-efficiency ({PROTOCOL_VERSION}; linear per-token head, val-selected (t, k, lr); mIoU)",
        "",
    ]
    for dataset in spec["datasets"]:
        lines += [
            f"### {dataset} (mIoU; mean ± sample std (ddof=1) over seeds; 100 % = single fit, 95 % image-bootstrap CI)",
            "",
        ]
        lines.append(
            "| model | "
            + " | ".join(f"{f * 100:g} %" for f in fractions)
            + " | AULC | Δ AULC vs teacher | selected @100 % |"
        )
        lines.append("|---|" + "---|" * (len(fractions) + 3))
        for model in spec["models"]:
            r = summary.get((model["name"], dataset))
            if not r:
                continue
            cells = []
            for f in fractions:
                c = r["per_fraction"].get(f)
                if c is None:
                    cells.append("MISSING")
                elif f == 1.0:
                    cells.append(f"{c['mean']:.4f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]")
                else:
                    cells.append(f"{c['mean']:.4f} ± {c['std']:.4f}")
            t = r["teacher"]
            delta = "—"
            if t and summary.get((t, dataset), {}).get("aulc") is not None and r["aulc"] is not None:
                delta = f"{r['aulc'] - summary[(t, dataset)]['aulc']:+.4f}"
            sel = r["per_fraction"].get(1.0, {}).get("selected", {})
            lines.append(f"| {model['name']} | " + " | ".join(cells) + f" | {r['aulc']:.4f} | {delta} | {sel.get('candidate', '')} lr {sel.get('lr', '')} |"
                         if r["aulc"] is not None else f"| {model['name']} | " + " | ".join(cells) + " | MISSING | — | |")  # fmt: skip
        lines.append("")
    if missing:
        lines += ["MISSING cells:", ""] + [
            f"- {m['model']} x {m['dataset']} f={m['frac']:g} s={m['seed']}" for m in missing
        ]
    return "\n".join(lines)


def report(spec: dict, out_dir: str) -> str:
    cells, missing = collect(spec, out_dir)
    summary = summarize(spec, cells)
    md = tables_markdown(spec, summary, missing)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({f"{m}|{d}": v for (m, d), v in summary.items()}, f, indent=1)
    print(md)
    print(f"-> {out_dir}/summary.md, summary.json; MISSING cells: {len(missing)}")
    return md


def load_spec(path: str) -> dict:
    spec = json.load(open(path))
    for k in ("datasets", "models"):
        if not spec.get(k):
            raise SystemExit(f"{path}: no {k!r}")
    names = [m["name"] for m in spec["models"]]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate model names {names}")
    for m in spec["models"]:
        if m.get("teacher") and m["teacher"] not in names:
            raise SystemExit(f"{path}: model {m['name']} names an unknown teacher {m['teacher']!r}")
    for d in spec["datasets"]:
        F.official_seg_spec(d)
    return spec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--models", nargs="*", default=None)
    r.add_argument("--datasets", nargs="*", default=None)
    r.add_argument("--out-dir", default=OUT_DIR)
    r.add_argument("--redo", action="store_true")
    r.add_argument("--device", default="cuda")
    r.add_argument(
        "--train-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="GPU storage of the standardized train tokens",
    )
    q = sub.add_parser("report")
    q.add_argument("--spec", required=True)
    q.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)
    spec = load_spec(args.spec)
    if args.cmd == "run":
        models = [m for m in spec["models"] if not args.models or m["name"] in args.models]
        datasets = [d for d in spec["datasets"] if not args.datasets or d in args.datasets]
        if args.models and len(models) != len(args.models):
            raise SystemExit(f"unknown --models among {args.models}")
        run(spec, models, datasets, args.out_dir, args.redo, args.device, args.train_dtype)
    else:
        report(spec, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
