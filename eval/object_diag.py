"""Object-level representation diagnostic on the cached segmentation tokens. CPU only, no training.

    python -m eval.object_diag run    --dataset m_pv4ger_seg [--spec eval/seg_specs/phase2_arms.json] [--max-images 48]
    python -m eval.object_diag report [--datasets ...] [--smoke]
    python -m eval.object_diag figure --dataset m_pv4ger_seg

Question: does semantic adaptation make the pretrained generator's token representation more
object-discriminative, or does it mainly alter global image semantics? The same images are scored
for DINOv3-Sat (the alignment teacher), DINOv3-web, frozen FLUX and every adapted FLUX arm of the
spec (block 28, every cached timestep: NO timestep is selected, rule 1). Masks are used strictly to
SCORE the tokens; nothing is fitted on them.

Per image (the independent unit, rule 4), on the tokens eval.seg_probe feeds its linear head before
the head's channel standardization (FLUX: DiTF, the block's adaLN applied, L2; DINO: the normed
patch tokens):
  * eta2: share of the within-image token variance explained by the mask classes (between-class
    sum of squares / total, tokens centred on the image);
  * gap: object-background cosine similarity gap, mean cosine of same-class token pairs minus
    mean cosine of different-class pairs (tokens centred on the image, then L2);
  * nn: class-balanced share of tokens whose most similar token is of the same class, among tokens
    more than NN_EXCLUDE grid steps away (adjacent tokens share receptive fields);
  * cka_t: debiased linear CKA between the image's token Gram and the teacher's (the spatial
    structure the alignment loss is meant to transfer). No mask involved.
Each statistic is reported against a SWAP NULL: the same statistic with the mask (for cka_t, the
teacher tokens) of another image. A mask of another image has the same spatial statistics, so
anything a positional code alone would score cancels; "true - swap" is the image-specific part.
Both directions are needed to read a row (rule 3b's analogue): the swap-corrected value says the
structure is tied to this image's objects, the raw value says how much of it there is.

Per dataset:
  * what adaptation changed (FLUX arms, same noise seed as the frozen cache): the squared token
    change adapted - frozen split into a dataset-constant shift, an image-global shift (constant
    over the image's tokens) and a local, spatially varying part; and the eta2 of the local part
    against the masks (is the local change aligned with objects?);
  * global variance share: between-image share of the token variance;
  * global CKA to the teacher on the image-mean tokens.

Executed gates (rule 13), synthetic tokens scored through the same code on the real masks:
  (a) noise control and (a2) position-only control: every "true - swap" interval must contain 0
      (Bonferroni over the gate's null checks);
  (b) object control (class embedding + position + noise) must beat its swap null on eta2 / gap /
      nn, and a noisy random projection of the teacher tokens must beat its swap null on cka_t.
On FAIL the report prints "do NOT quote any cell above".

Per-image vectors of every arm are cached under results/object_diag/arms/<dataset>/<arm>.npz
(rule 12); `report` and `figure` read only those.
"""

from __future__ import annotations

import argparse
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
from eval import seg_probe as SP  # noqa: E402

VERSION = "2026-11-10"  # bump when a code change alters what an arm computes
OUT_DIR = "results/object_diag"
DEFAULT_SPEC = "eval/seg_specs/phase2_arms.json"
DATASETS = ("m_pv4ger_seg", "m_chesapeake_landcover", "m_SA_crop_type")
TEACHER = "dinov3-sat"
FROZEN = "flux-pretrained"
BLOCK = 28  # the adapted arms' caches hold block 28 only; the frozen arm is read at the same block
LABELS = {
    "dinov3-sat": "DINOv3-Sat (teacher)",
    "dinov3-web": "DINOv3-web",
    "flux-pretrained": "frozen FLUX",
    "flux-naive": "joint-adapted FLUX",
    "flux-proj": "projected-joint FLUX",
    "flux-sem": "semantic-only FLUX",
}
SEED = 0
N_BOOT = 10000
VALID_FRAC = 0.5  # a token is labelled when >= this share of its pixels is not ignore
MIN_TOK = 8  # a class counts in an image with >= this many tokens; an image needs two such classes
N_SWAP = 4  # swap-null partners per image (averaged)
NN_EXCLUDE = 2  # Chebyshev radius (grid steps) excluded from the nearest-neighbour search
CHUNK = 8  # images per token read
PCA_N = 6  # images of the patch-PCA figure (seeded draw from the eligible images)
FIG_T = 260  # FLUX timestep of the figure: the middle of the cached grid, fixed before any result
CTRL_DIM = 32
CTRL_NOISE = 0.5
CTRL_OBJ = {"ctrl-object": 0.5, "ctrl-objectw": 0.1}  # class-embedding scale; the weak one is informational
CTRLS = ("ctrl-noise", "ctrl-position", "ctrl-object", "ctrl-objectw", "ctrl-teacherproj")
MASK_STATS = ("eta2", "gap", "nn")


# ----------------------------------------------------------------------------- labels and nulls
def token_labels(
    masks: np.ndarray, grid: tuple[int, int], n_cls: int, valid_frac: float = VALID_FRAC
) -> np.ndarray:
    """(n, H, W) uint8 class-index masks -> (n, h*w) int16 token labels: the majority class of the
    token's pixel block, -1 when fewer than valid_frac of its pixels carry a class."""
    n, H, W = masks.shape
    h, w = grid
    if H % h or W % w:
        raise SystemExit(f"mask {H}x{W} is not a multiple of the token grid {h}x{w}")
    bad = np.setdiff1d(np.unique(masks), np.r_[np.arange(n_cls), F.SEG_IGNORE])
    if bad.size:
        raise SystemExit(f"mask values {bad.tolist()} outside the {n_cls} classes + ignore")
    blocks = masks.reshape(n, h, H // h, w, W // w)
    counts = np.stack([(blocks == c).sum((2, 4)) for c in range(n_cls)], -1)  # n, h, w, n_cls
    lab = counts.argmax(-1).astype(np.int16)
    lab[counts.sum(-1) < valid_frac * (H // h) * (W // w)] = -1
    return lab.reshape(n, h * w)


def eligible_labels(lab: np.ndarray, min_tok: int = MIN_TOK) -> tuple[np.ndarray, np.ndarray]:
    """Drop (-1) the tokens of classes with < min_tok tokens in their image; an image is eligible
    with >= 2 remaining classes. A function of the masks alone: identical for every arm."""
    lab = lab.copy()
    eligible = np.zeros(len(lab), bool)
    for i in range(len(lab)):
        cls, cnt = np.unique(lab[i][lab[i] >= 0], return_counts=True)
        for c in cls[cnt < min_tok]:
            lab[i][lab[i] == c] = -1
        eligible[i] = int((cnt >= min_tok).sum()) >= 2
    return lab, eligible


def swap_partners(members: np.ndarray, n: int, n_swap: int = N_SWAP, seed: int = SEED) -> np.ndarray:
    """(n, n_swap) partner image per member image, -1 for non-members: the k-th successors in a
    seeded cyclic order of the members, so no image is its own partner and partners are distinct."""
    members = np.asarray(members)
    if len(members) <= n_swap:
        raise SystemExit(f"swap null: {len(members)} images, need more than {n_swap}")
    perm = np.random.default_rng([seed, 11]).permutation(members)
    out = np.full((n, n_swap), -1, np.int64)
    for k in range(1, n_swap + 1):
        out[perm, k - 1] = np.roll(perm, -k)
    return out


# ----------------------------------------------------------------------------- per-image statistics
def gram(x):
    """(L, C) tokens -> float64 Gram of the tokens centred on their mean."""
    x = x.double()
    x = x - x.mean(0, keepdim=True)
    return x @ x.T


def mask_stats(G, y: np.ndarray, grid: tuple[int, int]) -> tuple[float, float, float]:
    """(eta2, gap, nn) of one image's token Gram against token labels y (-1 = not scored)."""
    import torch

    v = np.flatnonzero(y >= 0)
    cls, inv = np.unique(y[v], return_inverse=True)
    if len(cls) < 2:
        raise ValueError("mask_stats needs two classes")
    vi = torch.from_numpy(v)
    g = G[vi][:, vi]
    g = g - g.mean(0, keepdim=True) - g.mean(1, keepdim=True) + g.mean()  # recentre on the scored tokens
    onehot = torch.zeros((len(v), len(cls)), dtype=g.dtype)
    onehot[torch.arange(len(v)), torch.from_numpy(inv)] = 1.0
    nc = onehot.sum(0)
    diag = g.diagonal()
    ss_total = diag.sum()
    if not ss_total > 0:
        raise RuntimeError("constant tokens: zero within-image variance")
    eta2 = float(((onehot.T @ g @ onehot).diagonal() / nc).sum() / ss_total)
    d = diag.clamp_min(1e-30).sqrt()
    s = g / (d[:, None] * d[None, :])
    same = onehot @ onehot.T
    n_same = float(same.sum()) - len(v)  # ordered same-class pairs, self excluded
    within = (float((s * same).sum()) - float(s.diagonal().sum())) / n_same
    cross = float((s * (1 - same)).sum()) / float((1 - same).sum())
    w = grid[1]
    r, c = torch.from_numpy(v // w), torch.from_numpy(v % w)
    near = ((r[:, None] - r[None, :]).abs() <= NN_EXCLUDE) & ((c[:, None] - c[None, :]).abs() <= NN_EXCLUDE)
    best, arg = s.masked_fill(near, -math.inf).max(1)
    ok = torch.isfinite(best).numpy()
    hit = inv[arg.numpy()] == inv
    rates = [hit[ok & (inv == k)].mean() if (ok & (inv == k)).any() else np.nan for k in range(len(cls))]
    nn = float(np.mean(rates))  # NaN when a class has no token with a non-adjacent candidate
    return eta2, within - cross, nn


def hsic_u(K, L):
    """Unbiased HSIC estimator (Song et al. 2012) of two (n, n) kernel matrices."""
    n = K.shape[0]
    K = K - K.diagonal().diag()
    L = L - L.diagonal().diag()
    ks, ls = K.sum(1), L.sum(1)
    return ((K * L).sum() + ks.sum() * ls.sum() / ((n - 1) * (n - 2)) - 2 * (ks @ ls) / (n - 2)) / (
        n * (n - 3)
    )


def cka_u(K, L) -> float:
    """Debiased linear CKA (Nguyen et al. 2021) from two Gram matrices."""
    den = float(hsic_u(K, K)) * float(hsic_u(L, L))
    return float(hsic_u(K, L)) / math.sqrt(den) if den > 0 else float("nan")


def position_features(grid: tuple[int, int], dim: int = CTRL_DIM, seed: int = SEED) -> np.ndarray:
    """(h*w, dim) smooth functions of the grid position, the same for every image."""
    rng = np.random.default_rng([seed, 77])
    h, w = grid
    rr, cc = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    pos = np.stack([rr.ravel(), cc.ravel()], 1)
    return np.cos(pos @ rng.normal(0, 2.0, (2, dim)) + rng.uniform(0, 2 * np.pi, dim)).astype(np.float32)


def control_tokens(name: str, i: int, y: np.ndarray, grid: tuple[int, int], teacher_x=None, seed: int = SEED):
    """Synthetic (L, CTRL_DIM) tokens of image i for the gate. y: the image's TRUE token labels."""
    import torch

    rng = np.random.default_rng(
        [seed, 100 + CTRLS.index(name), int(i), 1]
    )  # no all-zero tail: SeedSequence zero-pads
    L = grid[0] * grid[1]
    noise = rng.standard_normal((L, CTRL_DIM)).astype(np.float32)
    if name == "ctrl-noise":
        x = noise
    elif name == "ctrl-position":
        x = position_features(grid, seed=seed) + CTRL_NOISE * noise
    elif name in CTRL_OBJ:
        emb = np.r_[rng.standard_normal((int(y.max()) + 1, CTRL_DIM)), np.zeros((1, CTRL_DIM))].astype(
            np.float32
        )
        x = (
            position_features(grid, seed=seed) + CTRL_NOISE * noise + CTRL_OBJ[name] * emb[y]
        )  # y = -1 -> the zero row
    elif name == "ctrl-teacherproj":
        t = teacher_x.float()
        proj = torch.from_numpy(
            np.random.default_rng([seed, 78]).standard_normal((t.shape[1], CTRL_DIM)).astype(np.float32)
        )
        p = (t - t.mean(0, keepdim=True)) @ proj
        return p / p.std() + torch.from_numpy(noise)
    else:
        raise KeyError(name)
    return torch.from_numpy(x)


# ----------------------------------------------------------------------------- arms
def arm_slug(key: str) -> str:
    return key.replace("/", "__")


def arm_path(out_dir: str, dataset: str, key: str) -> str:
    return os.path.join(out_dir, "arms", dataset, arm_slug(key) + ".npz")


def strided(n: int, k: int | None) -> np.ndarray:
    return np.arange(n) if not k or k >= n else np.unique(np.linspace(0, n - 1, k).round().astype(np.int64))


def check_same_images(got: list[str], ref: list[str], what: str) -> None:
    """The token cache's own ordered image list must be the official listing (rules 6 / 11)."""
    if list(got) != list(ref):
        first = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
        raise SystemExit(
            f"{what}: token cache image list != the official listing ({len(got)} vs {len(ref)} images, first difference at row {first})"
        )


class TokenArm:
    """One (model, candidate) token file of one split, DiTF-normalized on read exactly as eval.seg_probe does."""

    def __init__(self, model: dict, dataset: str, split: str, cand: str, images: list[str]):
        import torch

        self.name, self.kind, self.cand = model["name"], model["kind"], cand
        self.key = f"{self.name}/{cand}"
        src = SP.TokenSource(model, dataset)
        check_same_images(src._paths(split), images, f"{self.name} {dataset} {split}")
        if cand not in src.files:
            raise SystemExit(f"{self.name} {dataset}: no candidate {cand} in the cache ({sorted(src.files)})")
        self.path = src.files[cand][split]
        self.mm = np.load(self.path, mmap_mode="r")
        g = src.grid()
        if self.mm.shape[0] != len(images) or self.mm.shape[1] != g[0] * g[1]:
            raise SystemExit(
                f"{self.path}: {self.mm.shape} rows x tokens, the listing has {len(images)} images on a {g} grid"
            )
        self.grid = g
        mod = src.mod_for(cand, split)
        self.mod = None if mod is None else torch.from_numpy(np.asarray(mod))
        self.discard = src.discard_for(split)
        roles = [f"{cand}/{split}"] + ([f"meta/{split}"] if self.kind == "flux" else [])
        paths = src.input_roles()
        self.inputs = {r: {"path": paths[r], "sha1": SP.file_sha1(paths[r])} for r in roles}
        self.meta = src.meta[split]

    def load(self, rows: np.ndarray):
        import torch

        x = (
            SP._prep(self.mm[rows], self.mod, "cpu")
            if self.discard is None
            else SP._prep(self.mm[rows], self.mod, "cpu", self.discard)
        )
        if not torch.isfinite(x).all():
            raise RuntimeError(f"{self.key}: non-finite tokens in rows {rows.tolist()}")
        return x


class ControlArm:
    kind, inputs, meta = "ctrl", {}, {}

    def __init__(self, name: str, ctx: dict, teacher: TokenArm | None):
        self.name, self.cand, self.key, self.ctx, self.teacher = name, "ctrl", f"{name}/ctrl", ctx, teacher
        self.grid = ctx["grid"]

    def load(self, rows: np.ndarray):
        import torch

        tx = self.teacher.load(rows) if self.name == "ctrl-teacherproj" else [None] * len(rows)
        pos = np.searchsorted(self.ctx["rows"], rows)  # split row -> position in this run's image list
        if not np.array_equal(self.ctx["rows"][pos], rows):
            raise RuntimeError(f"{self.key}: rows {rows.tolist()} are not all in this run's image list")
        return torch.stack(
            [
                control_tokens(self.name, int(r), self.ctx["lab_true"][pos[j]], self.grid, tx[j])
                for j, r in enumerate(rows)
            ]
        )


def arm_config(arm, dataset: str, split: str, rows: np.ndarray, n_total: int) -> dict:
    return {
        "version": VERSION, "dataset": dataset, "split": split, "arm": arm.key, "kind": arm.kind,
        "n_images": int(len(rows)), "n_split": int(n_total), "smoke": bool(len(rows) != n_total),
        "valid_frac": VALID_FRAC, "min_tok": MIN_TOK, "n_swap": N_SWAP, "nn_exclude": NN_EXCLUDE, "seed": SEED,
        "teacher": TEACHER, "frozen": FROZEN, "block": BLOCK, "grid": [int(g) for g in arm.grid],
    }  # fmt: skip


def new_record(n: int, dim: int, delta: bool) -> dict:
    nan = lambda: np.full(n, np.nan)  # noqa: E731
    rec = {
        k: nan()
        for k in ("eta2", "gap", "nn", "eta2_null", "gap_null", "nn_null", "cka_t", "cka_t_null", "ss_within")
    }
    rec["img_mean"] = np.zeros((n, dim), np.float32)
    if delta:
        rec.update({k: nan() for k in ("d_total", "d_local", "d_cos", "d_eta2", "d_eta2_null")})
        rec["d_mean"] = np.zeros((n, dim), np.float32)
    return rec


def score_masks(G, j: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """((3,) true, (3,) swap-null mean) mask statistics of image position j, NaN when not eligible."""
    if not ctx["eligible"][j]:
        return np.full(3, np.nan), np.full(3, np.nan)
    true = np.array(mask_stats(G, ctx["lab"][j], ctx["grid"]))
    null = np.array([mask_stats(G, ctx["lab"][p], ctx["grid"]) for p in ctx["partners"][j]])
    with np.errstate(invalid="ignore"):
        null_mean = np.array(
            [np.nanmean(null[:, k]) if np.isfinite(null[:, k]).any() else np.nan for k in range(3)]
        )
    return true, null_mean


def fill(rec: dict, j: int, x, G, ctx: dict, tg) -> None:
    true, null = score_masks(G, j, ctx)
    for k, name in enumerate(MASK_STATS):
        rec[name][j], rec[name + "_null"][j] = true[k], null[k]
    if tg is not None:
        rec["cka_t"][j] = cka_u(G, tg[j].double())
        rec["cka_t_null"][j] = float(np.mean([cka_u(G, tg[p].double()) for p in ctx["partners_all"][j]]))
    rec["ss_within"][j] = float(G.diagonal().sum())
    rec["img_mean"][j] = x.double().mean(0).float().numpy()


def fill_delta(rec: dict, j: int, xa, x0, ctx: dict) -> None:
    d = xa.double() - x0.double()
    dm = d.mean(0)
    rec["d_total"][j] = float((d * d).sum())
    rec["d_local"][j] = float(((d - dm) ** 2).sum())
    rec["d_cos"][j] = float((xa.double() * x0.double()).sum(-1).mean())
    rec["d_mean"][j] = dm.float().numpy()
    if ctx["eligible"][j]:
        Gd = gram(d)
        rec["d_eta2"][j] = mask_stats(Gd, ctx["lab"][j], ctx["grid"])[0]
        rec["d_eta2_null"][j] = float(
            np.mean([mask_stats(Gd, ctx["lab"][p], ctx["grid"])[0] for p in ctx["partners"][j]])
        )


def pca_rgb(G):
    """(L, 3) projections of the tokens on their top-3 principal components, from the centred Gram."""
    import torch

    val, vec = torch.linalg.eigh(G)
    return (vec[:, -3:].flip(1) * val[-3:].flip(0).clamp_min(0).sqrt()).float().numpy()


def sweep(
    group: list,
    ctx: dict,
    tg,
    out_dir: str,
    dataset: str,
    split: str,
    store_tg: bool = False,
    base: str | None = None,
) -> None:
    """Score every arm of `group` over the images of ctx (one read per chunk), write one npz per arm.
    store_tg: the group is the teacher alone, its normalized Grams go into tg (no CKA to itself)."""
    import torch

    rows, n = ctx["rows"], len(ctx["rows"])
    recs, pcas = {}, {a.key: {} for a in group}
    t0 = time.time()
    for c0 in range(0, n, CHUNK):
        xs = {a.key: a.load(rows[c0 : c0 + CHUNK]) for a in group}
        for a in group:
            x = xs[a.key]
            if a.key not in recs:
                recs[a.key] = new_record(n, x.shape[-1], delta=base is not None and a.key != base)
            for b in range(x.shape[0]):
                j = c0 + b
                G = gram(x[b])
                if store_tg:
                    tg[j] = (G / G.diagonal().mean()).to(torch.float16)
                fill(recs[a.key], j, x[b], G, ctx, None if store_tg else tg)
                if base is not None and a.key != base:
                    fill_delta(recs[a.key], j, x[b], xs[base][b], ctx)
                if j in ctx["pca_pos"]:
                    pcas[a.key][j] = pca_rgb(G)
        if (c0 // CHUNK) % 10 == 0:
            print(
                f"    {[a.key for a in group]} {min(c0 + CHUNK, n)}/{n} images {time.time() - t0:.0f}s",
                flush=True,
            )
    for a in group:
        p = arm_path(out_dir, dataset, a.key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pos = sorted(pcas[a.key])
        np.savez(
            p + ".tmp.npz",
            **recs[a.key],
            eligible=ctx["eligible"],
            rows=rows,
            paths=np.array([F.path_key(q) for q in ctx["images"]]),
            n_classes_img=ctx["n_classes_img"],
            pca_pos=np.array(pos, np.int64),
            pca_rgb=np.stack([pcas[a.key][q] for q in pos]) if pos else np.zeros((0, 0, 3), np.float32),
            config=np.array(json.dumps(arm_config(a, dataset, split, rows, ctx["n_split"]))),
            inputs=np.array(json.dumps(a.inputs)),
            meta=np.array(json.dumps(a.meta)),
        )
        os.replace(p + ".tmp.npz", p)
        print(f"  wrote {p} ({time.time() - t0:.0f}s)", flush=True)


def build_ctx(dataset: str, split: str, grid: tuple[int, int], max_images: int | None) -> dict:
    spec_d = F.official_seg_spec(dataset)
    images, mask_files = F.list_official_seg_split(dataset, split)
    rows = strided(len(images), max_images)
    masks = SP.load_masks([mask_files[r] for r in rows])
    if masks.shape[1:] != tuple(spec_d["mask_hw"]):
        raise SystemExit(f"{dataset} {split}: masks {masks.shape[1:]} != registered {spec_d['mask_hw']}")
    lab_true = token_labels(masks, grid, len(spec_d["classes"]))
    lab, eligible = eligible_labels(lab_true)
    n = len(rows)
    el = np.flatnonzero(eligible)
    pca_pos = set(
        np.sort(np.random.default_rng([SEED, 5]).choice(el, min(PCA_N, len(el)), replace=False)).tolist()
    )
    return {
        "rows": rows, "images": [images[r] for r in rows], "all_images": images, "n_split": len(images), "grid": grid,
        "lab_true": lab_true, "lab": lab, "eligible": eligible, "partners": swap_partners(el, n),
        "partners_all": swap_partners(np.arange(n), n), "pca_pos": pca_pos,
        "n_classes_img": np.array([len(np.unique(y[y >= 0])) for y in lab], np.int16),
    }  # fmt: skip


def plan_arms(spec: dict, dataset: str) -> list[tuple[str, dict, str]]:
    """[(group, model, candidate)]: the teacher, the other DINO rows, then per FLUX timestep the frozen
    arm and every adapted arm of the spec at block BLOCK."""
    models = {m["name"]: m for m in spec["models"]}
    for need in (TEACHER, FROZEN):
        if need not in models:
            raise SystemExit(f"spec has no model {need!r}")
    plan = [("teacher", models[TEACHER], "patch")]
    plan += [("dino", m, "patch") for m in spec["models"] if m["kind"] == "dino" and m["name"] != TEACHER]
    flux = [models[FROZEN]] + [m for m in spec["models"] if m["kind"] == "flux" and m["name"] != FROZEN]
    ts = [int(t) for t in SP._resolve(models[FROZEN], dataset)["timesteps"]]
    for m in flux[1:]:
        missing = set(ts) - {int(t) for t in SP._resolve(m, dataset)["timesteps"]}
        if missing:
            raise SystemExit(
                f"{m['name']}: timesteps {sorted(missing)} of the frozen arm are not in its spec entry"
            )
    for t in ts:
        plan += [(f"t{t}", m, f"t{t}_k{BLOCK}") for m in flux]
    return plan


def stored_ok(p: str, cfg: dict) -> bool:
    if not os.path.exists(p):
        return False
    with np.load(p) as z:
        old = json.loads(str(z["config"]))
    if old != cfg:
        raise SystemExit(
            f"{p}: stored config differs from this run's (--redo to recompute)\n  stored {old}\n  now    {cfg}"
        )
    return True


def run(args) -> int:
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    spec = SP.load_spec(args.spec)
    dataset, split = args.dataset, args.split
    out_dir = os.path.join(args.out_dir, "smoke") if args.max_images else args.out_dir
    plan = plan_arms(spec, dataset)
    groups = list(dict.fromkeys(g for g, _, _ in plan)) + ["ctrl"]
    want = set(args.groups.split(",")) if args.groups else set(groups)
    if want - set(groups):
        raise SystemExit(f"--groups {sorted(want - set(groups))} not in {groups}")
    images = F.list_official_seg_split(dataset, split)[0]
    teacher = TokenArm(plan[0][1], dataset, split, "patch", images)
    ctx = build_ctx(dataset, split, teacher.grid, args.max_images)
    n, L = len(ctx["rows"]), teacher.grid[0] * teacher.grid[1]
    print(f"{dataset} {split}: {n} images ({int(ctx['eligible'].sum())} eligible: >= 2 classes with >= {MIN_TOK} tokens), "
          f"grid {teacher.grid}, groups {sorted(want)} -> {out_dir}", flush=True)  # fmt: skip

    def pending(arm) -> bool:
        p = arm_path(out_dir, dataset, arm.key)
        return args.redo or not stored_ok(p, arm_config(arm, dataset, split, ctx["rows"], ctx["n_split"]))

    todo: dict[str, list] = {}
    for g in groups[1:-1]:
        if g in want:
            arms = [TokenArm(m, dataset, split, cand, images) for gg, m, cand in plan if gg == g]
            for a in arms:
                if a.grid != teacher.grid:
                    raise SystemExit(f"{a.key}: grid {a.grid} != the teacher's {teacher.grid}")
            if any(pending(a) for a in arms):
                todo[g] = arms  # a FLUX group is recomputed whole: the delta needs the frozen tokens
    if "ctrl" in want:
        ctrls = [ControlArm(c, ctx, teacher) for c in CTRLS]
        if any(pending(a) for a in ctrls):
            todo["ctrl"] = ctrls
    teacher_pending = "teacher" in want and pending(teacher)
    done = len([1 for g, _, _ in plan if g in want]) + (len(CTRLS) if "ctrl" in want else 0)
    if not todo and not teacher_pending:
        print(f"OBJECT_DIAG_RUN_OK {dataset} arms={done} computed=0 out={out_dir}")
        return 0
    tg = torch.empty(
        (n, L, L), dtype=torch.float16
    )  # the teacher's normalized Grams: every CKA and its swap null
    if teacher_pending:
        sweep([teacher], ctx, tg, out_dir, dataset, split, store_tg=True)
    else:
        print("  teacher arm cached; recomputing its Grams for the CKA of the pending arms", flush=True)
        for c0 in range(0, n, CHUNK):
            for b, x in enumerate(teacher.load(ctx["rows"][c0 : c0 + CHUNK])):
                G = gram(x)
                tg[c0 + b] = (G / G.diagonal().mean()).to(torch.float16)
    if not torch.isfinite(tg).all():
        raise RuntimeError("non-finite teacher Gram (float16 overflow)")
    computed = int(teacher_pending)
    for g, arms in todo.items():
        sweep(
            arms,
            ctx,
            tg,
            out_dir,
            dataset,
            split,
            base=None if g in ("dino", "ctrl") else f"{FROZEN}/{arms[0].cand}",
        )
        computed += len(arms)
    print(f"OBJECT_DIAG_RUN_OK {dataset} arms={done} computed={computed} out={out_dir}")
    return 0


# ----------------------------------------------------------------------------- report
def boot_ci(
    d: np.ndarray, level: float = 0.95, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile image bootstrap of the mean of the finite entries. The index stream
    depends only on (seed, count), so contrasts on the same images share resamples."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng([seed, len(d)]).integers(0, len(d), (n_boot, len(d)))
    means = np.concatenate([d[i].mean(1) for i in np.array_split(idx, 20)])
    a = (1 - level) / 2
    return float(d.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def load_arm(out_dir: str, dataset: str, key: str) -> dict | None:
    p = arm_path(out_dir, dataset, key)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        arm = {k: z[k] for k in z.files}
    arm["config"] = json.loads(str(arm["config"]))
    if arm["config"]["version"] != VERSION:
        raise SystemExit(f"{p}: version {arm['config']['version']} != {VERSION} (re-run with --redo)")
    return arm


def check_arms_paired(arms: dict, smoke: bool) -> None:
    """Every arm of a dataset must be scored on the same images, in the same order, with the same
    eligibility (rule 6: paired on the image list, never on counts)."""
    ref_key = next(iter(arms))
    ref = arms[ref_key]
    for key, a in arms.items():
        if a["config"]["smoke"] != smoke:
            raise SystemExit(
                f"{key}: smoke={a['config']['smoke']} cache in a smoke={smoke} report -- refusing"
            )
        if a["paths"].tolist() != ref["paths"].tolist() or a["rows"].tolist() != ref["rows"].tolist():
            raise SystemExit(f"{key} and {ref_key}: scored on different images -- refusing")
        if not np.array_equal(a["eligible"], ref["eligible"]):
            raise SystemExit(f"{key} and {ref_key}: different eligible sets -- refusing")


def diff_of(arm: dict, stat: str) -> np.ndarray:
    return arm[stat] - arm[stat + "_null"]


def global_share(arm: dict, L: int, n_boot: int = 2000) -> tuple[float, np.ndarray]:
    """Between-image share of the token variance and its image-bootstrap draws (shared index stream)."""
    m = arm["img_mean"].astype(np.float64)
    M = m @ m.T
    n = len(m)
    idx = np.random.default_rng([SEED, n, 3]).integers(0, n, (n_boot, n))

    def share(counts):
        between = L * (counts @ M.diagonal() - counts @ M @ counts / n)
        return between / (between + counts @ arm["ss_within"])

    draws = np.array([share(np.bincount(i, minlength=n).astype(float)) for i in idx])
    return float(share(np.ones(n))), draws


def global_cka(a: dict, t: dict) -> tuple[float, np.ndarray]:
    """Debiased CKA of the image-mean tokens against the teacher's, with half-sample draws (n/2
    images without replacement; a resample with replacement would add duplicate rows to a kernel)."""
    import torch

    def g(arm):
        m = torch.from_numpy(arm["img_mean"].astype(np.float64))
        m = m - m.mean(0, keepdim=True)
        return m @ m.T

    K, L_ = g(a), g(t)
    n = K.shape[0]
    rng = np.random.default_rng([SEED, n, 4])
    draws = []
    for _ in range(1000):
        s = torch.from_numpy(np.sort(rng.choice(n, n // 2, replace=False)))
        draws.append(cka_u(K[s][:, s], L_[s][:, s]))
    return cka_u(K, L_), np.array(draws)


def delta_shares(arm: dict, L: int) -> dict:
    """Shares of the squared token change adapted - frozen: dataset-constant shift, image-global
    shift, local (spatially varying) part; and the mean token cosine to the frozen tokens."""
    dm = arm["d_mean"].astype(np.float64)
    total = float(arm["d_total"].sum())
    const = len(dm) * L * float((dm.mean(0) ** 2).sum())
    local = float(arm["d_local"].sum())
    return {"constant": const / total, "image_global": (total - local - const) / total, "local": local / total,
            "cosine_to_frozen": float(arm["d_cos"].mean()), "mean_sq_change": total / (len(dm) * L)}  # fmt: skip


def _f(t, scale: float = 1.0, nd: int = 2) -> str:
    v, lo, hi = (scale * x for x in t)
    return "n/a" if not np.isfinite(v) else f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def gate(arms: dict) -> dict:
    """(a) noise and (a2) position-only controls: every swap-corrected interval contains 0, at a
    Bonferroni level over the null checks; (b) the object control beats its null on the mask
    statistics and the teacher-projection control on cka_t."""
    nulls = [(c, s) for c in ("ctrl-noise", "ctrl-position") for s in (*MASK_STATS, "cka_t")]
    level = 1 - 0.05 / len(nulls)
    lines, ok = [], True
    for c, s in nulls:
        a = arms.get(f"{c}/ctrl")
        if a is None:
            return {"pass": False, "lines": [f"control {c} not computed"], "level": level}
        v, lo, hi = boot_ci(diff_of(a, s), level)
        good = bool(lo <= 0 <= hi)
        ok &= good
        raw = float(np.nanmean(a[s]))
        lines.append(
            f"{c} {s}: true - swap {v:+.4f} [{lo:+.4f}, {hi:+.4f}] (raw {raw:+.4f}) {'PASS' if good else 'FAIL'}"
        )
    for c, s in [("ctrl-object", s) for s in MASK_STATS] + [("ctrl-teacherproj", "cka_t")]:
        a = arms.get(f"{c}/ctrl")
        if a is None:
            return {"pass": False, "lines": [f"control {c} not computed"], "level": level}
        v, lo, hi = boot_ci(diff_of(a, s))
        good = bool(lo > 0)
        ok &= good
        lines.append(f"{c} {s}: true - swap {v:+.4f} [{lo:+.4f}, {hi:+.4f}] {'PASS' if good else 'FAIL'}")
    weak = arms.get("ctrl-objectw/ctrl")
    if weak is not None:
        seen = [s for s in MASK_STATS if boot_ci(diff_of(weak, s))[1] > 0]
        lines.append(
            f"(informational) weak object control (embedding scale {CTRL_OBJ['ctrl-objectw']}): detected on {seen or 'no statistic'}"
        )
    return {"pass": bool(ok), "lines": lines, "level": level}


def report_dataset(out_dir: str, dataset: str, spec: dict, smoke: bool) -> tuple[dict, str] | None:
    plan = plan_arms(spec, dataset)
    keys = [f"{m['name']}/{cand}" for _, m, cand in plan] + [f"{c}/ctrl" for c in CTRLS]
    arms = {k: a for k in keys if (a := load_arm(out_dir, dataset, k)) is not None}
    missing = [k for k in keys if k not in arms]
    if not arms:
        return None
    check_arms_paired(arms, smoke)
    any_arm = next(iter(arms.values()))
    n, n_el = int(any_arm["config"]["n_images"]), int(any_arm["eligible"].sum())
    grids = {tuple(a["config"]["grid"]) for a in arms.values()}
    if len(grids) != 1:
        raise SystemExit(f"{dataset}: arms on different token grids {sorted(grids)} -- refusing")
    L = int(np.prod(next(iter(grids))))
    if f"{TEACHER}/patch" not in arms:
        raise SystemExit(f"{dataset}: the teacher arm is not computed -- refusing")
    g = gate(arms)
    res: dict = {
        "dataset": dataset,
        "n_images": n,
        "n_eligible": n_el,
        "smoke": smoke,
        "missing": missing,
        "gate": g,
        "arms": {},
        "contrasts": {},
        "delta": {},
    }
    real = [k for k in keys if k in arms and not k.startswith("ctrl-")]
    tkey = f"{TEACHER}/patch"

    def label(k: str) -> str:
        name, cand = k.split("/")
        return LABELS.get(name, name) + ("" if cand == "patch" else f" {cand.split('_')[0]}")

    md = [f"## {dataset}", "",
          f"{n} images of the {any_arm['config']['split']} split, {n_el} eligible (>= 2 classes with >= {MIN_TOK} tokens); "
          f"mean classes per eligible image {float(any_arm['n_classes_img'][any_arm['eligible']].mean()):.2f}. "
          f"Image bootstrap, {N_BOOT} draws, 95 % percentile intervals. Swap null: mean over the masks (teacher tokens) of {N_SWAP} other images.", ""]  # fmt: skip
    if missing:
        md += [f"MISSING arms: {', '.join(missing)}", ""]
    md += ["### Object alignment of the tokens (true mask minus swapped mask)", "",
           "| arm | eta2 true (%) | eta2 swap (%) | eta2 true - swap (pts) | cosine gap true - swap | NN agreement true - swap (pts) |", "|---|---|---|---|---|---|"]  # fmt: skip
    for k in real:
        a = arms[k]
        e, gp, nn = boot_ci(diff_of(a, "eta2")), boot_ci(diff_of(a, "gap")), boot_ci(diff_of(a, "nn"))
        res["arms"][k] = {"eta2_true": float(np.nanmean(a["eta2"])), "eta2_null": float(np.nanmean(a["eta2_null"])),
                          "eta2_diff": e, "gap_true": float(np.nanmean(a["gap"])), "gap_diff": gp,
                          "nn_true": float(np.nanmean(a["nn"])), "nn_diff": nn, "n_nn": int(np.isfinite(diff_of(a, "nn")).sum())}  # fmt: skip
        md.append(
            f"| {label(k)} | {100 * res['arms'][k]['eta2_true']:.2f} | {100 * res['arms'][k]['eta2_null']:.2f} | {_f(e, 100)} | {_f(gp, 1, 4)} | {_f(nn, 100)} |"
        )
    md += ["", "### Adapted minus frozen FLUX, same timestep, paired per image (difference of the swap-corrected statistics)", "",
           "| arm | eta2 raw, no null (pts) | eta2 (pts) | cosine gap | NN agreement (pts) | local CKA to teacher |", "|---|---|---|---|---|---|"]  # fmt: skip
    for k in real:
        name, cand = k.split("/")
        base = arms.get(f"{FROZEN}/{cand}")
        if name == FROZEN or cand == "patch" or base is None:
            continue
        c = {s: boot_ci(diff_of(arms[k], s) - diff_of(base, s)) for s in (*MASK_STATS, "cka_t")}
        c["eta2_raw"] = boot_ci(arms[k]["eta2"] - base["eta2"])
        res["contrasts"][k] = c
        md.append(
            f"| {label(k)} | {_f(c['eta2_raw'], 100)} | {_f(c['eta2'], 100)} | {_f(c['gap'], 1, 4)} | {_f(c['nn'], 100)} | {_f(c['cka_t'], 1, 4)} |"
        )
    md += ["", "### Spatial structure shared with the teacher (debiased linear CKA) and the global side", "",
           "| arm | local CKA true | local CKA swap | local true - swap | global CKA (image means) | between-image variance share (%) |", "|---|---|---|---|---|---|"]  # fmt: skip
    shares, gckas = {}, {}
    for k in real:
        a = arms[k]
        shares[k] = global_share(a, L)
        sh = (shares[k][0], *np.quantile(shares[k][1], [0.025, 0.975]))
        row = {"global_share": sh}
        if k != tkey and tkey in arms:
            gckas[k] = global_cka(a, arms[tkey])
            row.update({"cka_true": float(np.nanmean(a["cka_t"])), "cka_null": float(np.nanmean(a["cka_t_null"])),
                        "cka_diff": boot_ci(diff_of(a, "cka_t")), "global_cka": (gckas[k][0], *np.quantile(gckas[k][1], [0.025, 0.975]))})  # fmt: skip
            md.append(f"| {label(k)} | {row['cka_true']:.4f} | {row['cka_null']:.4f} | {_f(row['cka_diff'], 1, 4)} | "
                      f"{row['global_cka'][0]:.4f} [{row['global_cka'][1]:.4f}, {row['global_cka'][2]:.4f}] | {100 * sh[0]:.2f} [{100 * sh[1]:.2f}, {100 * sh[2]:.2f}] |")  # fmt: skip
        else:
            md.append(
                f"| {label(k)} | — | — | — | — | {100 * sh[0]:.2f} [{100 * sh[1]:.2f}, {100 * sh[2]:.2f}] |"
            )
        res["arms"][k].update(row)
    md += ["", "Global CKA interval: 2.5 / 97.5 % of 1000 half-samples (n/2 images without replacement), wider than a full-sample interval.", "",
           "### What adaptation changed: squared token change adapted - frozen (same noise seed)", "",
           "| arm | mean token cosine to frozen | constant shift (%) | image-global (%) | local (%) | eta2 of the local change, true - swap (pts) | global CKA, adapted - frozen | variance share, adapted - frozen (pts) |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k in real:
        name, cand = k.split("/")
        fk = f"{FROZEN}/{cand}"
        if name == FROZEN or cand == "patch" or fk not in arms:
            continue
        a = arms[k]
        ds = delta_shares(a, L)
        de = boot_ci(a["d_eta2"] - a["d_eta2_null"])
        gd = gckas[k][1] - gckas[fk][1]
        gc = (gckas[k][0] - gckas[fk][0], *np.quantile(gd, [0.025, 0.975]))
        sd = shares[k][1] - shares[fk][1]
        sc = (shares[k][0] - shares[fk][0], *np.quantile(sd, [0.025, 0.975]))
        res["delta"][k] = {**ds, "d_eta2_true": float(np.nanmean(a["d_eta2"])), "d_eta2_null": float(np.nanmean(a["d_eta2_null"])),
                           "d_eta2_diff": de, "global_cka_diff": gc, "global_share_diff": sc}  # fmt: skip
        md.append(f"| {label(k)} | {ds['cosine_to_frozen']:.4f} | {100 * ds['constant']:.1f} | {100 * ds['image_global']:.1f} | {100 * ds['local']:.1f} | "
                  f"{_f(de, 100)} | {_f(gc, 1, 4)} | {_f(sc, 100)} |")  # fmt: skip
    md += ["", f"GATE for {dataset} (null checks at the Bonferroni level {100 * g['level']:.2f} %):"]
    md += [f"- {line}" for line in g["lines"]]
    md.append(
        "- => GATE PASS: the rows above are readable."
        if g["pass"]
        else "- => GATE FAIL: do NOT quote any cell above."
    )
    md.append("")
    return res, "\n".join(md)


def run_report(args) -> int:
    spec = SP.load_spec(args.spec)
    out_dir = os.path.join(args.out_dir, "smoke") if args.smoke else args.out_dir
    datasets = args.datasets or list(DATASETS)
    results, parts = {}, []
    for d in datasets:
        r = report_dataset(out_dir, d, spec, args.smoke)
        if r is None:
            print(f"{d}: no arms under {out_dir}")
            continue
        results[d], md = r
        parts.append(md)
    if not results:
        raise SystemExit(f"no arms under {out_dir}")
    head = [
        "# Object-level representation diagnostic" + (" (SMOKE: never quotable)" if args.smoke else ""), "",
        f"eval/object_diag.py {VERSION}. Tokens of block {BLOCK}; every cached FLUX timestep is listed, none is selected. Masks score the "
        "tokens only. eta2: share of the within-image token variance explained by the mask classes. Cosine gap: mean same-class minus "
        f"mean different-class token cosine. NN agreement: class-balanced share of tokens whose most similar non-adjacent token (> {NN_EXCLUDE} "
        "grid steps) has the same class. Every statistic is given against the swap null (the mask, or the teacher tokens, of another image).",
        "No multiplicity correction across rows. The semantic-only arm appears only when its token cache exists in the spec.", "",
    ]  # fmt: skip
    text = "\n".join(head + parts)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(text)
    print(text)
    bad = [d for d, r in results.items() if not r["gate"]["pass"]]
    print(f"OBJECT_DIAG_REPORT_OK datasets={len(results)} gate_not_passed={len(bad)} missing_arms={sum(len(r['missing']) for r in results.values())} "
          f"-> {out_dir}/report.{{json,md}}")  # fmt: skip
    return 0


# ----------------------------------------------------------------------------- figure
def run_figure(args) -> int:
    """Patch-PCA maps (top-3 within-image principal components as RGB) of the seeded PCA_N eligible
    images: image, token labels, teacher, frozen FLUX and every adapted arm at t = FIG_T."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    spec = SP.load_spec(args.spec)
    out_dir = os.path.join(args.out_dir, "smoke") if args.smoke else args.out_dir
    dataset = args.dataset
    keys = [f"{m['name']}/{cand}" for g, m, cand in plan_arms(spec, dataset) if g in ("teacher", f"t{FIG_T}")]
    arms = {k: a for k in keys if (a := load_arm(out_dir, dataset, k)) is not None}
    if f"{TEACHER}/patch" not in arms or len(arms) < 2:
        raise SystemExit(f"{dataset}: teacher and t{FIG_T} arms needed under {out_dir}, found {sorted(arms)}")
    check_arms_paired(arms, args.smoke)
    ref = arms[f"{TEACHER}/patch"]
    pos = ref["pca_pos"]
    split = ref["config"]["split"]
    images, mask_files = F.list_official_seg_split(dataset, split)
    n_cls = len(F.official_seg_spec(dataset)["classes"])
    side = int(round(math.sqrt(ref["pca_rgb"].shape[1])))
    cols = ["image", "mask (token labels)"] + [LABELS.get(k.split("/")[0], k) for k in arms]
    fig, axes = plt.subplots(
        len(pos), len(cols), figsize=(1.35 * len(cols), 1.35 * len(pos) + 0.3), squeeze=False
    )
    for r, j in enumerate(pos):
        row = int(ref["rows"][j])
        if F.path_key(images[row]) != str(ref["paths"][j]):
            raise SystemExit(f"{dataset}: image list changed since the arms were scored (row {row})")
        axes[r][0].imshow(Image.open(images[row]).convert("RGB"))
        lab = (
            token_labels(SP.load_masks([mask_files[row]]), (side, side), n_cls)[0]
            .reshape(side, side)
            .astype(float)
        )
        lab[lab < 0] = np.nan
        axes[r][1].imshow(lab, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        for c, k in enumerate(arms, start=2):
            if arms[k]["pca_pos"].tolist() != pos.tolist():
                raise SystemExit(f"{k}: PCA images differ from the teacher's")
            p = arms[k]["pca_rgb"][r]
            p = (p - p.min(0)) / np.maximum(p.max(0) - p.min(0), 1e-12)
            axes[r][c].imshow(p.reshape(side, side, 3), interpolation="nearest")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for c, name in enumerate(cols):
        axes[0][c].set_title(name, fontsize=6)
    fig.tight_layout(pad=0.3)
    stem = os.path.join(out_dir, f"{dataset}_patch_pca")
    fig.savefig(stem + ".png", dpi=250)
    fig.savefig(stem + ".pdf")
    print(
        f"OBJECT_DIAG_FIGURE_OK {dataset} images={len(pos)} arms={len(arms)} t={FIG_T} -> {stem}.{{png,pdf}}"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report", "figure"):
        p = sub.add_parser(name)
        p.add_argument("--spec", default=DEFAULT_SPEC)
        p.add_argument("--out-dir", default=OUT_DIR)
        if name == "run":
            p.add_argument("--dataset", required=True, choices=DATASETS)
            p.add_argument("--split", default="test", choices=("val", "test"))
            p.add_argument(
                "--max-images",
                type=int,
                default=None,
                help="strided smoke subset -> <out-dir>/smoke, never quotable",
            )
            p.add_argument(
                "--groups", default=None, help="comma list of teacher,dino,t<timestep>,ctrl (default: all)"
            )
            p.add_argument("--redo", action="store_true")
        else:
            p.add_argument("--smoke", action="store_true")
            if name == "report":
                p.add_argument("--datasets", nargs="*", choices=DATASETS)
            else:
                p.add_argument("--dataset", required=True, choices=DATASETS)
    args = ap.parse_args(argv)
    return {"run": run, "report": run_report, "figure": run_figure}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
