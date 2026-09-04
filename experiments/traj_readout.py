"""Phase B2 — nonlinear trajectory readout on the multi-timestep EuroSAT cache.

Trains and evaluates ONE arm on the existing (N, K, C) pooled pre-normalization
cache (multistep_train_feats.npz from task='extract') and appends per-fold results
to a CSV. Non-interactive; one arm / one norm / one seed per invocation.

Arms:
  traj    — 1-2 layer Transformer encoder over the K ordered timesteps, learned
            positional encodings on the time axis, mean-pool over K, linear head.
            Tests whether the *ordered trajectory* is readable.
  mlp     — MLP on the single most-separable timestep (t=260; see BEST_T below),
            parameter-count matched to `traj` within ~10%. The content-only baseline.
  shuffle — identical to `traj`, but each sample's time axis is randomly permuted
            (fixed per sample, seeded). Preserves content, destroys ordering — the
            control that isolates whatever `traj` gains from temporal structure.

Preprocessing:
  --norm normalized applies DiTF normalization/channel-filtering offline, mirroring
  plot_multistep_diagnostics.apply_ditf_normalization (zero discard_channels ->
  LayerNorm(C, eps=1e-6, no affine) -> (1+scale)*x + shift from cached adaLN mods ->
  L2 norm). Same pooled-vs-per-token LayerNorm caveat as the diagnostics script.
  Every arm additionally standardizes inputs per-channel using TRAIN-FOLD statistics
  only (no leakage); applied identically to all arms and both norms.

Evaluation: 5-fold stratified CV. One row per fold appended to --out-csv:
  arm, norm, seed, fold, acc, macro_f1, param_count, timestamp

--dry-run loads the cache, prints shapes and BOTH arms' param counts (so traj/mlp
parity is verifiable), and exits without training. CPU-only, finishes in seconds.

--self-test is the permutation-sensitivity gate (CPU-only, seconds, no training):
forwards one real cached sample through a freshly built traj model as-is and with its
time axis permuted, and exits nonzero if the outputs match within 1e-6 — i.e. the
encoder cannot see ordering and the traj-vs-shuffle experiment is meaningless. Run it
before submitting any sweep.

--control is the forward-vs-reversed POSITIVE control, and it answers the question the
self-test cannot: the self-test only proves the encoder is order-sensitive at init, on an
untrained model. --control proves it after training. It builds a binary task in which each
cached trajectory appears twice — once as-is, once time-reversed — so every image sits in
both classes and direction of travel is the only signal that generalizes. traj should
solve it; shuffle cannot (a random permutation erases direction) and pins the floor. If
traj is also at chance, the encoder is blind to ordering and the traj-vs-shuffle result on
the real task is uninformative. Each fold also records what `pos` did during training
(RMS at init vs trained, its size relative to input_proj output, and post-training
permutation sensitivity), which is what distinguishes "no ordering signal in the data"
from "the positional encoding died".

Usage:
    python experiments/traj_readout.py --cache-path <npz> --self-test
    python experiments/traj_readout.py --cache-path <npz> --arm traj --dry-run
    python experiments/traj_readout.py --cache-path <npz> --arm traj \\
        --norm normalized --seed 0 --out-csv results/b2_results_v2.csv
    python experiments/traj_readout.py --cache-path <npz> --control --seed 42
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# --- DiTF normalization config (matches run.py / plot_multistep_diagnostics defaults) ---
DISCARD_CHANNELS = [154, 1446]

# --- Default best single timestep for the `mlp` arm (override with --best-t). ----------
# 260 comes from the ONE-SHOT multistep_diagnostics/separability_table.csv (features
# signal, lda_cv_acc): t=260 -> raw 0.928 (max over raw single timesteps), normalized
# 0.926; t=340 -> raw 0.924, normalized 0.926. That ranking was measured on one-shot
# features — on the inversion cache the most-separable single timestep can differ, which
# would hand the mlp content-baseline a suboptimal t. Pass --best-t with the strongest
# single t from the Tier-1 per-timestep linear probes on the cache being read.
DEFAULT_BEST_T = 260

# --- Architecture (fixed; parity between traj and mlp is asserted at runtime) -----------
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.0  # tiny data; keep training deterministic
MLP_HIDDEN = 214  # chosen so mlp param count matches traj within ~10% (see report)

# --- Training config --------------------------------------------------------------------
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
N_FOLDS = 5

# --- Positional encoding mode (set from --pos-enc / --pos-scale in main) ----------------
# "learned"    — nn.Parameter, trunc_normal_(std=0.02). Original behaviour.
# "sinusoidal" — fixed non-learnable buffer. Two consequences that matter here: it is not
#                in model.parameters(), so WEIGHT_DECAY cannot shrink it, and its RMS is
#                ~0.707 instead of 0.02. The direction control showed the learned table
#                decaying to a permutation-invariant solution in 23/50 folds (perm_delta
#                ~1e-4); a fixed table means the model is never permutation-invariant at
#                any point in training, so that basin does not exist to fall into.
POS_ENC = "learned"
POS_SCALE = 1.0

CSV_FIELDS = [
    "arm",
    "norm",
    "seed",
    "fold",
    "acc",
    "macro_f1",
    # Encoder diagnostics, same four the direction control records. Without these the
    # class task cannot distinguish "ordering carries no signal" from "this particular
    # model stopped being able to read ordering" — the learned table collapsed into a
    # permutation-invariant solution in 23/50 control folds, and a collapsed traj model
    # IS shuffle. Logging them per fold makes each row self-verifying: order-sensitivity
    # and the accuracy gap are measured on the SAME model rather than inferred across
    # experiments. None for arms with no `pos` (mlp).
    "pos_rms_init",
    "pos_rms_final",
    "pos_signal_ratio",
    "perm_delta_final",
    # Which positional encoding produced this row. Load-bearing: the learned-vs-sinusoidal
    # comparison is unrecoverable from the data without it -- a sinusoidal re-run appended
    # to the same CSV has identical (arm, norm, seed, fold) keys otherwise.
    "pos_enc",
    "pos_scale",
    "param_count",
    "timestamp",
]

# The direction control records encoder diagnostics alongside the score, so it gets its
# own schema and its own CSV rather than widening CSV_FIELDS for the main sweep.
CONTROL_CSV_FIELDS = [
    "task",
    "arm",
    "norm",
    "seed",
    "fold",
    "acc",
    "macro_f1",
    "n_train",
    "n_val",
    "pos_rms_init",
    "pos_rms_final",
    "pos_signal_ratio",
    "perm_delta_final",
    # Which positional encoding produced this row. Load-bearing: the learned-vs-sinusoidal
    # comparison is unrecoverable from the data without it -- a sinusoidal re-run appended
    # to the same CSV has identical (arm, norm, seed, fold) keys otherwise.
    "pos_enc",
    "pos_scale",
    "param_count",
    "timestamp",
]


# =======================================================================================
# Preprocessing
# =======================================================================================
def apply_ditf_normalization(feats: np.ndarray, mods: np.ndarray, discard_channels: list[int]) -> np.ndarray:
    """Offline DiTF normalization on pooled features. feats (N, K, C), mods (K, 3, C).

    Copied from plot_multistep_diagnostics.apply_ditf_normalization to keep this script
    standalone. See that module for the pooled-vs-per-token LayerNorm caveat.
    """
    x = feats.astype(np.float64).copy()
    if discard_channels:
        bad = [ch for ch in discard_channels if ch < 0 or ch >= x.shape[-1]]
        if bad:
            raise ValueError(f"discard_channels {bad} out of range for feature dim {x.shape[-1]}")
        x[:, :, discard_channels] = 0.0
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased variance, matches nn.LayerNorm
    x = (x - mu) / np.sqrt(var + 1e-6)
    shift = mods[None, :, 0, :].astype(np.float64)
    scale = mods[None, :, 1, :].astype(np.float64)
    x = (1.0 + scale) * x + shift
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)  # F.normalize
    return x.astype(np.float32)


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Per-channel standardization fit on `train` only (over all leading axes), applied
    to train and every array in `others`. Works for (N, C) and (N, K, C)."""
    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, keepdims=True)
    std = train.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)  # constant channels (e.g. discarded) -> passthrough
    return [((a - mean) / std).astype(np.float32) for a in (train, *others)]


def per_sample_time_shuffle(feats: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute the K (time) axis of each sample. Fixed per sample, seeded.
    Destroys temporal ordering while preserving the multiset of per-timestep vectors."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(feats)
    for i in range(feats.shape[0]):
        out[i] = feats[i, rng.permutation(feats.shape[1])]
    return out


def build_direction_dataset(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-vs-reversed positive control. feats (N, K, C) -> X (2N, K, C), y, groups.

    Every trajectory contributes BOTH copies: sample 2i is the chain as cached (label 0),
    sample 2i+1 is the same chain with its time axis reversed (label 1). Because each
    image appears once in each class, per-image content carries exactly zero information
    about the label — the ONLY thing separating the classes is the direction of travel.

    Consequences that make this a decisive gate on the traj encoder:
      * `shuffle` must sit at chance: a random permutation erases direction.
      * an order-blind `traj` must sit at EXACTLY chance — with pos == 0 the encoder is
        permutation-equivariant and mean-pool makes it permutation-invariant, so the two
        copies produce identical logits and every prediction is a coin flip.
      * so traj >> 0.5 is the only outcome consistent with a working positional encoding.

    `groups` is the originating image index; grouped CV must keep both copies of an image
    in the same fold, otherwise the model can match near-duplicate content across the
    train/val boundary instead of reading direction.

    NOTE: call this AFTER apply_ditf_normalization. That routine indexes mods[k] by slot,
    so normalizing a reversed chain would apply each timestep's modulation to the wrong
    state.
    """
    n, k, _ = feats.shape
    if k < 2:
        raise SystemExit(f"FATAL: direction control needs K >= 2 timesteps, got K={k}")
    x = np.empty((2 * n, *feats.shape[1:]), dtype=feats.dtype)
    x[0::2] = feats  # forward
    x[1::2] = feats[:, ::-1, :]  # reversed (assignment materializes the negative stride)
    y = np.zeros(2 * n, dtype=np.int64)
    y[1::2] = 1
    groups = np.repeat(np.arange(n), 2)
    return x, y, groups


# =======================================================================================
# Models
# =======================================================================================
def sinusoidal_pos_table(k: int, d: int) -> torch.Tensor:
    """Standard transformer sinusoidal table, shape (1, k, d), RMS ~0.707.

    At K=7 the sinusoid's usual selling points (relative distance, extrapolation beyond
    trained lengths) are irrelevant — what matters here is only that the table is FIXED
    and of the same order of magnitude as input_proj's output (measured RMS ~0.74 on this
    cache), so position is a comparable signal rather than a ~2% perturbation.
    """
    position = torch.arange(k, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(k, d, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.unsqueeze(0)


class TrajEncoder(nn.Module):
    """Transformer encoder over K ordered timestep tokens -> mean-pool -> linear head."""

    def __init__(self, c: int, k: int, n_classes: int,
                 pos_enc: str | None = None, pos_scale: float | None = None):
        super().__init__()
        # Explicit args beat the module globals so tests and multi-config processes can
        # build both encoders side by side; None falls back to the globals main() sets.
        pos_enc = POS_ENC if pos_enc is None else pos_enc
        pos_scale = POS_SCALE if pos_scale is None else pos_scale
        if pos_enc not in ("learned", "sinusoidal"):
            # A typo ("sinusoid") must not silently become the learned table.
            raise ValueError(f"unknown pos_enc {pos_enc!r}; expected 'learned' or 'sinusoidal'")
        self.input_proj = nn.Linear(c, D_MODEL)
        # Learned positional encoding. Non-zero init is load-bearing: with pos == 0 the
        # encoder (permutation-equivariant) + mean-pool is exactly permutation-invariant,
        # so traj and shuffle start as the same function and gradients to all shared
        # weights stay identical between the arms — the first sweep produced bit-identical
        # fold accuracies because of this. std=0.02 follows the ViT/BERT learned-pos-enc
        # convention.
        # `pos` is a Parameter when learned and a buffer when sinusoidal. Every diagnostic
        # below (pos_rms / pos_signal_ratio / perm_sensitivity) gates on hasattr(model,
        # "pos") and calls .detach(), both of which hold either way.
        if pos_enc == "sinusoidal":
            self.register_buffer("pos", sinusoidal_pos_table(k, D_MODEL) * pos_scale)
        else:
            self.pos = nn.Parameter(torch.empty(1, k, D_MODEL))
            nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, NHEAD, dim_feedforward=DIM_FF, dropout=DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, NUM_LAYERS)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, K, C)
        h = self.input_proj(x) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class MLPHead(nn.Module):
    """One-hidden-layer MLP on a single timestep's feature vector."""

    def __init__(self, c: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c, MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(MLP_HIDDEN, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C)
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(arm: str, c: int, k: int, n_classes: int,
                pos_enc: str | None = None, pos_scale: float | None = None) -> nn.Module:
    if arm in ("traj", "shuffle"):
        return TrajEncoder(c, k, n_classes, pos_enc=pos_enc, pos_scale=pos_scale)
    if arm == "mlp":
        return MLPHead(c, n_classes)
    raise ValueError(f"unknown arm {arm!r}")


# =======================================================================================
# Self-test (permutation-sensitivity gate)
# =======================================================================================
def run_self_test(feats: np.ndarray, labels: np.ndarray) -> None:
    """Permutation-sensitivity gate. CPU-only, seconds, no training.

    Builds the traj model, forwards one real cached sample as-is (y1) and with its time
    axis permuted (y2). If ||y1 - y2|| is ~0 the encoder cannot distinguish an ordered
    trajectory from a shuffled one, and the traj-vs-shuffle ordering experiment is
    meaningless — exit nonzero so batch scripts abort before burning GPU time.
    """
    _, k, c = feats.shape
    n_classes = int(np.unique(labels).size)
    torch.manual_seed(0)  # deterministic init; the gate must not depend on the draw
    model = build_model("traj", c, k, n_classes)
    model.eval()

    pos = model.pos.detach()
    print(
        f"pos-enc: shape {tuple(pos.shape)}  non-zero: {bool((pos != 0).any())}  "
        f"max|pos| = {pos.abs().max().item():.6f}"
    )

    x = torch.as_tensor(feats[:1], dtype=torch.float32)  # one real sample, (1, K, C)
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(1))
    if torch.equal(perm, torch.arange(k)):  # randperm can draw identity; force a real permutation
        perm = torch.roll(perm, 1)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x[:, perm, :])
    delta = torch.linalg.vector_norm(y1 - y2).item()
    print(f"self-test: time-axis perm {perm.tolist()}  ||y1 - y2|| = {delta:.6e}")
    if delta < 1e-6:
        raise SystemExit(
            f"FATAL: ||y1 - y2|| = {delta:.3e} < 1e-6 — the traj encoder is "
            "permutation-invariant as built (positional information is not reaching the "
            "encoder). The traj-vs-shuffle comparison is meaningless until this is fixed."
        )
    print("self-test OK — traj encoder output is permutation-sensitive.")


# =======================================================================================
# Encoder diagnostics (what the positional encoding is actually doing)
# =======================================================================================
def pos_rms(model: nn.Module) -> float | None:
    """RMS of the learned positional table, or None for arms that have none."""
    if not hasattr(model, "pos"):
        return None
    return float(model.pos.detach().pow(2).mean().sqrt())


@torch.no_grad()
def perm_sensitivity(model: nn.Module, x: np.ndarray, device: torch.device) -> float | None:
    """Mean ||y(x) - y(perm(x))|| on a TRAINED model.

    Same idea as the pre-flight self-test, but measured after training — this is the
    number that says whether the encoder still distinguishes orderings once Adam has had
    200 epochs to do what it likes with `pos`. A value that decays toward 0 means the
    model converged on a permutation-invariant solution regardless of how it was built.
    """
    if not hasattr(model, "pos"):
        return None
    model.eval()
    xt = torch.as_tensor(x[:64], dtype=torch.float32, device=device)
    k = xt.shape[1]
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(1))
    if torch.equal(perm, torch.arange(k)):
        perm = torch.roll(perm, 1)
    y1 = model(xt)
    y2 = model(xt[:, perm.to(device), :])
    return float(torch.linalg.vector_norm(y1 - y2, dim=1).mean())


@torch.no_grad()
def pos_signal_ratio(model: nn.Module, x: np.ndarray, device: torch.device) -> float | None:
    """RMS(pos) / RMS(input_proj(x)) — how large the positional term is relative to the
    content it is added to in `h = input_proj(x) + pos`. If this is ~0.02, position is a
    2% perturbation on the content signal and attention logits barely move."""
    if not hasattr(model, "pos") or not hasattr(model, "input_proj"):
        return None
    model.eval()
    xt = torch.as_tensor(x[:64], dtype=torch.float32, device=device)
    proj = float(model.input_proj(xt).pow(2).mean().sqrt())
    if proj < 1e-12:
        return None
    return float(model.pos.detach().pow(2).mean().sqrt()) / proj


# =======================================================================================
# Train / eval
# =======================================================================================
def train_one(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    device: torch.device,
    seed: int,
) -> nn.Module:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    xt = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    n = xt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE].to(device)
            opt.zero_grad()
            loss = crit(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    return model(xt).argmax(dim=1).cpu().numpy()


def run_cv(arm: str, feats: np.ndarray, labels: np.ndarray, seed: int, device: torch.device) -> list[dict]:
    """5-fold stratified CV for one arm. `feats` is (N, K, C) for traj/shuffle, (N, C) for mlp."""
    n_classes = int(np.unique(labels).size)
    k = feats.shape[1] if feats.ndim == 3 else 1
    c = feats.shape[-1]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    rows: list[dict] = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        x_tr, x_va = standardize(feats[tr], feats[va])
        fold_seed = seed * 1000 + fold
        torch.manual_seed(fold_seed)  # deterministic weight init
        model = build_model(arm, c, k, n_classes)
        rms_init = pos_rms(model)

        train_one(model, x_tr, labels[tr], device, fold_seed)
        y_pred = predict(model, x_va, device)

        acc = accuracy_score(labels[va], y_pred)
        macro_f1 = f1_score(labels[va], y_pred, average="macro")
        rms_final = pos_rms(model)
        delta = perm_sensitivity(model, x_va, device)
        ratio = pos_signal_ratio(model, x_va, device)
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "acc": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "pos_rms_init": None if rms_init is None else round(rms_init, 6),
                "pos_rms_final": None if rms_final is None else round(rms_final, 6),
                "pos_signal_ratio": None if ratio is None else round(ratio, 6),
                "perm_delta_final": None if delta is None else round(delta, 6),
                "param_count": count_params(model),
            }
        )
        bits = [f"  fold {fold}: acc={acc:.4f}  macro_f1={macro_f1:.4f}"]
        if delta is not None:
            # The load-bearing number: if this is ~0 the model is permutation-invariant,
            # so its score says nothing about ordering no matter what the accuracy is.
            bits.append(f"perm_delta {delta:.3e}")
        print("  ".join(bits))
    return rows


# =======================================================================================
# Positive control (forward vs reversed)
# =======================================================================================
def run_control_arm(
    arm: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    best_t_index: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    """Grouped 5-fold CV of one arm on the direction task.

    Folds are drawn over IMAGE indices and then expanded to both copies, so an image's
    forward and reversed versions always land on the same side of the split and each fold
    is exactly 50/50 by construction (chance = 0.5).
    """
    from sklearn.model_selection import KFold

    n_images = int(groups.max()) + 1
    n_classes = 2
    c = x.shape[-1]
    k = x.shape[1]

    # Arm-specific view of the direction dataset.
    if arm == "shuffle":
        arm_x = per_sample_time_shuffle(x, seed)
    elif arm == "mlp":
        arm_x = x[:, best_t_index, :]
    else:
        arm_x = x

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rows: list[dict] = []
    for fold, (g_tr, g_va) in enumerate(kf.split(np.arange(n_images))):
        tr = np.sort(np.concatenate([2 * g_tr, 2 * g_tr + 1]))
        va = np.sort(np.concatenate([2 * g_va, 2 * g_va + 1]))
        x_tr, x_va = standardize(arm_x[tr], arm_x[va])

        fold_seed = seed * 1000 + fold
        torch.manual_seed(fold_seed)
        model = build_model(arm, c, k, n_classes)
        rms_init = pos_rms(model)

        train_one(model, x_tr, y[tr], device, fold_seed)
        y_pred = predict(model, x_va, device)

        acc = accuracy_score(y[va], y_pred)
        macro_f1 = f1_score(y[va], y_pred, average="macro")
        rms_final = pos_rms(model)
        delta = perm_sensitivity(model, x_va, device)
        ratio = pos_signal_ratio(model, x_va, device)

        rows.append(
            {
                "task": "direction",
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "acc": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "n_train": len(tr),
                "n_val": len(va),
                "pos_rms_init": None if rms_init is None else round(rms_init, 6),
                "pos_rms_final": None if rms_final is None else round(rms_final, 6),
                "pos_signal_ratio": None if ratio is None else round(ratio, 6),
                "perm_delta_final": None if delta is None else round(delta, 6),
                "param_count": count_params(model),
            }
        )
        bits = []
        if rms_final is not None:
            bits.append(f"pos_rms {rms_init:.4f}->{rms_final:.4f}")
        if ratio is not None:
            bits.append(f"pos/proj {ratio:.4f}")
        if delta is not None:
            bits.append(f"perm_delta {delta:.3e}")
        extra = ("  " + "  ".join(bits)) if bits else ""
        print(f"  fold {fold}: acc={acc:.4f}  macro_f1={macro_f1:.4f}{extra}")
    return rows


def report_control_verdict(by_arm: dict[str, list[dict]]) -> None:
    """Interpret the control. The verdict rests on traj vs chance and traj vs shuffle."""
    print("\n=== direction control summary (chance = 0.5000) ===")
    means: dict[str, float] = {}
    for arm, rows in by_arm.items():
        accs = np.array([r["acc"] for r in rows], dtype=float)
        means[arm] = float(accs.mean())
        # 95% CI on the fold means; N_FOLDS is small so this is indicative, not exact.
        sem = accs.std(ddof=1) / np.sqrt(len(accs)) if len(accs) > 1 else 0.0
        lo, hi = means[arm] - 1.96 * sem, means[arm] + 1.96 * sem
        print(f"  {arm:<8} acc {means[arm]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    traj_rows = by_arm.get("traj")
    if not traj_rows:
        print("\n  (no traj arm run — verdict needs --control-arms to include traj)")
        return

    traj = means["traj"]
    finals = [r["pos_rms_final"] for r in traj_rows if r["pos_rms_final"] is not None]
    deltas = [r["perm_delta_final"] for r in traj_rows if r["perm_delta_final"] is not None]
    if finals:
        inits = [r["pos_rms_init"] for r in traj_rows if r["pos_rms_init"] is not None]
        print(f"\n  traj pos RMS: {np.mean(inits):.5f} (init) -> {np.mean(finals):.5f} (trained)")
        if deltas:
            print(f"  traj post-training permutation sensitivity: {np.mean(deltas):.3e}")

    print()
    if traj >= 0.70:
        print("  VERDICT: PASS — the traj encoder reads ordering when ordering is the signal.")
        print("  The EuroSAT traj-vs-shuffle null is therefore a statement about the data,")
        print("  not an artifact of an order-blind readout.")
    elif traj >= 0.55:
        print("  VERDICT: WEAK — traj is above chance but far from solving a task whose")
        print("  only signal is direction. The encoder sees ordering poorly; treat the")
        print("  EuroSAT null as underpowered rather than settled.")
    else:
        print("  VERDICT: FAIL — traj is at chance on a task where direction is the ONLY")
        print("  signal. The readout cannot see ordering at all, so the EuroSAT")
        print("  traj-vs-shuffle null says nothing about diffusion trajectories.")
        if finals and np.mean(finals) < 0.005:
            print("  pos RMS collapsed toward zero during training — the model converged on")
            print("  a permutation-invariant solution. Start with exempting `pos` from")
            print("  weight decay and raising its scale relative to input_proj output.")

    if "mlp" in means:
        print(f"\n  note: mlp scored {means['mlp']:.4f}, and above chance is EXPECTED here — it")
        print("  reads a fixed SLOT, and reversing the chain puts a different timestep in that")
        print("  slot. That is single-slot content leakage of direction, not a trajectory read.")
        print("  The load-bearing comparison is traj vs shuffle.")


# =======================================================================================
# CSV
# =======================================================================================
def _check_existing_header(out_csv: str, fields: list[str]) -> None:
    # Refuse to append under a stale header. The schema has grown twice; appending
    # 12-field rows below an 8-column header silently misaligns the diagnostic columns
    # the collapse analysis depends on.
    import csv

    if os.path.getsize(out_csv) == 0:
        return  # empty file: the caller's write_header path will populate it
    with open(out_csv, newline="") as f:
        existing = next(csv.reader(f), None)
    if existing is not None and existing != fields:
        raise SystemExit(
            f"FATAL: {out_csv} has header {existing}, but this script writes {fields}. "
            "Point --out-csv at a fresh file (or migrate the old one) instead of mixing "
            "schemas in place."
        )


def _append(out_csv: str, rows: list[dict], norm: str, fields: list[str]) -> None:
    """Single writer for both sweep and control CSVs. The two schemas previously had
    line-for-line duplicate writers, and every schema fix (empty-file handling, header
    validation, pos_enc stamping) had to land twice -- this diff itself demonstrated the
    drift mechanism the duplication invites."""
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    if not write_header:
        _check_existing_header(out_csv, fields)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            # Row-level value wins over the module global (rows built by run_cv /
            # run_control_arm stamp the encoder actually constructed).
            w.writerow({"pos_enc": POS_ENC, "pos_scale": POS_SCALE,
                        **r, "norm": norm, "timestamp": ts})
    print(f"appended {len(rows)} row(s) to {out_csv}")


def append_rows(out_csv: str, rows: list[dict], norm: str) -> None:
    _append(out_csv, rows, norm, CSV_FIELDS)


def append_control_rows(out_csv: str, rows: list[dict], norm: str) -> None:
    _append(out_csv, rows, norm, CONTROL_CSV_FIELDS)


# =======================================================================================
# Main
# =======================================================================================
def report_param_counts(c: int, k: int, n_classes: int, best_t: int) -> None:
    traj = count_params(build_model("traj", c, k, n_classes))
    mlp = count_params(build_model("mlp", c, k, n_classes))
    diff_pct = 100.0 * abs(traj - mlp) / traj
    print("param counts (parity must be within ~10%):")
    print(f"  traj/shuffle : {traj:,}")
    print(f"  mlp          : {mlp:,}  (t={best_t}, hidden={MLP_HIDDEN})")
    print(f"  |Δ|/traj     : {diff_pct:.2f}%")
    if diff_pct > 10.0:
        raise SystemExit(f"FATAL: arm param counts differ by {diff_pct:.1f}% (> 10%); adjust MLP_HIDDEN.")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase B2 nonlinear trajectory readout (one arm per invocation).")
    p.add_argument(
        "--cache-path",
        required=True,
        help="Path to multistep_train_feats.npz (N, K, C) cache.",
    )
    p.add_argument("--arm", choices=["traj", "mlp", "shuffle"], default="traj")
    p.add_argument("--norm", choices=["raw", "normalized"], default="raw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--best-t",
        type=int,
        default=DEFAULT_BEST_T,
        help="Single timestep the `mlp` content-baseline reads (default %(default)s). Set to the "
        "most-separable t from the Tier-1 per-timestep probes on THIS cache; the default was "
        "measured on the one-shot cache and may not be optimal for inversion.",
    )
    p.add_argument("--out-csv", default="results/b2_results_v2.csv")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load cache, print shapes + both arms' param counts, exit without training (CPU-only).",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Permutation-sensitivity gate: forward one cached sample ordered vs time-permuted "
        "through the traj model; exit nonzero if outputs match (CPU-only, no training).",
    )
    p.add_argument(
        "--control",
        action="store_true",
        help="Run the forward-vs-reversed positive control instead of the class task. Builds a "
        "binary dataset where each trajectory appears both as cached and time-reversed, so "
        "direction is the only signal, then reports whether traj can read it.",
    )
    p.add_argument(
        "--control-arms",
        default="traj,shuffle,mlp",
        help="Comma-separated arms for --control (default %(default)s).",
    )
    p.add_argument(
        "--control-out-csv",
        default="results/direction_control_v2.csv",
        help="CSV for --control rows (default %(default)s). Separate schema from --out-csv.",
    )
    p.add_argument(
        "--pos-enc",
        choices=["learned", "sinusoidal"],
        default="learned",
        help="Positional encoding for the traj/shuffle encoder (default %(default)s). "
        "'sinusoidal' is a fixed buffer: WEIGHT_DECAY cannot reach it and the model is "
        "never permutation-invariant during training, which is the basin the learned "
        "table fell into in 23/50 direction-control folds.",
    )
    p.add_argument(
        "--pos-scale",
        type=float,
        default=1.0,
        help="Multiplier on the sinusoidal table (default %(default)s; ignored when "
        "--pos-enc learned). At 1.0 its RMS ~0.707 is comparable to input_proj's ~0.74.",
    )
    args = p.parse_args()

    # Validate the output-CSV header BEFORE any training: a schema mismatch discovered at
    # append time costs the entire GPU sweep and is then swallowed by the caller's `|| WARN`.
    # Scope to the CSV the SELECTED MODE actually writes -- checking both unconditionally
    # made every invocation (self-test included) die on whichever default file was stale.
    if getattr(args, "control", False):
        path, fields = getattr(args, "control_out_csv", None), CONTROL_CSV_FIELDS
    else:
        path, fields = getattr(args, "out_csv", None), CSV_FIELDS
    if path and os.path.exists(path):
        _check_existing_header(path, fields)

    global POS_ENC, POS_SCALE
    POS_ENC = args.pos_enc
    POS_SCALE = args.pos_scale
    if POS_ENC != "learned":
        print(f"positional encoding: {POS_ENC} (scale {POS_SCALE})")

    d = np.load(args.cache_path)
    feats_raw, labels, mods, timesteps = (
        d["feats"],
        d["labels"],
        d["mods"],
        d["timesteps"],
    )
    n, k, c = feats_raw.shape
    n_classes = int(np.unique(labels).size)
    timesteps = timesteps.tolist()
    print(f"loaded {args.cache_path}")
    print(f"  feats {feats_raw.shape}  labels {labels.shape} ({n_classes} classes)  timesteps {timesteps}")

    best_t = args.best_t
    if best_t not in timesteps:
        raise SystemExit(f"FATAL: --best-t={best_t} not in cached timesteps {timesteps}")
    best_t_index = timesteps.index(best_t)

    report_param_counts(c, k, n_classes, best_t)

    if args.self_test:
        run_self_test(feats_raw, labels)
        return

    if args.dry_run:
        mlp_input = feats_raw[:, best_t_index, :]
        print("dry-run arm input shapes:")
        print(f"  traj/shuffle : {feats_raw.shape}  (K={k} ordered timestep tokens)")
        print(f"  mlp          : {mlp_input.shape}  (single timestep t={best_t}, index {best_t_index})")
        print(f"cuda available: {torch.cuda.is_available()} (dry-run stays on CPU)")
        print("dry-run OK — no training performed.")
        return

    # DiTF normalization (offline) if requested.
    feats = feats_raw.astype(np.float32)
    if args.norm == "normalized":
        feats = apply_ditf_normalization(feats_raw, mods, DISCARD_CHANNELS)
        print(f"applied DiTF normalization (discard_channels={DISCARD_CHANNELS})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Positive control: can the traj encoder read ordering when ordering is ALL there is?
    # Built from `feats` (post-normalization) so each state keeps its own adaLN modulation.
    if args.control:
        arms = [a.strip() for a in args.control_arms.split(",") if a.strip()]
        bad = [a for a in arms if a not in ("traj", "shuffle", "mlp")]
        if bad:
            raise SystemExit(f"FATAL: unknown --control-arms {bad}")

        x_dir, y_dir, groups = build_direction_dataset(feats)
        print(
            f"\ndirection control: {x_dir.shape} from {n} trajectories "
            f"({np.bincount(y_dir).tolist()} per class, chance = 0.5000)"
        )
        print(f"  norm={args.norm}  seed={args.seed}  device={device}  arms={arms}")
        print("  folds are grouped by source image so both copies stay on one side\n")

        by_arm: dict[str, list[dict]] = {}
        for arm in arms:
            print(f"--- control arm={arm} ---")
            rows = run_control_arm(arm, x_dir, y_dir, groups, best_t_index, args.seed, device)
            by_arm[arm] = rows
            append_control_rows(args.control_out_csv, rows, args.norm)
        report_control_verdict(by_arm)
        return

    # Arm-specific input.
    if args.arm == "mlp":
        arm_feats = feats[:, best_t_index, :]  # (N, C) at t=best_t
    elif args.arm == "shuffle":
        arm_feats = per_sample_time_shuffle(feats, args.seed)  # (N, K, C), time axis permuted per sample
    else:  # traj
        arm_feats = feats  # (N, K, C)

    print(f"arm={args.arm}  norm={args.norm}  seed={args.seed}  device={device}  input {arm_feats.shape}")

    rows = run_cv(args.arm, arm_feats, labels, args.seed, device)
    append_rows(args.out_csv, rows, args.norm)

    accs = [r["acc"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    print(f"done: acc {np.mean(accs):.4f}±{np.std(accs):.4f}  macro_f1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
