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

Usage:
    python experiments/traj_readout.py --cache-path <npz> --self-test
    python experiments/traj_readout.py --cache-path <npz> --arm traj --dry-run
    python experiments/traj_readout.py --cache-path <npz> --arm traj \\
        --norm normalized --seed 0 --out-csv results/b2_results_v2.csv
"""

from __future__ import annotations

import argparse
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

CSV_FIELDS = [
    "arm",
    "norm",
    "seed",
    "fold",
    "acc",
    "macro_f1",
    "param_count",
    "timestamp",
]


# =======================================================================================
# Preprocessing
# =======================================================================================
def apply_ditf_normalization(
    feats: np.ndarray, mods: np.ndarray, discard_channels: list[int]
) -> np.ndarray:
    """Offline DiTF normalization on pooled features. feats (N, K, C), mods (K, 3, C).

    Copied from plot_multistep_diagnostics.apply_ditf_normalization to keep this script
    standalone. See that module for the pooled-vs-per-token LayerNorm caveat.
    """
    x = feats.astype(np.float64).copy()
    if discard_channels:
        bad = [ch for ch in discard_channels if ch < 0 or ch >= x.shape[-1]]
        if bad:
            raise ValueError(
                f"discard_channels {bad} out of range for feature dim {x.shape[-1]}"
            )
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
    std = np.where(
        std < 1e-6, 1.0, std
    )  # constant channels (e.g. discarded) -> passthrough
    return [((a - mean) / std).astype(np.float32) for a in (train, *others)]


def per_sample_time_shuffle(feats: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute the K (time) axis of each sample. Fixed per sample, seeded.
    Destroys temporal ordering while preserving the multiset of per-timestep vectors."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(feats)
    for i in range(feats.shape[0]):
        out[i] = feats[i, rng.permutation(feats.shape[1])]
    return out


# =======================================================================================
# Models
# =======================================================================================
class TrajEncoder(nn.Module):
    """Transformer encoder over K ordered timestep tokens -> mean-pool -> linear head."""

    def __init__(self, c: int, k: int, n_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(c, D_MODEL)
        # Learned positional encoding. Non-zero init is load-bearing: with pos == 0 the
        # encoder (permutation-equivariant) + mean-pool is exactly permutation-invariant,
        # so traj and shuffle start as the same function and gradients to all shared
        # weights stay identical between the arms — the first sweep produced bit-identical
        # fold accuracies because of this. std=0.02 follows the ViT/BERT learned-pos-enc
        # convention.
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


def build_model(arm: str, c: int, k: int, n_classes: int) -> nn.Module:
    if arm in ("traj", "shuffle"):
        return TrajEncoder(c, k, n_classes)
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
    if torch.equal(
        perm, torch.arange(k)
    ):  # randperm can draw identity; force a real permutation
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


def run_cv(
    arm: str, feats: np.ndarray, labels: np.ndarray, seed: int, device: torch.device
) -> list[dict]:
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
        train_one(model, x_tr, labels[tr], device, fold_seed)
        y_pred = predict(model, x_va, device)

        acc = accuracy_score(labels[va], y_pred)
        macro_f1 = f1_score(labels[va], y_pred, average="macro")
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "acc": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "param_count": count_params(model),
            }
        )
        print(f"  fold {fold}: acc={acc:.4f}  macro_f1={macro_f1:.4f}")
    return rows


# =======================================================================================
# CSV
# =======================================================================================
def append_rows(out_csv: str, rows: list[dict], norm: str) -> None:
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({**r, "norm": norm, "timestamp": ts})
    print(f"appended {len(rows)} row(s) to {out_csv}")


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
        raise SystemExit(
            f"FATAL: arm param counts differ by {diff_pct:.1f}% (> 10%); adjust MLP_HIDDEN."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase B2 nonlinear trajectory readout (one arm per invocation)."
    )
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
    args = p.parse_args()

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
    print(
        f"  feats {feats_raw.shape}  labels {labels.shape} ({n_classes} classes)  timesteps {timesteps}"
    )

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
        print(
            f"  mlp          : {mlp_input.shape}  (single timestep t={best_t}, index {best_t_index})"
        )
        print(f"cuda available: {torch.cuda.is_available()} (dry-run stays on CPU)")
        print("dry-run OK — no training performed.")
        return

    # DiTF normalization (offline) if requested.
    feats = feats_raw.astype(np.float32)
    if args.norm == "normalized":
        feats = apply_ditf_normalization(feats_raw, mods, DISCARD_CHANNELS)
        print(f"applied DiTF normalization (discard_channels={DISCARD_CHANNELS})")

    # Arm-specific input.
    if args.arm == "mlp":
        arm_feats = feats[:, best_t_index, :]  # (N, C) at t=best_t
    elif args.arm == "shuffle":
        arm_feats = per_sample_time_shuffle(
            feats, args.seed
        )  # (N, K, C), time axis permuted per sample
    else:  # traj
        arm_feats = feats  # (N, K, C)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"arm={args.arm}  norm={args.norm}  seed={args.seed}  device={device}  input {arm_feats.shape}"
    )

    rows = run_cv(args.arm, arm_feats, labels, args.seed, device)
    append_rows(args.out_csv, rows, args.norm)

    accs = [r["acc"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    print(
        f"done: acc {np.mean(accs):.4f}±{np.std(accs):.4f}  macro_f1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}"
    )


if __name__ == "__main__":
    main()
