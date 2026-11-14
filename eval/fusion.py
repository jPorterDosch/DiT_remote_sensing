"""Learned per-slot (timestep) weighting head -- the SatDiFuser-style "global weighting" probe on
POOLED features, as an official-protocol head (eval.probe --head tsweight, eval.sweep block
--with-tsweight).

    slots   S feature vectors per image of one width C: FLUX = the cached timesteps at one block
            (N, T, 3072); DINO = the last-4-block CLS tokens + the last-block patch mean (N, 5, D).
    head    standardize each (slot, channel) on TRAIN; alpha = softmax(theta) over slots (S learned
            scalars); h = sum_s alpha_s x_s; logits = W h + b. Single-label: cross-entropy, argmax.
            Multi-label: BCE, sigmoid > 0.5 (the same rule as protocols.MultiLabelLR).
    recipe  SatDiFuser's: AdamW, 40 epochs, linear warmup 5 epochs then cosine, batch 256.
    select  (lr, weight decay, epoch) on the official VAL split (rule 1), then ONE test evaluation
            with the selected epoch's weights. Test labels never touch training or selection.
    grids   --fusion-grid (FUSION_GRIDS, recorded in every result's config):
            satdifuser  lr {1e-3, 1e-2} x wd {0, 1e-4}, weight decay on ALL parameters. The first
                        run (2026-10-12). wd 0 / 1e-4 gave identical curves: no regularization.
            reg         lr {1e-3, 1e-2} x wd {0, 1e-2, 1e-1, 1, 10}, weight decay on the linear
                        layer ONLY (theta undecayed: decaying the mixing logits pulls the weights
                        toward uniform, i.e. regularizes the fusion away). Fixed a priori
                        2026-10-12 after the satdifuser run picked epochs 3-4 with uniform weights
                        on the hard tasks; identical for every dataset and backbone. AdamW decay is
                        decoupled (shrink factor ~ exp(-lr * wd * steps)): at ~1000 steps
                        (m-forestnet) lr*wd*steps spans 0 .. ~50, i.e. none to heavy shrinkage --
                        the role C in 1e-6..1 plays for the LR head (rule 8).

What it is NOT: SatDiFuser fuses spatial maps before pooling through a learned 384-d projection;
the caches here are spatially pooled, so this matches their weighting idea and training recipe,
not their architecture. Name it that way in write-ups.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from eval import protocols as P

FUSION_EPOCHS = 40
FUSION_WARMUP = 5
FUSION_BATCH = 256
FUSION_SEED = 0
# SatDiFuser: lr 0.01. Their weight decay is NOT verified against their released code (rule 16).
FUSION_GRIDS = {
    "satdifuser": {"lr_grid": (1e-3, 1e-2), "wd_grid": (0.0, 1e-4), "decay_theta": True},
    "reg": {"lr_grid": (1e-3, 1e-2), "wd_grid": (0.0, 1e-2, 1e-1, 1.0, 10.0), "decay_theta": False},
}
FUSION_LR_GRID = FUSION_GRIDS["satdifuser"]["lr_grid"]  # back-compat names (default grid)
FUSION_WD_GRID = FUSION_GRIDS["satdifuser"]["wd_grid"]


def fusion_constants(grid: str = "satdifuser") -> dict:
    """Every constant of the head's protocol, recorded in each result's config."""
    if grid not in FUSION_GRIDS:
        raise SystemExit(f"unknown --fusion-grid {grid!r}; have {sorted(FUSION_GRIDS)}")
    return {
        "grid": grid,
        "epochs": FUSION_EPOCHS,
        "warmup_epochs": FUSION_WARMUP,
        "batch": FUSION_BATCH,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in FUSION_GRIDS[grid].items()},
        "optimizer": "AdamW (decoupled wd), linear warmup then cosine, per-step",
        "standardize": "per (slot, channel), train statistics",
        "seed": FUSION_SEED,
        "select": "val metric over (lr, wd, epoch); first max wins; one test evaluation",
    }


FUSION_CONSTANTS = fusion_constants("satdifuser")


class SlotWeightHead(nn.Module):
    def __init__(self, n_slots: int, dim: int, n_out: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_slots))  # uniform weights at init
        self.lin = nn.Linear(dim, n_out)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.theta, dim=0)

    def forward(self, x):  # x: (B, S, C), already standardized
        return self.lin(torch.einsum("bsc,s->bc", x, self.weights()))


def standardize(Xtr: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    """Per-(slot, channel) z-scoring with TRAIN statistics only."""
    mu = Xtr.mean(0, keepdims=True, dtype=np.float64)
    sd = Xtr.std(0, keepdims=True, dtype=np.float64) + eps
    return [((X - mu) / sd).astype(np.float32) for X in (Xtr, *others)]


def _predict(model, X: torch.Tensor, multilabel: bool, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 1024):
            z = model(X[b : b + 1024].to(device))
            out.append(((torch.sigmoid(z) > 0.5).long() if multilabel else z.argmax(1)).cpu().numpy())
    return np.concatenate(out)


def _train_config(Xtr, ytr, Xva, yva, n_out, lr, wd, multilabel, device, epochs, decay_theta=True):
    """One (lr, wd) run; the val metric after every epoch and the best epoch's weights.
    decay_theta=False: weight decay on the linear layer only (FUSION_GRIDS["reg"])."""
    torch.manual_seed(FUSION_SEED)
    model = SlotWeightHead(Xtr.shape[1], Xtr.shape[2], n_out).to(device)
    if decay_theta:  # the satdifuser grid exactly as first run: one group, every parameter decayed
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(
            [
                {"params": [model.theta], "weight_decay": 0.0},
                {"params": model.lin.parameters(), "weight_decay": wd},
            ],
            lr=lr,
        )
    steps = math.ceil(len(Xtr) / FUSION_BATCH)
    warm, total = FUSION_WARMUP * steps, epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda i: (
            (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm)))
        ),
    )
    yt = torch.from_numpy(ytr).to(device)
    yt = yt.float() if multilabel else yt.long()
    loss_fn = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    rng = np.random.default_rng(FUSION_SEED)
    curve, best = [], (-1.0, -1, None)
    for ep in range(epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(Xtr)))
        for b in range(steps):
            idx = order[b * FUSION_BATCH : (b + 1) * FUSION_BATCH]
            loss = loss_fn(model(Xtr[idx].to(device)), yt[idx.to(device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        v = P.metric(P.per_image(yva, _predict(model, Xva, multilabel, device)))
        curve.append(v)
        if v > best[0]:  # first max wins (earliest epoch)
            best = (v, ep + 1, copy.deepcopy(model.state_dict()))
    return best, curve


def fit_select(
    Xtr, ytr, Xva, yva, n_out: int, device=None, epochs: int = FUSION_EPOCHS, grid: str = "satdifuser"
):
    """Select (lr, wd, epoch) on val. Returns (model at the selected epoch, selection dict,
    table rows: one per (lr, wd) with its best epoch and full val curve)."""
    multilabel = ytr.ndim == 2
    # CPU is the protocol device: every reported tsweight block ran on a CPU node
    # (probe_multiblock.sbatch), and float32 reductions differ between CPU and GPU kernels at
    # the 1e-3 level the first-max (epoch, lr) selection is sensitive to, so a GPU re-probe is
    # not the same protocol. The device and torch thread count are recorded in the selection
    # (-> the block's `compute` stamp; review 2026-10-16).
    device = torch.device(device or "cpu")
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    g = fusion_constants(grid)
    rows, best = [], None
    for lr in g["lr_grid"]:
        for wd in g["wd_grid"]:
            (v, ep, state), curve = _train_config(
                Xtr_t, ytr, Xva_t, yva, n_out, lr, wd, multilabel, device, epochs, g["decay_theta"]
            )
            rows.append(
                {"lr": lr, "wd": wd, "epoch": ep, "val_acc": v, "curve": [round(c, 6) for c in curve]}
            )
            if best is None or v > best[0]:
                best = (v, lr, wd, ep, state)
    v, lr, wd, ep, state = best
    model = SlotWeightHead(Xtr.shape[1], Xtr.shape[2], n_out).to(device)
    model.load_state_dict(state)
    sel = {
        "candidate": "tsweight",
        "fusion_grid": grid,
        "lr": lr,
        "wd": wd,
        "epoch": ep,
        "val_acc": v,
        "slot_weights": [round(float(w), 6) for w in model.weights().detach().cpu()],
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
    }
    return model, sel, rows, device


def tsweight_predict(
    Xtr,
    ytr,
    Xva,
    yva,
    Xte,
    n_out: int,
    slots: list[str],
    device=None,
    epochs=FUSION_EPOCHS,
    grid="satdifuser",
):
    """Standardize on train, select (lr, wd, epoch) on val, predict test -- WITHOUT scoring it
    (test labels are never an input). eval.sweep stores the predictions sealed.
    Returns (test preds, selected dict, val table rows)."""
    if not (Xtr.ndim == Xva.ndim == Xte.ndim == 3) or len(slots) != Xtr.shape[1]:
        raise ValueError(f"tsweight needs (N, S, C) inputs with {len(slots)} named slots, got {Xtr.shape}")
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    model, sel, rows, device = fit_select(Xtr, ytr, Xva, yva, n_out, device, epochs, grid)
    pred = _predict(model, torch.from_numpy(Xte), ytr.ndim == 2, device)
    sel["slots"] = list(slots)
    sel["test_warn"] = 0  # API parity with run_official (no convergence warnings for SGD)
    return pred, sel, rows


def run_tsweight(
    Xtr,
    ytr,
    Xva,
    yva,
    Xte,
    yte,
    n_out: int,
    slots: list[str],
    device=None,
    epochs=FUSION_EPOCHS,
    grid="satdifuser",
):
    """The official-protocol head (eval.probe --head tsweight): tsweight_predict, then ONE test
    evaluation. Returns (test per_image vector, test preds, selected dict, val table rows) -- the
    shape of protocols.run_official, so eval.probe reports and saves it the same way."""
    pred, sel, rows = tsweight_predict(Xtr, ytr, Xva, yva, Xte, n_out, slots, device, epochs, grid)
    return P.per_image(yte, pred), pred, sel, rows


def n_outputs(dataset: str, ytr) -> int:
    """Output width from the REGISTRY (class list / label width), never from test labels."""
    from eval import features as F

    return int(ytr.shape[1]) if ytr.ndim == 2 else len(F.official_classes(dataset))


def flux_slots(X: np.ndarray, timesteps) -> tuple[np.ndarray, list[str]]:
    """(N, T, C) single-block FLUX features -> slots = timesteps."""
    if X.ndim != 3:
        raise ValueError(f"FLUX slots need (N, T, C), got {X.shape}")
    return X, [f"t{int(t)}" for t in timesteps]


def dino_slots(d) -> tuple[np.ndarray, list[str]]:
    """DINO cache -> (N, 5, D): the CLS tokens of the last 4 blocks (cls4, block -4..-1) and the
    last-block patch mean (mp). Refuses banked caches without cls4. These 5 vectors are exactly
    the inputs of DINOv2/v3's own linear eval at use_n_blocks=4, use_avgpool=True
    (create_linear_input in dinov3/eval/linear.py, verified in the shipped hub copy 2026-10-12;
    DINOv3's default n_last_blocks_list is (1,), i.e. cls_b-1 + mp). There the 5 are
    concatenated; here they are mixed with learned softmax weights over DEPTH -- for DINO this
    head is layer weighting, not timestep weighting."""
    if "cls4" not in d.files:
        raise SystemExit("tsweight on DINO needs cls4 (re-extract with eval.extract_dino)")
    c4 = np.asarray(d["cls4"])
    D = np.asarray(d["mp"]).shape[1]
    if c4.shape[1] != 4 * D:
        raise SystemExit(f"cls4 width {c4.shape[1]} != 4 x mp width {D}")
    X = np.concatenate([c4.reshape(len(c4), 4, D), np.asarray(d["mp"])[:, None, :]], axis=1)
    return X, ["cls_b-4", "cls_b-3", "cls_b-2", "cls_b-1", "mp"]
