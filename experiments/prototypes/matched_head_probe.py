"""Q2 (9.3): the SAME MLP head on FLUX and DINOv2 features — is the 6ac gap readout-stable?

6x measured MLP +3.5 over LIN on FLUX t260 tokens; the 6ac gap sizes were measured under a
LINEAR probe only. Any published gap size must come from the same nonlinear head on both
arms. Head = the 6x MLP arm verbatim (LN -> 4x-GELU-MLP residual -> LN -> linear; AdamW,
WD 1e-4, cosine, 40 epochs, batch 256; inner-lr selection on {3e-4, 1e-3}), generalized
only in input width D (hidden stays 12288 = the 6x capacity at D=3072 for every arm, so
capacity does not scale with the wider FLUX input).

GATE (rule 2 / reproduction): feeding this head the mean-pooled t260 tokens as length-1
token sequences is numerically the 6x MLP arm (same init order, same seeds, same data
order); it must reproduce results/attentive_probe_resisc45.npz MLP_s* within noise.

Splits: the 6x stratified_split (n_eval=1000, seeds 0/1/2) — NOT the 6ac folds, because
this probe is GPU-trained per fit; deltas are still per-image paired on each seed's shared
eval set. Arms: FLUX ditf 7-t concat 21504d (no PCA — the head handles width), FLUX t260
3072d (gate arm), DINOv2 clsmp 2048d, DINOv2 mean-patch 1024d.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from experiments.prototypes.attentive_probe import (  # noqa: E402  (6x machinery, verbatim)
    BATCH,
    EPOCHS,
    LR_GRID,
    OUTER_SEEDS,
    WD,
    stratified_split,
)
from experiments.prototypes._absorption_harness import RESISC45_BASE, ditf  # noqa: E402

HIDDEN = 12288  # fixed at 6x capacity (4 x 3072) for ALL widths


class MLPHead(nn.Module):
    """6x Readout MLP arm with D as a parameter. For D=3072 the parameter shapes and the
    creation order match 6x exactly, so torch.manual_seed gives identical init."""

    def __init__(self, d: int, n_cls: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, d))
        self.head = nn.Linear(d, n_cls)

    def forward(self, h):
        h = h + self.mlp(self.ln1(h))
        return self.head(self.ln2(h))


def train_eval(X, y, tr, ev, lr, n_cls, device, seed, epochs):
    torch.manual_seed(seed)
    model = MLPHead(X.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    steps = int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(tr))
        for b in range(steps):
            idx = tr[order[b * BATCH : (b + 1) * BATCH]]
            xb = torch.from_numpy(X[idx]).to(device, dtype=torch.float32)
            loss = nn.functional.cross_entropy(model(xb), torch.from_numpy(y[idx]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = np.zeros(len(ev), dtype=np.int8)
    with torch.no_grad():
        for b in range(0, len(ev), BATCH):
            idx = ev[b : b + BATCH]
            xb = torch.from_numpy(X[idx]).to(device, dtype=torch.float32)
            out[b : b + len(idx)] = (model(xb).argmax(1).cpu().numpy() == y[idx]).astype(np.int8)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 3 if args.smoke else EPOCHS

    inv = np.load(RESISC45_BASE)
    y = inv["labels"].astype(np.int64)
    fi = ditf(inv["feats"], inv["mods"])
    dino = np.load("results/dinov2_resisc45_feats_n5000.npz", allow_pickle=True)
    r6ac = np.load("results/dinov2_resisc45_paired.npz", allow_pickle=True)
    assert np.array_equal(r6ac["labels"], y), "6ac labels mismatch"
    # GATE arm input = mean-pooled RAW t260 tokens: exactly what the 6x MLP arm consumed
    # (6x pools raw tokens inside forward; ditf-normalized feats would NOT reproduce it).
    tok = np.load("models/resisc45_tokens_t260/resisc45_flux_3f1c1aa0+42/multistep_train_tokens_oneshot_g1.0_t260.npz",
                  mmap_mode="r")
    assert np.array_equal(np.array(tok["labels"]), y), "token cache labels mismatch"
    t260_raw = np.ascontiguousarray(tok["tokens"][:, 0].mean(axis=1)).astype(np.float32)

    arms = {
        "flux_t260_gate": t260_raw,  # GATE arm: must reproduce 6x MLP_s*
        "flux_7t": np.ascontiguousarray(fi.reshape(len(y), -1)).astype(np.float32),
        "dino_clsmp": np.ascontiguousarray(
            np.concatenate([dino["cls"], dino["mp"]], axis=1)).astype(np.float32),
        "dino_mp": np.ascontiguousarray(dino["mp"]).astype(np.float32),
    }
    ref = np.load("results/attentive_probe_resisc45.npz", allow_pickle=True)
    n_cls = int(y.max()) + 1
    n_eval = 1000 if not args.smoke else 225
    if args.smoke:
        keep = np.linspace(0, len(y) - 1, 900).astype(int)
        y = y[keep]
        arms = {k: v[keep] for k, v in arms.items()}
    for k, v in arms.items():
        print(f"{k}: {v.shape}")

    results = {"labels": y}
    for seed in OUTER_SEEDS:
        tr, ev = stratified_split(y, n_eval, seed)
        itr, iev = stratified_split(y[tr], max(len(tr) // 5, n_cls), 100 + seed)
        for name, X in arms.items():
            inner = {lr: train_eval(X, y, tr[itr], tr[iev], lr, n_cls, device, seed, epochs).mean()
                     for lr in LR_GRID}
            lr_star = max(LR_GRID, key=lambda k: inner[k])
            c = train_eval(X, y, tr, ev, lr_star, n_cls, device, seed, epochs)
            results[f"{name}_s{seed}_correct"] = c
            results[f"{name}_s{seed}_eval_idx"] = ev
            gate = ""
            if name == "flux_t260_gate" and not args.smoke:
                ref_acc = float(ref[f"MLP_s{seed}_correct"].mean())
                gate = f"  [6x MLP ref {ref_acc:.4f}  {'GATE-PASS' if abs(c.mean()-ref_acc)<0.02 else 'GATE-FAIL'}]"
            print(f"seed {seed} {name:<15s} lr*={lr_star} acc {c.mean():.4f}{gate}", flush=True)
        for a, b in (("dino_clsmp", "flux_7t"), ("dino_mp", "flux_7t"), ("flux_7t", "flux_t260_gate")):
            da = results[f"{a}_s{seed}_correct"].astype(int) - results[f"{b}_s{seed}_correct"].astype(int)
            rng = np.random.default_rng(0)
            m = da[rng.integers(0, len(da), (10000, len(da)))].mean(1)
            print(f"  seed {seed} {a}-{b}: {da.mean():+.4f} [{np.quantile(m,0.025):+.4f},{np.quantile(m,0.975):+.4f}]", flush=True)

    if not args.smoke:
        np.savez("results/matched_head_resisc45.npz", **results,
                 protocol=np.array("Q2: 6x MLP head (hidden 12288) on FLUX 7t/t260 and DINOv2 clsmp/mp, 6x splits+selection"))
        print("cached to results/matched_head_resisc45.npz")


if __name__ == "__main__":
    main()
