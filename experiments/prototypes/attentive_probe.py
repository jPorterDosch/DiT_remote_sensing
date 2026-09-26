"""6x: attentive probing (V-JEPA-style) vs capacity-matched MLP vs linear, on per-token
block-28 features. See RESEARCH_NOTES 6x pre-registration; protocol is fixed there —
this script implements it and nothing else.

Arms differ ONLY as pre-registered:
  LIN:  mean-pool -> LN -> Linear
  MLP:  mean-pool -> h + MLP(LN(h)) -> Linear(LN(h))        (no attention)
  ATT:  learnable query -> MHA(q, tokens) residual -> h + MLP(LN(h)) -> Linear(LN(h))

lr selected per (arm, seed) on an inner 3200/800 split (rule 1), retrain on 4000, ONE
final-epoch eval on the held-out 1000. Per-image correctness vectors cached (rule 12).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

D = 3072
HEADS = 12
MLP_HIDDEN = 4 * D
EPOCHS = 40
BATCH = 256
WD = 1e-4
LR_GRID = (3e-4, 1e-3)
OUTER_SEEDS = (0, 1, 2)


class Readout(nn.Module):
    def __init__(self, arm: str, n_cls: int):
        super().__init__()
        self.arm = arm
        if arm == "ATT":
            self.q = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        if arm in ("MLP", "ATT"):
            self.mlp = nn.Sequential(nn.Linear(D, MLP_HIDDEN), nn.GELU(), nn.Linear(MLP_HIDDEN, D))
        self.head = nn.Linear(D, n_cls)

    def forward(self, tokens):  # (B, L, D) fp32
        if self.arm == "ATT":
            q = self.q.expand(tokens.shape[0], -1, -1)
            a, _ = self.attn(q, tokens, tokens, need_weights=False)
            h = q + a  # (B, 1, D)
            h = h.squeeze(1)
        else:
            h = tokens.mean(dim=1)  # (B, D)
            if self.arm == "LIN":
                return self.head(self.ln2(h))
        h = h + self.mlp(self.ln1(h))
        return self.head(self.ln2(h))


def train_eval(arm, X, y, tr, ev, lr, n_cls, device, seed):
    torch.manual_seed(seed)
    model = Readout(arm, n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    steps_per_epoch = int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        order = rng.permutation(len(tr))
        for b in range(steps_per_epoch):
            idx = tr[order[b * BATCH : (b + 1) * BATCH]]
            xb = torch.from_numpy(X[idx]).to(device, dtype=torch.float32)
            yb = torch.from_numpy(y[idx]).to(device)
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    correct = np.zeros(len(ev), dtype=np.int8)
    with torch.no_grad():
        for b in range(0, len(ev), BATCH):
            idx = ev[b : b + BATCH]
            xb = torch.from_numpy(X[idx]).to(device, dtype=torch.float32)
            pred = model(xb).argmax(1).cpu().numpy()
            correct[b : b + len(idx)] = (pred == y[idx]).astype(np.int8)
    return correct


def stratified_split(y, n_eval, seed):
    rng = np.random.default_rng(seed)
    ev = []
    per = n_eval // len(np.unique(y))
    for cls in np.unique(y):
        pos = np.flatnonzero(y == cls)
        ev.append(rng.choice(pos, size=per, replace=False))
    ev = np.sort(np.concatenate(ev))
    tr = np.setdiff1d(np.arange(len(y)), ev)
    return tr, ev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", default="models/resisc45_tokens_t260/")
    p.add_argument("--glob", default="**/*_tokens_*.npz")
    p.add_argument("--out", default="results/attentive_probe_resisc45.npz")
    p.add_argument("--smoke", action="store_true", help="tiny run on whatever cache is given")
    args = p.parse_args()

    import glob as _g

    hits = sorted(_g.glob(os.path.join(args.tokens, args.glob), recursive=True))
    assert len(hits) == 1, f"expected exactly one token cache, got {hits}"
    d = np.load(hits[0], mmap_mode="r")
    toks = d["tokens"]  # (N, S, L, C)
    y = np.array(d["labels"])
    assert toks.ndim == 4 and toks.shape[-1] == D, toks.shape
    # ONE sequential disk read into RAM as fp16 (~7.9 GB at n=5000): mmap fancy-indexing
    # per batch would re-read ~0.8 GB/step from disk. fp32 cast happens per-batch on GPU.
    X = np.ascontiguousarray(toks[:, 0]).astype(np.float16)  # (N, L, C)
    n_cls = int(y.max()) + 1
    global EPOCHS
    n_eval = 1000 if not args.smoke else max(len(y) // 5, n_cls)
    if args.smoke:
        EPOCHS = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"tokens {toks.shape} classes {n_cls} device {device} eval_n {n_eval}")

    results: dict = {"labels": y, "subset_indices": np.array(d["subset_indices"])}
    summary = []
    for seed in OUTER_SEEDS:
        tr, ev = stratified_split(y, n_eval, seed)
        # inner lr selection on 3200/800 of the outer-train (rule 1: never touches ev)
        itr, iev = stratified_split(y[tr], max(len(tr) // 5, n_cls), 100 + seed)
        for arm in ("LIN", "MLP", "ATT"):
            inner = {}
            for lr in LR_GRID:
                c = train_eval(arm, X, y, tr[itr], tr[iev], lr, n_cls, device, seed)
                inner[lr] = c.mean()
            lr_star = max(LR_GRID, key=lambda k: inner[k])
            correct = train_eval(arm, X, y, tr, ev, lr_star, n_cls, device, seed)
            acc = float(correct.mean())
            results[f"{arm}_s{seed}_correct"] = correct
            results[f"{arm}_s{seed}_eval_idx"] = ev
            summary.append((arm, seed, lr_star, acc))
            print(f"seed {seed} {arm}: lr*={lr_star} (inner {inner}) eval acc {acc:.4f}", flush=True)
        # paired deltas on the SHARED eval set of this seed
        for a, b in (("ATT", "MLP"), ("ATT", "LIN"), ("MLP", "LIN")):
            da = results[f"{a}_s{seed}_correct"].astype(int) - results[f"{b}_s{seed}_correct"].astype(int)
            rng = np.random.default_rng(0)
            m = da[rng.integers(0, len(da), (10000, len(da)))].mean(1)
            print(f"  seed {seed} {a}-{b}: {da.mean():+.4f} [{np.quantile(m, 0.025):+.4f},{np.quantile(m, 0.975):+.4f}]", flush=True)

    os.makedirs("results", exist_ok=True)
    np.savez(args.out, **results,
             summary=np.array([f"{a},{s},{lr},{acc:.6f}" for a, s, lr, acc in summary]),
             protocol=np.array("6x pre-registration: 3 arms, inner-lr selection, final-epoch eval, 3 seeds"))
    print(f"cached to {args.out}")


if __name__ == "__main__":
    main()
