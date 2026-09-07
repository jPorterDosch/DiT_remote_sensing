"""traj vs shuffle on the ONE substrate where the ablation is admissible.

Gate result (timestep_removal.py, EuroSAT n=500): per-timestep ZCA whitening in a shared PCA
basis takes MLP timestep-identification to 0.176 vs chance 0.143 -- unrecoverable -- while
class accuracy survives (0.873 vs 0.940 raw). Per-timestep z-scoring, which the section-1
CORRECTION proposed, leaves 0.646 and is NOT admissible; INLP leaves 0.739.

On whitened features the readout can no longer re-sort shuffled tokens by content, so
`per_sample_time_shuffle` finally removes what it claims to. This is the first honest run of
the section-1 experiment.

Whitening is fit on the TRAIN fold only, inside the CV loop. Both arms see identically
whitened features; they differ only in whether the time axis is permuted per sample.

FINDINGS (2026-09-01, RESEARCH_NOTES 1). The substrate positive control FAILED: the encoder
cannot solve forward-vs-reversed on whitened features (0.576 vs chance 0.5), so the observed
traj = shuffle tie (0.7973 both) is UNINTERPRETABLE -- recorded as such, not as a null. The
ordering question was subsequently closed by proof, not experiment (section 1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

PCA_DIM, SEEDS, EPOCHS, D_MODEL = 256, [0, 1, 2], 120, 128
CACHE = (
    "models/paired_500_eurosat_inv_redo/eurosat_flux_5cad8aad+42/multistep_train_feats_inversion_g1.0_n50.npz"
)


def whiten_fit(tr):
    st = []
    for i in range(tr.shape[1]):
        z = tr[:, i, :]
        mu = z.mean(0)
        cov = np.cov((z - mu).T) + 1e-4 * np.eye(z.shape[1])
        w, V = np.linalg.eigh(cov)
        st.append((mu, V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T))
    return st


def whiten(st, x):
    return np.stack([(x[:, i, :] - mu) @ W for i, (mu, W) in enumerate(st)], axis=1)


class Traj(nn.Module):
    def __init__(self, c, k, n_cls):
        super().__init__()
        self.proj = nn.Linear(c, D_MODEL)
        self.pos = nn.Parameter(torch.empty(1, k, D_MODEL))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True, dropout=0.1), 2
        )
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        return self.head(self.enc(self.proj(x) + self.pos).mean(1))


def run(x, y, shuffle, seed):
    accs, deltas, rms = [], [], []
    for s in SEEDS:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(x[:, 0], y):
            p = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
            n, k, c = x.shape
            p.fit(x[tr].reshape(-1, c))
            a = p.transform(x[tr].reshape(-1, c)).reshape(len(tr), k, -1)
            b = p.transform(x[va].reshape(-1, c)).reshape(len(va), k, -1)
            st = whiten_fit(a)
            a, b = whiten(st, a), whiten(st, b)
            if shuffle:
                rng = np.random.default_rng(seed * 1000 + s)
                for arr in (a, b):
                    for i in range(len(arr)):
                        arr[i] = arr[i, rng.permutation(k)]
            torch.manual_seed(seed * 100 + s)
            m = Traj(a.shape[2], k, len(np.unique(y))).cuda()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            xa = torch.tensor(a, dtype=torch.float32).cuda()
            ya = torch.tensor(y[tr]).cuda()
            for _ in range(EPOCHS):
                m.train()
                opt.zero_grad()
                nn.functional.cross_entropy(m(xa), ya).backward()
                opt.step()
            m.eval()
            xb = torch.tensor(b, dtype=torch.float32).cuda()
            with torch.no_grad():
                pr = m(xb).argmax(1).cpu().numpy()
                # PERMUTATION-SENSITIVITY GATE (section 1's own lesson, applied to this run).
                # If the trained encoder is permutation-INVARIANT, traj and shuffle are
                # literally the same function and a tie is vacuous -- which is exactly the
                # "24/25 folds exactly tied" signature that started this whole thread.
                # Measured on the TRAINED model, per fold, not at init.
                g = torch.Generator(device="cpu").manual_seed(0)
                perm = torch.randperm(xb.shape[1], generator=g).cuda()
                delta = (m(xb) - m(xb[:, perm, :])).abs().mean().item()
                pos_rms = m.pos.detach().pow(2).mean().sqrt().item()
            accs.append(float((pr == y[va]).mean()))
            deltas.append(delta)
            rms.append(pos_rms)
    return np.array(accs), np.array(deltas), np.array(rms)


def direction_gate(x, y, seed=42):
    """POSITIVE CONTROL FOR THE SUBSTRATE (section 1's design, applied to whitened features).

    Each image contributes BOTH its trajectory and its time-reversed trajectory, labelled
    0/1. Content is identical between the classes by construction, so direction of travel is
    the only signal -- and on whitened features content cannot leak the timestep either.
    If this encoder cannot solve THIS at well above 0.5, then it cannot use order on this
    substrate at all, and the traj-vs-shuffle tie below says nothing about ordering.
    """
    accs = []
    for s_ in SEEDS:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s_).split(x[:, 0], y):
            p_ = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
            n, k, c = x.shape
            p_.fit(x[tr].reshape(-1, c))
            a = p_.transform(x[tr].reshape(-1, c)).reshape(len(tr), k, -1)
            b = p_.transform(x[va].reshape(-1, c)).reshape(len(va), k, -1)
            st = whiten_fit(a)
            a, b = whiten(st, a), whiten(st, b)
            # forward + reversed copies of every sample
            a2 = np.concatenate([a, a[:, ::-1, :]])
            ya = np.r_[np.zeros(len(a)), np.ones(len(a))]
            b2 = np.concatenate([b, b[:, ::-1, :]])
            yb = np.r_[np.zeros(len(b)), np.ones(len(b))]
            torch.manual_seed(seed * 100 + s_)
            m = Traj(a2.shape[2], k, 2).cuda()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            xa = torch.tensor(a2, dtype=torch.float32).cuda()
            ya_t = torch.tensor(ya, dtype=torch.long).cuda()
            for _ in range(EPOCHS):
                m.train()
                opt.zero_grad()
                nn.functional.cross_entropy(m(xa), ya_t).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                pr = m(torch.tensor(b2, dtype=torch.float32).cuda()).argmax(1).cpu().numpy()
            accs.append(float((pr == yb).mean()))
    return np.array(accs)


if __name__ == "__main__":
    d = np.load(CACHE)
    x, y = d["feats"].astype(np.float64), d["labels"]
    g = direction_gate(x, y)
    print(
        f"SUBSTRATE GATE (forward vs reversed on whitened features): {g.mean():.4f} "
        f"+- {g.std():.4f}  (chance 0.5)"
    )
    if g.mean() < 0.7:
        print(
            "  GATE FAIL: the encoder cannot read order on this substrate. The "
            "traj-vs-shuffle comparison below is UNINTERPRETABLE -- it cannot distinguish "
            "'ordering carries no class signal' from 'this readout cannot use ordering here'."
        )
    else:
        print(
            "  GATE PASS: the encoder CAN read order on whitened features, so a "
            "traj-vs-shuffle tie is a statement about class signal, not about the readout."
        )

    t, td, trms = run(x, y, False, 42)
    s, sd, srms = run(x, y, True, 42)
    print(f"GATE perm_delta (trained model): traj {td.mean():.3e}  shuffle {sd.mean():.3e}")
    print(f"GATE pos_rms:                    traj {trms.mean():.4f}  shuffle {srms.mean():.4f}")
    if td.mean() < 1e-3:
        print(
            "GATE FAIL: the traj encoder is permutation-INVARIANT -- traj and shuffle are "
            "the same function, so any comparison below is vacuous."
        )
    else:
        print("GATE PASS: the trained encoder is order-sensitive; the comparison is readable.")
    diff = t - s
    boot = np.random.default_rng(0).integers(0, len(diff), (10000, len(diff)))
    lo, hi = np.quantile(diff[boot].mean(1), [0.025, 0.975])
    print("\nWHITENED substrate (t-ID MLP 0.176 vs chance 0.143 -- ablation admissible)")
    print(f"  traj     {t.mean():.4f} +- {t.std():.4f}")
    print(f"  shuffle  {s.mean():.4f} +- {s.std():.4f}")
    print(
        f"  traj - shuffle {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
        f"{'SIGNIF' if lo > 0 or hi < 0 else 'ns'}"
    )
