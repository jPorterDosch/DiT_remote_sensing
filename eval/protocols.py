"""The four probe protocols. Fitting code is copied VERBATIM from the validated prototype
scripts (source named per function) so tests/test_eval_gates.py can demand bit-identical per-image
vectors; change behaviour only in a new, re-gated function, never in place.

  cv        section-13 instrument: 3 seeds x StratifiedKFold(5), in-fold scaler, LR C=0.1
            [+ optional in-fold PCA / nested C]. Per-image correctness averaged over seeds.
  budget    k labels/class x 3 seeds, train drawn stratified, eval = the remaining rows.
  mlp       6x MLP head (hidden 12288), the 6x study's holdout split per seed (3 splits, n_eval=1000), inner-lr selection.
  official  select (candidate x C) on the official val split, ONE test evaluation.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

C = 0.1  # standing a-priori probe C (never selected on eval data)
SEEDS = [0, 1, 2]
C_GRID = (0.01, 0.1, 1.0)


def ci(d, n=10000, seed=0):
    """Image-level bootstrap 95% CI of mean(d) (rule 4). Source: _absorption_harness.ci."""
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ------------------------------------------------------------------------------ cv
def fold_plain(X, y, C, pca, nested, tr, va):
    """Source: dinov2_resisc45._fold_plain."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(X[tr])
        a, b = sc.transform(X[tr]), sc.transform(X[va])
        if pca:
            k = min(pca, a.shape[1], len(tr) - 1)
            if k < a.shape[1]:
                p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
                a, b = p.transform(a), p.transform(b)
        if nested:
            # Selection touches ONLY the training rows of this outer fold (rule 1).
            best = None
            for Ci in nested:
                accs = []
                for itr, iva in StratifiedKFold(3, shuffle=True, random_state=0).split(a, y[tr]):
                    mi = LogisticRegression(C=Ci, max_iter=2000).fit(a[itr], y[tr][itr])
                    accs.append(float((mi.predict(a[iva]) == y[tr][iva]).mean()))
                s = float(np.mean(accs))
                if best is None or s > best[0]:
                    best = (s, Ci)
            C = best[1]
        m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
        pred = m.predict(b)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return va, pred, C, n_warn


def fold_sec13(base, extra, y, tr, va):
    """Section-13 FLUX treatment: PCA-512-protected base + `extra` appended raw.
    Source: dinov2_resisc45._fold_flux."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(base[tr])
        a, b = sc.transform(base[tr]), sc.transform(base[va])
        k = min(512, a.shape[1], len(tr) - 1)
        if k < a.shape[1]:
            p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
            a, b = p.transform(a), p.transform(b)
        if extra is not None:
            s2 = StandardScaler().fit(extra[tr])
            a = np.hstack([a, s2.transform(extra[tr])])
            b = np.hstack([b, s2.transform(extra[va])])
        m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
        pred = m.predict(b)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return va, pred, C, n_warn


def cv_folds(y, seed):
    """Folds depend only on (n, y, seed) -- the pairing premise, fire-tested in tests/test_eval_gates.py."""
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y))


def run_cv(y, fold_fn, fold_args, n_jobs=7):
    """Source: dinov2_resisc45.run_arm (printing moved to the caller).
    Returns (per-image correctness averaged over seeds, n_warn, C picks)."""
    out = np.zeros(len(y))
    warn_tot, picks = 0, []
    for s in SEEDS:
        jobs = cv_folds(y, s)
        for va, pred, Cp, nw in Parallel(n_jobs=n_jobs)(
            delayed(fold_fn)(*fold_args, tr, va) for tr, va in jobs
        ):
            out[va] += pred == y[va]
            warn_tot += nw
            picks.append(Cp)
    return out / len(SEEDS), warn_tot, picks


# -------------------------------------------------------------------------- budget
BUDGETS = (10, 25, 50, 100)


def budget_draw(y, per_class, seed):
    """Source: label_budget_curves.draw."""
    rng = np.random.default_rng(seed)
    tr = np.sort(
        np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in np.unique(y)])
    )
    ev = np.setdiff1d(np.arange(len(y)), tr)
    return tr, ev


def budget_fit_sec13(base, others, y, tr, ev):
    """Source: label_budget_curves.fit_flux."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(base[tr])
        a, b = sc.transform(base[tr]), sc.transform(base[ev])
        k = min(512, a.shape[1], len(tr) - 1)
        if k < a.shape[1]:
            p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(a)
            a, b = p.transform(a), p.transform(b)
        s2 = StandardScaler().fit(others[tr])
        a = np.hstack([a, s2.transform(others[tr])])
        b = np.hstack([b, s2.transform(others[ev])])
        m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr])
        nw = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return (m.predict(b) == y[ev]).astype(np.int8), nw


def budget_fit_plain(X, y, tr, ev):
    """Source: label_budget_curves.fit_plain."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        nw = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return (m.predict(sc.transform(X[ev])) == y[ev]).astype(np.int8), nw


# ----------------------------------------------------------------------------- mlp
MLP_HIDDEN = 12288  # 6x capacity (4 x 3072), fixed for ALL input widths
MLP_EPOCHS = 40
MLP_BATCH = 256
MLP_WD = 1e-4
MLP_LR_GRID = (3e-4, 1e-3)
MLP_N_EVAL = 1000


class MLPHead(nn.Module):
    """Source: matched_head_probe.MLPHead (= the 6x MLP arm, D as a parameter)."""

    def __init__(self, d: int, n_cls: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, MLP_HIDDEN), nn.GELU(), nn.Linear(MLP_HIDDEN, d))
        self.head = nn.Linear(d, n_cls)

    def forward(self, h):
        h = h + self.mlp(self.ln1(h))
        return self.head(self.ln2(h))


def mlp_train_eval(X, y, tr, ev, lr, n_cls, device, seed, epochs=MLP_EPOCHS):
    """Source: matched_head_probe.train_eval."""
    torch.manual_seed(seed)
    model = MLPHead(X.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=MLP_WD)
    steps = int(np.ceil(len(tr) / MLP_BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(tr))
        for b in range(steps):
            idx = tr[order[b * MLP_BATCH : (b + 1) * MLP_BATCH]]
            xb = torch.from_numpy(X[idx]).to(device, dtype=torch.float32)
            loss = nn.functional.cross_entropy(model(xb), torch.from_numpy(y[idx]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = np.zeros(len(ev), dtype=np.int8)
    with torch.no_grad():
        for b in range(0, len(ev), MLP_BATCH):
            idx = ev[b : b + MLP_BATCH]
            xb = torch.from_numpy(X[idx]).to(device, dtype=torch.float32)
            out[b : b + len(idx)] = (model(xb).argmax(1).cpu().numpy() == y[idx]).astype(np.int8)
    return out


def mlp_split(y, n_eval, seed):
    """Source: attentive_probe.stratified_split."""
    rng = np.random.default_rng(seed)
    ev = []
    per = n_eval // len(np.unique(y))
    for cls in np.unique(y):
        pos = np.flatnonzero(y == cls)
        ev.append(rng.choice(pos, size=per, replace=False))
    ev = np.sort(np.concatenate(ev))
    tr = np.setdiff1d(np.arange(len(y)), ev)
    return tr, ev


def run_mlp_seed(X, y, seed, device, epochs=MLP_EPOCHS, n_eval=MLP_N_EVAL):
    """One outer seed of matched_head_probe.main: inner-lr selection on train rows only."""
    n_cls = int(y.max()) + 1
    tr, ev = mlp_split(y, n_eval, seed)
    itr, iev = mlp_split(y[tr], max(len(tr) // 5, n_cls), 100 + seed)
    inner = {
        lr: mlp_train_eval(X, y, tr[itr], tr[iev], lr, n_cls, device, seed, epochs).mean()
        for lr in MLP_LR_GRID
    }
    lr_star = max(MLP_LR_GRID, key=lambda k: inner[k])
    return mlp_train_eval(X, y, tr, ev, lr_star, n_cls, device, seed, epochs), ev, lr_star


# ------------------------------------------------------------------------ official
OFFICIAL_MAX_ITER = 3000


def official_fit_eval(Xtr, ytr, Xev, Cv):
    """Source: m_eurosat_probe.fit_eval / dinov2_m_eurosat.fit_eval (identical)."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=Cv, max_iter=OFFICIAL_MAX_ITER).fit(Xtr, ytr)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
    return clf.predict(Xev), n_warn


def run_official(cands, ytr, yva, yte):
    """Select (candidate x C) on val, then ONE test evaluation. Source: the selection loop
    shared by m_eurosat_probe.main and dinov2_m_eurosat.main (first max wins, candidate
    order then C order). cands: {name: (Xtr, Xva, Xte)} in selection order.
    Returns (test correctness int8, test preds, selected dict, val table rows)."""
    best, table = None, []
    for name, (Atr, Ava, _) in cands.items():
        sc = StandardScaler().fit(Atr)
        A, V = sc.transform(Atr), sc.transform(Ava)
        for Cv in C_GRID:
            pred, n_warn = official_fit_eval(A, ytr, V, Cv)
            acc = float((pred == yva).mean())
            table.append({"candidate": name, "C": Cv, "val_acc": acc, "conv_warnings": n_warn})
            if best is None or acc > best[0]:
                best = (acc, name, Cv, n_warn)
    val_acc, name, Cv, sel_warn = best
    Atr, _, Ate = cands[name]
    sc = StandardScaler().fit(Atr)
    pred, n_warn = official_fit_eval(sc.transform(Atr), ytr, sc.transform(Ate), Cv)
    correct = (pred == yte).astype(np.int8)
    sel = {"candidate": name, "C": Cv, "val_acc": val_acc, "sel_warn": sel_warn, "test_warn": n_warn}
    return correct, pred, sel, table
