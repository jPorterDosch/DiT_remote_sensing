"""The four probe protocols. Fitting code is copied VERBATIM from the validated prototype
scripts (source named per function) so tests/test_eval_gates.py can demand bit-identical per-image
vectors; change behaviour only in a new, re-gated function, never in place.

  cv        section-13 instrument: 3 seeds x StratifiedKFold(5), in-fold scaler, LR C=0.1
            [+ optional in-fold PCA / nested C]. Per-image correctness averaged over seeds.
  budget    k labels/class x 3 seeds, train drawn stratified, eval = the remaining rows.
  mlp       6x MLP head (hidden 12288), the 6x study's holdout split per seed (3 splits, n_eval=1000), inner-lr selection.
  official  select (candidate x C) on the official val split, ONE test evaluation. Multi-hot
            labels (m-bigearthnet): one-vs-rest LR, metric micro-F1 instead of top-1.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

C = 0.1  # standing a-priori probe C (never selected on eval data)
SEEDS = [0, 1, 2]
C_GRID = (0.01, 0.1, 1.0)  # default official grid: the banked gates reproduce under it
# Extended downward 2026-10-07: 5/6 FLUX sweeps and 9/12 DINO arms selected the 0.01 edge and
# grid-mean val was still rising there. Opt-in (--c-grid), never in place (gates, test-use history).
C_GRID_WIDE = (1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0)
# 1e-6/1e-5 added the same day: the forestnet k=56 wide-grid smoke still peaked at 1e-4 (grid-mean
# val 45.1 at C=1 rising to 53.1 at 1e-4), while DINO underfits there -- the grid must bracket both arms.


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


class MultiLabelLR(OneVsRestClassifier):
    """One binary LR per label, predicting p > 0.5 (SatDiFuser utils/val_logger.py:
    sigmoid(logits) > 0.5). Not OneVsRestClassifier.predict: sklearn takes the threshold from
    estimators_[0] alone, and when label 0 is constant in train that is a constant predictor,
    so EVERY label gets thresholded at decision_function > 0.5 (tests/test_geobench.py)."""

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.int64)


OVR_FIT_THREADS = 1  # protocol constant (2026-10-15): BLAS threads of EVERY one-vs-rest label fit


def official_lr(Cv, y):
    """Single-label y: the prototype LR, unchanged. Multi-hot y: MultiLabelLR, its one-vs-rest
    fits spread over OVR_N_JOBS workers (default 1), each fit at OVR_FIT_THREADS BLAS threads
    (_CountingLR). Each label's fit is independent and lbfgs is deterministic for a fixed thread
    count, so predictions are identical for any worker count
    (tests/test_geobench.py::test_multilabel_n_jobs_identical); only wall time changes."""
    if y.ndim != 2:
        return LogisticRegression(C=Cv, max_iter=OFFICIAL_MAX_ITER)
    lr = _CountingLR(C=Cv, max_iter=OFFICIAL_MAX_ITER)  # counts its own warnings (worker-safe)
    return MultiLabelLR(lr, n_jobs=ovr_n_jobs())


_OVR_PIN: tuple[int, int] | None = None  # (requested, pinned) OVR_N_JOBS of the last budget_ovr_jobs call


def ovr_n_jobs() -> int:
    """OVR_N_JOBS resolved ONCE to a positive worker count (joblib semantics: -1 = all CPUs of
    this process's affinity set, -2 = all but one, ...; 0 is invalid). It sizes the one-vs-rest
    pool only; each fit's BLAS threads are the constant OVR_FIT_THREADS (_CountingLR)."""
    raw = int(os.environ.get("OVR_N_JOBS", "1"))
    if raw == 0:
        raise ValueError("OVR_N_JOBS=0 is invalid (joblib n_jobs)")
    cpus = len(os.sched_getaffinity(0))
    return raw if raw > 0 else max(1, cpus + 1 + raw)


def _cgroup_limit_files(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> list[str]:
    """This process's OWN cgroup memory-limit files (v1 `memory` controller, v2 unified), then
    the root ones. The root limit alone is unlimited on cgroup v1 nodes even inside a job."""
    files = []
    try:
        for line in open(proc_cgroup):
            _, ctrls, path = line.rstrip("\n").split(":", 2)
            if "memory" in ctrls.split(","):
                files.append(os.path.join(root, "memory", path.lstrip("/"), "memory.limit_in_bytes"))
            elif ctrls == "":
                files.append(os.path.join(root, path.lstrip("/"), "memory.max"))
    except (OSError, ValueError):
        pass
    return files + [os.path.join(root, "memory.max"), os.path.join(root, "memory", "memory.limit_in_bytes")]


def _mem_limit_bytes() -> int | None:
    """This job's memory limit, smallest-scope source first: Slurm --mem (SLURM_MEM_PER_NODE, MB),
    Slurm --mem-per-cpu (SLURM_MEM_PER_CPU x CPUs on the node), this process's cgroup, then
    physical RAM (MemTotal). None only if none is readable (review 2026-10-11: --mem-per-cpu and
    cgroup-v1 jobs used to fall through to None, disabling the budget)."""
    v = os.environ.get("SLURM_MEM_PER_NODE")
    if v and v.isdigit():
        return int(v) << 20
    v = os.environ.get("SLURM_MEM_PER_CPU")
    n = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit() and n and n.isdigit():
        return (int(v) * int(n)) << 20
    for f in _cgroup_limit_files():
        try:
            raw = open(f).read().strip()
            if raw.isdigit() and int(raw) < (1 << 50):
                return int(raw)
        except OSError:
            pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) << 10
    except (OSError, ValueError, IndexError):
        pass
    return None


def _rss_bytes() -> int:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) << 10
    except OSError:
        pass
    return 0


def budget_ovr_jobs(cands, y) -> int:
    """Size the one-vs-rest worker pool to the job's MEMORY before any parallel fit, and pin the
    result in OVR_N_JOBS (read by ovr_n_jobs; the per-fit BLAS threads are the constant
    OVR_FIT_THREADS, so the pool size only changes wall time). Measured 2026-10-08: m-bigearthnet DINO (largest X 3.3 GB) used
    9.7 GB serially but 82 GB with 18 workers -- each worker effectively holds a copy of X -- and
    was OOM-killed at 64 GB. Worst-case model: per worker 1.3 x the largest train matrix (float64)
    + 0.5 GB; the parent keeps its current RSS + 2 such matrices; use 85% of the limit.
    Single-label (no workers) leaves OVR_N_JOBS untouched; an unreadable limit refuses. The pin
    is written to OVR_N_JOBS, so a later call in the same process (matrix run, next cell) that
    still sees the previous pin budgets from the ORIGINAL request; a changed OVR_N_JOBS is a new
    request."""
    global _OVR_PIN
    cur = ovr_n_jobs()
    want = _OVR_PIN[0] if _OVR_PIN is not None and cur == _OVR_PIN[1] else cur
    if np.asarray(y).ndim != 2 or want <= 1:
        return want
    limit = _mem_limit_bytes()
    if limit is None:  # never run several workers blind: that is how 2026-10-08 OOMed
        raise SystemExit(f"OVR_N_JOBS={want} but no memory limit is readable -- set OVR_N_JOBS=1 or --mem")
    biggest = max(Xs[0].shape[0] * int(np.prod(Xs[0].shape[1:])) * 8 for Xs in cands.values())
    room = 0.85 * limit - _rss_bytes() - 2 * biggest
    fit = max(1, int(room // (1.3 * biggest + (512 << 20))))
    n = min(want, fit)
    if n < want:
        print(
            f"OVR workers {want} -> {n}: memory budget (limit {limit >> 30} GB, largest X {biggest / 2**30:.1f} GB)"
        )
    os.environ["OVR_N_JOBS"] = str(n)
    _OVR_PIN = (want, n)
    return n


def trivial_baseline(y) -> float:
    """Best val metric a CONSTANT predictor can reach -- the floor a positive control must beat.
    Single-label: the majority-class rate. Multi-label (micro-F1): the optimal constant set is
    the top-m most prevalent labels for some m (micro-F1 of a constant set grows by adding labels
    in prevalence order until it peaks), so maximise over m = 0..L exactly."""
    y = np.asarray(y)
    if y.ndim == 1:
        return float(np.bincount(y).max() / len(y))
    order = np.argsort(-y.mean(0), kind="stable")
    best = 0.0
    for m in range(1, y.shape[1] + 1):
        pred = np.zeros_like(y)
        pred[:, order[:m]] = 1
        best = max(best, metric(per_image(y, pred)))
    return best


def per_image(y, pred):
    """The per-image vector a result file stores: correctness (N,) int8, or for multi-hot
    labels the (N, 3) true-positive / false-positive / false-negative counts (micro-F1 is not
    a per-image mean, but it is a function of these sums)."""
    if y.ndim == 1:
        return (pred == y).astype(np.int8)
    return np.stack([(pred * y).sum(1), (pred * (1 - y)).sum(1), ((1 - pred) * y).sum(1)], 1)


def _micro_f1(sums):
    tp, fp, fn = np.moveaxis(sums, -1, 0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def metric(v) -> float:
    """top-1 from a correctness vector, micro-F1 (SatDiFuser: average="micro") from counts."""
    return float(v.mean()) if v.ndim == 1 else float(_micro_f1(v.sum(0)))


def paired_delta(va, vb, n=10000, seed=0):
    """metric(B) - metric(A) and its image-level bootstrap 95% CI (rule 4). Correctness
    vectors: mean(vb - va) with ci(), as before; counts: micro-F1 of every resample."""
    if va.ndim == 1:
        d = vb - va
        return (float(d.mean()), *ci(d, n, seed))
    idx = np.random.default_rng(seed).integers(0, len(va), (n, len(va)))
    ds = np.concatenate([_micro_f1(vb[i].sum(1)) - _micro_f1(va[i].sum(1)) for i in np.array_split(idx, 10)])
    return metric(vb) - metric(va), float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))


class _CountingLR(LogisticRegression):
    """LogisticRegression that records its OWN ConvergenceWarnings during fit (n_conv_warn_).
    Used for the one-vs-rest label fits: with OVR_N_JOBS > 1 they run in worker processes whose
    warnings never reach the parent, but this attribute travels back with the pickled fitted
    estimator. It counts every warning sklearn raises -- max_iter AND early lbfgs stops
    (status != 0, e.g. ABNORMAL_TERMINATION_IN_LNSRCH) -- exactly what the serial warning count
    saw. Predictions are those of the parent class."""

    def fit(self, X, y, sample_weight=None):
        # Thread policy (protocol constant, 2026-10-15): EVERY one-vs-rest fit runs with
        # OVR_FIT_THREADS BLAS threads -- serial or in a worker, whatever the allocation. Until
        # 2026-10-15 a worker's share was cpus // OVR_N_JOBS read from the WORKER's environment:
        # joblib keeps spawned workers across Parallel calls, so a re-pinned OVR_N_JOBS
        # (budget_ovr_jobs, next cell) need not reach them, and the share followed --cpus-per-task
        # and --mem. lbfgs on the wide designs moves ~1e-3 with the thread count
        # (eval.matrix.blas_threads), so compared arms could sit at different operating points
        # (CLAUDE.md rule 8). A constant makes OVR_N_JOBS wall-time only. Also protects against
        # joblib's explicitly exported OMP_NUM_THREADS (16 workers x 16 threads on 16 cores).
        with (
            threadpool_limits(limits=OVR_FIT_THREADS, user_api="blas"),
            warnings.catch_warnings(record=True) as wl,
        ):
            warnings.simplefilter("always", ConvergenceWarning)
            super().fit(X, y, sample_weight=sample_weight)
        self.n_conv_warn_ = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
        for w in wl:  # only the counted ConvergenceWarnings are consumed; everything else stays visible
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return self


def _n_not_converged(clf, wl) -> int:
    """ConvergenceWarnings of a fitted official classifier (rule 8: measured, not assumed).
    Single-label: the warnings caught in this process, exactly as before. Multi-label: the sum
    of each label fit's own count (_CountingLR), which survives worker processes. Constant-label
    columns get sklearn's _ConstantPredictor (no fit) and contribute 0."""
    if isinstance(clf, MultiLabelLR):
        total = 0
        for e in clf.estimators_:
            if isinstance(e, _CountingLR):
                total += e.n_conv_warn_
            elif isinstance(e, _ConstantPredictor):
                continue
            else:  # a plain LR here would report "converged" with no measurement (rule 8)
                raise TypeError(f"{type(e).__name__} in estimators_ does not count ConvergenceWarnings")
        return total
    return sum(issubclass(w.category, ConvergenceWarning) for w in wl)


def official_fit_eval(Xtr, ytr, Xev, Cv):
    """Source: m_eurosat_probe.fit_eval / dinov2_m_eurosat.fit_eval (identical)."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = official_lr(Cv, ytr).fit(Xtr, ytr)
    return clf.predict(Xev), _n_not_converged(clf, wl)


SHARE_MIN_BYTES = 64 << 20  # below this, joblib's own per-call dump is cheap


@contextlib.contextmanager
def sigterm_as_exit():
    """Within the block, SIGTERM raises SystemExit(128+15) so `finally` clauses run (Slurm's
    --time / scancel send SIGTERM before SIGKILL; Python's default action skips `finally`).
    Main thread only (signal.signal's constraint); elsewhere a no-op. The one copy of this logic:
    eval.sweep.run_block uses it too."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    prev = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, prev if prev is not None else signal.SIG_DFL)


@contextlib.contextmanager
def _shared_for_workers(make_A, y):
    """Multi-label fits with OVR_N_JOBS > 1: back the (large) design matrix by ONE on-disk memmap
    for all its C fits. joblib otherwise re-dumps X to its temp folder for EVERY one-vs-rest
    Parallel call (each C), ~4.4 GB per call for m-bigearthnet's 27648-d concat -- hundreds of
    GB of scratch writes per block; a memmap is handed to workers by reference. Same values,
    so identical fits. Serial / single-label / small inputs: A unchanged. Takes a FACTORY
    (make_A) because contextlib's context-manager object keeps its call arguments alive -- a
    passed array could never be freed, a factory's result lives only in this frame."""
    A = make_A()
    folder = os.environ.get("JOBLIB_TEMP_FOLDER")
    # No JOBLIB_TEMP_FOLDER (the probe scripts set it to Lustre): do not share -- TMPDIR is node-local
    # /tmp in jobs and must never receive bulky data (scratch_env.sh).
    if y.ndim != 2 or ovr_n_jobs() <= 1 or A.nbytes < SHARE_MIN_BYTES:
        yield A
        return
    # Every probe sbatch sets the folder; a worker pool without it re-dumps X per C into node-local /tmp.
    if not folder:
        raise SystemExit(
            f"OVR_N_JOBS={ovr_n_jobs()} with a {A.nbytes / 2**30:.1f} GB design matrix but JOBLIB_TEMP_FOLDER is unset: "
            "set it to a Lustre path (scratch_env.sh) or OVR_N_JOBS=1"
        )
    import tempfile

    # SIGTERM -> SystemExit BEFORE the file exists, so a kill during the multi-GB save also
    # removes it. (SIGKILL/OOM orphans: the probe scripts sweep ovr_X_* older than 7 days.)
    path = None
    with sigterm_as_exit():
        try:
            os.makedirs(folder, exist_ok=True)
            fd, path = tempfile.mkstemp(
                prefix=f"ovr_X_{os.environ.get('SLURM_JOB_ID', 'nojob')}_", suffix=".npy", dir=folder
            )
            os.close(fd)
            np.save(path, np.ascontiguousarray(A))
            del A  # the memmap replaces it: do not keep a second multi-GB copy resident
            yield np.load(path, mmap_mode="r")
        finally:
            try:
                if path is not None:
                    os.remove(path)
            except OSError:
                pass


def run_official(cands, ytr, yva, yte, c_grid=C_GRID):
    """Worker pool sized to memory first (budget_ovr_jobs). Select (candidate x C) on val by metric() (the "val_acc" key holds micro-F1 for
    multi-hot labels), then ONE test evaluation. Source: the selection loop
    shared by m_eurosat_probe.main and dinov2_m_eurosat.main (first max wins, candidate
    order then C order). cands: {name: (Xtr, Xva, Xte)} in selection order.
    Returns (test per_image vector, test preds, selected dict, val table rows)."""
    budget_ovr_jobs(cands, ytr)
    best, table = None, []
    for name, (Atr, Ava, _) in cands.items():
        sc = StandardScaler().fit(Atr)
        V = sc.transform(Ava)
        with _shared_for_workers(lambda: sc.transform(Atr), ytr) as A:
            for Cv in c_grid:
                pred, n_warn = official_fit_eval(A, ytr, V, Cv)
                acc = metric(per_image(yva, pred))
                table.append({"candidate": name, "C": Cv, "val_acc": acc, "conv_warnings": n_warn})
                if best is None or acc > best[0]:
                    best = (acc, name, Cv, n_warn)
            del A  # release the (deleted) memmap before the next candidate writes its own
    val_acc, name, Cv, sel_warn = best
    Atr, _, Ate = cands[name]
    sc = StandardScaler().fit(Atr)
    with _shared_for_workers(lambda: sc.transform(Atr), ytr) as A:
        pred, n_warn = official_fit_eval(A, ytr, sc.transform(Ate), Cv)
        del A
    correct = per_image(yte, pred)
    sel = {"candidate": name, "C": Cv, "val_acc": val_acc, "sel_warn": sel_warn, "test_warn": n_warn}
    return correct, pred, sel, table


def official_all_cells(cands, ytr, yva, yte, c_grid=C_GRID):
    """Worker pool sized to memory first (budget_ovr_jobs). Every (candidate x C) cell of the official protocol, for sweeps: val accuracy AND the
    test predictions, from ONE fit per cell. Same math as run_official (scaler fit on train
    per candidate, LR C, max_iter 3000); run_official refits the selected cell on the same
    train rows, which is deterministic, so its test vector equals this cell's (checked by
    the sweep gate on ISAAC). Test predictions are returned for SEALED storage only --
    eval/sweep.py selects on val and opens exactly one of them."""
    budget_ovr_jobs(cands, ytr)
    rows, test_preds = [], {}
    for name, (Atr, Ava, Ate) in cands.items():
        sc = StandardScaler().fit(Atr)
        V, T = sc.transform(Ava), sc.transform(Ate)
        with _shared_for_workers(lambda: sc.transform(Atr), ytr) as A:
            for Cv in c_grid:
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter("always", ConvergenceWarning)
                    clf = official_lr(Cv, ytr).fit(A, ytr)
                n_warn = _n_not_converged(clf, wl)
                rows.append(
                    {
                        "candidate": name,
                        "C": Cv,
                        "val_acc": metric(per_image(yva, clf.predict(V))),
                        "conv_warnings": n_warn,
                    }
                )
                test_preds[(name, Cv)] = clf.predict(T)
            del A  # release the (deleted) memmap before the next candidate writes its own
    return rows, test_preds
