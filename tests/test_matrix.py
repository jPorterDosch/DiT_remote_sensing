"""eval/matrix.py: nested label subsets (stratified / multi-hot), the per-cell protocol on a
synthetic separable problem, cell caching with the changed-inputs guard, and the report with
explicit missing cells. CPU, no feature files: load_cands is monkeypatched."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import matrix as M  # noqa: E402

FR = (0.01, 0.1, 1.0)


def test_label_subsets_nested_and_stratified():
    y = np.repeat(np.arange(5), [300, 200, 100, 40, 7])
    s = M.label_subsets(y, FR, seed=3)
    for f in FR:
        counts = np.bincount(y[s[f]], minlength=5)
        assert (counts == [math.ceil(f * n) for n in (300, 200, 100, 40, 7)]).all()  # per class ceil
        assert (np.diff(s[f]) > 0).all()  # sorted, unique
    assert set(s[0.01]) <= set(s[0.1]) <= set(s[1.0]) and len(s[1.0]) == len(y)
    assert not np.array_equal(s[0.1], M.label_subsets(y, FR, seed=4)[0.1])  # seed matters
    assert np.array_equal(s[0.1], M.label_subsets(y, FR, seed=3)[0.1])  # deterministic
    with pytest.raises(ValueError):
        M.label_subsets(y, (0.0, 1.0), 0)


def test_label_subsets_multihot_is_random_prefix():
    y = (np.random.default_rng(0).random((500, 6)) < 0.3).astype(int)
    s = M.label_subsets(y, FR, seed=0)
    assert len(s[0.01]) == 5 and len(s[0.1]) == 50 and len(s[1.0]) == 500
    assert set(s[0.01]) <= set(s[0.1])


def test_blas_threads_scales_with_rows_up_to_a_fixed_cap():
    t = [M.blas_threads(n) for n in (1, 69, 646, 6464, 10**6)]
    assert t[0] == 1 and t[1] == 1 and t == sorted(t) and t[-1] == M.BLAS_THREADS_CAP
    assert t[-1] != len(os.sched_getaffinity(0)) or M.BLAS_THREADS_CAP == len(os.sched_getaffinity(0))


def _synthetic(n_tr=400, n_ev=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(4, d)) * 3

    def make(n):
        y = rng.integers(0, 4, n)
        return centres[y] + rng.normal(size=(n, d)), y

    Xtr, ytr = make(n_tr)
    Xva, yva = make(n_ev)
    Xte, yte = make(n_ev)
    noise = {k: rng.normal(size=v.shape) for k, v in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    cands = {"good": (Xtr, Xva, Xte), "junk": (noise["tr"], noise["va"], noise["te"])}
    return cands, ytr, yva, yte


def test_fit_cell_selects_the_informative_candidate_and_shrinks_with_labels():
    cands, ytr, yva, yte = _synthetic()
    full = np.arange(len(ytr))
    correct, pred, sel, table, compute = M.fit_cell(cands, full, ytr, yva, yte, (0.01, 1.0))
    assert compute["blas_threads"] == M.blas_threads(len(full)) and compute["ovr_n_jobs"] >= 1
    assert sel["candidate"] == "good" and correct.mean() > 0.9 and len(table) == 4
    small = M.label_subsets(ytr, (0.02, 1.0), 0)[0.02]  # 8 rows
    c2, _, sel2, _, _ = M.fit_cell(cands, small, ytr, yva, yte, (0.01, 1.0))
    assert sel2["candidate"] == "good" and c2.mean() <= correct.mean() + 1e-9
    assert correct.dtype == np.int8 and correct.shape == (len(yte),)


def test_fit_cell_multilabel_workers_identical(monkeypatch):
    """Multi-hot cell: the one-vs-rest pool (OVR_N_JOBS) is sized by protocols.budget_ovr_jobs
    inside fit_cell and 2 workers give the same cell as 1 (lbfgs is deterministic)."""
    from eval import protocols as P

    rng = np.random.default_rng(1)
    w = rng.normal(size=(12, 5))
    X = {k: rng.normal(size=(n, 12)) for k, n in (("tr", 300), ("va", 120), ("te", 120))}
    y = {k: ((v @ w + 0.5 * rng.normal(size=(len(v), 5))) > 0).astype(int) for k, v in X.items()}
    cands = {"good": (X["tr"], X["va"], X["te"]), "junk": tuple(rng.normal(size=v.shape) for v in X.values())}
    tr = np.arange(200)
    monkeypatch.setattr(P, "_mem_limit_bytes", lambda: 64 << 30)
    monkeypatch.setenv("OVR_N_JOBS", "1")
    serial = M.fit_cell(cands, tr, y["tr"], y["va"], y["te"], (0.1, 1.0))[:4]
    monkeypatch.setenv("OVR_N_JOBS", "2")
    par = M.fit_cell(cands, tr, y["tr"], y["va"], y["te"], (0.1, 1.0))[:4]
    assert os.environ["OVR_N_JOBS"] == "2"  # pinned by budget_ovr_jobs (64 GB budget keeps 2)
    assert np.array_equal(serial[0], par[0]) and np.array_equal(serial[1], par[1])
    assert serial[2] == par[2] and serial[3] == par[3] and serial[2]["candidate"] == "good"
    assert serial[1].shape == (120, 5)


def _spec(tmp_path):
    return {
        "fractions": list(FR),
        "seeds": [0, 1],
        "c_grid": [0.01, 1.0],
        "datasets": ["m_eurosat", "m_forestnet"],
        "models": [
            {"name": "T", "kind": "dino", "features": "x_{split}.npz", "teacher": None},
            {"name": "A", "kind": "dino", "features": "x_{split}.npz", "teacher": "T"},
        ],
    }


def test_run_caches_cells_guards_inputs_and_reports_missing(tmp_path, monkeypatch, capsys):
    cands, ytr, yva, yte = _synthetic()
    paths = [f"p{i}" for i in range(len(yte))]
    sha = {"x": "aaa"}
    calls = []

    def fake_load(model, dataset):
        calls.append((model["name"], dataset))
        return cands, ytr, yva, yte, paths, ["x"]

    monkeypatch.setattr(M, "load_cands", fake_load)
    monkeypatch.setattr(M, "inputs_sha", lambda files: dict(sha))
    monkeypatch.setattr(M, "input_files", lambda m, d: ["x"])
    # m_forestnet has no features (missing), m_eurosat has them
    monkeypatch.setattr(
        M, "features_present", lambda m, d: (d == "m_eurosat", "" if d == "m_eurosat" else "nope")
    )
    spec = _spec(tmp_path)
    out = str(tmp_path / "res")
    log = M.run(spec, spec["models"], spec["datasets"], out, redo=False)
    done = [r for r in log if r["status"] == "done"]
    # 2 models x (2 fractions x 2 seeds + one 100 % cell) = 2 x 5
    assert len(done) == 10 and sum(r["status"] == "missing" for r in log) == 2
    assert len(calls) == 2  # features loaded once per (model, dataset)
    cell = M.cell_path(out, "A", "m_eurosat", 0.01, 1)
    d = np.load(cell, allow_pickle=True)
    assert json.loads(str(d["inputs"])) == sha and int(d["n_train"]) == len(M.label_subsets(ytr, FR, 1)[0.01])
    # second run: everything cached, nothing recomputed
    calls.clear()
    log2 = M.run(spec, spec["models"], spec["datasets"], out, redo=False)
    assert all(r["status"] in ("cached", "missing") for r in log2) and calls == []
    # changed inputs: refused without --redo, recomputed with it
    sha["x"] = "bbb"
    with pytest.raises(SystemExit, match="inputs changed"):
        M.run(spec, spec["models"], ["m_eurosat"], out, redo=False)
    log3 = M.run(spec, spec["models"], ["m_eurosat"], out, redo=True)
    assert sum(r["status"] == "done" for r in log3) == 10
    # changed protocol (C grid) with unchanged inputs: refused too (guard must be able to fail)
    spec2 = {**spec, "c_grid": [0.01, 1.0, 10.0]}
    with pytest.raises(SystemExit, match="protocol/config changed"):
        M.run(spec2, spec2["models"], ["m_eurosat"], out, redo=False)
    # the report re-validates: a stale cell (inputs changed) or a spec edit is refused there too
    with pytest.raises(SystemExit, match="protocol/config differs"):
        M.report(spec2, out)
    sha["x"] = "ccc"
    with pytest.raises(SystemExit, match="inputs changed"):
        M.report(spec, out)
    sha["x"] = "bbb"
    # inputs no longer on disk: the cell is reported with inputs_absent (sha kept in the cell)
    monkeypatch.setattr(M, "features_present", lambda m, d: (False, "gone"))
    M.report(spec, out)
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["summary"]["A"]["m_eurosat"]["inputs_absent"] is True
    # teacher / student test vectors must be the same images in the same order
    monkeypatch.setattr(M, "features_present", lambda m, d: (True, ""))
    cell_t = M.cell_path(out, "T", "m_eurosat", 1.0, 0)
    orig = dict(np.load(cell_t, allow_pickle=True))
    d = dict(orig)
    d["paths"] = np.array(d["paths"].tolist()[::-1])
    np.savez(cell_t, **d)
    with pytest.raises(SystemExit, match="ordered differently"):
        M.report(spec, out)
    d["paths"] = orig["paths"]
    d["labels"] = orig["labels"][::-1].copy()
    np.savez(cell_t, **d)
    with pytest.raises(SystemExit, match="test images differ"):
        M.report(spec, out)
    np.savez(cell_t, **orig)  # restore the untouched teacher cell
    # report: tables, AULC, teacher delta with CI at 100 %, missing cells listed
    md = M.report(spec, out)
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["summary"]["A"]["m_eurosat"]["aulc"] is not None
    dl = s["summary"]["A"]["m_eurosat"]["delta_vs_teacher"]["1.0"]
    assert set(dl) == {"delta", "ci_lo", "ci_hi"} and abs(dl["delta"]) < 1e-9  # same features -> same score
    assert "MISSING cells: 10" in md and "m_forestnet" in md
    assert os.path.exists(os.path.join(out, "curve_m_eurosat.png"))


def test_load_spec_validates(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"datasets": ["m_eurosat"], "models": [{"name": "A", "teacher": "Z"}]}))
    with pytest.raises(SystemExit, match="teacher"):
        M.load_spec(str(p))
    p.write_text(json.dumps({"datasets": ["nope"], "models": [{"name": "A"}]}))
    with pytest.raises(SystemExit):
        M.load_spec(str(p))


def test_blas_threads_is_a_fixed_policy_and_cells_carry_the_protocol_version():
    """The thread count must not depend on the job's CPU allocation (lbfgs on wide designs is
    only reproducible for a fixed count), and a code-only protocol change must invalidate cells."""
    assert (
        M.blas_threads(69) == 1 and M.blas_threads(2048) == 8 and M.blas_threads(20000) == M.BLAS_THREADS_CAP
    )
    cfg = M.cell_config(
        {"name": "m", "kind": "dino", "views": "cls", "features": "x_{dataset}"},
        "m_eurosat",
        0.01,
        0,
        (0.1, 1.0),
    )
    assert cfg["protocol"] == M.PROTOCOL_VERSION and cfg["blas_threads_cap"] == M.BLAS_THREADS_CAP


def test_config_current_accepts_only_bit_identical_old_cells():
    """PROTOCOL_VERSION 2026-10-15 fixed the one-vs-rest BLAS threads at protocols.OVR_FIT_THREADS.
    Cells written under 2026-10-13b are reused only where that changes nothing: single-label cells,
    and multi-label cells whose fits already ran at that share; a multi-label cell at another
    share, any other config difference or any older protocol is stale."""
    from eval import protocols as P

    cfg = {"protocol": M.PROTOCOL_VERSION, "c_grid": [0.1], "x": 1}
    old = {**cfg, "protocol": "2026-10-13b"}
    assert M.PROTOCOL_VERSION == "2026-10-15" and P.OVR_FIT_THREADS == 1
    assert M.config_current(cfg, cfg, {}, True)
    assert M.config_current(old, cfg, {"ovr_n_jobs": 7, "cpus": 16}, False)  # single-label: unchanged
    assert M.config_current(old, cfg, {"ovr_n_jobs": 9, "cpus": 16}, True)  # 16 // 9 = 1 = OVR_FIT_THREADS
    assert M.config_current(old, cfg, {"ovr_n_jobs": 21, "cpus": 35}, True)
    assert not M.config_current(old, cfg, {"ovr_n_jobs": 7, "cpus": 16}, True)  # 16 // 7 = 2: stale
    assert not M.config_current(old, cfg, {"ovr_n_jobs": 16, "cpus": 35}, True)
    assert M.config_current(old, cfg, {"ovr_n_jobs": 1, "cpus": 16, "blas_threads": 1}, True)  # serial at 1
    assert not M.config_current(old, cfg, {"ovr_n_jobs": 1, "cpus": 16, "blas_threads": 4}, True)
    assert M.config_current(
        old, cfg, {"ovr_n_jobs": 7, "cpus": 16, "ovr_threads": 1}, True
    )  # stamped share wins
    assert not M.config_current(old, cfg, {"ovr_n_jobs": 9, "cpus": 16, "ovr_threads": 2}, True)
    assert not M.config_current({**old, "x": 2}, cfg, {"ovr_n_jobs": 9, "cpus": 16}, False)
    assert not M.config_current({**old, "protocol": "2026-10-13"}, cfg, {}, False)
    assert not M.config_current(old, {**cfg, "protocol": "2026-10-16"}, {}, False)  # only the one known bump
    # --accept-thread-share: the share mismatch alone is waived, nothing else is
    assert M.config_current(old, cfg, {"ovr_n_jobs": 7, "cpus": 16}, True, accept_thread_share=True)
    assert not M.config_current(
        {**old, "x": 2}, cfg, {"ovr_n_jobs": 7, "cpus": 16}, True, accept_thread_share=True
    )
    assert (
        M.ovr_share({"ovr_n_jobs": 7, "cpus": 16}) == 2
        and M.ovr_share({"ovr_threads": 1, "cpus": 16, "ovr_n_jobs": 7}) == 1
    )
    assert M.ovr_share({"ovr_n_jobs": 1, "blas_threads": 4}) == 4 and M.ovr_share({}) is None


def test_summary_std_is_the_sample_std():
    """'mean ± std over seeds' is the SAMPLE std (ddof=1): three seeds at 0.80/0.82/0.84 read
    ± 0.0200, not the population 0.0163 (review 2026-10-16)."""
    spec = {"fractions": [1.0], "models": [{"name": "m"}], "datasets": ["m_fake"]}
    cells = {
        "m": {
            "m_fake": {1.0: [{"score": s, "n_train": 10, "inputs_absent": False} for s in (0.80, 0.82, 0.84)]}
        }
    }
    row = M.summarize(spec, cells)["m"]["m_fake"]["fractions"][1.0]
    assert abs(row["std"] - 0.02) < 1e-12 and row["n_seeds"] == 3
    one = {"m": {"m_fake": {1.0: [{"score": 0.8, "n_train": 10, "inputs_absent": False}]}}}
    assert M.summarize(spec, one)["m"]["m_fake"]["fractions"][1.0]["std"] == 0.0


def test_nonfinite_rows_detection():
    import numpy as np

    from eval.matrix import nonfinite_rows

    Atr = np.zeros((5, 3))
    Atr[2, 1] = np.nan
    Atr[4, 0] = np.inf
    cands = {
        "a": (Atr, np.zeros((2, 3)), np.zeros((2, 3))),
        "b": (np.zeros((5, 3)), np.zeros((2, 3)), np.zeros((2, 3))),
    }
    bad, bad_eval = nonfinite_rows(cands)
    assert bad.tolist() == [2, 4] and bad_eval == []
    cands["b"] = (np.zeros((5, 3)), np.full((2, 3), np.nan), np.zeros((2, 3)))
    assert nonfinite_rows(cands)[1] == ["b"]
