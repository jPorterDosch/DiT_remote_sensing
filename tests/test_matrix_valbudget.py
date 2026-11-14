"""eval.matrix "selection": "val-budget" -- model selection on a val subset drawn at the cell's budget."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import matrix as M  # noqa: E402
from tests.test_matrix import _spec, _synthetic  # noqa: E402

FR = (0.05, 0.25, 1.0)


def test_selection_stamp_and_rows():
    assert M.selection_stamp("full-val", 0.01) == M.FULL_VAL_STAMP
    assert M.selection_stamp("val-budget", 1.0) == M.FULL_VAL_STAMP  # shared 100 % cells
    assert M.selection_stamp("val-budget", 0.05).startswith("val-budget")
    with pytest.raises(ValueError, match="selection must be"):
        M.selection_stamp("test-set", 0.05)
    y = np.repeat(np.arange(4), 50)
    assert len(M.val_rows("full-val", y, 0.05, 0)) == 200
    rows = M.val_rows("val-budget", y, 0.05, 0)
    assert len(rows) == 12 and sorted(set(y[rows].tolist())) == [0, 1, 2, 3]  # ceil(0.05 * 50) per class
    assert np.array_equal(rows, M.label_subsets(y, FR, 0)[0.05])  # the same draw as the train rule
    assert len(M.val_rows("val-budget", y, 1.0, 0)) == 200


def _wire(monkeypatch):
    cands, ytr, yva, yte = _synthetic()
    monkeypatch.setattr(
        M, "load_cands", lambda m, d: (cands, ytr, yva, yte, [f"p{i}" for i in range(len(yte))], ["x"])
    )
    monkeypatch.setattr(M, "inputs_sha", lambda files: {"x": "aaa"})
    monkeypatch.setattr(M, "input_files", lambda m, d: ["x"])
    monkeypatch.setattr(M, "features_present", lambda m, d: (True, ""))
    return yva


def test_run_val_budget_cells_select_on_the_budget_rows(tmp_path, monkeypatch):
    yva = _wire(monkeypatch)
    spec = {**_spec(tmp_path), "fractions": list(FR), "datasets": ["m_eurosat"], "selection": "val-budget"}
    out = str(tmp_path / "vb")
    log = M.run(spec, spec["models"], spec["datasets"], out, redo=False)
    assert sum(r["status"] == "done" for r in log) == 2 * (2 * 2 + 1)
    low = np.load(M.cell_path(out, "A", "m_eurosat", 0.05, 1), allow_pickle=True)
    expect = M.label_subsets(yva, FR, 1)[0.05]
    assert np.array_equal(low["val_rows"], expect) and int(low["n_val"]) == len(expect) < len(yva)
    assert json.loads(str(low["config"]))["selection"].startswith("val-budget")
    # the val accuracies in the selection table are computed on the subset: integer multiples of 1/n_val
    table = json.loads(str(low["val_table"]))
    assert all(abs(r["val_acc"] * len(expect) - round(r["val_acc"] * len(expect))) < 1e-9 for r in table)
    full = np.load(M.cell_path(out, "A", "m_eurosat", 1.0, 0), allow_pickle=True)
    assert int(full["n_val"]) == len(yva) and json.loads(str(full["config"]))["selection"] == M.FULL_VAL_STAMP
    # a 100 % cell of the default protocol is bit-identical in config: shareable between the two specs
    spec_full = {**spec, "selection": "full-val"}
    out_full = str(tmp_path / "fv")
    M.run(spec_full, spec_full["models"], spec_full["datasets"], out_full, redo=False)
    ref = np.load(M.cell_path(out_full, "A", "m_eurosat", 1.0, 0), allow_pickle=True)
    assert json.loads(str(ref["config"])) == json.loads(str(full["config"]))
    assert float(ref["score"]) == float(full["score"])
    # a low cell of the default protocol differs in config (selection stamp) -> refused by collect
    ref_low = np.load(M.cell_path(out_full, "A", "m_eurosat", 0.05, 1), allow_pickle=True)
    assert json.loads(str(ref_low["config"]))["selection"] == M.FULL_VAL_STAMP
    assert int(ref_low["n_val"]) == len(yva)
    # report under the val-budget spec reads its own cells and labels the protocol
    md = M.report(spec, out)
    assert "Protocol (val-budget)" in md
    with pytest.raises(SystemExit, match="protocol/config differs"):
        M.collect(spec, out_full)


def test_load_spec_rejects_unknown_selection(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({**_spec(tmp_path), "selection": "test"}))
    with pytest.raises(SystemExit, match="selection must be"):
        M.load_spec(str(p))
