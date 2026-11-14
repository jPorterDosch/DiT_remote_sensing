"""Their-protocol anchor readout (eval/geocore_anchor.py).

pytest tests/test_geocore_anchor.py -v
"""

import json
import os
import sys

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from eval import geocore_anchor as GA  # noqa: E402


def test_anchor_accuracy_separable():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 32)) * 5
    y = rng.integers(0, 3, size=300)
    X = centers[y] + rng.normal(size=(300, 32)) * 0.3
    assert GA.anchor_accuracy(X, y, epochs=40, batch=64) > 0.95


def test_anchor_accuracy_is_deterministic():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(120, 8)), rng.integers(0, 2, size=120)
    assert GA.anchor_accuracy(X, y) == GA.anchor_accuracy(X, y)


def test_rejects_ensemble_cache(tmp_path):
    p = tmp_path / "multistep_train_feats_oneshot_g0.0_k4-8.npz"
    np.savez(p, feats=np.zeros((2, 2, 2, 4), np.float32), labels=np.zeros(2, int), subset_indices=np.arange(2),
             paths=np.array(["a", "b"]), timesteps=np.array([260, 580]), mods=np.zeros((2, 2, 3, 4), np.float32))  # fmt: skip
    (tmp_path / "multistep_train_feats_oneshot_g0.0_k4-8_meta.json").write_text(
        json.dumps(
            {"ensemble_size": 8, "t": [260, 580], "k": [4, 8], "layout": "N,K,B,C", "weights": "geocore-9b"}
        )
    )
    with pytest.raises(SystemExit, match="ensemble_size"):
        GA._load_all_splits(str(tmp_path / "multistep_{split}_feats_oneshot_g0.0_k4-8.npz"), [4, 8], 260)


# ------------------------------------------------------------------ the anchor is a gate (item 10)
def test_verdict_passes_inside_and_fails_outside_the_tolerance():
    assert GA.anchor_verdict(0.9805)[0]  # the measured 224 px anchor, +0.55
    assert GA.anchor_verdict(0.975)[0]
    assert GA.anchor_verdict(0.965)[0] and GA.anchor_verdict(0.985)[0]  # exactly one point, either side
    for bad in (0.9649, 0.9851, 0.5, 0.0, float("nan"), float("inf"), 97.5, -0.975):
        ok, line = GA.anchor_verdict(bad)
        assert not ok and line.startswith("GATE anchor FAIL") and "do NOT ship" in line, bad
    assert not GA.anchor_verdict(0.9805, tolerance=0.5)[0]
    assert not GA.anchor_verdict(0.975, tolerance=0.0)[0]  # a zero tolerance is a broken gate, not a pass
    assert not GA.anchor_verdict(0.975, tolerance=float("nan"))[0]


def _result(tmp_path, accuracy, reported=GA.THEIR_EUROSAT):
    p = tmp_path / f"anchor_{accuracy}.json"
    p.write_text(
        json.dumps(
            {"accuracy": accuracy, "geocore_reported": reported, "img_size": [224, 224], "n_images": 18192}
        )
    )
    return str(p)


def test_check_result_exit_code_follows_the_verdict(tmp_path, capsys):
    assert GA.main(["--check-result", _result(tmp_path, 0.9804837703704834)]) == 0
    assert "GATE anchor PASS: 98.05 vs GeoCore 97.5 (gap +0.55" in capsys.readouterr().out
    assert GA.main(["--check-result", _result(tmp_path, 0.93)]) == 1
    assert "GATE anchor FAIL" in capsys.readouterr().out
    assert GA.main(["--check-result", _result(tmp_path, 0.9804837703704834), "--tolerance", "0.5"]) == 1
    with pytest.raises(SystemExit, match="geocore_reported"):
        GA.main(["--check-result", _result(tmp_path, 0.98, reported=98.0)])


@pytest.mark.parametrize("tol", ["50", "1.01", "0", "-1", "nan", "inf"])
def test_cli_tolerance_can_only_tighten(tmp_path, tol):
    """`--tolerance 50` used to turn a 93.0 readout into GATE anchor PASS (staged-delta finding 8)."""
    with pytest.raises(SystemExit) as e:
        GA.main(["--check-result", _result(tmp_path, 0.93), "--tolerance", tol])
    assert e.value.code == 2  # argparse refuses the flag before any verdict is printed


def test_main_returns_one_and_never_overwrites(tmp_path, monkeypatch, capsys):
    """main() on a failing readout: exit 1, the json says FAIL, and a second run refuses the file."""
    monkeypatch.setattr(
        GA, "_load_all_splits", lambda *a: (np.zeros((10, 4)), np.zeros(10, int), [224, 224], "geocore-9b")
    )
    monkeypatch.setattr(GA, "anchor_accuracy", lambda *a: 0.93)
    out = tmp_path / "anchor.json"
    assert GA.main(["--features", "unused_{split}", "--out", str(out)]) == 1
    assert "GATE anchor FAIL: 93.00" in capsys.readouterr().out
    assert json.loads(out.read_text())["gate"] == "FAIL"
    monkeypatch.setattr(GA, "anchor_accuracy", lambda *a: 0.98)
    with pytest.raises(SystemExit, match="never overwritten"):
        GA.main(["--features", "unused_{split}", "--out", str(out)])
    assert json.loads(out.read_text())["accuracy"] == 0.93
    assert GA.main(["--features", "unused_{split}", "--out", str(tmp_path / "anchor2.json")]) == 0
