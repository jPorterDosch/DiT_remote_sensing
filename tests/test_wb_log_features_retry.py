"""eval.wb.log_features retries the producer-side hash while a freshly written file settles."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import wb  # noqa: E402


class _Run:
    def __init__(self):
        self.logged = []

    def log_artifact(self, art):
        self.logged.append(art)


def _patch(
    monkeypatch, tmp_path, failures, message="x changed while being hashed -- retry once it is stable"
):
    path = tmp_path / "feats.npz"
    path.write_bytes(b"0" * 16)
    calls = []

    def fake_artifact(p):
        calls.append(p)
        if len(calls) <= failures:
            raise OSError(message)
        return ("artifact", p)

    run = _Run()
    monkeypatch.setattr(wb, "features_artifact", fake_artifact)
    monkeypatch.setattr(wb.wandb, "run", run, raising=False)
    monkeypatch.setattr(wb.time, "sleep", lambda s: None)
    return str(path), calls, run


def test_retries_the_settle_error_then_logs_once(monkeypatch, tmp_path):
    path, calls, run = _patch(monkeypatch, tmp_path, failures=2)
    wb.log_features(path, attempts=4, settle_s=0)
    assert len(calls) == 3
    assert run.logged == [("artifact", path)]


def test_gives_up_after_attempts(monkeypatch, tmp_path):
    path, calls, run = _patch(monkeypatch, tmp_path, failures=10)
    with pytest.raises(OSError, match="changed while being hashed"):
        wb.log_features(path, attempts=3, settle_s=0)
    assert len(calls) == 3 and run.logged == []


def test_other_oserror_is_not_retried(monkeypatch, tmp_path):
    path, calls, run = _patch(monkeypatch, tmp_path, failures=1, message="No such file")
    with pytest.raises(OSError, match="No such file"):
        wb.log_features(path, attempts=4, settle_s=0)
    assert len(calls) == 1 and run.logged == []


def test_no_run_is_a_noop(monkeypatch, tmp_path):
    path, calls, run = _patch(monkeypatch, tmp_path, failures=0)
    monkeypatch.setattr(wb.wandb, "run", None, raising=False)
    wb.log_features(path)
    assert calls == []
