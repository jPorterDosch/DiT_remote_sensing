"""Guards added in the 2026-10-14 review of eval/protocols.py: each one is fed the condition it
exists to catch (CLAUDE.md rule 6: a guard that has never fired in a test is unverified)."""

import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import _ConstantPredictor

from eval import protocols as P


def test_n_not_converged_refuses_an_estimator_that_does_not_count():
    """A plain LogisticRegression inside the one-vs-rest list carries no n_conv_warn_: reporting
    0 for it would claim convergence without a measurement (rule 8), so it must raise."""
    clf = P.MultiLabelLR(LogisticRegression(), n_jobs=1)
    counting = P._CountingLR()
    counting.n_conv_warn_ = 2
    clf.estimators_ = [counting, _ConstantPredictor(), counting]
    assert P._n_not_converged(clf, []) == 4
    clf.estimators_ = [counting, LogisticRegression()]
    with pytest.raises(TypeError, match="does not count ConvergenceWarnings"):
        P._n_not_converged(clf, [])


def test_counting_lr_counts_convergence_warnings_and_re_emits_the_rest(monkeypatch):
    def fake_fit(self, X, y, sample_weight=None):
        warnings.warn("lbfgs failed to converge", ConvergenceWarning)
        warnings.warn("something unrelated", FutureWarning)
        return self

    monkeypatch.setattr(LogisticRegression, "fit", fake_fit)
    monkeypatch.setenv("OVR_N_JOBS", "1")
    clf = P._CountingLR()
    with pytest.warns(FutureWarning, match="something unrelated"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)  # the counted one must NOT escape
            clf.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    assert clf.n_conv_warn_ == 1


def test_shared_for_workers_refuses_a_worker_pool_without_a_temp_folder(monkeypatch):
    monkeypatch.setenv("OVR_N_JOBS", "2")
    monkeypatch.delenv("JOBLIB_TEMP_FOLDER", raising=False)
    y = np.zeros((8, 3), dtype=np.int8)
    big = np.zeros(P.SHARE_MIN_BYTES // 8 + 1, dtype=np.float64)  # just over the sharing threshold
    with pytest.raises(SystemExit, match="JOBLIB_TEMP_FOLDER is unset"):
        with P._shared_for_workers(lambda: big, y):
            pass
    small = np.zeros(8, dtype=np.float64)
    with P._shared_for_workers(lambda: small, y) as A:  # below the threshold: handed through
        assert A is small
    monkeypatch.setenv("OVR_N_JOBS", "1")
    with P._shared_for_workers(lambda: big, y) as A:  # serial: handed through
        assert A is big


def test_budget_ovr_jobs_budgets_from_the_requested_count_not_the_previous_pin(monkeypatch):
    """The pin is written to OVR_N_JOBS (workers read it), so a second cell in the same process
    must not treat the previous cell's pin as the request."""
    monkeypatch.setattr(P, "_OVR_PIN", None)
    monkeypatch.setattr(P, "_mem_limit_bytes", lambda: 64 << 30)
    monkeypatch.setattr(P, "_rss_bytes", lambda: 0)
    monkeypatch.setenv("OVR_N_JOBS", "8")
    y = np.zeros((10, 3), dtype=np.int8)

    class Shape:  # cands carry (Atr, Ava, Ate) and only Atr.shape is read: an 8 GB float64 stand-in
        shape = (1 << 30, 1)

    pinned = P.budget_ovr_jobs({"a": (Shape(), None, None)}, y)
    assert pinned < 8 and os.environ["OVR_N_JOBS"] == str(pinned)
    tiny = P.budget_ovr_jobs({"a": (np.zeros((10, 2)), None, None)}, y)
    assert tiny == 8 and os.environ["OVR_N_JOBS"] == "8"
    monkeypatch.setenv("OVR_N_JOBS", "3")  # a changed setting is a new request, not the old pin
    assert P.budget_ovr_jobs({"a": (np.zeros((10, 2)), None, None)}, y) == 3
