"""Chest X-ray probing: eval.probe accepts the OFFICIAL_CXR datasets, and multi-label results carry
the mean per-label AUROC next to micro-F1 (micro-F1 stays the score and the selection metric).
Synthetic features only; no images, no patient data."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval import features as F  # noqa: E402
from eval import matrix as M  # noqa: E402
from eval import probe as PR  # noqa: E402
from eval import protocols as P  # noqa: E402


def _multilabel(rng, n, d=10, L=4, w=None):
    """Synthetic multi-hot data; pass the same `w` for every split so train and test share one rule."""
    w = rng.normal(size=(d, L)) if w is None else w
    X = rng.normal(size=(n, d))
    y = ((X @ w + 0.7 * rng.normal(size=(n, L))) > 0).astype(np.int64)
    return X, y


def _splits(rng, sizes, d=10, L=4):
    w = rng.normal(size=(d, L))
    return [_multilabel(rng, n, d, L, w) for n in sizes]


def test_auroc_per_label_known_values_and_skips_single_class_labels():
    y = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]])
    proba = np.array([[0.9, 0.1, 0.5], [0.2, 0.2, 0.5], [0.7, 0.3, 0.5], [0.4, 0.4, 0.5]])
    r = P.auroc_per_label(y, proba)
    assert r["auroc_per_label"][0] == 1.0  # label 0 perfectly ranked
    assert r["auroc_per_label"][1] is None and r["auroc_per_label"][2] is None  # one class in test
    assert (r["auroc_mean"], r["auroc_n_labels"]) == (1.0, 1)
    with pytest.raises(ValueError, match="needs"):
        P.auroc_per_label(y, proba[:, :2])


def test_run_official_adds_auroc_and_leaves_predictions_and_score_alone(monkeypatch):
    monkeypatch.setenv("OVR_N_JOBS", "1")
    rng = np.random.default_rng(0)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(rng, (200, 80, 80))
    correct, pred, sel, _ = P.run_official({"x": (Xtr, Xva, Xte)}, ytr, yva, yte, c_grid=(0.1, 1.0))
    assert (
        0.8 < sel["auroc_mean"] <= 1.0 and sel["auroc_n_labels"] == 4
    )  # learnable rule: high AUROC, so the probability columns are not inverted or permuted
    # the same fit without probabilities gives the same predictions (AUROC is added, nothing replaced)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    pred2, _ = P.official_fit_eval(sc.transform(Xtr), ytr, sc.transform(Xte), sel["C"])
    assert np.array_equal(pred, pred2) and P.metric(correct) == P.metric(P.per_image(yte, pred2))
    # single-label results get no AUROC fields
    ys = ytr[:, 0], yva[:, 0], yte[:, 0]
    assert "auroc_mean" not in P.run_official({"x": (Xtr, Xva, Xte)}, *ys, c_grid=(0.1,))[2]


def test_matrix_fit_cell_returns_probabilities_only_on_request(monkeypatch):
    monkeypatch.setenv("OVR_N_JOBS", "1")
    monkeypatch.setattr(P, "_mem_limit_bytes", lambda: 64 << 30)
    rng = np.random.default_rng(1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(rng, (200, 80, 80))
    cands = {"x": (Xtr, Xva, Xte)}
    out5 = M.fit_cell(cands, np.arange(200), ytr, yva, yte, (0.1, 1.0))
    out6 = M.fit_cell(cands, np.arange(200), ytr, yva, yte, (0.1, 1.0), return_proba=True)
    assert len(out5) == 5 and len(out6) == 6
    proba, sel = out6[5], out6[2]
    assert proba.shape == yte.shape and "_proba" not in sel and "_proba" not in out5[2]
    assert sel["auroc_mean"] == P.auroc_per_label(yte, proba)["auroc_mean"]
    assert np.array_equal(out5[1], out6[1]) and out5[2] == sel  # same fit, same record


def test_probe_run_official_accepts_a_chest_xray_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("OVR_N_JOBS", "1")
    rng = np.random.default_rng(2)
    data = dict(zip(("train", "val", "test"), _splits(rng, (120, 40, 40))))
    files = {s: [f"/x/{s}/{i:03d}.png" for i in range(len(y))] for s, (_, y) in data.items()}
    for s, (X, y) in data.items():
        np.savez(
            tmp_path / f"medsiglip448-rev_224up_chestxray14_{s}.npz",
            cls=X,
            mp=X[:, :5],
            paths=np.array(files[s]),
            labels=y,
            preset=np.array("medsiglip448-rev_224up"),
        )
    sizes = {s: len(v) for s, v in files.items()}
    monkeypatch.setattr(
        F, "official_spec", lambda ds: {"sizes": sizes, "multilabel": True, "root": str(tmp_path)}
    )
    monkeypatch.setattr(F, "list_official_split", lambda ds, s: (files[s], data[s][1]))
    assert (
        "chestxray14" not in F.OFFICIAL and "chestxray14" in F.OFFICIAL_CXR
    )  # the case that used to be refused
    args = SimpleNamespace(
        dataset="chestxray14",
        kind="dino",
        features=str(tmp_path / "medsiglip448-rev_224up_chestxray14_{split}.npz"),
        view=None,
        sizes=None,
        head="lr",
        c_grid_t=(0.1, 1.0),
        fusion_grid=None,
    )
    cells, info, paths, labels = PR.run_official(args)
    assert info["selected"]["candidate"] in F.MEDSIGLIP_OFFICIAL_VIEWS.split(",")
    assert "auroc_mean" in info and "micro_f1" in info  # AUROC added next to micro-F1
