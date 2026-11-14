"""CPU tests for eval/fusion.py (the learned per-slot weighting head, --head tsweight).

Rule-6 style: each guarantee is tested by feeding the case it must catch -- the head must find
the informative slot, selection must be blind to test labels, standardization must use train
statistics only, and the DINO slot layout must match extract_dino's cls4 order.

    pytest tests/test_fusion.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from eval import fusion as FU

EP = 8  # short schedule for CPU tests (warmup 5 + 3 cosine epochs)


def _data(n=3000, S=3, C=16, informative=1, seed=0, n_cls=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cls, n)
    X = rng.normal(size=(n, S, C)).astype(np.float32)
    proto = rng.normal(size=(n_cls, C)).astype(np.float32) * 2.5
    X[:, informative] += proto[y]
    return X, y


def _splits(X, y):
    a, b = int(0.7 * len(X)), int(0.85 * len(X))  # ~8 optimizer steps per epoch at batch 256
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def test_head_learns_the_informative_slot_and_beats_chance():
    X, y = _data(informative=2)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(X, y)
    correct, pred, sel, rows = FU.run_tsweight(Xtr, ytr, Xva, yva, Xte, yte, 3, ["a", "b", "c"], "cpu", EP)
    w = np.array(sel["slot_weights"])
    assert w.argmax() == 2 and w[2] > 0.5 and np.isclose(w.sum(), 1, atol=1e-5)
    assert correct.mean() > 0.8 and pred.shape == (len(yte),)
    assert len(rows) == len(FU.FUSION_LR_GRID) * len(FU.FUSION_WD_GRID)
    assert all(len(r["curve"]) == EP for r in rows) and sel["slots"] == ["a", "b", "c"]
    # the selected val is the max over every (lr, wd, epoch) and its epoch is the curve's argmax
    best = max(rows, key=lambda r: r["val_acc"])
    assert np.isclose(sel["val_acc"], best["val_acc"]) and sel["epoch"] == int(np.argmax(best["curve"])) + 1


def test_deterministic():
    X, y = _data(seed=3)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(X, y)
    a = FU.run_tsweight(Xtr, ytr, Xva, yva, Xte, yte, 3, ["a", "b", "c"], "cpu", EP)
    b = FU.run_tsweight(Xtr, ytr, Xva, yva, Xte, yte, 3, ["a", "b", "c"], "cpu", EP)
    assert np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_selection_is_blind_to_test_labels():
    X, y = _data(seed=4)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(X, y)
    _, pred1, sel1, _ = FU.run_tsweight(Xtr, ytr, Xva, yva, Xte, yte, 3, ["a", "b", "c"], "cpu", EP)
    yperm = np.random.default_rng(9).permutation(yte)
    c2, pred2, sel2, _ = FU.run_tsweight(Xtr, ytr, Xva, yva, Xte, yperm, 3, ["a", "b", "c"], "cpu", EP)
    assert sel1 == sel2 and np.array_equal(pred1, pred2)  # only the scoring of test changed
    assert np.array_equal(c2, (pred2 == yperm).astype(c2.dtype))


def test_standardize_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(5, 2, (50, 2, 4)), rng.normal(0, 1, (20, 2, 4))
    a_tr, a_va = FU.standardize(Xtr, Xva)
    b_tr, _ = FU.standardize(Xtr, Xva * 100 + 7)  # changing val must not move train's scaling
    assert np.allclose(a_tr, b_tr) and abs(a_tr.mean()) < 1e-5
    assert not np.allclose(a_va.mean(), 0, atol=0.5)  # val is NOT re-centred on itself


def test_multilabel_path_uses_threshold_and_micro_f1():
    rng = np.random.default_rng(1)
    n, L = 3000, 5
    y = (rng.random((n, L)) < 0.3).astype(np.int64)
    X = rng.normal(size=(n, 2, 12)).astype(np.float32)
    X[:, 0, :L] += 3 * y  # slot 0 carries the labels
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(X, y)
    correct, pred, sel, _ = FU.run_tsweight(Xtr, ytr, Xva, yva, Xte, yte, L, ["s0", "s1"], "cpu", EP)
    assert pred.shape == (len(yte), L) and set(np.unique(pred)) <= {0, 1}
    assert sel["slot_weights"][0] > 0.5
    from eval import protocols as P

    assert 0.7 < P.metric(correct) <= 1.0  # micro-F1 from the per-image count vectors


def test_run_tsweight_refuses_bad_shapes():
    X, y = _data()
    with pytest.raises(ValueError, match="named slots"):
        FU.run_tsweight(X[:40, 0], y[:40], X[40:50, 0], y[40:50], X[50:60, 0], y[50:60], 3, ["a"], "cpu", EP)
    with pytest.raises(ValueError, match="named slots"):
        FU.run_tsweight(X[:40], y[:40], X[40:50], y[40:50], X[50:60], y[50:60], 3, ["a", "b"], "cpu", EP)


class _Npz(dict):
    @property
    def files(self):
        return list(self)


def test_dino_slots_layout_and_refusal():
    rng = np.random.default_rng(0)
    D, n = 6, 4
    blocks = [rng.normal(size=(n, D)) for _ in range(4)]  # block -4 .. -1
    d = _Npz(cls4=np.concatenate(blocks, axis=1), mp=rng.normal(size=(n, D)), cls=blocks[-1])
    X, names = FU.dino_slots(d)
    assert X.shape == (n, 5, D) and names[-1] == "mp" and names[3] == "cls_b-1"
    assert (
        np.allclose(X[:, 3], d["cls"]) and np.allclose(X[:, 0], blocks[0]) and np.allclose(X[:, 4], d["mp"])
    )
    with pytest.raises(SystemExit, match="needs cls4"):
        FU.dino_slots(_Npz(cls=blocks[-1], mp=d["mp"]))


# ------------------------------------------------------------------ --fusion-grid reg
def test_reg_grid_does_not_decay_theta_and_records_itself():
    import torch

    X, y = _data(seed=5, informative=0)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(X, y)
    Xtr_s, Xva_s = FU.standardize(Xtr, Xva)
    # heavy decay: with decay_theta=False the slot weights still move off uniform
    (_, _, st_free), _ = FU._train_config(torch.from_numpy(Xtr_s), ytr, torch.from_numpy(Xva_s), yva, 3,
                                          1e-2, 10.0, False, "cpu", EP, decay_theta=False)  # fmt: skip
    (_, _, st_dec), _ = FU._train_config(torch.from_numpy(Xtr_s), ytr, torch.from_numpy(Xva_s), yva, 3,
                                         1e-2, 10.0, False, "cpu", EP, decay_theta=True)  # fmt: skip
    w_free = torch.softmax(st_free["theta"], 0)
    w_dec = torch.softmax(st_dec["theta"], 0)
    assert w_free[0] > w_dec[0]  # decaying theta pulls the mix toward uniform; reg grid must not
    c = FU.fusion_constants("reg")
    assert c["grid"] == "reg" and c["decay_theta"] is False and c["wd_grid"] == [0.0, 1e-2, 1e-1, 1.0, 10.0]
    with pytest.raises(SystemExit, match="unknown --fusion-grid"):
        FU.fusion_constants("nope")


def test_satdifuser_grid_unchanged():
    # the first run's protocol must stay reproducible bit-for-bit
    c = FU.fusion_constants("satdifuser")
    assert c["lr_grid"] == [1e-3, 1e-2] and c["wd_grid"] == [0.0, 1e-4] and c["decay_theta"] is True
