"""eval.paired_ci: paired image-bootstrap intervals from stored per-image results.

Each guard is fed the mismatch it exists to catch (CLAUDE.md rule 6)."""

import json
import os

import numpy as np
import pytest

from eval import paired_ci as P


def _miou_ref(conf):
    conf = conf.astype(np.float64)
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    ok = union > 0
    return float((inter[ok] / union[ok]).mean()) if ok.any() else float("nan")


def _write_cls(root, arm, dataset, frac, seed, correct, paths, labels, score=None, train=None):
    d = os.path.join(root, arm, dataset)
    os.makedirs(d, exist_ok=True)
    if score is None:
        score = float(P.metric_of(correct, np.ones((1, len(correct))))[0])
    np.savez(
        os.path.join(d, f"f{frac:g}_s{seed}.npz"),
        correct=correct,
        score=np.array(score),
        paths=np.array(paths),
        labels=labels,
        train_rows=np.arange(5) if train is None else train,
    )


def _write_seg(root, arm, dataset, frac, seed, conf, paths):
    d = os.path.join(root, arm, dataset)
    os.makedirs(d, exist_ok=True)
    np.savez(
        os.path.join(d, f"f{frac:g}_s{seed}.npz"),
        test_conf_per_image=conf,
        test_miou=np.array(_miou_ref(conf.sum(0))),
        test_paths=np.array(paths),
        train_indices=np.arange(5),
    )


def _cls_pair(tmp_path, n=1000, flip=60, seeds=(0,)):
    rng = np.random.default_rng(1)
    paths = [f"img_{i:04d}" for i in range(n)]
    labels = rng.integers(0, 5, n)
    root = str(tmp_path / "cells")
    for s in seeds:
        a = (rng.random(n) < 0.6).astype(np.int8)
        b = a.copy()
        b[np.flatnonzero(a == 0)[:flip]] = 1  # the arm fixes `flip` of the base's errors, breaks none
        _write_cls(root, "base", "ds", 1.0, s, a, paths, labels)
        _write_cls(root, "arm", "ds", 1.0, s, b, paths, labels)
    return root, paths, labels


def test_miou_batch_matches_reference_and_skips_empty_classes():
    rng = np.random.default_rng(0)
    conf = rng.integers(0, 50, (7, 4, 4))
    conf[:, 3, :] = 0
    conf[:, :, 3] = 0  # class 3 never present nor predicted: skipped, not counted as IoU 0
    got = P.miou_batch(conf)
    assert np.allclose(got, [_miou_ref(c) for c in conf], atol=1e-12)
    assert np.isnan(P.miou_batch(np.zeros((1, 3, 3)))[0])


def test_micro_f1_from_counts():
    sums = np.array([[8.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    assert np.allclose(P.micro_f1_batch(sums), [0.8, 0.0])


def test_known_effect_is_recovered_and_paired_interval_is_tight(tmp_path):
    root, _, _ = _cls_pair(tmp_path, n=1000, flip=60)
    res = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=2000, seed=0)
    assert not res["refused"]
    (r,) = res["rows"]
    assert r["delta"] == pytest.approx(0.06, abs=1e-12)
    assert r["ci_lo"] > 0 and r["excludes_zero"]
    # every flipped image moves the same way, so the paired interval is far narrower than the
    # ~0.06 half-width two independent accuracies at n = 1000 would give
    assert r["ci_hi"] - r["ci_lo"] < 0.04
    assert all(g["pass"] for g in res["gates"].values())


def test_no_effect_interval_covers_zero(tmp_path):
    rng = np.random.default_rng(3)
    n = 800
    paths, labels = [f"p{i}" for i in range(n)], rng.integers(0, 3, n)
    root = str(tmp_path / "cells")
    _write_cls(root, "base", "ds", 0.1, 0, (rng.random(n) < 0.5).astype(np.int8), paths, labels)
    _write_cls(root, "arm", "ds", 0.1, 0, (rng.random(n) < 0.5).astype(np.int8), paths, labels)
    (r,) = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=2000, seed=0)["rows"]
    assert r["ci_lo"] < 0 < r["ci_hi"] and not r["excludes_zero"]


def test_reordered_test_images_are_refused(tmp_path):
    root, paths, labels = _cls_pair(tmp_path)
    a = np.load(os.path.join(root, "arm", "ds", "f1_s0.npz"))
    _write_cls(root, "arm", "ds", 1.0, 0, a["correct"], paths[::-1], labels)  # same set, other order
    res = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=200, seed=0)
    assert not res["rows"]
    assert len(res["refused"]) == 1 and "test paths differ" in res["refused"][0]["reason"]


def test_different_labels_are_refused(tmp_path):
    root, paths, labels = _cls_pair(tmp_path)
    a = np.load(os.path.join(root, "arm", "ds", "f1_s0.npz"))
    _write_cls(root, "arm", "ds", 1.0, 0, a["correct"], paths, (labels + 1) % 5)
    res = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=200, seed=0)
    assert not res["rows"] and "test labels differ" in res["refused"][0]["reason"]


def test_seed_mismatch_is_refused(tmp_path):
    root, paths, labels = _cls_pair(tmp_path, seeds=(0, 1))
    os.remove(os.path.join(root, "arm", "ds", "f1_s1.npz"))
    res = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=200, seed=0)
    assert not res["rows"] and "seeds differ" in res["refused"][0]["reason"]


def test_reproduction_gate_fails_on_a_stored_score_the_data_does_not_give(tmp_path):
    root, paths, labels = _cls_pair(tmp_path)
    a = np.load(os.path.join(root, "arm", "ds", "f1_s0.npz"))
    _write_cls(root, "arm", "ds", 1.0, 0, a["correct"], paths, labels, score=0.123)
    res = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=200, seed=0)
    assert not res["gates"]["reproduction"]["pass"]
    out = str(tmp_path / "out")
    assert P.main(["--kind", "cls", "--cells", root, "--base", "base", "--n-boot", "200", "--out", out]) == 1


def test_multiseed_mean_and_train_subset_flag(tmp_path):
    root, paths, labels = _cls_pair(tmp_path, n=500, flip=20, seeds=(0, 1, 2))
    a = np.load(os.path.join(root, "arm", "ds", "f1_s2.npz"))
    _write_cls(root, "arm", "ds", 1.0, 2, a["correct"], paths, labels, train=np.arange(5) + 1)
    (r,) = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=500, seed=0)["rows"]
    assert r["n_seeds"] == 3 and r["delta"] == pytest.approx(0.04, abs=1e-12)
    assert r["same_train_subsets"] is False


def test_micro_f1_cells(tmp_path):
    rng = np.random.default_rng(5)
    n = 400
    paths, labels = [f"p{i}" for i in range(n)], rng.integers(0, 2, (n, 6))
    a = rng.integers(0, 4, (n, 3))
    b = a.copy()
    b[:, 0] += 1  # one more true positive per image
    root = str(tmp_path / "cells")
    _write_cls(root, "base", "ds", 1.0, 0, a, paths, labels)
    _write_cls(root, "arm", "ds", 1.0, 0, b, paths, labels)
    res = P.compare(root, "cls", "base", ["arm"], ["ds"], n_boot=500, seed=0)
    (r,) = res["rows"]
    want = P.micro_f1_batch(b.sum(0)[None])[0] - P.micro_f1_batch(a.sum(0)[None])[0]
    assert r["delta"] == pytest.approx(want, abs=1e-12) and r["ci_lo"] > 0
    assert res["gates"]["reproduction"]["pass"]


def test_seg_cells_end_to_end_cli(tmp_path, capsys):
    rng = np.random.default_rng(7)
    n, c = 120, 3
    paths = [f"m_{i}.npy" for i in range(n)]
    root = str(tmp_path / "cells")
    for s in (0, 1):
        ref = rng.integers(20, 60, (n, c))
        base = np.zeros((n, c, c), np.int64)
        arm = np.zeros((n, c, c), np.int64)
        for k in range(c):
            wrong = rng.integers(5, 15, n)
            base[:, k, k], base[:, k, (k + 1) % c] = ref[:, k] - wrong, wrong
            arm[:, k, k], arm[:, k, (k + 1) % c] = ref[:, k] - wrong // 2, wrong // 2  # half the errors
        _write_seg(root, "base", "ds", 0.25, s, base, paths)
        _write_seg(root, "arm", "ds", 0.25, s, arm, paths)
    out = str(tmp_path / "res" / "seg")
    assert P.main(["--kind", "seg", "--cells", root, "--base", "base", "--n-boot", "500", "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "GATE reproduction PASS" in printed and "GATE null PASS" in printed and "PAIRED_CI_OK" in printed
    res = json.load(open(out + ".json"))
    (r,) = res["rows"]
    assert r["delta"] > 0 and r["ci_lo"] > 0 and r["n_seeds"] == 2 and r["n_images"] == n
    assert "| arm | 0.25 |" in open(out + ".md").read()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        P.main(["--kind", "seg", "--cells", root, "--base", "base", "--out", out])


def test_missing_base_dataset_is_listed_not_skipped(tmp_path):
    root, _, _ = _cls_pair(tmp_path)
    res = P.compare(root, "cls", "base", ["arm"], ["ds", "absent"], n_boot=200, seed=0)
    assert any(r["dataset"] == "absent" and "no cells for base" in r["reason"] for r in res["refused"])
