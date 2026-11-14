"""eval/complementarity.py: agreement statistics, the identity guards (fed the mismatch they exist
to catch, rule 6), the null / anchor blocks, the verdict rule (3b), and the run -> report path on a
synthetic problem with a complementary block, a junk block and the gate controls. CPU, no feature
files: load_role / load_cell are monkeypatched."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import complementarity as C  # noqa: E402


# ----------------------------------------------------------------------------- agreement
def test_agreement_identical_and_independent():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 4000)
    pa = np.where(rng.random(4000) < 0.7, y, (y + 1) % 4)
    same, _ = C.agreement(y, pa, pa, n_boot=200)
    assert same["kappa"]["value"] == pytest.approx(1.0)
    assert same["agreement"]["value"] == 1.0 and same["a_only"]["value"] == 0.0
    assert same["same_wrong_frac"]["value"] == 1.0
    pb = np.where(rng.random(4000) < 0.7, y, (y + 2) % 4)  # independent errors, never the same wrong class
    ind, boot = C.agreement(y, pa, pb, n_boot=500)
    assert abs(ind["kappa"]["value"]) < 0.06
    assert ind["kappa"]["ci"][0] < 0 < ind["kappa"]["ci"][1]
    assert ind["same_wrong_frac"]["value"] == 0.0
    parts = sum(ind[k]["value"] for k in ("both_right", "a_only", "b_only", "both_wrong"))
    assert parts == pytest.approx(1.0)
    assert ind["oracle"]["value"] == pytest.approx(1 - ind["both_wrong"]["value"])
    assert boot["kappa"].shape == (500,)
    assert ind["n_images"] == ind["n_units"] == 4000


def test_agreement_multilabel_units_are_clustered_by_image():
    rng = np.random.default_rng(1)
    y = (rng.random((50, 7)) < 0.3).astype(int)
    pa, pb = y.copy(), y.copy()
    pa[:10] = 1 - pa[:10]  # A wrong on every label of 10 images
    pb[5:15] = 1 - pb[5:15]
    S = C.agreement_counts(y, pa, pb)
    assert S.shape == (50, 7) and (S[:, 0] == 7).all()
    out, _ = C.agreement(y, pa, pb, n_boot=100)
    assert out["n_images"] == 50 and out["n_units"] == 350
    assert out["both_wrong"]["value"] == pytest.approx(5 / 50)
    assert out["a_only"]["value"] == pytest.approx(5 / 50)  # images 10-14: only B wrong
    assert out["b_only"]["value"] == pytest.approx(5 / 50)  # images 0-4: only A wrong
    with pytest.raises(ValueError):
        C.agreement_counts(y, pa[:, :6], pb)


def test_bootstrap_stream_is_shared_between_pairs():
    S = np.random.default_rng(2).integers(0, 2, (300, 7))
    assert np.array_equal(C.boot_sums(S, 50), C.boot_sums(S, 50))
    assert np.array_equal(
        C.boot_sums(S, 50)[:, 0], C.boot_sums(2 * S, 50)[:, 0] // 2
    )  # same resample indices


def _cell(paths, labels):
    return {"paths": np.array(paths), "labels": np.asarray(labels)}


def test_same_image_guard_fires_on_disjoint_subsets_with_equal_labels():
    labels = np.repeat(np.arange(3), 4)  # sorted stratified labels: identical for any subset (6o-F)
    a = _cell([f"data/x/test/c{c}/id_{i}.png" for i, c in enumerate(labels)], labels)
    b = _cell([f"data/x/test/c{c}/id_{i + 100}.png" for i, c in enumerate(labels)], labels)
    C.check_same_images(a, a, "self")
    with pytest.raises(SystemExit, match="image lists differ"):
        C.check_same_images(a, b, "disjoint")
    swapped = list(a["paths"])
    swapped[0], swapped[1] = swapped[1], swapped[0]  # same set, same labels, different order
    with pytest.raises(SystemExit, match="image lists differ"):
        C.check_same_images(a, _cell(swapped, labels), "reordered")
    with pytest.raises(SystemExit, match="labels differ"):
        C.check_same_images(a, _cell(a["paths"], (labels + 1) % 3), "labels")


# ----------------------------------------------------------------------------- blocks
def test_null_keeps_marginals_and_breaks_rows_within_split():
    rng = np.random.default_rng(3)
    block = tuple(rng.standard_normal((n, 5)) + 10 * i for i, n in enumerate((40, 20, 30)))
    null = C.null_of(block, "FA")
    for B, N in zip(block, null):
        assert B.shape == N.shape and not np.array_equal(B, N)
        assert np.array_equal(np.sort(B, axis=0), np.sort(N, axis=0))  # same marginals per column
        assert {tuple(r) for r in B} == {tuple(r) for r in N}  # whole rows move; none crosses a split
    again = C.null_of(block, "FA")
    assert all(np.array_equal(a, b) for a, b in zip(null, again))  # the same in every job
    other = C.null_of(block, "F0")
    assert not np.array_equal(null[0], other[0])


def test_anchor_contains_the_labels_and_weak_anchor_has_unit_aggregate_snr():
    ys = tuple(np.repeat(np.arange(4), n) for n in (30, 10, 20))
    strong = C.anchor_block(ys, 22, "anchor")
    assert [a.shape for a in strong] == [(120, 22), (40, 22), (80, 22)]
    assert C.anchor_sd("anchor", 22, 4) == 4.0
    assert C.anchor_sd("anchorw", 3072, 10) == pytest.approx(np.sqrt(307))
    big = C.anchor_block(ys, 4000, "anchor")[0]  # mean over a label's 1000 copies recovers the label
    means = np.stack([big[:, k::4].mean(1) for k in range(4)], 1)
    assert (means.argmax(1) == ys[0]).all()
    multi = tuple(
        (np.random.default_rng(i).random((n, 6)) < 0.4).astype(int) for i, n in enumerate((30, 10, 20))
    )
    assert C.anchor_block(multi, 15, "anchorw")[2].shape == (20, 15)
    noise = C.noise_block([30, 10, 20], 6)
    assert [x.shape for x in noise] == [(30, 6), (10, 6), (20, 6)]
    assert np.array_equal(noise[0], C.noise_block([30, 10, 20], 6)[0])


def test_verdict_needs_both_bars():
    assert C.verdict((0.02, 0.01, 0.03), (0.03, 0.02, 0.04)).startswith("ADDS")
    assert C.verdict((0.00, -0.01, 0.01), (0.03, 0.02, 0.04)).startswith(
        "image-linked"
    )  # redundant copy (3b)
    assert C.verdict((0.02, 0.01, 0.03), (0.01, -0.01, 0.02)) == "no detectable addition"
    assert C.verdict((-0.02, -0.03, -0.01), (0.03, 0.02, 0.04)) == "hurts"
    assert C.verdict((0.0, -0.01, 0.01), (0.0, -0.01, 0.01)) == "no detectable addition"


def test_pick_view_takes_best_val_among_single_views():
    table = [
        {"candidate": "t100", "C": 0.1, "val_acc": 0.80},
        {"candidate": "t100", "C": 1.0, "val_acc": 0.90},
        {"candidate": "t180", "C": 0.1, "val_acc": 0.90},  # tie: the first max wins
        {"candidate": "concat9t", "C": 0.1, "val_acc": 0.99},  # never a fixed view (capacity)
    ]
    p = C.pick_view({"val_table": table, "path": "x"})
    assert p["view"] == "t100" and p["cell_C"] == 1.0 and p["cell_val_column"] == {0.1: 0.80, 1.0: 0.90}
    with pytest.raises(SystemExit):
        C.pick_view({"val_table": [table[-1]], "path": "x"})


def test_every_arm_is_wired_and_named_uniquely():
    assert len({C.slug(a) for a in C.ARMS}) == len(C.ARMS)
    for _, base, plus, null in C.PAIRS + C.CONTROL_PAIRS:
        assert {base, plus, null} <= set(C.ARMS)
        assert C.ARMS[base][1] is None
    assert C.arm_roles("T+FA") == {"T", "FA"} and C.arm_roles("T+noise6") == {"T"}
    assert C.arm_roles("T+anchor") == {"T", "FA"}  # width-matched to the adapted block


# ----------------------------------------------------------------------------- run -> report
def _synthetic_roles(seed=0, n=(900, 450, 900), k=3):
    """T: weak label signal. FA: an INDEPENDENT view of the label (complementary). F0: junk."""
    rng = np.random.default_rng(seed)
    ys = tuple(np.sort(rng.integers(0, k, m)) for m in n)

    def view(width, sd):
        return tuple(
            np.hstack([np.eye(k)[y]] * (width // k + 1))[:, :width]
            + sd * rng.standard_normal((len(y), width))
            for y in ys
        )

    X = {"T": view(12, 2.6), "FA": view(48, 5.0), "F0": tuple(rng.standard_normal((len(y), 48)) for y in ys)}
    paths = [f"data/m_eurosat/test/c{c}/id_{i}.png" for i, c in enumerate(ys[2])]
    return {
        r: {
            "model": C.ROLES[r],
            "X": X[r],
            "y": ys,
            "paths": paths,
            "inputs": {"f": "sha"},
            "pick": {"view": "v", "cell_C": 1.0, "cell_val": 0.0, "cell_val_column": {}},
            "entry": {"kind": "synthetic", "name": C.ROLES[r], "view": "v"},
            "cell": f"cells/{r}",
        }
        for r in X
    }


def _patch(monkeypatch, roles):
    by_model = {v["model"]: v for v in roles.values()}
    monkeypatch.setattr(C, "load_role", lambda cells_dir, model, ds: {**by_model[model]})
    monkeypatch.setattr(C, "load_cell", lambda cells_dir, model, ds: {"config": {"c_grid": [0.01, 1.0]}})
    monkeypatch.setattr(C, "N_BOOT", 2000)


def test_run_and_report_end_to_end(tmp_path, monkeypatch, capsys):
    roles = _synthetic_roles()
    _patch(monkeypatch, roles)
    out = str(tmp_path / "comp")
    base = ["run", "--dataset", "m_eurosat", "--out-dir", out, "--smoke-train", "900", "--c-grid", "0.01,1"]
    assert C.main(base + ["--arms", "T,FA,T+FA,T+null(FA)"]) == 0
    assert C.main(base) == 0  # the rest; the four above are cached
    log = capsys.readouterr().out
    assert log.count(": cached") == 4 and "COMPLEMENTARITY_RUN_OK m_eurosat done=10 cached=4" in log
    arm = np.load(C.arm_path(os.path.join(out, "smoke"), "m_eurosat", "T+FA"), allow_pickle=True)
    assert (
        arm["test_pred"].shape == (2, 900) and int(arm["width_base"]) == 12 and int(arm["width_block"]) == 48
    )
    assert json.loads(str(arm["config"]))["smoke_train"] == 900

    assert C.main(["report", "--datasets", "m_eurosat", "--out-dir", out, "--smoke"]) == 0
    text = capsys.readouterr().out
    assert "SMOKE RUN" in text and "GATE PASS" in text
    res = json.load(open(os.path.join(out, "smoke", "concat.json")))["datasets"]["m_eurosat"]
    assert res["gate"]["a_probe_sanity"]["pass"] and res["gate"]["b_detection_power"]["pass"]
    assert "c_reproduction" not in res["gate"]  # smoke: the cells' val columns are not comparable
    p = res["pairs"]
    assert p["teacher + adapted FLUX"]["verdict"].startswith("ADDS")  # complementary block
    assert p["adapted FLUX + teacher"]["verdict"].startswith("ADDS")
    assert not p["teacher + frozen FLUX"]["verdict"].startswith("ADDS")  # junk block
    assert p["CTRL strong anchor (sd 4)"]["verdict"].startswith("ADDS")
    assert p["teacher + adapted FLUX"]["raw"][0] > 0.03


def test_report_marks_incomplete_gate_and_changed_config_is_refused(tmp_path, monkeypatch, capsys):
    roles = _synthetic_roles(seed=1, n=(300, 150, 300))
    _patch(monkeypatch, roles)
    out = str(tmp_path / "comp")
    base = ["run", "--dataset", "m_eurosat", "--out-dir", out, "--smoke-train", "300"]
    assert C.main(base + ["--c-grid", "0.01,1", "--arms", "T,FA,T+FA,T+null(FA)"]) == 0
    assert C.main(["report", "--datasets", "m_eurosat", "--out-dir", out, "--smoke"]) == 0
    text = capsys.readouterr().out
    assert "GATE INCOMPLETE" in text and "do NOT quote" in text and "MISSING arms" in text
    with pytest.raises(SystemExit, match="config changed"):  # another C grid: never silently reused
        C.main(base + ["--c-grid", "0.1,1", "--arms", "T"])
    assert C.main(base + ["--c-grid", "0.1,1", "--arms", "T", "--redo"]) == 0
    assert os.listdir(os.path.join(out, "smoke", "stale", "m_eurosat"))
    with pytest.raises(SystemExit, match="C grids differ"):  # T was refit on another grid
        C.main(["report", "--datasets", "m_eurosat", "--out-dir", out, "--smoke"])
    with pytest.raises(SystemExit, match="smoke option"):
        C.main(["run", "--dataset", "m_eurosat", "--out-dir", out, "--c-grid", "0.1,1"])


def test_role_guard_fires_when_roles_hold_different_images(monkeypatch):
    roles = _synthetic_roles(seed=2, n=(60, 30, 60))
    C.check_roles(roles)
    moved = {**roles["FA"], "paths": [p.replace("id_", "id_9") for p in roles["FA"]["paths"]]}
    with pytest.raises(SystemExit, match="test image lists differ"):
        C.check_roles({"T": roles["T"], "FA": moved})
    y = roles["T"]["y"]
    relabeled = {**roles["FA"], "y": (y[0], (y[1] + 1) % 3, y[2])}
    with pytest.raises(SystemExit, match="val rows / labels differ"):
        C.check_roles({"T": roles["T"], "FA": relabeled})


def test_pair_stats_refuses_arms_on_different_test_images():
    y = np.repeat(np.arange(2), 10)
    paths = np.array([f"p{i}" for i in range(20)])

    def arm(p):
        pred = np.stack([y, y])
        return {
            "paths": p,
            "labels": y,
            "c_grid": np.array([0.1, 1.0]),
            "sel": 0,
            "vec": [(q == y).astype(np.int8) for q in pred],
        }

    arms = {"T": arm(paths), "T+FA": arm(paths), "T+null(FA)": arm(paths[::-1])}
    with pytest.raises(SystemExit, match="test images differ"):
        C.pair_stats(arms, "T", "T+FA", "T+null(FA)")
