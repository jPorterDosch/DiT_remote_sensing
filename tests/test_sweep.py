"""Fast synthetic tests for eval/sweep.py's rule-1 machinery (no caches, no GPU):
selection reads val only, exactly one sealed test vector can be opened, and blocks from
different test sets / protocols cannot be mixed into one sweep."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval import sweep

CANDS = ["t100", "t180", "concat2t"]
C_GRID = (0.01, 0.1, 1.0)
N_TEST = 12


def write_block(path, k, val_accs, test_paths=None, pins=None, test_acc_of=None, dataset="m_eurosat"):
    """A block result file. test_acc_of(cand, C) -> fraction correct of that cell's sealed
    predictions; defaults to the INVERSE of val accuracy so a selector that peeked at test
    would pick a different cell."""
    y = np.arange(N_TEST) % 3
    rows, sealed = [], {}
    for i, c in enumerate(CANDS):
        for j, Cv in enumerate(C_GRID):
            va = val_accs[i][j]
            rows.append({"candidate": c, "C": Cv, "val_acc": va, "conv_warnings": 0})
            frac = test_acc_of(c, Cv) if test_acc_of else 1.0 - va
            n_ok = int(round(frac * N_TEST))
            pred = y.copy()
            pred[n_ok:] = (pred[n_ok:] + 1) % 3
            sealed[f"sealed__{c}__C{Cv}"] = pred
    np.savez(
        path,
        k=np.array(k),
        dataset=np.array(dataset),
        arm=np.array("flux-test"),
        candidates=np.array(CANDS),
        C_grid=np.array(C_GRID),
        val_table=np.array(json.dumps(rows)),
        timesteps=np.array([100, 180]),
        test_paths=np.array(test_paths or [f"test/c/{i}.png" for i in range(N_TEST)]),
        test_labels=y,
        pins=np.array(json.dumps(pins or {"ensemble_size": 8, "k": k})),
        run_name=np.array(f"blk{k}"),
        smoke=np.array(True),
        **sealed,
    )
    return str(path)


def flat(v):
    return [[v] * 3 for _ in CANDS]


def test_select_uses_val_and_first_max(tmp_path):
    a = flat(0.5)
    a[1][2] = 0.9  # k=19 t180 C=1.0
    b = flat(0.5)
    b[0][0] = 0.9  # k=33 t100 C=0.01: ties with k=19's best -> first (lower k) wins
    p1 = write_block(tmp_path / "b19.npz", 19, a)
    p2 = write_block(tmp_path / "b33.npz", 33, b)
    blocks, _, _ = sweep.load_blocks([p1, p2], "m_eurosat")
    best, grid = sweep.select(blocks)
    assert (best["k"], best["candidate"], best["C"]) == (19, "t180", 1.0)
    assert len(grid) == 2 * len(CANDS) * len(C_GRID)


def test_seal_allows_exactly_one_open(tmp_path):
    p = write_block(tmp_path / "b19.npz", 19, flat(0.5))
    blocks, _, _ = sweep.load_blocks([p], "m_eurosat")
    sealed = sweep.SealedTests(blocks)
    sealed.open(19, "t100", 0.1)
    with pytest.raises(RuntimeError, match="seal violation"):
        sealed.open(19, "t180", 0.1)


def test_mixed_test_sets_refused(tmp_path):
    p1 = write_block(tmp_path / "b19.npz", 19, flat(0.5))
    p2 = write_block(
        tmp_path / "b33.npz", 33, flat(0.5), test_paths=[f"other/{i}.png" for i in range(N_TEST)]
    )
    with pytest.raises(SystemExit, match="not one sweep"):
        sweep.load_blocks([p1, p2], "m_eurosat")


def test_mixed_pins_refused(tmp_path):
    p1 = write_block(tmp_path / "b19.npz", 19, flat(0.5), pins={"ensemble_size": 8, "k": 19})
    p2 = write_block(tmp_path / "b33.npz", 33, flat(0.5), pins={"ensemble_size": 1, "k": 33})
    with pytest.raises(SystemExit, match="not one sweep"):
        sweep.load_blocks([p1, p2], "m_eurosat")


def test_duplicate_block_refused(tmp_path):
    p1 = write_block(tmp_path / "a.npz", 19, flat(0.5))
    p2 = write_block(tmp_path / "b.npz", 19, flat(0.5))
    with pytest.raises(SystemExit, match="two result files for block k=19"):
        sweep.load_blocks([p1, p2], "m_eurosat")


def test_select_end_to_end_reports_selected_cell_only(tmp_path, monkeypatch):
    """run_select must report the val-selected cell's test accuracy, not the test-best one."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    a = flat(0.5)
    a[2][1] = 0.8  # k=19 concat2t C=0.1 is val-best; its test acc is set to 0.25 below
    p = write_block(
        tmp_path / "b19.npz",
        19,
        a,
        test_acc_of=lambda c, Cv: 0.25 if (c, Cv) == ("concat2t", 0.1) else 1.0,
    )
    out = sweep.main(
        ["select", "--dataset", "m_eurosat", "--arm", "flux-test", "--blocks", p, "--out-dir", str(tmp_path)]
    )
    d = np.load(out, allow_pickle=True)
    assert abs(d["correct__test"].mean() - 0.25) < 1e-9
    assert json.loads(str(d["info"]))["selected"]["candidate"] == "concat2t"


def test_sweep_is_dataset_agnostic(tmp_path, monkeypatch):
    """A task registered ONLY in features.OFFICIAL runs select end to end: nothing in the
    sweep depends on m-eurosat (the GEO-Bench tasks arrive in a separate PR)."""
    from eval import features as F

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setitem(
        F.OFFICIAL,
        "fake_task",
        {
            "sizes": {"train": 30, "val": N_TEST, "test": N_TEST},
            "root": "data/fake",
            "classes": lambda: ["a", "b", "c"],
        },
    )
    a = flat(0.5)
    a[0][1] = 0.7
    p = write_block(tmp_path / "b24.npz", 24, a, dataset="fake_task")
    out = sweep.main(
        ["select", "--dataset", "fake_task", "--arm", "flux-test", "--blocks", p, "--out-dir", str(tmp_path)]
    )
    d = np.load(out, allow_pickle=True)
    assert str(d["dataset"]) == "fake_task"
    assert json.loads(str(d["info"]))["selected"]["candidate"] == "t100"


def test_unregistered_dataset_refused():
    from eval import features as F

    with pytest.raises(SystemExit, match="not registered in eval/features.OFFICIAL"):
        F.official_classes("m_forestnet_unregistered")


def test_verify_block_catches_truncated_file(tmp_path):
    """verify_block derives the expected cells from the t-grid, so a file missing one
    sealed vector fails even though its own candidate list looks consistent."""
    p = write_block(tmp_path / "b19.npz", 19, flat(0.5))
    # write_block's grid is t100, t180 + concat2t -> complete file passes
    sweep.verify_block(p)
    z = dict(np.load(p, allow_pickle=True))
    del z["sealed__t180__C0.1"]
    bad = str(tmp_path / "trunc.npz")
    np.savez(bad, **z)
    with pytest.raises(SystemExit, match="cells incomplete"):
        sweep.verify_block(bad)


def _write_shard(d, idx, seed, n_per=3, n_shards=2):
    import os

    run = os.path.join(d, f"run{idx}")
    os.makedirs(run, exist_ok=True)
    path = os.path.join(run, "multistep_train_feats_inversion_g1.0_n50.npz")
    sub = np.arange(idx * n_per, (idx + 1) * n_per)
    np.savez(
        path,
        feats=np.zeros((n_per, 2, 4), np.float32),
        labels=sub % 2,
        subset_indices=sub,
        paths=np.array([f"train/c/{i}.png" for i in sub]),
        timesteps=np.array([100, 180]),
    )
    meta = {
        "split": "train",
        "guidance_scale": 1.0,
        "k": 28,
        "weights": "flux-dev",
        "dataset": "fake",
        "extraction_mode": "INVERSION",
        "num_inversion_steps": 50,
        "num_shards": n_shards,
        "shard_index": idx,
        "seed": seed,
        "eps_seed": None,
    }
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f)


@pytest.mark.parametrize("seeds, warns", [((42, 42), True), ((100, 101), False)])
def test_shard_seed_warning(tmp_path, capsys, seeds, warns):
    """Shards of one split sharing a seed draw position-paired noise (audit F1); the merge
    must say so. Distinct per-shard seeds must stay silent."""
    from eval import features as F

    for i, s in enumerate(seeds):
        _write_shard(str(tmp_path), i, s)
    pins = {
        "guidance_scale": 1.0,
        "k": 28,
        "weights": "flux-dev",
        "dataset": "fake",
        "extraction_mode": "INVERSION",
        "num_inversion_steps": 50,
    }
    pattern = str(tmp_path / "*" / "multistep_{split}_feats_inversion_g1.0_n50.npz")
    feats, *_ = F.load_flux_split(pattern, pins, "train", 6)
    assert feats.shape == (6, 2, 4)
    assert ("WARNING train: shards share" in capsys.readouterr().out) == warns
