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
        [
            "select",
            "--dataset",
            "m_eurosat",
            "--arm",
            "flux-test",
            "--blocks",
            p,
            "--out-dir",
            str(tmp_path),
            "--skip-control",
        ]
    )
    d = np.load(out, allow_pickle=True)
    assert abs(d["correct__test"].mean() - 0.25) < 1e-9
    assert json.loads(str(d["info"]))["selected"]["candidate"] == "concat2t"


def test_sweep_is_dataset_agnostic(tmp_path, monkeypatch):
    """A task registered ONLY in features.OFFICIAL runs select end to end: nothing in the
    sweep depends on m-eurosat."""
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
        [
            "select",
            "--dataset",
            "fake_task",
            "--arm",
            "flux-test",
            "--blocks",
            p,
            "--out-dir",
            str(tmp_path),
            "--skip-control",
        ]
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


def _write_shard(
    d, idx, seed, n_per=3, n_shards=2, split="train", mode="INVERSION", eps_seed=None, run=None, base=0
):
    """One fake run.py cache: shard idx of n_shards (1 = unsharded) of one split."""
    import os

    run = os.path.join(d, run or f"run_{split}{idx}")
    os.makedirs(run, exist_ok=True)
    fname = "multistep_{split}_feats_inversion_g1.0_n50.npz" if mode == "INVERSION" else FNAME_ONESHOT
    path = os.path.join(run, fname.format(split=split))
    sub = np.arange(base + idx * n_per, base + (idx + 1) * n_per)
    np.savez(
        path,
        feats=np.zeros((n_per, 2, 4), np.float32),
        labels=sub % 2,
        subset_indices=sub - base,
        paths=np.array([f"{split}/c/{i}.png" for i in sub]),
        timesteps=np.array([100, 180]),
    )
    meta = {
        "split": split,
        "guidance_scale": 1.0,
        "k": 28,
        "weights": "flux-dev",
        "dataset": "fake",
        "extraction_mode": mode,
        "num_inversion_steps": 50 if mode == "INVERSION" else None,
        "ensemble_size": None if mode == "INVERSION" else 2,
        "num_shards": n_shards,
        "shard_index": idx,
        "seed": seed,
        "eps_seed": eps_seed,
    }
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f)
    return path


FNAME_ONESHOT = "multistep_{split}_feats_oneshot_g1.0.npz"
PINS_INV = {
    "guidance_scale": 1.0,
    "k": 28,
    "weights": "flux-dev",
    "dataset": "fake",
    "extraction_mode": "INVERSION",
    "num_inversion_steps": 50,
}
PINS_ONESHOT = {
    "guidance_scale": 1.0,
    "k": 28,
    "weights": "flux-dev",
    "dataset": "fake",
    "extraction_mode": "ONESHOT",
    "ensemble_size": 2,
}


@pytest.mark.parametrize("seeds, warns", [((42, 42), True), ((100, 101), False)])
def test_shard_seed_warning(tmp_path, capsys, seeds, warns):
    """Shards of one split sharing a seed draw position-paired noise (audit F1); the merge
    must say so. Distinct per-shard seeds must stay silent."""
    from eval import features as F

    for i, s in enumerate(seeds):
        _write_shard(str(tmp_path), i, s)
    pattern = str(tmp_path / "*" / "multistep_{split}_feats_inversion_g1.0_n50.npz")
    feats, *_ = F.load_flux_split(pattern, PINS_INV, "train", 6)
    assert feats.shape == (6, 2, 4)
    assert ("WARNING train: shards share" in capsys.readouterr().out) == warns


def test_mixed_shard_generations_refused(tmp_path):
    """Banked shared-seed shards beside a per-shard-seed re-extraction (inversion.sbatch's
    documented situation) must be refused, not merged or die on the range check."""
    from eval import features as F

    for i, s in enumerate((42, 42)):
        _write_shard(str(tmp_path), i, s, run=f"old{i}")
    for i, s in enumerate((100, 101)):
        _write_shard(str(tmp_path), i, s, run=f"new{i}")
    pattern = str(tmp_path / "*" / "multistep_{split}_feats_inversion_g1.0_n50.npz")
    with pytest.raises(SystemExit, match="more than one extraction generation"):
        F.load_flux_split(pattern, PINS_INV, "train", 6)


def test_unsharded_multi_hit_refused(tmp_path):
    """Two unsharded caches under one glob used to die with a raw KeyError('shard_index')."""
    from eval import features as F

    _write_shard(str(tmp_path), 0, 42, n_shards=1, run="a")
    _write_shard(str(tmp_path), 0, 43, n_shards=1, run="b")
    pattern = str(tmp_path / "*" / "multistep_{split}_feats_inversion_g1.0_n50.npz")
    with pytest.raises(SystemExit, match="unsharded"):
        F.load_flux_split(pattern, PINS_INV, "train", 3)


def _fake_official(monkeypatch, n_train):
    from eval import features as F

    monkeypatch.setitem(
        F.OFFICIAL,
        "fake",
        {
            "sizes": {"train": n_train, "val": 3, "test": 3},
            "root": "data/fake",
            "classes": lambda: ["a", "b"],
        },
    )


def _official_flux(tmp_path, pins):
    from types import SimpleNamespace

    from eval import probe as PR

    fname = (
        FNAME_ONESHOT
        if pins["extraction_mode"] == "ONESHOT"
        else "multistep_{split}_feats_inversion_g1.0_n50.npz"
    )
    ns = SimpleNamespace(
        dataset="fake", features=str(tmp_path / "*" / fname), pins=pins, view="all", sizes=None
    )
    return PR.official_flux(ns)


@pytest.mark.parametrize("test_eps, fires", [(43, True), (45, False)])
def test_cross_split_eps_guard_sees_every_shard(tmp_path, monkeypatch, test_eps, fires):
    """ONESHOT train shards seeded 42, 43 + test eps 43: the collision is in shard 1, which
    a guard reading only shard 0's meta cannot see (rule 6). Distinct seeds must pass."""
    _fake_official(monkeypatch, 6)
    for i, s in enumerate((42, 43)):
        _write_shard(str(tmp_path), i, s, mode="ONESHOT", eps_seed=s)
    _write_shard(str(tmp_path), 0, 44, n_shards=1, split="val", mode="ONESHOT", eps_seed=44)
    _write_shard(str(tmp_path), 0, test_eps, n_shards=1, split="test", mode="ONESHOT", eps_seed=test_eps)
    if fires:
        with pytest.raises(SystemExit, match="eps_seed shared between train and test"):
            _official_flux(tmp_path, PINS_ONESHOT)
    else:
        cands, ytr, yva, yte, *_ = _official_flux(tmp_path, PINS_ONESHOT)
        assert len(ytr) == 6 and len(yva) == 3 and len(yte) == 3 and "concat2t" in cands


def test_cross_split_seed_guard_inversion(tmp_path, monkeypatch):
    """INVERSION draws no eps; its VAE-posterior stream comes from --seed, so the same
    cross-split check runs on seed (inversion.sbatch header, 6t F1)."""
    _fake_official(monkeypatch, 6)
    for i in range(2):
        _write_shard(str(tmp_path), i, 42)
    _write_shard(str(tmp_path), 0, 44, n_shards=1, split="val")
    _write_shard(str(tmp_path), 0, 42, n_shards=1, split="test")
    with pytest.raises(SystemExit, match="seed shared between train and test"):
        _official_flux(tmp_path, PINS_INV)


def test_banked_absolute_paths_join_the_sweep(tmp_path):
    """m-eurosat's banked k=28 caches were extracted with an absolute --dataset.path, the sweep
    blocks with a relative one: the same images must still form one sweep, while a truly
    different image is still refused."""
    rel = [f"data/m_eurosat_rgb/test/c/{i}.png" for i in range(N_TEST)]
    absolute = [f"/nfs/home/u/repo/{p}" for p in rel]
    p1 = write_block(tmp_path / "b28.npz", 28, flat(0.5), test_paths=absolute)
    p2 = write_block(tmp_path / "b33.npz", 33, flat(0.5), test_paths=rel)
    blocks, _, _ = sweep.load_blocks([p1, p2], "m_eurosat")
    assert sorted(blocks) == [28, 33]
    other = rel[:-1] + ["data/m_eurosat_rgb/test/c/999.png"]
    p3 = write_block(tmp_path / "b38.npz", 38, flat(0.5), test_paths=other)
    with pytest.raises(SystemExit, match="not one sweep"):
        sweep.load_blocks([p1, p3], "m_eurosat")


def _with_val_cache(tmp_path, path, sha_of=None, labels=None):
    """Record a val cache (labels in subset order) in a block's config, as run_block does."""
    from eval import wb

    v = tmp_path / "val_cache.npz"
    if not v.exists():
        y = labels if labels is not None else np.arange(N_TEST) % 3
        np.savez(v, labels=y, subset_indices=np.arange(len(y)))
    ident = wb.file_identity(str(v))
    if sha_of:
        ident = {**ident, "sha1": sha_of}
    d = dict(np.load(path, allow_pickle=True))
    d.pop("config", None)
    cfg = {
        "features": {
            "train": [{"path": "t.npz", "sha1": "a"}],
            "val": [ident],
            "test": [{"path": "x", "sha1": "b"}],
        }
    }
    np.savez(path, **d, config=np.array(json.dumps(cfg)))
    return str(v)


def _select(tmp_path, p, *extra):
    return sweep.main(["select", "--dataset", "m_eurosat", "--arm", "flux-test", "--blocks", p,
                       "--out-dir", str(tmp_path), *extra])  # fmt: skip


def test_select_control_gates_the_test_vector(tmp_path, monkeypatch):
    """The positive control runs before the one test use and can FAIL (rule 6): a selected val
    cell at the constant-predictor rate keeps the test sealed; a real one passes and is recorded."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    p = write_block(tmp_path / "b19.npz", 19, flat(0.40))  # labels balanced 1/3: skill 0.10 < 0.25
    _with_val_cache(tmp_path, p)
    with pytest.raises(SystemExit, match="control FAILED"):
        _select(tmp_path, p)
    assert not list(tmp_path.glob("sweep-select_*.npz"))  # nothing reported
    p = write_block(tmp_path / "b19.npz", 19, flat(0.90))
    _with_val_cache(tmp_path, p)
    out = _select(tmp_path, p)
    c = json.loads(str(np.load(out, allow_pickle=True)["info"]))["control"]
    assert c["pass"] and abs(c["chance"] - 1 / 3) < 1e-9


def test_select_control_refuses_missing_or_changed_val_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    p = write_block(tmp_path / "b19.npz", 19, flat(0.90))
    with pytest.raises(SystemExit, match="do not all record one val cache"):
        _select(tmp_path, p)  # legacy block: no recorded val cache
    _with_val_cache(tmp_path, p, sha_of="0" * 40)
    with pytest.raises(SystemExit, match="missing or changed"):
        _select(tmp_path, p)


def test_blocks_from_different_extractions_refused(tmp_path):
    """Same test set / pins / grid but different recorded input caches: not one sweep."""
    p1 = write_block(tmp_path / "b19.npz", 19, flat(0.5))
    p2 = write_block(tmp_path / "b33.npz", 33, flat(0.5))
    _with_val_cache(tmp_path, p1)
    _with_val_cache(tmp_path, p2)
    sweep.load_blocks([p1, p2], "m_eurosat")  # same recorded inputs: one sweep
    _with_val_cache(tmp_path, p2, sha_of="f" * 40)
    with pytest.raises(SystemExit, match="not one sweep"):
        sweep.load_blocks([p1, p2], "m_eurosat")


def test_control_skill_is_imbalance_aware():
    assert sweep.control_skill(0.9399, 0.8468) > sweep.CONTROL_MIN_SKILL  # m-pv4ger: passes on skill
    assert 0.9399 - 0.8468 < 0.10  # ...though the old absolute rule failed it
    assert sweep.control_skill(0.40, 1 / 3) < sweep.CONTROL_MIN_SKILL
    assert sweep.control_skill(1.0, 1.0) == 0.0


# ------------------------------------------------------------------ plot: per-cell best C
def _grid():
    # cell (28, t100) peaks at C=0.1; cell (28, t180) at C=0.01 and is the global best
    vals = {(28, "t100"): {0.01: 0.50, 0.1: 0.60}, (28, "t180"): {0.01: 0.70, 0.1: 0.55},
            (33, "t100"): {0.01: 0.40, 0.1: 0.45}, (33, "t180"): {0.01: 0.30, 0.1: 0.65}}  # fmt: skip
    return [
        {"k": k, "candidate": c, "C": C, "val_acc": v} for (k, c), d in vals.items() for C, v in d.items()
    ]


def test_best_over_C_shows_each_cells_own_best():
    M, Cb = sweep.best_over_C(_grid(), [28, 33], ["t100", "t180"])
    assert np.allclose(M, [[0.60, 0.70], [0.45, 0.65]]) and np.allclose(Cb, [[0.1, 0.01], [0.1, 0.1]])


def test_plot_outlines_selected_cell_at_its_own_value(tmp_path):
    best = {"k": 28, "candidate": "t180", "C": 0.01, "val_acc": 0.70}
    out = tmp_path / "g.png"
    sweep.plot(_grid(), best, ["t100", "t180"], "t", str(out), test_score=0.68)
    assert out.exists() and out.stat().st_size > 0


def test_plot_refuses_a_selection_the_grid_contradicts(tmp_path):
    # the old grid-wide-C bug in reverse: a "selected" value that is not the cell's best-over-C
    bad = {"k": 28, "candidate": "t180", "C": 0.1, "val_acc": 0.55}
    with pytest.raises(RuntimeError, match="plot/select mismatch"):
        sweep.plot(_grid(), bad, ["t100", "t180"], "t", str(tmp_path / "g.png"))
    not_max = {"k": 28, "candidate": "t100", "C": 0.1, "val_acc": 0.60}  # a cell beats it
    with pytest.raises(RuntimeError, match="exceeds the selected val"):
        sweep.plot(_grid(), not_max, ["t100", "t180"], "t", str(tmp_path / "g.png"))


# ------------------------------------------------------------------ --head tsweight blocks
def _tsw_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _fake_official(monkeypatch, 6)
    d = str(tmp_path / "caches")
    _write_shard(d, 0, 42, n_per=6, n_shards=1, mode="ONESHOT", eps_seed=42)
    _write_shard(d, 0, 44, n_shards=1, split="val", mode="ONESHOT", eps_seed=44)
    _write_shard(d, 0, 43, n_shards=1, split="test", mode="ONESHOT", eps_seed=43)
    return d, str(tmp_path / "caches" / "*" / FNAME_ONESHOT)


def _tsw_block(tmp_path, pattern, *extra):
    return sweep.main(["block", "--dataset", "fake", "--k", "28", "--arm", "flux-tsw", "--features", pattern,
                       "--expect", "extraction_mode=ONESHOT", "ensemble_size=2", "--out-dir", str(tmp_path / "res"),
                       *extra])  # fmt: skip


def test_tsweight_block_end_to_end_and_head_guards(tmp_path, monkeypatch):
    cache_dir, pattern = _tsw_caches(tmp_path, monkeypatch)
    out = _tsw_block(tmp_path, pattern, "--head", "tsweight")
    d = np.load(out, allow_pickle=True)
    assert sweep.head_of(d) == "tsweight" and list(d["candidates"]) == ["tsweight"]
    assert tuple(d["C_grid"]) == sweep.TSWEIGHT_GRID and "sealed__tsweight__C0.0" in d.files
    row = json.loads(str(d["val_table"]))[0]
    assert row["slots"] == ["t100", "t180"] and np.isclose(sum(row["slot_weights"]), 1, atol=1e-5)
    sweep.verify_block(out, expect_head="tsweight")
    with pytest.raises(SystemExit, match="block head 'tsweight' != expected 'lr'"):
        sweep.verify_block(out)  # an lr caller must not accept a tsweight block (rule 6)
    assert sweep.done_status(out, pattern, "fake", (0.01,), cache_dir, head="tsweight")[0] == 0
    assert sweep.done_status(out, pattern, "fake", (0.01,), cache_dir, head="lr")[0] == 5
    with pytest.raises(SystemExit, match="no C grid"):
        _tsw_block(tmp_path, pattern, "--head", "tsweight", "--c-grid", "0.1")


def test_lr_and_tsweight_blocks_are_not_one_sweep(tmp_path, monkeypatch):
    _, pattern = _tsw_caches(tmp_path, monkeypatch)
    p_t = _tsw_block(tmp_path, pattern, "--head", "tsweight")
    p_l = _tsw_block(tmp_path / "lr", pattern)
    d = dict(np.load(p_l, allow_pickle=True))
    d["k"] = np.array(33)  # another block, so the protocol-identity check (not the duplicate-k check) decides
    p_l33 = str(tmp_path / "lr33.npz")
    np.savez(p_l33, **d)
    with pytest.raises(SystemExit, match="not one sweep"):
        sweep.load_blocks([p_t, p_l33], "fake")


def test_tsweight_fusion_grid_recorded_and_enforced(tmp_path, monkeypatch):
    cache_dir, pattern = _tsw_caches(tmp_path, monkeypatch)
    p_reg = _tsw_block(tmp_path / "reg", pattern, "--head", "tsweight", "--fusion-grid", "reg")
    d = np.load(p_reg, allow_pickle=True)
    assert sweep.fusion_grid_of(d) == "reg"
    assert sweep.done_status(p_reg, pattern, "fake", (0.01,), cache_dir, "tsweight", "reg")[0] == 0
    assert sweep.done_status(p_reg, pattern, "fake", (0.01,), cache_dir, "tsweight", "satdifuser")[0] == 3
    p_sd = _tsw_block(tmp_path / "sd", pattern, "--head", "tsweight")
    dd = dict(np.load(p_sd, allow_pickle=True))
    dd["k"] = np.array(33)
    p_sd33 = str(tmp_path / "sd33.npz")
    np.savez(p_sd33, **dd)
    with pytest.raises(SystemExit, match="not one sweep"):
        sweep.load_blocks([p_reg, p_sd33], "fake")
    with pytest.raises(SystemExit, match="applies to --head tsweight only"):
        _tsw_block(tmp_path / "x", pattern, "--fusion-grid", "reg")


def _val_cache_at(tmp_path, name, labels=None):
    v = tmp_path / name
    y = labels if labels is not None else np.arange(N_TEST) % 3
    np.savez(v, labels=y, subset_indices=np.arange(len(y)))
    return str(v)


def _record_inputs(path, train, val, test):
    """Record one cache per split in a block's config (sweep_block.sh's per-block layout)."""
    from eval import wb

    d = dict(np.load(path, allow_pickle=True))
    d.pop("config", None)
    cfg = {"features": {"train": [{"path": train, "sha1": "a"}], "val": [wb.file_identity(val)],
                        "test": [{"path": test, "sha1": "b"}]}}  # fmt: skip
    np.savez(path, **d, config=np.array(json.dumps(cfg)))


def test_per_block_layout_is_one_sweep(tmp_path, monkeypatch):
    """sweep_block.sh probes block k on its own k<K>/ caches, so blocks share no input path:
    still one sweep (same pins / test set), and the control reads every block's val cache,
    which must all hold the same labels."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    p1 = write_block(tmp_path / "b19.npz", 19, flat(0.90))
    p2 = write_block(tmp_path / "b33.npz", 33, flat(0.90))
    _record_inputs(p1, "k19/train.npz", _val_cache_at(tmp_path, "k19_val.npz"), "k19/test.npz")
    _record_inputs(p2, "k33/train.npz", _val_cache_at(tmp_path, "k33_val.npz"), "k33/test.npz")
    args = [
        "select",
        "--dataset",
        "m_eurosat",
        "--arm",
        "flux-test",
        "--blocks",
        p1,
        p2,
        "--out-dir",
        str(tmp_path),
    ]
    info = json.loads(str(np.load(sweep.main(args), allow_pickle=True)["info"]))
    assert info["control"]["pass"] and info["compute"] == {"19": {}, "33": {}}  # legacy blocks: no stamp
    other = _val_cache_at(tmp_path, "k33_val.npz", labels=(np.arange(N_TEST) + 1) % 3)
    _record_inputs(p2, "k33/train.npz", other, "k33/test.npz")
    with pytest.raises(SystemExit, match="different labels"):
        sweep.main(args)


def test_block_compute_stamps_the_protocol_thread_count():
    from eval import protocols as P

    c = sweep.block_compute("lr", multilabel=False)
    assert c["blas_threads"] == sweep.SWEEP_FIT_THREADS == 8 and c["ovr_n_jobs"] == 1 and c["cpus"] >= 1
    assert sweep.block_compute("lr", multilabel=True)["blas_threads"] == P.OVR_FIT_THREADS
    t = sweep.block_compute("tsweight", False, {"device": "cpu", "torch_threads": 4})
    assert (t["device"], t["torch_threads"]) == ("cpu", 4)
