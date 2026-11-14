"""eval.seg_overlap_drawn: which retained rows near a segmentation task a training run drew.

Each guard is fed the mismatch it exists to catch (CLAUDE.md rule 6)."""

import csv
import json
import os

import numpy as np
import pytest
import torch

from eval import seg_overlap_drawn as S

STRIDE = 1 << 20  # datasets.git10m.ROW_ID_STRIDE, written out so a change there fails here
HOLD = 7
STEPS, ACCUM, BATCH, WARM = 5, 2, 2, 3


def _index(tmp, shards=3, per_shard=200):
    shard = np.repeat(np.arange(shards), per_shard)
    row = np.tile(np.arange(per_shard), shards)
    p = os.path.join(tmp, "index.npz")
    np.savez(p, shard=shard, row=row, bucket_code=(row % 5 == 0).astype(np.int8),
             buckets=np.array(["retained", "unassessable"]), one_row_per_image=np.array(True))  # fmt: skip
    hold = (shard.astype(np.int64) * STRIDE + row) % HOLD == 0
    return p, shard, row, np.flatnonzero(~hold), np.flatnonzero(hold)


def _main_pos(train_pos, seed, n=STEPS * ACCUM * BATCH):
    """The main draw written out independently of PermutationSampler."""
    perm = torch.randperm(len(train_pos), generator=torch.Generator().manual_seed(seed * 100_003))
    return train_pos[perm[:n].numpy()]


def _run(tmp, name, seed, index_path, n_train, micro=None, flow=None, **over):
    d = os.path.join(tmp, "runs", name)
    os.makedirs(d)
    cfg = {"seed": seed, "batch_size": BATCH, "gradient_accumulation_steps": ACCUM, "use_gradient_accumulation": True,
           "max_train_steps": STEPS, "max_samples": None, "git10m": {"index": index_path, "holdout_every": HOLD},
           "align": {"proj_warmup_steps": WARM, "resume": ""}}  # fmt: skip
    meta = {"cfg": cfg, "resumed_from": None, "train_size": n_train, "samples_seen": STEPS * ACCUM * BATCH}
    for k, v in over.items():
        (cfg if k in cfg else meta)[k] = v
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    arrays = {"step/flow_loss": np.asarray(flow if flow is not None else [0.5] * STEPS, dtype=np.float32)}
    if micro is not None:
        arrays["step/micro_row"] = np.asarray(micro, dtype=np.int64)
    np.savez(os.path.join(d, "stats.npz"), **arrays)
    return d


def _pair(shard, row, split="test", name="1.0,2.0", dist=5.0):
    return {"shard": int(shard), "row": int(row), "dataset": "m_pv4ger_seg", "split": split, "sample_name": name,
            "distance_m": float(dist), "img_name": f"img_{shard}_{row}.jpg", "status": "retained"}  # fmt: skip


def _write_matches(path, pairs, header=S.MATCH_HEADER):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[p[k] for k in S.MATCH_HEADER] for p in pairs])


def _gates(rep):
    return {g["gate"]: g["pass"] for g in rep["gates"]}


@pytest.fixture
def world(tmp_path):
    tmp = str(tmp_path)
    index_path, shard, row, train_pos, hold_pos = _index(tmp)
    main = _main_pos(train_pos, 1)
    warm = train_pos[S.warm_draw(len(train_pos), 1, BATCH, WARM).ravel()]
    warm_only = [p for p in warm if p not in set(main.tolist())][0]
    k0, k1 = [k for k in range(len(main)) if main[k] not in set(warm.tolist())][
        :2
    ]  # drawn in the main loop only
    undrawn = [p for p in train_pos if p not in set(main.tolist()) | set(warm.tolist())][:2]
    pairs = [
        _pair(shard[main[k0]], row[main[k0]], "test", "a", 0.0),  # drawn in the main loop, overlaps sample a
        _pair(shard[main[k0]], row[main[k0]], "train", "b", 30.0),  # the same row near a second sample
        _pair(shard[main[k1]], row[main[k1]], "valid", "c", 12.5),  # drawn in the main loop
        _pair(shard[warm_only], row[warm_only], "test", "d", 1.0),  # warm-up only
        _pair(shard[hold_pos[0]], row[hold_pos[0]], "test", "e", 2.0),  # holdout: never trained on
        _pair(9, 5, "test", "f", 3.0),  # not in the index
        _pair(shard[undrawn[0]], row[undrawn[0]], "test", "g", 4.0),
        _pair(shard[undrawn[1]], row[undrawn[1]], "train", "h", 0.0),
    ]
    logged = _run(tmp, "logged_s1", 1, index_path, len(train_pos), micro=main.reshape(STEPS, ACCUM * BATCH))
    return {"tmp": tmp, "index": index_path, "train_pos": train_pos, "main": main, "pairs": pairs, "logged": logged,
            "k": (k0, k1)}  # fmt: skip


def test_main_draw_is_the_seeded_permutation(world):
    tp = world["train_pos"]
    got = tp[S.main_draw(len(tp), 1, BATCH, STEPS * ACCUM).ravel()]
    assert np.array_equal(got, world["main"])
    assert not np.array_equal(got, _main_pos(tp, 2))


def test_warm_draw_is_seeded_and_does_not_depend_on_workers():
    a = S.warm_draw(500, 1, BATCH, WARM)
    assert np.array_equal(a, S.warm_draw(500, 1, BATCH, WARM, num_workers=2))  # training used 8 workers
    assert not np.array_equal(a, S.warm_draw(500, 2, BATCH, WARM))
    assert not np.array_equal(a, S.main_draw(500, 1, BATCH, WARM))
    assert S.warm_draw(500, 1, BATCH, 0).shape == (0, BATCH)


def test_a_second_pass_is_refused():
    with pytest.raises(ValueError, match="one pass"):
        S.main_draw(10, 1, BATCH, 6)


def test_read_matches_validates(tmp_path):
    p = str(tmp_path / "m.csv")
    _write_matches(p, [_pair(0, 1)])
    assert S.read_matches(p, "m_pv4ger_seg")[0]["row"] == 1
    with pytest.raises(ValueError, match="no row for dataset"):
        S.read_matches(p, "m_other")
    _write_matches(p, [_pair(0, 1)], header=S.MATCH_HEADER[::-1])
    with pytest.raises(ValueError, match="header"):
        S.read_matches(p, "m_pv4ger_seg")
    _write_matches(p, [_pair(0, 1) | {"status": "excluded"}])
    with pytest.raises(ValueError, match="status"):
        S.read_matches(p, "m_pv4ger_seg")
    _write_matches(p, [_pair(0, 1, dist=float("nan"))])
    with pytest.raises(ValueError, match="distance_m"):
        S.read_matches(p, "m_pv4ger_seg")


def test_counts_only_the_rows_the_run_drew(world):
    rep, rows = S.measure(world["index"], HOLD, world["pairs"], [world["logged"]], 7)
    assert rep["pass"], rep["gates"]
    assert rep["matched_rows"] == {"total": 7, "not_in_index": 1, "in_index": 6, "holdout": 1, "train_view": 5,
                                   "in_index_by_bucket": rep["matched_rows"]["in_index_by_bucket"]}  # fmt: skip
    d = rep["draws"]["1"]
    assert d["main"]["samples"] == STEPS * ACCUM * BATCH and d["main"]["matched_rows_drawn"] == 2
    assert d["main"]["eval_samples_near_a_drawn_row"] == {
        "test": {"within_buffer": 1, "overlapped": 1},
        "train": {"within_buffer": 1, "overlapped": 0},
        "valid": {"within_buffer": 1, "overlapped": 0},
    }
    assert d["warmup"]["matched_rows_drawn"] == 1 and d["main_or_warmup"]["matched_rows_drawn"] == 3
    assert d["main"]["expected_uniform"] == pytest.approx(5 * 20 / len(world["train_pos"]))
    assert rep["runs"] == [{"run": "logged_s1", "seed": 1, "verification": "logged"}]
    got = {(r[1], r[2], r[3], r[4]) for r in rows}
    per_step = ACCUM * BATCH
    for k in world["k"]:
        assert ("main", k // per_step, k % per_step, int(world["main"][k])) in got
    assert sum(r[1] == "warmup" for r in rows) == 1 and len(rows) == 3
    first = next(r for r in rows if r[1] == "main" and r[4] == int(world["main"][world["k"][0]]))
    assert first[8] == 2 and first[9] == "0.000000" and first[10] == "test+train"


@pytest.mark.parametrize("damage", ["one_slot", "unfilled", "shape"])
def test_a_log_that_differs_from_the_reconstruction_fails(world, damage):
    micro = world["main"].reshape(STEPS, ACCUM * BATCH).copy()
    if damage == "one_slot":
        micro[3, 1] = (
            world["train_pos"][-1] if micro[3, 1] != world["train_pos"][-1] else world["train_pos"][0]
        )
    elif damage == "unfilled":
        micro[4, 3] = -1
    else:
        micro = micro[:-1]
    bad = _run(world["tmp"], "bad", 1, world["index"], len(world["train_pos"]), micro=micro)
    rep, _ = S.measure(world["index"], HOLD, world["pairs"], [world["logged"], bad], 7)
    assert not rep["pass"] and not _gates(rep)["row log"]
    assert {r["run"]: r["verification"] for r in rep["runs"]} == {
        "logged_s1": "logged",
        "bad": "LOG MISMATCH",
    }


def test_unlogged_runs_are_labelled_by_their_evidence(world):
    n, t = len(world["train_pos"]), world["tmp"]
    same = _run(t, "same_start", 1, world["index"], n, flow=[0.5, 0.4, 0.3, 0.2, 0.1])
    other = _run(t, "other_start", 1, world["index"], n, flow=[0.6, 0.5, 0.5, 0.5, 0.5])
    lone = _run(t, "no_logged_peer", 3, world["index"], n)
    rep, _ = S.measure(world["index"], HOLD, world["pairs"], [world["logged"], same, other, lone], 7)
    assert rep["pass"]
    v = {r["run"]: r for r in rep["runs"]}
    assert v["same_start"]["verification"] == "fingerprint" and v["same_start"]["reference"] == "logged_s1"
    assert v["same_start"]["flow_loss_median_abs_diff"] == pytest.approx(0.2, abs=1e-6)
    assert (
        v["other_start"]["verification"] == "unverified"
        and v["no_logged_peer"]["verification"] == "unverified"
    )
    assert "reference" not in v["no_logged_peer"]
    assert rep["verification_counts"] == {"logged": 1, "fingerprint": 1, "unverified": 2, "LOG MISMATCH": 0}
    assert sorted(rep["draws"]) == ["1", "3"] and rep["draws"]["1"]["runs"] == [
        "logged_s1",
        "same_start",
        "other_start",
    ]


def test_no_logged_run_at_all_fails(world):
    lone = _run(world["tmp"], "lone", 1, world["index"], len(world["train_pos"]))
    rep, _ = S.measure(world["index"], HOLD, world["pairs"], [lone], 7)
    assert not rep["pass"] and not _gates(rep)["row log"]


@pytest.mark.parametrize(
    "over, gate",
    [
        ({"resumed_from": "ckpt_s500.pt"}, "loader config"),
        ({"max_samples": 64}, "loader config"),
        ({"git10m": {"index": "/elsewhere/index.npz", "holdout_every": HOLD}}, "loader config"),
        ({"git10m": {"index": None, "holdout_every": HOLD + 1}}, "loader config"),
        ({"samples_seen": 21}, "loader config"),
        ({"train_size": 3}, "train size"),
    ],
)
def test_a_run_from_another_loader_fails(world, over, gate):
    if "git10m" in over and over["git10m"]["index"] is None:
        over = {"git10m": {"index": world["index"], "holdout_every": HOLD + 1}}
    odd = _run(world["tmp"], "odd", 1, world["index"], len(world["train_pos"]), **over)
    rep, _ = S.measure(world["index"], HOLD, world["pairs"], [world["logged"], odd], 7)
    assert not rep["pass"] and not _gates(rep)[gate]


def test_the_match_count_and_the_control_can_fail(world, monkeypatch):
    rep, _ = S.measure(world["index"], HOLD, world["pairs"], [world["logged"]], 8)
    assert not rep["pass"] and not _gates(rep)["matched rows"]
    monkeypatch.setattr(S, "count_drawn", lambda match_pos, drawn_pos: np.ones(len(match_pos), dtype=bool))
    rep, _ = S.measure(world["index"], HOLD, world["pairs"], [world["logged"]], 7)
    assert not rep["pass"] and not _gates(rep)["control"]


def test_an_index_with_a_repeated_row_is_refused(tmp_path):
    p = str(tmp_path / "dup.npz")
    np.savez(p, shard=np.array([0, 0]), row=np.array([4, 4]), bucket_code=np.zeros(2, np.int8),
             buckets=np.array(["retained"]), one_row_per_image=np.array(True))  # fmt: skip
    with pytest.raises(ValueError, match="twice"):
        S.load_index(p, HOLD)


def _argv(world, out, n_runs=1):
    m, o = os.path.join(world["tmp"], "m.csv"), os.path.join(world["tmp"], "ov.json")
    if not os.path.exists(m):
        _write_matches(m, world["pairs"])
        with open(o, "w") as f:
            json.dump(
                {"buffer_m": 1000.0, "tasks": {"m_pv4ger_seg": {"overlap": {"of_which_retained": 7}}}}, f
            )
    return ["--index", world["index"], "--holdout-every", str(HOLD), "--matches", m, "--overlap-json", o,
            "--dataset", "m_pv4ger_seg", "--runs-root", os.path.join(world["tmp"], "runs"),
            "--expect-runs", str(n_runs), "--out", out]  # fmt: skip


def test_main_writes_once_and_only_on_pass(world, capsys):
    out = os.path.join(world["tmp"], "res", "drawn")
    with pytest.raises(SystemExit, match="expected 2"):
        S.main(_argv(world, out, n_runs=2))
    assert S.main(_argv(world, out)) == 0
    text = capsys.readouterr().out
    assert "GATE seg-overlap-drawn PASS: 6/6 checks" in text and "SEG_OVERLAP_DRAWN_OK" in text
    with open(out + ".json") as f:
        rep = json.load(f)
    assert rep["dataset"] == "m_pv4ger_seg" and rep["draws"]["1"]["main"]["matched_rows_drawn"] == 2
    with open(out + ".drawn_rows.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == S.ROW_HEADER and len(rows) == 4
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.main(_argv(world, out))
    _run(world["tmp"], "resumed", 1, world["index"], len(world["train_pos"]), resumed_from="ckpt.pt")
    out2 = os.path.join(world["tmp"], "res", "drawn2")
    assert S.main(_argv(world, out2, n_runs=2)) == 1
    assert "do NOT quote" in capsys.readouterr().out
    assert not os.path.exists(out2 + ".json") and not os.path.exists(out2 + ".drawn_rows.csv")
