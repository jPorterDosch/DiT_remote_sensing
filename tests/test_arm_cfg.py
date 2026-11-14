"""check_arm_cfg.py: an arm may differ from its reference run in declared fields only.

Each guard is fed the mismatch it exists to catch (CLAUDE.md rule 6). No torch."""

import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "isaac", "git10m"))
import check_arm_cfg as C  # noqa: E402

REF = "git10m_flux_aaaa+42+proj-0000"


def _meta(**align):
    a = {
        "arm": "proj",
        "feat": "ditf",
        "target": "dinov3_vit7b16_sat",
        "weight": 0.5,
        "ckpt_every": 500,
        "resume": "",
    }
    a.update(align)
    return {
        "cfg": {"seed": 42, "k": 28, "hash": "aaaa", "save_dir": "/somewhere", "align": a},
        "train_data": {
            "index": "idx.npz",
            "n": 100,
            "flip_seed": 42,
            "manifest_sha256": {"retained.csv": "d4"},
        },
        "align": {
            "arm": a["arm"],
            "discard_channels": [154, 1446],
            "target": {"name": a["target"], "norm": [[0.4], [0.2]]},
        },
    }


def _write(root, name, meta):
    os.makedirs(os.path.join(root, name))
    with open(os.path.join(root, name, "meta.json"), "w") as f:
        json.dump(meta, f)


def _expect(tmp_path, runs):
    p = str(tmp_path / "expect.json")
    with open(p, "w") as f:
        json.dump({"reference": REF, "runs": runs}, f)
    return p


def _cmp(run_meta, declared):
    return C.compare(C.view(_meta()), C.view(run_meta), declared)


def test_identical_run_passes_with_nothing_declared():
    r = _cmp(_meta(), {})
    assert r["pass"] and not r["declared"] and not r["undeclared"]


def test_save_dir_is_not_compared():
    m = _meta()
    m["cfg"]["save_dir"] = "/elsewhere"
    assert _cmp(m, {})["pass"]


def test_undeclared_difference_fails():
    r = _cmp(_meta(weight=2.0), {})
    assert not r["pass"] and r["undeclared"] == {"cfg.align.weight": [0.5, 2.0]}


def test_undeclared_difference_beside_a_declared_one_fails():
    m = _meta(weight=2.0)
    m["cfg"]["k"] = 56
    r = _cmp(m, {"cfg.align.weight": 2.0})
    assert (
        not r["pass"] and list(r["undeclared"]) == ["cfg.k"] and list(r["declared"]) == ["cfg.align.weight"]
    )


def test_declared_difference_passes():
    r = _cmp(_meta(weight=2.0), {"cfg.align.weight": 2.0})
    assert r["pass"] and r["declared"] == {"cfg.align.weight": [0.5, 2.0]}


def test_declared_key_that_equals_the_reference_fails():
    r = _cmp(_meta(), {"cfg.align.weight": 2.0})
    assert not r["pass"] and "equals the reference" in r["declared_not_met"]["cfg.align.weight"]


def test_declared_key_with_another_value_fails():
    r = _cmp(_meta(weight=4.0), {"cfg.align.weight": 2.0})
    assert not r["pass"] and "expected 2.0" in r["declared_not_met"]["cfg.align.weight"]


def test_declared_key_in_neither_meta_fails():
    r = _cmp(_meta(), {"cfg.align.no_such_key": 1})
    assert not r["pass"] and "cfg.align.no_such_key" in r["declared_not_met"]


def test_train_data_and_resolved_blocks_are_compared():
    m = _meta()
    m["train_data"]["manifest_sha256"]["retained.csv"] = "ff"
    m["align"]["discard_channels"] = []
    r = _cmp(m, {})
    assert set(r["undeclared"]) == {"train_data.manifest_sha256.retained.csv", "resolved.discard_channels"}


def test_logging_only_keys_are_reported_not_failed():
    assert "ckpt_every" in C.AlignConfig.LOGGING_ONLY and "snapshot_every" in C.AlignConfig.LOGGING_ONLY
    r = _cmp(_meta(ckpt_every=1000, snapshot_every=2000), {})
    assert r["pass"] and set(r["logging_only"]) == {"cfg.align.ckpt_every", "cfg.align.snapshot_every"}


def test_key_added_by_newer_code_at_its_default_counts_as_equal():
    assert C.AlignConfig().stop_step == 0 and C.AlignConfig().sem_t_weights is None
    m = _meta(stop_step=0, sem_t_weights=None)
    m["cfg"]["max_images"] = None
    r = _cmp(m, {})
    assert r["pass"] and set(r["default"]) == {
        "cfg.align.stop_step",
        "cfg.align.sem_t_weights",
        "cfg.max_images",
    }


def test_key_added_by_newer_code_off_its_default_fails():
    r = _cmp(_meta(stop_step=5000), {})
    assert not r["pass"] and r["undeclared"] == {"cfg.align.stop_step": [C.ABSENT, 5000]}
    m = _meta()
    m["cfg"]["max_images"] = 8
    assert not _cmp(m, {})["pass"]
    m = _meta()
    m["train_data"]["new_key"] = None  # no default is known outside cfg: never waved through
    assert not _cmp(m, {})["pass"]


def test_meta_without_a_required_block_is_refused():
    m = _meta()
    del m["train_data"]
    with pytest.raises(ValueError, match="train_data"):
        C.view(m)


def test_check_and_cli_pass_then_fail_on_each_directory_mismatch(tmp_path, capsys):
    root = str(tmp_path / "models")
    _write(root, REF, _meta())
    run = "git10m_flux_aaaa+42+proj-1111"
    _write(root, run, _meta(weight=2.0))
    good = {run: {"role": "weight 2", "differs": {"cfg.align.weight": 2.0}}}
    out = str(tmp_path / "report.json")
    assert C.main(["--models-dir", root, "--expect", _expect(tmp_path, good), "--out", out]) == 0
    assert "GATE arm cfg PASS: 1/1" in capsys.readouterr().out
    assert json.load(open(out))["pass"] is True
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        C.main(["--models-dir", root, "--expect", _expect(tmp_path, good), "--out", out])

    # an undeclared difference
    assert C.main(["--models-dir", root, "--expect", _expect(tmp_path, {run: {"differs": {}}})]) == 1
    printed = capsys.readouterr().out
    assert "GATE arm cfg FAIL" in printed and "UNDECLARED" in printed and "do NOT compare" in printed

    # a run directory nobody declared
    _write(root, "git10m_flux_aaaa+42+proj-2222", _meta())
    res = C.check(root, json.load(open(_expect(tmp_path, good))))
    assert not res["pass"] and res["unlisted"] == ["git10m_flux_aaaa+42+proj-2222"]

    # a declared run that is not on disk
    both = copy.deepcopy(good) | {
        "git10m_flux_aaaa+42+proj-2222": {"differs": {}},
        "git10m_flux_aaaa+42+proj-3333": {"differs": {}},
    }
    res = C.check(root, json.load(open(_expect(tmp_path, both))))
    assert not res["pass"] and res["missing"] == ["git10m_flux_aaaa+42+proj-3333"] and not res["unlisted"]


def test_empty_expectation_cannot_pass(tmp_path):
    root = str(tmp_path / "models")
    _write(root, REF, _meta())
    assert not C.check(root, {"reference": REF, "runs": {}})["pass"]
    with pytest.raises(ValueError, match="must not be listed"):
        C.check(root, {"reference": REF, "runs": {REF: {"differs": {}}}})
    with pytest.raises(FileNotFoundError):
        C.check(root, {"reference": "absent", "runs": {}})


def test_shipped_expectation_file_is_well_formed():
    exp = json.load(open(os.path.join(HERE, "..", "experiments", "isaac", "git10m", "arm_cfg_expect.json")))
    assert exp["reference"] == "git10m_flux_ca8e6ce7+42+proj-ffcd7963" and exp["reference"] not in exp["runs"]
    for name, spec in exp["runs"].items():
        assert name.startswith("git10m_flux_") and spec["role"] and isinstance(spec["differs"], dict)
        seed = int(name.split("+")[1])
        assert spec["differs"].get("cfg.seed", 42) == seed, (
            name
        )  # the declared seed is the one in the run name
        assert all(k.split(".")[0] in ("cfg", "train_data", "resolved") for k in spec["differs"])
