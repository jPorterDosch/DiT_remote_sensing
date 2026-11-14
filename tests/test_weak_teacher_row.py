"""Weak-teacher control (review 2026-11-12, item 6): the weak arm is scored against the teacher it
trained with (DINOv3 ViT-L/16 web), and a CPU extraction of that teacher is float32 and says so.

CPU only, no weights, no dataset: the contract of eval/extract_dino.py's --device / --dtype, the
phase-11 spec, and the spec builder inside experiments/isaac/git10m/arm_full.sh (run as the job
runs it, fed the mismatch it exists to catch).
"""

import json
import os
import re
import subprocess
import sys

import pytest
import torch

from eval import extract_dino as E
from eval import gate_marker as GM
from eval import matrix as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "eval/matrix_specs/phase11_weak_teacher.json")
ARM_FULL = os.path.join(ROOT, "experiments/isaac/git10m/arm_full.sh")
ROW = "dinov3-vitl16-web"
PRESET = "dinov3_vitl16_web"


# ------------------------------------------------------------------ extractor: device / dtype
def test_cpu_extraction_must_be_float32():
    with pytest.raises(SystemExit, match="needs --dtype float32"):
        E.resolve_dtype(PRESET, "cpu", "preset")
    assert E.resolve_dtype(PRESET, "cpu", "float32") is torch.float32
    assert E.resolve_dtype(PRESET, "cuda", "preset") is torch.bfloat16
    assert E.resolve_dtype(PRESET, "cuda", "float32") is torch.float32
    assert (
        E.resolve_dtype("dinov2_vitl14", "cpu", "preset") is torch.float32
    )  # a float32 preset needs no flag


def test_bad_device_dtype_pair_fails_before_any_io(tmp_path):
    """The dataset name does not exist: reaching the dataset lookup would raise another message."""
    common = ["--preset", PRESET, "--dataset", "no_such_dataset", "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit, match="needs --dtype float32"):
        E.main([*common, "--device", "cpu"])
    with pytest.raises(SystemExit, match="--tokens runs at the preset dtype on cuda only"):
        E.main([*common, "--tokens", "--dtype", "float32"])
    with pytest.raises(SystemExit, match="neither a CV_IDENTITY nor an OFFICIAL dataset"):
        E.main([*common, "--device", "cpu", "--dtype", "float32"])  # the good pair gets past the check
    assert not os.listdir(tmp_path)


def test_load_model_dtype_override_leaves_presets_alone(monkeypatch):
    before = {k: dict(v) for k, v in E.PRESETS.items()}
    monkeypatch.setattr(torch.hub, "load", lambda repo, entry, **kw: torch.nn.Linear(2, 2))
    model, spec = E.load_model("dinov2_vitl14", None, "cpu", torch.float64)
    assert spec["dtype"] is torch.float64 and model.weight.dtype is torch.float64
    model, spec = E.load_model("dinov2_vitl14", None, "cpu")
    assert spec["dtype"] is torch.float32 and model.weight.dtype is torch.float32
    assert E.PRESETS == before


# ------------------------------------------------------------------ the teacher's matrix row
def test_phase11_spec_is_the_vitl_teacher_on_the_phase5_grid():
    spec = M.load_spec(SPEC)
    ref = M.load_spec(os.path.join(ROOT, "eval/matrix_specs/phase5_web_sweep.json"))
    for k in ("fractions", "seeds", "c_grid", "datasets"):
        assert spec[k] == ref[k], k
    web, vitl = spec["models"]
    # identical dict = identical cell identity (eval.matrix.cell_config): the 7B cells are reused
    assert web == next(m for m in ref["models"] if m["name"] == "dinov3-web")
    assert vitl["name"] == ROW and vitl["kind"] == "dino" and vitl["teacher"] is None and vitl["view"] is None
    assert vitl["features"] == f"models/dino_feats/{PRESET}_{{dataset}}_{{split}}.npz"
    assert PRESET in E.PRESETS and E.PRESETS[PRESET]["entry"] == "dinov3_vitl16"


def _spec_builder() -> str:
    """The python heredoc of arm_full.sh that writes the per-arm spec, verbatim."""
    src = open(ARM_FULL).read()
    m = re.search(
        r'^PYTHONPATH="src:\." python3 - "\$SPEC" .*<<\'PY\' \|\| exit 1\n(.*?)^PY$', src, re.S | re.M
    )
    assert m, "arm_full.sh: spec-builder heredoc not found"
    return m.group(1)


def _build(tmp_path, tag, teacher, target):
    out = tmp_path / f"phase10_{tag}.json"
    run = "git10m_flux_ca8e6ce7+42+proj-00000000"
    r = subprocess.run(
        [sys.executable, "-c", _spec_builder(), str(out), tag, "42", run, teacher, target, "", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return r, out


def test_arm_full_scores_weak_against_its_own_teacher(tmp_path):
    src = open(ARM_FULL).read()
    assert f"weak) TEACHER_ROW={ROW} ;;" in src
    assert "web|weak) TEACHER_ROW=dinov3-web" not in src  # the old line: 7B web as the weak arm's teacher
    r, out = _build(tmp_path, "weak", ROW, PRESET)
    assert r.returncode == 0, r.stderr
    spec = M.load_spec(str(out))
    teacher, arm = spec["models"]
    assert teacher == M.load_spec(SPEC)["models"][1]
    assert arm["name"] == "flux-proj-weak-s42" and arm["teacher"] == ROW
    # the cell array of arm_full.sh is 6-11 = model index 1
    assert [m["name"] for m in spec["models"]].index(arm["name"]) == 1 and len(spec["datasets"]) == 6


def test_arm_full_spec_builder_refuses_an_unknown_teacher(tmp_path):
    r, out = _build(tmp_path, "weak", "dinov3-vitl14-web", PRESET)
    assert r.returncode != 0 and "is in none of the teacher specs" in r.stderr
    assert not out.exists()


def test_arm_full_spec_builder_refuses_to_change_an_existing_spec(tmp_path):
    r, out = _build(tmp_path, "weak", ROW, PRESET)
    assert r.returncode == 0, r.stderr
    spec = json.load(open(out))
    spec["models"][0] = M.load_spec(SPEC)["models"][0]  # as written while weak was scored against 7B web
    json.dump(spec, open(out, "w"))
    r, _ = _build(tmp_path, "weak", ROW, PRESET)
    assert r.returncode != 0 and "exists with different content" in r.stderr


# ------------------------------------------------------------------ the row is gated (staged-delta finding 3)
MATRIX_SBATCH = os.path.join(ROOT, "experiments/isaac/eval/matrix.sbatch")
DINO_PROBE = os.path.join(ROOT, "experiments/isaac/geobench/dino_probe.sbatch")
DINO_SBATCH = os.path.join(ROOT, "experiments/isaac/geobench/dino.sbatch")


def _features(root, preset=PRESET, dataset="m_eurosat", fill=b"x" * 64):
    d = root / "models" / "dino_feats"
    d.mkdir(parents=True, exist_ok=True)
    for split in GM.SPLITS:
        (d / f"{preset}_{dataset}_{split}.npz").write_bytes(fill + split.encode())
    return f"models/dino_feats/{preset}_{dataset}_{{split}}.npz"


def test_phase11_spec_requires_the_dino_gate_for_the_vitl_row():
    spec = json.load(open(SPEC))
    assert spec["requires_gate"] == {ROW: {"gate": "dino_control", "preset": PRESET}}
    M.load_spec(SPEC)  # the top-level key does not upset the matrix's own spec check


def test_matrix_task_is_refused_until_the_gate_passed_on_these_features(tmp_path, monkeypatch, capsys):
    """Fed the mismatches (CLAUDE.md rule 6): no marker; a marker for another dataset / preset; a
    marker whose feature files were replaced afterwards."""
    monkeypatch.chdir(tmp_path)
    pattern = _features(tmp_path)
    req = ["require", "--spec", SPEC, "--model", ROW, "--dataset", "m_eurosat"]
    assert GM.main(req) == 1
    assert "GATE required-marker FAIL" in capsys.readouterr().out
    assert (
        GM.main(["require", "--spec", SPEC, "--model", "dinov3-web", "--dataset", "m_eurosat"]) == 0
    )  # not listed
    # a PASS for another dataset, and one for another preset, do not unlock this task
    GM.write(
        GM.DEFAULT_DIR, "dino_control", PRESET, "m_pv4ger", _features(tmp_path, dataset="m_pv4ger"), "11"
    )
    other = _features(tmp_path, preset="dinov3_vit7b16_web")
    GM.write(GM.DEFAULT_DIR, "dino_control", "dinov3_vit7b16_web", "m_eurosat", other, "12")
    GM.write(GM.DEFAULT_DIR, "other_gate", PRESET, "m_eurosat", pattern, "13")
    assert GM.main(req) == 1
    # the gate passes on these files: unlocked
    w = [
        "write",
        "--gate",
        "dino_control",
        "--preset",
        PRESET,
        "--dataset",
        "m_eurosat",
        "--features",
        pattern,
    ]
    assert GM.main([*w, "--job", "14"]) == 0
    assert GM.main(req) == 0
    assert "GATE required-marker PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="never overwritten"):
        GM.main([*w, "--job", "14"])
    # the features are extracted again (same size, later mtime): the old PASS no longer counts
    f = tmp_path / pattern.format(split="val")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    ok, msg = GM.require(SPEC, ROW, "m_eurosat", GM.DEFAULT_DIR)
    assert not ok and "val features changed since the gate ran" in msg
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert GM.require(SPEC, ROW, "m_eurosat", GM.DEFAULT_DIR)[0]
    f.write_bytes(b"y" * 10)  # another size
    assert not GM.require(SPEC, ROW, "m_eurosat", GM.DEFAULT_DIR)[0]
    assert GM.main([*w, "--job", "15"]) == 0  # gated again on the new files
    assert GM.main(req) == 0
    # a marker edited to FAIL, or a non-marker json, never unlocks
    for name in os.listdir(GM.DEFAULT_DIR):
        if name.endswith(("_14.json", "_15.json")):
            os.remove(os.path.join(GM.DEFAULT_DIR, name))
    m = GM.file_stamps(pattern)
    bad = {"gate": "dino_control", "verdict": "FAIL", "preset": PRESET, "dataset": "m_eurosat", "features": m}
    (tmp_path / GM.DEFAULT_DIR / f"dino_control_{PRESET}_m_eurosat_16.json").write_text(json.dumps(bad))
    (tmp_path / GM.DEFAULT_DIR / f"dino_control_{PRESET}_m_eurosat_17.json").write_text("[1]")
    (tmp_path / GM.DEFAULT_DIR / f"dino_control_{PRESET}_m_eurosat_18.json").write_text("{not json")
    ok, msg = GM.require(SPEC, ROW, "m_eurosat", GM.DEFAULT_DIR)
    assert not ok and "verdict='FAIL'" in msg and "not a json object" in msg and "unreadable" in msg


def test_gate_marker_refuses_missing_features_and_bad_specs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pattern = _features(tmp_path)
    os.remove(tmp_path / pattern.format(split="test"))
    with pytest.raises(SystemExit, match="feature file .*_test.npz"):
        GM.write(GM.DEFAULT_DIR, "dino_control", PRESET, "m_eurosat", pattern, "1")
    with pytest.raises(SystemExit, match="feature file .*_test.npz"):
        GM.require(SPEC, ROW, "m_eurosat", GM.DEFAULT_DIR)  # nothing to score, nothing a gate judged
    assert not os.path.exists(GM.DEFAULT_DIR)
    (tmp_path / pattern.format(split="test")).write_bytes(b"")
    with pytest.raises(SystemExit, match="is empty"):
        GM.write(GM.DEFAULT_DIR, "dino_control", PRESET, "m_eurosat", pattern, "1")
    with pytest.raises(SystemExit, match="no {split}"):
        GM.file_stamps("models/dino_feats/a.npz")
    with pytest.raises(SystemExit, match="not a plain name"):
        GM.write(GM.DEFAULT_DIR, "dino_control", PRESET, "../m_eurosat", pattern, "1")
    spec = json.load(open(SPEC))
    for edit, why in [
        (
            {"dinov3-vitl16": spec["requires_gate"][ROW]},
            "not among the spec's models",
        ),  # misspelt: gates nothing
        ({ROW: {"gate": "dino_control"}}, "exactly"),
        ([ROW], "must be a map"),
    ]:
        p = tmp_path / "spec.json"
        p.write_text(json.dumps({**spec, "requires_gate": edit}))
        with pytest.raises(SystemExit, match=why):
            GM.require(str(p), ROW, "m_eurosat", GM.DEFAULT_DIR)
    with pytest.raises(SystemExit, match="no model"):
        GM.require(SPEC, "dinov3-sat", "m_eurosat", GM.DEFAULT_DIR)


def test_job_scripts_write_and_require_the_marker():
    body = [ln for ln in open(MATRIX_SBATCH).read().splitlines() if not ln.lstrip().startswith("#")]
    i_req = next(i for i, ln in enumerate(body) if "eval.gate_marker require" in ln)
    i_run = next(i for i, ln in enumerate(body) if "eval.matrix run" in ln)
    assert i_req < i_run and '--spec "$SPEC" --model "$MODEL" --dataset "$DATASET"' in body[i_req]
    assert "||" not in body[i_req]  # under set -e: a refusal ends the task
    assert any(ln.strip() == "set -euo pipefail" for ln in body[:i_req])
    body = [ln for ln in open(DINO_PROBE).read().splitlines() if not ln.lstrip().startswith("#")]
    i_gate = next(i for i, ln in enumerate(body) if '! gate_passed "$GATE_LOG"' in ln)
    i_write = next(i for i, ln in enumerate(body) if "eval.gate_marker write" in ln)
    i_only = next(i for i, ln in enumerate(body) if '"${GATE_ONLY:-0}" = 1' in ln)
    assert i_gate < i_write < i_only  # written only after the PASS, and also on a GATE_ONLY run
    assert '[ "$SMOKE" != 1 ]' in body[i_write - 1] and "|| return 1" in body[i_write + 1]


# ------------------------------------------------------------------ dino.sbatch cache reuse (finding 6)
def _reuse_check(out_dir, want):
    src = open(DINO_SBATCH).read()
    start = src.index("<<'PY' || rc=$?\n") + len("<<'PY' || rc=$?\n")
    block = src[start : src.index("\nPY\n", start)]
    r = subprocess.run(
        [sys.executable, "-", str(out_dir), PRESET, "m_eurosat", want],
        input=block, capture_output=True, text=True, cwd=ROOT,
    )  # fmt: skip
    return r.returncode, r.stderr


def _cache(out_dir, split, n, dtype):
    import numpy as np

    kw = {"preset": np.array(PRESET), "labels": np.zeros(n, np.int64), "paths": np.zeros(n, "U1")}
    if dtype is not None:
        kw["dtype"] = np.array(dtype)
    np.savez(os.path.join(out_dir, f"{PRESET}_m_eurosat_{split}.npz"), **kw)


def test_dino_reuse_check_reports_a_dtype_conflict_behind_a_missing_split(tmp_path):
    """Exit 4 (another dtype's cache: fatal) must win over exit 1 (extract) whichever split comes
    first: the old loop stopped at a missing train split and never looked at val / test."""
    from eval.features import official_spec

    sizes = official_spec("m_eurosat")["sizes"]
    first, *rest = list(sizes)
    for split in rest:
        _cache(tmp_path, split, sizes[split], "torch.bfloat16")
    rc, err = _reuse_check(tmp_path, "torch.float32")  # first split missing, later ones bf16
    assert rc == 4 and "holds dtype torch.bfloat16, this run extracts torch.float32" in err, (rc, err)
    for split in rest:
        _cache(tmp_path, split, sizes[split], "torch.float32")
    rc, err = _reuse_check(tmp_path, "torch.float32")  # first split missing, nothing conflicting: extract
    assert rc == 1 and f"_{first}.npz" in err, (rc, err)
    _cache(tmp_path, first, sizes[first], "torch.float32")
    assert _reuse_check(tmp_path, "torch.float32")[0] == 0  # complete and of this dtype: reuse
    assert _reuse_check(tmp_path, "preset")[0] == 4  # a cuda (bf16) run must not reuse the float32 caches
    _cache(tmp_path, first, sizes[first] - 1, "torch.float32")
    assert _reuse_check(tmp_path, "torch.float32")[0] == 1  # short split: extract again
