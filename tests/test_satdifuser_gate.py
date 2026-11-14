"""SatDiFuser reproduction gate (experiments/isaac/satdifuser/gate_check.py; review 2026-11-12, item 10).

Stdlib logic, fed the mismatches it exists to catch (CLAUDE.md rule 6), plus the two job scripts'
use of it.
"""

import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF = os.path.join(ROOT, "experiments/isaac/satdifuser")
_spec = importlib.util.spec_from_file_location("sdf_gate_check", os.path.join(SDF, "gate_check.py"))
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)

LOG = "INFO Epoch 40 completed in 812.30 seconds\nINFO Loading best checkpoint from step 19800 for final test evaluation\n"


def _log(metric):
    return LOG + f"2026-11-12 10:00:00 INFO \x1b[32mFinal test metric: {metric}\x1b[0m\n"


def test_pass_within_half_a_point_either_side():
    for m in ("0.9770", "0.9720", "0.9820", "0.9745"):
        v = GC.verdict(_log(m), "gw")
        assert v["gate"] == "PASS" and v["metric"] == float(m), m
    assert GC.verdict(_log("0.9680"), "lw")["gate"] == "PASS"  # each fuser against its own published number
    assert GC.verdict(_log("0.9680"), "gw")["gate"] == "FAIL"


@pytest.mark.parametrize(
    "text, why",
    [
        (_log("0.9719"), "gap"),  # just below
        (_log("0.9821"), "gap"),  # just above: too good is also a discrepancy
        (_log("0.1000"), "gap"),  # chance
        (_log("nan"), "not a fraction"),
        (_log("97.70"), "not a fraction"),  # a percentage where a fraction is expected
        (_log("abc"), "unparseable"),
        (LOG, "found 0"),  # the run died before the test evaluation
        (_log("0.9770") + _log("0.9770"), "found 2"),  # two runs in one log
        (
            _log("0.9770") + "WARNING No best checkpoint found; skipping final test evaluation\n",
            "No best checkpoint",
        ),
        ("", "found 0"),
    ],
)
def test_fail_cases(text, why):
    v = GC.verdict(text, "gw")
    assert v["gate"] == "FAIL" and why in v["reason"], v
    assert "do NOT dump" in GC.line(v)


def test_bad_tolerance_and_unknown_fuser_never_pass():
    assert GC.verdict(_log("0.9770"), "gw", tolerance=0.0)["gate"] == "FAIL"
    assert GC.verdict(_log("0.9770"), "gw", tolerance=float("nan"))["gate"] == "FAIL"
    with pytest.raises(SystemExit, match="no published"):
        GC.verdict(_log("0.9770"), "mean")


def test_verdict_cli_writes_once_and_exit_code_follows(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text(_log("0.9770"))
    out = tmp_path / "gate" / "eurosat_gw_1.json"
    assert GC.main(["verdict", "--log", str(log), "--fuser", "gw", "--out", str(out), "--job", "1"]) == 0
    assert "GATE satdifuser-eurosat PASS [gw]" in capsys.readouterr().out
    assert json.loads(out.read_text())["gate"] == "PASS"
    with pytest.raises(SystemExit, match="never overwritten"):
        GC.main(["verdict", "--log", str(log), "--fuser", "gw", "--out", str(out)])
    log.write_text(_log("0.9000"))
    bad = tmp_path / "gate" / "eurosat_gw_2.json"
    assert GC.main(["verdict", "--log", str(log), "--fuser", "gw", "--out", str(bad), "--job", "2"]) == 1
    assert "GATE satdifuser-eurosat FAIL" in capsys.readouterr().out
    assert json.loads(bad.read_text())["gate"] == "FAIL"  # the FAIL is on disk too


def _stored(fuser="gw", metric=0.977, **kw):
    """A verdict file as `verdict` writes it."""
    return json.dumps({**GC.verdict(_log(f"{metric:.4f}"), fuser), **kw})


def test_require_needs_a_pass_and_no_fail(tmp_path):
    d = tmp_path / "gate"
    assert GC.main(["require", "--dir", str(d), "--fuser", "gw"]) == 1  # no directory, no verdict
    d.mkdir()
    (d / "eurosat_lw_1.json").write_text(_stored("lw", 0.968))
    assert not GC.require(str(d), "gw")[0]  # another fuser's PASS does not count
    (d / "eurosat_gw_2.json").write_text(_stored())
    assert GC.require(str(d), "gw")[0]
    assert GC.main(["require", "--dir", str(d), "--fuser", "gw"]) == 0
    (d / "eurosat_gw_3.json").write_text(_stored(metric=0.90))
    assert json.loads((d / "eurosat_gw_3.json").read_text())["gate"] == "FAIL"
    assert not GC.require(str(d), "gw")[0]  # a later FAIL blocks, an earlier PASS does not excuse it
    os.remove(d / "eurosat_gw_3.json")
    (d / "eurosat_gw_4.json").write_text("{not json")
    assert not GC.require(str(d), "gw")[0]
    (d / "eurosat_gw_4.json").write_text(_stored("lw", 0.968))  # mislabelled file
    assert not GC.require(str(d), "gw")[0]


@pytest.mark.parametrize(
    "stored, why",
    [
        ({"gate": "PASS", "fuser": "gw"}, "published=None"),  # the bare stamp the old `require` accepted
        ({"tolerance_pp": 5.0, "metric": 0.93}, "tolerance_pp=5.0"),  # written with a loose --tolerance
        ({"tolerance_pp": 0.0}, "tolerance_pp=0.0"),
        ({"tolerance_pp": None}, "tolerance_pp=None"),
        ({"tolerance_pp": True}, "tolerance_pp=True"),
        ({"metric": 0.93}, "exceeds its tolerance"),  # PASS stamped on a metric 4.7 points off
        ({"metric": None}, "metric=None"),
        ({"metric": float("nan")}, "metric=nan"),
        ({"metric": 97.7}, "metric=97.7"),
        ({"published": 96.8}, "published=96.8"),  # judged against another fuser's number
        ({"tolerance_pp": 0.2, "metric": 0.9745}, "exceeds its tolerance"),  # outside its own tighter bound
    ],
)
def test_require_rejudges_the_stored_verdict(tmp_path, stored, why):
    """Fed the mismatch (CLAUDE.md rule 6): each file says gate=PASS for gw and must not unlock the dump."""
    good = json.loads(_stored())
    assert GC.stored_ok(good, "gw") == (True, "")
    v = stored if "published=None" in why else {**good, **stored}
    assert v["gate"] == "PASS" and v["fuser"] == "gw"
    ok, reason = GC.stored_ok(v, "gw")
    assert not ok and why in reason, reason
    (tmp_path / "eurosat_gw_9.json").write_text(json.dumps(v))
    ok, msg = GC.require(str(tmp_path), "gw")
    assert not ok and why in msg
    assert GC.stored_ok([v], "gw") == (False, "not a json object")
    assert GC.stored_ok({**good, "tolerance_pp": 0.2}, "gw")[0]  # a stricter tolerance that holds is fine


@pytest.mark.parametrize("tol", ["5", "0.51", "0", "-0.5", "nan", "inf"])
def test_cli_tolerance_can_only_tighten(tmp_path, tol):
    log = tmp_path / "run.log"
    log.write_text(_log("0.9770"))
    out = tmp_path / "eurosat_gw_1.json"
    with pytest.raises(SystemExit) as e:
        GC.main(["verdict", "--log", str(log), "--fuser", "gw", "--out", str(out), "--tolerance", tol])
    assert e.value.code == 2 and not out.exists()  # argparse refuses before anything is written


def test_cli_tolerance_tighter_is_accepted(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(_log("0.9745"))  # 0.25 below: inside 0.5, outside 0.2
    args = ["verdict", "--log", str(log), "--fuser", "gw"]
    assert GC.main([*args, "--out", str(tmp_path / "eurosat_gw_1.json"), "--tolerance", "0.2"]) == 1
    assert GC.main([*args, "--out", str(tmp_path / "eurosat_gw_2.json"), "--tolerance", "0.5"]) == 0


def test_job_scripts_call_the_gate():
    gate = open(os.path.join(SDF, "gate_eurosat.sbatch")).read()
    body = [ln for ln in gate.splitlines() if not ln.lstrip().startswith("#")]
    i_run = next(i for i, ln in enumerate(body) if ln.startswith("python3 run.py"))
    i_check = next(i for i, ln in enumerate(body) if 'gate_check.py" verdict' in ln)
    i_ok = next(i for i, ln in enumerate(body) if ln.strip() == 'echo "SDF_GATE_OK"')
    assert i_run < i_check < i_ok and sum("SDF_GATE_OK" in ln for ln in body) == 1
    assert "set -euo pipefail" in gate and "| tee" in body[i_run]
    dump = open(os.path.join(SDF, "dump.sbatch")).read()
    body = [ln for ln in dump.splitlines() if not ln.lstrip().startswith("#")]
    i_req = next(i for i, ln in enumerate(body) if 'gate_check.py" require' in ln)
    i_first = next(i for i, ln in enumerate(body) if "eval.satdifuser dump" in ln or 'rm -rf "$OUT"' in ln)
    assert i_req < i_first  # the gate is checked before anything is removed or dumped
