"""Gates that live in job scripts as shell functions (review 2026-11-12, item 10).

Each function is taken verbatim from its script and fed, in bash, the outputs it must refuse
(CLAUDE.md rule 6: a guard that has never fired in a test is unverified).
"""

import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "experiments/isaac/geocore/smoke.sbatch")


def _function(script: str, name: str) -> str:
    src = open(script).read()
    m = re.search(rf"^{name}\(\) \{{.*?^\}}$", src, re.S | re.M)
    assert m, f"{script}: shell function {name} not found"
    return m.group(0)


def _accepts(tmp_path, summary: str) -> bool:
    log = tmp_path / "pytest.out"
    log.write_text(f"collected output\n...\n{summary}\n")
    fn = _function(SMOKE, "pytest_all_passed")
    r = subprocess.run(
        ["bash", "-c", f'{fn}\npytest_all_passed "$1"', "bash", str(log)], capture_output=True, text=True
    )
    assert r.stderr == "", r.stderr
    return r.returncode == 0


@pytest.mark.parametrize(
    "summary",
    [
        "3 passed in 41.20s",
        "1 passed, 2 warnings in 3.10s",
        "\x1b[32m\x1b[1m12 passed\x1b[0m\x1b[32m in 61.00s (0:01:01)\x1b[0m",
        "============================== 3 passed in 41.20s ==============================",
    ],
)
def test_smoke_pytest_gate_accepts(tmp_path, summary):
    assert _accepts(tmp_path, summary)


@pytest.mark.parametrize(
    "summary",
    [
        "3 skipped in 0.40s",  # no CUDA: pytest exits 0, nothing ran
        "2 passed, 1 skipped in 20.00s",
        "1 failed, 2 passed in 20.00s",
        "2 passed, 1 error in 20.00s",
        "3 deselected in 0.10s",
        "2 passed, 1 deselected in 9.00s",
        "2 passed, 1 xfailed in 9.00s",  # an expected failure is a gate that did not hold
        "2 passed, 1 xpassed in 9.00s",
        "2 passed, 1 rerun in 9.00s",
        "no tests ran in 0.02s",
        "0 passed in 0.02s",
        "",  # pytest died before a summary
        "Killed",
    ],
)
def test_smoke_pytest_gate_refuses(tmp_path, summary):
    assert not _accepts(tmp_path, summary)


def test_smoke_gate_one_uses_the_function_and_no_tail_pipe():
    body = [ln for ln in open(SMOKE).read().splitlines() if not ln.lstrip().startswith("#")]
    run = next(ln for ln in body if "tests/test_geocore_null_text.py" in ln)
    assert "| tail" not in run and "-rs" in run and "|| g1_rc=$?" in run
    assert any("! pytest_all_passed" in ln for ln in body)
