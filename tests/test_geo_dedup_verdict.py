"""The brute-force verification of the dedup manifests must be able to FAIL (review 2026-11-12,
item 7). verify.py used to write results.json and print DONE whatever the numbers said.

CPU only, stdlib: experiments/isaac/geo_dedup/verify_bruteforce/verdict.py is fed a passing
result set, then each mismatch it exists to catch.
"""

import copy
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "experiments", "isaac", "geo_dedup", "verify_bruteforce")


def _load():
    spec = importlib.util.spec_from_file_location("verify_verdict", os.path.join(DIR, "verdict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load()


def _good() -> dict:
    """Shape and magnitudes of the 2026-10-13 run (README.md table)."""
    return {
        "city_boxes_max_abs_deg_diff": 1.4e-14,
        "c2_excluded": {
            "n": 2000,
            "pass": 2000,
            "fail": 0,
            "max_abs_diff_vs_artifact_min_distance_m": 0.199,
            "disagree": [],
        },
        "c3_retained": {"n": 20000, "pass": 20000, "fail": 0, "disagree": []},
        "c4_v2excluded_subset_of_v3": {"v2_excluded": 418713, "v2_not_in_v3": 0},
        "c4_v3_new_excluded": {
            "n": 2000,
            "pass": 2000,
            "fail": 0,
            "all_datasets_contain_m_so2sat": True,
            "max_abs_diff_vs_artifact_min_distance_m": 0.206,
            "disagree": [],
        },
        "c4_v3_retained": {"n": 2000, "pass": 2000, "fail": 0, "disagree": []},
        "c5_matches": {
            "n": 500,
            "max_abs_diff_m": 0.104,
            "n_diff_gt_1m": 0,
            "n_zero_artifact": 91,
            "n_zero_mine": 91,
            "all_le_buffer_mine": True,
        },
        "c6_tiles": {
            "n": 20,
            "max_abs_deg_diff_vs_code": 0.0,
            "max_abs_deg_diff_vs_meta_parquet": 0.0,
            "name_matches_location": 20,
        },
        "c6_ssl4eo": {
            "n": 20,
            "source_id_matches_centre_id": 20,
            "max_centre_offset_m": 0.0,
            "max_abs_half_extent_err_m_vs_1320": 5.4e-10,
        },
    }


def _set(d, path, value):
    *head, last = path
    for k in head:
        d = d[k]
    d[last] = value


def test_good_results_pass(capsys):
    assert V.report(_good()) is True
    out = capsys.readouterr().out
    assert out.count("PASS ") == len(V.SECTIONS) and "FAIL" not in out
    assert f"GATE brute-force verify PASS: {len(V.SECTIONS)}/{len(V.SECTIONS)} checks" in out


MISMATCHES = [
    (("city_boxes_max_abs_deg_diff",), 1e-6, "city boxes"),
    (("c2_excluded", "fail"), 1, "v2 excluded"),
    (("c2_excluded", "max_abs_diff_vs_artifact_min_distance_m"), 1.5, "v2 excluded"),
    (("c2_excluded", "max_abs_diff_vs_artifact_min_distance_m"), float("nan"), "v2 excluded"),
    (("c3_retained", "disagree"), [{"shard": 3, "row": 7}], "v2 retained"),
    (("c3_retained", "pass"), 19999, "v2 retained"),
    (("c3_retained", "n"), 0, "v2 retained"),
    (("c4_v2excluded_subset_of_v3", "v2_not_in_v3"), 2, "subset of v3"),
    (("c4_v3_new_excluded", "all_datasets_contain_m_so2sat"), False, "city box"),
    (("c4_v3_retained", "fail"), 3, "outside every city buffer"),
    (("c5_matches", "n_zero_mine"), 90, "matches.csv"),
    (("c5_matches", "all_le_buffer_mine"), False, "matches.csv"),
    (("c6_tiles", "name_matches_location"), 19, "tile bounds"),
    (("c6_tiles", "max_abs_deg_diff_vs_meta_parquet"), 1e-3, "tile bounds"),
    (("c6_ssl4eo", "source_id_matches_centre_id"), 19, "SSL4EO"),
    (("c6_ssl4eo", "max_centre_offset_m"), 2640.0, "SSL4EO"),
]


@pytest.mark.parametrize("path,value,name", MISMATCHES)
def test_each_mismatch_fails_its_own_check(path, value, name, capsys):
    d = _good()
    _set(d, path, value)
    assert V.report(d) is False
    failed = [n for n, ok, _ in V.checks(d) if not ok]
    assert len(failed) == 1 and name in failed[0], failed
    out = capsys.readouterr().out
    assert "GATE brute-force verify FAIL" in out and "do NOT quote" in out


@pytest.mark.parametrize("key", [k for k in _good()])
def test_a_missing_section_fails_instead_of_being_skipped(key):
    d = copy.deepcopy(_good())
    del d[key]
    res = V.checks(d)
    assert len(res) == len(V.SECTIONS)  # the check still runs, and fails
    assert [ok for _, ok, _ in res].count(False) == 1
    assert V.report(d) is False


def test_sections_are_exactly_what_verify_py_fills():
    src = open(os.path.join(DIR, "verify.py")).read()
    for key in V.SECTIONS:
        assert f'OUT["{key}"]' in src, key
    assert "if not verdict.report(OUT):\n    sys.exit(1)" in src
    assert 'open("results.json", "x")' in src  # never overwrites a recorded result
