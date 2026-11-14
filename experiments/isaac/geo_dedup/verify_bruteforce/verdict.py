"""PASS / FAIL verdict over the numbers verify.py collects (review 2026-11-12, item 7).

verify.py used to write results.json and end with DONE whatever the numbers said: a sampled row
that disagreed with the artifact was a line in a json file. This module turns each number into a
check that prints PASS or FAIL; verify.py exits 1 when any check fails.

Stdlib only, no side effects on import: tests/test_geo_dedup_verdict.py feeds it the mismatches it
exists to catch.

Tolerances (fixed 2026-11-12, after the 2026-10-13 run; recorded so the choice is auditable):
  DIST_TOL_M 1.0   independent gap vs the artifact's distance. The artifacts predate the poleward
                   shared-latitude change of box_gap_m and read up to 0.2 m high (README.md); the
                   2026-10-13 run measured 0.104-0.206 m. 1 m is 0.1 % of the 1,000 m buffer.
  DEG_TOL 1e-9     box corners rebuilt from the same formula (about 0.1 mm).
  SSL4EO_TOL_M 0.01  box centre vs the authors' centre, and half extent vs 1,320 m.
"""

from __future__ import annotations

DIST_TOL_M = 1.0
DEG_TOL = 1e-9
SSL4EO_TOL_M = 0.01
# The keys verify.py must have filled. A missing key is a FAIL, never a skipped check.
SECTIONS = (
    "city_boxes_max_abs_deg_diff",
    "c2_excluded",
    "c3_retained",
    "c4_v2excluded_subset_of_v3",
    "c4_v3_new_excluded",
    "c4_v3_retained",
    "c5_matches",
    "c6_tiles",
    "c6_ssl4eo",
)


def checks(out: dict) -> list[tuple[str, bool, str]]:
    """-> [(name, passed, detail)]. Raises nothing: a missing or non-numeric value fails its check."""
    res: list[tuple[str, bool, str]] = []

    def add(name, fn):
        try:
            ok, detail = fn()
        except (KeyError, TypeError, ValueError) as e:
            ok, detail = False, f"result missing or malformed: {type(e).__name__}: {e}"
        res.append((name, bool(ok), detail))

    def le(value, tol):
        # NaN compares False: a NaN discrepancy fails
        return float(value) <= tol

    def sampled(key, extra=None):
        def fn():
            c = out[key]
            ok = c["n"] > 0 and c["fail"] == 0 and c["pass"] == c["n"] and not c["disagree"]
            detail = f"n={c['n']} pass={c['pass']} fail={c['fail']}"
            if extra:
                e_ok, e_detail = extra(c)
                ok, detail = ok and e_ok, f"{detail} {e_detail}"
            return ok, detail

        return fn

    def dist(field):
        return lambda c: (le(c[field], DIST_TOL_M), f"{field}={float(c[field]):.6f} (tol {DIST_TOL_M})")

    add(
        "city boxes rebuilt from so2sat_cities.csv",
        lambda: (
            le(out["city_boxes_max_abs_deg_diff"], DEG_TOL),
            f"max abs diff {float(out['city_boxes_max_abs_deg_diff']):.3e} deg (tol {DEG_TOL})",
        ),
    )
    add(
        "v2 excluded rows have gap <= buffer",
        sampled("c2_excluded", dist("max_abs_diff_vs_artifact_min_distance_m")),
    )
    add("v2 retained rows have gap > buffer", sampled("c3_retained"))
    add(
        "v2 excluded is a subset of v3 excluded",
        lambda: (
            out["c4_v2excluded_subset_of_v3"]["v2_excluded"] > 0
            and out["c4_v2excluded_subset_of_v3"]["v2_not_in_v3"] == 0,
            f"{out['c4_v2excluded_subset_of_v3']['v2_not_in_v3']} of "
            f"{out['c4_v2excluded_subset_of_v3']['v2_excluded']} v2-excluded rows are not excluded in v3",
        ),
    )

    def v3_new(c):
        d_ok, d_detail = dist("max_abs_diff_vs_artifact_min_distance_m")(c)
        return d_ok and c["all_datasets_contain_m_so2sat"] is True, (
            f"{d_detail} all_datasets_contain_m_so2sat={c['all_datasets_contain_m_so2sat']}"
        )

    add("v3-only excluded rows lie within buffer of a city box", sampled("c4_v3_new_excluded", v3_new))
    add("v3 retained rows lie outside every city buffer", sampled("c4_v3_retained"))

    def matches():
        c = out["c5_matches"]
        ok = (
            c["n"] > 0
            and le(c["max_abs_diff_m"], DIST_TOL_M)
            and c["n_diff_gt_1m"] == 0
            and c["n_zero_artifact"] == c["n_zero_mine"]
            and c["all_le_buffer_mine"] is True
        )
        return ok, (
            f"n={c['n']} max_abs_diff_m={float(c['max_abs_diff_m']):.6f} zero gaps artifact/mine "
            f"{c['n_zero_artifact']}/{c['n_zero_mine']} all_le_buffer={c['all_le_buffer_mine']}"
        )

    add("matches.csv distance_m equals the independent gap", matches)

    def tiles():
        c = out["c6_tiles"]
        ok = (
            c["n"] > 0
            and le(c["max_abs_deg_diff_vs_code"], DEG_TOL)
            and le(c["max_abs_deg_diff_vs_meta_parquet"], DEG_TOL)
            and c["name_matches_location"] == c["n"]
        )
        return ok, (
            f"n={c['n']} vs code {float(c['max_abs_deg_diff_vs_code']):.3e} deg, vs meta "
            f"{float(c['max_abs_deg_diff_vs_meta_parquet']):.3e} deg, names {c['name_matches_location']}/{c['n']}"
        )

    add("tile bounds equal the formula and the meta parquet", tiles)

    def ssl4eo():
        c = out["c6_ssl4eo"]
        ok = (
            c["n"] > 0
            and c["source_id_matches_centre_id"] == c["n"]
            and le(c["max_centre_offset_m"], SSL4EO_TOL_M)
            and le(c["max_abs_half_extent_err_m_vs_1320"], SSL4EO_TOL_M)
        )
        return ok, (
            f"n={c['n']} ids {c['source_id_matches_centre_id']}/{c['n']} centre offset "
            f"{float(c['max_centre_offset_m']):.3e} m half-extent error "
            f"{float(c['max_abs_half_extent_err_m_vs_1320']):.3e} m"
        )

    add("SSL4EO box centre equals the authors' centre (n // 4 rule)", ssl4eo)
    return res


def report(out: dict) -> bool:
    """Prints one PASS / FAIL line per check and the gate line. -> True when every check passed."""
    res = checks(out)
    for name, ok, detail in res:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    n_fail = sum(not ok for _, ok, _ in res)
    passed = n_fail == 0 and len(res) == len(SECTIONS)
    print(
        f"GATE brute-force verify {'PASS' if passed else 'FAIL'}: {len(res) - n_fail}/{len(res)} checks",
        flush=True,
    )
    if not passed:
        print("do NOT quote the dedup manifests as independently verified", flush=True)
    return passed
