"""Compare two `eval.geo_dedup dedup` manifest directories row by row.

Written for the box_gap_m poleward-latitude fix (2026-10-13): the old gap read up to 0.2 m
high, so rows whose true edge gap lies in (buffer - 0.2, buffer] metres could have been
retained by the old run and are excluded by the fixed one. Nothing can move the other way.

    python experiments/isaac/geo_dedup/diff_manifests.py OLD_DIR NEW_DIR [--buffer-m 1000] [--tol-m 0.2]

Prints the per-manifest row counts, every row whose decision changed (with its matched
distance from the new run's matches.csv), and MANIFEST_DIFF_PASS when every changed row is
old-retained -> new-excluded with the new distance inside (buffer - tol, buffer]; otherwise
MANIFEST_DIFF_FAIL with the offending rows. Identical manifests also PASS (0 moved rows).
Exit status 0 on PASS, 1 on FAIL.
"""

import argparse
import os
import sys

import pandas as pd

MANIFESTS = ("retained", "excluded", "unassessable")


def decisions(d):
    out = {}
    for m in MANIFESTS:
        df = pd.read_csv(
            os.path.join(d, f"{m}.csv"), usecols=["shard", "row"], dtype={"shard": "int32", "row": "int64"}
        )
        out[m] = pd.MultiIndex.from_frame(df)
        if not out[m].is_unique:
            raise SystemExit(f"{d}/{m}.csv has duplicate (shard,row) keys")
    keys = out["retained"].append(out["excluded"]).append(out["unassessable"])
    if not keys.is_unique:
        raise SystemExit(f"{d}: a (shard,row) appears in more than one manifest")
    s = pd.Series(pd.NA, index=keys, dtype="object")
    for m in MANIFESTS:
        s.loc[out[m]] = m
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--buffer-m", type=float, default=1000.0)
    ap.add_argument("--tol-m", type=float, default=0.2, help="old gap over-read bound (m)")
    ap.add_argument("--out", help="csv of moved rows")
    a = ap.parse_args()
    old, new = decisions(a.old), decisions(a.new)
    for m in MANIFESTS:
        print(f"{m:13s} old={int((old == m).sum()):>9d} new={int((new == m).sum()):>9d}")
    only_old, only_new = old.index.difference(new.index), new.index.difference(old.index)
    if len(only_old) or len(only_new):
        print(
            f"MANIFEST_DIFF_FAIL: row universes differ (only old {len(only_old)}, only new {len(only_new)})"
        )
        sys.exit(1)
    new = new.reindex(old.index)
    moved = old.index[(old.values != new.values)]
    print(f"moved rows: {len(moved)}")
    if len(moved) == 0:
        print("MANIFEST_DIFF_PASS: manifests identical")
        return
    matches = pd.read_csv(os.path.join(a.new, "matches.csv"), usecols=["shard", "row", "distance_m"])
    dist = matches.groupby(["shard", "row"])["distance_m"].min()
    rows = pd.DataFrame({"old": old.loc[moved].values, "new": new.loc[moved].values}, index=moved)
    rows["new_min_distance_m"] = dist.reindex(moved).values
    ok = (
        (rows["old"] == "retained")
        & (rows["new"] == "excluded")
        & (rows["new_min_distance_m"] > a.buffer_m - a.tol_m)
        & (rows["new_min_distance_m"] <= a.buffer_m)
    )
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(rows.reset_index().to_string(index=False, max_rows=200))
    if a.out:
        rows.reset_index().to_csv(a.out, index=False)
    bad = rows[~ok]
    if len(bad):
        print(
            f"MANIFEST_DIFF_FAIL: {len(bad)} moved rows are not retained->excluded inside ({a.buffer_m - a.tol_m}, {a.buffer_m}] m"
        )
        sys.exit(1)
    print(
        f"MANIFEST_DIFF_PASS: {len(rows)} rows retained->excluded, new gap in ({a.buffer_m - a.tol_m}, {a.buffer_m}] m (old box_gap_m over-read)"
    )


if __name__ == "__main__":
    main()
