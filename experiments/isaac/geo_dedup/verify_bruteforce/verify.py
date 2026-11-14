"""Brute-force verification of dedup_v2_b1000 / dedup_v3_b1000_cities. Read-only on artifacts."""

import json
import math
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mygeo
import verdict

G = os.path.join(os.environ.get("STORE", "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing"), "geo_dedup")
V2, V3 = f"{G}/dedup_v2_b1000", f"{G}/dedup_v3_b1000_cities"
BUF = 1000.0
OUT = {}
t0 = time.time()


# Aggregates below go through .to_numpy(): pandas' Series.max() / .min() SKIP NaN, so one NaN gap
# (a row the independent implementation could not place) used to vanish from the reported maximum.
# numpy propagates it, and verdict.py fails a non-finite value (review 2026-11-12, staged-delta 10).
def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


# Outputs land in the working directory. Nothing is overwritten: checked before any work.
OUTPUTS = (
    "results.json",
    "c2_excluded_sample.csv",
    "c3_retained_sample.csv",
    "c3_closest200.csv",
    "c4_v3_new_excluded_sample.csv",
    "c4_v3_retained_sample.csv",
    "c5_matches_sample.csv",
)
_present = [f for f in OUTPUTS if os.path.exists(f)]
if _present:
    sys.exit(f"FATAL: {_present} exist in {os.getcwd()}: run in an empty directory (nothing is overwritten)")


# ---------------------------------------------------------------- evaluation boxes
pts2 = pd.read_csv(f"{G}/points_v2.csv")
pts3 = pd.read_csv(f"{G}/points_v3_cities.csv")
cities = pts3[pts3["source"] == "city_roi"].reset_index(drop=True)
log("points_v2", len(pts2), "points_v3", len(pts3), "city boxes", len(cities))
assert len(pts3) == len(pts2) + len(cities)
# cities: independent rebuild from the csv in the repo (centre +- 25 km)
cc = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "so2sat_cities.csv"))
dlat = math.degrees(25000 / mygeo.R)
my_city = pd.DataFrame(
    {
        "sample_name": "city:" + cc["city"],
        "lon_min": cc["lon"] - dlat / np.cos(np.radians(cc["lat"])),
        "lat_min": cc["lat"] - dlat,
        "lon_max": cc["lon"] + dlat / np.cos(np.radians(cc["lat"])),
        "lat_max": cc["lat"] + dlat,
    }
)
m = cities.merge(my_city, on="sample_name", suffixes=("", "_my"))
assert len(m) == 52 == len(cc)
city_box_err = max(
    float(np.abs(m[c] - m[c + "_my"]).to_numpy().max()) for c in ("lon_min", "lat_min", "lon_max", "lat_max")
)
OUT["city_boxes_max_abs_deg_diff"] = city_box_err
log("city boxes rebuilt, max abs deg diff", city_box_err)


def ev_arrays(df):
    return {c: df[c].to_numpy(dtype=float) for c in ("lon_min", "lat_min", "lon_max", "lat_max")}


EV2, EV3C = ev_arrays(pts2), ev_arrays(cities)


def min_gap(box, ev, window_deg):
    """min gap (m) of one box to all eval boxes in ev, via a coarse lat/lon prefilter.
    Returns (min_gap, argmin or -1). Rows outside the window are > window (not computed)."""
    lon1, lat1, lon2, lat2 = box
    clat = max(math.cos(math.radians(max(abs(lat1), abs(lat2)))), 0.05)
    wlon = window_deg / clat
    cand = (ev["lat_min"] <= lat2 + window_deg) & (ev["lat_max"] >= lat1 - window_deg)
    lonc = np.zeros(len(ev["lon_min"]), bool)
    for k in (-360.0, 0.0, 360.0):
        lonc |= (ev["lon_min"] + k <= lon2 + wlon) & (ev["lon_max"] + k >= lon1 - wlon)
    cand &= lonc
    idx = np.nonzero(cand)[0]
    if len(idx) == 0:
        return math.inf, -1
    d = mygeo.box_gap(
        tuple(np.float64(v) for v in box),
        (ev["lon_min"][idx], ev["lat_min"][idx], ev["lon_max"][idx], ev["lat_max"][idx]),
    )
    j = int(np.argmin(d))
    return float(d[j]), int(idx[j])


# ---------------------------------------------------------------- Git-10M boxes
src_boxes = pd.read_parquet(f"{G}/source_boxes.parquet").set_index("img_name")
log("source boxes", len(src_boxes))
TILE_RE = re.compile(r"^(\d+)_(\d+)_(\d+)$")


def git_box(img_name, google_location):
    """independent box: XYZ tile bounds from the name, else source_boxes.parquet."""
    if isinstance(google_location, str) and google_location:
        z, x, y = (int(v) for v in TILE_RE.match(google_location).groups())
        return mygeo.tile_bounds(z, x, y), "tile"
    r = src_boxes.loc[img_name]
    return (float(r.lon_min), float(r.lat_min), float(r.lon_max), float(r.lat_max)), "source"


def sample_csv(path, n, seed, usecols):
    """uniform sample of n rows from a big csv in two chunked passes (memory-safe)."""
    total = 0
    for ch in pd.read_csv(path, usecols=["shard"], chunksize=1_000_000):
        total += len(ch)
    rng = np.random.default_rng(seed)
    want = np.sort(rng.choice(total, size=n, replace=False))
    parts, off = [], 0
    for ch in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        sel = want[(want >= off) & (want < off + len(ch))] - off
        if len(sel):
            parts.append(ch.iloc[sel])
        off += len(ch)
    assert off == total
    return pd.concat(parts, ignore_index=True), total


def check_rows(df, ev, window_deg):
    gaps, kinds = [], []
    for r in df.itertuples(index=False):
        b, kind = git_box(r.img_name, r.google_location)
        g, _ = min_gap(b, ev, window_deg)
        gaps.append(g)
        kinds.append(kind)
    df = df.copy()
    df["my_gap_m"] = gaps
    df["box_kind"] = kinds
    return df


# ================================================================ check 2: v2 excluded
exc2 = pd.read_csv(
    f"{V2}/excluded.csv", usecols=["shard", "row", "img_name", "google_location", "source", "min_distance_m"]
)
log("v2 excluded rows", len(exc2))
s2 = exc2.sample(2000, random_state=0).reset_index(drop=True)
s2 = check_rows(s2, EV2, 0.05)
bad2 = s2[~(s2["my_gap_m"] <= BUF)]
OUT["c2_excluded"] = {
    "n": len(s2),
    "pass": int((s2["my_gap_m"] <= BUF).sum()),
    "fail": len(bad2),
    "max_abs_diff_vs_artifact_min_distance_m": float(
        np.abs(s2["my_gap_m"] - s2["min_distance_m"]).to_numpy().max()
    ),
    "by_source": s2.groupby("source").size().to_dict(),
    "disagree": bad2[["shard", "row", "img_name", "my_gap_m", "min_distance_m"]].to_dict("records"),
}
log("check2", OUT["c2_excluded"])
s2.to_csv("c2_excluded_sample.csv", index=False)
del exc2

# ================================================================ check 3: v2 retained
r2, n_ret2 = sample_csv(
    f"{V2}/retained.csv", 20000, 0, ["shard", "row", "img_name", "google_location", "source"]
)
log("v2 retained total", n_ret2, "sampled", len(r2))
r2 = check_rows(r2, EV2, 0.5)
bad3 = r2[r2["my_gap_m"] <= BUF]
closest = r2.nsmallest(200, "my_gap_m")
q = closest["my_gap_m"].quantile([0, 0.05, 0.25, 0.5, 0.75, 1.0]).to_dict()
OUT["c3_retained"] = {
    "n": len(r2),
    "total_retained": int(n_ret2),
    "pass": int((r2["my_gap_m"] > BUF).sum()),
    "fail": len(bad3),
    "finite_within_0.5deg_window": int(np.isfinite(r2["my_gap_m"]).sum()),
    "by_source": r2.groupby("source").size().to_dict(),
    "closest200_quantiles_m": {str(k): float(v) for k, v in q.items()},
    "closest200_n_le_1100m": int((closest["my_gap_m"] <= 1100).sum()),
    "closest200_n_le_1500m": int((closest["my_gap_m"] <= 1500).sum()),
    "disagree": bad3[["shard", "row", "img_name", "my_gap_m"]].to_dict("records"),
}
log("check3", OUT["c3_retained"])
r2.to_csv("c3_retained_sample.csv", index=False)
closest.to_csv("c3_closest200.csv", index=False)

# ================================================================ check 4: v3 cities
exc2k = pd.read_csv(f"{V2}/excluded.csv", usecols=["shard", "row"])
exc3 = pd.read_csv(
    f"{V3}/excluded.csv",
    usecols=["shard", "row", "img_name", "google_location", "source", "datasets", "min_distance_m"],
)
log("v3 excluded rows", len(exc3))
k2 = set(zip(exc2k["shard"].to_numpy(), exc2k["row"].to_numpy()))
new3 = exc3[[(s, r) not in k2 for s, r in zip(exc3["shard"].to_numpy(), exc3["row"].to_numpy())]]
# v2-excluded must be a subset of v3-excluded (v3 only adds boxes)
k3 = set(zip(exc3["shard"].to_numpy(), exc3["row"].to_numpy()))
OUT["c4_v2excluded_subset_of_v3"] = {
    "v2_excluded": len(k2),
    "v2_not_in_v3": int(sum(1 for k in k2 if k not in k3)),
}
log("v3-only excluded", len(new3), OUT["c4_v2excluded_subset_of_v3"])
s4 = new3.sample(2000, random_state=0).reset_index(drop=True)
s4 = check_rows(s4, EV3C, 0.05)
bad4 = s4[~(s4["my_gap_m"] <= BUF)]
OUT["c4_v3_new_excluded"] = {
    "n": len(s4),
    "pass": int((s4["my_gap_m"] <= BUF).sum()),
    "fail": len(bad4),
    "all_datasets_contain_m_so2sat": bool(s4["datasets"].str.contains("m_so2sat").all()),
    "max_abs_diff_vs_artifact_min_distance_m": float(
        np.abs(s4["my_gap_m"] - s4["min_distance_m"]).to_numpy().max()
    ),
    "by_source": s4.groupby("source").size().to_dict(),
    "disagree": bad4[["shard", "row", "img_name", "my_gap_m", "min_distance_m"]].to_dict("records"),
}
log("check4a", OUT["c4_v3_new_excluded"])
s4.to_csv("c4_v3_new_excluded_sample.csv", index=False)
del exc3, exc2k, new3
r3, n_ret3 = sample_csv(
    f"{V3}/retained.csv", 2000, 0, ["shard", "row", "img_name", "google_location", "source"]
)
r3 = check_rows(r3, EV3C, 0.5)
bad4b = r3[r3["my_gap_m"] <= BUF]
OUT["c4_v3_retained"] = {
    "n": len(r3),
    "total_retained": int(n_ret3),
    "pass": int((r3["my_gap_m"] > BUF).sum()),
    "fail": len(bad4b),
    "min_gap_to_city_m": float(r3["my_gap_m"].to_numpy().min()),
    "disagree": bad4b[["shard", "row", "img_name", "my_gap_m"]].to_dict("records"),
}
log("check4b", OUT["c4_v3_retained"])
r3.to_csv("c4_v3_retained_sample.csv", index=False)

# ================================================================ check 5: matches.csv distances
mt = pd.read_csv(f"{V2}/matches.csv")
log("matches", len(mt))
s5 = mt.sample(500, random_state=0).reset_index(drop=True)
ev_idx = pts2.set_index(["dataset", "split", "sample_name"])
meta_cache = {}


def meta_row(shard, row):
    if shard not in meta_cache:
        meta_cache[shard] = (
            pq.read_table(f"{G}/git10m_meta/meta-{shard:05d}.parquet").to_pandas().set_index("row")
        )
    return meta_cache[shard].loc[row]


my5 = []
for r in s5.itertuples(index=False):
    mr = meta_row(int(r.shard), int(r.row))
    b, _ = git_box(mr.img_name, mr.google_location if isinstance(mr.google_location, str) else None)
    e = ev_idx.loc[(r.dataset, r.split, r.sample_name)]
    my5.append(float(mygeo.box_gap(b, (e.lon_min, e.lat_min, e.lon_max, e.lat_max))))
s5["my_gap_m"] = my5
diff5 = np.abs(s5["my_gap_m"] - s5["distance_m"])
OUT["c5_matches"] = {
    "n": len(s5),
    "max_abs_diff_m": float(diff5.to_numpy().max()),
    "n_diff_gt_1m": int((diff5 > 1).sum()),
    "n_zero_artifact": int((s5["distance_m"] == 0).sum()),
    "n_zero_mine": int((s5["my_gap_m"] == 0).sum()),
    "all_le_buffer_mine": bool((s5["my_gap_m"] <= BUF).all()),
}
log("check5", OUT["c5_matches"])
s5.to_csv("c5_matches_sample.csv", index=False)
del mt

# ================================================================ check 6: sanity
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)  # repo root
from eval import geo_dedup as gd  # noqa: E402  imported only for comparison

rng = np.random.default_rng(0)
tiles = r2[r2["box_kind"] == "tile"].sample(20, random_state=0)
tb = []
for r in tiles.itertuples(index=False):
    z, x, y = (int(v) for v in TILE_RE.match(r.google_location).groups())
    mine = np.array(mygeo.tile_bounds(z, x, y))
    theirs = np.array(gd.tile_bounds(z, x, y))
    mr = meta_row(int(r.shard), int(r.row))
    meta = np.array([mr.lon_min, mr.lat_min, mr.lon_max, mr.lat_max])
    tb.append(
        (
            float(np.abs(mine - theirs).max()),
            float(np.abs(mine - meta).max()),
            r.img_name == f"Google-{z}-{z}_{x}_{y}.jpg",
        )
    )
OUT["c6_tiles"] = {
    "n": 20,
    "max_abs_deg_diff_vs_code": max(t[0] for t in tb),
    "max_abs_deg_diff_vs_meta_parquet": max(t[1] for t in tb),
    "name_matches_location": int(sum(t[2] for t in tb)),
}
log("check6 tiles", OUT["c6_tiles"])
cen = (
    pd.read_csv(f"{G}/source_meta/ssl4eo/ssl4eo_centers.bin", header=None, names=["id", "lon", "lat"])
    .sort_values("id")
    .reset_index(drop=True)
)
ss = src_boxes[src_boxes["source"] == "ssl4eo"].sample(20, random_state=0)
errs, half_errs, id_ok = [], [], 0
for name, r in ss.iterrows():
    n = int(re.match(r"^SSL4EO-S12-rgb-(\d+)\.jpg$", name).group(1))
    i = n // 4
    c = cen.iloc[i]
    id_ok += int(str(int(c.id)) == str(r.source_id))
    clat, clon = (r.lat_min + r.lat_max) / 2, (r.lon_min + r.lon_max) / 2
    errs.append(float(mygeo.hav(clat, clon, c.lat, c.lon)))
    half_errs.append(abs(float(mygeo.hav(r.lat_min, clon, r.lat_max, clon)) / 2 - 1320.0))
OUT["c6_ssl4eo"] = {
    "n": 20,
    "centres_file_rows": len(cen),
    "source_id_matches_centre_id": id_ok,
    "max_centre_offset_m": max(errs),
    "max_abs_half_extent_err_m_vs_1320": max(half_errs),
}
log("check6 ssl4eo", OUT["c6_ssl4eo"])

with open("results.json", "x") as f:
    json.dump(OUT, f, indent=1, default=str)
log("results.json written")
# The verdict decides, not the fact that the script reached its end (review 2026-11-12, item 7).
if not verdict.report(OUT):
    sys.exit(1)
log("DONE")
