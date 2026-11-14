"""Tile-centre coordinates for the geo-held-out reference rows (DiffusionSat geo mode, OUTLINE §4.8).

Joins <ref>/rows.csv (shard,row,img_name,datasets) with the Git-10M metadata parquet shards
($STORE/DiT_remote_sensing/geo_dedup/git10m_meta/meta-*.parquet: shard,row,img_name,z,x,y,
lon_min,lat_min,lon_max,lat_max; bounding boxes exist for the Google tiles only) on (shard,row)
and writes <out> csv: shard,row,img_name,lon,lat with lon / lat = bbox centre. Rows without a
bbox are left out (the sampler counts them and uses lon = lat = 0). Prints HELDOUT_COORDS_OK.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import pandas as pd
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref", default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/git10m/fid_ref_geoheldout_n5000"
    )
    ap.add_argument(
        "--meta", default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/geo_dedup/git10m_meta"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    rows = pd.read_csv(os.path.join(a.ref, "rows.csv"))
    shards = sorted(rows["shard"].unique().tolist())
    parts = []
    for s in shards:
        fs = glob.glob(os.path.join(a.meta, f"meta-{s:05d}.parquet"))
        if len(fs) != 1:
            raise SystemExit(f"shard {s}: {len(fs)} parquet files")
        t = pq.read_table(
            fs[0], columns=["shard", "row", "img_name", "lon_min", "lat_min", "lon_max", "lat_max"]
        ).to_pandas()
        parts.append(t[t["row"].isin(rows.loc[rows["shard"] == s, "row"])])
    meta = pd.concat(parts)
    m = rows.merge(meta, on=["shard", "row"], how="left", suffixes=("", "_meta"))
    if len(m) != len(rows):
        raise SystemExit(f"join changed the row count {len(rows)} -> {len(m)}")
    bad = (m["img_name_meta"].notna()) & (m["img_name_meta"] != m["img_name"])
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} rows: img_name differs between rows.csv and the parquet")
    has = m["lon_min"].notna() & m["lat_min"].notna() & m["lon_max"].notna() & m["lat_max"].notna()
    m["lon"] = (m["lon_min"] + m["lon_max"]) / 2
    m["lat"] = (m["lat_min"] + m["lat_max"]) / 2
    out = m.loc[has, ["shard", "row", "img_name", "lon", "lat"]]
    out.to_csv(a.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(
        f"HELDOUT_COORDS_OK {len(out)}/{len(rows)} rows with a bbox; lon [{out.lon.min():.2f}, {out.lon.max():.2f}] "
        f"lat [{out.lat.min():.2f}, {out.lat.max():.2f}] -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
