"""Geographic exclusion of Git-10M samples that sit at or near GEO-Bench evaluation locations.

    python -m eval.geo_dedup git10m-meta --shard-dir DIR --shards 0-786 --out META [--hf-download] [--rm-shard]
    python -m eval.geo_dedup eval-points --geobench-root DIR --out POINTS.csv
    python -m eval.geo_dedup dedup --meta META --expect-shards 787 --points POINTS.csv --buffer-m 1000 \\
        --out OUT [--dry-run] [--allow-partial]

Git-10M (lcybuaa/Git-10M, 787 arrow shards) carries no lat/lon: the Google Earth tiles have
`Google_location = "z_x_y"` (XYZ Web Mercator tile, the dataset card's XYZToLonLat), the rows
taken from public datasets have `Google_location = None`. A tile's footprint is exact (its
Mercator bounds); a None row is UNASSESSABLE and is listed, never retained silently.

GEO-Bench side, decided from the shipped artifact (first sample of the partition):
  * footprint: the hdf5 band attrs carry transform + crs (m-brick-kiln: EPSG:4326). The box is
    the lon/lat bbox of the reprojected corners, a conservative superset of the true footprint.
  * centre+nominal: no georeferencing in the hdf5 (m-forestnet, m-pv4ger, verified), but the
    sample NAME is a centre (`lat_lon_YYYY_MM_DD`; `lat,lon` sometimes written `lon,lat`). The
    box is centre +- EVAL_HALF_EXTENT_M, a documented nominal extent, never a footprint.
  * unassessable: `id_N` names and no georeferencing (m-eurosat, m-so2sat, m-bigearthnet, see
    UNASSESSABLE): no Git-10M row is excluded on their account; they are reported, not skipped.
    m-eurosat / m-bigearthnet are then located by pixel join to their originals (eval-points
    --originals-root); m-so2sat, whose source publishes no coordinates at all, can only be
    covered at CITY level (`city-points`: the 52 So2Sat LCZ42 cities as ~50 x 50 km boxes).
Rule: a tile is excluded when the great-circle gap between its box and an evaluation box is
<= --buffer-m (0 m = the boxes overlap).

This only addresses overlap between Git-10M and GEO-Bench. It says nothing about whether
the DINOv3-SAT, CLIP, RemoteCLIP or GeoRSCLIP pretraining corpora were independent of
GEO-Bench, and it is not an image-similarity deduplication (none exists in this repo).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import pickle
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.spatial import cKDTree

HF_REPO = "lcybuaa/Git-10M"
N_SHARDS = 787
EARTH_R = 6_371_008.8  # metres, IUGG mean radius
MAX_LAT = math.degrees(math.atan(math.sinh(math.pi)))  # Web Mercator limit, 85.0511...
TILE_RE = re.compile(r"^(\d+)_(\d+)_(\d+)$")
META_COLS = ["img_name", "Google_location"]  # read from the shards; image bytes never decoded
# Nominal half extent (m) of one evaluation sample = image px x coarsest band resolution / 2.
# px: eval.export_geobench.SPECS img_hw (checked on every sample); resolution: shipped
# task_specs.pkl (test_geo_dedup checks both when the files are present). Conservative:
# the coarsest band is used, not the finest.
EVAL_HALF_EXTENT_M = {"m_forestnet": 256 * 30 / 2, "m_pv4ger": 256 * 0.1 / 2}
# So2Sat LCZ42 (Zhu et al. 2020, sec. III.B.2): each city's region of interest is "a rectangle of
# approximately 50 x 50 kilometers around the city center". City-level boxes are centre +- this.
SO2SAT_CITY_HALF_EXTENT_M = 25_000.0
# An evaluation box wider than this (a city ROI) is matched by brute force against every Git-10M
# row instead of through the KD-tree, whose query radius would otherwise be tens of km for
# every tile (millions of candidates per chunk).
BIG_EVAL_BOX_M = 10_000.0
UNASSESSABLE = {
    "m_eurosat": "samples named id_N; shipped hdf5 has no crs/transform (verified id_0000)",
    "m_so2sat": "samples named id_N; shipped hdf5 has no crs/transform (verified id_0005); So2Sat LCZ42 publishes no coordinates",
    "m_bigearthnet": "samples named id_N; shipped hdf5 has no crs/transform (verified id_0034)",
}
GERMANY_LAT, GERMANY_LON = (47.2, 55.1), (5.8, 15.1)  # disjoint ranges: lat/lon order is decidable
FORESTNET_RE = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_\d{4}_\d{2}_\d{2}$")
PV4GER_RE = re.compile(r"^\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*$")


# --------------------------------------------------------------------------- geometry
def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) of XYZ tile z/x/y (Web Mercator, y down)."""
    if not (0 <= z <= 30 and 0 <= x < 2**z and 0 <= y < 2**z):
        raise ValueError(f"tile out of range: z={z} x={x} y={y}")
    n = 2.0**z

    def lat(yy):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; arguments in degrees, broadcasting numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _unit_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


# ----------------------------------------------------------------------- Git-10M side
def parse_google_location(loc: str | None, img_name: str) -> tuple[int, int, int] | None:
    """`z_x_y` -> ints, None -> None (unassessable). Malformed or inconsistent values raise."""
    if loc is None:
        if img_name.startswith("Google-"):
            raise ValueError(f"{img_name}: Google tile without Google_location")
        return None
    m = TILE_RE.match(loc)
    if not m:
        raise ValueError(f"{img_name}: cannot parse Google_location {loc!r}")
    z, x, y = (int(v) for v in m.groups())
    tile_bounds(z, x, y)  # range check
    if img_name.startswith("Google-") and img_name != f"Google-{z}-{z}_{x}_{y}.jpg":
        raise ValueError(f"{img_name}: name does not match Google_location {loc!r}")
    return z, x, y


def shard_meta(path: str, shard: int) -> pd.DataFrame:
    """Per-row metadata of one Git-10M arrow shard (image bytes are never decoded)."""
    rows = []
    with pa.memory_map(path) as src:
        for batch in ipc.open_stream(src):
            rows.append(batch.select(META_COLS).to_pandas())
    df = pd.concat(rows, ignore_index=True)
    if df["img_name"].isna().any() or (df["img_name"] == "").any():
        raise ValueError(f"{path}: empty img_name")
    tiles = [parse_google_location(loc, name) for loc, name in zip(df["Google_location"], df["img_name"])]
    out = pd.DataFrame({"shard": np.int32(shard), "row": np.arange(len(df), dtype=np.int64)})
    out["img_name"] = df["img_name"].astype(str)
    out["google_location"] = df["Google_location"]
    has = np.array([t is not None for t in tiles])
    for col in ("z", "x", "y"):
        out[col] = pd.array([None] * len(df), dtype="Int32")
    for col in ("lon_min", "lat_min", "lon_max", "lat_max"):
        out[col] = np.nan
    if has.any():
        zxy = np.array([t for t in tiles if t is not None], dtype=np.int64)
        for i, col in enumerate(("z", "x", "y")):
            out.loc[has, col] = zxy[:, i]
        b = np.array([tile_bounds(*t) for t in zxy.tolist()])
        out.loc[has, ["lon_min", "lat_min", "lon_max", "lat_max"]] = b
    return out


def git10m_meta(shard_dir: str, shards: range, out_dir: str, hf_download: bool, rm_shard: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i in shards:
        fname = f"data-{i:05d}-of-{N_SHARDS:05d}.arrow"
        out = os.path.join(out_dir, f"meta-{i:05d}.parquet")
        if os.path.exists(out):  # written via .tmp + os.replace, so present == complete: resumable
            print(f"shard {i}: {out} exists, skipping", flush=True)
            continue
        path = os.path.join(shard_dir, "train", fname)  # the Hub's layout
        if not os.path.exists(path):
            if not hf_download:
                raise SystemExit(f"{path} missing (pass --hf-download)")
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(HF_REPO, f"train/{fname}", repo_type="dataset", local_dir=shard_dir)
            if os.path.realpath(got) != os.path.realpath(path):
                raise SystemExit(f"hf_hub_download landed at {got}, expected {path}")
        df = shard_meta(path, i)
        df.to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)  # a partial file never looks complete
        print(f"shard {i}: {len(df)} rows, {int(df['z'].notna().sum())} with tile location", flush=True)
        if rm_shard:
            os.remove(path)


def git10m_text(
    shard_dir: str, shards: range, out_dir: str, prefix: str, hf_download: bool, rm_shard: bool
) -> None:
    """text-NNNNN.parquet: shard, row, img_name, caption for the rows whose img_name starts with
    `prefix` (the captions are needed to verify a public-source name -> location mapping; the
    meta parquets deliberately carry none). Same download / resume / cleanup rules as git10m_meta."""
    os.makedirs(out_dir, exist_ok=True)
    for i in shards:
        fname = f"data-{i:05d}-of-{N_SHARDS:05d}.arrow"
        out = os.path.join(out_dir, f"text-{i:05d}.parquet")
        if os.path.exists(out):
            print(f"shard {i}: {out} exists, skipping", flush=True)
            continue
        path = os.path.join(shard_dir, "train", fname)
        if not os.path.exists(path):
            if not hf_download:
                raise SystemExit(f"{path} missing (pass --hf-download)")
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(HF_REPO, f"train/{fname}", repo_type="dataset", local_dir=shard_dir)
            if os.path.realpath(got) != os.path.realpath(path):
                raise SystemExit(f"hf_hub_download landed at {got}, expected {path}")
        rows = []
        with pa.memory_map(path) as src:
            for batch in ipc.open_stream(src):
                rows.append(batch.select(["img_name", "caption"]).to_pandas())
        df = pd.concat(rows, ignore_index=True)
        df.insert(0, "row", np.arange(len(df), dtype=np.int64))
        df.insert(0, "shard", np.int32(i))
        df = df[df["img_name"].astype(str).str.startswith(prefix)].reset_index(drop=True)
        df.to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        print(f"shard {i}: {len(df)} rows with prefix {prefix!r}", flush=True)
        if rm_shard:
            os.remove(path)


def load_meta(meta_dir: str, expect_shards: int, allow_partial: bool) -> tuple[pd.DataFrame, list[int]]:
    files = sorted(f for f in os.listdir(meta_dir) if re.fullmatch(r"meta-\d{5}\.parquet", f))
    present = [int(f[5:10]) for f in files]
    missing = sorted(set(range(expect_shards)) - set(present))
    if (missing or max(present, default=-1) >= expect_shards) and not allow_partial:
        raise SystemExit(f"{len(present)}/{expect_shards} shards in {meta_dir}; missing {missing[:10]}...")
    df = pd.concat([pd.read_parquet(os.path.join(meta_dir, f)) for f in files], ignore_index=True)
    df = df.sort_values(["shard", "row"], kind="stable").reset_index(drop=True)
    # Git-10M repeats some public-dataset images on adjacent rows (one row per caption;
    # 1,287,310 rows in pass 1, 602,844 once the SSL4EO / SkyScript rows are located; all
    # without Google_location, verified 2026-10-11). Rows stay keyed by (shard, row), so
    # repeats are reported, not fatal. A repeated Google tile name would mean a parsing fault.
    tile = df["google_location"].notna()
    dup = df["img_name"].duplicated(keep=False)
    if (dup & tile).any():
        raise SystemExit(
            f"{int((dup & tile).sum())} tile rows share an img_name, "
            f"e.g. {df.loc[dup & tile, 'img_name'].iloc[:3].tolist()}"
        )
    return df, missing


# --------------------------------------------------------------------- GEO-Bench side
class Unassessable(Exception):
    pass


class _AffineShim:
    """Old geobench pickles hold affine<3 namedtuples (NEWOBJ with 9 args); affine 3.x is an
    attrs class whose __new__ ignores them. Rebuild as the plain 9-tuple (a b c d e f 0 0 1)."""

    def __new__(cls, *args):
        return tuple(args)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("affine", "Affine"):
            return _AffineShim
        return super().find_class(module, name)


def hdf5_footprint(blob: bytes, name: str) -> tuple[float, float, float, float] | None:
    """Lon/lat box of one shipped GEO-Bench hdf5 sample from its per-band transform + crs, or
    None when the writer stored no georeferencing (keys absent). Reprojected corner bbox:
    a conservative superset of the true footprint. A sample with the keys but None values,
    or with georeferencing on only some bands, is an error, not None."""
    import geobench  # noqa: F401  registers the ccb.* module aliases the pickles reference
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as h:
        attrs = _Unpickler(io.BytesIO(ast.literal_eval(h.attrs["pickle"]))).load()
        boxes = []
        for band_name in attrs["bands_order"]:
            band = attrs[band_name]
            if "transform" not in band and "crs" not in band:
                if boxes:
                    raise ValueError(f"{name}: band {band_name!r} has no georeferencing but earlier bands do")
                continue
            t, crs = band.get("transform"), band.get("crs")
            if t is None or crs is None:
                raise ValueError(f"{name}/{band_name}: transform or crs is None")
            rows, cols = h[band_name].shape[:2]
            try:
                boxes.append(raster_box(crs, t, rows, cols))
            except ValueError as e:
                raise ValueError(f"{name}/{band_name}: {e}") from None
        if not boxes:
            return None
        if len(boxes) != len(attrs["bands_order"]):
            raise ValueError(f"{name}: georeferencing on only some bands")
    arr = np.array(boxes)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())


def raw_reader(task: str, geobench_root: str, raw_root: str | None):
    """name -> hdf5 bytes, from <raw_root>/<task>/data.zip or <geobench_root>/<task>_meta/<name>.hdf5."""
    zpath = os.path.join(raw_root or "", task, "data.zip")
    if raw_root and os.path.exists(zpath):
        try:
            z = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile as e:
            raise SystemExit(f"{zpath}: {e} (incomplete download? check the Zenodo md5)") from None
        return lambda name: z.read(f"{name}.hdf5")
    d = os.path.join(geobench_root, f"{task}_meta")

    def read(name):
        p = os.path.join(d, f"{name}.hdf5")
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing and no {zpath}: raw samples are needed to read footprints")
        with open(p, "rb") as f:
            return f.read()

    return read


def _check_latlon(lat: float, lon: float, name: str) -> tuple[float, float]:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{name}: lat/lon out of range ({lat}, {lon})")
    return lat, lon


def parse_eval_name(task: str, name: str) -> tuple[float, float]:
    """Nominal centre (lat, lon) of one GEO-Bench sample, from its sample name."""
    if task == "m_forestnet":
        m = FORESTNET_RE.match(name)
        if not m:
            raise ValueError(f"m_forestnet: cannot parse {name!r}")
        return _check_latlon(float(m.group(1)), float(m.group(2)), name)
    if task == "m_pv4ger":
        m = PV4GER_RE.match(name)
        if not m:
            raise ValueError(f"m_pv4ger: cannot parse {name!r}")
        a, b = float(m.group(1)), float(m.group(2))
        in_lat, in_lon = (
            (lambda v: GERMANY_LAT[0] <= v <= GERMANY_LAT[1]),
            (lambda v: GERMANY_LON[0] <= v <= GERMANY_LON[1]),
        )
        if in_lat(a) and in_lon(b):
            return a, b
        if in_lat(b) and in_lon(a):
            return b, a
        raise ValueError(f"m_pv4ger: {name!r} is not a lat,lon or lon,lat pair inside Germany")
    if task in UNASSESSABLE:
        raise Unassessable(UNASSESSABLE[task])
    raise ValueError(f"{task}: no georeferencing in the hdf5 and no name parser")


def centre_box(lat: float, lon: float, half_m: float) -> tuple[float, float, float, float]:
    """Lon/lat box of half side half_m around a point (metres -> degrees at that latitude)."""
    dlat = math.degrees(half_m / EARTH_R)
    dlon = dlat / math.cos(math.radians(lat))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


BOX = ["lon_min", "lat_min", "lon_max", "lat_max"]


def eval_boxes(task: str, partition: dict, read=None) -> pd.DataFrame:
    """Every sample of every split of default_partition.json -> dataset, split, sample_name,
    source, lon/lat box. `read(name)` gives the raw hdf5 bytes; when the first sample carries
    georeferencing every sample must (source "footprint"), otherwise the name parsers give a
    nominal centre box (source "centre+nominal"). Raises Unassessable for the id_N tasks."""
    names = [(split, n) for split in partition for n in partition[split]]
    if len({n for _, n in names}) != len(names):
        raise ValueError(f"{task}: duplicate sample names across splits")
    footprint = read is not None and hdf5_footprint(read(names[0][1]), names[0][1]) is not None
    rows = []
    for split, name in names:
        if footprint:
            box = hdf5_footprint(read(name), name)
            if box is None:
                raise ValueError(f"{task}: {name} has no georeferencing but {names[0][1]} does")
            rows.append((task, split, name, "footprint", *box))
        else:
            lat, lon = parse_eval_name(task, name)
            rows.append(
                (task, split, name, "centre+nominal", *centre_box(lat, lon, EVAL_HALF_EXTENT_M[task]))
            )
    df = pd.DataFrame(rows, columns=["dataset", "split", "sample_name", "source", *BOX])
    return df.sort_values(["dataset", "split", "sample_name"], kind="stable").reset_index(drop=True)


def collect_eval_boxes(
    geobench_root: str, raw_root: str | None, tasks: list[str], originals_root: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """originals_root: <root>/<ORIGINALS path> per task; an id_N task whose originals are
    present is located by pixel join, otherwise it is reported unassessable."""
    frames, report = [], {}
    for task in tasks:
        part = os.path.join(geobench_root, f"{task}_meta", "default_partition.json")
        if not os.path.exists(part):
            raise SystemExit(f"{part} missing")
        with open(part) as f:
            partition = json.load(f)
        n = sum(len(v) for v in partition.values())
        read = raw_reader(task, geobench_root, raw_root)
        info = {}
        try:
            df = eval_boxes(task, partition, read)
        except Unassessable as e:
            orig = (
                os.path.join(originals_root or "", *ORIGINALS[task][0].split("/"))
                if task in ORIGINALS
                else ""
            )
            if not (originals_root and os.path.exists(orig)):
                report[task] = {"assessable": False, "n_samples": n, "reason": str(e)}
                continue
            _, loader, keys = ORIGINALS[task]
            print(f"{task}: indexing originals at {orig}", flush=True)
            df, info = join_originals(task, partition, read, loader(orig), keys)
            info["originals"] = orig
        frames.append(df.drop(columns=[c for c in df.columns if c == "original"]))
        report[task] = {"assessable": True, "n_samples": n, "source": df["source"].iloc[0], **info}
        if df["source"].iloc[0] != "footprint":
            report[task]["half_extent_m"] = EVAL_HALF_EXTENT_M[task]
        print(f"{task}: {n} samples, {report[task]}", flush=True)
    if not frames:
        raise SystemExit("no assessable GEO-Bench task: nothing to exclude against")
    return pd.concat(frames, ignore_index=True), report


def city_boxes(cities: pd.DataFrame, half_extent_m: float, task: str = "m_so2sat") -> pd.DataFrame:
    """City-level evaluation boxes for a task whose samples cannot be located (m-so2sat): one
    box of centre +- half_extent_m per city of `cities` (columns group, city, lat, lon; group
    "train" = the 42 LCZ42 training cities, "test" = the 10 held-out cities that GEO-Bench's
    valid AND test partitions come from -- So2Sat v2 validation.h5 / testing.h5 are the western
    / eastern halves of those 10, torchgeo So2Sat version 2). Source "city_roi": these are not
    sample footprints and do not say which samples sit where; a hit means "inside a city that
    So2Sat sampled", nothing finer."""
    need = {"group", "city", "lat", "lon"}
    if not need <= set(cities.columns):
        raise ValueError(f"cities csv needs columns {sorted(need)}, has {list(cities.columns)}")
    if cities["city"].duplicated().any():
        raise ValueError(f"duplicate city: {cities.loc[cities['city'].duplicated(), 'city'].tolist()}")
    bad = set(cities["group"]) - {"train", "test"}
    if bad:
        raise ValueError(f"group must be train | test, got {sorted(bad)}")
    if half_extent_m <= 0:
        raise ValueError("half_extent_m must be positive")
    rows = []
    for c in cities.itertuples(index=False):
        lat, lon = float(c.lat), float(c.lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{c.city}: lat/lon {lat}, {lon} out of range")
        rows.append((task, c.group, f"city:{c.city}", "city_roi", *centre_box(lat, lon, half_extent_m)))
    df = pd.DataFrame(rows, columns=["dataset", "split", "sample_name", "source", *BOX])
    return df.sort_values(["dataset", "split", "sample_name"], kind="stable").reset_index(drop=True)


def city_points(points_csv: str, cities_csv: str, half_extent_m: float, task: str = "m_so2sat"):
    """points csv (+ its .report.json) with `task`, unassessable there, replaced by city boxes.
    -> (points DataFrame, report dict). Refuses a points file that already locates the task."""
    pts = pd.read_csv(points_csv)
    with open(points_csv + ".report.json") as f:
        rep = json.load(f)
    if task not in rep:
        raise SystemExit(f"{points_csv}.report.json has no entry for {task}")
    if rep[task]["assessable"] or (pts["dataset"] == task).any():
        raise SystemExit(f"{task} is already located in {points_csv}: city boxes would double-count")
    cities = pd.read_csv(cities_csv)
    boxes = city_boxes(cities, half_extent_m, task)
    rep = dict(rep)
    rep[task] = {
        "assessable": True,
        "n_samples": rep[task]["n_samples"],
        "source": "city_roi",
        "located_by": f"{len(boxes)} So2Sat LCZ42 city centres +- {half_extent_m:g} m ({os.path.basename(cities_csv)})",
        "cities": {g: int(n) for g, n in boxes["split"].value_counts().items()},
        "half_extent_m": float(half_extent_m),
        "cities_sha256": _sha256_file(cities_csv),
        "previous_reason": rep[task].get("reason"),
    }
    out = pd.concat([pts, boxes], ignore_index=True)
    return out.sort_values(["dataset", "split", "sample_name"], kind="stable").reset_index(drop=True), rep


def check_city_points(points_csv: str, cities_csv: str, half_extent_m: float, task: str = "m_so2sat") -> dict:
    """An EXISTING city-points csv may be reused only when it was built from this cities csv at
    this half extent. dedup_v2.sbatch used to reuse the file by name alone, so an edited city
    list produced a manifest from the old cities under the new list's name. -> the task's report
    entry; SystemExit on any mismatch."""
    with open(points_csv + ".report.json") as f:
        rep = json.load(f)
    ent = rep.get(task)
    if not isinstance(ent, dict) or ent.get("source") != "city_roi":
        raise SystemExit(f"{points_csv}.report.json: {task} is not located by city boxes")
    want = _sha256_file(cities_csv)
    if ent.get("cities_sha256") != want:
        raise SystemExit(
            f"{points_csv} was built from a cities csv with sha256 {ent.get('cities_sha256')}, "
            f"{cities_csv} has {want}: move the points file aside and rebuild"
        )
    if float(ent.get("half_extent_m", float("nan"))) != float(half_extent_m):
        raise SystemExit(
            f"{points_csv} was built at half extent {ent.get('half_extent_m')} m, this run asks {half_extent_m:g} m"
        )
    n_csv = int((pd.read_csv(points_csv, usecols=["dataset", "source"])["source"] == "city_roi").sum())
    n_cities = len(pd.read_csv(cities_csv))
    if not (n_csv == n_cities == sum(ent["cities"].values())):
        raise SystemExit(
            f"{points_csv} holds {n_csv} city boxes, its report {sum(ent['cities'].values())}, {cities_csv} {n_cities}"
        )
    return ent


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- Git-10M public sources
# Git-10M rows taken from public datasets carry no Google_location. Whether a source can be
# located is decided from its shipped artifact (rule 16), per source:
#   SSL4EO-S12  "SSL4EO-S12-rgb-<n>.jpg". The authors' centre-coordinates file lists 251,079
#               locations (id, lon, lat; ids 0..251175 minus 97 void ids). Git-10M's n is
#               4 * i + 1 with i in 0..251078, one row per location (verified on all 251,079
#               names, 2026-10-11): the second of four seasonal images in a location-major
#               enumeration, so the location is the i-th centre in ascending id order. That
#               ordering is a hypothesis until `ssl4eo-verify` confirms it (Git-10M captions
#               vs the authors' per-location land-cover labels, against a permuted null).
#               Box: centre +- 1320 m (264 px at 10 m, the published patch extent).
#   SkyScript   "Sky-images<k>-<stem>.jpg". The authors' meta<k>.zip holds one pickle per
#               image with bbox = (lon_min, lat_min, lon_max, lat_max): an exact footprint
#               (checked on w963383994_DE_: a 50 m box in Brandenburg, matching the DE alias).
#   Million-AID, GeoPileV0, DIOR, AID, RSICB, VGoogle: no coordinates are published. Those
#               rows stay unassessable, the source named in the reason.
SOURCE_PREFIXES = (
    ("SSL4EO-S12-", "ssl4eo"),
    ("Sky-", "skyscript"),
    ("Million-AID-", "million_aid"),
    ("GeoPileV0-", "geopile"),
    ("DIOR-", "dior"),
    ("AID-", "aid"),
    ("RSICB", "rsicb"),
    ("VGoogle-", "vgoogle"),
)
SOURCE_REASON = {
    "ssl4eo": "SSL4EO-S12 row not located (run source-boxes and pass --source-boxes)",
    "skyscript": "SkyScript row not located (no meta pickle for this name, or no --source-boxes)",
    "million_aid": "Million-AID publishes no coordinates",
    "geopile": "GeoPile publishes no coordinates",
    "dior": "DIOR publishes no coordinates",
    "aid": "AID publishes no coordinates",
    "rsicb": "RSI-CB publishes no coordinates",
    "vgoogle": "VGoogle publishes no coordinates",
    "unknown": "unrecognised public-source name",
}
SSL4EO_N_LOCATIONS = 251_079
SSL4EO_SEASONS = 4
SSL4EO_HALF_EXTENT_M = 264 * 10 / 2  # 2640 m patches (SSL4EO-S12 paper / repo README)
SSL4EO_RE = re.compile(r"^SSL4EO-S12-rgb-(\d+)\.jpg$")
SKYSCRIPT_RE = re.compile(r"^Sky-(images[2-7])-(.+)\.jpg$")
# SkyScript mixes sources (image_sources.py): 10-50 cm orthophotos (CH, DE, FI, ES, US) with
# 50-200 m boxes, but also Sentinel-2 (S2) and Landsat (L8/L9) scenes of 10-30 km. The limit
# only catches a misread pickle (e.g. a1000364136_L8_21: 8 x 15 km, real).
SKYSCRIPT_MAX_SIDE_M = 100_000


def source_of(names: pd.Series, google_location: pd.Series) -> pd.Series:
    """Per-row source label: google_tile for rows with a Google_location, else by name prefix."""
    out = pd.Series("unknown", index=names.index, dtype=object)
    for prefix, src in SOURCE_PREFIXES:
        out[names.str.startswith(prefix)] = src
    out[google_location.notna()] = "google_tile"
    return out


def _centre_boxes(lat: np.ndarray, lon: np.ndarray, half_m: float) -> np.ndarray:
    """Vectorised centre_box: (N, 4) lon_min, lat_min, lon_max, lat_max."""
    dlat = math.degrees(half_m / EARTH_R)
    dlon = dlat / np.cos(np.radians(lat))
    return np.stack([lon - dlon, lat - dlat, lon + dlon, lat + dlat], axis=1)


def ssl4eo_centres(centres_csv: str, void_csv: str) -> pd.DataFrame:
    """The authors' centre file (id, lon, lat; no header) in ascending id order, checked against
    the published void-id list and the published location count."""
    c = pd.read_csv(centres_csv, header=None, names=["id", "lon", "lat"])
    void = pd.read_csv(void_csv, header=None)[0]
    if len(c) != SSL4EO_N_LOCATIONS or c["id"].nunique() != len(c):
        raise SystemExit(
            f"{centres_csv}: {len(c)} rows / {c['id'].nunique()} unique ids, expected {SSL4EO_N_LOCATIONS}"
        )
    if c["id"].isin(void).any():
        raise SystemExit(f"{centres_csv}: {int(c['id'].isin(void).sum())} void ids present")
    if not (c["lon"].between(-180, 180).all() and c["lat"].between(-90, 90).all()):
        raise SystemExit(f"{centres_csv}: lon/lat out of range")
    return c.sort_values("id").reset_index(drop=True)


def ssl4eo_index(names: pd.Series) -> pd.Series:
    """Location index i (0-based, into the ascending-id centre list) of each Git-10M SSL4EO name."""
    n = names.str.extract(SSL4EO_RE)[0]
    if n.isna().any():
        raise SystemExit(f"unparseable SSL4EO names, e.g. {names[n.isna()].iloc[:3].tolist()}")
    n = n.astype(np.int64)
    season = n % SSL4EO_SEASONS
    if season.nunique() != 1:
        raise SystemExit(
            f"SSL4EO names span several season indices {sorted(season.unique())}: the n = 4i + s rule does not hold"
        )
    return n // SSL4EO_SEASONS


def ssl4eo_boxes(names: pd.Series, centres: pd.DataFrame) -> pd.DataFrame:
    """img_name -> centre+nominal box for unique Git-10M SSL4EO names."""
    names = names.drop_duplicates().reset_index(drop=True)
    i = ssl4eo_index(names)
    if i.duplicated().any() or (i >= len(centres)).any():
        raise SystemExit("SSL4EO names do not map one-to-one onto the centre list")
    pick = centres.iloc[i.to_numpy()]
    b = _centre_boxes(pick["lat"].to_numpy(), pick["lon"].to_numpy(), SSL4EO_HALF_EXTENT_M)
    out = pd.DataFrame(
        {"img_name": names, "source": "ssl4eo", "source_id": pick["id"].astype(str).to_numpy()}
    )
    for k, col in enumerate(BOX):
        out[col] = b[:, k]
    return out


class _SkyUnpickler(pickle.Unpickler):
    """Only what SkyScript's meta pickles need (numpy float64 scalars); every other global is refused,
    so a planted pickle cannot run code."""

    def find_class(self, module, name):
        if (module, name) == ("numpy.core.multiarray", "scalar"):
            return np.core.multiarray.scalar
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def skyscript_box(path: str, data: bytes | None = None) -> tuple[float, float, float, float]:
    if data is None:
        with open(path, "rb") as f:
            data = f.read()
    d = _SkyUnpickler(io.BytesIO(data)).load()
    b = tuple(float(v) for v in d["bbox"])
    if len(b) != 4 or not (-180 <= b[0] <= b[2] <= 180 and -90 <= b[1] <= b[3] <= 90):
        raise ValueError(f"{path}: bbox {b} is not lon_min, lat_min, lon_max, lat_max")
    if haversine_m(b[1], b[0], b[3], b[2]) > SKYSCRIPT_MAX_SIDE_M * math.sqrt(2):
        raise ValueError(f"{path}: bbox {b} spans more than {SKYSCRIPT_MAX_SIDE_M} m")
    return b


def skyscript_boxes(names: pd.Series, meta: str) -> tuple[pd.DataFrame, list[str]]:
    """img_name -> footprint box for unique Git-10M SkyScript names "Sky-images<k>-<stem>.jpg",
    from meta<k>/<stem>.pickle inside the authors' meta<k>.zip (`meta` = the zip; ~40 min for
    954k entries) or under an extracted directory (`meta` = its parent; ~6x slower on Lustre).
    Names whose pickle is absent are returned separately, never silently dropped."""
    names = names.drop_duplicates().reset_index(drop=True)
    m = names.str.extract(SKYSCRIPT_RE)
    if m[0].isna().any():
        raise SystemExit(f"unparseable SkyScript names, e.g. {names[m[0].isna()].iloc[:3].tolist()}")
    rows, missing = [], []
    if meta.endswith(".zip"):
        zf = zipfile.ZipFile(meta)
        have = set(zf.namelist())
        for name, part, stem in zip(names, m[0], m[1]):
            entry = f"{part.replace('images', 'meta')}/{stem}.pickle"  # images6 -> meta6/
            if entry not in have:
                missing.append(name)
                continue
            rows.append((name, "skyscript", stem, *skyscript_box(f"{meta}:{entry}", zf.read(entry))))
    else:
        for name, part, stem in zip(names, m[0], m[1]):
            p = os.path.join(meta, part.replace("images", "meta"), f"{stem}.pickle")
            if not os.path.exists(p):
                missing.append(name)
                continue
            rows.append((name, "skyscript", stem, *skyscript_box(p)))
    df = pd.DataFrame(rows, columns=["img_name", "source", "source_id", *BOX])
    return df, missing


def source_boxes(
    git: pd.DataFrame, centres: pd.DataFrame | None, skyscript_root: str | None
) -> tuple[pd.DataFrame, dict]:
    """Boxes for every locatable non-tile row of the Git-10M metadata + a per-source report."""
    src = source_of(git["img_name"], git["google_location"])
    report, frames = {}, []
    for s in sorted(set(src) - {"google_tile"}):
        names = git.loc[src == s, "img_name"]
        entry = {"rows": int(len(names)), "unique_names": int(names.nunique()), "located_names": 0}
        if s == "ssl4eo" and centres is not None:
            df = ssl4eo_boxes(names, centres)
            frames.append(df)
            entry.update(located_names=int(len(df)), box="centre +- %g m" % SSL4EO_HALF_EXTENT_M)
        elif s == "skyscript" and skyscript_root is not None:
            df, missing = skyscript_boxes(names, skyscript_root)
            frames.append(df)
            diag = (
                haversine_m(df["lat_min"], df["lon_min"], df["lat_max"], df["lon_max"])
                if len(df)
                else np.array([np.nan])
            )
            entry.update(
                located_names=int(len(df)),
                missing_meta=len(missing),
                box="footprint",
                box_diagonal_m={
                    "min": float(np.min(diag)),
                    "median": float(np.median(diag)),
                    "max": float(np.max(diag)),
                },
            )
            if missing:
                entry["missing_examples"] = missing[:5]
        else:
            entry["reason"] = SOURCE_REASON[s]
        report[s] = entry
    boxes = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["img_name", "source", "source_id", *BOX])
    )
    if boxes["img_name"].duplicated().any():
        raise SystemExit("source boxes: an img_name resolved twice")
    return boxes.sort_values("img_name", kind="stable").reset_index(drop=True), report


def apply_source_boxes(git: pd.DataFrame, boxes: pd.DataFrame | None) -> pd.DataFrame:
    """Adds `source`; with boxes, fills the box columns of the non-tile rows whose img_name is
    located. A located name that belongs to a tile row, or that is not in the metadata at all,
    means the boxes file was made from other metadata: fatal."""
    git = git.copy()
    git["source"] = source_of(git["img_name"], git["google_location"])
    if boxes is None:
        return git
    if boxes["img_name"].duplicated().any():
        raise SystemExit("source boxes: duplicate img_name")
    tile_names = set(git.loc[git["google_location"].notna(), "img_name"])
    hit_tile = boxes["img_name"].isin(tile_names)
    if hit_tile.any():
        raise SystemExit(f"source boxes name {boxes.loc[hit_tile, 'img_name'].iloc[0]!r} is a Google tile")
    known = boxes["img_name"].isin(set(git["img_name"]))
    if not known.all():
        raise SystemExit(
            f"source boxes name {boxes.loc[~known, 'img_name'].iloc[0]!r} is not in the metadata"
        )
    idx = boxes.set_index("img_name")
    sel = git["img_name"].isin(idx.index)
    git.loc[sel, BOX] = idx.loc[git.loc[sel, "img_name"], BOX].to_numpy()
    srcs = idx.loc[git.loc[sel, "img_name"], "source"].to_numpy()
    if (git.loc[sel, "source"].to_numpy() != srcs).any():
        raise SystemExit("source boxes: source label disagrees with the name prefix")
    return git


# ------------------------------------------------------------- SSL4EO mapping verification
# Dynamic World class ids as used by the SSL4EO-S12 multilabel file (0 water, 1 trees, 2 grass,
# 3 flooded vegetation, 4 crops, 5 shrub & scrub, 6 built, 7 bare, 8 snow & ice; labels are
# per 2.6 km patch and are pooled over the four seasons here, so most classes are present at
# most locations: base rates 0.33-0.82 on the real file). The gate tests the NAME -> LOCATION
# hypothesis, not the class semantics: under a wrong mapping every caption/label association
# is at its permuted-null value, and under the "folder = i" alternative (voids ignored) the
# association survives only for ids below the first void.
DW_KEYWORDS = {
    "0": ("water", "river", "lake", "sea", "ocean", "coast", "coastline", "reservoir", "bay", "shoreline"),
    "1": ("forest", "forested", "trees", "wooded", "woodland"),
    "4": ("agricultural", "farmland", "crops", "crop", "cultivated", "fields"),
    "6": ("urban", "buildings", "city", "town", "residential", "built-up", "developed"),
    "7": ("desert", "sand", "sandy", "barren", "bare", "dunes", "arid"),
    "8": ("snow", "snow-covered", "ice", "glacier", "glaciers"),
}
SSL4EO_VERIFY_MIN_HITS = 200
# Criteria, fixed 2026-10-11 AFTER the first real run (recorded so the choice is auditable):
# the first draft asked every class for a gain >= 0.15 in both halves and failed on class 0
# (water: 71% base rate, gain 0.02/0.00) and class 7 (gain 0.150/0.149) while classes 1, 4,
# 6, 8 gained 0.21-0.44 identically in both halves. The statistic that separates the ORDER
# hypotheses is the across-void agreement, so that is what every class must satisfy; the
# gain floor on a majority of classes guards against a dead instrument.
SSL4EO_VERIFY_MIN_GAIN = 0.10  # precision above the permuted null, both halves
SSL4EO_VERIFY_MIN_CLASSES = 4  # ... on at least this many of the keyword classes
SSL4EO_VERIFY_MAX_HALF_DIFF = 0.05  # |gain below first void - gain above| for EVERY class


def ssl4eo_verify(texts: pd.DataFrame, multilabel: dict, centres: pd.DataFrame, seed: int = 0) -> dict:
    """Executed gate for the name -> location hypothesis. Per keyword class and per id half
    (below / above the first void id): precision = P(label present | caption keyword), null =
    the same under a random permutation of the mapping. PASS when every class gains the same
    in both halves (within SSL4EO_VERIFY_MAX_HALF_DIFF) and at least SSL4EO_VERIFY_MIN_CLASSES
    classes gain >= SSL4EO_VERIFY_MIN_GAIN in both. Returns {"pass", "detail", "stats", ...}."""
    i = ssl4eo_index(texts["img_name"]).to_numpy()
    ids = centres["id"].to_numpy()[i]
    present = {}
    for k in DW_KEYWORDS:
        present[k] = np.array(
            [k in {lab for ts in multilabel[f"{d:07d}"].values() for lab in ts} for d in ids]
        )
    cap = texts["caption"].fillna("").str.lower()
    rng = np.random.default_rng(seed)
    first_void = int(np.setdiff1d(np.arange(ids.max() + 1), ids).min()) if len(ids) else 0
    halves = {"lo": ids < first_void, "hi": ids >= first_void}
    detail, stats, strong = [], {}, 0
    consistent = True
    for k, words in DW_KEYWORDS.items():
        pat = r"\b(?:%s)\b" % "|".join(re.escape(w) for w in words)
        hit = cap.str.contains(pat, regex=True).to_numpy()
        gain = {}
        for hname, hmask in halves.items():
            m = hit & hmask
            n = int(m.sum())
            prec = float(present[k][m].mean()) if n else float("nan")
            null = (
                float(np.mean([present[k][rng.permutation(len(ids))][m].mean() for _ in range(5)]))
                if n
                else float("nan")
            )
            gain[hname] = prec - null
            stats[f"class{k}/{hname}"] = {"hits": n, "precision": prec, "null": null, "gain": gain[hname]}
            if n < SSL4EO_VERIFY_MIN_HITS:
                consistent = False
                detail.append(f"FAIL ssl4eo-verify class {k} {hname}: only {n} keyword hits")
        agree = abs(gain["lo"] - gain["hi"]) <= SSL4EO_VERIFY_MAX_HALF_DIFF
        consistent &= agree
        big = min(gain.values()) >= SSL4EO_VERIFY_MIN_GAIN
        strong += big
        detail.append(
            f"{'PASS' if agree else 'FAIL'} ssl4eo-verify class {k}: gain below/above first void "
            f"{gain['lo']:+.3f}/{gain['hi']:+.3f}{' (strong)' if big else ''}"
        )
    enough = strong >= SSL4EO_VERIFY_MIN_CLASSES
    detail.append(
        f"{'PASS' if enough else 'FAIL'} ssl4eo-verify: {strong}/{len(DW_KEYWORDS)} classes gain >= {SSL4EO_VERIFY_MIN_GAIN} in both halves"
    )
    ok = consistent and enough
    return {
        "pass": bool(ok),
        "first_void_id": first_void,
        "rows": int(len(texts)),
        "detail": detail,
        "stats": stats,
    }


# ------------------------------------------------ GEO-Bench originals (pixel join)
# GEO-Bench's converters (make_benchmark/dataset_converters/{eurosat,bigearthnet}.py) wrote the
# original pixel values unchanged (EuroSAT: uint16 -> int16; BigEarthNet: the 120 x 120 10 m
# bands through a same-size bilinear resample, i.e. identity, as int32) but stored no
# georeferencing ("TODO can't find the GPS coordinates"). So a sample is located by hashing
# the stored pixels of a band and finding the one original with the same pixels; the
# original's GeoTIFF (EuroSAT) or *_labels_metadata.json (BigEarthNet) gives the footprint.
EUROSAT_JOIN = {"02 - Blue": 1, "04 - Red": 3}  # GEO-Bench key -> 0-based GeoTIFF band (B01..B12, B8A)
BIGEARTHNET_JOIN_KEY, BIGEARTHNET_JOIN_BAND = "02 - Blue", "B02"


def pixel_hash(*arrays: np.ndarray) -> str:
    """sha1 of the int32 pixels (shape included) of one or more bands, in order."""
    import hashlib

    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.int32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sample_hash(blob: bytes, keys: list[str]) -> str:
    """pixel_hash of the named bands of a shipped GEO-Bench hdf5 sample."""
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as h:
        arrays = []
        for k in keys:
            if k not in h:
                raise ValueError(f"band {k!r} missing; sample has {list(h.keys())}")
            a = h[k][()]
            if not np.array_equal(a, np.round(a)):
                raise ValueError(f"band {k!r} is not integer-valued: the converter rescaled it")
            arrays.append(a)
    return pixel_hash(*arrays)


def raster_box(crs, transform6, rows: int, cols: int) -> tuple[float, float, float, float]:
    """Lon/lat bbox of the four reprojected raster corners (a conservative superset)."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp

    a, b, c, d, e, f = transform6[:6]
    px = [(0, 0), (cols, 0), (cols, rows), (0, rows)]
    xs = [a * col + b * row + c for col, row in px]
    ys = [d * col + e * row + f for col, row in px]
    lons, lats = warp(CRS.from_user_input(crs), "EPSG:4326", xs, ys)
    if not all(-180 <= v <= 180 for v in lons) or not all(-90 <= v <= 90 for v in lats):
        raise ValueError("corners outside lon/lat range")
    return min(lons), min(lats), max(lons), max(lats)


def eurosat_index(ms_dir: str) -> pd.DataFrame:
    """Every GeoTIFF under the EuroSAT MS directory -> path, pixel hash of B02+B04 (as the
    int16 GEO-Bench stored), footprint box."""
    import glob

    import rasterio

    paths = sorted(glob.glob(os.path.join(ms_dir, "*", "*.tif")))
    if not paths:
        raise SystemExit(f"{ms_dir}: no <class>/<name>.tif")
    rows = []
    for p in paths:
        with rasterio.open(p) as src:
            if src.count != 13 or src.dtypes[0] != "uint16":
                raise SystemExit(f"{p}: {src.count} bands of {src.dtypes[0]}, expected 13 x uint16")
            bands = [src.read(j + 1).view(np.int16) for j in EUROSAT_JOIN.values()]
            t = src.transform
            box = raster_box(src.crs, (t.a, t.b, t.c, t.d, t.e, t.f), src.height, src.width)
        rows.append((os.path.relpath(p, ms_dir), pixel_hash(*bands), *box))
    return pd.DataFrame(rows, columns=["original", "hash", *BOX])


def bigearthnet_index(tar_path: str) -> pd.DataFrame:
    """One streaming pass over BigEarthNet-S2-v1.0.tar.gz: per patch, the pixel hash of its
    B02 band (uint16 -> int32, as GEO-Bench stored) and the footprint from the patch json's
    corner coordinates + projection. Nothing is extracted to disk."""
    import tarfile

    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp

    pending: dict[str, dict] = {}
    rows = []

    def flush(patch, d):
        meta = json.loads(d["json"])
        c, proj = meta["coordinates"], meta["projection"]
        with rasterio.MemoryFile(d["tif"]) as mf, mf.open() as src:
            if src.count != 1 or src.dtypes[0] != "uint16":
                raise SystemExit(f"{patch}: B02 is {src.count} x {src.dtypes[0]}")
            h = pixel_hash(src.read(1))
            crs_tif = src.crs
        xs = [c["ulx"], c["lrx"], c["lrx"], c["ulx"]]
        ys = [c["uly"], c["uly"], c["lry"], c["lry"]]
        lons, lats = warp(CRS.from_wkt(proj), "EPSG:4326", xs, ys)
        if not all(-180 <= v <= 180 for v in lons) or not all(-90 <= v <= 90 for v in lats):
            raise SystemExit(f"{patch}: corners outside lon/lat range")
        rows.append(
            (
                patch,
                h,
                min(lons),
                min(lats),
                max(lons),
                max(lats),
                meta.get("tile_source"),
                meta.get("acquisition_date"),
                str(crs_tif),
            )
        )

    with tarfile.open(tar_path, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            patch = os.path.basename(os.path.dirname(m.name))
            if base.endswith("_labels_metadata.json"):
                key = "json"
            elif base.endswith(f"_{BIGEARTHNET_JOIN_BAND}.tif"):
                key = "tif"
            else:
                continue
            d = pending.setdefault(patch, {})
            if key in d:
                raise SystemExit(f"{patch}: two {key} members")
            d[key] = tar.extractfile(m).read()
            if "json" in d and "tif" in d:
                flush(patch, pending.pop(patch))
                if len(rows) % 50_000 == 0:
                    print(f"  {len(rows)} patches", flush=True)
    if pending:
        raise SystemExit(
            f"{len(pending)} patches without both json and {BIGEARTHNET_JOIN_BAND}, e.g. {list(pending)[:3]}"
        )
    df = pd.DataFrame(rows, columns=["original", "hash", *BOX, "tile_source", "acquisition_date", "crs"])
    if df["original"].duplicated().any():
        raise SystemExit("duplicate patch names in the archive")
    return df.sort_values("original", kind="stable").reset_index(drop=True)


def join_originals(
    task: str, partition: dict, read, index: pd.DataFrame, keys: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Every partition sample -> the original(s) with identical pixels -> footprint box.
    A sample with no match is fatal (the converter changed pixels, or the originals differ
    from what GEO-Bench used); a sample matching several originals (duplicated patches in the
    source dataset) gets the union box and is counted."""
    by_hash = index.groupby("hash")
    groups = {h: g for h, g in by_hash}
    rows, unmatched, multi = [], [], 0
    for split in partition:
        for name in partition[split]:
            h = sample_hash(read(name), keys)
            g = groups.get(h)
            if g is None:
                unmatched.append(name)
                continue
            if len(g) > 1:
                multi += 1
            box = (g["lon_min"].min(), g["lat_min"].min(), g["lon_max"].max(), g["lat_max"].max())
            rows.append((task, split, name, "footprint", *box, ";".join(g["original"])))
    if unmatched:
        raise SystemExit(
            f"{task}: {len(unmatched)} samples match no original by pixels, e.g. {unmatched[:5]}"
        )
    df = pd.DataFrame(rows, columns=["dataset", "split", "sample_name", "source", *BOX, "original"])
    info = {"located_by": f"pixel join on {keys} to {len(index)} originals", "multi_match": multi}
    return df.sort_values(["dataset", "split", "sample_name"], kind="stable").reset_index(drop=True), info


ORIGINALS = {  # task -> (relative path under --originals-root, loader, join keys)
    "m_eurosat": ("eurosat/EuroSAT_MS", eurosat_index, list(EUROSAT_JOIN)),
    "m_bigearthnet": ("bigearthnet/patch_index.parquet", pd.read_parquet, [BIGEARTHNET_JOIN_KEY]),
}


# ---------------------------------------------------------------------------- matching
def box_gap_m(a: dict, b: dict):
    """Great-circle gap (m) between lon/lat boxes a (scalars) and b (arrays); 0 when they
    overlap. Nearest pair = facing edges; when the latitude ranges overlap the shared latitude
    FARTHEST from the equator (meridians converge poleward, so that is where two lon-separated
    edges are closest; the equator-nearest choice overestimated a 1,000 m gap between city-sized
    boxes at 50 N by ~1 %); b is tried shifted by +-360 deg so boxes touching the antimeridian
    are handled."""
    lo, hi = np.maximum(a["lat_min"], b["lat_min"]), np.minimum(a["lat_max"], b["lat_max"])
    below = a["lat_max"] < b["lat_min"]  # a entirely south of b
    poleward = np.where(np.abs(lo) >= np.abs(hi), lo, hi)
    lat_a = np.where(lo <= hi, poleward, np.where(below, a["lat_max"], a["lat_min"]))
    lat_b = np.where(lo <= hi, lat_a, np.where(below, b["lat_min"], b["lat_max"]))
    best = np.full(np.shape(lat_b), np.inf)
    for k in (-360.0, 0.0, 360.0):
        bl, bh = b["lon_min"] + k, b["lon_max"] + k
        lo2, hi2 = np.maximum(a["lon_min"], bl), np.minimum(a["lon_max"], bh)
        west = a["lon_max"] < bl  # a entirely west of b
        lon_a = np.where(lo2 <= hi2, lo2, np.where(west, a["lon_max"], a["lon_min"]))
        lon_b = np.where(lo2 <= hi2, lo2, np.where(west, bl, bh))
        best = np.minimum(best, haversine_m(lat_a, lon_a, lat_b, lon_b))
    return best


def _half_diag_m(df: pd.DataFrame) -> np.ndarray:
    lat_c, lon_c = (
        (df["lat_min"] + df["lat_max"]).to_numpy() / 2,
        (df["lon_min"] + df["lon_max"]).to_numpy() / 2,
    )
    return np.max(
        [
            haversine_m(lat_c, lon_c, df[a].to_numpy(), df[b].to_numpy())
            for a in ("lat_min", "lat_max")
            for b in ("lon_min", "lon_max")
        ],
        axis=0,
    )


def nearby(git: pd.DataFrame, ev: pd.DataFrame, buffer_m: float, chunk: int = 500_000) -> pd.DataFrame:
    """Every (git row, eval sample) pair whose box gap <= buffer_m (0 = the boxes overlap).
    A KD-tree on the eval box centres' unit vectors is queried per tile with radius = the
    tile's half-diagonal + the largest eval half-diagonal + buffer (a superset); candidates
    then take the exact test. Returns shard, row, dataset, split, sample_name, distance_m."""
    cols = ["shard", "row", "dataset", "split", "sample_name", "distance_m"]
    if len(ev) == 0 or len(git) == 0:
        return pd.DataFrame(columns=cols)
    big = _half_diag_m(ev) > BIG_EVAL_BOX_M
    out = _nearby_big(git, ev[big], buffer_m, chunk) if big.any() else []
    ev = ev[~big]
    if len(ev) == 0:
        df = pd.DataFrame(out, columns=cols)
        return df.sort_values(["shard", "row", "dataset", "sample_name"], kind="stable").reset_index(
            drop=True
        )
    ev_lat, ev_lon = (
        (ev["lat_min"] + ev["lat_max"]).to_numpy() / 2,
        (ev["lon_min"] + ev["lon_max"]).to_numpy() / 2,
    )
    tree = cKDTree(_unit_xyz(ev_lat, ev_lon))
    slack = float(_half_diag_m(ev).max()) + buffer_m
    ev_box = {c: ev[c].to_numpy() for c in BOX}
    for start in range(0, len(git), chunk):
        g = git.iloc[start : start + chunk]
        lat_c, lon_c = (
            (g["lat_min"] + g["lat_max"]).to_numpy() / 2,
            (g["lon_min"] + g["lon_max"]).to_numpy() / 2,
        )
        r = 2 * np.sin(np.minimum((_half_diag_m(g) + slack) / EARTH_R, math.pi) / 2)  # chord length
        for gi, cand in enumerate(tree.query_ball_point(_unit_xyz(lat_c, lon_c), r)):
            if not cand:
                continue
            row, cand = g.iloc[gi], np.array(sorted(cand))
            d = box_gap_m({c: float(row[c]) for c in BOX}, {c: v[cand] for c, v in ev_box.items()})
            for j, dist in zip(cand[d <= buffer_m], d[d <= buffer_m]):
                q = ev.iloc[j]
                out.append(
                    (
                        int(row["shard"]),
                        int(row["row"]),
                        q["dataset"],
                        q["split"],
                        q["sample_name"],
                        float(dist),
                    )
                )
    df = pd.DataFrame(out, columns=cols)
    return df.sort_values(["shard", "row", "dataset", "sample_name"], kind="stable").reset_index(drop=True)


def _nearby_big(git: pd.DataFrame, ev: pd.DataFrame, buffer_m: float, chunk: int) -> list[tuple]:
    """Exact box-gap test of every Git-10M row against each (few, wide) evaluation box."""
    g_box = {c: git[c].to_numpy() for c in BOX}
    shard, row = git["shard"].to_numpy(), git["row"].to_numpy()
    out = []
    for q in ev.itertuples(index=False):
        a = {c: float(getattr(q, c)) for c in BOX}
        for start in range(0, len(git), chunk):
            sl = slice(start, start + chunk)
            d = box_gap_m(a, {c: v[sl] for c, v in g_box.items()})
            for i in np.nonzero(d <= buffer_m)[0]:
                out.append(
                    (
                        int(shard[start + i]),
                        int(row[start + i]),
                        q.dataset,
                        q.split,
                        q.sample_name,
                        float(d[i]),
                    )
                )
    return out


def check_boxes(df: pd.DataFrame, what: str, allow_unlocated: bool) -> np.ndarray:
    """-> boolean array: the row has a box. A box is four FINITE corners with min <= max on both
    axes. A row with one to three missing corners, or an infinite one, is fatal: every comparison
    with NaN is False, so such a row can never match an evaluation sample and would be written to
    retained.csv as if it had been assessed. An INVERTED box (lat_min > lat_max or lon_min >
    lon_max) is fatal for the same reason: box_gap_m then reads an overlapping pair as disjoint
    (a box across the antimeridian is stored with lon_max > 180, never inverted; box_gap_m tries
    the +-360 shifts). Rows with no corner at all are unlocated (allowed for Git-10M only)."""
    v = df[BOX].to_numpy(dtype=float)
    missing = np.isnan(v)
    none, bad = missing.all(axis=1), missing.any(axis=1) & ~missing.all(axis=1)
    bad |= np.isinf(v).any(axis=1)
    if bad.any():
        raise SystemExit(
            f"{what}: {int(bad.sum())} rows with a partial or infinite box, first:\n{df[bad].head(3).to_string()}"
        )
    lon_min, lat_min, lon_max, lat_max = (
        v[:, BOX.index(c)] for c in ("lon_min", "lat_min", "lon_max", "lat_max")
    )
    with np.errstate(invalid="ignore"):  # all-NaN (unlocated) rows compare False: not inverted
        inverted = (lat_min > lat_max) | (lon_min > lon_max)
    if inverted.any():
        raise SystemExit(
            f"{what}: {int(inverted.sum())} rows with an inverted box (min > max), first:\n"
            f"{df[inverted].head(3).to_string()}"
        )
    if none.any() and not allow_unlocated:
        raise SystemExit(
            f"{what}: {int(none.sum())} rows without a box, first:\n{df[none].head(3).to_string()}"
        )
    return ~none


def dedup(
    git: pd.DataFrame, ev: pd.DataFrame, buffer_m: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> retained, excluded, unassessable, matches (all sorted by shard,row)."""
    if "source" not in git:
        git = apply_source_boxes(git, None)
    assessable = check_boxes(git, "Git-10M rows", allow_unlocated=True)
    check_boxes(ev, "evaluation boxes", allow_unlocated=False)
    matches = nearby(git[assessable], ev, buffer_m)
    key = ["shard", "row"]
    agg = (
        matches.groupby(key, sort=True)
        .agg(
            datasets=("dataset", lambda s: ";".join(sorted(set(s)))),
            matched_ids=("sample_name", lambda s: ";".join(s)),  # already sorted by dataset, name
            n_matches=("sample_name", "size"),
            min_distance_m=("distance_m", "min"),
        )
        .reset_index()
    )
    ids = git[key + ["img_name", "google_location", "source"]]
    excluded = ids.merge(agg, on=key, how="inner")
    excluded.insert(5, "reason", f"box gap <= {buffer_m:g} m to an evaluation sample")
    hit = pd.MultiIndex.from_frame(git[key]).isin(pd.MultiIndex.from_frame(agg[key]))
    retained = ids[assessable & ~hit].reset_index(drop=True)
    unassessable = ids[~assessable].drop(columns="google_location").reset_index(drop=True)
    unassessable["reason"] = unassessable["source"].map(SOURCE_REASON)
    if unassessable["reason"].isna().any():
        raise SystemExit(
            f"unassessable rows with source {unassessable.loc[unassessable['reason'].isna(), 'source'].unique()}"
        )
    assert len(retained) + len(excluded) + len(unassessable) == len(git)
    return retained, excluded, unassessable, matches


def control(ev: pd.DataFrame, buffer_m: float) -> dict:
    """Executed instrument gate (rule 13): for every assessable dataset plant one synthetic z=17
    tile on the centre of its first sample (must be excluded at distance 0), one at the
    antipode (must be retained), the sample's own box as a source-located row (distance 0),
    and that box shifted east to a gap of 0.99 x buffer (must hit, measured gap within 3 %)
    and 1.01 x buffer (must miss). Returns {"pass": bool, "detail": [...]}; the CLI stops on FAIL."""
    rows, want = [], []
    for k, (dataset, q) in enumerate(ev.groupby("dataset", sort=True).first().iterrows()):
        lat, lon = (q["lat_min"] + q["lat_max"]) / 2, (q["lon_min"] + q["lon_max"]) / 2
        for j, (la, lo) in enumerate([(lat, lon), (-lat, lon + 180 if lon < 0 else lon - 180)]):
            n = 2**17  # rows per dataset: on-sample tile, antipode tile, 3 source-box rows
            x = min(int((lo + 180) / 360 * n), n - 1)
            y = min(
                int(
                    (1 - math.log(math.tan(math.radians(la)) + 1 / math.cos(math.radians(la))) / math.pi)
                    / 2
                    * n
                ),
                n - 1,
            )
            b = tile_bounds(17, x, y)
            rows.append(
                dict(
                    shard=-1,
                    row=5 * k + j,
                    img_name=f"control-{dataset}-{j}",
                    google_location=f"17_{x}_{y}",
                    z=17,
                    x=x,
                    y=y,
                    lon_min=b[0],
                    lat_min=b[1],
                    lon_max=b[2],
                    lat_max=b[3],
                )
            )
            want.append((dataset, j == 0))
        # the same sample as a non-tile row located through source boxes (no z/x/y): must hit;
        # then the same box pushed east by 0.99 x buffer (must hit, at ~0.99 buffer) and by
        # 1.01 x buffer (must miss): a wrong buffer unit or a gap error > 1 % fails here
        width = q["lon_max"] - q["lon_min"]
        # the gap of lon-separated boxes is measured at the POLEWARD shared latitude (box_gap_m),
        # so the shift is converted at that latitude: at lat_max for a southern box the 1.01 row
        # of a 50 km city box would read 1004.7 m, and a taller one < 1000 m
        lat_pole = max(abs(q["lat_min"]), abs(q["lat_max"]))
        for tag, factor, expect in (
            ("box", 0.0, True),
            ("inside-buffer", 0.99, True),
            ("outside-buffer", 1.01, False),
        ):
            shift = width + math.degrees(factor * buffer_m / EARTH_R) / math.cos(math.radians(lat_pole))
            rows.append(
                dict(
                    shard=-1,
                    row=len(rows),
                    img_name=f"control-{dataset}-{tag}",
                    google_location=None,
                    z=None,
                    x=None,
                    y=None,
                    lon_min=q["lon_min"] + shift,
                    lat_min=q["lat_min"],
                    lon_max=q["lon_max"] + shift,
                    lat_max=q["lat_max"],
                )
            )
            want.append((dataset, expect, factor, q["sample_name"]))
    m = nearby(pd.DataFrame(rows), ev, buffer_m)
    got = (
        m.groupby(["row", "dataset"])["distance_m"].min().to_dict()
    )  # min: neighbours of the sample also match
    # the shifted boxes are scored against the PLANTED sample only: real footprints have
    # neighbours (adjacent BigEarthNet / EuroSAT patches at 0-300 m) that also fall inside the
    # buffer, so "min over the dataset" / "no match at all" failed on the real points file
    own = m.set_index(["row", "dataset", "sample_name"])["distance_m"].to_dict()
    detail, ok = [], True
    kinds = (
        "on-sample tile",
        "antipode tile",
        "on-sample source-box row",
        "box at 0.99 buffer",
        "box at 1.01 buffer",
    )
    for r, w in enumerate(want):
        dataset, expect_hit = w[0], w[1]
        d = got.get((r, dataset))
        if len(w) > 2 and w[2] > 0:  # shifted box rows: the gap to the planted sample itself
            d_own = own.get((r, dataset, w[3]))
            if expect_hit:  # inside-buffer row: hit, and the gap must read ~0.99 buffer
                passed = d_own is not None and 0.97 * buffer_m <= d_own <= buffer_m
            else:  # outside-buffer row: no match to the planted sample (neighbours may match)
                passed = d_own is None
        elif expect_hit:
            passed = d == 0.0
        else:
            passed = all(rr != r for rr, _ in got)
        ok &= passed
        detail.append(f"{'PASS' if passed else 'FAIL'} control: {dataset} {kinds[r % 5]}")
    return {"pass": bool(ok), "detail": detail}


def report(git, ev, ev_report, retained, excluded, unassessable, matches, buffer_m, missing_shards) -> dict:
    if "source" not in git:
        git = apply_source_boxes(git, None)
    per = {}
    for task, info in ev_report.items():
        m = matches[matches["dataset"] == task]
        per[task] = dict(
            info,
            git10m_excluded=int(m[["shard", "row"]].drop_duplicates().shape[0]),
            eval_samples_hit=int(m["sample_name"].nunique()) if info["assessable"] else None,
        )
    return {
        "buffer_m": buffer_m,
        "rule": "great-circle gap between the tile's lon/lat box and the evaluation sample's box <= buffer_m",
        "complete": not missing_shards,
        "missing_shards": missing_shards,
        "shards": int(git["shard"].nunique()),
        "git10m_rows": int(len(git)),
        "retained": int(len(retained)),
        "excluded": int(len(excluded)),
        "unassessable": int(len(unassessable)),
        "unassessable_rows_with_repeated_img_name": int(
            unassessable["img_name"].duplicated(keep=False).sum()
        ),
        "per_source": {
            src: {
                "rows": int((git["source"] == src).sum()),
                "retained": int((retained["source"] == src).sum()),
                "excluded": int((excluded["source"] == src).sum()),
                "unassessable": int((unassessable["source"] == src).sum()),
            }
            for src in sorted(git["source"].unique())
        },
        "eval_samples": int(len(ev)),
        "per_dataset": per,
    }


# -------------------------------------------------------------------------------- CLI
def _shard_range(s: str) -> range:
    a, _, b = s.partition("-")
    return range(int(a), int(b or a) + 1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("git10m-meta", help="per-shard metadata parquet (img_name, tile, lon/lat bounds)")
    g.add_argument("--shard-dir", required=True)
    g.add_argument("--shards", required=True, type=_shard_range, help="inclusive range, e.g. 0-49")
    g.add_argument("--out", required=True)
    g.add_argument(
        "--hf-download", action="store_true", help="fetch missing shards from the Hub into --shard-dir/train"
    )
    g.add_argument("--rm-shard", action="store_true", help="delete each shard after its parquet is written")
    t = sub.add_parser(
        "git10m-text", help="per-shard captions of the rows whose img_name starts with --prefix"
    )
    t.add_argument("--shard-dir", required=True)
    t.add_argument("--shards", required=True, type=_shard_range)
    t.add_argument("--out", required=True)
    t.add_argument("--prefix", required=True)
    t.add_argument("--hf-download", action="store_true")
    t.add_argument("--rm-shard", action="store_true")
    e = sub.add_parser("eval-points", help="GEO-Bench evaluation boxes (all six tasks) -> csv + report")
    e.add_argument(
        "--geobench-root", required=True, help="dir holding <task>_meta/default_partition.json [+ *.hdf5]"
    )
    e.add_argument("--raw-root", default=None, help="dir holding <task>/data.zip (Zenodo) for the footprints")
    e.add_argument(
        "--originals-root",
        default=None,
        help="dir with <task originals> (eurosat/EuroSAT_MS, bigearthnet/patch_index.parquet): pixel-join the id_N tasks",
    )
    e.add_argument("--out", required=True)
    cp = sub.add_parser(
        "city-points",
        help="points csv + report with m-so2sat covered at city level (So2Sat LCZ42 ROIs, ~50 x 50 km)",
    )
    cp.add_argument("--points", required=True, help="eval-points output (m_so2sat unassessable there)")
    cp.add_argument(
        "--cities",
        required=True,
        help="csv: group,city,lat,lon (experiments/isaac/geo_dedup/so2sat_cities.csv)",
    )
    cp.add_argument("--half-extent-m", type=float, default=SO2SAT_CITY_HALF_EXTENT_M)
    cp.add_argument("--task", default="m_so2sat")
    cp.add_argument("--out", required=True)
    cp.add_argument(
        "--check-existing",
        action="store_true",
        help="--out exists: verify it was built from --cities at --half-extent-m (exit 1 if not); writes nothing",
    )
    b = sub.add_parser(
        "bigearthnet-index",
        help="stream BigEarthNet-S2-v1.0.tar.gz -> patch_index.parquet (hash + footprint)",
    )
    b.add_argument("--tar", required=True)
    b.add_argument("--out", required=True)
    sb = sub.add_parser(
        "source-boxes", help="boxes for the locatable public-source rows (SSL4EO-S12, SkyScript) -> parquet"
    )
    sb.add_argument("--meta", required=True)
    sb.add_argument("--expect-shards", type=int, required=True)
    sb.add_argument("--allow-partial", action="store_true")
    sb.add_argument("--ssl4eo-centres", default=None, help="the authors' centre file (id,lon,lat)")
    sb.add_argument("--ssl4eo-void", default=None, help="the authors' data/void_ids.csv")
    sb.add_argument(
        "--skyscript-meta",
        default=None,
        help="the authors' meta<k>.zip (or a dir holding meta<k>/<stem>.pickle)",
    )
    sb.add_argument("--out", required=True)
    v = sub.add_parser(
        "ssl4eo-verify", help="gate: Git-10M SSL4EO captions vs the authors' labels of the mapped location"
    )
    v.add_argument(
        "--text", required=True, help="dir of text-NNNNN.parquet (git10m-text --prefix SSL4EO-S12)"
    )
    v.add_argument("--multilabel", required=True, help="the authors' ssl4eo_multilabel.json")
    v.add_argument("--ssl4eo-centres", required=True)
    v.add_argument("--ssl4eo-void", required=True)
    v.add_argument("--out", required=True, help="json report")
    d = sub.add_parser("dedup", help="retained / excluded / unassessable manifests + report.json")
    d.add_argument("--meta", required=True)
    d.add_argument("--expect-shards", type=int, required=True)
    d.add_argument(
        "--allow-partial", action="store_true", help="run on a subset of shards (report.complete=false)"
    )
    d.add_argument("--points", required=True)
    d.add_argument("--buffer-m", type=float, required=True)
    d.add_argument(
        "--source-boxes", default=None, help="source-boxes parquet: locates the SSL4EO-S12 / SkyScript rows"
    )
    d.add_argument("--out", required=True)
    d.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    a = p.parse_args(argv)

    if a.cmd == "git10m-meta":
        git10m_meta(a.shard_dir, a.shards, a.out, a.hf_download, a.rm_shard)
    elif a.cmd == "git10m-text":
        git10m_text(a.shard_dir, a.shards, a.out, a.prefix, a.hf_download, a.rm_shard)
    elif a.cmd == "eval-points":
        from eval.features import OFFICIAL

        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: refusing to overwrite")
        pts, rep = collect_eval_boxes(a.geobench_root, a.raw_root, list(OFFICIAL), a.originals_root)
        pts.to_csv(a.out, index=False)
        with open(a.out + ".report.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
    elif a.cmd == "city-points" and a.check_existing:
        ent = check_city_points(a.out, a.cities, a.half_extent_m, a.task)
        print(
            f"CITY_POINTS_MATCH {a.out} cities_sha256={ent['cities_sha256']} half_extent_m={ent['half_extent_m']:g}"
        )
    elif a.cmd == "city-points":
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: refusing to overwrite")
        pts, rep = city_points(a.points, a.cities, a.half_extent_m, a.task)
        pts.to_csv(a.out, index=False)
        with open(a.out + ".report.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep[a.task], indent=1))
        print(f"CITY_POINTS_OK {len(pts)} boxes -> {a.out}")
    elif a.cmd == "bigearthnet-index":
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: refusing to overwrite")
        df = bigearthnet_index(a.tar)
        df.to_parquet(a.out + ".tmp", index=False)
        os.replace(a.out + ".tmp", a.out)
        print(f"BIGEARTHNET_INDEX_OK {len(df)} patches -> {a.out}")
    elif a.cmd == "source-boxes":
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: refusing to overwrite")
        if (a.ssl4eo_centres is None) != (a.ssl4eo_void is None):
            raise SystemExit("pass both --ssl4eo-centres and --ssl4eo-void")
        git, missing = load_meta(a.meta, a.expect_shards, a.allow_partial)
        centres = ssl4eo_centres(a.ssl4eo_centres, a.ssl4eo_void) if a.ssl4eo_centres else None
        boxes, rep = source_boxes(git, centres, a.skyscript_meta)
        rep = {"meta": a.meta, "missing_shards": missing, "located_names": int(len(boxes)), "per_source": rep}
        boxes.to_parquet(a.out + ".tmp", index=False)
        os.replace(a.out + ".tmp", a.out)
        with open(a.out + ".report.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
    elif a.cmd == "ssl4eo-verify":
        files = sorted(f for f in os.listdir(a.text) if re.fullmatch(r"text-\d{5}\.parquet", f))
        if not files:
            raise SystemExit(f"{a.text}: no text-NNNNN.parquet")
        texts = pd.concat([pd.read_parquet(os.path.join(a.text, f)) for f in files], ignore_index=True)
        with open(a.multilabel) as f:
            ml = json.load(f)
        rep = ssl4eo_verify(texts, ml, ssl4eo_centres(a.ssl4eo_centres, a.ssl4eo_void))
        print("\n".join(rep["detail"]))
        print("SSL4EO MAPPING:", "PASS" if rep["pass"] else "FAIL")
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        if not rep["pass"]:
            raise SystemExit("ssl4eo-verify FAILED: do not use the SSL4EO boxes")
    else:
        if os.path.exists(a.out) and not a.dry_run:
            raise SystemExit(f"{a.out} exists: refusing to overwrite")
        git, missing = load_meta(a.meta, a.expect_shards, a.allow_partial)
        boxes = pd.read_parquet(a.source_boxes) if a.source_boxes else None
        git = apply_source_boxes(git, boxes)
        pts = pd.read_csv(a.points)
        with open(a.points + ".report.json") as f:
            pts_report = json.load(f)
        if set(pts["dataset"]) != {k for k, v in pts_report.items() if v["assessable"]}:
            raise SystemExit("points csv and its report disagree on the assessable datasets")
        gate = control(pts, a.buffer_m)
        print("\n".join(gate["detail"]), flush=True)
        if not gate["pass"]:
            raise SystemExit(
                "instrument control FAILED: diagnose nearby()/box_gap_m before trusting any count"
            )
        parts = dedup(git, pts, a.buffer_m)
        rep = report(git, pts, pts_report, *parts, a.buffer_m, missing)
        rep["control"] = gate
        rep["source_boxes"] = a.source_boxes
        print(json.dumps(rep, indent=1))
        if a.dry_run:
            return
        os.makedirs(a.out)
        for name, df in zip(("retained", "excluded", "unassessable", "matches"), parts):
            df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
