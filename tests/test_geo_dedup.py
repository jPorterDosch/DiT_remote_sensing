"""Geographic exclusion (eval/geo_dedup.py) on synthetic shards and points: tile geometry,
identifier/coordinate validation, the distance rule across the antimeridian, accounting,
determinism and the refuse-to-overwrite / refuse-partial guards (each fed the mismatch it
exists to catch, rule 6). No downloads."""

from __future__ import annotations

import io
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from eval import geo_dedup as G


# ------------------------------------------------------------------ tile geometry
def test_tile_bounds_matches_dataset_card_formula():
    def xyz_to_lonlat(x, y, z):  # verbatim from the Git-10M dataset card
        n = 2**z * 1.0
        return x / n * 360.0 - 180.0, math.degrees(math.atan(math.sinh(math.pi * (1 - 2.0 * y / n))))

    lon_min, lat_min, lon_max, lat_max = G.tile_bounds(10, 0, 218)
    assert (lon_min, lat_max) == xyz_to_lonlat(0, 218, 10)
    assert (lon_max, lat_min) == xyz_to_lonlat(1, 219, 10)
    assert G.tile_bounds(0, 0, 0) == pytest.approx((-180, -G.MAX_LAT, 180, G.MAX_LAT))
    assert lat_min < lat_max and lon_min < lon_max


@pytest.mark.parametrize("zxy", [(10, 1024, 0), (10, 0, 1024), (-1, 0, 0), (31, 0, 0)])
def test_tile_bounds_rejects_out_of_range(zxy):
    with pytest.raises(ValueError):
        G.tile_bounds(*zxy)


def test_parse_google_location():
    assert G.parse_google_location("10_0_218", "Google-10-10_0_218.jpg") == (10, 0, 218)
    assert G.parse_google_location(None, "VGoogle-tennis_court-0.jpg") is None
    for loc, name in [
        ("10_0", "Google-10-10_0.jpg"),  # malformed
        ("10_0_219", "Google-10-10_0_218.jpg"),  # name/location disagree
        ("10_0_1024", "Google-10-10_0_1024.jpg"),  # out of range
        (None, "Google-10-10_0_218.jpg"),  # tile row without a location
    ]:
        with pytest.raises(ValueError):
            G.parse_google_location(loc, name)


# ------------------------------------------------------------------- shard reading
def _write_shard(path, names, locs):
    schema = pa.schema(
        [
            ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("text", pa.string()),
            ("img_name", pa.string()),
            ("img_quality_score", pa.float64()),
            ("caption", pa.string()),
            ("caption_length", pa.int64()),
            ("resolution", pa.string()),
            ("Google_location", pa.string()),
        ]
    )
    n = len(names)
    batch = pa.record_batch(
        [
            pa.array([{"bytes": b"\x89PNG-not-decoded", "path": None}] * n, type=schema.field("image").type),
            pa.array(["t"] * n),
            pa.array(names),
            pa.array([3.0] * n),
            pa.array(["c"] * n),
            pa.array([1] * n),
            pa.array([None if lo is None else "128m/pix" for lo in locs]),
            pa.array(locs),
        ],
        schema=schema,
    )
    with pa.OSFile(path, "wb") as f, ipc.new_stream(f, schema) as w:
        w.write_batch(batch)


def test_shard_meta_reads_tiles_and_flags_unassessable(tmp_path):
    p = str(tmp_path / "s.arrow")
    _write_shard(p, ["Google-10-10_0_218.jpg", "VGoogle-tennis_court-0.jpg"], ["10_0_218", None])
    df = G.shard_meta(p, 3)
    assert df["shard"].tolist() == [3, 3] and df["row"].tolist() == [0, 1]
    assert df["z"].tolist()[0] == 10 and pd.isna(df["z"].iloc[1]) and np.isnan(df["lon_min"].iloc[1])
    assert tuple(df.loc[0, ["lon_min", "lat_min", "lon_max", "lat_max"]]) == G.tile_bounds(10, 0, 218)


# ---------------------------------------------------------------- GEO-Bench names
def test_eval_name_parsers():
    assert G.parse_eval_name("m_forestnet", "2.14547959681366_117.26891152087542_2015_01_01") == (
        2.14547959681366,
        117.26891152087542,
    )
    assert G.parse_eval_name("m_pv4ger", "51.52163602611816,7.087774713501749") == (
        51.52163602611816,
        7.087774713501749,
    )
    assert G.parse_eval_name("m_pv4ger", "6.335539,51.746694") == (
        51.746694,
        6.335539,
    )  # lon,lat written order
    assert G.parse_eval_name("m_pv4ger", "6.544908108305756, 50.55091602295795") == (
        50.55091602295795,
        6.544908108305756,
    )
    for task, name in [
        ("m_forestnet", "id_0001"),
        ("m_forestnet", "95.0_117.0_2015_01_01"),  # lat out of range
        ("m_pv4ger", "10.0,10.0"),  # neither order inside Germany: ambiguous
        ("m_pv4ger", "51.5;7.0"),
        ("m_unknown", "x"),
    ]:
        with pytest.raises(ValueError):
            G.parse_eval_name(task, name)
    for task in ("m_eurosat", "m_so2sat", "m_bigearthnet"):
        with pytest.raises(G.Unassessable):
            G.parse_eval_name(task, "id_0000")
    with pytest.raises(ValueError, match="no name parser"):
        G.parse_eval_name("m_brick_kiln", "examples_40_141")  # footprint task run without raw data


def test_eval_boxes_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        G.eval_boxes("m_pv4ger", {"train": ["51.5,7.0"], "test": ["51.5,7.0"]})


def test_eval_boxes_centre_nominal():
    df = G.eval_boxes("m_pv4ger", {"train": ["51.5,7.0"]})
    assert df["source"].tolist() == ["centre+nominal"]
    assert G.haversine_m(df.lat_min[0], 7.0, df.lat_max[0], 7.0) == pytest.approx(2 * 12.8, rel=1e-6)
    assert G.haversine_m(51.5, df.lon_min[0], 51.5, df.lon_max[0]) == pytest.approx(2 * 12.8, rel=1e-3)


# ------------------------------------------------------------------ hdf5 footprints
def _hdf5_blob(bands, transform, crs, shape=(4, 4)):
    """A GEO-Bench-style hdf5 (attrs['pickle'] = str(pickle.dumps(...))) in memory. transform is
    pickled as an old affine<3 namedtuple (module affine, NEWOBJ with 9 args) when given."""
    import collections
    import pickle

    import affine
    import h5py

    Affine = collections.namedtuple("Affine", "a b c d e f g h i")
    Affine.__module__ = "affine"
    attrs = {"label": 0, "bands_order": bands}
    for b in bands:
        attrs[b] = {
            "date": None,
            "date_id": None,
            "spatial_resolution": 10,
            "band_info": None,
            "meta_info": None,
        }
        if transform is not None:
            attrs[b]["transform"], attrs[b]["crs"] = Affine(*transform, 0.0, 0.0, 1.0), crs
    real = affine.Affine
    affine.Affine = Affine  # pickle resolves the global by identity; restored below
    try:
        blob = pickle.dumps(attrs)
    finally:
        affine.Affine = real
    bio = io.BytesIO()
    with h5py.File(bio, "w") as h:
        for b in bands:
            h.create_dataset(b, data=np.zeros(shape, np.uint16))
        h.attrs["pickle"] = str(blob)
    return bio.getvalue()


def test_hdf5_footprint_epsg4326_and_utm():
    # 4x4 px at 0.001 deg, upper-left (10, 50): box (10, 49.996, 10.004, 50)
    box = G.hdf5_footprint(_hdf5_blob(["b1", "b2"], (0.001, 0, 10.0, 0, -0.001, 50.0), "EPSG:4326"), "s")
    assert box == pytest.approx((10.0, 49.996, 10.004, 50.0))
    # UTM 32N, 10 m px, 4x4 at (500000, 5500000): reprojected box ~40 m wide near (9E, 49.66N)
    box = G.hdf5_footprint(_hdf5_blob(["b1"], (10.0, 0, 500000.0, 0, -10.0, 5500000.0), "EPSG:32632"), "s")
    assert abs(box[0] - 9.0) < 0.01 and 49.6 < box[1] < 49.7
    assert G.haversine_m(box[1], box[0], box[1], box[2]) == pytest.approx(40, abs=1)
    assert G.hdf5_footprint(_hdf5_blob(["b1"], None, None), "s") is None  # keys absent: no georef


def test_hdf5_footprint_rejects_partial_or_none_georef():
    blob = _hdf5_blob(["b1"], (0.001, 0, 10.0, 0, -0.001, 50.0), None)
    with pytest.raises(ValueError, match="None"):
        G.hdf5_footprint(blob, "s")


def test_eval_boxes_footprint_requires_every_sample():
    geo = _hdf5_blob(["b1"], (0.001, 0, 10.0, 0, -0.001, 50.0), "EPSG:4326")
    none = _hdf5_blob(["b1"], None, None)
    df = G.eval_boxes("m_brick_kiln", {"train": ["a"], "test": ["b"]}, {"a": geo, "b": geo}.get)
    assert df["source"].tolist() == ["footprint"] * 2 and df["lon_max"].tolist() == pytest.approx(
        [10.004] * 2
    )
    with pytest.raises(ValueError, match="no georeferencing"):
        G.eval_boxes("m_brick_kiln", {"train": ["a"], "test": ["b"]}, {"a": geo, "b": none}.get)


def test_half_extent_matches_shipped_specs():
    """EVAL_HALF_EXTENT_M = export img_hw x coarsest task_specs resolution / 2 (when shipped)."""
    from eval import export_geobench as E

    for key, half in G.EVAL_HALF_EXTENT_M.items():
        px = E.SPECS[key.replace("_", "-")].img_hw[0]
        d = os.path.join(
            os.environ.get("GEOBENCH_META_ROOT", "data"), f"{key}_meta"
        )  # ISAAC: $STORE/datasets
        if not os.path.exists(os.path.join(d, "task_specs.pkl")):
            pytest.skip(f"{d} not present")
        import geobench

        res = max(b.spatial_resolution for b in geobench.load_task_specs(d).bands_info)
        assert half == px * res / 2


# -------------------------------------------------------------------- distance rule
def _git(rows):
    """rows: (shard, row, z, x, y | None)."""
    out = []
    for s, r, t in rows:
        b = G.tile_bounds(*t) if t else (np.nan,) * 4
        out.append(
            dict(
                shard=s,
                row=r,
                img_name=f"n{s}_{r}",
                google_location="_".join(map(str, t)) if t else None,
                z=t[0] if t else None,
                x=t[1] if t else None,
                y=t[2] if t else None,
                lon_min=b[0],
                lat_min=b[1],
                lon_max=b[2],
                lat_max=b[3],
            )
        )
    df = pd.DataFrame(out)
    for c in ("z", "x", "y"):
        df[c] = df[c].astype("Int32")
    return df


def _ev(rows):
    """rows: (dataset, split, name, lat, lon, half_m) -> centre+nominal boxes."""
    out = [(d, sp, n, "centre+nominal", *G.centre_box(lat, lon, h)) for d, sp, n, lat, lon, h in rows]
    return pd.DataFrame(out, columns=["dataset", "split", "sample_name", "source", *G.BOX])


def _box(lon_min, lat_min, lon_max, lat_max):
    return dict(zip(G.BOX, (lon_min, lat_min, lon_max, lat_max)))


def _arr(**boxes):
    return {c: np.array([b[c] for b in boxes.values()]) for c in G.BOX}


def test_box_gap_m():
    a = _box(10.0, 50.0, 10.1, 50.1)
    gaps = G.box_gap_m(
        a,
        _arr(
            overlap=_box(10.05, 50.05, 10.2, 50.2),
            inside=_box(10.02, 50.02, 10.03, 50.03),
            north=_box(10.0, 50.2, 10.1, 50.3),
            east=_box(10.2, 50.0, 10.3, 50.1),
            diag=_box(10.2, 50.2, 10.3, 50.3),
        ),
    )
    assert gaps[0] == 0 and gaps[1] == 0
    assert gaps[2] == pytest.approx(G.haversine_m(50.1, 10.0, 50.2, 10.0))  # facing edges, shared lon
    assert gaps[3] == pytest.approx(
        G.haversine_m(50.1, 10.1, 50.1, 10.2)
    )  # shared lat farthest from the equator
    assert gaps[4] == pytest.approx(G.haversine_m(50.1, 10.1, 50.2, 10.2))  # corner to corner
    wrap = G.box_gap_m(_box(179.9, 0.0, 180.0, 0.1), _arr(w=_box(-180.0, 0.0, -179.9, 0.1)))
    assert wrap[0] == 0


def test_box_gap_m_lon_separated_is_the_true_minimum():
    """Meridians converge poleward: for boxes side by side in longitude the nearest pair sits at
    the poleward end of the shared latitude range. Checked against a brute-force scan on both
    hemispheres and for a city-sized box, where the equator-nearest choice was ~1 % too large."""
    for lat_lo, lat_hi in ((50.0, 50.45), (-50.45, -50.0)):
        a = _box(10.0, lat_lo, 10.5, lat_hi)
        b = _box(10.52, lat_lo, 11.0, lat_hi)
        code = float(G.box_gap_m(a, _arr(b=b))[0])
        lats = np.linspace(a["lat_min"], a["lat_max"], 4001)
        brute = G.haversine_m(lats, a["lon_max"], lats, b["lon_min"]).min()
        assert code == pytest.approx(brute, rel=1e-6)
        equator_nearest = lat_lo if abs(lat_lo) < abs(lat_hi) else lat_hi
        assert code < G.haversine_m(equator_nearest, a["lon_max"], equator_nearest, b["lon_min"]) * 0.995


def test_nearby_inside_outside_and_buffer():
    lon_min, lat_min, lon_max, lat_max = G.tile_bounds(10, 0, 218)  # ~40 km tile near 71N, 180W
    inside = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    git = _git([(0, 0, (10, 0, 218)), (0, 1, (10, 512, 512))])  # second tile: equator, Greenwich
    far_east = (inside[0], lon_max + 0.5)  # 0.5 deg east of the box edge, ~18 km at 71N
    ev = _ev([("m_pv4ger", "train", "in", *inside, 12.8), ("m_pv4ger", "test", "east", *far_east, 12.8)])
    m = G.nearby(git, ev, buffer_m=1000.0)
    assert (
        m[["shard", "row", "sample_name"]].values.tolist() == [[0, 0, "in"]] and m["distance_m"].iloc[0] == 0
    )
    gap = float(
        G.box_gap_m(_box(lon_min, lat_min, lon_max, lat_max), {c: ev[c].to_numpy()[1:] for c in G.BOX})[0]
    )
    assert 15_000 < gap < 20_000  # 0.5 deg of longitude at ~71N
    assert G.nearby(git, ev, buffer_m=gap + 1).shape[0] == 2  # buffer now reaches 'east'
    assert G.nearby(git, ev, buffer_m=gap - 1).shape[0] == 1


def test_nearby_across_antimeridian():
    git = _git([(0, 0, (10, 1023, 512))])  # tile touching lon=180 just south of the equator
    ev = _ev([("m_forestnet", "train", "w", -0.1, -179.999, 3840.0)])  # just west of -180 == east of the tile
    m = G.nearby(git, ev, buffer_m=500.0)
    assert len(m) == 1 and m["distance_m"].iloc[0] == 0  # the nominal box crosses the antimeridian


def test_dedup_accounting_and_report():
    git = _git([(0, 0, (10, 0, 218)), (0, 1, (10, 512, 512)), (1, 0, None)])
    lon_min, lat_min, lon_max, lat_max = G.tile_bounds(10, 0, 218)
    ev = _ev(
        [
            ("m_pv4ger", "train", "a", (lat_min + lat_max) / 2, (lon_min + lon_max) / 2, 12.8),
            ("m_forestnet", "test", "b", lat_min, lon_min, 3840.0),
        ]
    )
    retained, excluded, unassessable, matches = G.dedup(git, ev, 100.0)
    assert len(retained) == 1 and retained["row"].tolist() == [1]
    assert excluded[["shard", "row", "datasets", "matched_ids", "n_matches"]].values.tolist() == [
        [0, 0, "m_forestnet;m_pv4ger", "b;a", 2]
    ]
    assert unassessable["img_name"].tolist() == ["n1_0"] and "reason" in unassessable
    rep = G.report(
        git,
        ev,
        {
            "m_pv4ger": {"assessable": True},
            "m_forestnet": {"assessable": True},
            "m_so2sat": {"assessable": False, "reason": "x"},
        },
        retained,
        excluded,
        unassessable,
        matches,
        100.0,
        [],
    )
    assert rep["retained"] + rep["excluded"] + rep["unassessable"] == rep["git10m_rows"] == 3
    assert (
        rep["per_dataset"]["m_pv4ger"]["git10m_excluded"] == 1
        and rep["per_dataset"]["m_so2sat"]["git10m_excluded"] == 0
    )
    assert rep["complete"] is True


# --------------------------------------------------------------- end-to-end + guards
def _meta_dir(tmp_path, shards):
    d = tmp_path / "meta"
    d.mkdir()
    for s, names, locs in shards:
        p = str(tmp_path / f"s{s}.arrow")
        _write_shard(p, names, locs)
        G.shard_meta(p, s).to_parquet(d / f"meta-{s:05d}.parquet", index=False)
    return str(d)


def _points_csv(tmp_path):
    lon_min, lat_min, lon_max, lat_max = G.tile_bounds(10, 0, 218)
    pts = _ev([("m_pv4ger", "train", "a", (lat_min + lat_max) / 2, (lon_min + lon_max) / 2, 12.8)])
    p = tmp_path / "points.csv"
    pts.to_csv(p, index=False)
    (tmp_path / "points.csv.report.json").write_text(
        json.dumps({"m_pv4ger": {"assessable": True, "n_samples": 1}})
    )
    return str(p)


def test_cli_dedup_deterministic_dry_run_and_guards(tmp_path, capsys):
    meta = _meta_dir(
        tmp_path,
        [
            (0, ["Google-10-10_0_218.jpg", "VGoogle-x-0.jpg"], ["10_0_218", None]),
            (1, ["Google-10-10_512_512.jpg"], ["10_512_512"]),
        ],
    )
    pts = _points_csv(tmp_path)
    base = ["dedup", "--meta", meta, "--expect-shards", "2", "--points", pts, "--buffer-m", "100"]
    G.main(base + ["--out", str(tmp_path / "dry"), "--dry-run"])
    assert not (tmp_path / "dry").exists() and '"excluded": 1' in capsys.readouterr().out
    G.main(base + ["--out", str(tmp_path / "o1")])
    G.main(base + ["--out", str(tmp_path / "o2")])
    for f in ("retained.csv", "excluded.csv", "unassessable.csv", "matches.csv", "report.json"):
        assert (tmp_path / "o1" / f).read_bytes() == (tmp_path / "o2" / f).read_bytes()
    rep = json.loads((tmp_path / "o1" / "report.json").read_text())
    assert (rep["retained"], rep["excluded"], rep["unassessable"], rep["git10m_rows"]) == (1, 1, 1, 3)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        G.main(base + ["--out", str(tmp_path / "o1")])
    with pytest.raises(SystemExit, match="shards"):  # 2 of 3 expected shards
        G.main(base[:4] + ["3"] + base[5:] + ["--out", str(tmp_path / "o3")])
    G.main(base[:4] + ["3"] + base[5:] + ["--allow-partial", "--out", str(tmp_path / "o3")])
    assert json.loads((tmp_path / "o3" / "report.json").read_text())["complete"] is False


def test_git10m_meta_skips_complete_parquet(tmp_path, capsys):
    p = str(tmp_path / "train" / "data-00003-of-00787.arrow")
    os.makedirs(os.path.dirname(p))
    _write_shard(p, ["Google-10-10_0_218.jpg"], ["10_0_218"])
    G.git10m_meta(str(tmp_path), range(3, 4), str(tmp_path / "meta"), False, False)
    first = (tmp_path / "meta" / "meta-00003.parquet").read_bytes()
    G.git10m_meta(str(tmp_path), range(3, 4), str(tmp_path / "meta"), False, False)
    assert (
        "skipping" in capsys.readouterr().out
        and (tmp_path / "meta" / "meta-00003.parquet").read_bytes() == first
    )


def test_control_gate_passes_and_can_fail(monkeypatch):
    east = math.degrees(80.0 / G.EARTH_R) / math.cos(math.radians(51.5))  # 80 m east of a's centre
    ev = _ev(
        [
            ("m_pv4ger", "train", "a", 51.5, 7.0, 12.8),
            ("m_pv4ger", "train", "b", 51.5005, 7.0, 12.8),
            ("m_pv4ger", "train", "n", 51.5, 7.0 + east, 12.8),
            ("m_forestnet", "test", "c", -2.0, 115.0, 3840.0),
            ("m_so2sat", "test", "city:south", -34.0, 151.0, 47_000.0),
        ]
    )  # b: a 55 m neighbour also matches the planted tile; n: a neighbour 19-21 m from the
    # shifted boxes (real footprints have adjacent patches), which must not fail the 0.99 / 1.01
    # rows (they are scored against sample a); city:south: a 94 km southern box, where a shift
    # converted at lat_max instead of the poleward latitude reads the 1.01 row under the buffer
    gate = G.control(ev, 100.0)
    assert gate["pass"] and len(gate["detail"]) == 15 and all(d.startswith("PASS") for d in gate["detail"])
    assert gate == G.control(ev, 100.0)  # deterministic
    real = G.box_gap_m
    monkeypatch.setattr(G, "box_gap_m", lambda a, b: np.full(np.shape(b["lat_min"]), 1e6))  # dead instrument
    assert G.control(ev, 100.0)["pass"] is False
    monkeypatch.setattr(
        G, "box_gap_m", lambda a, b: real(a, b) / 1000.0
    )  # gap read in km: 1.01-buffer row hits
    bad = G.control(ev, 100.0)
    assert not bad["pass"] and any(d.startswith("FAIL") and "1.01 buffer" in d for d in bad["detail"])
    monkeypatch.setattr(
        G, "box_gap_m", lambda a, b: real(a, b) * 1.05
    )  # 5 % overestimate: 0.99-buffer row misses
    bad = G.control(ev, 100.0)
    assert not bad["pass"] and any(d.startswith("FAIL") and "0.99 buffer" in d for d in bad["detail"])


def test_nearby_and_dedup_handle_empty_eval_frame():
    git = _git([(0, 0, (10, 0, 218))])
    ev = _ev([("m_pv4ger", "train", "a", 51.5, 7.0, 12.8)]).iloc[:0]
    assert len(G.nearby(git, ev, 100.0)) == 0
    retained, excluded, unassessable, matches = G.dedup(git, ev, 100.0)
    assert len(retained) == 1 and len(excluded) == 0 and len(matches) == 0


def test_load_meta_rejects_duplicate_img_names(tmp_path):
    meta = _meta_dir(
        tmp_path,
        [(0, ["Google-10-10_0_218.jpg"], ["10_0_218"]), (1, ["Google-10-10_0_218.jpg"], ["10_0_218"])],
    )
    with pytest.raises(SystemExit, match="tile rows share an img_name"):
        G.load_meta(meta, 2, False)


def test_load_meta_keeps_repeated_non_tile_names(tmp_path):
    # Real Git-10M: one public-dataset image on adjacent rows, no Google_location.
    meta = _meta_dir(tmp_path, [(0, ["GeoPileV0-naip300k-class0-1.png"] * 3, [None] * 3)])
    git, missing = G.load_meta(meta, 1, False)
    assert len(git) == 3 and missing == []
    assert git[["shard", "row"]].drop_duplicates().shape[0] == 3


# ------------------------------------------------------- Git-10M public sources
def _centres(tmp_path, ids):
    rng = np.random.default_rng(0)
    c = pd.DataFrame(
        {"id": ids, "lon": rng.uniform(-170, 170, len(ids)), "lat": rng.uniform(-60, 60, len(ids))}
    )
    c.to_csv(tmp_path / "centres.csv", header=False, index=False)
    void = sorted(set(range(max(ids) + 1)) - set(ids))
    pd.Series(void).to_csv(tmp_path / "void.csv", header=False, index=False)
    return c.sort_values("id").reset_index(drop=True)


def test_source_of_prefixes():
    names = pd.Series(
        [
            "Google-10-10_0_218.jpg",
            "SSL4EO-S12-rgb-5.jpg",
            "Sky-images6-w1_DE_.jpg",
            "Million-AID-x.jpg",
            "zzz.jpg",
        ]
    )
    loc = pd.Series(["10_0_218", None, None, None, None])
    assert G.source_of(names, loc).tolist() == [
        "google_tile",
        "ssl4eo",
        "skyscript",
        "million_aid",
        "unknown",
    ]


def test_ssl4eo_boxes_follow_the_4i_plus_1_rule(tmp_path, monkeypatch):
    ids = [0, 1, 2, 4, 5]  # 3 is void
    monkeypatch.setattr(G, "SSL4EO_N_LOCATIONS", len(ids))
    c = (
        G.ssl4eo_centres(tmp_path / "centres.csv", tmp_path / "void.csv")
        if False
        else _centres(tmp_path, ids)
    )
    c2 = G.ssl4eo_centres(str(tmp_path / "centres.csv"), str(tmp_path / "void.csv"))
    assert c2["id"].tolist() == c["id"].tolist()
    names = pd.Series(
        [f"SSL4EO-S12-rgb-{4 * i + 1}.jpg" for i in (0, 3, 4)] + ["SSL4EO-S12-rgb-13.jpg"]
    )  # 13 repeats i=3
    b = G.ssl4eo_boxes(names, c2)
    assert b["source_id"].tolist() == ["0", "4", "5"]  # i=3 -> the 4th centre, id 4 (void 3 skipped)
    lat, lon = c2.loc[1, ["lat", "lon"]] if False else c2.loc[3, ["lat", "lon"]]
    exp = G.centre_box(lat, lon, G.SSL4EO_HALF_EXTENT_M)
    assert np.allclose(b.loc[1, G.BOX].to_numpy(dtype=float), exp)
    with pytest.raises(SystemExit, match="season"):
        G.ssl4eo_boxes(pd.Series(["SSL4EO-S12-rgb-1.jpg", "SSL4EO-S12-rgb-2.jpg"]), c2)
    with pytest.raises(SystemExit, match="one-to-one"):
        G.ssl4eo_boxes(pd.Series(["SSL4EO-S12-rgb-21.jpg"]), c2)  # i=5 >= 5 centres
    with pytest.raises(SystemExit, match="void"):
        pd.Series([3]).to_csv(tmp_path / "void2.csv", header=False, index=False)
        pd.concat([c2, pd.DataFrame({"id": [3], "lon": [0.0], "lat": [0.0]})]).to_csv(
            tmp_path / "c3.csv", header=False, index=False
        )
        monkeypatch.setattr(G, "SSL4EO_N_LOCATIONS", 6)
        G.ssl4eo_centres(str(tmp_path / "c3.csv"), str(tmp_path / "void2.csv"))


def _sky_pickle(path, bbox):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(
            {"bbox": tuple(np.float64(v) for v in bbox), "time": None, "center_tags": {}}, f, protocol=3
        )


def test_skyscript_unpickler_reads_bbox_and_refuses_code(tmp_path):
    import pickle

    (tmp_path / "meta6").mkdir()
    _sky_pickle(tmp_path / "meta6" / "w1_DE_.pickle", (13.6867, 52.4648, 13.6874, 52.4652))
    df, missing = G.skyscript_boxes(
        pd.Series(["Sky-images6-w1_DE_.jpg", "Sky-images6-w2_US_19.jpg"]), str(tmp_path)
    )
    assert df["img_name"].tolist() == ["Sky-images6-w1_DE_.jpg"] and missing == ["Sky-images6-w2_US_19.jpg"]
    assert np.allclose(df.loc[0, G.BOX].to_numpy(dtype=float), (13.6867, 52.4648, 13.6874, 52.4652))
    with open(tmp_path / "meta6" / "evil.pickle", "wb") as f:
        pickle.dump({"bbox": (0, 0, 0, 0), "x": __import__("os").system}, f)
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        G.skyscript_box(str(tmp_path / "meta6" / "evil.pickle"))
    _sky_pickle(tmp_path / "meta6" / "latlon.pickle", (13.68, 95.0, 13.69, 95.1))  # a latitude past 90
    with pytest.raises(ValueError, match="not lon_min"):
        G.skyscript_box(str(tmp_path / "meta6" / "latlon.pickle"))
    _sky_pickle(tmp_path / "meta6" / "huge.pickle", (13.0, 52.0, 16.0, 55.0))
    with pytest.raises(ValueError, match="spans more than"):
        G.skyscript_box(str(tmp_path / "meta6" / "huge.pickle"))
    import zipfile

    with zipfile.ZipFile(tmp_path / "meta6.zip", "w") as z:  # the authors' layout: meta6/<stem>.pickle
        z.write(tmp_path / "meta6" / "w1_DE_.pickle", "meta6/w1_DE_.pickle")
    dz, missing_z = G.skyscript_boxes(
        pd.Series(["Sky-images6-w1_DE_.jpg", "Sky-images6-w2_US_19.jpg"]), str(tmp_path / "meta6.zip")
    )
    assert dz.equals(df) and missing_z == missing


def test_apply_source_boxes_locates_rows_and_guards():
    git = _git([(0, 0, (10, 0, 218)), (0, 1, None), (0, 2, None)])
    git.loc[1, "img_name"] = "Sky-images6-w1_DE_.jpg"
    git.loc[2, "img_name"] = "Million-AID-x.jpg"
    boxes = pd.DataFrame(
        [("Sky-images6-w1_DE_.jpg", "skyscript", "w1_DE_", 7.0, 51.5, 7.001, 51.501)],
        columns=["img_name", "source", "source_id", *G.BOX],
    )
    out = G.apply_source_boxes(git, boxes)
    assert out["source"].tolist() == ["google_tile", "skyscript", "million_aid"]
    assert out.loc[1, "lon_min"] == 7.0 and np.isnan(out.loc[2, "lon_min"])
    ev = _ev([("m_pv4ger", "train", "a", 51.5005, 7.0005, 12.8)])
    retained, excluded, unassessable, _ = G.dedup(out, ev, 100.0)
    assert excluded["img_name"].tolist() == ["Sky-images6-w1_DE_.jpg"]
    assert unassessable["reason"].tolist() == ["Million-AID publishes no coordinates"]
    assert "source" in retained and retained["source"].tolist() == ["google_tile"]
    with pytest.raises(SystemExit, match="is a Google tile"):
        G.apply_source_boxes(git, boxes.assign(img_name=["n0_0"]))
    with pytest.raises(SystemExit, match="not in the metadata"):
        G.apply_source_boxes(git, boxes.assign(img_name=["Sky-images6-other.jpg"]))
    with pytest.raises(SystemExit, match="disagrees"):
        G.apply_source_boxes(git, boxes.assign(source=["ssl4eo"]))


def test_ssl4eo_verify_passes_true_mapping_and_fails_shuffled(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = 4000
    ids = np.array([i for i in range(n + 1) if i != 2000])  # one void id in the middle
    monkeypatch.setattr(G, "SSL4EO_N_LOCATIONS", len(ids))
    c = _centres(tmp_path, ids.tolist())
    cls = rng.integers(0, 9, n)
    word = {
        0: "a lake",
        1: "a forest",
        4: "farmland",
        6: "an urban area",
        7: "a desert",
        8: "snow",
        2: "grass",
        3: "marsh",
        5: "shrubs",
    }
    texts = pd.DataFrame(
        {
            "img_name": [f"SSL4EO-S12-rgb-{4 * i + 1}.jpg" for i in range(n)],
            "caption": [f"The image shows {word[k]}." for k in cls],
        }
    )
    ml = {f"{ids[i]:07d}": {"t0": [str(cls[i])], "t1": [str(cls[i])]} for i in range(n)}
    rep = G.ssl4eo_verify(texts, ml, c)
    assert rep["pass"] and rep["first_void_id"] == 2000 and all(d.startswith("PASS") for d in rep["detail"])
    # a mapping that ignored the void (folder = i) is wrong for every row past the void
    ml_shift = {f"{i:07d}": {"t0": [str(cls[i])]} for i in range(n)}
    ml_shift = {k: v for k, v in ml_shift.items() if int(k) != 2000}
    ml_shift["0004000"] = {"t0": ["0"]}
    rep2 = G.ssl4eo_verify(texts, ml_shift, c)
    assert not rep2["pass"] and sum(d.startswith("FAIL") and "below/above" in d for d in rep2["detail"]) == 6
    assert all(
        v["gain"] > 0.5 for k, v in rep2["stats"].items() if k.endswith("/lo")
    )  # the first half still agrees
    rep3 = G.ssl4eo_verify(texts.assign(caption=rng.permutation(texts["caption"].to_numpy())), ml, c)
    assert not rep3["pass"] and any("0/6 classes" in d for d in rep3["detail"])


# ------------------------------------------------ GEO-Bench originals (pixel join)
def _hdf5_pixels(bands: dict):
    import pickle

    import h5py

    attrs = {"label": 0, "bands_order": list(bands)}
    for b in bands:
        attrs[b] = {"date": None, "spatial_resolution": 10}
    bio = io.BytesIO()
    with h5py.File(bio, "w") as h:
        for b, a in bands.items():
            h.create_dataset(b, data=a)
        h.attrs["pickle"] = str(pickle.dumps(attrs))
    return bio.getvalue()


def _write_tif(path, data, crs="EPSG:32633", origin=(453599.2, 5165120.7), res=10.0):
    import rasterio
    from rasterio.transform import from_origin

    data = np.asarray(data)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(data)


def test_eurosat_pixel_join_locates_and_fails_fast(tmp_path):
    rng = np.random.default_rng(2)
    ms = tmp_path / "EuroSAT_MS"
    (ms / "Forest").mkdir(parents=True)
    tifs = {
        n: rng.integers(0, 40000, (13, 4, 4)).astype(np.uint16) for n in ("Forest_1", "Forest_2", "Forest_3")
    }
    tifs["Forest_3"] = tifs["Forest_2"].copy()  # a duplicated patch in the originals
    for n, a in tifs.items():
        _write_tif(ms / "Forest" / f"{n}.tif", a, origin=(453599.2 + (int(n[-1]) * 1000), 5165120.7))
    index = G.eurosat_index(str(ms))
    assert len(index) == 3 and index["hash"].nunique() == 2
    assert index["lon_min"].between(14, 15).all() and index["lat_min"].between(46, 47).all()

    def sample(a):
        return _hdf5_pixels({"02 - Blue": a[1].view(np.int16), "04 - Red": a[3].view(np.int16)})

    blobs = {"id_0": sample(tifs["Forest_1"]), "id_1": sample(tifs["Forest_2"])}
    df, info = G.join_originals(
        "m_eurosat", {"train": ["id_0"], "test": ["id_1"]}, blobs.__getitem__, index, list(G.EUROSAT_JOIN)
    )
    assert df["sample_name"].tolist() == ["id_1", "id_0"] and (df["source"] == "footprint").all()
    assert info["multi_match"] == 1 and df.loc[0, "original"] == "Forest/Forest_2.tif;Forest/Forest_3.tif"
    assert df.loc[0, "lon_max"] > df.loc[1, "lon_max"]  # union of the two duplicate footprints
    blobs["id_2"] = sample(rng.integers(0, 40000, (13, 4, 4)).astype(np.uint16))
    with pytest.raises(SystemExit, match="1 samples match no original"):
        G.join_originals(
            "m_eurosat", {"train": ["id_0", "id_2"]}, blobs.__getitem__, index, list(G.EUROSAT_JOIN)
        )
    with pytest.raises(ValueError, match="not integer-valued"):
        G.sample_hash(_hdf5_pixels({"02 - Blue": np.full((4, 4), 0.5, np.float32)}), ["02 - Blue"])


def test_bigearthnet_index_streams_tar(tmp_path):
    import tarfile

    from rasterio.crs import CRS

    rng = np.random.default_rng(3)
    root = tmp_path / "BigEarthNet-v1.0"
    arrays = {}
    for p in ("S2A_20170613T101031_26_57", "S2A_20170613T101031_27_55"):
        d = root / p
        d.mkdir(parents=True)
        arrays[p] = rng.integers(0, 10000, (1, 4, 4)).astype(np.uint16)
        _write_tif(d / f"{p}_B02.tif", arrays[p])
        _write_tif(d / f"{p}_B03.tif", rng.integers(0, 10000, (1, 4, 4)).astype(np.uint16))
        meta = {
            "labels": ["Pastures"],
            "coordinates": {"ulx": 453599.2, "uly": 5165120.7, "lrx": 453639.2, "lry": 5165080.7},
            "projection": CRS.from_epsg(32633).to_wkt(),
            "tile_source": "S2A_MSIL2A_20170613T101031_N0205_R022_T33UUP_20170613T101608.SAFE",
            "acquisition_date": "2017-06-13 10:10:31",
        }
        (d / f"{p}_labels_metadata.json").write_text(json.dumps(meta))
    tar = tmp_path / "BigEarthNet-S2-v1.0.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(root, arcname="BigEarthNet-v1.0")
    index = G.bigearthnet_index(str(tar))
    assert index["original"].tolist() == sorted(arrays)
    for _, r in index.iterrows():
        assert r["hash"] == G.pixel_hash(arrays[r["original"]][0])
        assert 14 < r["lon_min"] < r["lon_max"] < 15 and 46 < r["lat_min"] < r["lat_max"] < 47
    blob = _hdf5_pixels({"02 - Blue": arrays["S2A_20170613T101031_26_57"][0].astype(np.float32)})
    df, _ = G.join_originals(
        "m_bigearthnet", {"train": ["id_0"]}, {"id_0": blob}.__getitem__, index, [G.BIGEARTHNET_JOIN_KEY]
    )
    assert df["original"].tolist() == ["S2A_20170613T101031_26_57"]
    (root / "S2A_20170613T101031_27_55" / "S2A_20170613T101031_27_55_labels_metadata.json").unlink()
    with tarfile.open(tar, "w:gz") as t:
        t.add(root, arcname="BigEarthNet-v1.0")
    with pytest.raises(SystemExit, match="without both json"):
        G.bigearthnet_index(str(tar))


def test_collect_eval_boxes_uses_originals_when_present(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    gb = tmp_path / "gb"
    (gb / "m_eurosat_meta").mkdir(parents=True)
    (gb / "m_eurosat_meta" / "default_partition.json").write_text(json.dumps({"train": ["id_0"]}))
    ms = tmp_path / "orig" / "eurosat" / "EuroSAT_MS" / "River"
    ms.mkdir(parents=True)
    a = rng.integers(0, 40000, (13, 4, 4)).astype(np.uint16)
    _write_tif(ms / "River_1.tif", a)
    (gb / "m_eurosat_meta" / "id_0.hdf5").write_bytes(
        _hdf5_pixels({"02 - Blue": a[1].view(np.int16), "04 - Red": a[3].view(np.int16)})
    )
    pts, rep = G.collect_eval_boxes(str(gb), None, ["m_eurosat"], str(tmp_path / "orig"))
    assert rep["m_eurosat"]["assessable"] and rep["m_eurosat"]["source"] == "footprint" and len(pts) == 1
    assert "original" not in pts.columns
    with pytest.raises(SystemExit, match="no assessable"):
        G.collect_eval_boxes(str(gb), None, ["m_eurosat"], None)


# ------------------------------------------------------------------ city-level boxes (m-so2sat)
def _cities():
    return pd.DataFrame(
        [("train", "Munich", 48.1371, 11.5754), ("test", "Sydney", -33.8698, 151.2083)],
        columns=["group", "city", "lat", "lon"],
    )


def test_city_boxes_are_50km_rois_and_validate():
    b = G.city_boxes(_cities(), G.SO2SAT_CITY_HALF_EXTENT_M)
    assert b["dataset"].tolist() == ["m_so2sat"] * 2 and b["source"].tolist() == ["city_roi"] * 2
    assert b["split"].tolist() == ["test", "train"] and b["sample_name"].tolist() == [
        "city:Sydney",
        "city:Munich",
    ]
    for r in b.itertuples(index=False):
        lat_c = (r.lat_min + r.lat_max) / 2
        assert abs(G.haversine_m(r.lat_min, r.lon_min, r.lat_max, r.lon_min) - 50_000) < 50  # N-S side
        assert abs(G.haversine_m(lat_c, r.lon_min, lat_c, r.lon_max) - 50_000) < 50  # E-W side at centre
    assert float(G._half_diag_m(b).min()) > G.BIG_EVAL_BOX_M  # takes the brute-force path
    with pytest.raises(ValueError, match="duplicate city"):
        G.city_boxes(pd.concat([_cities(), _cities()]), 25_000.0)
    with pytest.raises(ValueError, match="train \\| test"):
        G.city_boxes(_cities().assign(group=["valid", "test"]), 25_000.0)
    with pytest.raises(ValueError, match="needs columns"):
        G.city_boxes(_cities().drop(columns="lon"), 25_000.0)
    with pytest.raises(ValueError, match="out of range"):
        G.city_boxes(_cities().assign(lat=[91.0, 0.0]), 25_000.0)


def test_nearby_big_box_brute_force_matches_tree(monkeypatch):
    city = G.city_boxes(_cities().iloc[:1], 25_000.0)  # Munich
    lat, lon = 48.1371, 11.5754
    z = 17
    n = 2**z
    tiles = []
    # centre; on the north edge (25 km = 0.2248 deg); ~0.5 km beyond it; ~4 km beyond it
    for j, dlat in enumerate((0.0, 0.2248, 0.2300, 0.26)):
        la = lat + dlat
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(math.radians(la)) + 1 / math.cos(math.radians(la))) / math.pi) / 2 * n)
        tiles.append((0, j, (z, x, y)))
    git = _git(tiles)
    gaps = G.box_gap_m({c: float(city[c].iloc[0]) for c in G.BOX}, {c: git[c].to_numpy() for c in G.BOX})
    assert gaps[0] == 0 and gaps[1] < 300 and 300 < gaps[2] < 1000 and gaps[3] > 3000
    brute = G.nearby(git, city, buffer_m=1000.0)
    assert brute["row"].tolist() == [0, 1, 2] and brute["distance_m"].tolist() == [float(g) for g in gaps[:3]]
    monkeypatch.setattr(G, "BIG_EVAL_BOX_M", 1e9)  # force the KD-tree path on the same inputs
    tree = G.nearby(git, city, buffer_m=1000.0)
    pd.testing.assert_frame_equal(brute, tree)
    # mixed frame: a small box and the city box in one call, both paths used, matches merged
    ev = pd.concat([city, _ev([("m_pv4ger", "train", "p", 0.0, 0.0, 12.8)])], ignore_index=True)
    monkeypatch.setattr(G, "BIG_EVAL_BOX_M", 10_000.0)
    both = G.nearby(pd.concat([git, _git([(1, 0, (17, n // 2, n // 2))])], ignore_index=True), ev, 1000.0)
    assert both["sample_name"].tolist() == ["city:Munich"] * 3 + ["p"]


def test_city_points_merges_and_refuses_double_location(tmp_path):
    pts = _ev([("m_pv4ger", "train", "p", 0.0, 0.0, 12.8)])
    p = tmp_path / "points.csv"
    pts.to_csv(p, index=False)
    rep = {
        "m_pv4ger": {"assessable": True, "n_samples": 1, "source": "centre+nominal"},
        "m_so2sat": {"assessable": False, "n_samples": 21964, "reason": "no coordinates"},
    }
    (tmp_path / "points.csv.report.json").write_text(json.dumps(rep))
    c = tmp_path / "cities.csv"
    _cities().to_csv(c, index=False)
    out, r = G.city_points(str(p), str(c), 25_000.0)
    assert len(out) == 3 and set(out["dataset"]) == {"m_pv4ger", "m_so2sat"}
    assert r["m_so2sat"]["assessable"] and r["m_so2sat"]["cities"] == {"train": 1, "test": 1}
    assert r["m_so2sat"]["n_samples"] == 21964 and r["m_so2sat"]["previous_reason"] == "no coordinates"
    assert r["m_pv4ger"] == rep["m_pv4ger"]
    rep["m_so2sat"]["assessable"] = True
    (tmp_path / "points.csv.report.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="already located"):
        G.city_points(str(p), str(c), 25_000.0)
    # the dedup CLI consistency check accepts the merged pair
    rep["m_so2sat"]["assessable"] = False
    (tmp_path / "points.csv.report.json").write_text(json.dumps(rep))
    out, r = G.city_points(str(p), str(c), 25_000.0)
    assert set(out["dataset"]) == {k for k, v in r.items() if v["assessable"]}


def test_shipped_city_list_matches_the_paper():
    """52 cities of So2Sat LCZ42 appendix B: 42 training + 10 testing/validation."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(here, "experiments", "isaac", "geo_dedup", "so2sat_cities.csv"))
    assert df["group"].value_counts().to_dict() == {"train": 42, "test": 10}
    assert set(df.loc[df["group"] == "test", "city"]) == {
        "Guangzhou",
        "Jakarta",
        "Moscow",
        "Mumbai",
        "Munich",
        "Nairobi",
        "San Francisco",
        "Santiago de Chile",
        "Sydney",
        "Tehran",
    }
    b = G.city_boxes(df, G.SO2SAT_CITY_HALF_EXTENT_M)
    assert len(b) == 52 and b["lat_min"].between(-90, 90).all()


# ------------------------------------------------------------------ fail-fast guards (review 2026-11-12, item 7)
def test_dedup_refuses_partial_and_infinite_boxes():
    """A row with a NaN corner can never match (NaN comparisons are False): before the guard it was
    written to retained.csv as assessed. Fed the mismatch: one NaN corner, one infinite corner."""
    lon_min, lat_min, lon_max, lat_max = G.tile_bounds(10, 0, 218)
    ev = _ev([("m_pv4ger", "train", "a", (lat_min + lat_max) / 2, (lon_min + lon_max) / 2, 12.8)])
    git = _git([(0, 0, (10, 0, 218)), (0, 1, (10, 512, 512)), (1, 0, None)])
    assert G.check_boxes(git, "git", allow_unlocated=True).tolist() == [True, True, False]
    assert len(G.dedup(git, ev, 100.0)[1]) == 1  # the clean frame: row (0, 0) is excluded
    for col, bad in [("lat_max", np.nan), ("lon_min", np.nan), ("lat_min", np.inf)]:
        g = git.copy()
        g.loc[0, col] = bad  # the row that overlaps the evaluation sample
        with pytest.raises(SystemExit, match="Git-10M rows: 1 rows with a partial or infinite box"):
            G.dedup(g, ev, 100.0)
        e = ev.copy()
        e.loc[0, col] = bad
        with pytest.raises(SystemExit, match="evaluation boxes: 1 rows with a partial or infinite box"):
            G.dedup(git, e, 100.0)
    e = ev.copy()
    e.loc[0, G.BOX] = np.nan  # an evaluation sample without any box is not "unlocated": it is an error
    with pytest.raises(SystemExit, match="evaluation boxes: 1 rows without a box"):
        G.dedup(git, e, 100.0)


def _city_points_files(tmp_path):
    pts = _ev([("m_pv4ger", "train", "p", 0.0, 0.0, 12.8)])
    p = tmp_path / "points.csv"
    pts.to_csv(p, index=False)
    rep = {
        "m_pv4ger": {"assessable": True, "n_samples": 1, "source": "centre+nominal"},
        "m_so2sat": {"assessable": False, "n_samples": 21964, "reason": "no coordinates"},
    }
    (tmp_path / "points.csv.report.json").write_text(json.dumps(rep))
    c = tmp_path / "cities.csv"
    _cities().to_csv(c, index=False)
    out = tmp_path / "points_cities.csv"
    G.main(["city-points", "--points", str(p), "--cities", str(c), "--out", str(out)])
    return p, c, out


def test_city_points_reuse_check_accepts_the_same_list_and_refuses_another(tmp_path, capsys):
    p, c, out = _city_points_files(tmp_path)
    before = out.read_bytes(), (tmp_path / "points_cities.csv.report.json").read_bytes()
    argv = ["city-points", "--points", str(p), "--cities", str(c), "--out", str(out), "--check-existing"]
    G.main(argv)
    assert "CITY_POINTS_MATCH" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="this run asks 10000 m"):
        G.main([*argv, "--half-extent-m", "10000"])
    # the mismatch the check exists to catch: the city list is edited after the points were built
    edited = _cities()
    edited.loc[0, "lat"] += 1.0
    edited.to_csv(c, index=False)
    with pytest.raises(SystemExit, match="was built from a cities csv with sha256"):
        G.main(argv)
    # a points file that does not locate the task by cities at all
    with pytest.raises(SystemExit, match="is not located by city boxes"):
        G.main(["city-points", "--points", str(p), "--cities", str(c), "--out", str(p), "--check-existing"])
    assert before == (
        out.read_bytes(),
        (tmp_path / "points_cities.csv.report.json").read_bytes(),
    )  # read-only


def test_dedup_v2_sbatch_checks_an_existing_city_points_file():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "experiments", "isaac", "geo_dedup", "dedup_v2.sbatch")).read()
    body = src.split('if [ ! -f "$POINTS" ]; then', 1)[1].split("\n    fi\n", 1)[0]
    build, reuse = body.split("\n    else\n")
    assert "--check-existing" not in build and '--out "$POINTS"' in build
    assert '--cities "$CITIES" --out "$POINTS" --check-existing' in reuse


def test_dedup_refuses_inverted_boxes():
    """box_gap_m assumes min <= max: an inverted box reads an overlapping pair as disjoint, so the
    row would be retained as assessed (review 2026-11-12, staged-delta finding 1). Fed the
    mismatch on each axis, on the Git-10M side and on the evaluation side."""
    lon_min, lat_min, lon_max, lat_max = G.tile_bounds(10, 0, 218)
    ev = _ev([("m_pv4ger", "train", "a", (lat_min + lat_max) / 2, (lon_min + lon_max) / 2, 12.8)])
    git = _git([(0, 0, (10, 0, 218)), (0, 1, (10, 512, 512)), (1, 0, None)])
    for lo, hi in [("lat_min", "lat_max"), ("lon_min", "lon_max")]:
        g = git.copy()
        g.loc[0, [lo, hi]] = g.loc[0, [hi, lo]].to_numpy()
        assert g.loc[0, lo] > g.loc[0, hi]
        with pytest.raises(SystemExit, match="Git-10M rows: 1 rows with an inverted box"):
            G.dedup(g, ev, 100.0)
        e = ev.copy()
        e.loc[0, [lo, hi]] = e.loc[0, [hi, lo]].to_numpy()
        assert e.loc[0, lo] > e.loc[0, hi]
        with pytest.raises(SystemExit, match="evaluation boxes: 1 rows with an inverted box"):
            G.dedup(git, e, 100.0)
    p = git.copy()  # a degenerate box (a point) is ordered: accepted
    p.loc[1, ["lat_max", "lon_max"]] = p.loc[1, ["lat_min", "lon_min"]].to_numpy()
    assert G.check_boxes(p, "git", allow_unlocated=True).tolist() == [True, True, False]


GEO_DEDUP_DIR = os.environ.get(
    "GEO_DEDUP_DIR", "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/geo_dedup"
)


@pytest.mark.parametrize(
    "name, n_min", [("source_boxes.parquet", 1_000_000), ("points_v3_cities.csv", 10_000)]
)
def test_shipped_box_files_pass_check_boxes(name, n_min):
    """The box files the shipped manifest (dedup_v3_b1000_cities) was built from hold only complete,
    finite, ordered boxes: the guards added on 2026-11-12 would not have fired on that run."""
    path = os.path.join(GEO_DEDUP_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    df = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
    assert len(df) >= n_min, len(df)
    assert G.check_boxes(df, name, allow_unlocated=False).all()
