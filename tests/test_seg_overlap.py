"""eval.seg_overlap: measurement of Git-10M training rows near the GEO-Bench segmentation tasks.

Each guard is fed the mismatch it exists to catch (CLAUDE.md rule 6)."""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from eval import geo_dedup as G
from eval import seg_overlap as S
from tests.test_geo_dedup import _hdf5_blob

GEOREF = ((0.001, 0, 7.0, 0, -0.001, 51.5), "EPSG:4326")
PV_A, PV_B = "7.000000,51.500000", "51.600000,7.200000"  # lon,lat and lat,lon: both orders ship


def _task_dir(root, task, samples):
    """samples: {split: {name: blob}} -> <root>/<dir>/default_partition.json + <name>.hdf5."""
    d = os.path.join(root, S.SEG_TASKS[task]["dir"])
    os.makedirs(d)
    with open(os.path.join(d, "default_partition.json"), "w") as f:
        json.dump({split: sorted(v) for split, v in samples.items()}, f)
    for v in samples.values():
        for name, blob in v.items():
            with open(os.path.join(d, f"{name}.hdf5"), "wb") as f:
                f.write(blob)
    return d


def _seg_root(tmp_path):
    root = str(tmp_path / "seg")
    plain = _hdf5_blob(["b1"], None, None, shape=(320, 320))  # no georeferencing; 320 px at 10 m
    small = _hdf5_blob(["b1"], None, None, shape=(8, 8))
    _task_dir(root, "m_pv4ger_seg", {"train": {PV_A: plain}, "test": {PV_B: plain}})
    _task_dir(root, "m_chesapeake_landcover", {"train": {"id_0": small}, "test": {"id_1": small}})
    _task_dir(root, "m_SA_crop_type", {"train": {"ref_x_0001_0": small}, "test": {"ref_x_0002_1": small}})
    return root


def _git():
    """Seven training rows: 50 m boxes; the last has no coordinates."""
    east = math.degrees(2000.0 / G.EARTH_R) / math.cos(math.radians(51.5))  # 2,000 m east of PV_A
    pts = [
        (51.5, 7.0, "google_tile"),  # r0 on PV_A                          -> gap 0
        (51.5, 7.0 + east, "google_tile"),  # r1 2,000 - 1,600 - 50 = 350 m from PV_A
        (51.6, 7.2, "ssl4eo"),  # r2 on PV_B, already excluded
        (53.0, 10.0, "google_tile"),  # r3 Germany, far from both
        (39.0, -76.5, "google_tile"),  # r4 Maryland: inside the Chesapeake region
        (-33.0, 19.0, "skyscript"),  # r5 Western Cape: inside the South Africa region
    ]
    rows = [
        dict(
            shard=0,
            row=i,
            img_name=f"n{i}",
            google_location=None,
            source=src,
            **dict(zip(G.BOX, G.centre_box(lat, lon, 50.0))),
        )  # fmt: skip
        for i, (lat, lon, src) in enumerate(pts)
    ]
    rows.append(dict(shard=0, row=6, img_name="n6", google_location=None, source="other",
                     **dict(zip(G.BOX, (np.nan,) * 4))))  # fmt: skip
    return pd.DataFrame(rows)


def _manifests(retained=(0, 1, 3, 4), excluded=(2, 5)):
    return (
        pd.DataFrame({"shard": 0, "row": list(retained)}),
        pd.DataFrame({"shard": 0, "row": list(excluded)}),
    )


def test_task_boxes_reads_centre_from_names_and_half_extent_from_the_artifact(tmp_path):
    root = _seg_root(tmp_path)
    df, rep = S.task_boxes("m_pv4ger_seg", os.path.join(root, "m-pv4ger-seg"))
    assert rep["assessable"] and rep["source"] == "centre+nominal" and rep["files_opened"] == 2
    assert (rep["rows"], rep["cols"], rep["spatial_resolution_m"], rep["half_extent_m"]) == (
        320,
        320,
        10.0,
        1600.0,
    )
    a = df.set_index("sample_name").loc[PV_A]
    b = df.set_index("sample_name").loc[PV_B]
    assert (a["lon_min"] + a["lon_max"]) / 2 == pytest.approx(7.0) and (
        a["lat_min"] + a["lat_max"]
    ) / 2 == pytest.approx(51.5)
    assert (b["lon_min"] + b["lon_max"]) / 2 == pytest.approx(7.2) and (
        b["lat_min"] + b["lat_max"]
    ) / 2 == pytest.approx(51.6)
    assert G.haversine_m(a["lat_min"], 7.0, a["lat_max"], 7.0) == pytest.approx(3200, rel=1e-6)


def test_task_without_coordinates_is_reported_not_guessed(tmp_path):
    root = _seg_root(tmp_path)
    df, rep = S.task_boxes("m_chesapeake_landcover", os.path.join(root, "m-chesapeake-landcover"))
    assert len(df) == 0 and rep["assessable"] is False and rep["files_opened"] == 2
    assert rep["files_with_transform_and_crs"] == 0 and "no transform / crs" in rep["reason"]


def test_footprints_are_used_when_every_sample_has_them_and_mixed_is_refused(tmp_path):
    geo = _hdf5_blob(["b1"], *GEOREF)
    plain = _hdf5_blob(["b1"], None, None)
    d = _task_dir(
        str(tmp_path / "all"), "m_chesapeake_landcover", {"train": {"id_0": geo}, "test": {"id_1": geo}}
    )
    df, rep = S.task_boxes("m_chesapeake_landcover", d)
    assert rep["assessable"] and rep["source"] == "footprint" and df["source"].tolist() == ["footprint"] * 2
    assert df["lon_max"].tolist() == pytest.approx([7.004] * 2)
    d = _task_dir(
        str(tmp_path / "mix"), "m_chesapeake_landcover", {"train": {"id_0": geo}, "test": {"id_1": plain}}
    )
    with pytest.raises(SystemExit, match="mixed"):
        S.task_boxes("m_chesapeake_landcover", d)


def test_samples_of_different_size_are_refused(tmp_path):
    a = _hdf5_blob(["b1"], None, None, shape=(320, 320))
    b = _hdf5_blob(["b1"], None, None, shape=(256, 256))
    d = _task_dir(str(tmp_path / "s"), "m_pv4ger_seg", {"train": {PV_A: a}, "test": {PV_B: b}})
    with pytest.raises(SystemExit, match="differ in size"):
        S.task_boxes("m_pv4ger_seg", d)


def test_duplicate_names_across_splits_are_refused(tmp_path):
    a = _hdf5_blob(["b1"], None, None)
    d = _task_dir(str(tmp_path / "s"), "m_pv4ger_seg", {"train": {PV_A: a}, "test": {PV_A: a}})
    with pytest.raises(SystemExit, match="duplicate"):
        S.task_boxes("m_pv4ger_seg", d)


def test_measure_counts_distinct_rows_and_splits_retained_from_excluded(tmp_path):
    root = _seg_root(tmp_path)
    points = pd.DataFrame(
        {"dataset": ["m_pv4ger", "m_pv4ger", "m_forestnet"], "sample_name": [PV_A, "9.0,50.0", PV_B]}
    )
    rep, keep = S.measure(root, _git(), *_manifests(), points, 1000.0)
    assert rep["control"]["pass"]
    o = rep["tasks"]["m_pv4ger_seg"]["overlap"]
    assert o == {
        "buffer_m": 1000.0,
        "eval_samples_hit": 2,
        "training_rows_within_buffer": 3,
        "of_which_already_excluded": 1,
        "of_which_retained": 2,
        "retained_rows_overlapping_a_sample": 1,
        "retained_by_source": {"google_tile": 2},
    }
    assert rep["tasks"]["m_pv4ger_seg"]["names_shared_with_m_pv4ger"] == 1  # the forestnet row does not count
    ch = rep["tasks"]["m_chesapeake_landcover"]
    assert ch["assessable"] is False and "overlap" not in ch and "state:MD" in ch["region"]
    assert (
        ch["region_upper_bound"]["training_rows_in_region"],
        ch["region_upper_bound"]["of_which_retained"],
    ) == (1, 1)
    assert "NOT a hit count" in ch["region_upper_bound"]["meaning"]
    sa = rep["tasks"]["m_SA_crop_type"]["region_upper_bound"]
    assert (sa["training_rows_in_region"], sa["of_which_already_excluded"], sa["of_which_retained"]) == (
        1,
        1,
        0,
    )
    assert rep["git10m"] == {"rows": 7, "located": 6, "retained": 4, "excluded": 2, "without_coordinates": 1}
    # the row list holds retained rows near LOCATED samples only: no region rows, no excluded rows
    assert sorted(keep["row"]) == [0, 1] and set(keep["dataset"]) == {"m_pv4ger_seg"}
    assert keep.set_index("row").loc[0, "distance_m"] == 0.0
    assert keep.set_index("row").loc[1, "distance_m"] == pytest.approx(350, abs=5)


def test_buffer_decides_the_near_row(tmp_path):
    root = _seg_root(tmp_path)
    rep, keep = S.measure(root, _git(), *_manifests(), None, 300.0)  # r1 sits at ~350 m
    o = rep["tasks"]["m_pv4ger_seg"]["overlap"]
    assert (o["training_rows_within_buffer"], o["of_which_retained"]) == (2, 1) and sorted(keep["row"]) == [0]
    assert "names_shared_with_m_pv4ger" not in rep["tasks"]["m_pv4ger_seg"]


def test_manifests_from_other_metadata_are_refused(tmp_path):
    root = _seg_root(tmp_path)
    with pytest.raises(SystemExit, match="do not partition the located rows"):  # r3 in neither
        S.measure(root, _git(), *_manifests(retained=(0, 1, 4)), None, 1000.0)
    with pytest.raises(SystemExit, match="do not partition the located rows"):  # r0 in both
        S.measure(root, _git(), *_manifests(excluded=(0, 2, 5)), None, 1000.0)
    with pytest.raises(SystemExit, match="do not partition the located rows"):  # the unlocated row listed
        S.measure(root, _git(), *_manifests(retained=(0, 1, 3, 4, 6)), None, 1000.0)


def test_dead_instrument_stops_before_any_count(tmp_path, monkeypatch):
    root = _seg_root(tmp_path)
    monkeypatch.setattr(G, "box_gap_m", lambda a, b: np.full(np.shape(b["lat_min"]), 1e6))
    rep, keep = S.measure(root, _git(), *_manifests(), None, 1000.0)
    assert rep["control"]["pass"] is False and len(keep) == 0
    assert all("overlap" not in t and "region_upper_bound" not in t for t in rep["tasks"].values())


def test_cli_writes_once_and_fails_on_a_failed_gate(tmp_path, monkeypatch, capsys):
    root = _seg_root(tmp_path)
    dedup = tmp_path / "dedup"
    dedup.mkdir()
    retained, excluded = _manifests()
    retained.assign(img_name="x").to_csv(dedup / "retained.csv", index=False)
    excluded.assign(img_name="x").to_csv(dedup / "excluded.csv", index=False)
    boxes = tmp_path / "boxes.parquet"
    pd.DataFrame({"img_name": ["unused"]}).to_parquet(boxes)
    monkeypatch.setattr(G, "load_meta", lambda meta, n, partial: (_git(), []))
    monkeypatch.setattr(G, "apply_source_boxes", lambda git, b: git)
    out = str(tmp_path / "res" / "seg_overlap")
    argv = ["--seg-root", root, "--meta", "unused", "--source-boxes", str(boxes), "--dedup", str(dedup),
            "--buffer-m", "1000", "--out", out]  # fmt: skip
    assert S.main(argv) == 0
    printed = capsys.readouterr().out
    assert "GATE instrument control PASS" in printed and "SEG_OVERLAP_OK" in printed
    rep = json.load(open(out + ".json"))
    assert rep["tasks"]["m_pv4ger_seg"]["overlap"]["of_which_retained"] == 2 and rep["buffer_m"] == 1000.0
    assert len(pd.read_csv(out + ".retained_matches.csv")) == 2
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.main(argv)
    monkeypatch.setattr(G, "box_gap_m", lambda a, b: np.full(np.shape(b["lat_min"]), 1e6))
    out2 = str(tmp_path / "res" / "dead")
    assert S.main(argv[:-1] + [out2]) == 1
    assert "GATE instrument control FAIL" in capsys.readouterr().out and not os.path.exists(out2 + ".json")
