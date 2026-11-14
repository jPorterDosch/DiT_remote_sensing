"""DiffusionSat conditioning helpers (experiments/isaac/baselines/dsat_cond.py): the metadata
encoding must reproduce the authors' shipped vector, and its gate must reject the encoding this
repo used before 2026-11-11."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "isaac", "baselines"
    ),
)
import dsat_cond as DC  # noqa: E402


def normalize(m):  # diffusionsat.data_util.metadata_normalize, restated (that package needs its own env)
    lon, lat, gsd, cloud, year, month, day = m
    return [
        lon / 360 * 1000,
        lat / 180 * 1000,
        gsd / 1.0 * 1000,
        cloud * 1000,
        year / 120 * 1000,
        month / 12 * 1000,
        day / 31 * 1000,
    ]


def test_gate_passes_on_training_convention():
    assert DC.known_answer_gate(normalize).startswith("GATE metadata encoding PASS")


def test_gate_fails_on_legacy_encoding():
    k = DC.KNOWN

    def legacy():
        return DC.legacy_metadata(k["lon"], k["lat"], k["json_gsd"], 0.0, k["year"], k["month"], k["day"])

    with pytest.raises(SystemExit, match="GATE metadata encoding FAIL"):
        DC.known_answer_gate(normalize, encode=legacy)


def test_gate_fails_when_it_cannot_tell_the_encodings_apart():
    with pytest.raises(SystemExit, match="cannot detect"):
        DC.known_answer_gate(lambda m: DC.KNOWN["normalized"][:2] + [0.0] + DC.KNOWN["normalized"][3:])


def test_raw_metadata_offsets_and_ranges():
    assert DC.raw_metadata(10.0, 51.0, 0.6, 0.0, 2020, 6, 15) == [190.0, 141.0, 0.6, 0.0, 40.0, 6.0, 15.0]
    n = normalize(DC.raw_metadata(10.0, 51.0, 0.6, 0.0, 2020, 6, 15))
    assert max(n[i] for i in (0, 1, 4, 5, 6)) <= 1000.0
    for bad in [
        (200.0, 0.0, 0.6, 0.0, 2020, 6, 15),
        (0.0, 0.0, 0.6, 5.0, 2020, 6, 15),
        (0.0, 0.0, 0.0, 0.0, 2020, 6, 15),
        (0.0, 0.0, 0.6, 0.0, 1979, 6, 15),
        (0.0, 0.0, 0.6, 0.0, 2020, 13, 15),
    ]:
        with pytest.raises(ValueError):
            DC.raw_metadata(*bad)


def test_parse_fmow_path():
    assert DC.parse_fmow_path("val/airport_hangar/airport_hangar_10/airport_hangar_10_2_rgb.jpg") == (
        "airport_hangar",
        10,
        2,
        "airport_hangar_10_2_rgb.jpg",
    )
    assert DC.parse_fmow_path(
        "val/multi-unit_residential/multi-unit_residential_120/multi-unit_residential_120_3_rgb.jpg"
    )[:3] == ("multi-unit_residential", 120, 3)
    assert DC.fmow_json_url("val/airport/airport_0/airport_0_0_rgb.jpg").endswith(
        "/fmow-rgb/val/airport/airport_0/airport_0_0_rgb.json"
    )
    for bad in [
        "val/airport/airport_1/airport_0_0_rgb.jpg",
        "val/airport/airport_0/airport_0_0_msrgb.jpg",
        "airport_0_0_rgb.jpg",
    ]:
        with pytest.raises(ValueError):
            DC.parse_fmow_path(bad)


def test_tile_gsd():
    lon = (491 + 0.5) / 2**10 * 360 - 180
    g = DC.tile_gsd("Google-10-10_491_331.jpg", lon, 53.435590187258626)
    assert g == pytest.approx(45.5355, abs=1e-3)
    assert DC.tile_gsd(
        "Google-11-11_982_662.jpg", (982.5) / 2**11 * 360 - 180, 53.435590187258626
    ) == pytest.approx(g / 2, rel=1e-9)
    with pytest.raises(ValueError):
        DC.tile_gsd("Google-10-10_491_331.jpg", 0.0, 53.4)  # centre does not match the tile
    with pytest.raises(ValueError):
        DC.tile_gsd("Sky-images6-abc.jpg", 0.0, 0.0)


def test_cast_overflow_count_sees_the_float16_gsd_overflow():
    ok = [705.32, 603.23, 619.98, 0.0, 208.33, 916.67, 548.39]  # an fMoW row
    far = [500.0, 800.0, 76440.0, 0.0, 333.33, 500.0, 483.87]  # 76.44 m tile, normalized
    assert DC.cast_overflow_count([ok], "float16") == 0
    assert DC.cast_overflow_count([ok, far], "float16") == 1
    assert DC.cast_overflow_count([[65503.0], [65520.0]], "float16") == 1  # float16 max is 65,504
    assert DC.cast_overflow_count([ok, far], "float32") == 0
    with pytest.raises(ValueError):
        DC.cast_overflow_count([float("inf")], "float32")


def test_flat_count_fires_on_black_and_constant_images_only():
    import numpy as np

    rng = np.random.default_rng(0)
    real = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)
    black = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    grey = np.full((1, 32, 32, 3), 127, dtype=np.uint8)
    assert DC.flat_count(real) == 0
    assert DC.flat_count(np.concatenate([real, black, grey])) == 3
    with pytest.raises(ValueError):
        DC.flat_count(real.astype(np.float32))


def test_gsd_subset_keeps_located_rows_at_or_below_the_limit_only():
    rows = [
        {"i": "0", "has_metadata": "1", "gsd_n": "548.0"},  # 0.548 m
        {"i": "1", "has_metadata": "1", "gsd_n": "10000.0"},  # exactly 10 m: kept
        {"i": "2", "has_metadata": "1", "gsd_n": "10000.5"},  # just above: dropped
        {"i": "3", "has_metadata": "0", "gsd_n": "0.0"},  # unlocated: never kept
        {"i": "4", "has_metadata": "1", "gsd_n": "76436.7"},
    ]
    assert DC.gsd_subset(rows, 10.0) == [0, 1]
    assert DC.gsd_subset(rows, 100.0) == [0, 1, 2, 4]
    with pytest.raises(ValueError, match="no located row"):
        DC.gsd_subset(rows, 0.1)
    with pytest.raises(ValueError, match="located row with gsd"):
        DC.gsd_subset([{"i": "0", "has_metadata": "1", "gsd_n": "0.0"}], 10.0)
    with pytest.raises(ValueError, match="twice"):
        DC.gsd_subset(rows + [rows[0]], 10.0)
    with pytest.raises(ValueError, match="has_metadata"):
        DC.gsd_subset([{"i": "0", "has_metadata": "True", "gsd_n": "1.0"}], 10.0)
