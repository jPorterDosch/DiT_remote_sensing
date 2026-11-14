"""Conditioning helpers for the DiffusionSat baseline (OUTLINE §4.8). No torch, no diffusers:
importable in any env, so the encoding can be checked without loading the model.

DiffusionSat's UNet takes a 7-value metadata vector. The TRAINING code builds it in two steps
(DiffusionSat/diffusionsat/data_util.py):

  1. `fmow_numerical_metadata` (and the spacenet / satlas twins) return
       [lon + 180, lat + 90, gsd, cloud_cover (fraction), year - 1980, month, day]
     with gsd = sensor gsd * stored_min_side / target_resolution.
  2. `metadata_normalize` divides by (360, 180, 1, 1, 120, 12, 31) and multiplies by 1000.

`metadata_normalize` does NOT apply the offsets itself, so calling it on raw lon / lat / calendar
year (what this repo's sampler did until 2026-11-11, and what the second example in the authors'
notebook does) puts lon / lat in the wrong half of the range and the year at ~16,800 against a
trained range of 0-1000. `raw_metadata` is step 1 for sources that are not fMoW; `known_answer_gate`
checks any step-1 + step-2 chain against the one normalized vector the authors ship
(notebooks/single-image.ipynb, example 1) and proves the gate rejects the old encoding.
"""

from __future__ import annotations

import math
import os
import re

BASE_LON, BASE_LAT, BASE_YEAR = 180.0, 90.0, 1980  # data_util.fmow_numerical_metadata defaults
EARTH_CIRCUMFERENCE_M = 40075016.686  # WGS84 equator, web-mercator tile maths

# Authors' notebook example 1: "a fmow satellite image of a amusement park in Australia",
# normalized metadata [925.8798, 345.2111, 411.4541, 0.0, 308.3333, 166.6667, 354.8387].
# It is fMoW val amusement_park_23_0 (datasets/fmow-val-meta.csv row 1652, timestamp
# 2017-02-11T23:52:26Z; polygon centroid below). The gsd entry is not checked: it depends on the
# stored image size of the authors' shards, which the release does not document (the fMoW JSON
# says 0.5147183, the notebook 0.4114541).
KNOWN = {
    "category": "amusement_park",
    "location_id": 23,
    "image_id": 0,
    "img_filename": "amusement_park_23_0_rgb.jpg",
    "timestamp": "2017-02-11T23:52:26Z",
    "lon": 153.316736017525,
    "lat": -27.861996400980017,
    "year": 2017,
    "month": 2,
    "day": 11,
    "json_gsd": 0.5147183,
    "cloud_cover_pct": 0.0,
    "country_code": "AUS",
    "normalized": [925.8798, 345.2111, None, 0.0, 308.3333, 166.6667, 354.8387],
}
KNOWN_TOL = 2e-3  # the notebook prints 4 decimals

_FMOW_RE = re.compile(r"^(?P<cat>[a-z_-]+)_(?P<loc>\d+)_(?P<img>\d+)_rgb\.jpg$")
_TILE_RE = re.compile(r"^Google-(?P<z>\d+)-(?P<z2>\d+)_(?P<x>\d+)_(?P<y>\d+)\.jpg$")


def raw_metadata(
    lon: float, lat: float, gsd: float, cloud_frac: float, year: int, month: int, day: int
) -> list[float]:
    """Step 1 of the training encoding (the input `metadata_normalize` expects). lon / lat in
    degrees (-180..180, -90..90), cloud cover as a fraction, calendar year."""
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"lon / lat out of range: {lon}, {lat}")
    if not (math.isfinite(gsd) and gsd > 0):
        raise ValueError(f"gsd must be positive and finite, got {gsd}")
    if not (0.0 <= cloud_frac <= 1.0):
        raise ValueError(f"cloud cover must be a fraction in [0, 1], got {cloud_frac}")
    if not (BASE_YEAR <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"date out of range: {year}-{month}-{day}")
    return [
        lon + BASE_LON,
        lat + BASE_LAT,
        float(gsd),
        float(cloud_frac),
        float(year - BASE_YEAR),
        float(month),
        float(day),
    ]


def legacy_metadata(
    lon: float, lat: float, gsd: float, cloud_frac: float, year: int, month: int, day: int
) -> list[float]:
    """The encoding this repo's sampler used before 2026-11-11 (no offsets). Kept ONLY as the
    negative control of `known_answer_gate`; never feed it to the model."""
    return [float(lon), float(lat), float(gsd), float(cloud_frac), float(year), float(month), float(day)]


def known_answer_errors(normalized) -> list[float]:
    """|normalized - notebook| for the checked entries (lon, lat, cloud, year, month, day)."""
    got = [float(v) for v in normalized]
    if len(got) != 7:
        raise ValueError(f"expected 7 metadata values, got {len(got)}")
    return [abs(g - e) for g, e in zip(got, KNOWN["normalized"]) if e is not None]


def known_answer_gate(normalize, encode=None) -> str:
    """Executable gate. `normalize` = diffusionsat.metadata_normalize. `encode` (optional) returns
    the step-1 vector of the KNOWN image through the code path under test (default: raw_metadata).
    Raises unless (a) the chain reproduces the authors' normalized vector and (b) the SAME check
    rejects the legacy encoding (a gate that cannot fail verifies nothing). Returns the PASS line."""
    k = KNOWN
    args = (k["lon"], k["lat"], k["json_gsd"], k["cloud_cover_pct"] / 100.0, k["year"], k["month"], k["day"])
    step1 = encode() if encode is not None else raw_metadata(*args)
    err = max(known_answer_errors(normalize(step1)))
    if not err <= KNOWN_TOL:
        raise SystemExit(
            f"GATE metadata encoding FAIL: max |normalized - authors' notebook| = {err:.4f} > {KNOWN_TOL} "
            f"(got {[round(float(v), 4) for v in normalize(step1)]}); do NOT sample"
        )
    bad = max(known_answer_errors(normalize(legacy_metadata(*args))))
    if not bad > 100 * KNOWN_TOL:
        raise SystemExit(
            f"GATE metadata encoding FAIL: the legacy (no-offset) encoding passes the check "
            f"(max error {bad:.4f}); the gate cannot detect the bug it exists for"
        )
    return (
        f"GATE metadata encoding PASS: authors' notebook vector reproduced (max error {err:.4f} <= {KNOWN_TOL}); "
        f"negative control: legacy no-offset encoding rejected (max error {bad:.1f})"
    )


def parse_fmow_path(path: str) -> tuple[str, int, int, str]:
    """'val/airport/airport_0/airport_0_3_rgb.jpg' -> ('airport', 0, 3, 'airport_0_3_rgb.jpg').
    The category comes from the directory, and must agree with the file name."""
    parts = path.split("/")
    if len(parts) != 4:
        raise ValueError(f"unexpected fMoW path {path!r}")
    cat, fname = parts[1], parts[3]
    m = _FMOW_RE.match(fname)
    if not m or m.group("cat") != cat or parts[2] != f"{cat}_{m.group('loc')}":
        raise ValueError(
            f"fMoW path does not parse as <split>/<cat>/<cat>_<loc>/<cat>_<loc>_<img>_rgb.jpg: {path!r}"
        )
    return cat, int(m.group("loc")), int(m.group("img")), fname


def fmow_json_url(path: str) -> str:
    """Public SpaceNet bucket URL of the per-image fMoW-rgb metadata JSON."""
    parse_fmow_path(path)
    return (
        "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/"
        + os.path.splitext(path)[0]
        + ".json"
    )


def tile_gsd(img_name: str, lon: float, lat: float, target_px: int = 512) -> float:
    """gsd (m / px at the model's `target_px` output) of a Git-10M Google tile named
    'Google-<z>-<z>_<x>_<y>.jpg'. One image is one web-mercator tile at zoom z, so its ground
    width is circumference * cos(lat) / 2^z; DiffusionSat's convention is ground extent /
    target resolution. `lon` must be the tile centre implied by x (guards the one-tile reading)."""
    m = _TILE_RE.match(img_name)
    if not m or m.group("z") != m.group("z2"):
        raise ValueError(f"not a Git-10M Google tile name: {img_name!r}")
    z, x = int(m.group("z")), int(m.group("x"))
    centre = (x + 0.5) / 2**z * 360.0 - 180.0
    if abs(centre - lon) > 1e-6:
        raise ValueError(f"{img_name}: tile centre lon {centre} != coords lon {lon}")
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / 2**z / target_px


def cast_overflow_count(values, dtype: str) -> int:
    """Number of finite entries of `values` that are no longer finite after a cast to `dtype`
    ('float16' / 'float32'). float16 tops out at 65,504: a normalized gsd of 65.5 m x 1000 or more
    becomes inf there, the sinusoidal metadata embedding of inf is NaN, and the sample decodes
    flat (found 2026-11-12: the pipeline casts metadata to the fp16 weights' dtype)."""
    import numpy as np

    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("values must be finite before the cast")
    with np.errstate(over="ignore"):
        return int((~np.isfinite(a.astype(np.dtype(dtype)))).sum())


FLAT_STD = 2.0  # uint8 units; a black / NaN-decoded / constant image has std 0


def flat_count(imgs_uint8, thresh: float = FLAT_STD) -> int:
    """Number of images (N, H, W, 3 uint8) whose pixel std is below `thresh`: black frames from a
    NaN latent or a tripped safety filter, never a real overhead scene at 512 px."""
    import numpy as np

    a = np.asarray(imgs_uint8)
    if a.ndim != 4 or a.dtype != np.uint8:
        raise ValueError(f"expected (N, H, W, C) uint8, got {a.shape} {a.dtype}")
    return int((a.reshape(len(a), -1).astype(np.float32).std(axis=1) < thresh).sum())


GSD_NORM = 1000.0  # metadata_normalize: gsd / 1 * 1000, so conditioning.csv's gsd_n = metres x 1000


def gsd_subset(rows, gsd_max_m: float) -> list[int]:
    """Image indices of the located rows of a geo run's conditioning.csv (dict rows: i,
    has_metadata, gsd_n) whose tile gsd is <= `gsd_max_m` metres. Rows without metadata are never
    in the subset. Used for the declared sensitivity score "located rows inside DiffusionSat's
    trained gsd range" (ledger 2026-11-12: gsd <= 10 m, n = 2,886 of 5,000)."""
    if not (math.isfinite(gsd_max_m) and gsd_max_m > 0):
        raise ValueError(f"gsd_max_m must be positive and finite, got {gsd_max_m}")
    out, seen = [], set()
    for r in rows:
        i = int(r["i"])
        if i in seen:
            raise ValueError(f"conditioning row {i} appears twice")
        seen.add(i)
        if r["has_metadata"] not in ("0", "1"):
            raise ValueError(f"row {i}: has_metadata must be 0 or 1, got {r['has_metadata']!r}")
        if r["has_metadata"] == "0":
            continue
        gsd = float(r["gsd_n"]) / GSD_NORM
        if not (math.isfinite(gsd) and gsd > 0):
            raise ValueError(f"row {i}: located row with gsd {gsd}")
        if gsd <= gsd_max_m:
            out.append(i)
    if not out:
        raise ValueError(f"no located row with gsd <= {gsd_max_m} m")
    return out
