"""eval/ssl4eo_pixel_check.py: the score separates a matching patch from an unrelated one under
both geometric fits, the gate fires on each criterion, and the zarr reader decodes a synthetic
v1.1-style zip (numcodecs required for that one test)."""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import ssl4eo_pixel_check as PC  # noqa: E402


def _scene(seed: int, side: int = 264) -> np.ndarray:
    """Smooth random field with edges: correlated across resize / crop, uncorrelated across seeds."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(side // 8 + 2, side // 8 + 2, 3))
    from PIL import Image

    im = Image.fromarray(((base - base.min()) / np.ptp(base) * 255).astype(np.uint8)).resize(
        (side, side), Image.Resampling.BICUBIC
    )
    return np.asarray(im)


def test_score_prefers_the_matching_patch_under_either_fit():
    patches = np.stack([_scene(s) for s in range(4)])  # 4 "seasons"
    from PIL import Image

    # Git-10M image derived from season 2 by resize, and another by centre crop
    resized = np.asarray(Image.fromarray(patches[2]).resize((256, 256), Image.Resampling.BICUBIC))
    cropped = patches[1][4:260, 4:260]
    sc, season, fit = PC.score(resized, patches)
    assert sc > 0.95 and season == 2 and fit == 0
    sc, season, fit = PC.score(cropped, patches)
    assert sc > 0.95 and season == 1 and fit == 1
    null, _, _ = PC.score(resized, np.stack([_scene(s) for s in range(10, 14)]))
    assert abs(null) < 0.3


def _rows(mapped, null, ids):
    return pd.DataFrame({"mapped": mapped, "null": null, "id": ids})


def test_gate_passes_and_fires_per_criterion():
    n = 100
    ids = np.arange(n) * 3000  # half below, half above a void at 150000
    good = _rows(np.full(n, 0.85), np.full(n, 0.05), ids)
    assert PC.gate(good, 150_000)["pass"]
    assert not PC.gate(_rows(np.full(n, 0.5), np.full(n, 0.05), ids), 150_000)["pass"]  # mapped too low
    assert not PC.gate(_rows(np.full(n, 0.85), np.full(n, 0.4), ids), 150_000)["pass"]  # null too high
    m = np.full(n, 0.85)
    m[:10] = 0.0  # 10 % below null
    assert not PC.gate(_rows(m, np.full(n, 0.05), ids), 150_000)["pass"]
    # upper half wrong (the "do not skip voids" failure mode): overall still 92 % > null? no: 50 %
    m = np.full(n, 0.85)
    m[ids >= 150_000] = 0.0
    g = PC.gate(_rows(m, np.full(n, 0.05), ids), 150_000)
    assert not g["pass"] and g["halves"]["above_first_void"]["frac_mapped_gt_null"] == 0.0
    # a half with no rows fails (the split must be exercised)
    assert not PC.gate(good, 10**9)["pass"]


def test_read_zarr_rgb_decodes_a_synthetic_v11_zip():
    numcodecs = pytest.importorskip("numcodecs")
    codec = numcodecs.get_codec({"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1, "blocksize": 0})
    bands = np.random.default_rng(0).integers(0, 255, (4, 3, 264, 264), dtype=np.uint8)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "bands/.zarray",
            json.dumps(
                {
                    "chunks": [4, 3, 264, 264],
                    "shape": [4, 3, 264, 264],
                    "dtype": "|u1",
                    "compressor": codec.get_config(),
                }
            ),
        )
        z.writestr("bands/0.0.0.0", codec.encode(np.ascontiguousarray(bands)))
        names = np.array(["B04", "B03", "B02"], dtype="<U3")
        z.writestr(
            "band/.zarray",
            json.dumps({"chunks": [3], "shape": [3], "dtype": "<U3", "compressor": codec.get_config()}),
        )
        z.writestr("band/0", codec.encode(names))
        z.writestr(
            "sample/.zarray", json.dumps({"chunks": [], "shape": [], "dtype": "<U7", "compressor": None})
        )
        z.writestr("sample/0", np.array("0001234", dtype="<U7").tobytes())
    arr, sid = PC.read_zarr_rgb(buf.getvalue())
    assert sid == "0001234" and arr.shape == (4, 264, 264, 3)
    assert np.array_equal(arr[:, :, :, 0], bands[:, 0])  # R = B04
    # wrong band order is refused
    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as z:
        with zipfile.ZipFile(buf) as src:
            for n in src.namelist():
                z.writestr(
                    n,
                    src.read(n)
                    if n != "band/0"
                    else codec.encode(np.array(["B02", "B03", "B04"], dtype="<U3")),
                )
    with pytest.raises(ValueError, match="band order"):
        PC.read_zarr_rgb(buf2.getvalue())
