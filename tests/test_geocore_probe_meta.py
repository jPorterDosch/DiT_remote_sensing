"""Probe-side gates read width / depth / discard channels from the cache meta for non-FLUX
backbones and stay byte-identical for FLUX caches (eval/features.py).

    pytest tests/test_geocore_probe_meta.py -v
"""

import os
import sys

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from eval import features as F  # noqa: E402

STAMP = {"double_block_feature": F.DOUBLE_BLOCK_FEATURE}


def test_flux_meta_without_new_keys_unchanged():  # Review Focus 4
    F.check_double_block_stamp({"weights": "flux-dev", **STAMP}, 4, "x")
    F.check_double_block_stamp({"weights": "flux-dev"}, 28, "x")  # single block: no stamp needed
    F.check_double_block_stamp({}, 28, "x")  # historical cache without weights key
    F.check_double_block_stamp({"weights": "flux-dev+lora:proj", **STAMP}, 18, "x")
    with pytest.raises(SystemExit):
        F.check_double_block_stamp({"weights": "flux-dev"}, 4, "x")
    assert F.meta_discard_channels({"weights": "flux-dev"}, "x") == [154, 1446]
    assert F.meta_discard_channels({}, "x") == [154, 1446]


def test_geocore_meta_requires_keys():
    with pytest.raises(SystemExit, match="num_double_blocks"):
        F.check_double_block_stamp({"weights": "geocore-9b", **STAMP}, 4, "x")
    with pytest.raises(SystemExit, match="discard_channels"):
        F.meta_discard_channels({"weights": "geocore-9b"}, "x")


def test_geocore_stamp_uses_meta_count():
    m = {"weights": "geocore-9b", "num_double_blocks": 8}
    F.check_double_block_stamp(m, 8, "x")  # first single block: no stamp needed
    F.check_double_block_stamp({**m, **STAMP}, 4, "x")
    with pytest.raises(SystemExit):
        F.check_double_block_stamp(m, 4, "x")  # double block without the stamp
    assert F.meta_discard_channels({**m, "discard_channels": [7]}, "x") == [7]


def test_ditf_discard_param():
    """`discard` zeroes the listed channels BEFORE the per-token normalization (verbatim harness
    rule), so the output equals ditf of an input whose channels were pre-zeroed; the default list
    stays FLUX's [154, 1446]."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=(3, 2, 2000)).astype(np.float32)
    m = rng.normal(size=(2, 3, 2000)).astype(np.float32)
    pre = f.copy()
    pre[:, :, 0] = 0.0
    out = F.ditf(f, m, discard=[0])
    assert np.array_equal(out, F.ditf(pre, m, discard=[]))
    assert not np.array_equal(out, F.ditf(f, m, discard=[]))  # the parameter is live
    pre_default = f.copy()
    pre_default[:, :, [154, 1446]] = 0.0
    assert np.array_equal(F.ditf(f, m), F.ditf(pre_default, m, discard=[]))
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-5)
