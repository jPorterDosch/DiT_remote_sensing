"""Vendored GeoCore-9B model code: pinned commit, recorded file hashes, HF config parity.

pytest tests/test_geocore_vendored.py -v
"""

import hashlib
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from models.geocore import GEOCORE_COMMIT, flux2, vae_flux2  # noqa: E402


def test_commit_pin():
    assert GEOCORE_COMMIT == "17406c03a9bd090ff1570809bf6bd7f4a35e2359"


def test_vendored_file_sha1s_match_vendored_md():
    root = os.path.dirname(flux2.__file__)
    md = open(os.path.join(root, "VENDORED.md")).read()
    for name in ("flux2.py", "vae_flux2.py"):
        sha = hashlib.sha1(open(os.path.join(root, name), "rb").read()).hexdigest()
        assert sha in md, f"{name} sha1 {sha} not recorded in VENDORED.md"


def test_params_match_hf_config():
    p = flux2.GeoCore9BParams()
    assert (p.in_channels, p.hidden_size, p.num_heads, p.depth, p.depth_single_blocks) == (
        128,
        4096,
        32,
        8,
        24,
    )
    assert p.axes_dim == [32, 48, 48] and p.theta == 2000 and p.mlp_ratio == 3.0
    assert vae_flux2.AutoEncoderParams().z_channels == 32
