"""eval/gen_metrics.py: FID / KID numerics, the input contract and the identity record, with an
injected feature net (no Inception download in tests)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval import gen_metrics as GM  # noqa: E402


class _Net(torch.nn.Module):
    """(N, 3, 299, 299) -> (N, 8): fixed random projection of the mean colour + a few moments."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(6, 8), requires_grad=False)

    def forward(self, x):
        m = x.mean(dim=(2, 3))
        s = x.std(dim=(2, 3))
        return torch.cat([m, s], dim=1) @ self.w


def _imgs(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5 + shift, 0.2, size=(n, 32, 32, 3))
    return (np.clip(base, 0, 1) * 255).round().astype(np.uint8)


def test_fid_and_kid_zero_for_identical_sets_and_positive_for_shifted():
    fa = np.random.default_rng(0).normal(size=(500, 16))
    fb = np.random.default_rng(1).normal(size=(500, 16))
    assert GM.fid(fa, fa) < 1e-6
    km, ks = GM.kid(fa, fa, subsets=10, subset_size=200)
    assert abs(km) < 0.05  # same distribution: unbiased estimate scatters around 0
    same = GM.fid(fa, fb)
    far = GM.fid(fa, fb + 2.0)
    assert 0 <= same < far
    k_same, _ = GM.kid(fa, fb, subsets=20, subset_size=200)
    k_far, _ = GM.kid(fa, fb + 2.0, subsets=20, subset_size=200)
    assert k_same < k_far and k_far > 0.1
    with pytest.raises(ValueError):
        GM.fid(fa, fb[:, :8])


def test_extractor_input_contract_and_compare_identity():
    ex = GM.InceptionFeatures("cpu", net=_Net(), batch=16)
    real, fake = _imgs(40, 0), _imgs(40, 1, shift=0.2)
    f = ex(real)
    assert f.shape == (40, 8) and f.dtype == np.float64
    # float NCHW in [-1, 1] is the other accepted form and must give the same features
    as_float = torch.from_numpy(real).permute(0, 3, 1, 2).float() / 127.5 - 1
    assert np.allclose(ex(as_float), f, atol=1e-5)
    with pytest.raises(ValueError, match="NHWC"):
        ex(np.zeros((4, 3, 32, 32), np.uint8))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ex(torch.full((4, 3, 32, 32), 3.0))
    res = GM.compare(real, fake, ex, subsets=5)
    assert set(res) >= {"n", "fid", "kid_mean", "kid_std", "extractor", "real", "fake"}
    assert res["n"] == 40 and res["fid"] > 0 and res["real"]["sha1"] != res["fake"]["sha1"]
    assert res["real"] == GM.image_identity(real) and res["extractor"].startswith("injected/")
    same = GM.compare(real, real.copy(), ex, subsets=5)
    assert same["fid"] < 1e-6 and abs(same["kid_mean"]) < 0.05
    with pytest.raises(ValueError, match="same n"):
        GM.compare(real, fake[:10], ex)


def test_cli_on_png_dirs(tmp_path, monkeypatch):
    from PIL import Image

    for name, seed in (("real", 0), ("fake", 1)):
        d = tmp_path / name
        d.mkdir()
        for i, a in enumerate(_imgs(6, seed)):
            Image.fromarray(a).save(d / f"img_{i}.png")
        Image.fromarray(_imgs(1, 9)[0]).save(d / "x_grid.png")  # grids are skipped
    monkeypatch.setattr(GM, "InceptionFeatures", lambda dev: _Ex())
    out = tmp_path / "m.json"
    assert (
        GM.main(
            [
                "--real",
                str(tmp_path / "real"),
                "--fake",
                str(tmp_path / "fake"),
                "--n",
                "6",
                "--subsets",
                "3",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    import json

    r = json.loads(out.read_text())
    assert r["n"] == 6 and r["fid"] >= 0


class _Ex(GM.InceptionFeatures):
    def __init__(self):
        super().__init__("cpu", net=_Net(), batch=8)


def test_cli_refuses_unequal_counts(tmp_path, monkeypatch):
    """FID is biased in n: the CLI must not shorten the larger set to the smaller one."""
    from PIL import Image

    for name, k in (("real", 6), ("fake", 5)):
        d = tmp_path / name
        d.mkdir()
        for i, a in enumerate(_imgs(k, 0)):
            Image.fromarray(a).save(d / f"img_{i}.png")
    monkeypatch.setattr(GM, "InceptionFeatures", lambda dev: _Ex())
    for extra in ([], ["--n", "6"]):
        with pytest.raises(SystemExit, match="--real has 6 PNGs and --fake 5"):
            GM.main(
                ["--real", str(tmp_path / "real"), "--fake", str(tmp_path / "fake"), "--subsets", "3"] + extra
            )
