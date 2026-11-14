"""GeoCoreModel adapter contract on a tiny Flux2 + stub VAE (CPU, float32).

pytest tests/test_geocore_adapter.py -v
"""

import os
import sys
import types

import pytest
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from models.geocore import adapter as A  # noqa: E402
from models.geocore.flux2 import Flux2  # noqa: E402
from models.geocore.trunk import randomize_zero_init, tiny_params  # noqa: E402

IN_CH = 8


class StubVAE:
    """Deterministic 'encoder': average-pool 16x16 pixel blocks, tile to IN_CH channels."""

    def encode(self, x):
        b, c, h, w = x.shape
        pooled = x.reshape(b, c, h // 16, 16, w // 16, 16).mean(dim=(3, 5))  # (b, 3, h/16, w/16)
        return torch.cat([pooled, pooled, pooled[:, :2]], dim=1)  # (b, 8, h/16, w/16)

    def eval(self):
        return self


def _adapter(seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 7)
    m = randomize_zero_init(Flux2(tiny_params(in_channels=IN_CH)), g).eval()
    ctx = torch.randn(1, 5, 16, generator=g)
    ctx_ids = torch.zeros(1, 5, 3)
    ctx_ids[0, :, 0] = torch.arange(5)
    y = torch.randn(1, 8, generator=g)
    return A.GeoCoreModel.from_parts(m, StubVAE(), ctx, ctx_ids, y, device="cpu")


def _img(seed, size=32):
    return torch.rand(3, size, size, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def test_extract_raw_token_grid_224():  # Review Focus 1: 224 px -> 14x14 tokens
    m = _adapter()
    f, ada, lat, eps = m.extract_raw(_img(0, 224), timestep=260, block_idx=[0, 3], ensemble_size=2,
                                     generator=torch.Generator().manual_seed(0))  # fmt: skip
    assert f.shape == (2, 64, 14, 14) and ada.shape == (2, 3, 64)
    assert lat.shape == (2, IN_CH, 14, 14) and eps.shape == lat.shape
    assert torch.isfinite(f).all()


def test_extract_raw_int_k_shapes():
    f, ada, lat, eps = _adapter().extract_raw(_img(1), timestep=100, block_idx=2, ensemble_size=1)
    assert f.shape == (1, 64, 2, 2) and ada.shape == (1, 3, 64) and lat.shape == (1, IN_CH, 2, 2)


def test_int_k_equals_list_slice():
    m = _adapter()
    img = _img(2)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    f_int, a_int, *_ = m.extract_raw(img, 340, 2, 1, generator=g1)
    f_lst, a_lst, *_ = m.extract_raw(img, 340, [1, 2, 4], 1, generator=g2)
    assert torch.allclose(f_int[0], f_lst[1]) and torch.allclose(a_int[0], a_lst[1])


def test_latent_and_noise_reuse_across_timesteps():
    m = _adapter()
    img = _img(4)
    f1, _, lat, eps = m.extract_raw(img, 100, [1], 2, generator=torch.Generator().manual_seed(1))
    f2, _, lat2, eps2 = m.extract_raw(img, 500, [1], 2, latents=lat, noise=eps)
    assert torch.equal(lat, lat2) and torch.equal(eps, eps2)
    assert not torch.allclose(f1, f2)


def test_member_mean_equals_manual():
    m = _adapter()
    img = _img(5)
    f, ada, lat, eps = m.extract_raw(img, 420, [2], 2, generator=torch.Generator().manual_seed(9))
    f0, a0, *_ = m.extract_raw(img, 420, [2], 1, latents=lat[:1], noise=eps[:1])
    f1, a1, *_ = m.extract_raw(img, 420, [2], 1, latents=lat[1:], noise=eps[1:])
    assert torch.allclose(f, (f0 + f1) / 2, atol=1e-6) and torch.allclose(ada, (a0 + a1) / 2, atol=1e-6)
    assert not torch.equal(eps[0], eps[1])


def test_generator_isolation_reproducible():
    m = _adapter()
    a = m.extract_raw(_img(6), 260, [1], 2, generator=torch.Generator().manual_seed(5))
    torch.manual_seed(123)  # global RNG must not matter
    b = m.extract_raw(_img(6), 260, [1], 2, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[3], b[3])


def test_mods_shared_within_family():
    _, ada, *_ = _adapter().extract_raw(_img(7), 180, [0, 1, 2, 4], 1)
    assert torch.equal(ada[0], ada[1]) and torch.equal(ada[2], ada[3]) and not torch.equal(ada[1], ada[2])


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(guidance=1.0), "guidance"),
        (dict(caption="a photo"), "prompt"),
    ],
)
def test_extract_raw_pins(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _adapter().extract_raw(_img(8), 100, [1], 1, **kw)


def test_extract_raw_rejects_bad_image_size():
    with pytest.raises(ValueError, match="multiples of 16"):
        _adapter().extract_raw(_img(9, 40), 100, [1], 1)


def test_unsupported_paths_raise():
    m = _adapter()
    for fn in (m.extract, m.extract_inversion, m.roundtrip):
        with pytest.raises(NotImplementedError):
            fn(_img(0), timesteps=[100, 200], num_inversion_steps=10, block_idx=1)


def _cfg(**over):
    base = dict(lora_checkpoint="", extraction_mode="ONESHOT", guidance_scale=0.0, img_size=[224, 224],
                k=[4, 8, 12, 16, 20, 24, 28, 31], geocore_ckpt="/nonexistent/GeoCore-9B",
                geocore_vae="/nonexistent/vae.safetensors", device="cpu")  # fmt: skip
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize(
    "over,msg",
    [
        (dict(guidance_scale=1.0), "guidance_scale"),
        (dict(lora_checkpoint="x.pt"), "lora"),
        (dict(extraction_mode="INVERSION"), "one-shot"),
        (dict(img_size=[200, 200]), "multiple of 16"),
        (dict(k=[4, 40]), "k must"),
        (dict(geocore_ckpt=""), "required"),
    ],
)
def test_init_guards_raise_before_any_file_access(over, msg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("sha256 verification ran before the config guards")

    monkeypatch.setattr(A, "_verify_sha256", boom)
    with pytest.raises(ValueError, match=msg):
        A.GeoCoreModel(_cfg(**over), ["a"])


def test_sha256_mismatch_raises(tmp_path):
    p = tmp_path / "model.safetensors.index.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        A.GeoCoreModel(_cfg(geocore_ckpt=str(tmp_path)), ["a"])


def test_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        A.GeoCoreModel(_cfg(geocore_ckpt=str(tmp_path)), ["a"])


def test_provenance_meta_keys():
    meta = _adapter().provenance_meta()
    assert set(meta) == {"hidden_size", "num_double_blocks", "num_blocks", "vae_encode", "text", "geo_null",
                         "geocore_commit", "geocore_ckpt_sha256", "geocore_vae_sha256"}  # fmt: skip
    assert meta["hidden_size"] == 4096 and meta["num_double_blocks"] == 8 and meta["num_blocks"] == 32
    assert meta["vae_encode"] == "mean" and meta["text"] == "" and meta["geo_null"] == -999.0
    assert meta["geocore_commit"] == "17406c03a9bd090ff1570809bf6bd7f4a35e2359"


def test_registered():
    from registry import MODELS

    assert MODELS["geocore"] is A.GeoCoreModel
