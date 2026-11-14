"""Block-input tap on the vendored GeoCore trunk (tiny random Flux2, CPU, float32).

pytest tests/test_geocore_trunk.py -v
"""

import os
import sys

import pytest
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from models.geocore.flux2 import Flux2  # noqa: E402
from models.geocore.trunk import forward_feat, randomize_zero_init, tiny_params  # noqa: E402

L_TXT, H_LAT, W_LAT, B = 5, 3, 4, 2


def _model_and_inputs(seed=0):
    torch.manual_seed(seed)
    p = tiny_params()
    g = torch.Generator().manual_seed(seed + 1)
    m = randomize_zero_init(Flux2(p), g).eval()  # zero-init adaLN would make every block a no-op
    x = torch.randn(B, p.in_channels, H_LAT, W_LAT, generator=g)
    ids = torch.zeros(H_LAT, W_LAT, 3)
    ids[..., 1] = torch.arange(H_LAT)[:, None]
    ids[..., 2] = torch.arange(W_LAT)[None, :]
    x_ids = ids.reshape(1, -1, 3).repeat(B, 1, 1)
    ctx = torch.randn(B, L_TXT, p.context_in_dim, generator=g)
    ctx_ids = torch.zeros(B, L_TXT, 3)
    ctx_ids[:, :, 0] = torch.arange(L_TXT)
    y = torch.randn(B, p.y_in_dim, generator=g)
    t = torch.full((B,), 0.25)
    null = torch.full((B,), -999.0)
    return m, dict(x=x, x_ids=x_ids, timesteps=t, ctx=ctx, ctx_ids=ctx_ids, y=y, res=null, lon=null, lat=null)


def _reference_inputs(m, kw):
    """Image tokens entering every block, recorded by wrapping forward_kv_extract during Flux2.forward."""
    seen = {}
    n_double = len(m.double_blocks)
    for i, blk in enumerate(m.double_blocks):
        orig = blk.forward_kv_extract

        def wrapped(img, txt, *a, _i=i, _o=orig, **k):
            seen[_i] = img.clone()
            return _o(img, txt, *a, **k)

        blk.forward_kv_extract = wrapped
    for j, blk in enumerate(m.single_blocks):
        orig = blk.forward_kv_extract

        def wrapped(xx, pe, mod, num_txt, *a, _j=j, _o=orig, **k):
            seen[n_double + _j] = xx[:, num_txt:].clone()
            return _o(xx, pe, mod, num_txt, *a, **k)

        blk.forward_kv_extract = wrapped
    with torch.no_grad():
        m.forward(**kw)
    return seen


def test_block_input_matches_reference_every_k():
    m, kw = _model_and_inputs()
    ref = _reference_inputs(*_model_and_inputs())
    ks = list(range(5))
    out = forward_feat(m, ft_indices=ks, **kw)
    assert len(out) == 2 * len(ks)
    for j, k in enumerate(ks):
        assert out[2 * j].shape == (B, H_LAT * W_LAT, 64)
        assert torch.allclose(out[2 * j], ref[k], atol=1e-6), k


def test_subset_of_blocks_matches_full_tap():
    m, kw = _model_and_inputs()
    full = forward_feat(m, ft_indices=[0, 1, 2, 3, 4], **kw)
    sub = forward_feat(m, ft_indices=[1, 4], **kw)
    assert torch.equal(sub[0], full[2]) and torch.equal(sub[2], full[8])


def test_mods_per_block_shape_and_sharing():
    m, kw = _model_and_inputs()
    out = forward_feat(m, ft_indices=[0, 1, 2, 4], **kw)
    mods = out[1::2]
    assert all(md.shape == (B, 3, 64) for md in mods)
    assert torch.equal(mods[0], mods[1])  # all double blocks share one triple
    assert torch.equal(mods[2], mods[3])  # all single blocks share one triple
    assert not torch.equal(mods[1], mods[2])
    vec = m.make_vec(kw["timesteps"], kw["res"], kw["lon"], kw["lat"], kw["y"])
    shift, scale, gate = (
        x.reshape(B, -1) for x in m.double_stream_modulation_img(vec)[0]
    )  # (B, 1, C) chunks
    assert torch.equal(mods[0], torch.stack([shift, scale, gate], dim=1))


def test_early_exit_counts():
    m, kw = _model_and_inputs()
    calls = {"double": 0, "single": 0, "final": 0}
    for name, blocks in (("double", m.double_blocks), ("single", m.single_blocks)):
        for blk in blocks:
            orig = blk.forward_kv_extract

            def wrapped(*a, _o=orig, _n=name, **k):
                calls[_n] += 1
                return _o(*a, **k)

            blk.forward_kv_extract = wrapped
    fo = m.final_layer.forward

    def final(*a, **k):
        calls["final"] += 1
        return fo(*a, **k)

    m.final_layer.forward = final
    forward_feat(m, ft_indices=[3], **kw)  # input of single block 1: both doubles + single 0 run
    assert calls == {"double": 2, "single": 1, "final": 0}
    forward_feat(m, ft_indices=[0], **kw)  # input of double block 0: nothing runs
    assert calls == {"double": 2, "single": 1, "final": 0}
    forward_feat(m, ft_indices=[4], **kw)  # last block's input: everything but the last single block
    assert calls == {"double": 4, "single": 3, "final": 0}


@pytest.mark.parametrize("bad", [[], [2, 1], [5], [-1], [1, 1], [0.5]])
def test_bad_indices_raise(bad):
    m, kw = _model_and_inputs()
    with pytest.raises(ValueError):
        forward_feat(m, ft_indices=bad, **kw)


def test_null_metadata_uses_learned_null_embeddings():
    m, kw = _model_and_inputs()
    a = forward_feat(m, ft_indices=[1], **kw)[0]
    kw2 = {**kw, "res": torch.full((B,), 3.0), "lon": torch.full((B,), 10.0), "lat": torch.full((B,), 20.0)}
    b = forward_feat(m, ft_indices=[1], **kw2)[0]
    assert not torch.allclose(a, b)


def test_zero_init_model_is_degenerate_and_helper_fixes_it():
    """Documents why randomize_zero_init exists: a fresh Flux2 has zero adaLN gates, so every block
    input equals img_in(x) and the mods are all zero."""
    torch.manual_seed(0)
    p = tiny_params()
    m = Flux2(p).eval()
    _, kw = _model_and_inputs()
    out = forward_feat(m, ft_indices=[0, 4], **kw)
    assert torch.equal(out[0], out[2]) and torch.count_nonzero(out[1]) == 0
    randomize_zero_init(m, torch.Generator().manual_seed(3))
    out = forward_feat(m, ft_indices=[0, 4], **kw)
    assert not torch.equal(out[0], out[2]) and torch.count_nonzero(out[1]) > 0
