"""No-GPU tests for FLUX block feature capture (tiny random blocks, CPU, float64).

Guards the 2026-10-05 double-block fix: DoubleStreamBlock.forward_feat used to return a
mid-block, post-attention, already-modulated activation and no adaLN mod, so double-block
features were not the single blocks' quantity and up_ft lost its [feat, mod] pairing.

    pytest tests/test_flux_blocks.py -v
"""

import os
import sys

import pytest
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

from models.flux.model import Flux, FluxParams  # noqa: E402
from models.flux.modules.layers import DoubleStreamBlock, EmbedND, SingleStreamBlock  # noqa: E402

H, HEADS, AXES = 64, 4, [4, 6, 6]  # head_dim 16 = sum(AXES)
L_TXT, L_IMG = 3, 8


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, L_IMG, H, generator=g, dtype=torch.float64)
    txt = torch.randn(1, L_TXT, H, generator=g, dtype=torch.float64)
    vec = torch.randn(1, H, generator=g, dtype=torch.float64)
    ids = torch.randint(0, 4, (1, L_TXT + L_IMG, 3), generator=g).double()
    pe = EmbedND(dim=H // HEADS, theta=10_000, axes_dim=AXES)(ids)
    return img, txt, vec, pe


@pytest.mark.parametrize("return_feat", [True, False])
def test_double_block_feat_is_block_input_and_propagation_unchanged(return_feat):
    torch.manual_seed(0)
    block = DoubleStreamBlock(H, HEADS, mlp_ratio=2.0, qkv_bias=True).double().eval()
    img, txt, vec, pe = _inputs()
    with torch.no_grad():
        ref_img, ref_txt = block.forward(img, txt, vec, pe)
        out_img, out_txt, feat, mod = block.forward_feat(img, txt, vec, pe, return_feat=return_feat)
        mod1, _ = block.img_mod(vec)
    assert torch.equal(out_img, ref_img) and torch.equal(out_txt, ref_txt)  # no behaviour change
    if return_feat:
        assert torch.equal(feat, img)  # block INPUT, same definition as single blocks
    else:
        assert feat is None
    for f in ("shift", "scale", "gate"):
        assert torch.equal(getattr(mod, f), getattr(mod1, f))  # the img_norm1 adaLN triple


def test_single_block_feat_unchanged():
    torch.manual_seed(0)
    block = SingleStreamBlock(H, HEADS, mlp_ratio=2.0).double().eval()
    img, txt, vec, pe = _inputs()
    x = torch.cat((txt, img), 1)
    with torch.no_grad():
        ref = block.forward(x, vec, pe)
        out, feat, mod = block.forward_feat(x, (vec, vec, vec), pe, return_feat=True)
    assert torch.equal(out, ref) and torch.equal(feat, x)


def _tiny_flux():
    torch.manual_seed(0)
    p = FluxParams(
        in_channels=8,
        vec_in_dim=16,
        context_in_dim=12,
        hidden_size=H,
        mlp_ratio=2.0,
        num_heads=HEADS,
        depth=2,
        depth_single_blocks=2,
        axes_dim=AXES,
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )
    return Flux(p).double().eval()


def _trunk_inputs():
    g = torch.Generator().manual_seed(1)
    return dict(
        img=torch.randn(1, L_IMG, 8, generator=g, dtype=torch.float64),
        img_ids=torch.randint(0, 4, (1, L_IMG, 3), generator=g).double(),
        txt=torch.randn(1, L_TXT, 12, generator=g, dtype=torch.float64),
        txt_ids=torch.zeros(1, L_TXT, 3, dtype=torch.float64),
        timesteps=torch.tensor([0.3], dtype=torch.float64),
        y=torch.randn(1, 16, generator=g, dtype=torch.float64),
        guidance=torch.tensor([1.0], dtype=torch.float64),
    )


def test_trunk_returns_feat_mod_pair_for_every_block_and_same_velocity():
    flux, kw = _tiny_flux(), _trunk_inputs()
    with torch.no_grad():
        ref = flux.forward(**kw)
        up_ft = flux.forward_feat(**kw, ft_indices=[0, 1, 2, 3])  # 2 double + 2 single
        pred, up_ft_v = flux.forward_velocity_feat(**kw, ft_indices=[1, 3])
    assert len(up_ft) == 8
    for j in range(4):
        feat, mod = up_ft[2 * j], up_ft[2 * j + 1]
        assert feat.shape == (1, L_IMG, H)  # image tokens only, text sliced off
        assert all(hasattr(mod, f) for f in ("shift", "scale", "gate"))
    assert torch.equal(pred, ref)  # requesting features never changes the velocity
    assert torch.equal(up_ft_v[0], up_ft[2]) and torch.equal(up_ft_v[2], up_ft[6])
    # early exit at a double block still yields its pair
    assert len(flux.forward_feat(**kw, ft_indices=[0])) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_double_block_stamp_default_unchanged():
    from config_types import DOUBLE_BLOCK_FEATURE, double_block_stamp

    assert double_block_stamp(28) == {}
    assert double_block_stamp([4, 28]) == {"double_block_feature": DOUBLE_BLOCK_FEATURE}
    assert double_block_stamp([19, 56]) == {}


def test_double_block_stamp_num_double():
    from config_types import DOUBLE_BLOCK_FEATURE, double_block_stamp

    assert double_block_stamp([8, 12], num_double=8) == {}
    assert double_block_stamp([4, 8], num_double=8) == {"double_block_feature": DOUBLE_BLOCK_FEATURE}
    assert double_block_stamp(7, num_double=8) == {"double_block_feature": DOUBLE_BLOCK_FEATURE}
