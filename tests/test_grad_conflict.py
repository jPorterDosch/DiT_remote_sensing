"""CPU tests for experiments/grad_conflict/gradproj.py (no model, no GPU).

Covers: block grouping of LoRA names, flatten/unflatten round trip, cosine stats (NaN for
unreachable blocks), the one-sided projection (conflicting units fixed, agreeing units and the
FM gradient untouched, zero-FM units untouched), and -- rule 6 -- that the verifier FAILS on
an unprojected conflicting pair, so a PASS from it means something.
"""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models import gradproj as G  # noqa: E402


class _Tiny(torch.nn.Module):
    """Two double + two single blocks with LoRA-shaped A/B leaves, names as models/lora.py."""

    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([_Blk("img_attn.qkv") for _ in range(2)])
        self.single_blocks = torch.nn.ModuleList([_Blk("linear1") for _ in range(2)])
        self.other = torch.nn.Linear(2, 2)  # not LoRA, must be ignored


class _Blk(torch.nn.Module):
    def __init__(self, leaf: str):
        super().__init__()
        self.leaf = leaf
        mod = self
        for part in leaf.split(".")[:-1]:
            sub = torch.nn.Module()
            mod.add_module(part, sub)
            mod = sub
        lin = torch.nn.Module()
        lin.A = torch.nn.Parameter(torch.randn(2, 3))
        lin.B = torch.nn.Parameter(torch.randn(4, 2))
        mod.add_module(leaf.split(".")[-1], lin)


def _groups():
    torch.manual_seed(0)
    return G.lora_groups(_Tiny())


def test_block_of_names():
    assert G.block_of("double_blocks.3.img_attn.qkv.A", 19) == 3
    assert G.block_of("single_blocks.9.linear1.B", 19) == 28
    with pytest.raises(ValueError):
        G.block_of("final_layer.linear.weight", 19)


def test_groups_cover_lora_only_in_block_order():
    g = _groups()
    assert list(g) == [0, 1, 2, 3]
    for b, params in g.items():
        assert all(n.endswith((".A", ".B")) for n, _ in params)
        assert len(params) == 2
    assert all("other" not in n for params in g.values() for n, _ in params)


def test_flatten_unflatten_roundtrip_and_none_as_zero():
    g = _groups()
    grads = {n: torch.randn_like(p) for params in g.values() for n, p in params}
    name0 = g[0][0][0]
    grads[name0] = None  # loss did not reach this parameter
    flat = G.flatten(g, grads)
    assert flat[0].numel() == 2 * 3 + 4 * 2 and flat[0][: 2 * 3].abs().sum() == 0
    back = G.unflatten(g, flat)
    for params in g.values():
        for n, p in params:
            want = torch.zeros_like(p) if grads[n] is None else grads[n]
            assert torch.equal(back[n], want.float())


def test_stats_cosine_and_nan_for_unreached_block():
    g = _groups()
    a = OrderedDict((b, torch.randn(14)) for b in g)
    b = OrderedDict((k, v.clone()) for k, v in a.items())
    b[3] = torch.zeros(14)  # alignment cannot reach a block at/after k
    s = G.stats(a, b)
    assert s["blocks"] == [0, 1, 2, 3, "global"]
    assert torch.allclose(s["cos"][:3], torch.ones(3), atol=1e-6)
    assert math.isnan(float(s["cos"][3]))
    assert not math.isnan(float(s["cos"][4]))


def _conflict_pair():
    """Block 0 conflicts (cos < 0), block 1 agrees, block 2 orthogonal, block 3 has zero FM."""
    f = OrderedDict(
        {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([1.0, 0.0]),
            3: torch.tensor([0.0, 0.0]),
        }
    )
    s = OrderedDict(
        {
            0: torch.tensor([-1.0, 1.0]),
            1: torch.tensor([2.0, 1.0]),
            2: torch.tensor([0.0, 1.0]),
            3: torch.tensor([-5.0, 1.0]),
        }
    )
    return s, f


def test_projection_block_scope():
    s, f = _conflict_pair()
    f_before = OrderedDict((b, v.clone()) for b, v in f.items())
    p, hits = G.project_one_sided(s, f, scope="block")
    assert hits.tolist() == [True, False, False, False]
    assert torch.allclose(p[0], torch.tensor([0.0, 1.0]))  # opposing component removed
    assert torch.equal(p[1], s[1]) and torch.equal(p[2], s[2]) and torch.equal(p[3], s[3])
    for b in f:  # FM gradient untouched
        assert torch.equal(f[b], f_before[b])
    ok, cos, units = G.verify_nonneg(p, f)
    assert ok and units == [0, 1, 2, 3]
    assert float(cos[0]) >= 0.0


def test_projection_global_scope():
    s, f = _conflict_pair()
    # global dot: block0 -1 + block1 2 + block2 0 + block3 0 = +1 -> no conflict globally
    p, hits = G.project_one_sided(s, f, scope="global")
    assert hits.tolist() == [False]
    assert all(torch.equal(p[b], s[b]) for b in s)
    s[1] = torch.tensor([-2.0, 1.0])  # now global dot = -3 -> projected
    p, hits = G.project_one_sided(s, f, scope="global")
    assert hits.tolist() == [True]
    ok, cos, _ = G.verify_nonneg(p, f, scope="global")
    assert ok and abs(float(cos[0])) < 1e-6


def test_verifier_fails_on_unprojected_conflict():
    """Rule 6: the gate must be able to fail. Feed it the raw conflicting pair."""
    s, f = _conflict_pair()
    ok, cos, _ = G.verify_nonneg(s, f)
    assert not ok
    assert float(cos[0]) < 0


def test_verifier_tolerance_is_relative():
    s = OrderedDict({0: torch.tensor([1.0, -1e-7])})
    f = OrderedDict({0: torch.tensor([0.0, 1.0])})
    assert G.verify_nonneg(s, f, tol=1e-5)[0]
    assert not G.verify_nonneg(s, f, tol=1e-9)[0]


def test_projection_random_highdim_fp32():
    torch.manual_seed(1)
    f = OrderedDict((b, torch.randn(5000)) for b in range(6))
    s = OrderedDict((b, -f[b] + 0.3 * torch.randn(5000)) for b in range(6))
    p, hits = G.project_one_sided(s, f)
    assert hits.all()
    ok, cos, _ = G.verify_nonneg(p, f)
    assert ok and float(cos.abs().max()) < 1e-5
    # the projection is the minimal change: p - s is parallel to f
    for b in range(6):
        d = p[b] - s[b]
        c = float(d @ f[b]) / (d.norm() * f[b].norm())
        assert abs(abs(c) - 1.0) < 1e-5


def test_flatten_rejects_unknown_name():
    """A gradient dict missing a parameter is a wiring bug, not a zero gradient."""
    g = _groups()
    grads = {n: torch.randn_like(p) for params in g.values() for n, p in params}
    del grads[g[1][0][0]]
    with pytest.raises(KeyError):
        G.flatten(g, grads)


def test_two_autograd_grad_calls_match_separate_backward():
    """diagnostic.Runner.two_grads: g_FM and g_SEM from ONE graph (retain_graph on the first call)
    must equal the gradients of independent backward passes, and a loss that does not reach a
    parameter yields None (-> zeros in flatten)."""
    torch.manual_seed(0)
    lin1, lin2 = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    h = lin1(x)
    loss_a = (lin2(h) ** 2).mean()  # reaches lin1 and lin2
    loss_b = (h**2).mean()  # reaches lin1 only
    params = list(lin1.parameters()) + list(lin2.parameters())
    ga = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, allow_unused=True)
    assert gb[2] is None and gb[3] is None  # lin2 unreached by loss_b
    h2 = lin1(x)
    ref_a = torch.autograd.grad((lin2(h2) ** 2).mean(), params)
    ref_b = torch.autograd.grad((lin1(x) ** 2).mean(), list(lin1.parameters()))
    for g1, g2 in zip(ga, ref_a):
        assert torch.allclose(g1, g2)
    for g1, g2 in zip(gb[:2], ref_b):
        assert torch.allclose(g1, g2)


def test_fm_arm_update_is_exactly_fm_gradient():
    """With w = 0 the combined vector equals g_FM bit for bit (the fm arm applies no semantic part)."""
    torch.manual_seed(2)
    f = OrderedDict((b, torch.randn(100)) for b in range(3))
    sems = OrderedDict((b, torch.randn(100)) for b in range(3))
    combined = OrderedDict((b, f[b] + 0.0 * sems[b]) for b in f)
    assert all(torch.equal(combined[b], f[b]) for b in f)


# ------------------------------------------------------------------ fp32 LoRA-parameter gate
def _fp32_setup():
    params = [torch.randn(4, 3).requires_grad_(True) for _ in range(2)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    return params, opt


def _adam_step(params, opt):
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()


def test_fp32_gate_passes_on_fp32_params():
    params, opt = _fp32_setup()
    G.check_fp32_params(params, opt, where="init")
    _adam_step(params, opt)  # creates fp32 exp_avg / exp_avg_sq
    G.check_fp32_params(params, opt, check_grads=True, where="post")


def test_fp32_gate_fires_on_bf16_param():
    bad = [torch.randn(4, 3).to(torch.bfloat16).requires_grad_(True)]
    with pytest.raises(RuntimeError, match="param 0 is torch.bfloat16"):
        G.check_fp32_params(bad, torch.optim.AdamW(bad, lr=1e-3))


def test_fp32_gate_fires_when_optimizer_steps_other_tensors():
    # the pipeline's finetune-diffusion style: AdamW bound to the (bf16) model tensors instead
    params, _ = _fp32_setup()
    other = [torch.randn(4, 3).requires_grad_(True) for _ in range(2)]
    with pytest.raises(RuntimeError, match="does not step exactly the fp32 LoRA params"):
        G.check_fp32_params(params, torch.optim.AdamW(other, lr=1e-3))


def test_bf16_grad_on_fp32_param_is_refused_by_torch():
    # The gate's check_grads branch is a backstop: torch already refuses a bf16 .grad on an fp32
    # leaf, so the only way bf16 enters the update path is a bf16 parameter or a wrongly bound
    # optimizer (both covered above).
    params, opt = _fp32_setup()
    with pytest.raises(RuntimeError, match="dtype"):
        params[1].grad = torch.zeros(4, 3, dtype=torch.bfloat16)
    params[1].grad = torch.zeros(4, 3)
    G.check_fp32_params(params, opt, check_grads=True)


def test_fp32_gate_fires_on_bf16_optimizer_state():
    params, opt = _fp32_setup()
    _adam_step(params, opt)
    opt.state[params[0]]["exp_avg"] = opt.state[params[0]]["exp_avg"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="optimizer state 'exp_avg'"):
        G.check_fp32_params(params, opt)


def test_lora_fp32_params_match_bf16_compute_and_keep_fp32_grads():
    """LoRALinear(param_dtype=float32) on a bf16 base: forward equals the bf16-parameter layer bit
    for bit (A/B are cast to bf16 before the matmul), while .grad lands in fp32."""
    from models.lora import LoRALinear

    torch.manual_seed(0)
    base = torch.nn.Linear(6, 5).to(torch.bfloat16)
    bf = LoRALinear(base, rank=2, alpha=4.0)
    f32 = LoRALinear(base, rank=2, alpha=4.0, param_dtype=torch.float32)
    with torch.no_grad():
        bf.A.copy_(torch.randn(2, 6).to(torch.bfloat16))
        bf.B.copy_(torch.randn(5, 2).to(torch.bfloat16))
        f32.A.copy_(bf.A.float())
        f32.B.copy_(bf.B.float())
    x = torch.randn(3, 6).to(torch.bfloat16)
    ya, yb = bf(x), f32(x)
    assert ya.dtype == yb.dtype == torch.bfloat16 and torch.equal(ya, yb)
    yb.float().sum().backward()
    assert f32.A.grad.dtype == torch.float32 and f32.B.grad.dtype == torch.float32
    ya.float().sum().backward()
    assert torch.equal(f32.A.grad, bf.A.grad.float())  # same bf16 gradient values, fp32 storage


# ------------------------------------------------------------------ modular alignment targets
from models import align_targets as TG  # noqa: E402


class _FakeTarget(TG.AlignTarget):
    name, dim = "fake", 8

    def patches(self, imgs):
        return torch.ones(imgs.shape[0], 256, self.dim)


def make_fake(device):  # import:-spec factory
    return _FakeTarget()


def test_match_grid_identity_and_resample():
    z = torch.randn(2, 196, 5)
    assert TG.match_grid(z, 196) is z  # DINOv3/16 at 224 px == FLUX 14x14: untouched
    # 16x16 (DINOv2/14 at 224) -> 14x14: shape, and a constant field stays constant
    c = torch.full((2, 256, 5), 3.0)
    out = TG.match_grid(c, 196)
    assert out.shape == (2, 196, 5) and torch.allclose(out, torch.full_like(out, 3.0))
    # a left/right split field keeps its left/right structure after pooling
    g = torch.zeros(1, 16, 16, 1)
    g[:, :, 8:] = 1.0
    o = TG.match_grid(g.reshape(1, 256, 1), 196).reshape(14, 14)
    assert o[:, 0].max() == 0 and o[:, -1].min() == 1


def test_match_grid_refuses_non_square():
    with pytest.raises(RuntimeError, match="square"):
        TG.match_grid(torch.zeros(1, 200, 3), 196)


def test_build_import_spec_and_unknown_family():
    t = TG.build("import:test_grad_conflict:make_fake", "cpu")
    assert (
        isinstance(t, TG.AlignTarget)
        and t.dim == 8
        and t.patches(torch.zeros(3, 3, 8, 8)).shape == (3, 256, 8)
    )
    with pytest.raises(SystemExit, match="unknown alignment-target family"):
        TG.build("clip:ViT-L-14", "cpu")
    assert "dino" in TG.families()


# ------------------------------------------------------------------ fp64 accumulation
def _big_pair(n=2_000_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n, generator=g) * torch.rand(n, generator=g) ** 4  # heavy-tailed, like LoRA grads
    return f, g


def test_stats_cosine_never_exceeds_one_for_nearly_parallel_large_vectors():
    f, g = _big_pair()
    c = f + 0.003 * torch.randn(f.numel(), generator=g)
    cos = float(G.stats(OrderedDict([(0, c)]), OrderedDict([(0, f)]))["cos"][-1])
    assert cos <= 1.0 + 1e-12 and cos > 0.999


def test_projection_residual_far_below_verifier_tol_on_large_vectors():
    f, g = _big_pair(seed=1)
    s = -0.02 * f + 0.01 * torch.randn(f.numel(), generator=g)  # strongly conflicting
    out, hits = G.project_one_sided(OrderedDict([(0, s)]), OrderedDict([(0, f)]))
    ok, cos, _ = G.verify_nonneg(out, OrderedDict([(0, f)]), tol=1e-5)
    assert bool(hits[0]) and ok and abs(float(cos[0])) < 1e-6
