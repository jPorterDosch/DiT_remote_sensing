"""finetune-diffusion semantic alignment (tasks/align.py) and its run.py wiring, on CPU with a
57-block stand-in for FLUX: the two-gradient combine of every arm against plain autograd, the
executed verifiers, the statistics / identity records a resume relies on, the Git-10M dataset
wrapper, and the rule-9 invariants of RunConfig (existing run names must not move)."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_types import NUM_BLOCKS, NUM_DOUBLE_BLOCKS, AlignConfig, Git10MConfig  # noqa: E402
from models import gradproj as GP  # noqa: E402
from models.align_targets import AlignTarget  # noqa: E402
from tasks import align as AL  # noqa: E402

D, L, DIM = 6, 5, 3  # hidden size, tokens, target dim


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(D, D)
        self.A = torch.nn.Parameter(torch.randn(2, D) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(D, 2) * 0.1)

    def forward(self, x):
        return x + torch.tanh(self.lin(x)) + (x @ self.A.t()) @ self.B.t()


class _FakeFlux(torch.nn.Module):
    """double_blocks / single_blocks with LoRA-named A/B tensors; hidden_size as FLUX."""

    def __init__(self):
        super().__init__()
        self.hidden_size = D
        self.double_blocks = torch.nn.ModuleList([_Block() for _ in range(NUM_DOUBLE_BLOCKS)])
        self.single_blocks = torch.nn.ModuleList([_Block() for _ in range(NUM_BLOCKS - NUM_DOUBLE_BLOCKS)])
        for n, p in self.named_parameters():
            p.requires_grad = n.endswith((".A", ".B"))

    def forward(self, x):
        for b in list(self.double_blocks) + list(self.single_blocks):
            x = b(x)
        return x


class _FakeTarget(AlignTarget):
    name, dim = "fake", DIM

    def __init__(self):
        self.proj = torch.randn(3, DIM)

    @torch.no_grad()
    def patches(self, imgs):
        return torch.einsum("bli,id->bld", imgs, self.proj)


def _state(arm, w=0.5, seed=0):
    torch.manual_seed(seed)
    flux = _FakeFlux()
    cfg = AlignConfig(arm=arm, weight=w, proj_hidden=8, feat="raw")  # the stand-in has no modulation
    st = AL.AlignState(flux, cfg, k=28, device="cpu", target=_FakeTarget())
    x = torch.randn(2, L, 3)  # "images": (B, L, 3) so the fake target can project them
    tok = torch.randn(2, L, D)
    v_target = torch.randn(2, L, D)
    return flux, st, x, tok, v_target


def _losses(flux, st, x, tok, v_target):
    out = flux(tok)
    return ((out - v_target) ** 2).mean(), st.align_loss(out, x)


def _ref_grads(flux, st, x, tok, v_target, w, n_micro=2):
    """Gradient of mean_micro(flow + w * align) w.r.t. the LoRA tensors, by plain autograd."""
    tot = 0.0
    for _ in range(n_micro):
        fl, al = _losses(flux, st, x, tok, v_target)
        tot = tot + (fl + w * al) / n_micro
    return torch.autograd.grad(tot, st.params)


def test_naive_arm_grad_equals_plain_autograd_of_the_weighted_sum():
    flux, st, x, tok, v = _state("naive", w=0.7)
    ref = _ref_grads(flux, st, x, tok, v, 0.7)
    stats = AL.AlignStats(3, st.n_rows)
    for _ in range(2):
        fl, al = _losses(flux, st, x, tok, v)
        st.accumulate(fl / 2, al / 2)
    assert st.n_micro == 2
    summary = st.apply(stats, 0)
    for p, r in zip(st.params, ref):
        assert p.grad.dtype == torch.float32 and torch.allclose(p.grad, r, atol=1e-6)
    assert np.isfinite(stats.st["cos_pre"][0, :-1]).all() and np.isclose(stats.sc["w_eff"][0], 0.7)
    assert set(summary) == {"cos_lt_k_mean", "frac_neg", "global_cos", "nproj"} and summary["nproj"] == 0


def test_fm_arm_grad_is_exactly_the_fm_gradient_and_projector_still_trains():
    flux, st, x, tok, v = _state("fm", w=0.5)
    assert st.w == 0.0
    fl, al = _losses(flux, st, x, tok, v)
    ref = torch.autograd.grad(fl, st.params, retain_graph=True)
    fl, al = _losses(flux, st, x, tok, v)
    st.accumulate(fl, al)
    stats = AL.AlignStats(1, st.n_rows)
    st.apply(stats, 0)
    for p, r in zip(st.params, ref):
        assert torch.equal(p.grad, r)
    # the semantic gradient is still MEASURED (stats) and the projector still receives it
    assert np.isfinite(stats.st["norm_sem"][0, :-1]).all() and (stats.st["norm_sem"][0, :-1] > 0).any()
    before_proj = [p.detach().clone() for p in st.proj_params]
    opt = torch.optim.AdamW(st.params, lr=1e-3, weight_decay=0.0)
    st.pre_step_gate(opt, 0)
    before = [p.detach().clone() for p in st.params]
    opt.step()
    st.after_step(stats, 0, before, opt)
    assert any(not torch.equal(a, b) for a, b in zip(before_proj, st.proj_params))
    assert np.isfinite(stats.sc["cos_update"][0]) and st.acc_fm is None  # cleared for the next step


def test_proj_arm_removes_an_opposing_semantic_gradient_and_verifies():
    """align loss = -c * flow loss gives g_SEM = -c g_FM in every block (pure conflict): the
    one-sided projection must zero it, so the applied gradient is exactly g_FM and every block
    is flagged projected; the verifier passes."""
    flux, st, x, tok, v = _state("proj", w=0.5)
    fl, al = _losses(flux, st, x, tok, v)
    ref = torch.autograd.grad(fl, st.params, retain_graph=True)
    fl, _ = _losses(flux, st, x, tok, v)
    st.accumulate(fl, -3.0 * fl)
    stats = AL.AlignStats(1, st.n_rows)
    summary = st.apply(stats, 0)
    assert summary["nproj"] == NUM_BLOCKS and stats.st["projected"][0, :-1].all()
    assert stats.st["conflict"][0, :-1].all()
    for p, r in zip(st.params, ref):  # fp64 projection, one fp32 rounding of a ~1e1 value
        assert torch.allclose(p.grad, r, rtol=1e-5, atol=1e-5)
    assert stats.sc["applied_cos_min"][0] >= -st.cfg.applied_tol


def test_apply_raises_on_non_finite_gradient_and_before_accumulate():
    flux, st, x, tok, v = _state("naive")
    stats = AL.AlignStats(1, st.n_rows)
    with pytest.raises(RuntimeError, match="before any accumulate"):
        st.apply(stats, 0)
    fl, al = _losses(flux, st, x, tok, v)
    st.accumulate(fl, al)
    st.acc_fm[0][0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite gradient in block 0"):
        st.apply(stats, 0)


def test_projection_verifier_failure_is_fatal(monkeypatch):
    flux, st, x, tok, v = _state("proj")
    fl, al = _losses(flux, st, x, tok, v)
    st.accumulate(fl, al)
    monkeypatch.setattr(
        GP, "verify_nonneg", lambda *a, **k: (False, torch.full((NUM_BLOCKS,), -1.0), list(range(NUM_BLOCKS)))
    )
    with pytest.raises(RuntimeError, match="PROJECTION VERIFY FAIL"):
        st.apply(AL.AlignStats(1, st.n_rows), 0)


def test_measure_returns_per_t_rows_and_warmup_trains_the_projector():
    flux, st, x, tok, v = _state("naive")
    batches = [(x, tok), (x * 2, tok)]
    out = st.measure(
        batches, lambda imgs, lat, t, noise: _losses(flux, st, imgs, lat, v), [100, 500], seed=0, tag="t"
    )
    assert (
        set(out) == {100, 500}
        and out[100]["cos"].shape == (2, st.n_rows)
        and out[100]["agg_cos"].shape == (st.n_rows,)
    )
    before = [p.detach().clone() for p in st.proj_params]
    st.warmup(iter([(flux(tok).detach(), x)] * 3))
    assert any(not torch.equal(a, b) for a, b in zip(before, st.proj_params))


def test_batch_shuffle_control_rolls_the_targets_and_refuses_single_rows():
    """The shuffled-teacher control aligns row b to the target of row b-1: its loss equals the
    real loss on the images rolled by one (same targets, other pairing); a 1-row batch is
    refused (it would roll onto itself); the setting is recorded in describe()."""
    flux, st, x, tok, _ = _state("proj")
    cfg = AlignConfig(arm="proj", weight=0.5, proj_hidden=8, feat="raw", target_shuffle="batch")
    sh = AL.AlignState(flux, cfg, k=28, device="cpu", target=st.target)
    sh.projector.load_state_dict(st.projector.state_dict())
    out = flux(tok).detach()
    assert torch.allclose(sh.align_loss(out, x), st.align_loss(out, x.roll(1, dims=0)))
    assert not torch.allclose(sh.align_loss(out, x), st.align_loss(out, x))
    with pytest.raises(ValueError, match="batches of >= 2"):
        sh.align_loss(out[:1], x[:1])
    assert sh.describe()["target_shuffle"] == "batch" and st.describe()["target_shuffle"] == ""
    with pytest.raises(ValueError, match="target_shuffle"):
        AL.AlignState(
            flux, AlignConfig(arm="proj", feat="raw", target_shuffle="rows"), 28, "cpu", target=st.target
        )


def test_align_state_refuses_bad_arm_scope_and_projector_arch():
    flux = _FakeFlux()
    with pytest.raises(ValueError, match="align.arm"):
        AL.AlignState(flux, AlignConfig(arm="xx", feat="raw"), 28, "cpu", target=_FakeTarget())
    with pytest.raises(ValueError, match="proj_scope"):
        AL.AlignState(
            flux, AlignConfig(arm="proj", proj_scope="row", feat="raw"), 28, "cpu", target=_FakeTarget()
        )
    with pytest.raises(ValueError, match="only 'mlp'"):
        AL.AlignState(
            flux, AlignConfig(arm="proj", proj_arch="linear", feat="raw"), 28, "cpu", target=_FakeTarget()
        )


def test_stats_state_roundtrip_and_shape_guard():
    s = AL.AlignStats(4, 58)
    s.sc["flow_loss"][1] = 0.5
    s.st["projected"][2, 3] = True
    r = AL.AlignStats.from_state(s.state(), 4, 58)
    assert r.sc["flow_loss"][1] == 0.5 and r.st["projected"][2, 3]
    with pytest.raises(ValueError, match="shape"):
        AL.AlignStats.from_state(s.state(), 5, 58)
    assert "step/cos_pre" in s.npz_fields() and "step/flow_loss" in s.npz_fields()


# ------------------------------------------------------------------------- run.py wiring
@pytest.fixture(scope="module")
def run_mod():
    return importlib.import_module("run")


def _cfg(run_mod, **kw):
    base = dict(
        task="finetune-diffusion",
        dataset=run_mod.DatasetConfig(name="m_eurosat", path="data/m_eurosat_rgb"),
        t=[100, 180, 260, 340, 420, 500, 580],
        guidance_scale=1.0,
        mim_loss_weight=0.0,
        mask_ratio=0.0,
        k=28,
    )
    base.update(kw)
    return run_mod.RunConfig(**base)


def test_config_hashes_of_existing_runs_are_unchanged_by_the_new_fields(run_mod):
    """Rule 9: adding align / git10m to RunConfig must not repoint any existing run directory.
    Pinned 2026-10-14 from the code before the fields existed."""
    assert run_mod.RunConfig().config_hash() == "c752d737"
    assert _cfg(run_mod).config_hash() == "ecc99112"
    assert _cfg(run_mod).make_run_name() == "m_eurosat_flux_ecc99112+42"
    e = _cfg(run_mod, task="extract", mim_loss_weight=1.0, mask_ratio=0.75)
    assert e.config_hash() == "eacc27eb" and e.make_run_name() == "m_eurosat_flux_eacc27eb+42"


def test_shuffle_control_has_its_own_identity_and_leaves_the_real_arms_alone(run_mod):
    """Off, the identity dict is the pre-2026-10-16 one (no key at all): the run names of the
    arms in flight and their resume check do not move. On, the hash differs, and the control
    is refused without an arm or with 1-row batches (rule 10)."""
    real = _cfg(run_mod, align=AlignConfig(arm="proj"), batch_size=2)
    ctrl = _cfg(run_mod, align=AlignConfig(arm="proj", target_shuffle="batch"), batch_size=2)
    assert (
        "target_shuffle" not in real.align.identity() and ctrl.align.identity()["target_shuffle"] == "batch"
    )
    assert real.config_hash() == ctrl.config_hash() and real.align_hash() != ctrl.align_hash()
    with pytest.raises(ValueError, match="target_shuffle"):
        _cfg(run_mod, align=AlignConfig(arm="", target_shuffle="batch"), batch_size=2)
    with pytest.raises(ValueError, match="target_shuffle"):
        _cfg(run_mod, align=AlignConfig(arm="proj", target_shuffle="rows"), batch_size=2)
    with pytest.raises(ValueError, match="batch-size"):
        _cfg(run_mod, align=AlignConfig(arm="proj", target_shuffle="batch"), batch_size=1)


def test_gen_eval_task_reads_git10m_without_an_arm(run_mod):
    """run.py --task gen-eval scores a checkpoint's generation against the git10m holdout: no
    align.arm (the checkpoint carries it); any other non-training task is still refused."""
    g = run_mod.Git10MConfig(index="x.npz")
    ds = run_mod.DatasetConfig(name="git10m", path="arrow")
    _cfg(run_mod, task="gen-eval", dataset=ds, git10m=g, align=AlignConfig(fid_n=8))
    with pytest.raises(ValueError, match="training corpus"):
        _cfg(run_mod, task="extract", dataset=ds, git10m=g)
    with pytest.raises(ValueError, match="set align.arm"):
        _cfg(run_mod, dataset=ds, git10m=g)


def test_lora_tag_separates_a_shuffled_control_from_its_arm():
    """The cache filename / weights pin of a shuffled-teacher control is not the arm's."""
    from tasks.extraction import lora_tag

    assert lora_tag(None) == "finetune"
    assert lora_tag({"arm": "proj"}) == "proj"
    assert lora_tag({"arm": "proj", "target_shuffle": ""}) == "proj"
    assert lora_tag({"arm": "proj", "target_shuffle": "batch"}) == "proj-shufbatch"


def test_gen_eval_honors_no_fixed_cond(run_mod):
    """gen-eval never reaches Featurizer4Eval.forward, so FIXED_COND_T must be refused, not stamped."""
    assert "FIXED_COND_T" not in run_mod._task_honors("gen-eval")
    assert "FIXED_COND_T" in run_mod._task_honors("extract")


def test_gen_eval_task_is_registered(run_mod):
    """run.py's validation, the git10m dataset wrapper and gen_eval.sbatch all name task 'gen-eval';
    until 2026-11-12 the task module (src/tasks/gen_eval.py) existed only in another worktree, so
    `run.py --task gen-eval` on this branch stopped at "Unknown task" (CLAUDE.md rule 17)."""
    assert "gen-eval" in run_mod.TASKS
    sbatch = os.path.join(ROOT, "experiments", "isaac", "git10m", "gen_eval.sbatch")
    with open(sbatch) as f:
        assert "--task gen-eval" in f.read()


def test_t_weights_need_an_arm(run_mod):
    """Rule 10: align.t_weights is honored only by the aligned arms (the plain finetune samples
    the grid uniformly), so set-but-unhonored must raise instead of mislabeling the run."""
    _cfg(run_mod, align=AlignConfig(arm="proj", t_weights=[1] * 6 + [0.5]))
    with pytest.raises(ValueError, match="t_weights"):
        _cfg(run_mod, align=AlignConfig(arm="", t_weights=[1] * 6 + [0.5]))


def test_align_identity_lives_in_the_run_name_suffix_not_the_hash(run_mod):
    a = _cfg(run_mod, align=AlignConfig(arm="proj"))
    b = _cfg(run_mod, align=AlignConfig(arm="proj", weight=0.1))
    c = _cfg(run_mod, align=AlignConfig(arm="proj", resume="", ckpt_every=500))
    assert a.config_hash() == b.config_hash() == "ecc99112"
    assert a.make_run_name().startswith("m_eurosat_flux_ecc99112+42+proj-")
    assert a.make_run_name() != b.make_run_name()  # the weight is identity
    assert a.make_run_name() == c.make_run_name()  # the checkpoint cadence is not
    g = _cfg(
        run_mod,
        dataset=run_mod.DatasetConfig(name="git10m", path="/x/arrow"),
        align=AlignConfig(arm="fm"),
        git10m=Git10MConfig(index="/x/index_all_cities_unique.npz"),
    )
    # dataset.path is hashed (it always was; the name is not); only the align / git10m blocks are not
    plain = _cfg(run_mod, dataset=run_mod.DatasetConfig(name="m_eurosat", path="/x/arrow"))
    assert g.config_hash() == plain.config_hash() != "ecc99112"
    assert g.make_run_name().startswith(f"git10m_flux_{plain.config_hash()}+42+fm-")
    assert (
        g.make_run_name()
        != _cfg(
            run_mod,
            dataset=run_mod.DatasetConfig(name="git10m", path="/x/arrow"),
            align=AlignConfig(arm="fm"),
            git10m=Git10MConfig(index="/x/index_all_unique.npz"),
        ).make_run_name()
    )


def test_runconfig_guards_for_align_and_git10m(run_mod):
    with pytest.raises(ValueError, match="only honored by task='finetune-diffusion'"):
        _cfg(run_mod, task="extract", mim_loss_weight=1.0, mask_ratio=0.75, align=AlignConfig(arm="proj"))
    with pytest.raises(ValueError, match="align.arm must be one of"):
        _cfg(run_mod, align=AlignConfig(arm="repa"))
    with pytest.raises(ValueError, match="has no labels"):
        _cfg(
            run_mod,
            dataset=run_mod.DatasetConfig(name="git10m", path="/x"),
            git10m=Git10MConfig(index="i.npz"),
        )
    with pytest.raises(ValueError, match="needs git10m.index"):
        _cfg(run_mod, dataset=run_mod.DatasetConfig(name="git10m", path="/x"), align=AlignConfig(arm="fm"))
    with pytest.raises(ValueError, match="only honored by dataset.name='git10m'"):
        _cfg(run_mod, git10m=Git10MConfig(index="i.npz"))
    with pytest.raises(ValueError, match="align.resume does not exist"):
        _cfg(run_mod, align=AlignConfig(arm="fm", resume="/nonexistent/ckpt.pt"))
    with pytest.raises(ValueError, match="meas_t"):
        _cfg(run_mod, align=AlignConfig(arm="fm", meas_t=[500, 100]))


def test_resume_identity_changes_with_sampler_relevant_settings(run_mod):
    """A checkpoint resumed under a different batch size / accumulation / seed / index / hflip
    would silently re-position the PermutationSampler, so each must change the record."""
    kw = dict(
        dataset=run_mod.DatasetConfig(name="git10m", path="/x/arrow"),
        align=AlignConfig(arm="proj"),
        git10m=Git10MConfig(index="/x/i.npz"),
        batch_size=4,
        gradient_accumulation_steps=2,
    )
    data = {"manifest_sha256": {"a": "1"}, "rows": 5}
    ref = AL.resume_identity(_cfg(run_mod, **kw), data)
    assert ref == AL.resume_identity(_cfg(run_mod, **kw), dict(data))
    assert ref["grad_accum"] == 2 and ref["run_name"].startswith("git10m_flux_")
    for k, v in (
        ("batch_size", 8),
        ("gradient_accumulation_steps", 1),
        ("seed", 1),
        ("git10m", Git10MConfig(index="/x/i.npz", hflip=False)),
        ("git10m", Git10MConfig(index="/x/other.npz")),
        ("align", AlignConfig(arm="naive")),
        ("max_train_steps", 700),
    ):
        assert AL.resume_identity(_cfg(run_mod, **{**kw, k: v}), data) != ref, k
    assert AL.resume_identity(_cfg(run_mod, **kw), {**data, "manifest_sha256": {"a": "2"}}) != ref
    # the resume path and checkpoint cadence are NOT identity (a resume must match its own run)
    same = AL.resume_identity(_cfg(run_mod, **{**kw, "align": AlignConfig(arm="proj", ckpt_every=9)}), data)
    assert same == ref


# ------------------------------------------------------------------------- git10m wrapper
def _load_test_git10m():
    spec = importlib.util.spec_from_file_location(
        "test_git10m", os.path.join(ROOT, "tests", "test_git10m.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_git10m_wrapper_is_registered_label_free_and_positions_the_sampler(tmp_path):
    import registry
    from datasets import git10m as G

    T = _load_test_git10m()
    stream = tmp_path / "stream" / "train"
    stream.mkdir(parents=True)
    names = T._stream_shard(str(stream / G.shard_name(0)))
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    T._meta(str(meta_dir / "meta-00000.parquet"), names)
    arrow = tmp_path / "arrow"
    G.fetch_convert(range(0, 1), str(tmp_path / "stream"), str(arrow), str(meta_dir), False, True)
    ret = [(0, r, names[r], "10_1_1", "google_tile") for r in range(T.N_ROWS)]
    d = str(tmp_path / "m")
    T._manifests(d, ret, [])
    idx = str(tmp_path / "idx.npz")
    G.build_index(d, idx, include_unassessable=False, shards=range(0, 1))

    assert registry.DATASETS["git10m"] is G.Git10MWrapper
    mk = lambda **kw: SimpleNamespace(  # noqa: E731
        task="finetune-diffusion",
        git10m=Git10MConfig(**{"index": idx, "holdout_every": 10, **kw.pop("g", {})}),
        label_fraction=1.0,
        img_size=[32, 32],
        dataset=SimpleNamespace(name="git10m", path=str(arrow)),
        seed=3,
        max_samples=kw.pop("max_samples", 8),
        batch_size=2,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="smoke index"):
        G.Git10MWrapper(mk(max_samples=None))  # a shard-subset index needs --max-samples
    w = G.Git10MWrapper(mk())
    assert w.category_list == [] and len(w.train) == 8 and w.val_full.which == "val"
    loaders = w.get_data(mk())
    assert set(loaders) == {"train", "val"}
    full = [b["index"].tolist() for b in w.train_loader(mk(), 0)]
    resumed = [b["index"].tolist() for b in w.train_loader(mk(), 2)]  # two microbatches consumed
    assert resumed[: len(full) - 2] == full[2:]
    with pytest.raises(ValueError, match="task='finetune-diffusion'"):
        G.Git10MWrapper(SimpleNamespace(**{**vars(mk()), "task": "extract"}))
    # gen-eval reads the val view as its FID reference (the four 2026-10-17 gen-eval jobs died here).
    assert set(G.Git10MWrapper(SimpleNamespace(**{**vars(mk()), "task": "gen-eval"})).get_data(mk())) == {
        "train",
        "val",
    }
    with pytest.raises(ValueError, match="needs --git10m.index"):
        G.Git10MWrapper(mk(g={"index": ""}))


def test_flatten_order_matches_model_parameter_order():
    """AlignState.params (block order) must be the optimizer's exact parameter list: the fp32
    gate compares the two by identity and position."""
    flux = _FakeFlux()
    st = AL.AlignState(flux, AlignConfig(arm="fm", feat="raw"), 28, "cpu", target=_FakeTarget())
    model_order = [p for p in flux.parameters() if p.requires_grad]
    assert len(model_order) == len(st.params) and all(a is b for a, b in zip(model_order, st.params))
    opt = torch.optim.AdamW(st.params, lr=1e-3)
    GP.check_fp32_params(st.params, opt, where="test")
    assert isinstance(st.groups, OrderedDict) and list(st.groups) == list(range(NUM_BLOCKS))


def test_ditf_features_match_the_adapter_definition_and_raw_passes_through():
    """align.feat='ditf' = FluxModel.extract per token: discard channels zeroed, LayerNorm(eps 1e-6,
    no affine), (1 + scale) * x + shift with the block's modulation; 'raw' = the tokens."""
    from types import SimpleNamespace

    torch.manual_seed(0)
    flux = _FakeFlux()
    st = AL.AlignState(
        flux, AlignConfig(arm="naive", proj_hidden=8), 28, "cpu", target=_FakeTarget(), discard=[1]
    )
    tok = torch.randn(2, L, D)
    mod = SimpleNamespace(shift=torch.randn(2, 1, D), scale=torch.randn(2, 1, D))
    x = tok.clone()
    x[..., 1] = 0.0
    x = torch.nn.functional.layer_norm(x, (D,), eps=1e-6)
    ref = (1 + mod.scale) * x + mod.shift
    assert torch.allclose(st.features(tok, mod), ref, atol=1e-6)
    with pytest.raises(ValueError, match="modulation"):
        st.features(tok, None)
    with pytest.raises(ValueError, match="feat"):
        AL.AlignState(flux, AlignConfig(arm="naive", feat="pooled"), 28, "cpu", target=_FakeTarget())
    raw = AL.AlignState(
        flux, AlignConfig(arm="naive", proj_hidden=8, feat="raw"), 28, "cpu", target=_FakeTarget()
    )
    assert torch.equal(raw.features(tok, None), tok)
    ln = AL.AlignState(
        flux, AlignConfig(arm="naive", proj_hidden=8, feat="ln"), 28, "cpu", target=_FakeTarget(), discard=[1]
    )
    assert torch.allclose(ln.features(tok, None), x, atol=1e-6)  # no modulation needed, none applied
    # weighted timestep sampling: only the weighted grid points are drawn, in proportion
    from tasks.train_diffusion import _sample_timesteps

    g = torch.Generator().manual_seed(0)
    t = _sample_timesteps([100, 500, 900], 4000, torch.device("cpu"), g, weights=[1.0, 1.0, 0.0001])
    frac = {v: float((t == v / 1000).float().mean()) for v in (100, 500, 900)}
    assert abs(frac[100] - 0.5) < 0.05 and frac[900] < 0.01
    with pytest.raises(ValueError):
        _sample_timesteps([100, 500], 4, torch.device("cpu"), g, weights=[1.0])


# ------------------------------------------------------------------ 2026-10-23 additions
def test_sem_arm_applies_only_the_weighted_semantic_gradient_and_measures_fm():
    """The sem arm is the mirror of fm: .grad == w * g_SEM exactly, g_FM still measured."""
    flux, st, x, tok, v = _state("sem", w=0.5)
    assert st.w == 0.5 and st.w_fm == 0.0
    fl, al = _losses(flux, st, x, tok, v)
    ref = torch.autograd.grad(al, st.params, retain_graph=True)
    fl, al = _losses(flux, st, x, tok, v)
    st.accumulate(fl, al)
    stats = AL.AlignStats(1, st.n_rows)
    st.apply(stats, 0)
    for p, r in zip(st.params, ref):
        assert torch.allclose(p.grad, 0.5 * r, atol=1e-7)
    assert (stats.st["norm_fm"][0, :-1] > 0).any() and np.isclose(stats.sc["w_eff"][0], 0.5)


def test_stop_step_drops_the_semantic_gradient_from_that_step_on():
    flux, st, x, tok, v = _state("naive", w=0.7)
    st.cfg.stop_step = 1
    stats = AL.AlignStats(2, st.n_rows)
    for step in (0, 1):
        fl, al = _losses(flux, st, x, tok, v)
        g_fm = torch.autograd.grad(fl, st.params, retain_graph=True)
        g_al = torch.autograd.grad(al, st.params, retain_graph=True)
        st.accumulate(fl, al)
        st.apply(stats, step)
        w = 0.7 if step == 0 else 0.0
        for p, a, b in zip(st.params, g_fm, g_al):
            assert torch.allclose(p.grad, a + w * b, atol=1e-6)
        st._clear()
    assert np.isclose(stats.sc["w_eff"][0], 0.7) and stats.sc["w_eff"][1] == 0.0


def test_micro_log_roundtrip_and_old_checkpoints_without_it():
    s = AL.AlignStats(3, 58, accum=2, batch=2)
    s.log_micro(1, 1, 0.6, -0.4, [0.1, 0.9], [7, 8])
    assert s.mb["micro_flow"][1, 1] == np.float32(0.6) and list(s.mb["micro_row"][1, 2:4]) == [7, 8]
    assert np.isnan(s.mb["micro_t"][1, 0]) and np.isclose(s.mb["micro_t"][1, 3], 0.9)
    r = AL.AlignStats.from_state(s.state(), 3, 58, 2, 2)
    assert r.mb["micro_row"][1, 3] == 8 and "step/micro_t" in r.npz_fields()
    old = {"st": s.st, "sc": s.sc}  # a ckpt.pt written before the micro log existed
    r2 = AL.AlignStats.from_state(old, 3, 58, 2, 2)
    assert (r2.mb["micro_row"] == -1).all()
    assert "step/micro_t" not in AL.AlignStats(3, 58).npz_fields()  # accum=0: old layout


def test_new_fields_leave_existing_identities_alone(run_mod):
    """Rule 9: stop_step off and every logging-only field never enter the identity."""
    a = _cfg(run_mod, align=AlignConfig(arm="proj"))
    b = _cfg(
        run_mod,
        align=AlignConfig(
            arm="proj",
            snapshot_every=2000,
            keep_ckpt_every=5000,
            feat_stats=False,
            ckpt_every=500,
            vs_grad_batches=8,
        ),
    )
    assert a.make_run_name() == b.make_run_name()
    ident = a.align.identity()
    assert not {
        "stop_step",
        "snapshot_every",
        "keep_ckpt_every",
        "feat_stats",
        "split_eval",
        "virtual_steps",
        "vs_grad_batches",
        "vs_per_t",
        "grad_geometry",
    } & set(ident)
    c = _cfg(run_mod, align=AlignConfig(arm="proj", stop_step=1000), max_train_steps=2000)
    assert c.align.identity()["stop_step"] == 1000 and c.make_run_name() != a.make_run_name()
    s = _cfg(run_mod, align=AlignConfig(arm="sem"))
    assert s.make_run_name().split("+")[-1].startswith("sem-")


def test_guards_for_stop_step_and_kept_checkpoints(run_mod):
    with pytest.raises(ValueError, match="stop_step needs arm"):
        _cfg(run_mod, align=AlignConfig(arm="fm", stop_step=500), max_train_steps=2000)
    with pytest.raises(ValueError, match="stop_step needs arm"):
        _cfg(run_mod, align=AlignConfig(arm="sem", stop_step=500), max_train_steps=2000)
    with pytest.raises(ValueError, match="must lie in"):
        _cfg(run_mod, align=AlignConfig(arm="proj", stop_step=2000), max_train_steps=2000)
    with pytest.raises(ValueError, match="multiple of align.ckpt_every"):
        _cfg(run_mod, align=AlignConfig(arm="proj", ckpt_every=500, keep_ckpt_every=750))
    with pytest.raises(ValueError, match="multiple of align.ckpt_every"):
        _cfg(run_mod, align=AlignConfig(arm="proj", keep_ckpt_every=500))


def test_sem_t_weights_identity_lookup_and_guards(run_mod):
    """Noise-restricted alignment (align.sem_t_weights): off leaves every existing identity
    unchanged, on enters it; the per-image lookup maps each sampled t to its grid weight and
    refuses off-grid t; RunConfig rejects bad lengths, negatives, all-zero and the fm arm."""
    import torch
    from tasks import train_diffusion as TD

    grid = [100, 180, 260, 340, 420, 500, 580, 740, 900]
    a = _cfg(run_mod, t=grid, align=AlignConfig(arm="proj"))
    assert "sem_t_weights" not in a.align.identity()
    hi = _cfg(run_mod, t=grid, align=AlignConfig(arm="proj", sem_t_weights=[0, 0, 0, 0, 0, 0, 1, 1, 1]))
    lo = _cfg(run_mod, t=grid, align=AlignConfig(arm="proj", sem_t_weights=[1, 1, 0, 0, 0, 0, 0, 0, 0]))
    assert hi.align.identity()["sem_t_weights"] == [0.0] * 6 + [1.0] * 3
    assert len({a.make_run_name(), hi.make_run_name(), lo.make_run_name()}) == 3
    t = torch.tensor([0.1, 0.9, 0.58, 0.18])
    hi9 = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    # uniform t sampling: the window holds 3/9 of the mass -> weight 3 inside it
    w = TD._sem_t_weights_for(t, grid, hi9)
    assert w.tolist() == pytest.approx([0.0, 3.0, 3.0, 0.0])
    # the campaign sampling (1 x 7, 0.5 x 2): the window {580, 740, 900} holds 2/8 -> weight 4
    tw = [1, 1, 1, 1, 1, 1, 1, 0.5, 0.5]
    assert TD._sem_t_weights_for(t, grid, hi9, tw).tolist() == pytest.approx([0.0, 4.0, 4.0, 0.0])
    assert TD._sem_t_weights_for(t, grid, [1, 1, 0, 0, 0, 0, 0, 0, 0], tw).tolist() == pytest.approx(
        [4.0, 0.0, 0.0, 4.0]
    )
    assert TD._sem_t_weights_for(t, grid, [1] * 9, tw).tolist() == pytest.approx(
        [1.0] * 4
    )  # all ones unchanged
    with pytest.raises(ValueError, match="no mass"):
        TD._sem_t_weights_for(t, grid, [0] * 8 + [1], [1] * 8 + [0])
    with pytest.raises(ValueError, match="one weight per grid point"):
        TD._sem_t_weights_for(t, grid, hi9, [1] * 8)
    assert TD._sem_t_weights_for(t, grid, None) is None
    with pytest.raises(ValueError, match="not on the grid"):
        TD._sem_t_weights_for(torch.tensor([0.15]), grid, [1] * 9)
    for bad in ([1] * 8, [-1] + [1] * 8, [0] * 9):
        with pytest.raises(ValueError, match="one weight >= 0 per t"):
            _cfg(run_mod, t=grid, align=AlignConfig(arm="proj", sem_t_weights=bad))
    with pytest.raises(ValueError, match="naive / proj / sem"):
        _cfg(run_mod, t=grid, align=AlignConfig(arm="fm", sem_t_weights=[1] * 9))


def test_virtual_steps_guards_and_etas():
    """The virtual-steps diagnostic: fixed step sizes, needs an arm, excludes resume / split_eval,
    needs a positive gradient batch count (tasks/train_diffusion run-time guards)."""
    from tasks import train_diffusion as TD

    assert TD.VS_ETAS == (0.01, 0.03)
    assert "virtual_steps" in AlignConfig.LOGGING_ONLY and "vs_grad_batches" in AlignConfig.LOGGING_ONLY
    assert "vs_per_t" in AlignConfig.LOGGING_ONLY and AlignConfig().vs_per_t is False
    src = open(TD.__file__).read()
    assert "align.virtual_steps needs align.arm and excludes align.resume and align.split_eval" in src
    assert "align.vs_grad_batches must be positive" in src


def test_grad_geometry_guards_and_identity():
    """The gradient-geometry diagnostic (2026-10-28): logging-only, needs an arm, excludes resume /
    split_eval / virtual_steps, refuses to overwrite, gates the Gram matrix against the vectors."""
    from tasks import train_diffusion as TD

    assert "grad_geometry" in AlignConfig.LOGGING_ONLY and AlignConfig().grad_geometry == ""
    src = open(TD.__file__).read()
    assert (
        "align.grad_geometry needs align.arm and excludes align.resume, align.split_eval and align.virtual_steps"
        in src
    )
    assert "grad_geometry: {out_path} exists; remove it first" in src
    assert "GRAD_GEOMETRY gate FAIL" in src
    # run.py must let the diagnostic reuse the finished run directory (the smoke of 2026-10-28
    # died on the "already exists" guard) and must refuse a missing one.
    run_src = open(os.path.join(ROOT, "run.py")).read()
    assert "align.grad_geometry given but the run directory" in run_src
    assert (
        "cfg.align.resume or cfg.align.split_eval or cfg.align.virtual_steps or cfg.align.grad_geometry"
        in run_src
    )


def test_align_hash_of_every_existing_run_dir_is_unchanged(run_mod):
    """Rebuild each finished Git-10M run's alignment identity from its meta.json with the CURRENT
    code: the 8-hex suffix of its run directory must come out unchanged (else split-eval,
    resume and the extraction stamps would point at new directories)."""
    import glob
    import json
    from dataclasses import fields

    metas = sorted(glob.glob(os.path.join(ROOT, "models", "git10m", "git10m_flux_*", "meta.json")))
    if not metas:
        pytest.skip("no finished Git-10M runs on this machine")
    a_fields = {f.name for f in fields(AlignConfig)}
    g_fields = {f.name for f in fields(Git10MConfig)}
    checked = 0
    for m in metas:
        c = json.load(open(m))
        c = c.get("cfg", c)
        if not c.get("align", {}).get("arm"):
            continue
        al = AlignConfig(**{k: v for k, v in c["align"].items() if k in a_fields})
        g = Git10MConfig(**{k: v for k, v in c["git10m"].items() if k in g_fields})
        cfg = SimpleNamespace(align=al, git10m=g)
        h = run_mod.RunConfig.align_hash(cfg)
        run = os.path.basename(os.path.dirname(m))
        assert run.endswith(f"-{h}"), f"{run}: align hash now {h}"
        checked += 1
    assert checked >= 5


def test_lora_tag_separates_a_stop_step_branch_from_its_arm():
    from tasks.extraction import lora_tag

    assert lora_tag({"arm": "proj", "stop_step": 0}) == "proj"
    assert lora_tag({"arm": "naive", "stop_step": 5000}) == "naive-stop5000"
    assert lora_tag({"arm": "sem"}) == "sem"


def test_micro_log_places_a_short_microbatch_at_its_configured_slot():
    s = AL.AlignStats(2, 58, accum=3, batch=2)
    s.log_micro(0, 0, 0.5, -0.3, [0.1, 0.2], [1, 2])
    s.log_micro(0, 1, 0.6, -0.3, [0.3], [3])  # short microbatch at j = 1
    s.log_micro(0, 2, 0.7, -0.3, [0.5, 0.6], [5, 6])
    assert list(s.mb["micro_row"][0]) == [1, 2, 3, -1, 5, 6]
    with pytest.raises(ValueError, match="batch size"):
        s.log_micro(1, 0, 0.5, -0.3, [0.1, 0.2, 0.3], [1, 2, 3])


def test_sha1_sidecar_is_not_trusted_after_a_copy_that_keeps_size_and_mtime(tmp_path):
    """Rule 6: the guard must fire on the mismatch it exists for (cp -p / rsync -a over a cache)."""
    import importlib
    import time

    wb = importlib.import_module("eval.wb")
    p = tmp_path / "cache.npz"
    p.write_bytes(b"a" * 64)
    first = wb.file_identity(str(p))
    st = os.stat(p)
    time.sleep(0.05)
    p.write_bytes(b"b" * 64)  # same size, different content
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime preserved, as cp -p would
    wb._SHA_CACHE.clear()
    second = wb.file_identity(str(p))
    assert first != second
