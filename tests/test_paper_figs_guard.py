"""The sampling-identity gate of experiments/figs/paper_figs.py must fire on the mismatches it exists to
catch (CLAUDE.md rule 6): a different guidance, step count, resolution, seed list, sample count or
reference between two arms, and a held-out FID computed on other samples than the recorded ones."""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "figs"))
plt = pytest.importorskip("matplotlib")
import paper_figs as PF  # noqa: E402


def _gen():
    return {"n": 8, "gen_steps": 28, "guidance": 1.0, "seeds": [1000, 1001], "extractor": "inception",
            "sampler": {"gen_n": 4, "gen_seed": 0, "gen_steps": 28, "guidance": 1.0, "img_size": [224, 224]},
            "fake": {"hw": [224, 224], "n": 8, "sha1": "f"}, "real": {"hw": [224, 224], "n": 8, "sha1": "overlap"}}  # fmt: skip


def _held(gen):
    return {
        "fake": copy.deepcopy(gen["fake"]),
        "extractor": gen["extractor"],
        "real": {"hw": [224, 224], "n": 8, "sha1": "heldout"},
    }


def _ident(mutate=None):
    g = _gen()
    h = _held(g)
    if mutate:
        mutate(g, h)
    return PF.sampling_identity(g, h)


def test_identical_arms_pass():
    ref = PF.assert_same_sampling({"a": _ident(), "b": _ident()})
    assert ref["seeds"] == [1000, 1001] and ref["sampler"]["guidance"] == 1.0


def _guidance(g, h):
    g["guidance"] = g["sampler"]["guidance"] = 3.5


def _steps(g, h):
    g["gen_steps"] = g["sampler"]["gen_steps"] = 50


def _res(g, h):
    g["sampler"]["img_size"] = [256, 256]
    g["fake"]["hw"] = h["fake"]["hw"] = [256, 256]


def _seed(g, h):
    g["sampler"]["gen_seed"] = 7
    g["seeds"] = [1007, 1008]


def _count(g, h):
    g["n"] = 4
    g["seeds"] = [1000]


def _ref(g, h):
    h["real"]["sha1"] = "another reference"


def _extractor(g, h):
    g["extractor"] = h["extractor"] = "clip"


@pytest.mark.parametrize("mutate", [_guidance, _steps, _res, _seed, _count, _ref, _extractor])
def test_cross_arm_mismatch_fails(mutate):
    with pytest.raises(SystemExit, match="GATE sampling identity FAIL"):
        PF.assert_same_sampling({"a": _ident(), "b": _ident(mutate)})


def test_seed_list_must_follow_the_sampler():
    def bad(g, h):
        g["seeds"] = [1000, 1005]

    with pytest.raises(SystemExit, match="seeds are not"):
        _ident(bad)


def test_heldout_fid_on_other_samples_is_refused():
    def bad(g, h):
        h["fake"]["sha1"] = "other samples"

    with pytest.raises(SystemExit, match="different samples"):
        _ident(bad)


def test_sampler_block_must_agree_with_recorded_settings():
    def bad(g, h):
        g["sampler"]["guidance"] = 3.5

    with pytest.raises(SystemExit, match="disagrees"):
        _ident(bad)


# ---- the reference-identity gate of a baseline (DiffusionSat) against a FLUX arm
def _held_pair():
    g = _gen()
    h = _held(g) | {"n": 8}
    return copy.deepcopy(h), h


def test_baseline_scored_on_the_same_reference_passes(capsys):
    other, held = _held_pair()
    other["fake"]["sha1"] = (
        "the baseline's own samples"  # other samples are expected: only the reference must match
    )
    PF.assert_same_reference("baseline", other, held)
    assert "GATE reference identity PASS" in capsys.readouterr().out


def _other_ref(o):
    o["real"]["sha1"] = "train-overlap reference"


def _other_extractor(o):
    o["extractor"] = "clip"


def _other_n(o):
    o["n"] = 600


def _other_size(o):
    o["fake"]["hw"] = [512, 512]


def _no_ref(o):
    del o["real"]


def _no_fake(o):
    del o["fake"]


@pytest.mark.parametrize("mutate", [_other_ref, _other_extractor, _other_n, _other_size, _no_ref, _no_fake])
def test_baseline_scored_differently_is_refused(mutate):
    other, held = _held_pair()
    mutate(other)
    with pytest.raises(SystemExit, match="GATE reference identity FAIL"):
        PF.assert_same_reference("baseline", other, held)


def test_two_files_that_both_lack_the_reference_are_refused():
    other, held = _held_pair()
    del other["real"], held["real"]
    with pytest.raises(SystemExit, match="GATE reference identity FAIL"):
        PF.assert_same_reference("baseline", other, held)


# ---- the label gate (no truncated or overlapping text) must fire too
def _labelled_fig():
    import matplotlib.pyplot as mpl

    fig, ax = mpl.subplots(figsize=(3.0, 2.5))
    ax.plot([0.01, 0.02], [50, 60], label="arm")
    ax.set_xlabel("1 / FID")
    ax.set_ylabel("accuracy (%)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def test_fitting_labels_pass():
    fig, _ = _labelled_fig()
    assert PF.assert_labels_fit(fig, "ok") >= 5  # two axis labels, a legend entry, tick labels


def test_label_cut_by_the_canvas_fails():
    fig, ax = _labelled_fig()
    ax.set_xlabel("a label far wider than a three inch figure can hold " * 3)
    with pytest.raises(SystemExit, match="GATE labels fit FAIL.*outside the canvas"):
        PF.assert_labels_fit(fig, "cut")


def test_legend_entry_cut_by_the_canvas_fails():
    fig, _ = _labelled_fig()
    fig.legend(
        ["a figure legend entry that runs off the right edge of the page"],
        loc="lower left",
        bbox_to_anchor=(0.7, 0.0),
    )
    with pytest.raises(SystemExit, match="GATE labels fit FAIL.*outside the canvas"):
        PF.assert_labels_fit(fig, "legend")


def test_overlapping_labels_fail():
    fig, _ = _labelled_fig()
    fig.text(0.5, 0.5, "first annotation")
    fig.text(0.52, 0.5, "second annotation")
    with pytest.raises(SystemExit, match="GATE labels fit FAIL.*overlap"):
        PF.assert_labels_fit(fig, "overlap")


def test_undrawn_tick_labels_are_ignored():
    fig, ax = _labelled_fig()
    ax.set_xticks([0.0, 0.01, 0.02, 5.0])  # 0.0 and 5.0 are outside the view: matplotlib does not draw them
    ax.set_xlim(0.009, 0.021)
    PF.assert_labels_fit(fig, "ticks")


def test_figure_without_text_is_refused():
    import matplotlib.pyplot as mpl

    fig = mpl.figure(figsize=(2, 2))
    with pytest.raises(SystemExit, match="no text found"):
        PF.assert_labels_fit(fig, "empty")


# ---- prompted grid (prompt_gen.py --prompts runs): identity gate, arm gate, label gate, end to end on synthetic runs
PROMPT_KEYS = ("pretrained", "fm s42", "sem s42", "naive s42")
PROMPT_CAPS = [
    "a centered satellite photo of residential buildings or homes or apartments.",  # longest of rs_prompts.json
    "satellite photo of a harbor.",
    "Some white storage tanks are in a piece of bare land",
]


def _prompt_run(root, key, caps=PROMPT_CAPS, px=64, name="rs_prompts", figure=(0, 2), **over):
    import hashlib
    import json

    import numpy as np
    from PIL import Image

    tag = key.replace(" ", "-")
    d = root / f"{tag}_{name}_n{len(caps)}_{px}px_g3.5_s28"
    d.mkdir(parents=True)
    rng = np.random.default_rng(len(key))
    for i in range(len(caps)):
        Image.fromarray(rng.integers(0, 255, (px, px, 3), dtype=np.uint8)).save(d / f"{i:05d}.png")
    run = PF.TRADE_ARMS[key][3]
    man = {"tag": tag, "ckpt": f"/store/{run}/checkpoints/lora_last.pt" if run else "", "prompts_file": "/repo/rs_prompts.json",
           "captions_sha1": hashlib.sha1(json.dumps(caps, ensure_ascii=False).encode()).hexdigest(), "n": len(caps),
           "img_size": px, "steps": 28, "guidance": 3.5, "seed": 0, "noise": "rule",
           "prompts": [{"caption": c, "source": "test", "figure": i in figure} for i, c in enumerate(caps)]}  # fmt: skip
    man.update(over)
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "clip_per_pair.json").write_text(
        json.dumps({"fake": [20.0 + i + len(key) for i in range(len(caps))], "real": None})
    )
    return d


@pytest.fixture
def prompt_root(tmp_path, monkeypatch):
    out = tmp_path / "figs"
    out.mkdir()
    monkeypatch.setattr(PF, "OUT", str(out))
    monkeypatch.delenv("NOTITLE", raising=False)
    return tmp_path / "runs"


def test_prompt_grid_end_to_end(prompt_root):
    import json

    for k in PROMPT_KEYS:
        _prompt_run(prompt_root, k)
    PF.fig_prompt_grid(PROMPT_KEYS, root=str(prompt_root), n_boot=200)
    meta = json.load(open(f"{PF.OUT}/gen_grid_rs_prompts.json"))
    assert [r["prompt_index"] for r in meta["rows"]] == [0, 2]  # the flagged entries, in list order
    assert [r["noise_seed"] for r in meta["rows"]] == [0, 2]
    assert [c["key"] for c in meta["columns"]] == list(PROMPT_KEYS)
    assert meta["columns"][0]["clip_minus_pretrained"] == {"mean": 0.0, "ci95": [0.0, 0.0]}
    assert meta["columns"][1]["clip_minus_pretrained"]["mean"] == pytest.approx(
        len("fm s42") - len("pretrained")
    )
    for ext in ("png", "pdf", "svg"):
        assert os.path.getsize(f"{PF.OUT}/gen_grid_rs_prompts.{ext}") > 0
    PF.fig_prompt_grid(PROMPT_KEYS, idx="all", name="all_rows", root=str(prompt_root), n_boot=200)
    assert len(json.load(open(f"{PF.OUT}/all_rows.json"))["rows"]) == 3


@pytest.mark.parametrize("over", [{"guidance": 1.0}, {"steps": 4}, {"seed": 1}, {"noise": "another rule"}])
def test_prompt_sampling_mismatch_fails(prompt_root, over):
    for k in PROMPT_KEYS[:3]:
        _prompt_run(prompt_root, k)
    _prompt_run(prompt_root, PROMPT_KEYS[3], **over)
    with pytest.raises(SystemExit, match="GATE prompt sampling identity FAIL"):
        PF.fig_prompt_grid(PROMPT_KEYS, root=str(prompt_root), n_boot=200)


def test_prompt_list_mismatch_fails(prompt_root):
    for k in PROMPT_KEYS[:3]:
        _prompt_run(prompt_root, k)
    _prompt_run(
        prompt_root, PROMPT_KEYS[3], caps=PROMPT_CAPS[::-1]
    )  # same captions, another order = another noise per prompt
    with pytest.raises(SystemExit, match="GATE prompt sampling identity FAIL.*captions_sha1"):
        PF.fig_prompt_grid(PROMPT_KEYS, root=str(prompt_root), n_boot=200)


def test_prompt_resolution_mismatch_fails(prompt_root):
    for k in PROMPT_KEYS[:3]:
        _prompt_run(prompt_root, k)
    _prompt_run(prompt_root, PROMPT_KEYS[3], px=32)
    with pytest.raises(SystemExit, match="GATE prompt sampling identity FAIL.*img_size"):
        PF.fig_prompt_grid(PROMPT_KEYS, root=str(prompt_root), n_boot=200)


def test_prompt_arm_with_another_checkpoint_is_refused(prompt_root):
    _prompt_run(
        prompt_root, "fm s42", ckpt="/store/git10m_flux_ca8e6ce7+42+naive-11e56b1e/checkpoints/lora_last.pt"
    )
    with pytest.raises(SystemExit, match="GATE prompt arm FAIL.*needs"):
        PF.prompt_arm("fm s42", str(prompt_root))
    _prompt_run(prompt_root, "pretrained", ckpt="/store/some_lora.pt")
    with pytest.raises(SystemExit, match="GATE prompt arm FAIL.*pretrained"):
        PF.prompt_arm("pretrained", str(prompt_root))


def test_prompt_arm_needs_exactly_one_run(prompt_root):
    with pytest.raises(SystemExit, match="exactly one"):
        PF.prompt_arm("fm s42", str(prompt_root))
    _prompt_run(prompt_root, "fm s42")
    _prompt_run(prompt_root, "fm s42", px=32)
    with pytest.raises(SystemExit, match="exactly one"):
        PF.prompt_arm("fm s42", str(prompt_root))
    assert PF.prompt_arm("fm s42", str(prompt_root), run="_64px_")["identity"]["img_size"] == 64


def test_prompt_arm_with_edited_prompts_is_refused(prompt_root):
    import json

    d = _prompt_run(prompt_root, "sem s42")
    man = json.load(open(d / "manifest.json"))
    man["prompts"][0]["caption"] = "an edited caption"
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="do not hash"):
        PF.prompt_arm("sem s42", str(prompt_root))


def test_prompt_grid_overlong_caption_fails_the_label_gate(prompt_root):
    caps = ["a caption that no single line of a four tile figure can hold " * 4] + PROMPT_CAPS[1:]
    for k in PROMPT_KEYS:
        _prompt_run(prompt_root, k, caps=caps)
    with pytest.raises(SystemExit, match="GATE labels fit FAIL"):
        PF.fig_prompt_grid(PROMPT_KEYS, root=str(prompt_root), wrap=1000, n_boot=200)
    PF.fig_prompt_grid(
        PROMPT_KEYS, name="wrapped", root=str(prompt_root), n_boot=200
    )  # default wrap: it fits


def test_prompt_grid_without_flagged_rows_needs_indices(prompt_root):
    for k in PROMPT_KEYS:
        _prompt_run(prompt_root, k, figure=())
    with pytest.raises(SystemExit, match="no entry is flagged"):
        PF.fig_prompt_grid(PROMPT_KEYS, root=str(prompt_root), n_boot=200)
    with pytest.raises(SystemExit, match="not distinct indices"):
        PF.fig_prompt_grid(PROMPT_KEYS, idx=(0, 7), root=str(prompt_root), n_boot=200)
