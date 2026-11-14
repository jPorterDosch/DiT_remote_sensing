"""prompt_gen.py prompt-list mode: the shipped remote-sensing list is well formed and the loader refuses the
lists it must refuse (no torch: only the pure helpers are imported)."""

import collections
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "isaac", "git10m"))
import prompt_gen as PG  # noqa: E402

RS = os.path.join(HERE, "..", "experiments", "isaac", "git10m", "rs_prompts.json")


def test_rs_prompt_list_is_the_documented_one():
    ps = PG.load_prompts(RS)
    assert len(ps) == 26
    assert collections.Counter(p["source"] for p in ps) == {
        "clip-eurosat": 10,
        "clip-resisc45": 8,
        "text2earth": 4,
        "diffusionsat-format": 4,
    }
    assert [i for i, p in enumerate(ps) if p["figure"]] == [2, 7, 13, 16, 18, 22]
    assert all(
        p["caption"].startswith("a centered satellite photo of ") and p["caption"].endswith(".")
        for p in ps
        if p["source"] == "clip-eurosat"
    )
    assert all(
        p["caption"].startswith("satellite photo of a ") and p["caption"].endswith(".")
        for p in ps
        if p["source"] == "clip-resisc45"
    )
    # pinned: editing the list after sampling changes this and every manifest's captions_sha1
    assert PG.captions_sha1([p["caption"] for p in ps]).startswith("4bd9e808822f")


def test_first_n_and_order_identity():
    ps = PG.load_prompts(RS, 4)
    assert [p["caption"] for p in ps] == [p["caption"] for p in PG.load_prompts(RS)[:4]]
    caps = [p["caption"] for p in ps]
    assert PG.captions_sha1(caps) != PG.captions_sha1(caps[::-1])


@pytest.mark.parametrize("content, match", [
    ([], "non-empty"), ({"caption": "x"}, "non-empty"), ([{"text": "x"}], "no caption"), ([{"caption": "  "}], "no caption"),
    ([{"caption": "x"}, {"caption": "x"}], "repeated"), (["x"], "no caption")])  # fmt: skip
def test_bad_lists_are_refused(tmp_path, content, match):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(content))
    with pytest.raises(SystemExit, match=match):
        PG.load_prompts(str(p))


@pytest.mark.parametrize("n", [0, -1, 27])
def test_n_out_of_range_is_refused(n):
    with pytest.raises(SystemExit, match="out of range"):
        PG.load_prompts(RS, n)


def test_reference_and_prompt_list_are_exclusive():
    with pytest.raises(SystemExit):
        PG.main(["--tag", "x", "--out", "/nonexistent", "--ref", "a", "--prompts", "b"])
    with pytest.raises(SystemExit):
        PG.main(["--tag", "x", "--out", "/nonexistent"])
