"""CXR probe registry and the dataset-agnostic export core, on synthetic images only. Every guard
is fed the mismatch it exists to catch (rule 6). No restricted data is touched."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval import export_cxr as X
from eval import export_geobench
from eval import features as F

CLASSES = ["a", "b", "c"]
SIZES = {"train": 5, "val": 2, "test": 3}


def _img(w=40, h=30, mode="L", seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(1, 255, (h, w), dtype=np.uint8)
    return Image.fromarray(arr, "L").convert(mode)


def _rows(split, n, offset=0, w=40, h=30):
    out = []
    for i in range(n):
        y = np.zeros(len(CLASSES), np.uint8)
        y[(i + offset) % len(CLASSES)] = 1
        out.append((f"{split}_{i}", y, (lambda s=i + offset: _img(w, h, seed=s))))
    return out


def _splits(w=40, h=30):
    return {s: _rows(s, n, offset=k * 7, w=w, h=h) for k, (s, n) in enumerate(SIZES.items())}


@pytest.fixture
def fake_key(monkeypatch, tmp_path):
    """A registered CXR key whose manifest lives under tmp_path."""
    key = "fakecxr"
    monkeypatch.setitem(
        F.OFFICIAL_CXR,
        key,
        {"sizes": dict(SIZES), "root": "unused", "classes": lambda: list(CLASSES), "multilabel": True},
    )
    monkeypatch.setattr(F, "cxr_manifest_path", lambda k: str(tmp_path / "manifests" / f"{k}.json"))
    return key


# ------------------------------------------------------------------- geometry
def test_pad_square_keeps_all_pixels():
    for w, h in ((300, 100), (100, 300)):
        src = np.tile((np.arange(max(w, h)) % 250 + 1).astype(np.uint8), (min(w, h), 1))
        src = src if w > h else src.T
        out, padded = X.pad_square(Image.fromarray(src, "L"))
        a = np.asarray(out)
        assert out.size == (300, 300) and padded
        ox, oy = (300 - w) // 2, (300 - h) // 2
        np.testing.assert_array_equal(a[oy : oy + h, ox : ox + w], src)
        mask = np.ones(a.shape, bool)
        mask[oy : oy + h, ox : ox + w] = False
        assert (a[mask] == 0).all()
    sq = _img(200, 200)
    out, padded = X.pad_square(sq)
    assert not padded and np.array_equal(np.asarray(out), np.asarray(sq))


def test_to_tree_image_never_upsamples_and_caps_at_448():
    big, info = X.to_tree_image(_img(1024, 1024))
    assert info["side"] == 448 and big.size == (448, 448)
    small, info = X.to_tree_image(_img(300, 200))
    assert info == {"src_w": 300, "src_h": 200, "padded": True, "side": 300}
    a = np.asarray(small)
    assert small.mode == "RGB" and a.dtype == np.uint8 and a.shape == (300, 300, 3)
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    rgba, _ = X.to_tree_image(_img(64, 64, mode="RGBA"))
    assert rgba.mode == "RGB"
    for mode in ("P", "LA"):
        assert X.to_tree_image(_img(64, 64, mode=mode))[0].mode == "RGB"


@pytest.mark.parametrize("mode", ["I;16", "I", "F"])
def test_to_tree_image_rejects_16bit_and_float(mode):
    img = Image.new(mode, (8, 8))
    with pytest.raises(ValueError, match="unsupported image mode"):
        X.to_tree_image(img)


# ------------------------------------------------------------------- tree writer
def test_write_tree_layout_is_what_list_official_split_reads(fake_key, tmp_path):
    root = str(tmp_path / "tree")
    splits = _splits()
    rep = X.write_tree(root, fake_key, splits, CLASSES, {}, restricted=False)
    for split, rows in splits.items():
        files, y = F.list_official_split(fake_key, split, root)
        assert [os.path.basename(f) for f in files] == [f"{n}.png" for n, _, _ in rows]
        np.testing.assert_array_equal(y, np.stack([r[1] for r in rows]))
        assert rep["splits"][split]["n"] == len(rows)
    man = export_geobench.tree_manifest(fake_key, root, CLASSES)
    assert man["classes"] == CLASSES and man["splits"]["train"]["n"] == 5
    assert not os.path.exists(root + ".partial")


def test_write_tree_table_report_and_extra_columns(fake_key, tmp_path):
    root = str(tmp_path / "tree")
    extra = {
        "train_0": {"frontal_lateral": "Frontal", "AP/PA": "PA", "patient": "p1", "draw_rank": 7},
        "train_1": {"AP/PA": "AP"},
    }
    rep = X.write_tree(root, fake_key, _splits(w=60, h=30), CLASSES, extra, restricted=True)
    rows = list(csv.DictReader(open(os.path.join(root, "export_table.csv"))))
    assert list(rows[0])[:9] == X._TABLE_BASE
    assert list(rows[0])[9:] == ["AP/PA", "frontal_lateral"]
    r0 = rows[0]
    assert (r0["name"], r0["patient"], r0["draw_rank"], r0["AP/PA"]) == ("train_0", "p1", "7", "PA")
    assert rows[2]["draw_rank"] == "2" and rows[2]["AP/PA"] == ""
    assert (r0["src_w"], r0["src_h"], r0["padded"], r0["side"]) == ("60", "30", "True", "60")
    saved = json.load(open(os.path.join(root, "export_report.json")))
    assert saved == rep
    assert rep["splits"]["train"]["padded"] == 5 and rep["splits"]["train"]["src_sizes"] == {"60x30": 5}
    assert sum(rep["splits"]["train"]["counts"]) == 5
    assert "train_0" not in json.dumps(rep) and "p1" not in json.dumps(rep)


def test_export_refuses_existing_tree(fake_key, tmp_path):
    root = str(tmp_path / "tree")
    X.write_tree(root, fake_key, _splits(), CLASSES, {}, restricted=False)
    with pytest.raises(SystemExit, match="already exists"):
        X.write_tree(root, fake_key, _splits(), CLASSES, {}, restricted=False)
    other = str(tmp_path / "other")
    os.mkdir(other + ".partial")
    with pytest.raises(SystemExit, match="already exists"):
        X.write_tree(other, fake_key, _splits(), CLASSES, {}, restricted=False)
    assert os.listdir(other + ".partial") == []


def test_duplicate_names_fail_before_anything_is_written(fake_key, tmp_path):
    within = _splits()
    within["train"].append(within["train"][0])
    across = _splits()
    across["val"][0] = ("train_0", *across["val"][0][1:])
    for i, splits in enumerate((within, across)):
        root = str(tmp_path / f"t{i}")
        with pytest.raises(SystemExit, match="1 duplicate"):
            X.write_tree(root, fake_key, splits, CLASSES, {}, restricted=False)
        assert not os.path.exists(root) and not os.path.exists(root + ".partial")


@pytest.mark.parametrize("bad", ["a/b", "../x", ".hidden", "..", "", "a\\b"])
def test_unsafe_names_fail_before_anything_is_written(fake_key, tmp_path, bad):
    splits = _splits()
    splits["test"][1] = (bad, *splits["test"][1][1:])
    root = str(tmp_path / "t")
    with pytest.raises(SystemExit, match="1 unsafe"):
        X.write_tree(root, fake_key, splits, CLASSES, {}, restricted=False)
    assert not os.path.exists(root) and not os.path.exists(root + ".partial")


def test_restricted_name_failure_is_counts_only_and_logged(fake_key, tmp_path):
    splits = _splits()
    splits["val"][0] = ("patient777_view1", *splits["val"][0][1:])
    splits["test"][0] = ("patient777_view1", *splits["test"][0][1:])
    root = str(tmp_path / "t")
    with pytest.raises(SystemExit) as ei:
        X.write_tree(root, fake_key, splits, CLASSES, {}, restricted=True)
    assert "patient777" not in str(ei.value) and "1 duplicate" in str(ei.value)
    assert "patient777_view1" in open(root + ".errors.jsonl").read()


def test_restricted_write_failure_hides_the_path_and_keeps_partial(fake_key, tmp_path):
    def boom():
        raise FileNotFoundError("/scratch/chexpert/patient64541/study1/view1_frontal.jpg")

    splits = _splits()
    splits["val"][1] = (splits["val"][1][0], splits["val"][1][1], boom)
    root = str(tmp_path / "t")
    with pytest.raises(SystemExit) as ei:
        X.write_tree(root, fake_key, splits, CLASSES, {}, restricted=True)
    msg = str(ei.value)
    assert "patient64541" not in msg and "val" in msg and "item 1" in msg
    assert "patient64541" in open(root + ".errors.jsonl").read()
    assert os.path.isdir(root + ".partial") and not os.path.exists(root)
    with pytest.raises(SystemExit, match="already exists"):  # retry on the same root is refused
        X.write_tree(root, fake_key, _splits(), CLASSES, {}, restricted=True)


def test_unrestricted_write_failure_propagates(fake_key, tmp_path):
    splits = _splits()
    splits["train"][0] = (splits["train"][0][0], splits["train"][0][1], lambda: Image.new("I;16", (4, 4)))
    with pytest.raises(ValueError, match="unsupported image mode"):
        X.write_tree(str(tmp_path / "t"), fake_key, splits, CLASSES, {}, restricted=False)
    assert os.path.isdir(str(tmp_path / "t") + ".partial")


def test_label_width_mismatch_is_caught(fake_key, tmp_path):
    splits = _splits()
    splits["train"][0] = (splits["train"][0][0], np.zeros(2, np.uint8), splits["train"][0][2])
    with pytest.raises(ValueError, match="label shape"):
        X.write_tree(str(tmp_path / "t"), fake_key, splits, CLASSES, {}, restricted=False)


# ------------------------------------------------------------------- registry
def test_official_spec_sees_both_registries_and_OFFICIAL_is_unchanged():
    assert list(F.OFFICIAL) == [
        "m_eurosat",
        "m_forestnet",
        "m_so2sat",
        "m_brick_kiln",
        "m_pv4ger",
        "m_bigearthnet",
    ]
    assert F.official_spec("chestxray14")["multilabel"]
    assert F.official_spec("chestxray14")["sizes"] == {"train": 16000, "val": 2000, "test": 25596}
    assert F.official_spec("chexpert")["sizes"]["test"] == 202
    assert F.official_spec("vindr_cxr_kaggle")["sizes"] == {"train": 10500, "val": 1500, "test": 3000}
    assert F.official_spec("m_eurosat") is F.OFFICIAL["m_eurosat"]
    with pytest.raises(SystemExit, match="not registered") as ei:
        F.official_spec("nope")
    assert all(k in str(ei.value) for k in ("m_eurosat", "chestxray14", "chexpert", "vindr_cxr_kaggle"))
    # the registry roots name the tree versions that exist (a re-export gets a new name)
    assert {k: v["root"] for k, v in F.OFFICIAL_CXR.items()} == {
        "chestxray14": "data/cxr/chestxray14_v1",
        "chexpert": "data/cxr/chexpert_v1",
        "vindr_cxr_kaggle": "data/cxr/vindr_cxr_kaggle_v2",
    }


def test_cxr_manifest_paths_and_classes_without_manifest(monkeypatch, tmp_path):
    assert F.cxr_manifest_path("chestxray14").endswith("eval/manifests/chestxray14.json")
    for key in ("chexpert", "vindr_cxr_kaggle"):
        assert F.cxr_manifest_path(key).endswith(f"data/cxr/manifests/{key}.json")
    monkeypatch.setattr(F, "cxr_manifest_path", lambda k: str(tmp_path / f"{k}.json"))
    with pytest.raises(SystemExit, match="--write-manifest") as ei:
        F.official_classes("chexpert")
    assert not any(ch.isdigit() for ch in str(ei.value).replace(str(tmp_path), ""))
    (tmp_path / "chexpert.json").write_text(json.dumps({"classes": ["x", "y"]}))
    assert F.official_classes("chexpert") == ["x", "y"]


# ------------------------------------------------------------------- manifest check
def test_check_cxr_manifest_semantics(fake_key, tmp_path):
    path = F.cxr_manifest_path(fake_key)
    m = {"classes": CLASSES, "splits": {"train": {"n": 5}}}
    with pytest.raises(SystemExit, match="missing"):
        X.check_cxr_manifest(fake_key, m, write=False)
    X.check_cxr_manifest(fake_key, m, write=True)
    assert json.load(open(path)) == m and not os.path.exists(path + ".partial")
    X.check_cxr_manifest(fake_key, m, write=False)
    X.check_cxr_manifest(fake_key, m, write=True)  # identical: fine
    other = {"classes": CLASSES, "splits": {"train": {"n": 99999}}}
    for write in (False, True):
        with pytest.raises(SystemExit) as ei:
            X.check_cxr_manifest(fake_key, other, write=write)
        assert "['splits']" in str(ei.value) and "99999" not in str(ei.value)
    assert json.load(open(path)) == m  # never overwritten


# ------------------------------------------------------------------- main
def _install_splitter(monkeypatch, key):
    monkeypatch.setitem(X.SPLITTERS, key, lambda args: (_splits(), {}, list(CLASSES)))


def test_main_unknown_splitter(monkeypatch, tmp_path):
    monkeypatch.delitem(X.SPLITTERS, "chexpert", raising=False)
    with pytest.raises(SystemExit, match="no splitter"):
        X.main(["--dataset", "chexpert", "--out", str(tmp_path / "t")])


def test_main_exports_writes_then_verifies_manifest(fake_key, monkeypatch, tmp_path, capsys):
    _install_splitter(monkeypatch, fake_key)
    assert X.main(["--dataset", fake_key, "--out", str(tmp_path / "a"), "--write-manifest"]) == 0
    assert "CXR_EXPORT_OK dataset=fakecxr train=5 val=2 test=3" in capsys.readouterr().out
    assert X.main(["--dataset", fake_key, "--out", str(tmp_path / "b")]) == 0  # identical tree matches
    capsys.readouterr()
    # a tree with different pixels must not match the written manifest
    monkeypatch.setitem(X.SPLITTERS, fake_key, lambda args: (_splits(w=50, h=50), {}, list(CLASSES)))
    for extra in ([], ["--write-manifest"]):  # neither mode gives a differing tree its final name
        with pytest.raises(SystemExit, match="differs|refusing to overwrite") as ei:
            X.main(["--dataset", fake_key, "--out", str(tmp_path / "c"), *extra])
        assert str(ei.value).endswith(f"{tmp_path / 'c'}.partial left in place, {tmp_path / 'c'} not created")
        assert not os.path.exists(tmp_path / "c") and os.path.isdir(str(tmp_path / "c") + ".partial")
        shutil.rmtree(str(tmp_path / "c") + ".partial")
    assert json.load(open(F.cxr_manifest_path(fake_key)))["splits"]["train"]["n"] == 5  # never overwritten


def test_write_tree_before_rename_sees_the_complete_partial_and_can_stop_the_rename(fake_key, tmp_path):
    root = str(tmp_path / "t")
    seen = []

    def hook(partial):
        seen.append(sorted(os.listdir(partial)))
        assert partial == root + ".partial" and not os.path.exists(root)

    X.write_tree(root, fake_key, _splits(), CLASSES, {}, restricted=False, before_rename=hook)
    assert seen == [["export_report.json", "export_table.csv", "test", "train", "val"]] and os.path.isdir(
        root
    )

    def stop(partial):
        raise SystemExit("stop")

    root2 = str(tmp_path / "u")
    with pytest.raises(SystemExit, match="stop"):
        X.write_tree(root2, fake_key, _splits(), CLASSES, {}, restricted=True, before_rename=stop)
    assert not os.path.exists(root2) and os.path.isdir(root2 + ".partial")
    assert not os.path.exists(root2 + ".errors.jsonl")  # a count-only exit is not an export error


@pytest.mark.parametrize("key", ["chestxray14", "chexpert", "vindr_cxr_kaggle"])
def test_full_run_without_a_manifest_and_without_write_manifest_exits_before_any_work(
    key, monkeypatch, tmp_path, capsys
):
    """A full export that can only fail at its last step must fail at its first (rule 6: fed the case)."""
    ran = []
    monkeypatch.setitem(X.SPLITTERS, key, lambda args: ran.append(1))
    monkeypatch.setattr(F, "cxr_manifest_path", lambda k: str(tmp_path / "manifests" / f"{k}.json"))
    out = str(tmp_path / "t")
    with pytest.raises(SystemExit) as ei:
        X.main(["--dataset", key, "--out", out])
    msg = str(ei.value)
    assert key in msg and "--write-manifest" in msg
    assert not ran, "the splitter ran"
    for p in (out, out + ".partial", out + ".errors.jsonl"):
        assert not os.path.exists(p), p
    shown = capsys.readouterr()
    assert shown.out == "" and shown.err == ""


def test_the_manifest_pre_flight_does_not_block_write_manifest_or_max_samples(
    fake_key, monkeypatch, tmp_path
):
    _install_splitter(monkeypatch, fake_key)
    assert X.main(["--dataset", fake_key, "--out", str(tmp_path / "s"), "--max-samples", "2"]) == 0
    assert X.main(["--dataset", fake_key, "--out", str(tmp_path / "w"), "--write-manifest"]) == 0


def test_main_max_samples_takes_prefix_and_writes_no_manifest(fake_key, monkeypatch, tmp_path, capsys):
    _install_splitter(monkeypatch, fake_key)
    root = str(tmp_path / "s")
    assert X.main(["--dataset", fake_key, "--out", root, "--max-samples", "2"]) == 0
    assert "CXR_EXPORT_OK dataset=fakecxr train=2 val=2 test=2" in capsys.readouterr().out
    assert list(np.load(os.path.join(root, "train", "labels.npz"))["names"]) == ["train_0", "train_1"]
    assert not os.path.exists(F.cxr_manifest_path(fake_key))
    with pytest.raises(SystemExit, match="cannot be combined"):
        X.main(
            ["--dataset", fake_key, "--out", str(tmp_path / "s2"), "--max-samples", "2", "--write-manifest"]
        )


def test_restricted_tree_manifest_failure_hides_the_file_name(fake_key, monkeypatch, tmp_path):
    _install_splitter(monkeypatch, fake_key)
    monkeypatch.setattr(X, "RESTRICTED", {fake_key})

    def boom(key, root, classes=None):
        raise SystemExit(f"{root}/train/patient123_study1.png: uint8 (3,), expected uint8 RGB")

    monkeypatch.setattr(export_geobench, "tree_manifest", boom)
    root = str(tmp_path / "t")
    with pytest.raises(SystemExit) as ei:
        X.main(["--dataset", fake_key, "--out", root, "--write-manifest"])
    assert "patient123" not in str(ei.value) and "record 1" in str(ei.value)
    assert "patient123_study1.png" in open(root + ".errors.jsonl").read()
    assert not os.path.exists(root) and os.path.isdir(root + ".partial")
    assert not os.path.exists(root + ".partial.errors.jsonl")  # the log sits beside --out
    assert not os.path.exists(F.cxr_manifest_path(fake_key))


def test_unrestricted_tree_manifest_failure_keeps_the_detail(fake_key, monkeypatch, tmp_path):
    _install_splitter(monkeypatch, fake_key)

    def boom(key, root, classes=None):
        raise SystemExit("train/x_1.png: bad")

    monkeypatch.setattr(export_geobench, "tree_manifest", boom)
    with pytest.raises(SystemExit, match="x_1.png"):
        X.main(["--dataset", fake_key, "--out", str(tmp_path / "t"), "--write-manifest"])


# ------------------------------------------------------------------- fix round 1
@pytest.mark.parametrize("bad", [-1, 0.5, float("nan")])
@pytest.mark.parametrize("restricted", [False, True])
def test_label_values_must_be_exactly_zero_or_one(fake_key, tmp_path, bad, restricted):
    splits = _splits()
    name, y, load = splits["val"][1]
    y = y.astype(np.float64)
    y[0] = bad
    splits["val"][1] = ("patient555_v1" if restricted else name, y, load)
    root = str(tmp_path / "t")
    with pytest.raises((ValueError, SystemExit)) as ei:
        X.write_tree(root, fake_key, splits, CLASSES, {}, restricted=restricted)
    assert "0 or 1" in str(ei.value) or "record" in str(ei.value)
    assert "patient555" not in str(ei.value)
    assert not os.path.exists(root)
    if restricted:
        assert isinstance(ei.value, SystemExit) and "item 1" in str(ei.value)
        assert "0 or 1" in open(root + ".errors.jsonl").read()


def test_table_extra_collision_with_base_column_is_refused(fake_key, tmp_path):
    for col in ("src_w", "split"):
        root = str(tmp_path / col)
        with pytest.raises(SystemExit, match="collide"):
            X.write_tree(root, fake_key, _splits(), CLASSES, {"train_0": {col: "x"}}, restricted=False)
        assert not os.path.exists(root) and not os.path.exists(root + ".partial")


def test_restricted_failure_when_error_log_unwritable_leaks_nothing(fake_key, tmp_path, monkeypatch):
    def boom():
        raise FileNotFoundError("/s/patient888/view1.jpg")

    def no_log(root, detail):
        raise OSError("disk full: /s/patient888/view1.jpg")

    splits = _splits()
    splits["train"][2] = (splits["train"][2][0], splits["train"][2][1], boom)
    monkeypatch.setattr(X, "_log_error", no_log)
    with pytest.raises(SystemExit) as ei:
        X.write_tree(str(tmp_path / "t"), fake_key, splits, CLASSES, {}, restricted=True)
    assert "patient888" not in str(ei.value) and "could not be written" in str(ei.value)
    assert ei.value.__suppress_context__ or ei.value.__context__ is None
    assert ei.value.__cause__ is None


def test_main_rejects_max_samples_zero(fake_key, monkeypatch, tmp_path):
    _install_splitter(monkeypatch, fake_key)
    with pytest.raises(SystemExit, match="N >= 1"):
        X.main(["--dataset", fake_key, "--out", str(tmp_path / "t"), "--max-samples", "0"])
    assert not os.path.exists(str(tmp_path / "t"))


# ------------------------------------------------------------------- chestxray14 reader
# The synthetic corpus and manifest helper are Task 4's (tests/test_cxr14.py); importing the fixture
# keeps one definition. The CSV it writes has no View Position and one label, so `cx` rewrites it.
from test_cxr14 import _manifest, _png, corpus  # noqa: E402,F401  (corpus is a fixture)

from eval import cxr_splits as S  # noqa: E402

_FINDINGS = ["No Finding", "Atelectasis", "Mass|Nodule", "Hernia|Effusion|Edema"]


def _edit_csv(c, fn):
    path = os.path.join(c["raw"], "Data_Entry_2017_v2020.csv")
    df = pd.read_csv(path, dtype=str)
    fn(df)
    df.to_csv(path, index=False)


@pytest.fixture
def cx(corpus, monkeypatch):  # noqa: F811
    """Corpus + manifest `man` + a CSV with views and varied labels; PROBE_SIZES 8 / 4."""
    _manifest(corpus)
    corpus["man"] = str(corpus["tmp"] / "man")

    def add_columns(df):
        df["View Position"] = ["PA" if i % 3 else "AP" for i in range(len(df))]
        df["Finding Labels"] = [_FINDINGS[i % len(_FINDINGS)] for i in range(len(df))]

    _edit_csv(corpus, add_columns)
    monkeypatch.setitem(S.PROBE_SIZES, "train", 8)
    monkeypatch.setitem(S.PROBE_SIZES, "val", 4)
    return corpus


def _splits14(c):
    return X.chestxray14_splits(c["arrow"], c["man"], c["raw"])


def _manifest_names(c, fname):
    return pd.read_csv(os.path.join(c["man"], fname), dtype=str)["img_name"].tolist()


def test_chestxray14_splits_on_a_synthetic_corpus(cx):
    splits, table_extra, classes = _splits14(cx)
    csv = pd.read_csv(os.path.join(cx["raw"], "Data_Entry_2017_v2020.csv"), dtype=str).set_index(
        "Image Index"
    )
    assert classes == S.CXR14_CLASSES and list(splits) == ["train", "val", "test"]
    assert [len(splits[s]) for s in splits] == [8, 4, 20]
    names = {s: [n for n, _, _ in rows] for s, rows in splits.items()}
    assert not any(n.endswith(".png") for ns in names.values() for n in ns)
    patient = {s: {csv.at[n + ".png", "Patient ID"] for n in ns} for s, ns in names.items()}
    assert not (patient["train"] & patient["val"] or patient["train"] & patient["test"])
    assert not patient["val"] & patient["test"]
    assert {n + ".png" for n in names["train"]} <= set(_manifest_names(cx, "retained.csv"))
    assert {n + ".png" for n in names["val"]} <= set(_manifest_names(cx, "validation.csv"))
    rows = pd.read_csv(os.path.join(cx["man"], "rows.csv"), dtype={"img_name": str})
    want_test = sorted(rows.loc[rows["bucket"] == "test", "img_name"])
    assert [n + ".png" for n in names["test"]] == want_test
    # the draw runs on the suffixed names, in draw order
    assert [n + ".png" for n in names["train"]] == S.draw(
        rows.loc[rows["bucket"] == "align", "img_name"].tolist(), 8
    )
    for rows_ in splits.values():
        for name, y, _ in rows_:
            assert y.dtype == np.uint8 and y.shape == (14,)
            np.testing.assert_array_equal(y, S.cxr14_labels([csv.at[name + ".png", "Finding Labels"]])[0])
    assert any(y.sum() > 1 for _, y, _ in splits["train"] + splits["test"])  # multi-hot is exercised


def test_chestxray14_table_extra(cx):
    splits, extra, _ = _splits14(cx)
    rows = pd.read_csv(os.path.join(cx["man"], "rows.csv"), dtype={"img_name": str})
    ref = {n: f"{s}:{r}" for n, s, r in zip(rows["img_name"], rows["shard"], rows["row"], strict=True)}
    csv = pd.read_csv(os.path.join(cx["raw"], "Data_Entry_2017_v2020.csv"), dtype=str).set_index(
        "Image Index"
    )
    assert set(extra) == {n for rs in splits.values() for n, _, _ in rs}
    for rs in splits.values():
        for i, (n, _, _) in enumerate(rs):
            e = extra[n]
            assert e["draw_rank"] == i and e["source_ref"] == ref[n + ".png"]
            assert e["patient"] == csv.at[n + ".png", "Patient ID"] == str(int(n[:8]))  # "1", not "00000001"
            assert e["view"] == csv.at[n + ".png", "View Position"] and e["view"] in ("PA", "AP")


def test_chestxray14_load_is_lazy_and_decodes_the_addressed_row(cx):
    moved = cx["arrow"] + "_moved"
    os.rename(cx["arrow"], moved)  # building the splits must not touch a shard
    splits, _, _ = _splits14(cx)
    with pytest.raises(OSError):  # the shard is read at load time, and only then
        splits["val"][0][2]()
    os.rename(moved, cx["arrow"])
    seed = {n: i for i, (n, _) in enumerate(cx["rows"])}
    for split in ("train", "val", "test"):
        for name, _, load in splits[split][:3]:
            i = seed[name + ".png"]
            want = Image.open(io.BytesIO(_png(i, "L" if i % 2 == 0 else "RGBA")))
            got = load()
            assert got.mode == want.mode
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_chestxray14_load_refuses_a_row_that_holds_another_image(cx):
    path = os.path.join(cx["man"], "rows.csv")
    rows = pd.read_csv(path, dtype={"img_name": str})
    a, b = rows.index[rows["bucket"] == "test"][:2]
    swapped = set(rows.loc[[a, b], "img_name"])
    for col in ("shard", "row"):
        rows.loc[[a, b], col] = rows.loc[[b, a], col].to_numpy()
    rows.to_csv(path, index=False)
    splits, _, _ = _splits14(cx)
    loads = {n + ".png": load for n, _, load in splits["test"]}
    for name in swapped:
        with pytest.raises(ValueError, match="does not hold"):
            loads[name]()
    other = next(n for n in loads if n not in swapped)
    loads[other]()  # an untouched row still decodes


def test_chestxray14_guard_rows_and_csv_must_name_the_same_images(cx):
    _edit_csv(cx, lambda df: df.drop(index=[0, 1], inplace=True))
    with pytest.raises(SystemExit, match=r"2 only in rows.csv, 0 only in the CSV"):
        _splits14(cx)


def test_chestxray14_guard_every_name_must_end_in_png(cx):
    path = os.path.join(cx["man"], "rows.csv")
    rows = pd.read_csv(path, dtype={"img_name": str})
    old = rows.loc[rows["bucket"] == "test", "img_name"].iloc[:2].tolist()
    new = {n: n[: -len(".png")] + ".jpeg" for n in old}
    rows["img_name"] = rows["img_name"].replace(new)
    rows.to_csv(path, index=False)
    _edit_csv(cx, lambda df: df.__setitem__("Image Index", df["Image Index"].replace(new)))
    with pytest.raises(SystemExit, match=r"2 image names do not end in \.png"):
        _splits14(cx)


def test_chestxray14_guard_csv_names_missing_from_rows(cx):
    path = os.path.join(cx["man"], "rows.csv")
    rows = pd.read_csv(path, dtype={"img_name": str})
    rows.iloc[:3].to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"0 only in rows.csv, 97 only in the CSV"):
        _splits14(cx)


def test_chestxray14_guard_train_must_be_in_retained_csv(cx):
    train, _, _ = _splits14(cx)
    gone = train["train"][0][0] + ".png"
    path = os.path.join(cx["man"], "retained.csv")
    r = pd.read_csv(path, dtype={"img_name": str})
    r[r["img_name"] != gone].to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"1 train names are not in retained.csv, 0 val names"):
        _splits14(cx)


def test_chestxray14_guard_val_must_be_in_validation_csv(cx):
    splits, _, _ = _splits14(cx)
    gone = splits["val"][0][0] + ".png"
    path = os.path.join(cx["man"], "validation.csv")
    v = pd.read_csv(path, dtype={"img_name": str})
    v[v["img_name"] != gone].to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"0 train names are not in retained.csv, 1 val names"):
        _splits14(cx)


@pytest.mark.parametrize(("a", "b"), [("train", "val"), ("train", "test"), ("val", "test")])
def test_chestxray14_guard_a_shared_patient_fails(cx, a, b):
    splits, _, _ = _splits14(cx)
    src, dst = splits[a][0][0] + ".png", splits[b][0][0] + ".png"

    def share(df):
        pid = df.loc[df["Image Index"] == src, "Patient ID"].iloc[0]
        df.loc[df["Image Index"] == dst, "Patient ID"] = pid

    _edit_csv(cx, share)
    with pytest.raises(SystemExit, match=rf"1 patients are in both {a} and {b}"):
        _splits14(cx)


def test_chestxray14_guard_unknown_view_position_is_counted(cx):
    splits, _, _ = _splits14(cx)
    bad = {splits["train"][0][0] + ".png", splits["test"][0][0] + ".png"}
    _edit_csv(
        cx,
        lambda df: df.__setitem__(
            "View Position", df["View Position"].where(~df["Image Index"].isin(bad), "LL")
        ),
    )
    with pytest.raises(SystemExit, match=r"2 selected images have a View Position other than PA/AP"):
        _splits14(cx)


def test_chestxray14_asking_for_more_than_the_bucket_holds_fails(cx, monkeypatch):
    monkeypatch.setitem(S.PROBE_SIZES, "val", 9)  # the validation bucket holds 8 images
    with pytest.raises(ValueError, match="cannot draw 9 keys from a pool of 8"):
        _splits14(cx)


def test_chestxray14_unknown_finding_label_fails(cx):
    _edit_csv(cx, lambda df: df.__setitem__("Finding Labels", "Fibrosis|Made_Up"))
    with pytest.raises(ValueError, match="unknown ChestX-ray14 label"):
        _splits14(cx)


def test_chestxray14_main_needs_its_three_directories(cx, tmp_path):
    full = ["--raw-dir", cx["raw"], "--arrow-dir", cx["arrow"], "--manifest-dir", cx["man"]]
    for i in range(0, 6, 2):
        flags = full[:i] + full[i + 2 :]
        missing = full[i]
        with pytest.raises(SystemExit, match=missing):
            X.main(["--dataset", "chestxray14", "--out", str(tmp_path / f"o{i}"), *flags])
        assert not os.path.exists(tmp_path / f"o{i}.partial")


def test_chestxray14_main_end_to_end_with_max_samples(cx, tmp_path, capsys):
    root = str(tmp_path / "tree")
    argv = ["--dataset", "chestxray14", "--out", root, "--max-samples", "3"]
    argv += ["--raw-dir", cx["raw"], "--arrow-dir", cx["arrow"], "--manifest-dir", cx["man"]]
    assert X.main(argv) == 0
    assert "CXR_EXPORT_OK dataset=chestxray14 train=3 val=3 test=3" in capsys.readouterr().out
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str)
    full, _, _ = _splits14(cx)
    assert table.loc[table["split"] == "train", "name"].tolist() == [n for n, _, _ in full["train"][:3]]
    assert (
        set(table["view"]) <= {"PA", "AP"}
        and (table["patient"] == table["name"].str[:8].astype(int).astype(str)).all()
    )
    y = np.load(os.path.join(root, "test", "labels.npz"))["y"]
    assert y.shape == (3, 14) and y.dtype == np.uint8


# ------------------------------------------------------------------- main: restricted splitter failures
def test_restricted_splitter_failure_is_counts_only_and_logged(fake_key, monkeypatch, tmp_path, capsys):
    def boom(args):
        raise ValueError("patient00001 secret")

    monkeypatch.setitem(X.SPLITTERS, fake_key, boom)
    monkeypatch.setattr(X, "RESTRICTED", {fake_key})
    root = str(tmp_path / "t")
    with pytest.raises(SystemExit) as ei:
        X.main(["--dataset", fake_key, "--out", root, "--write-manifest"])
    out = capsys.readouterr()
    shown = str(ei.value) + out.out + out.err
    assert ei.value.code != 0 and "ValueError" in shown and "record 1 of" in shown
    assert "patient00001" not in shown and "secret" not in shown
    assert "patient00001 secret" in open(root + ".errors.jsonl").read()
    assert not os.path.exists(root) and not os.path.exists(root + ".partial")


def test_unrestricted_splitter_failure_still_propagates(fake_key, monkeypatch, tmp_path):
    def boom(args):
        raise ValueError("visible detail")

    monkeypatch.setitem(X.SPLITTERS, fake_key, boom)
    with pytest.raises(ValueError, match="visible detail"):
        X.main(["--dataset", fake_key, "--out", str(tmp_path / "t"), "--write-manifest"])


# ------------------------------------------------------------------- chexpert reader (synthetic zips)
import warnings  # noqa: E402
import zipfile  # noqa: E402

from test_cxr_count import make_fake_chexpert  # noqa: E402

CHEXPERT_SIZES = {"train": 4, "val": 1}


@pytest.fixture
def cs(monkeypatch):
    for k, v in CHEXPERT_SIZES.items():
        monkeypatch.setitem(S.PROBE_SIZES, k, v)


def _jpeg(mode="L", seed=0, w=64, h=48):
    buf = io.BytesIO()
    _img(w, h, mode=mode, seed=seed).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _rewrite(raw, fn):
    """Rewrite every zip of the container: member data := fn(name, data) (None keeps it)."""
    for p in sorted(os.path.join(raw, f) for f in os.listdir(raw)):
        with zipfile.ZipFile(p) as zf:
            items = [(i.filename, zf.read(i)) for i in zf.infolist()]
        with zipfile.ZipFile(p + ".new", "w") as zf:
            for n, d in items:
                new = fn(n, d)
                zf.writestr(n, d if new is None else new)
        os.replace(p + ".new", p)


def _container(tmp_path, **kw):
    """make_fake_chexpert, with a real greyscale JPEG behind every image member."""
    raw = make_fake_chexpert(tmp_path, n_train_patients=6, n_valid_patients=2, **kw)
    counter = iter(range(1, 10_000))  # every member gets its own pixels: a mix-up shows in the export
    _rewrite(raw, lambda n, d: None if n.endswith(".csv") else _jpeg(seed=next(counter)))
    return raw


def _csv_frames(raw):
    out = {}
    for p in sorted(os.path.join(raw, f) for f in os.listdir(raw) if f.endswith(".zip")):
        with zipfile.ZipFile(p) as zf:
            for i in zf.infolist():
                if i.filename.endswith(".csv"):
                    out[os.path.basename(i.filename)] = pd.read_csv(io.BytesIO(zf.read(i)))
    return out["train.csv"], out["valid.csv"]


def _selection(raw):
    return S.chexpert_split(*_csv_frames(raw))


def _patch(raw, key, data):
    tail = "/" + key.split("/", 1)[1]
    _rewrite(raw, lambda n, d: data if n.endswith(tail) else None)


def _name(key):
    return key.rsplit(".", 1)[0].replace("/", "_")


def _export(raw, root, extra=()):
    return X.main(["--dataset", "chexpert", "--out", root, "--raw-dir", raw, "--max-samples", "100", *extra])


def test_resolve_requires_exactly_one_member(tmp_path):
    raw = _container(tmp_path, drop_members=1, dup_members=1)
    train, valid = _csv_frames(raw)
    keys = [S.chexpert_key(p) for p in [*train["Path"], *valid["Path"]]]
    zm = X.ZipMembers(sorted(os.path.join(raw, f) for f in os.listdir(raw) if f.endswith(".zip")))
    with pytest.raises(SystemExit) as ei:
        zm.resolve(keys)
    msg = str(ei.value)
    assert "1 unresolved" in msg and "1 in more than one zip" in msg
    assert "patient" not in msg and "study" not in msg
    (tmp_path / "clean").mkdir()
    clean = _container(tmp_path / "clean")  # control: all members present exactly once resolves
    train, valid = _csv_frames(clean)
    zm = X.ZipMembers(sorted(os.path.join(clean, f) for f in os.listdir(clean) if f.endswith(".zip")))
    zm.resolve([S.chexpert_key(p) for p in [*train["Path"], *valid["Path"]]])


def test_resolve_counts_a_key_repeated_inside_one_zip(tmp_path):
    zp = str(tmp_path / "a.zip")
    with zipfile.ZipFile(zp, "w") as zf, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate name this test builds
        for _ in range(2):
            zf.writestr("batch (train 1)/patient00001/study1/view1_frontal.jpg", b"x")
    zm = X.ZipMembers([zp])
    with pytest.raises(SystemExit) as ei:
        zm.resolve(["train/patient00001/study1/view1_frontal.jpg"])
    assert "0 unresolved" in str(ei.value) and "1 in more than one zip" in str(ei.value)
    assert "patient" not in str(ei.value)


def test_zip_that_cannot_be_opened_is_named_by_ordinal(tmp_path):
    bad = tmp_path / "secret_name.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(SystemExit) as ei:
        X.ZipMembers([str(bad)])
    assert "zip ordinal 0" in str(ei.value) and "secret_name" not in str(ei.value)


def test_chexpert_tree_is_frontal_patient_disjoint_and_u_zeros(cs, tmp_path, capsys):
    raw = _container(tmp_path)
    root = str(tmp_path / "tree")
    assert _export(raw, root) == 0
    assert "CXR_EXPORT_OK dataset=chexpert train=4 val=1 test=2" in capsys.readouterr().out
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    assert len(table) == 7 and not table["name"].str.contains("lateral").any()
    per = table.groupby("patient")["split"].nunique()
    assert (per == 1).all()
    train, valid = _csv_frames(raw)
    by_name = {}
    for df in (train, valid):
        for k, (_, r) in zip([_name(S.chexpert_key(p)) for p in df["Path"]], df.iterrows(), strict=True):
            by_name[k] = r
    seen_uncertain = 0
    for split in ("train", "val", "test"):
        z = np.load(os.path.join(root, split, "labels.npz"))
        rows = pd.DataFrame([by_name[n] for n in z["names"]])
        np.testing.assert_array_equal(z["y"], S.chexpert_labels(rows))
        seen_uncertain += int((rows[S.CHEXPERT_CLASSES] == -1).to_numpy().sum())
        assert z["y"].dtype == np.uint8 and set(np.unique(z["y"])) <= {0, 1}
        assert sorted(os.listdir(os.path.join(root, split))) == sorted(
            [*(f"{n}.png" for n in z["names"]), "labels.npz"]
        )
    assert seen_uncertain > 0  # the synthetic labels do hold -1: the U-zeros claim is exercised
    # table extras: draw_rank follows each split's order, ap_pa and frontal_lateral come from the CSV
    assert set(table["frontal_lateral"]) == {"Frontal"} and set(table["ap_pa"]) == {"AP"}
    sel = _selection(raw).set_index("key")
    for _, r in table.iterrows():
        assert r["name"] in by_name and r["patient"].startswith("patient")
    ranks = {(s, _name(k)): rk for k, s, rk in zip(sel.index, sel["split"], sel["draw_rank"], strict=True)}
    assert {(r["split"], r["name"]): int(r["draw_rank"]) for _, r in table.iterrows()} == ranks
    # source_ref is the 0-based position of the zip holding the member in the sorted list
    zips = sorted(f for f in os.listdir(raw) if f.endswith(".zip"))
    where = {}
    for z, f in enumerate(zips):
        with zipfile.ZipFile(os.path.join(raw, f)) as zf:
            for n in zf.namelist():
                if not n.endswith(".csv"):
                    where[n[n.index("patient") :].rsplit(".", 1)[0].replace("/", "_")] = str(z)
    for _, r in table.iterrows():
        tail = r["name"][r["name"].index("patient") :]
        assert where[tail] == r["source_ref"]


def test_chexpert_blank_ap_pa_is_the_empty_string(cs, tmp_path):
    raw = _container(tmp_path)

    def blank_ap(n, d):
        if os.path.basename(n) not in ("train.csv", "valid.csv"):
            return None
        df = pd.read_csv(io.BytesIO(d))
        df["AP/PA"] = ""
        return df.to_csv(index=False).encode()

    _rewrite(raw, blank_ap)
    root = str(tmp_path / "tree")
    _export(raw, root)
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    assert set(table["ap_pa"]) == {""}


def test_export_fails_on_corrupt_zip_member_without_naming_it(cs, tmp_path, capsys):
    raw = _container(tmp_path)
    key = _selection(raw).query("split == 'train'")["key"].iloc[1]
    good = _jpeg(seed=5)
    _patch(raw, key, good[: int(len(good) * 0.6)])  # header intact, body cut
    root = str(tmp_path / "tree")
    with pytest.raises(SystemExit) as ei:
        _export(raw, root)
    out = capsys.readouterr()
    shown = str(ei.value) + out.out + out.err
    assert ei.value.code != 0 and "at item 1" in shown and "record 1 of" in shown
    assert "patient" not in shown and "study" not in shown
    assert key in open(root + ".errors.jsonl").read()
    assert not os.path.exists(root)


@pytest.mark.parametrize("what", ["rgb", "garbage"])
def test_non_greyscale_or_unreadable_member_fails_with_per_mode_counts(cs, tmp_path, capsys, what):
    raw = _container(tmp_path)
    key = _selection(raw).query("split == 'train'")["key"].iloc[0]
    _patch(raw, key, _jpeg("RGB") if what == "rgb" else b"not an image at all")
    root = str(tmp_path / "tree")
    with pytest.raises(SystemExit) as ei:
        _export(raw, root)
    out = capsys.readouterr()
    shown = str(ei.value) + out.out + out.err
    mode = "RGB" if what == "rgb" else "unreadable"
    assert f"'{mode}': 1" in shown and "'L': 6" in shown and "record 1 of" in shown
    assert "patient" not in shown and "study" not in shown
    assert json.loads(open(root + ".errors.jsonl").readline()) == {"stage": "modes", "items": [[key, mode]]}
    assert not os.path.exists(root) and not os.path.exists(root + ".partial")


def test_mode_prepass_without_an_errors_root_still_fails(cs, tmp_path):
    raw = _container(tmp_path)
    key = _selection(raw).query("split == 'test'")["key"].iloc[0]
    _patch(raw, key, _jpeg("RGB"))
    with pytest.raises(SystemExit, match="only L is accepted"):
        X.chexpert_splits(raw)


def test_csv_must_be_in_exactly_one_member_each(cs, tmp_path):
    raw = _container(tmp_path)
    with zipfile.ZipFile(os.path.join(raw, "a_batch2.zip"), "a") as zf:
        zf.writestr("elsewhere/train.csv", b"Path\n")
    with pytest.raises(SystemExit, match=r"2 members named train\.csv"):
        X.chexpert_splits(raw)
    none = tmp_path / "none"
    none.mkdir()
    with zipfile.ZipFile(none / "z.zip", "w") as zf:
        zf.writestr("only/patient00001/study1/view1_frontal.jpg", b"x")
    with pytest.raises(SystemExit, match=r"0 members named train\.csv"):
        X.chexpert_splits(str(none))


def test_loose_csv_files_in_raw_dir_are_never_read(cs, tmp_path):
    raw = _container(tmp_path)
    (tmp_path / "raw" / "train.csv").write_text("Path\nthis would fail to parse as the real file\n")
    splits, extra, classes = X.chexpert_splits(raw)
    assert [len(splits[s]) for s in ("train", "val", "test")] == [4, 1, 2] and classes == S.CHEXPERT_CLASSES


def test_label_guard_fires_when_a_selected_key_matches_two_csv_rows(cs, tmp_path):
    raw = _container(tmp_path)

    def duplicate_as_lateral(n, d):
        if os.path.basename(n) != "train.csv":
            return None
        df = pd.read_csv(io.BytesIO(d))
        fr = df[df["Frontal/Lateral"] == "Frontal"].copy()
        fr["Frontal/Lateral"] = "Lateral"  # same Path, so the frontal-only split cannot see it
        return pd.concat([df, fr]).to_csv(index=False).encode()

    _rewrite(raw, duplicate_as_lateral)
    with pytest.raises(SystemExit) as ei:
        X.chexpert_splits(raw)
    assert "more than one CSV row" in str(ei.value) and "patient" not in str(ei.value)


def test_label_guard_fires_when_a_selected_key_matches_no_csv_row(cs, tmp_path, monkeypatch):
    raw = _container(tmp_path)
    real = S.chexpert_split

    def shifted(train, valid):
        t = real(train, valid)
        t.loc[0, "key"] = "train/patient99999/study1/view1_frontal.jpg"
        return t

    monkeypatch.setattr(S, "chexpert_split", shifted)
    with pytest.raises(SystemExit) as ei:
        X.chexpert_splits(raw)
    assert "no CSV row" in str(ei.value) and "99999" not in str(ei.value)


def test_selected_key_missing_from_the_zips_fails_before_any_image_is_read(cs, tmp_path, monkeypatch):
    raw = _container(tmp_path)
    key = _selection(raw).query("split == 'test'")["key"].iloc[0]
    tail = "/" + key.split("/", 1)[1]
    zp = os.path.join(raw, "z_batch1.zip")
    with zipfile.ZipFile(zp) as zf:
        items = [(i.filename, zf.read(i)) for i in zf.infolist() if not i.filename.endswith(tail)]
    with zipfile.ZipFile(zp, "w") as zf:
        for n, d in items:
            zf.writestr(n, d)
    reads = []
    real = X.ZipMembers.read
    monkeypatch.setattr(X.ZipMembers, "read", lambda self, k: reads.append(k) or real(self, k))
    with pytest.raises(SystemExit, match="1 unresolved"):
        X.chexpert_splits(raw)
    assert reads == []


def test_chexpert_main_needs_raw_dir(tmp_path):
    with pytest.raises(SystemExit, match="--raw-dir"):
        X.main(["--dataset", "chexpert", "--out", str(tmp_path / "t")])
    assert "chexpert" in X.SPLITTERS


def test_every_exported_png_holds_the_pixels_of_its_own_member(cs, tmp_path):
    raw = _container(tmp_path)
    root = str(tmp_path / "tree")
    _export(raw, root)
    source = {}
    for f in sorted(f for f in os.listdir(raw) if f.endswith(".zip")):
        with zipfile.ZipFile(os.path.join(raw, f)) as zf:
            for n in zf.namelist():
                if not n.endswith(".csv"):
                    tail = n[n.index("patient") :].rsplit(".", 1)[0].replace("/", "_")
                    source[tail] = Image.open(io.BytesIO(zf.read(n)))
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    pixels = []
    for _, r in table.iterrows():
        got = np.asarray(Image.open(os.path.join(root, r["split"], f"{r['name']}.png")))
        want = np.asarray(X.to_tree_image(source[r["name"][r["name"].index("patient") :]])[0])
        np.testing.assert_array_equal(got, want)
        pixels.append(got.tobytes())
    assert len(table) == 7 and len(set(pixels)) == 7  # distinct sources: identical content cannot pass


# ------------------------------------------------------------------- main: more restricted-splitter paths
def test_restricted_splitter_systemexit_passes_through_untouched(fake_key, monkeypatch, tmp_path):
    def count_only(args):
        raise SystemExit("chexpert: 3 unresolved keys, 0 in more than one zip")

    monkeypatch.setitem(X.SPLITTERS, fake_key, count_only)
    monkeypatch.setattr(X, "RESTRICTED", {fake_key})
    root = str(tmp_path / "t")
    with pytest.raises(SystemExit) as ei:
        X.main(["--dataset", fake_key, "--out", root, "--write-manifest"])
    assert str(ei.value) == "chexpert: 3 unresolved keys, 0 in more than one zip"
    assert not os.path.exists(root + ".errors.jsonl")


def test_restricted_splitter_failure_with_unwritable_error_log_leaks_nothing(
    fake_key, monkeypatch, tmp_path, capsys
):
    def boom(args):
        raise ValueError("patient00001 secret")

    def no_log(root, detail):
        raise OSError("patient00001 secret")

    monkeypatch.setitem(X.SPLITTERS, fake_key, boom)
    monkeypatch.setattr(X, "RESTRICTED", {fake_key})
    monkeypatch.setattr(X, "_log_error", no_log)
    with pytest.raises(SystemExit) as ei:
        X.main(["--dataset", fake_key, "--out", str(tmp_path / "t"), "--write-manifest"])
    out = capsys.readouterr()
    shown = str(ei.value) + out.out + out.err
    assert ei.value.code != 0 and "ValueError" in shown and "could not be written" in shown
    assert "patient00001" not in shown and "secret" not in shown


# ------------------------------------------------------------------- VinDr-CXR Kaggle reader (synthetic zip)
import functools  # noqa: E402
import re  # noqa: E402

from test_cxr_dicom import EVR, IVR, J2K, make_dicom, scan  # noqa: E402

from eval import cxr_dicom as DCM  # noqa: E402
from eval import cxr_leakage as L  # noqa: E402

VD_SIZES = {"test": 8, "val": 4, "train": 28}
VD_IDS = [f"scanid{i:03d}" for i in range(40)]
VD_TEST_ONLY = [f"testonly{i:02d}" for i in range(5)]
VD_RADS = ("radx01", "radx02", "radx03")
# every string that identifies a scan, a member or a radiologist: none may reach stdout / stderr / an exception
VD_FORBIDDEN = ["scanid", "testonly", "radx", ".dicom", "XSECRETX", *VD_IDS, *VD_TEST_ONLY, *VD_RADS]
VD_TRUTH = {  # class -> which fixture indices all three radiologists (or two of them) marked
    "Cardiomegaly": lambda i: i % 2 == 0,
    "Nodule/Mass": lambda i: i % 3 == 0,
    "Aortic enlargement": lambda i: i % 7 == 0,
    "Pleural effusion": lambda i: False,  # one radiologist only: never two of three
}


@pytest.fixture
def vd(monkeypatch):
    for k, v in VD_SIZES.items():
        monkeypatch.setitem(S.VINDR_KAGGLE_SIZES, k, v)
    monkeypatch.setattr(S, "VINDR_MIN_TEST_POSITIVES", 1)
    monkeypatch.setattr(S, "VINDR_KAGGLE_CLASSES", tuple(_vd_expected([])[0]))  # the fixture's own pin


@functools.lru_cache(maxsize=None)
def _vd_member(i):
    """Scan i: its own pixels; transfer syntax, photometric and window vary with i."""
    mono1 = i % 2 == 1
    a = scan(seed=i)
    return make_dicom(
        arr=(4095 - a).astype(np.uint16) if mono1 else a,
        photometric="MONOCHROME1" if mono1 else "MONOCHROME2",
        ts=(IVR, EVR, J2K)[i % 3],
        window=([2048, 3000], [4000, 4000]) if i % 4 in (0, 1) else None,  # i%4==1: MONOCHROME1 + window
    )


def _vd_csv(ids=VD_IDS, bad_class=None, drop_col=None) -> bytes:
    rows = []
    for i, img in enumerate(ids):
        marks = {r: [] for r in VD_RADS}
        for r in VD_RADS:
            if VD_TRUTH["Cardiomegaly"](i):
                marks[r].append("Cardiomegaly")
            if VD_TRUTH["Aortic enlargement"](i):
                marks[r].append("Aortic enlargement")
        if VD_TRUTH["Nodule/Mass"](i):  # two of three radiologists
            marks[VD_RADS[0]].append("Nodule/Mass")
            marks[VD_RADS[1]].append("Nodule/Mass")
        if i % 5 == 0:  # one of three: never a positive
            marks[VD_RADS[0]].append("Pleural effusion")
        if bad_class:
            for r in VD_RADS:
                marks[r].append(bad_class)
        for r, classes in marks.items():
            for cls in classes or ["No finding"]:
                rows.append({"image_id": img, "class_name": cls, "class_id": 0, "rad_id": r, "x_min": ""})
    df = pd.DataFrame(rows)
    return df.drop(columns=[drop_col] if drop_col else []).to_csv(index=False).encode()


def _vd_zip(
    tmp_path, *, drop=(), dup=(), csv_copies=1, override=None, csv=None, name="kaggle.zip", raw="vraw"
):
    d = tmp_path / raw
    d.mkdir(exist_ok=True)
    override = override or {}
    with zipfile.ZipFile(d / name, "w") as zf, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate names these tests build
        for c in range(csv_copies):
            zf.writestr("train.csv" if c == 0 else f"elsewhere{c}/train.csv", csv or _vd_csv())
        for i, img in enumerate(VD_IDS):
            if img in drop:
                continue
            for _ in range(2 if img in dup else 1):
                zf.writestr(f"train/{img}.dicom", override.get(img, _vd_member(i)))
        for t in VD_TEST_ONLY:  # unlabelled: any attempt to read one fails the header pre-pass
            zf.writestr(f"test/{t}.dicom", b"not a dicom: never selected")
    return str(d)


def _vd_export(raw, root, extra=()):
    return X.main(
        ["--dataset", "vindr_cxr_kaggle", "--out", root, "--raw-dir", raw, "--max-samples", "100", *extra]
    )


def _clean(*texts):
    shown = "\n".join(texts)
    for bad in VD_FORBIDDEN:
        assert bad not in shown, bad


def _vd_fail(raw, root, capsys, extra=()):
    """Run the export expecting a count-only SystemExit; return (message, everything printed)."""
    with pytest.raises(SystemExit) as ei:
        _vd_export(raw, root, extra)
    out = capsys.readouterr()
    shown = str(ei.value) + out.out + out.err
    _clean(shown)
    return str(ei.value), shown


def _vd_expected(ids):
    """(classes, y) from the fixture's own truth table, independent of eval.cxr_splits."""
    idx = {s: [int(i[len("scanid") :]) for i in v] for s, v in S.vindr_kaggle_split(VD_IDS).items()}
    classes = sorted(c for c, f in VD_TRUTH.items() if any(f(i) for i in idx["test"]))
    y = np.array([[int(VD_TRUTH[c](int(i[len("scanid") :]))) for c in classes] for i in ids], dtype=np.uint8)
    return classes, y


def test_vindr_tree_from_a_synthetic_zip(vd, tmp_path, capsys, monkeypatch):
    raw = _vd_zip(tmp_path)
    root = str(tmp_path / "tree")
    calls = []
    real = DCM.decode_dicom_bytes
    monkeypatch.setattr(DCM, "decode_dicom_bytes", lambda b: calls.append(1) or real(b))
    assert _vd_export(raw, root) == 0
    out = capsys.readouterr()
    assert "CXR_EXPORT_OK dataset=vindr_cxr_kaggle train=28 val=4 test=8" in out.out
    _clean(out.out, out.err)
    assert len(calls) == 40  # every selected scan once; the test/ members were never touched
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    assert table["split"].value_counts().to_dict() == {"train": 28, "val": 4, "test": 8}
    assert sorted(table["name"]) == VD_IDS and set(table["photometric"]) == {"MONOCHROME1", "MONOCHROME2"}
    assert set(table["patient"]) == {""} and set(table["transfer_syntax"]) == {
        "implicit_le",
        "explicit_le",
        "jpeg2000_lossless",
    }
    assert not any(n.startswith("testonly") for n in table["name"])
    # check 7 of the leakage gate reads this table
    ok = L._check7(type("Inp", (), {"table": lambda self, k: table})())
    assert ok[0][1] is True
    # selection, order and labels
    classes, _ = _vd_expected([])
    sel = S.vindr_kaggle_split(VD_IDS)
    report = json.load(open(os.path.join(root, "export_report.json")))
    assert report["classes"] == classes and len(classes) >= 2
    for split in ("train", "val", "test"):
        z = np.load(os.path.join(root, split, "labels.npz"))
        assert list(z["names"]) == sel[split]
        np.testing.assert_array_equal(z["y"], _vd_expected(sel[split])[1])
        assert list(table.loc[table["split"] == split, "draw_rank"]) == [
            str(i) for i in range(len(sel[split]))
        ]
    assert set(table["source_ref"].str.split(":").str[0]) == {"0"}


def test_every_exported_vindr_png_holds_the_pixels_of_its_own_member(vd, tmp_path):
    raw = _vd_zip(tmp_path)
    root = str(tmp_path / "tree")
    _vd_export(raw, root)
    with zipfile.ZipFile(os.path.join(raw, "kaggle.zip")) as zf:
        source = {i: DCM.decode_dicom_bytes(zf.read(f"train/{i}.dicom"))[0] for i in VD_IDS}
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    pixels = []
    for _, r in table.iterrows():
        got = np.asarray(Image.open(os.path.join(root, r["split"], f"{r['name']}.png")))
        want = np.asarray(X.to_tree_image(Image.fromarray(source[r["name"]]))[0])
        np.testing.assert_array_equal(got, want)
        pixels.append(got.tobytes())
    assert len(table) == 40 and len(set(pixels)) == 40  # distinct scans (both photometrics): a mix-up shows


def test_vindr_report_carries_the_tallies_timings_and_brightness_quantiles(vd, tmp_path, capsys):
    raw = _vd_zip(tmp_path)
    root = str(tmp_path / "tree")
    _vd_export(raw, root)
    report = json.load(open(os.path.join(root, "export_report.json")))
    tal = report["header_tallies"]
    assert tal["photometric"] == {"MONOCHROME1": 20, "MONOCHROME2": 20}
    assert tal["transfer_syntax"] == {"implicit_le": 14, "explicit_le": 13, "jpeg2000_lossless": 13}
    assert tal["bits_stored"] == {"12": 40} and tal["had_voi"] == {"False": 20, "True": 20}
    assert tal["n_windows"] == {"0": 20, "2": 20}
    assert report["class_test_positives"] == dict(zip(report["classes"], report["splits"]["test"]["counts"]))
    for label, n in tal["transfer_syntax"].items():
        t = report["transfer_syntax_timing"][label]
        assert t["images"] == n and t["zip_read_s"] >= 0 and t["decode_s"] > 0
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    for split in ("train", "val", "test"):
        means = [
            np.asarray(Image.open(os.path.join(root, split, f"{n}.png"))).mean()
            for n in table.loc[table["split"] == split, "name"]
        ]
        want = [round(float(v), 1) for v in np.quantile(means, [0, 0.05, 0.5, 0.95, 1])]
        assert report["out_mean_quantiles"][split] == want
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if "decoded" in ln]
    assert len(line) == 1
    assert re.fullmatch(
        r"vindr_cxr_kaggle: decoded 40 images; transfer syntax explicit_le 13, implicit_le 14, "
        r"jpeg2000_lossless 13; zip read \d+\.\d{6} s, decode \d+\.\d{6} s",
        line[0],
    )


def test_write_tree_report_is_unchanged_without_a_report_hook(tmp_path):
    root = str(tmp_path / "t")
    rep = X.write_tree(root, "fakecxr", {"train": _rows("train", 2)}, CLASSES, {}, restricted=False)
    assert list(rep) == ["dataset", "classes", "splits"]
    assert json.load(open(os.path.join(root, "export_report.json"))) == rep


def test_vindr_main_needs_raw_dir_and_is_registered(tmp_path):
    with pytest.raises(SystemExit, match="--raw-dir"):
        X.main(["--dataset", "vindr_cxr_kaggle", "--out", str(tmp_path / "t"), "--write-manifest"])
    assert "vindr_cxr_kaggle" in X.SPLITTERS


# --- pre-pass failures: count-only, before any scan is decoded or anything is written
def _spy_decode(monkeypatch):
    calls = []
    real = DCM.decode_dicom_bytes
    monkeypatch.setattr(DCM, "decode_dicom_bytes", lambda b: calls.append(1) or real(b))
    return calls


def _nothing_written(root):
    assert not os.path.exists(root) and not os.path.exists(root + ".partial")


def _rle(i):
    data = _vd_member(i)
    assert b"1.2.840.10008.1.2.1\x00" in data
    return data.replace(b"1.2.840.10008.1.2.1\x00", b"1.2.840.10008.1.2.5\x00")  # same length: RLE Lossless


@pytest.mark.parametrize(
    "bad, reason",
    [
        pytest.param(_rle(1), "{'transfer_syntax': 1}", id="transfer_syntax"),
        pytest.param(b"garbage", "{'unreadable': 1}", id="unreadable"),
        pytest.param(
            make_dicom(arr=np.stack([scan()] * 3, -1), photometric="RGB", samples=3),
            "{'photometric': 1}",
            id="photometric",
        ),
        pytest.param(
            make_dicom(arr=np.stack([scan()] * 3, -1), samples=3), "{'samples_per_pixel': 1}", id="samples"
        ),
        pytest.param(make_dicom(arr=scan(), bits=17), "{'bits_stored': 1}", id="bits"),
    ],
)
def test_vindr_header_prepass_failure_is_count_only_and_decodes_nothing(
    vd, tmp_path, capsys, monkeypatch, bad, reason
):
    calls = _spy_decode(monkeypatch)
    raw = _vd_zip(tmp_path, override={VD_IDS[5]: bad})
    root = str(tmp_path / "tree")
    msg, _ = _vd_fail(raw, root, capsys)
    assert "header pre-pass over 40 images found 1 unusable" in msg and reason in msg
    assert "nothing written" in msg and "record 1 of" in msg
    assert calls == []
    _nothing_written(root)
    detail = json.loads(open(root + ".errors.jsonl").readline())
    assert detail["stage"] == "header" and detail["items"][0][0] == VD_IDS[5]  # detail only in the log


def test_vindr_prepass_tallies_name_only_allow_list_labels(vd, tmp_path, capsys):
    raw = _vd_zip(tmp_path, override={VD_IDS[5]: _rle(1)})
    msg, _ = _vd_fail(raw, str(tmp_path / "tree"), capsys)
    assert "'other': 1" in msg and "'implicit_le'" in msg and "MONOCHROME" in msg


def test_vindr_unresolved_selected_id_fails_before_any_header_is_read(vd, tmp_path, capsys, monkeypatch):
    calls = _spy_decode(monkeypatch)
    root = str(tmp_path / "tree")
    msg, _ = _vd_fail(_vd_zip(tmp_path, drop={VD_IDS[3]}), root, capsys)
    assert "1 of 40 selected image ids have no train member, 0 have more than one; nothing read" in msg
    assert calls == []
    _nothing_written(root)


def test_vindr_id_repeated_in_the_zip_fails(vd, tmp_path, capsys):
    root = str(tmp_path / "tree")
    msg, _ = _vd_fail(_vd_zip(tmp_path, dup={VD_IDS[3]}), root, capsys)
    assert "0 of 40 selected image ids have no train member, 1 have more than one; nothing read" in msg
    _nothing_written(root)


@pytest.mark.parametrize("copies, n", [(0, 0), (2, 2)])
def test_vindr_needs_exactly_one_train_csv(vd, tmp_path, capsys, copies, n):
    root = str(tmp_path / "tree")
    msg, _ = _vd_fail(_vd_zip(tmp_path, csv_copies=copies), root, capsys)
    assert msg == f"vindr_cxr_kaggle: {n} members named train.csv, expected 1"
    _nothing_written(root)


@pytest.mark.parametrize("zips", [0, 2])
def test_vindr_needs_exactly_one_zip(vd, tmp_path, capsys, zips):
    raw = tmp_path / "vraw"
    raw.mkdir()
    for z in range(zips):
        _vd_zip(tmp_path, name=f"k{z}.zip")
    msg, _ = _vd_fail(str(raw), str(tmp_path / "tree"), capsys)
    assert msg == f"vindr_cxr_kaggle: {zips} zip files under the raw dir, expected 1"


def test_vindr_zip_that_cannot_be_read_is_named_by_ordinal(vd, tmp_path, capsys):
    raw = tmp_path / "vraw"
    raw.mkdir()
    (raw / "radx_secret.zip").write_bytes(b"not a zip")
    msg, _ = _vd_fail(str(raw), str(tmp_path / "tree"), capsys)
    assert msg == "vindr_cxr_kaggle: the zip cannot be read (BadZipFile)"


def test_vindr_csv_without_a_required_column_fails_count_only(vd, tmp_path, capsys):
    msg, _ = _vd_fail(_vd_zip(tmp_path, csv=_vd_csv(drop_col="rad_id")), str(tmp_path / "tree"), capsys)
    assert msg == "vindr_cxr_kaggle: train.csv lacks 1 of the 3 required columns"


def test_vindr_wrong_number_of_images_fails_count_only(vd, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(S.VINDR_KAGGLE_SIZES, "train", 27)  # the CSV holds 40 ids, the sizes want 39
    with pytest.raises(SystemExit) as ei:
        _vd_export(_vd_zip(tmp_path), str(tmp_path / "tree"))
    out = capsys.readouterr()
    _clean(str(ei.value), out.out, out.err)
    assert "building the splits failed (ValueError)" in str(ei.value)


def test_vindr_class_name_outside_the_allow_list_fails_without_the_name(vd, tmp_path, capsys):
    raw = _vd_zip(tmp_path, csv=_vd_csv(bad_class="XSECRETX9"))
    msg, _ = _vd_fail(raw, str(tmp_path / "tree"), capsys)
    assert msg == "vindr_cxr_kaggle: 1 class names fail the class-name allow-list"


def test_vindr_no_class_reaching_the_threshold_fails(vd, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(S, "VINDR_MIN_TEST_POSITIVES", 99)
    msg, _ = _vd_fail(_vd_zip(tmp_path), str(tmp_path / "tree"), capsys)
    assert msg == "vindr_cxr_kaggle: no class reaches the test-positive threshold"


def test_vindr_label_set_that_differs_from_the_pin_fails_count_only(vd, tmp_path, capsys, monkeypatch):
    derived = _vd_expected([])[0]
    assert len(derived) == 3
    for pin, differ in ((derived[:-1], 1), ([*derived, "Atelectasis"], 1), (["Atelectasis"], 4)):
        monkeypatch.setattr(S, "VINDR_KAGGLE_CLASSES", tuple(pin))
        msg, _ = _vd_fail(_vd_zip(tmp_path), str(tmp_path / "tree"), capsys)
        assert msg == (
            f"vindr_cxr_kaggle: the threshold rule gives 3 classes, the pinned VINDR_KAGGLE_CLASSES "
            f"holds {len(pin)}, {differ} names are in one list only"
        )
        assert not os.path.exists(tmp_path / "tree")


def test_the_real_vindr_pin_is_twelve_sorted_allow_listed_names_without_the_excluded_ones():
    pin = S.VINDR_KAGGLE_CLASSES
    assert len(pin) == 12 and list(pin) == sorted(pin) and not set(pin) & set(S.VINDR_EXCLUDED)
    from eval.cxr_count import _CLASS_NAME_RE  # the allow-list vindr_kaggle_splits applies

    assert all(_CLASS_NAME_RE.fullmatch(c) for c in pin)


# --- load: a scan that passed the pre-pass but cannot be turned into a picture
def test_vindr_constant_scan_fails_the_export_without_naming_it(vd, tmp_path, capsys):
    bad = make_dicom(arr=np.full((16, 24), 900, np.uint16))
    raw = _vd_zip(tmp_path, override={VD_IDS[7]: bad})
    root = str(tmp_path / "tree")
    msg, shown = _vd_fail(raw, root, capsys)
    assert "export failed in split" in shown and "record 1 of" in shown and "RuntimeError" in shown
    log = open(root + ".errors.jsonl").read()
    assert VD_IDS[7] in log and "constant image" in log
    assert os.path.isdir(root + ".partial") and not os.path.exists(root)


def test_vindr_prepass_vs_decode_mismatch_raises(vd, tmp_path, capsys, monkeypatch):
    real = DCM.decode_dicom_bytes

    def shifted(b):
        out, info = real(b)
        return out, {**info, "bits_stored": info["bits_stored"] + 1}

    monkeypatch.setattr(DCM, "decode_dicom_bytes", shifted)
    root = str(tmp_path / "tree")
    msg, shown = _vd_fail(_vd_zip(tmp_path), root, capsys)
    assert "export failed in split" in shown and "RuntimeError" in shown
    log = open(root + ".errors.jsonl").read()
    assert "header pre-pass" in log and "bits_stored" in log


@pytest.mark.parametrize("field, value", [("photometric", "MONOCHROME1"), ("transfer_syntax", "other")])
def test_vindr_mismatch_guard_checks_each_header_field(vd, tmp_path, capsys, monkeypatch, field, value):
    real = DCM.decode_dicom_bytes

    def swapped(b):
        out, info = real(b)
        return out, {**info, field: value if info[field] != value else "MONOCHROME2"}

    monkeypatch.setattr(DCM, "decode_dicom_bytes", swapped)
    root = str(tmp_path / "tree")
    _vd_fail(_vd_zip(tmp_path), root, capsys)
    assert field in open(root + ".errors.jsonl").read()


# --- fix round 1: library warnings, raw-max instrument, max_samples
def _write_files(tmp_path):
    return b"".join(p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("where", ["decode", "header", "csv"])
def test_library_warnings_are_counted_by_class_and_never_shown(vd, tmp_path, capsys, monkeypatch, where):
    import pandas
    import pydicom

    text = "patient00001 secret-id"
    if where == "decode":
        real = DCM.decode_dicom_bytes
        monkeypatch.setattr(DCM, "decode_dicom_bytes", lambda b: warnings.warn(text) or real(b))
    elif where == "header":
        real_read = pydicom.dcmread

        def noisy(fh, *a, **kw):
            if kw.get("stop_before_pixels"):
                warnings.warn(text)
            return real_read(fh, *a, **kw)

        monkeypatch.setattr(pydicom, "dcmread", noisy)
    else:
        real_csv = pandas.read_csv
        monkeypatch.setattr(pandas, "read_csv", lambda *a, **kw: warnings.warn(text) or real_csv(*a, **kw))
    raw = _vd_zip(tmp_path)
    root = str(tmp_path / "tree")
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        assert _vd_export(raw, root) == 0
    out = capsys.readouterr()
    assert "secret-id" not in out.out + out.err
    assert not [w for w in outer if "secret-id" in str(w.message)]  # nothing escaped the instrument
    assert b"secret-id" not in _write_files(tmp_path)
    monkeypatch.undo()
    report = json.load(open(os.path.join(root, "export_report.json")))
    assert report["warning_counts"] == {"UserWarning": 1 if where == "csv" else 40}


def test_warning_counts_is_empty_without_warnings(vd, tmp_path):
    root = str(tmp_path / "tree")
    _vd_export(_vd_zip(tmp_path), root)
    assert json.load(open(os.path.join(root, "export_report.json")))["warning_counts"] == {}


def test_raw_max_quantiles_are_per_transfer_syntax_and_bits_stored(vd, tmp_path):
    root = str(tmp_path / "tree")
    _vd_export(_vd_zip(tmp_path), root)
    report = json.load(open(os.path.join(root, "export_report.json")))
    groups: dict[str, list[int]] = {}
    for i in range(40):
        a = scan(seed=i)
        stored = 4095 - a if i % 2 == 1 else a
        groups.setdefault(f"{('implicit_le', 'explicit_le', 'jpeg2000_lossless')[i % 3]}|12", []).append(
            int(stored.max())
        )
    assert set(report["raw_max_quantiles"]) == set(groups) and len(groups) == 3
    for key, vals in groups.items():
        got = report["raw_max_quantiles"][key]
        assert got["n"] == len(vals)
        assert got["quantiles"] == [round(float(q), 6) for q in np.quantile(vals, [0, 0.25, 0.5, 0.75, 1])]
    assert len({tuple(v["quantiles"]) for v in report["raw_max_quantiles"].values()}) > 1  # groups differ


def test_report_records_max_samples_only_when_set(vd, tmp_path):
    root = str(tmp_path / "tree")
    _vd_export(_vd_zip(tmp_path), root)  # main with --max-samples 100
    assert json.load(open(os.path.join(root, "export_report.json")))["max_samples"] == 100
    bare = X._VindrRun().report_extra({"classes": [], "splits": {}}, {})
    assert "max_samples" not in bare


# --- task 8 fix round 2: opt-in process pool for the zip read + DICOM decode (--workers)
import ast  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import signal  # noqa: E402

import cxr_pool_hooks as H  # noqa: E402  (worker hooks: a module a spawned worker imports cheaply)


def _imported(path):
    """Top-level names of every module a source file imports."""
    names = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def test_pool_hooks_come_from_a_module_that_imports_only_the_standard_library():
    """A spawned worker imports the module that defines its hook. From this test module that is
    torch and pandas per worker (212 s cold, job 6606235), enough to trip the 240 s alarm."""
    assert _imported(H.__file__) == {"__future__", "os", "warnings"}
    assert {"pandas", "pytest", "test_cxr14"} <= _imported(__file__)  # the reader does see heavy imports
    assert H.warn_secret.__module__ == H.die_on.__module__ == "cxr_pool_hooks"


@pytest.fixture
def alarm():
    """A hung pool must fail the test, not the 30 min job."""

    def fire(*_):
        raise TimeoutError("pool test exceeded 240 s")

    old = signal.signal(signal.SIGALRM, fire)
    signal.alarm(240)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


def _tree_digest(root):
    """Everything an export wrote, as comparable values; timing seconds are the one allowed difference."""
    files = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            path = os.path.join(dirpath, n)
            rel = os.path.relpath(path, root)
            if n == "labels.npz":
                z = np.load(path)
                files[rel] = (list(z["names"]), z["y"].tolist())
            elif n == "export_report.json":
                rep = json.load(open(path))
                for v in rep["transfer_syntax_timing"].values():
                    del v["zip_read_s"], v["decode_s"]
                files[rel] = rep
            else:
                files[rel] = open(path, "rb").read()
    return files


def test_vindr_workers_flag_default_is_one_and_no_pool_is_built(vd, tmp_path):
    run = X._VindrRun()
    X.vindr_kaggle_splits(_vd_zip(tmp_path), run=run)
    assert run.prefetch is None  # the default call keeps the serial path


@pytest.mark.parametrize("extra", [(), ("--max-samples", "5")], ids=["full", "max_samples"])
def test_vindr_workers3_tree_is_identical_to_workers1(vd, tmp_path, capfd, alarm, extra):
    raw = _vd_zip(tmp_path)
    one, three = str(tmp_path / "one"), str(tmp_path / "three")
    assert _vd_export(raw, one, ("--workers", "1", *extra)) == 0
    capfd.readouterr()
    assert _vd_export(raw, three, ("--workers", "3", *extra)) == 0
    out = capfd.readouterr()
    assert "CXR_EXPORT_OK dataset=vindr_cxr_kaggle" in out.out
    _clean(out.out, out.err)
    a, b = _tree_digest(one), _tree_digest(three)
    assert sorted(a) == sorted(b)
    assert len(a) >= (40 if not extra else 15)
    for k in a:
        assert a[k] == b[k], k.split("/")[0]  # the file name is an image id: name only the folder
    rep = b["export_report.json"]
    assert rep["warning_counts"] == {} and len(rep["raw_max_quantiles"]) == 3


def test_vindr_corrupt_member_with_workers_is_count_only_and_logged(vd, tmp_path, capfd, alarm):
    bad = make_dicom(arr=np.full((16, 24), 900, np.uint16))  # passes the header pre-pass, fails the decode
    raw = _vd_zip(tmp_path, override={VD_IDS[7]: bad})
    root = str(tmp_path / "tree")
    msg, shown = _vd_fail(raw, root, capfd, ("--workers", "3"))
    assert "export failed in split" in shown and "record 1 of" in shown and "RuntimeError" in shown
    assert "Traceback" not in shown and "concurrent" not in shown
    log = open(root + ".errors.jsonl").read()
    assert VD_IDS[7] in log and "constant image" in log
    assert os.path.isdir(root + ".partial") and not os.path.exists(root)


def test_vindr_worker_that_dies_ends_the_export_count_only(vd, tmp_path, capfd, monkeypatch, alarm):
    monkeypatch.setattr(X, "_VINDR_POOL_HOOK", functools.partial(H.die_on, VD_IDS[7]))
    root = str(tmp_path / "tree")
    msg, shown = _vd_fail(_vd_zip(tmp_path), root, capfd, ("--workers", "2"))
    assert "export failed in split" in shown and "RuntimeError" in shown
    assert "Traceback" not in shown
    assert "BrokenProcessPool" in open(root + ".errors.jsonl").read()
    assert os.path.isdir(root + ".partial") and not os.path.exists(root)


def test_vindr_warnings_raised_in_a_worker_are_tallied_by_class_only(vd, tmp_path, capfd, monkeypatch, alarm):
    monkeypatch.setattr(X, "_VINDR_POOL_HOOK", H.warn_secret)
    root = str(tmp_path / "tree")
    assert _vd_export(_vd_zip(tmp_path), root, ("--workers", "3")) == 0
    out = capfd.readouterr()
    _clean(out.out, out.err)
    assert "XSECRETX" not in out.out + out.err
    assert b"XSECRETX" not in _write_files(tmp_path)
    report = json.load(open(os.path.join(root, "export_report.json")))
    assert report["warning_counts"] == {"UserWarning": 40}


@pytest.mark.parametrize("dataset", ["chexpert", "chestxray14", "vindr_cxr_kaggle"])
def test_workers_zero_and_negative_are_refused(tmp_path, capsys, dataset):
    for bad in ("0", "-2", "x"):
        with pytest.raises(SystemExit) as ei:
            X.main(["--dataset", dataset, "--out", str(tmp_path / "t"), "--workers", bad])
        assert ei.value.code != 0
        assert "--workers" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "t") and not os.path.exists(str(tmp_path / "t") + ".partial")


@pytest.mark.parametrize("dataset", ["chexpert", "chestxray14"])
def test_workers_above_one_is_refused_for_datasets_that_ignore_it(tmp_path, dataset):
    with pytest.raises(SystemExit) as ei:
        X.main(["--dataset", dataset, "--out", str(tmp_path / "t"), "--workers", "2"])
    assert "--workers" in str(ei.value) and "vindr_cxr_kaggle" in str(ei.value)
    assert not os.path.exists(tmp_path / "t.errors.jsonl") and not os.path.exists(tmp_path / "t.partial")


class _FakePool:
    """Records submissions; every future is already done, so only the look-ahead logic is measured."""

    def __init__(self):
        self.submitted = []

    def submit(self, fn, img):
        self.submitted.append(img)
        f = cf.Future()
        f.set_result(("r", img))
        return f

    def shutdown(self, **kw):
        self.shut = kw


def test_prefetch_window_is_bounded_ordered_and_survives_skipped_ids():
    order = [f"id{i}" for i in range(60)]
    pf = X._VindrPrefetch("unused", 2, order)
    fake = pf._pool = _FakePool()
    assert pf.get("id0") == ("r", "id0")
    assert fake.submitted == order[:8]  # 4 * workers, never the whole list
    assert pf.get("id1") == ("r", "id1") and fake.submitted == order[:9]
    assert pf.get("id30") == ("r", "id30")  # non-contiguous: jumps, drops what was passed
    assert fake.submitted[9:] == order[30:38] and sorted(pf._futs) == list(range(31, 38))
    for k in range(31, 60):
        assert pf.get(f"id{k}") == ("r", f"id{k}") and len(pf._futs) <= 7
    assert len(fake.submitted) == 9 + 30 and len(set(fake.submitted)) == 39
    pf.close()
    assert fake.shut == {"wait": True, "cancel_futures": True}


def test_vindr_splitter_announces_only_the_consumed_prefix_and_starts_no_pool(vd, tmp_path):
    run = X._VindrRun()
    splits, _, _ = X.vindr_kaggle_splits(_vd_zip(tmp_path), run=run, workers=3, limit=2)
    assert len(run.prefetch.order) == 6 and run.prefetch._pool is None  # lazy: nothing spawned yet
    assert run.prefetch.order == [i for s in splits.values() for i, _, _ in s[:2]]
    run.close()  # closing a pool that never started is a no-op


# --- final fix round: stored values outside the BitsStored range (VinDr-CXR export)
def _over_member(n_over=3, value=4200, seed=6):
    """A JPEG 2000 member (the plugin does not mask) with `n_over` stored values above the 12-bit range."""
    a = scan(seed=seed)
    a.ravel()[:n_over] = value
    return make_dicom(arr=a, ts=J2K)


def _over_zip(tmp_path, **kw):
    return _vd_zip(tmp_path, override={VD_IDS[6]: _over_member(**kw)})


def test_vindr_few_over_range_pixels_export_identically_with_one_and_three_workers(
    vd, tmp_path, capfd, alarm
):
    raw = _over_zip(tmp_path)
    one, three = str(tmp_path / "one"), str(tmp_path / "three")
    assert _vd_export(raw, one, ("--workers", "1")) == 0
    out1 = capfd.readouterr()
    assert _vd_export(raw, three, ("--workers", "3")) == 0
    out3 = capfd.readouterr()
    for out in (out1, out3):
        _clean(out.out, out.err)
        assert (
            "1 images had stored values outside the BitsStored range, clipped (max fraction 0.0078125)"
            in out.out
        )
    a, b = _tree_digest(one), _tree_digest(three)
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k] == b[k], k.split("/")[0]
    rep = b["export_report.json"]
    assert rep["raw_over_range"] == {
        "jpeg2000_lossless|12": {"images": 1, "pixels": 3, "max_frac": 0.0078125}
    }
    assert set(rep["versions"]) == {"pillow", "pydicom", "numpy"}


def test_vindr_report_without_over_range_pixels_has_an_empty_raw_over_range(vd, tmp_path, capsys):
    root = str(tmp_path / "tree")
    assert _vd_export(_vd_zip(tmp_path), root) == 0
    out = capsys.readouterr().out
    assert "outside the BitsStored range" not in out
    assert json.load(open(os.path.join(root, "export_report.json")))["raw_over_range"] == {}


def test_vindr_wholesale_over_range_still_fails_the_export_count_only(vd, tmp_path, capsys):
    raw = _vd_zip(tmp_path, override={VD_IDS[6]: _over_member(n_over=20)})  # 20 of 384 = 5.2 %
    root = str(tmp_path / "tree")
    msg, shown = _vd_fail(raw, root, capsys)
    assert "export failed in split" in shown and "RuntimeError" in shown
    assert "20 of 384 stored values outside the 12-bit range" in open(root + ".errors.jsonl").read()


# --- final fix round: guards that had never fired
def test_vindr_worker_that_cannot_open_the_zip_ends_the_export_count_only(
    vd, tmp_path, capfd, monkeypatch, alarm
):
    """The parent has indexed the zip before the pool starts; the zip is then moved away, so only the
    workers' own `_VindrZip` fails (their initializer), the parent's index keeps working."""
    raw = _vd_zip(tmp_path)
    zpath = os.path.join(raw, "kaggle.zip")
    real_start = X._VindrPrefetch._start

    def start(self):
        if self._pool is None and os.path.exists(zpath):
            os.rename(zpath, zpath + ".moved")
        return real_start(self)

    monkeypatch.setattr(X._VindrPrefetch, "_start", start)
    root = str(tmp_path / "tree")
    try:
        msg, shown = _vd_fail(raw, root, capfd, ("--workers", "2"))
    finally:
        if os.path.exists(zpath + ".moved"):
            os.rename(zpath + ".moved", zpath)
    assert os.path.exists(zpath)  # the control: the zip did exist and was moved by the seam
    assert "export failed in split" in shown and "record 1 of" in shown and "RuntimeError" in shown
    assert "Traceback" not in shown and raw not in shown and "kaggle.zip" not in shown
    assert "worker initializer failed (SystemExit)" in open(root + ".errors.jsonl").read()
    assert os.path.isdir(root + ".partial") and not os.path.exists(root)


@pytest.mark.parametrize("field", ["bits_stored", "photometric", "transfer_syntax"])
def test_vindr_pooled_cross_check_catches_a_prepass_that_disagrees_with_the_decode(
    vd, tmp_path, capfd, monkeypatch, alarm, field
):
    real = X._vindr_header_prepass

    def lying(members, ids, errors_root, run):
        facts = real(members, ids, errors_root, run)
        for f in facts.values():
            f[field] = (
                f[field] + 1
                if field == "bits_stored"
                else "other"
                if field == "transfer_syntax"
                else "MONOCHROME1x"
            )
        return facts

    monkeypatch.setattr(X, "_vindr_header_prepass", lying)
    root = str(tmp_path / "tree")
    msg, shown = _vd_fail(_vd_zip(tmp_path), root, capfd, ("--workers", "2"))
    assert "export failed in split" in shown and "RuntimeError" in shown and "Traceback" not in shown
    log = open(root + ".errors.jsonl").read()
    assert "header pre-pass and decode disagree on " + field in log
