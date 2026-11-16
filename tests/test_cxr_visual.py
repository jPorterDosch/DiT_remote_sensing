"""eval/cxr_visual.py on SYNTHETIC trees and archives only (restricted data is never touched).

Worlds: a ChestX-ray14 corpus (tar -> Arrow -> manifest -> index -> tree), a CheXpert zip container and a
VinDr-CXR Kaggle zip, each exported by eval.export_cxr and then read back by the visual-check module.
Restricted keys: every text in a figure and everything printed or raised carries no fixture name, patient
id or member name (rule 6: the checks are fed the strings they exist to catch)."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.figure  # noqa: E402
import matplotlib.text  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from test_cxr14 import PER_PATIENT, N_ARCH, N_TEST_PATIENTS, N_TV_PATIENTS, _png, _tar  # noqa: E402
from test_export_cxr import (  # noqa: E402
    VD_FORBIDDEN,
    VD_IDS,
    _container,
    _export,
    _name,
    _rewrite,
    _selection,
    _vd_export,
    _vd_zip,
    cs,  # noqa: F401  (fixture)
    vd,  # noqa: F401  (fixture)
)

from datasets import cxr14 as G  # noqa: E402
from eval import cxr_dicom as DCM  # noqa: E402
from eval import cxr_splits as S  # noqa: E402
from eval import cxr_visual as V  # noqa: E402
from eval import export_cxr as X  # noqa: E402
from eval import features as F  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINDINGS = ["No Finding", "Atelectasis", "Mass|Nodule", "Hernia|Effusion|Edema"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _St:
    def __init__(self, mode=0o700, uid=None):
        self.st_mode = stat.S_IFDIR | mode
        self.st_uid = os.getuid() if uid is None else uid


@pytest.fixture
def private(monkeypatch):
    """The guard sees a private (0700, ours) directory: pytest's tmp_path may sit under any ancestor."""
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o700))


def _texts(fig):
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text)]


def _panels(fig):
    return [a.images[0].get_array() for a in fig.axes if a.images]


# ------------------------------------------------------------------- the ChestX-ray14 world
def _world14(tmp, mp, png_fn):
    """tar -> Arrow -> manifest -> index -> probe tree (6 / 4 / 6 images). Returns the paths and the
    source PNG bytes of every image name."""
    mp.setattr(G, "N_ARCHIVES", N_ARCH)
    rng = np.random.default_rng(0)
    tv = [f"{p:08d}" for p in range(1, N_TV_PATIENTS + 1)]
    te = [f"{p:08d}" for p in range(100, 100 + N_TEST_PATIENTS)]
    rows = [(f"{p}_{k:03d}.png", p) for p in tv + te for k in range(PER_PATIENT)]
    mp.setattr(G, "N_IMAGES", len(rows))
    order = rng.permutation(len(rows))
    raw = tmp / "raw14"
    raw.mkdir()
    per = -(-len(rows) // N_ARCH)
    src = {}
    for a in range(N_ARCH):
        members = []
        for i in order[a * per : (a + 1) * per]:
            src[rows[i][0][: -len(".png")]] = png_fn(int(i))
            members.append((rows[i][0], src[rows[i][0][: -len(".png")]]))
        _tar(raw / G.archive_name(a), members)
    pd.DataFrame(
        {
            "Image Index": [r[0] for r in rows],
            "Finding Labels": [FINDINGS[i % len(FINDINGS)] for i in range(len(rows))],
            "Follow-up #": 0,
            "Patient ID": [int(r[1]) for r in rows],
            "View Position": ["PA" if i % 3 else "AP" for i in range(len(rows))],
        }
    ).to_csv(raw / "Data_Entry_2017_v2020.csv", index=False)
    (raw / "train_val_list.txt").write_text("\n".join(n for n, p in rows if p in tv))
    (raw / "test_list.txt").write_text("\n".join(n for n, p in rows if p in te))
    arrow = tmp / "arrow14"
    for a in range(N_ARCH):
        G.convert_archive(str(raw / G.archive_name(a)), str(arrow), a)
    man = str(tmp / "man14")
    G.build_manifest(str(arrow), str(raw), man)
    mp.setitem(S.PROBE_SIZES, "train", 8)
    mp.setitem(S.PROBE_SIZES, "val", 4)
    index = str(tmp / "idx14.npz")
    G.build_index(man, index)
    root = str(tmp / "tree14")
    argv = ["--dataset", "chestxray14", "--out", root, "--max-samples", "6"]
    assert X.main(argv + ["--raw-dir", str(raw), "--arrow-dir", str(arrow), "--manifest-dir", man]) == 0
    return {"arrow": str(arrow), "index": index, "root": root, "src": src}


def _mixed(i):
    return _png(i, "L" if i % 2 == 0 else "RGBA")  # 20 x 16, alpha on odd ones


def _square_grey(i):
    return _png(i, "L", (16, 16))


@pytest.fixture
def w14(tmp_path, monkeypatch):
    return _world14(tmp_path, monkeypatch, _mixed)


# ------------------------------------------------------------------- notebook
def test_notebook_has_no_hardwired_worktree_path_and_no_default_gate_file():
    with open(os.path.join(REPO, "notebooks", "cxr_visual_check.ipynb")) as f:
        nb = json.load(f)
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "DiT_remote_sensing-cxr" not in code and "/nfs/" not in code
    assert "CXR_REPO" in code and "RuntimeError" in code and "plt" not in code
    assert re.search(r"^GATE_JSON = None$", code, re.M) and "leakage/gate.json" not in code


def test_notebook_is_tracked_clean():
    with open(os.path.join(REPO, "notebooks", "cxr_visual_check.ipynb")) as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    cells = nb["cells"]
    assert cells[0]["cell_type"] == "markdown" and "copy this file" in "".join(cells[0]["source"]).lower()
    called = set()
    for c in cells:
        if c["cell_type"] != "code":
            continue
        assert c["outputs"] == [] and c["execution_count"] is None
        called |= set(re.findall(r"\bV\.(\w+)\(", "".join(c["source"])))
    assert called and all(callable(getattr(V, n, None)) for n in called), sorted(called)
    heads = ["".join(c["source"]).splitlines()[0] for c in cells if c["cell_type"] == "markdown"]
    for title in (
        "ChestX-ray14 corpus (training view)",
        "ChestX-ray14 probe tree",
        "CheXpert",
        "VinDr-CXR (Kaggle copy)",
        "Leakage gate",
        "Training view against probe view",
    ):
        assert any(title in h for h in heads), title
    # the path cell runs before any call and puts the repository on sys.path
    first_code = next(c for c in cells if c["cell_type"] == "code")
    assert "sys.path" in "".join(first_code["source"]) and "V." not in "".join(first_code["source"])


# ------------------------------------------------------------------- figures on the synthetic trees
def test_figures_render_on_a_synthetic_tree(w14, tmp_path):
    f = V.corpus_grid(w14["arrow"], w14["index"], "train", n=5)
    assert len(_panels(f)) == 5 and _panels(f)[0].shape == (224, 224, 3)
    g = V.tree_grid("chestxray14", "train", n=4, root=w14["root"])
    assert len(_panels(g)) == 4
    assert any("#1" in t for t in _texts(g))
    pair = V.raw_vs_exported("chestxray14", w14["arrow"], n=3, root=w14["root"])
    assert len(_panels(pair)) == 6
    ext = V.extreme_aspect("chestxray14", w14["arrow"], k=2, root=w14["root"])
    assert len(_panels(ext)) == 8
    assert isinstance(V.size_histograms("chestxray14", root=w14["root"]), matplotlib.figure.Figure)

    prev = V.prevalence("chestxray14", root=w14["root"])
    classes = json.load(open(os.path.join(w14["root"], "export_report.json")))["classes"]
    for split in ("train", "val", "test"):
        y = np.load(os.path.join(w14["root"], split, "labels.npz"))["y"]
        sub = prev[prev["split"] == split].set_index("class").loc[classes]
        assert sub["positives"].tolist() == y.sum(0).tolist() and (sub["n"] == y.shape[0]).all()
        np.testing.assert_allclose(sub["rate"], y.mean(0))
    assert prev["positives"].sum() > 0

    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "checks": [
                    {"name": "1 a", "passed": True, "detail": "n=3"},
                    {"name": "2 b", "passed": False, "detail": "n=1"},
                ],
                "skipped": ["chexpert"],
                "notes": [],
                "verdict": "CXR_LEAKAGE_GATE FAIL",
            }
        )
    )
    gs = V.gate_summary(str(gate))
    assert gs["status"].tolist() == ["PASS", "FAIL", "SKIPPED", "FAIL"]
    assert list(gs.columns) == ["check", "status", "detail"]

    gap = V.resample_gap(w14["arrow"], w14["index"], w14["root"], n=5)
    assert gap["n"] == 5 and set(gap) >= {"mean_abs_diff", "max_abs_diff"}


def test_resample_gap_is_zero_for_identical_paths_and_positive_otherwise(tmp_path, monkeypatch):
    # identical paths: square 8-bit greyscale sources -> 16 px stays 16 px in the tree, then both go to 224
    (tmp_path / "a").mkdir()
    same = _world14(tmp_path / "a", monkeypatch, _square_grey)
    gap = V.resample_gap(same["arrow"], same["index"], same["root"], n=100)
    assert gap["n"] == 10 and gap["mean_abs_diff"] == 0.0 and gap["max_abs_diff"] == 0
    # different paths: 20 x 16 sources are squashed by the training view and padded by the probe view
    (tmp_path / "b").mkdir()
    other = _world14(tmp_path / "b", monkeypatch, _mixed)
    gap = V.resample_gap(other["arrow"], other["index"], other["root"], n=100)
    assert gap["mean_abs_diff"] > 1.0 and gap["max_abs_diff"] > 10


def test_resample_gap_refuses_a_tree_row_that_points_at_another_image(tmp_path, monkeypatch):
    w = _world14(tmp_path, monkeypatch, _square_grey)
    path = os.path.join(w["root"], "export_table.csv")
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    tv = t["split"] != "test"
    t.loc[tv, "source_ref"] = (
        t.loc[tv, "source_ref"].iloc[::-1].tolist()
    )  # train/val rows name each other's shard row
    t.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different images"):
        V.resample_gap(w["arrow"], w["index"], w["root"], n=100)
    t.loc[tv, "source_ref"] = t.loc[t["split"] == "test", "source_ref"].iloc[:1].tolist() * int(
        tv.sum()
    )  # a test row
    t.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not a row of the index"):
        V.resample_gap(w["arrow"], w["index"], w["root"], n=100)


def test_raw_vs_exported_pairs_each_export_with_its_own_source_chestxray14(w14):
    fig = V.raw_vs_exported("chestxray14", w14["arrow"], n=6, root=w14["root"])
    table = pd.read_csv(os.path.join(w14["root"], "export_table.csv"), dtype=str, keep_default_na=False)
    by_pixels = {
        np.asarray(Image.open(os.path.join(w14["root"], r["split"], f"{r['name']}.png"))).tobytes(): r["name"]
        for _, r in table.iterrows()
    }
    assert len(by_pixels) == len(table)
    panels = _panels(fig)
    for k in range(0, len(panels), 2):
        name = by_pixels[np.asarray(panels[k + 1]).astype(np.uint8).tobytes()]
        want = np.asarray(Image.open(io.BytesIO(w14["src"][name])).convert("L"))
        np.testing.assert_array_equal(panels[k], want)
    assert len({tuple(p.shape) for p in panels}) >= 2  # source and padded export differ in shape


def _by_pixels(root, table):
    """exported PNG bytes -> (split, name) for every tree row (RGB, as the figures show them)."""
    out = {
        np.asarray(Image.open(os.path.join(root, r["split"], f"{r['name']}.png")).convert("RGB")).tobytes(): (
            r["split"],
            r["name"],
        )
        for _, r in table.iterrows()
    }
    assert len(out) == len(table)  # distinct pixels: a panel pairs to exactly one row
    return out


def _check_tree_grid_labels(key, root, split):
    tree = V._Tree(key, root)
    by = _by_pixels(root, tree.table)
    fig = V.tree_grid(key, split, n=10_000, root=root)  # every row of the split
    axes = [a for a in fig.axes if a.images]
    assert len(axes) == int((tree.table["split"] == split).sum()) > 1
    shown = set()
    for a in axes:
        sp, name = by[np.asarray(a.images[0].get_array()).astype(np.uint8).tobytes()]
        assert sp == split
        want = V._class_text(tree.classes, tree.y_of(split, name))
        assert a.get_title().split("\n", 1)[1] == want
        shown.add(want)
    assert len(shown) > 1  # the rows carry different label vectors, so a mismatch could show


def test_tree_grid_puts_each_rows_own_classes_under_its_image_chestxray14(w14):
    _check_tree_grid_labels("chestxray14", w14["root"], "train")


@pytest.mark.usefixtures("cs")
def test_tree_grid_puts_each_rows_own_classes_under_its_image_chexpert(tmp_path):
    _raw, root, _table, _k, _f = _chexpert_world(tmp_path)
    _check_tree_grid_labels("chexpert", root, "train")


def _varied(i):
    return _png(i, "L", (14 + i, 20 - i % 7))  # many different width/height ratios


def test_extreme_aspect_shows_the_widest_and_tallest_rows(tmp_path, monkeypatch):
    w = _world14(tmp_path, monkeypatch, _varied)
    table = pd.read_csv(os.path.join(w["root"], "export_table.csv"), dtype=str, keep_default_na=False)
    asp = table["src_w"].astype(int) / table["src_h"].astype(int)
    assert asp.nunique() > 3 and (asp == asp.max()).sum() == 1 and (asp == asp.min()).sum() == 1
    by = _by_pixels(w["root"], table)
    fig = V.extreme_aspect("chestxray14", w["arrow"], k=2, root=w["root"])
    ims = [a for a in fig.axes if a.images]
    for tag, want_idx in (("widest", asp.idxmax()), ("tallest", asp.idxmin())):
        hit = [j for j in range(0, len(ims), 2) if ims[j].get_title().startswith(f"{tag} #1 ")]
        assert len(hit) == 1
        _split, name = by[np.asarray(ims[hit[0] + 1].images[0].get_array()).astype(np.uint8).tobytes()]
        row = table.loc[want_idx]
        assert name == row["name"]
        assert f"source {row['src_w']}x{row['src_h']}" in ims[hit[0]].get_title()


def test_raw_vs_exported_refuses_a_shard_row_that_holds_another_image_chestxray14(w14):
    path = os.path.join(w14["root"], "export_table.csv")
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    a, b = t.index[0], t.index[1]
    t.loc[[a, b], "name"] = [t.loc[b, "name"], t.loc[a, "name"]]  # names swapped, source_ref kept
    t.to_csv(path, index=False)
    with pytest.raises(ValueError, match="shard row does not hold the tree image"):
        V.raw_vs_exported("chestxray14", w14["arrow"], n=100, root=w14["root"])


@pytest.mark.usefixtures("cs")
def test_y_of_without_a_label_row_does_not_quote_any_name(tmp_path):
    _raw, root, table, _k, _f = _chexpert_world(tmp_path)
    z = np.load(os.path.join(root, "train", "labels.npz"))
    np.savez(
        os.path.join(root, "train", "labels.npz"),
        names=np.array([f"zz{i}" for i in range(len(z["names"]))]),
        y=z["y"],
    )
    tree = V._Tree("chexpert", root)
    name = table[table["split"] == "train"]["name"].iloc[0]
    with pytest.raises(SystemExit) as ei:
        tree.y_of("train", name)
    msg = str(ei.value)
    assert msg == "chexpert: matching a label row failed (ValueError)"
    assert not any(n in msg for n in table["name"]) and ei.value.__suppress_context__


def test_tree_root_resolution(monkeypatch):
    assert V.tree_root("chexpert", "/x/y") == "/x/y"
    assert V.tree_root("chexpert") == os.path.join(F.REPO_ROOT, "data", "cxr", "chexpert")
    monkeypatch.setitem(F.OFFICIAL_CXR, "chexpert", {**F.OFFICIAL_CXR["chexpert"], "root": "/abs/tree"})
    assert V.tree_root("chexpert") == "/abs/tree"
    with pytest.raises(SystemExit):
        V.tree_root("nonsense")


# ------------------------------------------------------------------- main: refusals
def test_main_refuses_existing_out_dir(w14, tmp_path, capsys):
    out = tmp_path / "vis"
    out.mkdir()
    argv = [
        "--dataset",
        "chestxray14",
        "--root",
        w14["root"],
        "--arrow-dir",
        w14["arrow"],
        "--index",
        w14["index"],
    ]
    with pytest.raises(SystemExit, match="already exists"):
        V.main([*argv, "--out", str(out)])
    assert os.listdir(out) == []  # nothing written into it
    partial = tmp_path / "vis2.partial"
    partial.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        V.main([*argv, "--out", str(tmp_path / "vis2")])
    assert os.listdir(partial) == []


def test_main_chestxray14_end_to_end_names_files_by_function_and_split(w14, tmp_path, capsys):
    out = str(tmp_path / "vis")
    argv = ["--dataset", "chestxray14", "--root", w14["root"], "--arrow-dir", w14["arrow"]]
    assert V.main([*argv, "--index", w14["index"], "--out", out, "--n", "4"]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[-1] == "CXR_VISUAL_OK dataset=chestxray14 figures=8 tables=2"
    assert sorted(os.listdir(out)) == sorted(
        [
            "corpus_grid_train.png", "corpus_grid_val.png", "tree_grid_train.png", "tree_grid_val.png",
            "tree_grid_test.png", "raw_vs_exported.png", "extreme_aspect.png", "size_histograms.png",
            "prevalence.csv", "resample_gap.csv",
        ]
    )  # fmt: skip
    assert not os.path.exists(out + ".partial") and stat.S_IMODE(os.stat(out).st_mode) == 0o700
    assert "figure tree_grid_train.png panels=4" in text and "table prevalence.csv rows=" in text


# ------------------------------------------------------------------- the private-directory guard
@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_refuses_a_restricted_out_when_the_root_is_not_private(key, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o755))
    out = tmp_path / "open" / "vis"
    with pytest.raises(SystemExit, match="mode 700"):
        V.main(
            ["--dataset", key, "--out", str(out), "--raw-dir", str(tmp_path), "--private-root", str(tmp_path)]
        )
    assert not out.exists() and not out.parent.exists() and not os.path.exists(f"{out}.partial")


def test_private_dir_refuses_a_mode_700_directory_owned_by_someone_else(monkeypatch):
    """The owner half of the guard: mode 0o700, uid + 1."""
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o700, uid=os.getuid() + 1))
    assert V._is_private_dir("anywhere") is False
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o700))  # control: ours and 700 is accepted
    assert V._is_private_dir("anywhere") is True


def test_the_guard_is_for_restricted_keys_only(w14, monkeypatch, tmp_path):
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o755))
    argv = ["--dataset", "chestxray14", "--root", w14["root"], "--arrow-dir", w14["arrow"]]
    assert V.main([*argv, "--index", w14["index"], "--out", str(tmp_path / "v"), "--n", "2"]) == 0


def test_private_root_uses_the_real_stat_without_patching(w14, tmp_path):
    """No patch of V._stat: directories this test creates, real os.chmod / os.stat."""
    priv, opend = tmp_path / "priv", tmp_path / "open"
    for d, mode in ((priv, 0o700), (opend, 0o755)):
        d.mkdir()
        os.chmod(d, mode)
    argv = ["--dataset", "chestxray14", "--root", w14["root"], "--arrow-dir", w14["arrow"]]
    argv += ["--index", w14["index"], "--n", "2"]
    assert V.main([*argv, "--out", str(priv / "vis"), "--private-root", str(priv)]) == 0  # accepted
    assert "tree_grid_train.png" in os.listdir(priv / "vis")
    with pytest.raises(SystemExit, match="mode 700"):  # refused: the root itself is 0755
        V.main([*argv, "--out", str(opend / "vis"), "--private-root", str(opend)])
    with pytest.raises(SystemExit, match="mode 700"):  # restricted key, real stat as well
        V.main(_argv_restricted("chexpert", tmp_path, opend / "vis", "--private-root", str(opend)))
    assert os.listdir(opend) == []
    os.chmod(opend, 0o750)  # group bits count
    with pytest.raises(SystemExit, match="mode 700"):
        V.main(_argv_restricted("vindr_cxr_kaggle", tmp_path, opend / "vis", "--private-root", str(opend)))
    assert os.listdir(opend) == []


def _argv_restricted(key, tmp_path, out, *extra):
    return ["--dataset", key, "--out", str(out), "--raw-dir", str(tmp_path), *extra]


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_requires_a_private_root_for_restricted_keys(key, private, tmp_path):
    out = tmp_path / "vis"
    with pytest.raises(SystemExit, match="--private-root is required"):
        V.main(_argv_restricted(key, tmp_path, out))
    assert not out.exists() and not os.path.exists(f"{out}.partial")


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_refuses_out_outside_the_private_root(key, private, tmp_path, capsys):
    (tmp_path / "root").mkdir()
    (tmp_path / "elsewhere").mkdir()  # private too (the fixture), but not under --private-root
    for out in (tmp_path / "elsewhere" / "vis", tmp_path / "vis", tmp_path / "rootx" / "vis"):
        with pytest.raises(SystemExit) as ei:
            V.main(_argv_restricted(key, tmp_path, out, "--private-root", str(tmp_path / "root")))
        assert str(ei.value) == f"{key}: --out is not under --private-root; nothing written"
        assert not out.exists() and not os.path.exists(f"{out}.partial")
    assert not (tmp_path / "rootx").exists()


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_refuses_a_private_root_that_is_not_private(key, monkeypatch, tmp_path):
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o755))
    with pytest.raises(SystemExit, match="mode 700"):
        V.main(_argv_restricted(key, tmp_path, tmp_path / "vis", "--private-root", str(tmp_path)))
    assert not (tmp_path / "vis").exists()


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_refuses_symlinked_outs_live_and_dangling(key, private, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    live, dangling = root / "live", root / "dangling"
    os.symlink(target, live)
    os.symlink(root / "not_yet", dangling)
    for out in (live, dangling):
        with pytest.raises(SystemExit, match="already exists"):
            V.main(_argv_restricted(key, tmp_path, out, "--private-root", str(root)))
    assert os.listdir(target) == [] and not (root / "not_yet").exists()
    assert not os.path.lexists(f"{dangling}.partial") and not os.path.lexists(f"{live}.partial")
    # a dangling symlink at <out>.partial is refused as well
    os.symlink(root / "nowhere", f"{root / 'fresh'}.partial")
    with pytest.raises(SystemExit, match="already exists"):
        V.main(_argv_restricted(key, tmp_path, root / "fresh", "--private-root", str(root)))
    assert not (root / "nowhere").exists() and not (root / "fresh").exists()


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_refuses_a_symlinked_parent_that_leaves_the_private_root(key, private, tmp_path):
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    os.symlink(other, root / "hop")
    with pytest.raises(SystemExit, match="not under --private-root"):
        V.main(_argv_restricted(key, tmp_path, root / "hop" / "vis", "--private-root", str(root)))
    assert os.listdir(other) == []


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
def test_main_refuses_dotdot_through_a_symlink_that_leaves_the_private_root(key, private, tmp_path):
    root, other = tmp_path / "root", tmp_path / "other"
    (other / "sub").mkdir(parents=True)
    root.mkdir()
    os.symlink(other / "sub", root / "hop")
    with pytest.raises(SystemExit, match="not under --private-root"):
        V.main(_argv_restricted(key, tmp_path, f"{root}/hop/../vis", "--private-root", str(root)))
    assert os.listdir(other) == ["sub"] and os.listdir(other / "sub") == [] and os.listdir(root) == ["hop"]


@pytest.mark.parametrize("key", ["chexpert", "vindr_cxr_kaggle"])
@pytest.mark.parametrize("tail", ["/", "/.", "/..", "//"])
def test_main_refuses_an_out_without_a_leaf_name(key, tail, private, tmp_path):
    root = tmp_path / "root"
    (root / "vis").mkdir(parents=True)
    with pytest.raises(SystemExit, match="names no new directory"):
        V.main(_argv_restricted(key, tmp_path, f"{root}/vis{tail}", "--private-root", str(root)))
    assert sorted(os.listdir(root)) == ["vis"] and os.listdir(root / "vis") == []


def test_main_writes_a_relative_out_at_the_resolved_location(w14, private, tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    monkeypatch.chdir(root)
    argv = ["--dataset", "chestxray14", "--root", w14["root"], "--arrow-dir", w14["arrow"]]
    argv += ["--index", w14["index"], "--n", "2", "--private-root", str(root)]
    assert V.main([*argv, "--out", "sub/../vis_rel"]) == 0
    assert "tree_grid_train.png" in os.listdir(root / "vis_rel") and sorted(os.listdir(root)) == [
        "sub",
        "vis_rel",
    ]
    with pytest.raises(SystemExit, match="not under --private-root"):
        V.main([*argv, "--out", "../outside"])
    assert not (tmp_path / "outside").exists()


def test_main_accepts_a_symlinked_parent_that_stays_inside_the_private_root(w14, private, tmp_path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    os.symlink(root / "real", root / "link")
    argv = ["--dataset", "chestxray14", "--root", w14["root"], "--arrow-dir", w14["arrow"]]
    argv += ["--index", w14["index"], "--n", "2", "--private-root", str(root)]
    assert V.main([*argv, "--out", f"{root}/link/vis"]) == 0
    assert "tree_grid_train.png" in os.listdir(root / "real" / "vis")
    assert sorted(os.listdir(root)) == ["link", "real"] and not os.path.lexists(f"{root}/real/vis.partial")


@pytest.mark.usefixtures("cs")
def test_main_accepts_out_under_the_private_root_and_writes(private, tmp_path, capsys):
    raw, root, _table, _k, _f = _chexpert_world(tmp_path)
    priv = tmp_path / "priv"
    priv.mkdir()
    out = priv / "deep" / "vis"
    argv = ["--dataset", "chexpert", "--root", root, "--raw-dir", raw, "--n", "2", "--out", str(out)]
    assert V.main([*argv, "--private-root", str(priv)]) == 0
    assert "tree_grid_train.png" in os.listdir(out) and not os.path.lexists(f"{out}.partial")


def test_private_root_is_checked_for_unrestricted_keys_when_given(w14, monkeypatch, tmp_path):
    argv = ["--dataset", "chestxray14", "--root", w14["root"], "--arrow-dir", w14["arrow"]]
    argv += ["--index", w14["index"], "--n", "2"]
    monkeypatch.setattr(V, "_stat", lambda p: _St(0o700))
    (tmp_path / "root").mkdir()
    with pytest.raises(SystemExit, match="not under --private-root"):
        V.main([*argv, "--out", str(tmp_path / "v"), "--private-root", str(tmp_path / "root")])
    assert (
        V.main([*argv, "--out", str(tmp_path / "root" / "v"), "--private-root", str(tmp_path / "root")]) == 0
    )


# ------------------------------------------------------------------- CheXpert world
def _members(raw):
    out = []
    for p in sorted(os.path.join(raw, f) for f in os.listdir(raw) if f.endswith(".zip")):
        with zipfile.ZipFile(p) as zf:
            out += [i.filename for i in zf.infolist()]
    return out


def _chexpert_world(tmp_path):
    raw = _container(tmp_path)
    root = str(tmp_path / "tree_cx")
    assert _export(raw, root) == 0
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    sel = _selection(raw)
    key_of = {_name(k): k for k in sel["key"]}
    forbidden = {
        *table["name"], *table["patient"], "patient0", "CheXpert", ".jpg", "study",
        *[m for m in _members(raw) if not m.endswith(".csv")],
    }  # fmt: skip
    return raw, root, table, key_of, sorted(forbidden)


def _chexpert_source(raw, key):
    tail = "/" + key.split("/", 1)[1]
    for p in sorted(os.path.join(raw, f) for f in os.listdir(raw) if f.endswith(".zip")):
        with zipfile.ZipFile(p) as zf:
            for i in zf.infolist():
                if i.filename.endswith(tail):
                    return np.asarray(Image.open(io.BytesIO(zf.read(i))).convert("L"))
    raise AssertionError("fixture member missing")


def _no_leak(strings, forbidden):
    shown = "\n".join(strings)
    for bad in forbidden:
        assert bad not in shown, "a restricted string reached the output"


@pytest.mark.usefixtures("cs")
def test_chexpert_figures_carry_only_class_names_and_ordinals(tmp_path):
    raw, root, table, _key_of, forbidden = _chexpert_world(tmp_path)
    assert len(forbidden) > 20
    figs = [
        V.tree_grid("chexpert", "train", n=6, root=root),
        V.raw_vs_exported("chexpert", raw, n=4, root=root),
        V.extreme_aspect("chexpert", raw, k=2, root=root),
        V.size_histograms("chexpert", root=root),
    ]
    for fig in figs:
        _no_leak(_texts(fig), forbidden)
    texts = _texts(figs[0])
    assert any(t.startswith("#1") for t in texts)
    assert any(c in t for t in texts for c in S.CHEXPERT_CLASSES + ["no positive class"])
    # control: the instrument can see a leak (feed it a name)
    with pytest.raises(AssertionError):
        _no_leak([*texts, table["name"].iloc[0]], forbidden)


@pytest.mark.usefixtures("cs")
def test_raw_vs_exported_pairs_each_export_with_its_own_source_chexpert(tmp_path):
    raw, root, table, key_of, _ = _chexpert_world(tmp_path)
    fig = V.raw_vs_exported("chexpert", raw, n=6, root=root)
    by_pixels = {
        np.asarray(Image.open(os.path.join(root, r["split"], f"{r['name']}.png"))).tobytes(): r["name"]
        for _, r in table.iterrows()
    }
    assert len(by_pixels) == len(table)
    panels = _panels(fig)
    assert len(panels) == 12
    for k in range(0, len(panels), 2):
        name = by_pixels[
            np.asarray(panels[k + 1]).astype(np.uint8).tobytes()
        ]  # the exported panel is the PNG
        np.testing.assert_array_equal(panels[k], _chexpert_source(raw, key_of[name]))
    assert len({p.tobytes() for p in panels[::2]}) == 6  # six distinct sources (distinct fixture pixels)


@pytest.mark.usefixtures("cs")
def test_raw_vs_exported_refuses_a_row_whose_zip_ordinal_differs_from_source_ref(tmp_path):
    raw, root, table, _k, _ = _chexpert_world(tmp_path)
    table["source_ref"] = table["source_ref"].map({"0": "1", "1": "0"})
    table.to_csv(os.path.join(root, "export_table.csv"), index=False)
    with pytest.raises(SystemExit) as ei:
        V.raw_vs_exported("chexpert", raw, n=4, root=root)
    assert str(ei.value) == "chexpert: reading the sources failed (ValueError)"


@pytest.mark.usefixtures("cs")
def test_main_chexpert_success_prints_counts_only(private, tmp_path, capsys):
    raw, root, table, _k, forbidden = _chexpert_world(tmp_path)
    capsys.readouterr()
    out = str(tmp_path / "vis_cx")
    argv = ["--dataset", "chexpert", "--root", root, "--raw-dir", raw, "--out", out, "--n", "4"]
    assert V.main([*argv, "--private-root", str(tmp_path)]) == 0
    shown = capsys.readouterr()
    _no_leak([shown.out, shown.err], forbidden)
    assert shown.out.splitlines()[-1] == "CXR_VISUAL_OK dataset=chexpert figures=6 tables=1"
    assert sorted(os.listdir(out)) == [
        "extreme_aspect.png", "prevalence.csv", "raw_vs_exported.png", "size_histograms.png",
        "tree_grid_test.png", "tree_grid_train.png", "tree_grid_val.png",
    ]  # fmt: skip
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o700
    prev = pd.read_csv(os.path.join(out, "prevalence.csv"))
    assert set(prev["class"]) == set(S.CHEXPERT_CLASSES)


@pytest.mark.usefixtures("cs")
def test_main_chexpert_failure_is_count_only(private, tmp_path, capsys):
    raw, root, table, _k, forbidden = _chexpert_world(tmp_path)
    _rewrite(raw, lambda n, d: None if n.endswith(".csv") else b"not an image: XSECRETX")
    capsys.readouterr()
    with pytest.raises(SystemExit) as ei:
        V.main(
            ["--dataset", "chexpert", "--root", root, "--raw-dir", raw, "--out", str(tmp_path / "v")]
            + ["--private-root", str(tmp_path)]
        )
    shown = capsys.readouterr()
    msg = str(ei.value)
    assert re.fullmatch(r"chexpert: reading the sources failed \(\w+\)", msg), msg
    _no_leak([msg, shown.out, shown.err], [*forbidden, "XSECRETX"])
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


@pytest.mark.usefixtures("cs")
def test_main_chexpert_missing_tree_is_count_only(private, tmp_path, capsys):
    raw, root, table, _k, forbidden = _chexpert_world(tmp_path)
    with pytest.raises(SystemExit) as ei:
        V.main(
            [
                "--dataset",
                "chexpert",
                "--root",
                str(tmp_path / "nowhere"),
                "--raw-dir",
                raw,
                "--out",
                str(tmp_path / "v"),
                "--private-root",
                str(tmp_path),
            ]
        )
    assert str(ei.value) == "chexpert: reading the tree tables failed (FileNotFoundError)"
    assert str(tmp_path) not in str(ei.value)


# ------------------------------------------------------------------- VinDr-CXR world
def _vindr_world(tmp_path):
    raw = _vd_zip(tmp_path)
    root = str(tmp_path / "tree_vd")
    assert _vd_export(raw, root) == 0
    table = pd.read_csv(os.path.join(root, "export_table.csv"), dtype=str, keep_default_na=False)
    return raw, root, table


@pytest.mark.usefixtures("vd")
def test_vindr_figures_carry_only_class_names_and_ordinals(tmp_path):
    raw, root, table = _vindr_world(tmp_path)
    refs = [r for r in table["source_ref"] if ":" in r]
    assert len(refs) == 40
    figs = [
        V.tree_grid("vindr_cxr_kaggle", "val", n=4, root=root),
        V.raw_vs_exported("vindr_cxr_kaggle", raw, n=4, root=root),
        V.extreme_aspect("vindr_cxr_kaggle", raw, k=2, root=root),
        V.photometric_pairs(root, n=3),
        V.size_histograms("vindr_cxr_kaggle", root=root),
    ]
    for fig in figs:
        _no_leak(_texts(fig), [*VD_FORBIDDEN, *refs, "scan"])
    assert len(_panels(figs[3])) == 6
    assert any("MONOCHROME1 #1" in t for t in _texts(figs[3]))


@pytest.mark.usefixtures("vd")
def test_photometric_pairs_with_an_empty_class_says_so_with_a_count(tmp_path):
    _raw, root, table = _vindr_world(tmp_path)
    one = str(tmp_path / "tree_one")
    shutil.copytree(root, one)
    t = pd.read_csv(os.path.join(one, "export_table.csv"), dtype=str, keep_default_na=False)
    t["photometric"] = "MONOCHROME2"
    t.to_csv(os.path.join(one, "export_table.csv"), index=False)
    fig = V.photometric_pairs(one, n=3)
    texts = _texts(fig)
    assert not _panels(fig)
    assert any("no rows of MONOCHROME1" in s and "'MONOCHROME2': 40" in s for s in texts), texts


@pytest.mark.usefixtures("vd")
def test_raw_vs_exported_pairs_each_export_with_its_own_source_vindr(tmp_path):
    raw, root, table = _vindr_world(tmp_path)
    fig = V.raw_vs_exported("vindr_cxr_kaggle", raw, n=6, root=root)
    by_pixels = {
        np.asarray(Image.open(os.path.join(root, r["split"], f"{r['name']}.png"))).tobytes(): r["name"]
        for _, r in table.iterrows()
    }
    assert len(by_pixels) == 40
    panels = _panels(fig)
    assert len(panels) == 12
    with zipfile.ZipFile(os.path.join(raw, "kaggle.zip")) as zf:
        for k in range(0, len(panels), 2):
            name = by_pixels[np.asarray(panels[k + 1]).astype(np.uint8).tobytes()]
            want = DCM.decode_dicom_bytes(zf.read(f"train/{name}.dicom"))[0]
            np.testing.assert_array_equal(panels[k], want)
    assert len({p.tobytes() for p in panels[::2]}) == 6


@pytest.mark.usefixtures("vd")
def test_raw_vs_exported_refuses_a_vindr_row_whose_ordinal_differs_from_source_ref(tmp_path):
    raw, root, table = _vindr_world(tmp_path)
    table["source_ref"] = table["source_ref"].iloc[::-1].tolist()
    table.to_csv(os.path.join(root, "export_table.csv"), index=False)
    with pytest.raises(SystemExit) as ei:
        V.raw_vs_exported("vindr_cxr_kaggle", raw, n=4, root=root)
    assert str(ei.value) == "vindr_cxr_kaggle: reading the sources failed (ValueError)"


@pytest.mark.usefixtures("vd")
def test_main_vindr_success_and_failure_are_count_only(private, tmp_path, capsys):
    raw, root, table = _vindr_world(tmp_path)
    capsys.readouterr()
    out = str(tmp_path / "vis_vd")
    vd_argv = [
        "--dataset",
        "vindr_cxr_kaggle",
        "--root",
        root,
        "--raw-dir",
        raw,
        "--private-root",
        str(tmp_path),
    ]
    assert V.main([*vd_argv, "--out", out]) == 0
    shown = capsys.readouterr()
    _no_leak([shown.out, shown.err], [*VD_FORBIDDEN, "scan"])
    assert shown.out.splitlines()[-1] == "CXR_VISUAL_OK dataset=vindr_cxr_kaggle figures=7 tables=1"
    assert "photometric_pairs.png" in os.listdir(out)
    # failure: every member is garbage that cannot be decoded
    with zipfile.ZipFile(os.path.join(raw, "kaggle.zip")) as zf:
        csv_bytes = zf.read("train.csv")
    with zipfile.ZipFile(os.path.join(raw, "kaggle.zip"), "w") as zf:
        zf.writestr("train.csv", csv_bytes)
        for i in VD_IDS:
            zf.writestr(f"train/{i}.dicom", b"not a dicom: XSECRETX")
    with pytest.raises(SystemExit) as ei:
        V.main([*vd_argv, "--out", str(tmp_path / "v2")])
    shown = capsys.readouterr()
    msg = str(ei.value)
    assert re.fullmatch(r"vindr_cxr_kaggle: reading the sources failed \(\w+\)", msg), msg
    _no_leak([msg, shown.out, shown.err], [*VD_FORBIDDEN, "scan", "XSECRETX"])


@pytest.mark.usefixtures("vd")
def test_library_warnings_are_tallied_by_category_and_never_shown(private, tmp_path, capsys, monkeypatch):
    import warnings

    raw, root, _t = _vindr_world(tmp_path)
    real = DCM.decode_dicom_bytes
    monkeypatch.setattr(DCM, "decode_dicom_bytes", lambda b: warnings.warn("patient secret-id") or real(b))
    capsys.readouterr()
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        assert (
            V.main(
                [
                    "--dataset",
                    "vindr_cxr_kaggle",
                    "--root",
                    root,
                    "--raw-dir",
                    raw,
                    "--out",
                    str(tmp_path / "v"),
                    "--private-root",
                    str(tmp_path),
                ]
            )
            == 0
        )
    shown = capsys.readouterr()
    assert "secret-id" not in shown.out + shown.err and not [
        w for w in outer if "secret-id" in str(w.message)
    ]
    assert "library warnings by category: {'UserWarning':" in shown.out
