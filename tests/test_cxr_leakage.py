"""Leakage gate (eval/cxr_leakage.py) on a synthetic corpus and synthetic trees. Every check is fed the
mismatch it exists to catch (rule 6). No restricted data is touched: the CheXpert tree comes from the
synthetic zip container of test_export_cxr, the VinDr-CXR Kaggle tree is written with write_tree."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from test_cxr14 import corpus  # noqa: F401  (fixture used by cx)
from test_export_cxr import _container, _img, cx  # noqa: F401  (cx is a fixture)

from datasets import cxr14 as G
from eval import cxr_leakage as L
from eval import cxr_splits as S
from eval import export_cxr as X
from eval import features as F

SIZES = {
    "chestxray14": {"train": 8, "val": 4, "test": 20},
    "chexpert": {"train": 4, "val": 1, "test": 2},
    "vindr_cxr_kaggle": {"train": 3, "val": 2, "test": 2},
}


def _edit_table(root, fn):
    path = os.path.join(root, "export_table.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fn(df)
    df.to_csv(path, index=False)


def _vindr_splits():
    out = {}
    for s, n in SIZES["vindr_cxr_kaggle"].items():
        out[s] = [
            (f"vd_{s}_{i}", np.array([i % 2, 1 - i % 2], np.uint8), (lambda k=i: _img(seed=k)))
            for i in range(n)
        ]
    return out


@pytest.fixture
def g(cx, tmp_path, monkeypatch):  # noqa: F811
    """A clean build: index + three trees, literals and OFFICIAL_CXR sizes moved to the synthetic corpus."""
    npz = str(tmp_path / "index.npz")
    G.build_index(cx["man"], npz)
    t14 = str(tmp_path / "trees" / "chestxray14")
    splits, extra, classes = X.chestxray14_splits(cx["arrow"], cx["man"], cx["raw"])
    X.write_tree(t14, "chestxray14", splits, classes, extra, restricted=False)
    for k, v in {"train": 4, "val": 1}.items():
        monkeypatch.setitem(S.PROBE_SIZES, k, v)
    cxp_dir = tmp_path / "cxp"
    cxp_dir.mkdir()
    raw = _container(cxp_dir)
    tcx = str(tmp_path / "trees" / "chexpert")
    assert X.main(["--dataset", "chexpert", "--out", tcx, "--raw-dir", raw, "--max-samples", "100"]) == 0
    tvd = str(tmp_path / "trees" / "vindr_cxr_kaggle")
    X.write_tree(tvd, "vindr_cxr_kaggle", _vindr_splits(), ["a", "b"], {}, restricted=True)
    for k, sizes in SIZES.items():
        monkeypatch.setitem(F.OFFICIAL_CXR, k, {**F.OFFICIAL_CXR[k], "sizes": dict(sizes)})
    z = np.load(npz)
    lists = {
        f: [x for x in open(os.path.join(cx["raw"], f)).read().split("\n") if x] for f in G.LABEL_FILES[1:]
    }
    monkeypatch.setattr(
        L,
        "PINNED",
        {
            "index_retained": int((z["bucket_code"] == 0).sum()),
            "index_validation": int((z["bucket_code"] == 1).sum()),
            "train_val_list": len(lists["train_val_list.txt"]),
            "test_list": len(lists["test_list.txt"]),
            "csv_rows": len(cx["rows"]),
        },
    )
    return {
        "npz": npz,
        "man": cx["man"],
        "raw": cx["raw"],
        "trees": {"chestxray14": t14, "chexpert": tcx, "vindr_cxr_kaggle": tvd},
        "tmp": tmp_path,
        "n": 0,
    }


def _run(g, capsys, trees=None):
    g["n"] += 1
    out = str(g["tmp"] / f"gate{g['n']}.json")
    argv = ["--index", g["npz"], "--manifest-dir", g["man"], "--raw-dir", g["raw"], "--out", out]
    for k, v in (g["trees"] if trees is None else trees).items():
        argv += ["--tree", f"{k}={v}"]
    rc = L.main(argv)
    text = capsys.readouterr()
    return rc, text.out + text.err, json.load(open(out))


def _status(rec):
    return {c["name"]: c["passed"] for c in rec["checks"]}


def _failed(rec):
    return {n.split()[0] for n, ok in _status(rec).items() if not ok}


# ------------------------------------------------------------------- clean build
def test_gate_passes_on_a_clean_build(g, capsys):
    rc, out, rec = _run(g, capsys)
    assert rc == 0 and out.rstrip("\n").splitlines()[-1] == "CXR_LEAKAGE_GATE PASS"
    assert all(_status(rec).values()) and rec["verdict"] == "CXR_LEAKAGE_GATE PASS" and rec["skipped"] == []
    assert {n.split()[0] for n in _status(rec)} == {"1", "1b", "1c", "2", "2b", "3", "4", "5", "6", "7", "8"}
    assert sum(n.startswith("8 ") for n in _status(rec)) == 3
    for note in L.NOTES:
        assert f"NOTE {note}" in out
    assert "NOTE no patient identifier in this release: patient overlap cannot be checked" in out
    assert not os.path.exists(g["tmp"] / "gate1.json.partial")
    assert check_matches_main(g, rec)


def check_matches_main(g, rec):
    res = L.check(g["npz"], g["man"], g["raw"], g["trees"])
    return [(n, ok) for n, ok, _ in res] == [(c["name"], c["passed"]) for c in rec["checks"]]


# ------------------------------------------------------------------- check 1 and 2
def _edit_index(g, fn):
    z = dict(np.load(g["npz"]))
    z = fn(z) or z
    os.rename(g["npz"], g["npz"] + ".orig")
    np.savez(g["npz"], **z)


def _index_names(g):
    """Image name of every index row, in npz order (test-side join through rows.csv)."""
    rows = pd.read_csv(os.path.join(g["man"], "rows.csv"), dtype={"img_name": str}).set_index(
        ["shard", "row"]
    )
    z = np.load(g["npz"])
    return [rows.at[(int(a), int(b)), "img_name"] for a, b in zip(z["shard"], z["row"], strict=True)]


def _tree_names(g, split):
    t = pd.read_csv(os.path.join(g["trees"]["chestxray14"], "export_table.csv"), dtype=str)
    return [n + ".png" for n in t.loc[t["split"] == split, "name"]]


@pytest.mark.parametrize("bucket_code", [0, 1], ids=["retained", "validation"])
def test_gate_fails_when_a_test_patient_is_injected_into_the_index(g, capsys, bucket_code):
    rows = pd.read_csv(os.path.join(g["man"], "rows.csv"), dtype={"img_name": str})
    t = rows[rows["bucket"] == "test"].iloc[0]

    def inject(z):
        for k, v in (
            ("shard", t["shard"]),
            ("row", t["row"]),
            ("bucket_code", bucket_code),
            ("patient_id", t["patient_id"]),
        ):
            z[k] = np.append(z[k], np.array(v, z[k].dtype))

    _edit_index(g, inject)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and out.rstrip("\n").splitlines()[-1] == "CXR_LEAKAGE_GATE FAIL"
    assert {"1", "2"} <= _failed(rec)
    assert "FAIL 1 " in out and "FAIL 2 " in out


def test_gate_fails_when_a_tree_val_patient_is_in_the_retained_bucket(g, capsys):
    val = set(_tree_names(g, "val"))
    pos = next(i for i, n in enumerate(_index_names(g)) if n in val)

    def move(z):
        z["bucket_code"] = z["bucket_code"].copy()
        z["bucket_code"][pos] = 0

    _edit_index(g, move)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and "1" in _failed(rec) and "2" in _failed(rec)
    assert "retained vs val+test shared 1" in out


def test_gate_fails_when_a_patient_is_in_both_index_buckets(g, capsys):
    train = set(_tree_names(g, "train"))
    pos = next(
        i
        for i, (n, b) in enumerate(zip(_index_names(g), np.load(g["npz"])["bucket_code"], strict=True))
        if b == 0 and n not in train
    )

    def move(z):  # one retained image goes to validation: its patient is now in both buckets
        z["bucket_code"] = z["bucket_code"].copy()
        z["bucket_code"][pos] = 1

    _edit_index(g, move)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and "2b" in _failed(rec) and not ({"1", "2"} & _failed(rec))


def test_gate_fails_when_tree_names_do_not_resolve_in_the_csv(g, capsys):
    """A format drift (here: the .png suffix left on the tree names) must not make checks 1 and 2 pass
    vacuously on two sets that cannot intersect."""
    _edit_table(g["trees"]["chestxray14"], lambda df: df.__setitem__("name", df["name"] + ".png"))
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and {"1", "2"} <= _failed(rec)
    detail = {c["name"].split()[0]: c["detail"] for c in rec["checks"]}
    assert "do not resolve to exactly one CSV row" in detail["1"] and "24" in detail["1"]


def test_gate_fails_when_an_index_name_does_not_resolve_in_the_csv(g, capsys):
    path = os.path.join(g["raw"], G.LABEL_FILES[0])
    df = pd.read_csv(path, dtype=str)
    retained = pd.read_csv(os.path.join(g["man"], "retained.csv"), dtype=str)["img_name"].iloc[0]
    df.loc[df["Image Index"] == retained, "Image Index"] = "renamed.png"
    df.to_csv(path, index=False)
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and {"1", "1b", "2"} <= _failed(rec)


def test_gate_fails_when_the_index_patient_column_disagrees_with_the_csv(g, capsys):
    z = dict(np.load(g["npz"]))
    z["patient_id"] = z["patient_id"].copy()
    z["patient_id"][0] += 1000
    os.rename(g["npz"], g["npz"] + ".orig")
    np.savez(g["npz"], **z)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and "1b" in _failed(rec) and "1 disagree" in out


def test_gate_fails_when_the_tree_patient_column_disagrees_with_the_csv(g, capsys):
    _edit_table(g["trees"]["chestxray14"], lambda df: df.__setitem__("patient", "424242"))
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and "1c" in _failed(rec)


# ------------------------------------------------------------------- check 3
def test_gate_fails_when_a_patient_is_in_train_val_and_test_lists(g, capsys):
    tv = open(os.path.join(g["raw"], "train_val_list.txt")).read().split("\n")[0]
    with open(os.path.join(g["raw"], "test_list.txt"), "a") as f:
        f.write("\n" + tv)
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and "3" in _failed(rec)


# ------------------------------------------------------------------- check 4
def test_gate_fails_when_a_tree_val_image_is_outside_the_validation_bucket(g, capsys):
    def edit(df):
        df.loc[df.index[df["split"] == "train"][0], "split"] = "val"  # a retained image now sits in val

    _edit_table(g["trees"]["chestxray14"], edit)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and "4" in _failed(rec) and "1 val names outside validation" in out


def test_gate_fails_when_a_tree_train_image_is_outside_the_retained_bucket(g, capsys):
    def edit(df):
        df.loc[df.index[df["split"] == "val"][0], "split"] = "train"

    _edit_table(g["trees"]["chestxray14"], edit)
    _, out, rec = _run(g, capsys)
    assert "4" in _failed(rec) and "1 train names outside retained" in out


def test_gate_fails_when_a_tree_test_split_is_not_the_official_list(g, capsys):
    def edit(df):
        df.loc[df.index[df["split"] == "test"][0], "split"] = "train"

    _edit_table(g["trees"]["chestxray14"], edit)
    _, out, rec = _run(g, capsys)
    assert "4" in _failed(rec) and "1 official test names missing" in out


# ------------------------------------------------------------------- check 5
@pytest.mark.parametrize("field", list(L.PINNED))
def test_gate_fails_when_a_pinned_count_differs(g, capsys, monkeypatch, field):
    monkeypatch.setattr(L, "PINNED", {**L.PINNED, field: L.PINNED[field] + 1})
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and _failed(rec) == {"5"} and field in out


# ------------------------------------------------------------------- check 6
def test_gate_fails_when_a_chexpert_patient_is_in_two_splits(g, capsys):
    def edit(df):
        tr = df.loc[df["split"] == "train", "patient"].iloc[0]
        df.loc[df.index[df["split"] == "val"][0], "patient"] = tr

    _edit_table(g["trees"]["chexpert"], edit)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and _failed(rec) == {"6"} and "train/val 1" in out


def test_gate_fails_when_a_chexpert_row_is_not_frontal(g, capsys):
    _edit_table(
        g["trees"]["chexpert"],
        lambda df: df.__setitem__("frontal_lateral", ["Lateral"] + ["Frontal"] * (len(df) - 1)),
    )
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and _failed(rec) == {"6"} and "1 not frontal" in out


def test_gate_fails_when_chexpert_patients_are_blank(g, capsys):
    """Blank patient ids would make 'no shared patient' vacuous."""
    _edit_table(g["trees"]["chexpert"], lambda df: df.__setitem__("patient", ""))
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and _failed(rec) == {"6"}


# ------------------------------------------------------------------- check 7
def test_gate_fails_when_a_vindr_image_id_is_in_two_splits(g, capsys):
    def edit(df):
        df.loc[df.index[df["split"] == "val"][0], "name"] = df.loc[df["split"] == "train", "name"].iloc[0]

    _edit_table(g["trees"]["vindr_cxr_kaggle"], edit)
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and "7" in _failed(rec) and "train/val 1" in out


# ------------------------------------------------------------------- check 8
def test_gate_fails_when_a_tree_size_differs_from_official(g, capsys, monkeypatch):
    sizes = {**SIZES["chexpert"], "train": 5}
    monkeypatch.setitem(F.OFFICIAL_CXR, "chexpert", {**F.OFFICIAL_CXR["chexpert"], "sizes": sizes})
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and _failed(rec) == {"8"} and not _status(rec)["8 tree_sizes_match_official[chexpert]"]


def test_gate_fails_when_a_png_is_missing_from_a_split_directory(g, capsys):
    d = os.path.join(g["trees"]["vindr_cxr_kaggle"], "train")
    os.remove(os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".png"))[0]))
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and not _status(rec)["8 tree_sizes_match_official[vindr_cxr_kaggle]"]


def test_gate_fails_when_labels_and_table_name_different_images(g, capsys):
    p = os.path.join(g["trees"]["vindr_cxr_kaggle"], "val", "labels.npz")
    z = dict(np.load(p))
    z["names"] = np.array(["other0", "other1"])
    np.savez(p, **z)
    rc, _, rec = _run(g, capsys)
    assert rc != 0 and not _status(rec)["8 tree_sizes_match_official[vindr_cxr_kaggle]"]


# ------------------------------------------------------------------- completeness and output
def test_gate_is_incomplete_when_a_tree_is_missing(g, capsys):
    trees = {k: v for k, v in g["trees"].items() if k != "vindr_cxr_kaggle"}
    rc, out, rec = _run(g, capsys, trees)
    lines = out.rstrip("\n").splitlines()
    assert rc != 0 and lines[-1] == "CXR_LEAKAGE_GATE INCOMPLETE" and "SKIPPED vindr_cxr_kaggle" in lines
    assert rec["skipped"] == ["vindr_cxr_kaggle"] and all(_status(rec).values())
    assert not any(n.startswith("7 ") for n in _status(rec))
    # a tree key whose directory does not exist is skipped the same way
    rc, out, _ = _run(g, capsys, {**trees, "vindr_cxr_kaggle": str(g["tmp"] / "nowhere")})
    assert rc != 0 and "SKIPPED vindr_cxr_kaggle" in out


def test_gate_fail_outranks_incomplete(g, capsys):
    _edit_table(g["trees"]["chexpert"], lambda df: df.__setitem__("patient", ""))
    trees = {k: v for k, v in g["trees"].items() if k != "vindr_cxr_kaggle"}
    rc, out, _ = _run(g, capsys, trees)
    assert rc != 0 and out.rstrip("\n").splitlines()[-1] == "CXR_LEAKAGE_GATE FAIL"
    assert "SKIPPED vindr_cxr_kaggle" in out


def test_gate_output_for_restricted_trees_has_counts_only(g, capsys):
    rc, out, rec = _run(g, capsys)
    text = out + json.dumps(rec)
    assert rc == 0 and "patient0" not in text and "vd_" not in text
    names = set()
    for k in ("chexpert", "vindr_cxr_kaggle"):
        t = pd.read_csv(os.path.join(g["trees"][k], "export_table.csv"), dtype=str, keep_default_na=False)
        names |= set(t["name"]) | set(t["patient"])
    assert len(names) > 5 and not any(n and n in text for n in names)
    # the failing path is counts only as well
    _edit_table(g["trees"]["chexpert"], lambda df: df.__setitem__("patient", df["patient"].iloc[0]))
    _, out, rec = _run(g, capsys)
    text = out + json.dumps(rec)
    assert "patient0" not in text and not any(n and n in text for n in names)


def test_unreadable_tree_file_is_a_fail_line_without_exception_text(g, capsys):
    os.remove(os.path.join(g["trees"]["chexpert"], "export_table.csv"))
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and {"6", "8"} <= _failed(rec)
    assert (
        "6 chexpert_patient_disjoint_and_frontal: cannot read export_table.csv of chexpert (FileNotFoundError)"
        in out
    )
    assert "No such file" not in out and "export_table.csv'" not in out and str(g["tmp"]) not in out


def test_unreadable_labels_npz_is_a_fail_line_without_exception_text(g, capsys):
    p = os.path.join(g["trees"]["vindr_cxr_kaggle"], "test", "labels.npz")
    open(p, "wb").write(b"not an npz")
    rc, out, rec = _run(g, capsys)
    assert rc != 0 and not _status(rec)["8 tree_sizes_match_official[vindr_cxr_kaggle]"]
    assert "cannot read labels.npz of vindr_cxr_kaggle (" in out and "not an npz" not in out


def test_existing_out_is_fatal_before_any_work(g, monkeypatch):
    out = g["tmp"] / "taken.json"
    out.write_text("keep")
    monkeypatch.setattr(L, "_run", lambda *a: pytest.fail("work started"))
    argv = ["--index", g["npz"], "--manifest-dir", g["man"], "--raw-dir", g["raw"], "--out", str(out)]
    with pytest.raises(SystemExit, match="exists"):
        L.main(argv)
    assert out.read_text() == "keep"
    (g["tmp"] / "other.json.partial").write_text("x")
    with pytest.raises(SystemExit, match="exists"):
        L.main([*argv[:-1], str(g["tmp"] / "other.json")])


def test_tree_option_rejects_unknown_key(g):
    argv = ["--index", g["npz"], "--manifest-dir", g["man"], "--raw-dir", g["raw"]]
    with pytest.raises(SystemExit):
        L.main([*argv, "--out", str(g["tmp"] / "o.json"), "--tree", "nope=/x"])
