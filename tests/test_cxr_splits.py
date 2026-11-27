"""Pinned split and label rules for the chest X-ray datasets (eval/cxr_splits.py).

CheXpert and VinDr-CXR tests use synthetic frames only; nothing restricted is read.
"""

import collections
import contextlib
import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from eval import cxr_splits as S

CXR14_RAW = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr/chestxray14/raw"


def need(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"inputs absent: {missing}")


def load_cxr14_lists(raw):
    df = pd.read_csv(
        os.path.join(raw, "Data_Entry_2017_v2020.csv"),
        usecols=["Image Index", "Patient ID"],
        dtype=str,
    )
    patient_of = dict(zip(df["Image Index"], df["Patient ID"], strict=True))

    def names(fn):
        with open(os.path.join(raw, fn)) as f:
            return {line.strip() for line in f if line.strip()}

    return patient_of, names("train_val_list.txt"), names("test_list.txt")


@contextlib.contextmanager
def monkeypatch_sizes(module, **sizes):
    with mock.patch.dict(module.PROBE_SIZES, sizes):
        yield


def fake_chexpert_frames(n_patients, images_per_patient, n_valid_patients=20, valid_offset=90000):
    def frame(split, first, count):
        rows = []
        for p in range(first, first + count):
            for j in range(images_per_patient):
                lateral = j % 5 == 4
                view = "lateral" if lateral else "frontal"
                row = {
                    "Path": f"CheXpert-v1.0/{split}/patient{p:05d}/study1/view{j}_{view}.jpg",
                    "Frontal/Lateral": "Lateral" if lateral else "Frontal",
                }
                for k, c in enumerate(S.CHEXPERT_CLASSES):
                    row[c] = [1.0, 0.0, -1.0, np.nan][(p + j + k) % 4]
                rows.append(row)
        return pd.DataFrame(rows)

    return frame("train", 1, n_patients), frame("valid", valid_offset, n_valid_patients)


def test_hash_rank_is_pinned():
    assert S.hash_rank("patient", "1") == 8271735379821893146
    assert S.hash_rank("image", "00000001_000.png") == 3443901661694405644


def test_draw_is_nested_and_refuses_shortfall():
    keys = [f"k{i}" for i in range(100)]
    assert S.draw(keys, 10) == S.draw(keys, 40)[:10] == S.draw(reversed(keys), 10)
    with pytest.raises(ValueError):
        S.draw(keys, 101)
    with pytest.raises(ValueError):
        S.ranked("image", ["a", "a"])


def test_pick_val_patients_is_a_hashed_fraction():
    pats = [str(i) for i in range(50)]
    v = S.pick_val_patients(pats)
    assert len(v) == 5 and v == set(S.ranked("patient", pats)[:5])
    assert v == S.pick_val_patients(reversed(pats))


def test_cxr14_buckets_against_the_shipped_lists():
    need(CXR14_RAW)
    patient_of, train_val, test = load_cxr14_lists(CXR14_RAW)
    b = S.cxr14_buckets(patient_of, train_val, test)
    n = collections.Counter(b.values())
    assert n == {"align": 77780, "validation": 8744, "test": 25596}

    def pats(k):
        return {patient_of[i] for i, v in b.items() if v == k}

    assert (len(pats("align")), len(pats("validation")), len(pats("test"))) == (25207, 2801, 2797)
    assert not (pats("align") & pats("validation"))
    assert not (pats("align") & pats("test"))
    assert not (pats("validation") & pats("test"))
    assert S.draw([i for i, v in b.items() if v == "align"], 16000)[0] == "00016486_000.png"
    assert S.draw([i for i, v in b.items() if v == "validation"], 2000)[0] == "00007291_000.png"


def test_cxr14_buckets_refuses_a_patient_on_both_sides():
    with pytest.raises(ValueError, match="both sides"):
        S.cxr14_buckets({"a_0.png": "1", "a_1.png": "1"}, {"a_0.png"}, {"a_1.png"})


def test_cxr14_buckets_refuses_bad_lists():
    p = {"a.png": "1", "b.png": "2"}
    with pytest.raises(ValueError, match="both official lists"):
        S.cxr14_buckets(p, {"a.png", "b.png"}, {"b.png"})
    with pytest.raises(ValueError, match="neither"):
        S.cxr14_buckets(p, {"a.png"}, set())
    with pytest.raises(ValueError, match="no patient"):
        S.cxr14_buckets({"a.png": "1"}, {"a.png", "z.png"}, set())


def test_cxr14_labels():
    y = S.cxr14_labels(["No Finding", "Effusion|Hernia"])
    assert y.dtype == np.uint8
    assert y.tolist() == [[0] * 14, [0, 0, 1] + [0] * 10 + [1]]
    with pytest.raises(ValueError):
        S.cxr14_labels(["Effusions"])


def test_chexpert_split_is_patient_disjoint_frontal_and_u_zeros():
    train, valid = fake_chexpert_frames(n_patients=400, images_per_patient=5)
    with monkeypatch_sizes(S, train=160, val=20):
        t = S.chexpert_split(train, valid)
    by = {s: set(t.patient[t.split == s]) for s in ("train", "val", "test")}
    assert not (by["train"] & by["val"]) and not (by["train"] & by["test"]) and not (by["val"] & by["test"])
    assert (t.split == "train").sum() == 160 and (t.split == "val").sum() == 20
    assert (t.split == "test").sum() == 20 * 4
    assert not t.key.str.contains("lateral").any()
    assert list(t.columns) == ["key", "patient", "split", "draw_rank"]
    # validation patients come from the hashed pool of train patients, train from the rest
    pool = S.pick_val_patients({f"patient{p:05d}" for p in range(1, 401)})
    assert by["val"] <= pool and not (by["train"] & pool)
    # draw order: draw_rank is the position in draw(); test rank is the position in ranked()
    for s in ("train", "val"):
        part = t[t.split == s]
        assert part.draw_rank.tolist() == list(range(len(part)))
    assert t.split.tolist() == sorted(t.split, key=["train", "val", "test"].index)
    test = t[t.split == "test"]
    assert test.key.tolist() == S.ranked("image", test.key)
    assert S.chexpert_key("CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg") == (
        "train/patient00001/study1/view1_frontal.jpg"
    )
    assert S.chexpert_patient("/x/valid/train/patient00007/study2/v.jpg") == "patient00007"
    assert S.chexpert_key("/x/valid/train/patient00007/study2/v.jpg") == "train/patient00007/study2/v.jpg"


def test_chexpert_labels_u_zeros_and_refusals():
    cells = {c: [v] for c, v in zip(S.CHEXPERT_CLASSES, [1, -1, 0, np.nan, 1.0])}
    assert S.chexpert_labels(pd.DataFrame(cells)).tolist() == [[1, 0, 0, 0, 1]]
    assert S.chexpert_labels(pd.DataFrame(cells)).dtype == np.uint8
    for bad in (2, 0.5):
        broken = pd.DataFrame({**cells, "Edema": [bad]})
        with pytest.raises(ValueError, match="1 label cells"):
            S.chexpert_labels(broken)


def test_chexpert_key_refuses_a_path_without_patient_directory():
    for bad in ("a/b/c.jpg", "train/study1/x.jpg", "valid/patientX/s/v.jpg"):
        with pytest.raises(ValueError):
            S.chexpert_key(bad)
        with pytest.raises(ValueError):
            S.chexpert_patient(bad)


def test_chexpert_split_refuses_patient_in_both_csvs():
    train, valid = fake_chexpert_frames(n_patients=50, images_per_patient=5)
    clash = valid.copy()
    clash.loc[0, "Path"] = "CheXpert-v1.0/valid/patient00003/study1/view0_frontal.jpg"
    with monkeypatch_sizes(S, train=10, val=2):
        with pytest.raises(ValueError, match="both") as ei:
            S.chexpert_split(train, clash)
    assert "patient00003" not in str(ei.value) and "CheXpert-v1.0" not in str(ei.value)


def test_chexpert_split_refuses_a_pool_shortfall_without_naming_rows():
    train, valid = fake_chexpert_frames(n_patients=50, images_per_patient=5)
    with monkeypatch_sizes(S, train=10**6, val=2):
        with pytest.raises(ValueError, match="cannot draw") as ei:
            S.chexpert_split(train, valid)
    assert "patient" not in str(ei.value) and "view" not in str(ei.value)


def test_vindr_two_of_three_and_label_set():
    rows = pd.DataFrame(
        {
            "image_id": ["a"] * 4 + ["b"] * 3,
            "class_name": ["X", "X", "Y", "No finding", "No finding", "No finding", "Y"],
            "rad_id": ["R1", "R2", "R3", "R3", "R1", "R2", "R3"],
        }
    )
    assert S.vindr_two_of_three(rows, ["a", "b"], ["X", "Y"]).tolist() == [[1, 0], [0, 0]]
    with pytest.raises(ValueError, match="3 radiologists") as ei:
        S.vindr_two_of_three(rows[rows.rad_id != "R2"], ["a", "b"], ["X", "Y"])
    assert "2 images" in str(ei.value)
    # an image with no rows at all is not "exactly 3 radiologists" either
    with pytest.raises(ValueError, match="3 radiologists"):
        S.vindr_two_of_three(rows, ["a", "b", "c"], ["X", "Y"])


def test_vindr_two_of_three_message_carries_no_image_id():
    rows = pd.DataFrame({"image_id": ["secretimage"] * 2, "class_name": ["X", "X"], "rad_id": ["R1", "R2"]})
    with pytest.raises(ValueError) as ei:
        S.vindr_two_of_three(rows, ["secretimage"], ["X"])
    assert "secretimage" not in str(ei.value)


def test_vindr_label_set_threshold_and_exclusions():
    # 12 test images, 3 radiologists each (R3 always marks "No finding"). P: 2-of-3 on 10 images (kept);
    # Q: 2-of-3 on 9 (below threshold); R: 1 of 3 on all 12; "Other lesion" is excluded despite 2-of-3.
    recs = []
    for i in range(12):
        img = f"i{i}"
        recs.append((img, "No finding", "R3"))
        recs.append((img, "R", "R1"))
        for rad in ("R1", "R2"):
            recs.append((img, "Other lesion", rad))
            if i < 10:
                recs.append((img, "P", rad))
            if i < 9:
                recs.append((img, "Q", rad))
    rows = pd.DataFrame(recs, columns=["image_id", "class_name", "rad_id"])
    ids = [f"i{i}" for i in range(12)]
    assert S.vindr_label_set(rows, ids) == ["P"]
    assert S.vindr_label_set(rows, ids[:9]) == []  # P drops to 9 positives
    assert S.vindr_label_set(rows, ids[:10]) == ["P"]  # exactly 10 counts


def test_vindr_kaggle_split_sizes():
    s = S.vindr_kaggle_split([f"{i:032x}" for i in range(15000)])
    assert {k: len(v) for k, v in s.items()} == S.VINDR_KAGGLE_SIZES
    assert len(set(sum(s.values(), []))) == 15000
    order = S.ranked("image", [f"{i:032x}" for i in range(15000)])
    assert s["test"] == order[:3000] and s["val"] == order[3000:4500] and s["train"] == order[4500:]
    with pytest.raises(ValueError):
        S.vindr_kaggle_split(["a"] * 15000)
    with pytest.raises(ValueError):
        S.vindr_kaggle_split([f"{i:032x}" for i in range(14999)])


def test_chexpert_labels_refuses_string_and_bool_columns():
    cells = {c: [v] for c, v in zip(S.CHEXPERT_CLASSES, [1, -1, 0, np.nan, 1.0])}
    for bad in ("1", True):
        with pytest.raises(ValueError, match="1 label columns are not numeric") as ei:
            S.chexpert_labels(pd.DataFrame({**cells, "Edema": [bad]}))
        assert "'1'" not in str(ei.value) and "True" not in str(ei.value)


def test_cxr14_buckets_refuses_non_string_patient_ids():
    for bad in (float("nan"), ""):
        with pytest.raises(ValueError, match="1 patient ids"):
            S.cxr14_buckets({"a.png": bad, "b.png": "2"}, {"a.png"}, {"b.png"})


def test_ranked_refuses_non_string_keys():
    with pytest.raises(ValueError, match="1 keys are not strings"):
        S.ranked("image", ["a", 7])
    with pytest.raises(ValueError, match="1 keys are not strings"):
        S.draw(["a", float("nan")], 1)


def test_vindr_two_of_three_counts_distinct_radiologists():
    def rows_for(extra):
        base = [("a", "No finding", "R2"), ("a", "No finding", "R3")] + extra
        return pd.DataFrame(base, columns=["image_id", "class_name", "rad_id"])

    twice = rows_for([("a", "X", "R1"), ("a", "X", "R1")])
    assert S.vindr_two_of_three(twice, ["a"], ["X"]).tolist() == [[0]]
    mixed = rows_for([("a", "X", "R1"), ("a", "X", "R1"), ("a", "X", "R2")])
    assert S.vindr_two_of_three(mixed, ["a"], ["X"]).tolist() == [[1]]


def test_vindr_two_of_three_empty_inputs():
    empty = pd.DataFrame(columns=["image_id", "class_name", "rad_id"])
    with pytest.raises(ValueError, match="1 images do not have exactly 3 radiologists"):
        S.vindr_two_of_three(empty, ["a"], ["X"])
    out = S.vindr_two_of_three(empty, [], ["X", "Y"])
    assert out.shape == (0, 2)


_NO_PANDAS = """
import importlib.abc, sys

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas is blocked in this process")
        return None

for k in [k for k in sys.modules if k == "pandas" or k.startswith("pandas.")]:
    del sys.modules[k]
sys.meta_path.insert(0, _Block())
import eval.features
import eval.cxr_splits
assert "pandas" not in sys.modules
"""


def test_features_and_splits_import_without_pandas():
    """The SatDiFuser env has no pandas and `eval.features` imports constants from `eval.cxr_splits`."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c", _NO_PANDAS], cwd=repo, capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr[-400:]
    # control: the blocker really blocks (a guard that never fired is unverified)
    blocked = _NO_PANDAS.split("import eval.features")[0] + "import pandas\n"
    r2 = subprocess.run(
        [sys.executable, "-c", blocked], cwd=repo, capture_output=True, text=True, timeout=300
    )
    assert r2.returncode != 0 and "blocked" in r2.stderr
