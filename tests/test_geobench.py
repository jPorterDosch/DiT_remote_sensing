"""GEO-Bench task registration, export guards and the multi-label (micro-F1) path, on
synthetic data (no GPU, no downloads). Every guard here is fed the mismatch it exists to
catch (rule 6). Real-data checks: the committed manifests (eval/manifests/) are verified by
every export, on the workstation and on ISAAC."""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from eval import export_geobench as E
from eval import features as F
from eval import protocols as P


# ------------------------------------------------------------------- registration
@pytest.mark.parametrize("key", list(F.OFFICIAL))
def test_registry_consistency(key):
    """OFFICIAL, TaskSpec, run.py's DATASETS and the committed manifest agree; where the
    shipped partition file is present, its counts equal OFFICIAL's (rule 16)."""
    sys.path.insert(0, os.path.join(F.REPO_ROOT, "src"))
    import datasets  # noqa: F401
    from registry import DATASETS

    assert key.replace("_", "-") in E.SPECS and key in DATASETS
    sizes = F.official_spec(key)["sizes"]
    man = json.load(open(E.manifest_path(key)))
    assert {s: v["n"] for s, v in man["splits"].items()} == sizes
    classes = F.official_classes(key)
    assert classes == man["classes"] and len(set(classes)) == len(classes)
    assert all(len(v["counts"]) == len(classes) for v in man["splits"].values())
    part = f"data/{key}_meta/default_partition.json"
    if os.path.exists(part):
        shipped = {E.OUT_SPLIT[s]: len(v) for s, v in json.load(open(part)).items()}
        assert shipped == sizes


# ------------------------------------------------------- export on a fake geobench
class _Sample:
    def __init__(self, i, label, px):
        self.sample_name, self.label, self._px = f"s{i}", label, px

    def pack_to_3d(self, band_names):
        return self._px, list(band_names)


def _fake_geobench(monkeypatch, multilabel, value, shape=(4, 4, 3)):
    """geobench stand-in: 3 classes, split sizes 6/3/3, every pixel = value."""
    names = ["class a", "class b", "class c"]
    lt = type("MultiLabelClassification" if multilabel else "Classification", (), {})()
    setattr(lt, "class_name" if multilabel else "class_names", names)

    def dataset(_dir, split, partition_name):
        n = {"train": 6, "valid": 3, "test": 3}[split]
        lab = (
            (lambda i: np.eye(3, dtype=int)[i % 3] | np.eye(3, dtype=int)[(i + 1) % 3])
            if multilabel
            else (lambda i: i % 3)
        )
        return [_Sample(i, lab(i), np.full(shape, value, np.float32)) for i in range(n)]

    mod = types.SimpleNamespace(
        load_task_specs=lambda d: types.SimpleNamespace(label_type=lt), GeobenchDataset=dataset
    )
    monkeypatch.setitem(sys.modules, "geobench", mod)
    monkeypatch.setitem(
        E.SPECS, "m-fake", E.TaskSpec(zenodo=0, rgb_bands=("r", "g", "b"), scale=100.0, img_hw=shape[:2])
    )
    monkeypatch.setitem(
        F.OFFICIAL,
        "m_fake",
        {
            "sizes": {"train": 6, "val": 3, "test": 3},
            "root": "unused",
            "classes": lambda: ["class_a", "class_b", "class_c"],
            "multilabel": multilabel,
        },
    )


@pytest.mark.parametrize("multilabel", [False, True])
def test_export_roundtrip(tmp_path, monkeypatch, multilabel):
    """Export -> list_official_split returns every image with its label; manifest counts
    match; a re-export matches the manifest."""
    _fake_geobench(monkeypatch, multilabel, value=50.0)
    man = E.export("m-fake", "unused", str(tmp_path / "tree"))
    files, y = F.list_official_split("m_fake", "train", str(tmp_path / "tree"))
    assert len(files) == 6 and all(os.path.exists(f) for f in files)
    want = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]] * 2) if multilabel else np.array([0, 0, 1, 1, 2, 2])
    assert np.array_equal(y, want)  # single-label: class-dir order, then sorted names
    assert man["splits"]["train"]["counts"] == ([4, 4, 4] if multilabel else [2, 2, 2])
    monkeypatch.setattr(E, "manifest_path", lambda t: str(tmp_path / "man.json"))
    E.check_manifest("m-fake", man, write=True)
    E.check_manifest("m-fake", E.export("m-fake", "unused", str(tmp_path / "again")))
    E.check_manifest("m-fake", E.tree_manifest("m_fake", str(tmp_path / "tree")))  # verify-only path


def _pixel(root):
    f = F.list_official_split("m_fake", "test", root)[0][0]
    a = np.asarray(Image.open(f)).copy()
    a[0, 0, 0] ^= 1
    Image.fromarray(a).save(f)


def _stray(root):
    Image.new("RGB", (4, 4)).save(os.path.join(root, "val", "class_a", "zz.png"))


def _relabel(root):
    if F.OFFICIAL["m_fake"]["multilabel"]:
        p = os.path.join(root, "train", "labels.npz")
        d = dict(np.load(p))
        d["y"][0, 2] ^= 1
        np.savez(p, **d)
    else:  # move one image to another class dir: same pixels, different label
        f = F.list_official_split("m_fake", "train", root)[0][0]
        os.replace(f, f.replace("class_a", "class_b"))


def _delete(root):
    os.remove(F.list_official_split("m_fake", "val", root)[0][0])


@pytest.mark.parametrize("multilabel", [False, True])
@pytest.mark.parametrize(
    "corrupt, match",
    [
        (_pixel, "differs"),
        (_stray, "stray|expected 3"),
        (_relabel, "differs"),
        (_delete, "expected 3|missing"),
    ],
)
def test_manifest_catches_corruption(tmp_path, monkeypatch, multilabel, corrupt, match):
    """The committed manifest refuses a tree with one flipped pixel bit, a stray PNG, one
    changed label, or one missing image (rule 6: each guard fed its mismatch)."""
    import shutil

    _fake_geobench(monkeypatch, multilabel, value=50.0)
    root = str(tmp_path / "tree")
    monkeypatch.setattr(E, "manifest_path", lambda t: str(tmp_path / "man.json"))
    E.check_manifest("m-fake", E.export("m-fake", "unused", root), write=True)
    bad = str(tmp_path / "bad")
    shutil.copytree(root, bad)
    if corrupt is _stray and multilabel:
        os.makedirs(os.path.join(bad, "val", "class_a"))
    corrupt(bad)
    with pytest.raises(SystemExit, match=match):
        E.check_manifest("m-fake", E.tree_manifest("m_fake", bad))


@pytest.mark.parametrize("value", [0.0, 1000.0])
def test_wrong_scale_refused(tmp_path, monkeypatch, value):
    """All-black (scale too large) or all-white (too small) exports are refused."""
    _fake_geobench(monkeypatch, False, value=value)
    with pytest.raises(SystemExit, match="scale 100.0 is wrong"):
        E.export("m-fake", "unused", str(tmp_path))


def test_wrong_shape_refused(tmp_path, monkeypatch):
    """m-forestnet's task_specs claims 332x332, its samples are 256x256: shape is per sample."""
    _fake_geobench(monkeypatch, False, value=50.0, shape=(5, 4, 3))
    monkeypatch.setitem(
        E.SPECS, "m-fake", E.TaskSpec(zenodo=0, rgb_bands=("r", "g", "b"), scale=100.0, img_hw=(4, 4))
    )
    with pytest.raises(SystemExit, match="has shape"):
        E.export("m-fake", "unused", str(tmp_path))


def test_label_kind_mismatch_refused(tmp_path, monkeypatch):
    _fake_geobench(monkeypatch, True, value=50.0)
    monkeypatch.setitem(F.OFFICIAL["m_fake"], "multilabel", False)
    with pytest.raises(SystemExit, match="multilabel flag"):
        E.export("m-fake", "unused", str(tmp_path))


def test_partial_tree_refused(tmp_path, monkeypatch):
    _fake_geobench(monkeypatch, False, value=50.0)
    E.export("m-fake", "unused", str(tmp_path))
    os.remove(F.list_official_split("m_fake", "val", str(tmp_path))[0][0])
    with pytest.raises(SystemExit, match="expected 3"):
        F.list_official_split("m_fake", "val", str(tmp_path))


# ------------------------------------------------------------------ micro-F1 path
def _multihot(rng, n, L=5):
    return (rng.random((n, L)) < 0.3).astype(np.int64)


def _satdifuser_f1(y, pred):
    """SatDiFuser utils/val_logger.py MultiLabelClsEvaluator, verbatim call."""
    from sklearn.metrics import precision_recall_fscore_support

    return precision_recall_fscore_support(y_true=y, y_pred=pred, average="micro", zero_division=0)[2]


def test_micro_f1_matches_satdifuser():
    """metric(per_image(...)) == SatDiFuser's micro-F1, incl. the zero_division edge cases."""
    rng = np.random.default_rng(0)
    y, pred = _multihot(rng, 200), _multihot(rng, 200)
    zeros = np.zeros_like(y)
    for yy, pp in ((y, pred), (y, zeros), (zeros, pred), (zeros, zeros), (y, y)):
        assert abs(P.metric(P.per_image(yy, pp)) - _satdifuser_f1(yy, pp)) < 1e-12
    c = P.per_image(y[:, 0], pred[:, 0])  # single-label path unchanged: top-1
    assert c.dtype == np.int8 and P.metric(c) == float((y[:, 0] == pred[:, 0]).mean())


def test_ovr_threshold_is_sigmoid_half():
    """official_lr on multi-hot labels predicts exactly sigmoid(logit) > 0.5 per label, even
    when label 0 is constant in train (where sklearn's own OneVsRest predict shifts every
    label's threshold -- fed here, and shown to differ)."""
    from sklearn.multiclass import OneVsRestClassifier

    rng = np.random.default_rng(4)
    X, y = rng.standard_normal((150, 6)), _multihot(rng, 150)
    y[:, 0] = 0
    clf = P.official_lr(1.0, y).fit(X, y)
    logit = np.stack([e.decision_function(X) for e in clf.estimators_], 1)
    want = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    assert np.array_equal(clf.predict(X), want)
    assert not np.array_equal(np.asarray(OneVsRestClassifier.predict(clf, X)), want)


def test_paired_delta():
    rng = np.random.default_rng(1)
    y = _multihot(rng, 300)
    good, bad = P.per_image(y, y), P.per_image(y, _multihot(rng, 300))
    assert P.paired_delta(good, good) == (0.0, 0.0, 0.0)
    d, lo, hi = P.paired_delta(bad, good)
    assert 0 < lo <= d <= hi and abs(d - (1.0 - P.metric(bad))) < 1e-12
    a, b = rng.integers(0, 2, 300).astype(np.int8), rng.integers(0, 2, 300).astype(np.int8)
    assert P.paired_delta(a, b) == (float((b - a).mean()), *P.ci(b - a))  # top-1: as before


def test_official_multilabel_protocol():
    """run_official / official_all_cells fit one-vs-rest on multi-hot labels, select by
    micro-F1, and return (N, 3) counts; a label-informative feature beats noise."""
    rng = np.random.default_rng(2)
    y = [_multihot(rng, n) for n in (120, 40, 40)]
    sig = [yy + 0.1 * rng.standard_normal(yy.shape) for yy in y]
    noise = [rng.standard_normal(yy.shape) for yy in y]
    cands = {"noise": tuple(noise), "signal": tuple(sig)}
    correct, pred, sel, table = P.run_official(cands, *y)
    assert sel["candidate"] == "signal" and correct.shape == (40, 3) and pred.shape == (40, 5)
    assert P.metric(correct) > 0.9
    rows, preds = P.official_all_cells(cands, *y)
    assert {(r["candidate"], r["C"]) for r in rows} == {(c, Cv) for c in cands for Cv in P.C_GRID}
    assert np.array_equal(preds[("signal", sel["C"])], pred)


def test_multilabel_subset_deterministic():
    sys.path.insert(0, os.path.join(F.REPO_ROOT, "src"))
    from tasks.extraction import _stratified_indices

    y = _multihot(np.random.default_rng(3), 50)
    a, b = _stratified_indices(y, 20, 42), _stratified_indices(y, 20, 42)
    assert np.array_equal(a, b) and len(np.unique(a)) == 20 and np.all(np.diff(a) > 0)


def test_dinov3_checkpoint_hash_guard(tmp_path):
    """A web checkpoint under a sat preset (or vice versa) is refused before torch.hub runs:
    the hub picks the ViT-L architecture from the filename hash, and the norms differ."""
    from eval import extract_dino as X

    web = X.PRESETS["dinov3_vitl16_web"]["weights"]
    assert X.PRESETS["dinov3_vitl16_sat"]["norm"] != X.PRESETS["dinov3_vitl16_web"]["norm"]
    with pytest.raises(SystemExit, match="lacks hash eadcf0ff"):
        X.load_model("dinov3_vitl16_sat", web, "cpu")
    with pytest.raises(SystemExit, match="would be ignored"):
        X.load_model("dinov2_vitl14", web, "cpu")
    with pytest.raises(SystemExit, match="missing"):
        X.load_model(
            "dinov3_vitl16_sat",
            str(tmp_path / os.path.basename(X.PRESETS["dinov3_vitl16_sat"]["weights"])),
            "cpu",
        )
