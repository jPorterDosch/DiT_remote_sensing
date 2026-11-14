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
    web and sat need different normalization."""
    from eval import extract_dino as X

    web = X.PRESETS["dinov3_vit7b16_web"]["weights"]
    assert X.PRESETS["dinov3_vit7b16_sat"]["norm"] != X.PRESETS["dinov3_vit7b16_web"]["norm"]
    with pytest.raises(SystemExit, match="lacks hash a6675841"):
        X.load_model("dinov3_vit7b16_sat", web, "cpu")
    with pytest.raises(SystemExit, match="would be ignored"):
        X.load_model("dinov2_vitl14", web, "cpu")
    with pytest.raises(SystemExit, match="missing"):
        X.load_model(
            "dinov3_vit7b16_sat",
            str(tmp_path / os.path.basename(X.PRESETS["dinov3_vit7b16_sat"]["weights"])),
            "cpu",
        )


def test_official_wide_c_grid():
    """The opt-in wider grid (2026-10-07) scores every (candidate x C) cell, run_official's
    refit of the selected cell equals that cell's sealed prediction, and parse_c_grid orders,
    dedups and validates."""
    from eval import probe as PR

    rng = np.random.default_rng(3)
    y = [_multihot(rng, n) for n in (120, 40, 40)]
    cands = {"signal": tuple(yy + 0.1 * rng.standard_normal(yy.shape) for yy in y)}
    correct, pred, sel, table = P.run_official(cands, *y, c_grid=P.C_GRID_WIDE)
    assert {r["C"] for r in table} == set(P.C_GRID_WIDE) and len(table) == len(P.C_GRID_WIDE)
    rows, preds = P.official_all_cells(cands, *y, c_grid=P.C_GRID_WIDE)
    assert {r["C"] for r in rows} == set(P.C_GRID_WIDE)
    assert np.array_equal(preds[("signal", sel["C"])], pred)
    assert PR.parse_c_grid(None) == P.C_GRID
    assert PR.parse_c_grid("1,1e-5,0.0001,0.01,1e-6,0.001,0.1,0.01") == P.C_GRID_WIDE  # unordered + dup
    with pytest.raises(SystemExit, match="positive"):
        PR.parse_c_grid("0,0.1")


def _fake_block(path, k, grid, rng, n=30):
    """A sweep block result file with the run_block layout (2 timesteps + concat)."""
    from eval import sweep as S

    cands = ["t100", "t180", "concat2t"]
    rows = [
        {"candidate": c, "C": Cv, "val_acc": float(rng.random()), "conv_warnings": 0}
        for c in cands
        for Cv in grid
    ]
    np.savez(
        path, k=np.array(k), dataset=np.array("m_fake"), arm=np.array("a"), candidates=np.array(cands),
        C_grid=np.array(grid), val_table=np.array(json.dumps(rows)), timesteps=np.array([100, 180]),
        test_paths=np.array([f"img{i}.png" for i in range(n)]), test_labels=np.arange(n) % 3,
        pins=np.array(json.dumps({"k": k})), run_name=np.array(f"b{k}"), smoke=np.array(False),
        **{S._sealed_key(c, Cv): rng.integers(0, 3, n) for c in cands for Cv in grid},
    )  # fmt: skip
    return str(path)


def test_sweep_blocks_carry_their_c_grid(tmp_path):
    """verify_block checks a block against the grid IT was run with, and select refuses to mix
    blocks run with different grids (they are not one sweep)."""
    from eval import sweep as S

    rng = np.random.default_rng(4)
    wide = [_fake_block(tmp_path / f"w{k}.npz", k, P.C_GRID_WIDE, rng) for k in (19, 28)]
    for p in wide:
        S.verify_block(p, P.C_GRID_WIDE)  # 3 candidates x 7 C, all sealed
    with pytest.raises(SystemExit, match="C grid"):
        S.verify_block(wide[0])  # expected (default) grid != the block's: the guard fires
    blocks, _, ref = S.load_blocks(wide, "m_fake")
    assert ref[4] == list(P.C_GRID_WIDE) and len(S.select(blocks)[1]) == 2 * 3 * len(P.C_GRID_WIDE)
    old = _fake_block(tmp_path / "o33.npz", 33, P.C_GRID, rng)
    S.verify_block(old)  # an old 3-C block still verifies against its own grid
    with pytest.raises(SystemExit, match="not one sweep"):
        S.load_blocks(wide + [old], "m_fake")


def test_multilabel_n_jobs_identical(monkeypatch):
    """OVR_N_JOBS spreads m-bigearthnet's one-vs-rest fits over workers; the fits are independent
    and deterministic, so every (candidate x C) prediction must be bit-identical to serial."""
    rng = np.random.default_rng(5)
    y = [_multihot(rng, n) for n in (120, 40, 40)]
    cands = {"signal": tuple(yy + 0.3 * rng.standard_normal(yy.shape) for yy in y)}
    monkeypatch.setenv("OVR_N_JOBS", "1")
    rows1, preds1 = P.official_all_cells(cands, *y, c_grid=P.C_GRID_WIDE)
    monkeypatch.setenv("OVR_N_JOBS", "4")
    assert P.official_lr(0.1, y[0]).n_jobs == 4
    rows4, preds4 = P.official_all_cells(cands, *y, c_grid=P.C_GRID_WIDE)
    assert rows1 == rows4 and all(np.array_equal(preds1[k], preds4[k]) for k in preds1)


def test_file_identity_sidecar(tmp_path, monkeypatch):
    """The sha1 is cached in <file>.sha1.json and reused across processes, but a rewritten file
    (new size/mtime) is re-hashed -- a stale sidecar must never vouch for new content."""
    from eval import wb

    f = tmp_path / "x.npz"
    f.write_bytes(b"a" * 1000)
    first = wb.file_identity(str(f))
    assert (tmp_path / "x.npz.sha1.json").exists()
    wb._SHA_CACHE.clear()  # a fresh process
    monkeypatch.setattr(wb.hashlib, "sha1", lambda: (_ for _ in ()).throw(AssertionError("re-hashed")))
    assert wb.file_identity(str(f)) == first  # served from the sidecar, no hashing
    monkeypatch.undo()
    wb._SHA_CACHE.clear()
    f.write_bytes(b"b" * 1001)  # rewritten: size (and mtime) change
    assert wb.file_identity(str(f))["sha1"] != first["sha1"]


def test_multilabel_convergence_counted_across_workers(monkeypatch):
    """With OVR_N_JOBS > 1 the one-vs-rest fits run in worker processes whose ConvergenceWarnings
    never reach the parent; non-convergence must still be counted (from n_iter_), identically to
    serial. max_iter=1 forces real non-convergence, so this guard can fail (rule 6)."""
    rng = np.random.default_rng(7)
    y = [_multihot(rng, n) for n in (120, 40, 40)]
    cands = {"x": tuple(rng.standard_normal((len(yy), 20)) + yy @ rng.standard_normal((5, 20)) for yy in y)}
    monkeypatch.setattr(P, "OFFICIAL_MAX_ITER", 1)
    counts = {}
    for jobs in ("1", "4"):
        monkeypatch.setenv("OVR_N_JOBS", jobs)
        rows, _ = P.official_all_cells(cands, *y, c_grid=P.C_GRID)
        counts[jobs] = [r["conv_warnings"] for r in rows]
        _, _, sel, _ = P.run_official(cands, *y)
        assert sel["test_warn"] > 0 and sel["sel_warn"] > 0
    assert counts["1"] == counts["4"] and all(c > 0 for c in counts["4"]), counts


def _mb_cache(dirpath, split, rng, blocks=(8, 19, 28), stamp=True, compressed=False, n=6):
    """A multi-block cache with the extractor's exact layout: feats (N, K, B, C), mods (K, B, 3, C),
    rows in a SHUFFLED subset order (the loader must restore dataset order)."""
    os.makedirs(dirpath, exist_ok=True)
    K, B, C = 2, len(blocks), 4
    idx = rng.permutation(n)
    feats = rng.standard_normal((n, K, B, C)).astype(np.float32)
    tag = "k" + "-".join(map(str, blocks))
    path = os.path.join(dirpath, f"multistep_{split}_feats_oneshot_g1.0_{tag}.npz")
    (np.savez_compressed if compressed else np.savez)(
        path, feats=feats, labels=idx % 3, mods=rng.standard_normal((K, B, 3, C)), timesteps=np.array([100, 180]),
        subset_indices=idx, paths=np.array([f"img{i}.png" for i in idx]), layout=np.array("N,K,B,C"),
        block_idx=np.array(list(blocks)),  # as extraction.py saves it
    )  # fmt: skip
    meta = {"split": split, "k": list(blocks), "layout": "N,K,B,C", "dataset": "m_fake", "num_shards": 1}
    if stamp:
        from config_types import double_block_stamp

        meta.update(double_block_stamp(list(blocks)))
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f)
    return path, feats, idx


def test_multiblock_loader_slices_and_guards(tmp_path):
    """features.load_flux_split serves block k of an (N,K,B,C) cache as the single-block (N,K,C)
    slice, in dataset order, through the npz memmap -- and every new guard fires (rule 6)."""
    rng = np.random.default_rng(8)
    path, feats, idx = _mb_cache(str(tmp_path / "ok" / "run"), "val", rng)
    pat = str(tmp_path / "ok" / "*" / "multistep_{split}_feats_oneshot_g1.0_k8-19-28.npz")
    order = np.argsort(idx)
    for j, k in enumerate((8, 19, 28)):  # double block 8 included
        X, y, ts, meta, paths, _ = F.load_flux_split(pat, {"k": k, "dataset": "m_fake"}, "val", 6)
        assert np.array_equal(X, feats[order][:, :, j].astype(np.float64)) and ts == [100, 180]
        assert meta["k"] == k and paths == [f"img{i}.png" for i in np.sort(idx)]
    assert np.array_equal(F.npz_member_memmap(path, "feats"), feats)
    with pytest.raises(SystemExit, match="not among"):
        F.load_flux_split(pat, {"k": 30, "dataset": "m_fake"}, "val", 6)
    _mb_cache(str(tmp_path / "nostamp" / "run"), "val", rng, stamp=False)
    pat2 = str(tmp_path / "nostamp" / "*" / "multistep_{split}_feats_oneshot_g1.0_k8-19-28.npz")
    with pytest.raises(SystemExit, match="double_block_feature"):
        F.load_flux_split(pat2, {"k": 8, "dataset": "m_fake"}, "val", 6)
    F.load_flux_split(pat2, {"k": 19, "dataset": "m_fake"}, "val", 6)  # single block: no stamp needed
    cp, _, _ = _mb_cache(str(tmp_path / "zip" / "run"), "val", rng, compressed=True)
    with pytest.raises(SystemExit, match="compressed"):
        F.npz_member_memmap(cp, "feats")


def test_single_block_double_k_needs_stamp(tmp_path):
    """An int-k cache at a double block must carry the stamp too (pre-fix definition otherwise)."""
    d = tmp_path / "run"
    d.mkdir()
    p = d / "multistep_val_feats_oneshot_g1.0.npz"
    np.savez(p, feats=np.zeros((2, 1, 4)), labels=np.array([0, 1]), mods=np.zeros((1, 3, 4)),
             timesteps=np.array([100]), subset_indices=np.array([0, 1]), paths=np.array(["a", "b"]))  # fmt: skip
    with open(str(p).replace(".npz", "_meta.json"), "w") as f:
        json.dump({"split": "val", "k": 8, "dataset": "m_fake", "num_shards": 1}, f)
    with pytest.raises(SystemExit, match="double_block_feature"):
        F.load_flux_split(
            str(tmp_path / "*" / "multistep_{split}_feats_oneshot_g1.0.npz"),
            {"k": 8, "dataset": "m_fake"},
            "val",
            2,
        )


def test_multilabel_warning_count_equals_serial_reference(monkeypatch):
    """The per-fit count (_CountingLR) must equal the ORIGINAL measurement -- ConvergenceWarnings
    caught around a plain serial one-vs-rest fit -- in serial and in parallel, so max_iter stops
    and early lbfgs stops are both counted; predictions must equal plain LogisticRegression's."""
    import warnings as W

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(9)
    y = _multihot(rng, 150)
    X = rng.standard_normal((150, 30)) + y @ rng.standard_normal((5, 30))
    for max_iter in (1, 3, 3000):
        monkeypatch.setattr(P, "OFFICIAL_MAX_ITER", max_iter)
        with W.catch_warnings(record=True) as wl:
            W.simplefilter("always", ConvergenceWarning)
            ref = P.MultiLabelLR(LogisticRegression(C=1.0, max_iter=max_iter)).fit(X, y)
        ref_n = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
        for jobs in ("1", "3"):
            monkeypatch.setenv("OVR_N_JOBS", jobs)
            with W.catch_warnings(record=True) as wl2:
                W.simplefilter("always", ConvergenceWarning)
                clf = P.official_lr(1.0, y).fit(X, y)
            assert P._n_not_converged(clf, wl2) == ref_n, (max_iter, jobs)
            assert np.array_equal(clf.predict(X), ref.predict(X))


def test_trivial_baseline_is_optimal_constant():
    """trivial_baseline = the best metric ANY constant predictor reaches: majority rate for
    single-label; for multi-label, brute force over every constant label set must not beat it."""
    import itertools

    rng = np.random.default_rng(10)
    y1 = rng.integers(0, 4, 200)
    assert P.trivial_baseline(y1) == np.bincount(y1).max() / 200
    y = (rng.random((150, 6)) < np.array([0.6, 0.45, 0.3, 0.2, 0.1, 0.05])).astype(np.int64)
    brute = max(
        P.metric(P.per_image(y, np.broadcast_to(np.array(s), y.shape).astype(np.int64)))
        for s in itertools.product([0, 1], repeat=6)
    )
    assert abs(P.trivial_baseline(y) - brute) < 1e-12
    assert P.trivial_baseline(y) >= P.metric(P.per_image(y, np.ones_like(y)))  # beats all-positive


def test_ovr_n_jobs_resolution(monkeypatch):
    import os as _os

    cpus = len(_os.sched_getaffinity(0))
    for raw, want in (("1", 1), ("4", 4), ("-1", cpus), ("-2", max(1, cpus - 1))):
        monkeypatch.setenv("OVR_N_JOBS", raw)
        assert P.ovr_n_jobs() == want and P.official_lr(0.1, np.zeros((2, 2))).n_jobs == want
    monkeypatch.setenv("OVR_N_JOBS", "0")
    with pytest.raises(ValueError, match="invalid"):
        P.ovr_n_jobs()


def test_file_identity_refuses_file_changed_while_hashing(tmp_path, monkeypatch):
    """A digest of bytes that changed mid-read matches neither version: refuse, never record it."""
    from eval import wb

    f = tmp_path / "y.npz"
    f.write_bytes(b"x" * (1 << 20))
    wb._SHA_CACHE.clear()
    real_sha1 = wb.hashlib.sha1

    class Mutating:
        def __init__(self):
            self.h = real_sha1()

        def update(self, b):
            f.write_bytes(b"z" * ((1 << 20) + 7))  # rewritten during the read
            self.h.update(b)

        def hexdigest(self):
            return self.h.hexdigest()

    monkeypatch.setattr(wb.hashlib, "sha1", Mutating)
    with pytest.raises(OSError, match="changed while being hashed"):
        wb.file_identity(str(f))
    assert not (tmp_path / "y.npz.sha1.json").exists()


def test_shared_memmap_fits_identical(monkeypatch, tmp_path):
    """Multi-label + workers: the design matrix is backed by ONE memmap per candidate (no joblib
    re-dump per C fit). Fits, convergence counts and sealed predictions must equal serial, and
    the temp file must be gone afterwards."""
    rng = np.random.default_rng(11)
    y = [_multihot(rng, n) for n in (120, 40, 40)]
    cands = {"x": tuple(rng.standard_normal((len(yy), 25)) + yy @ rng.standard_normal((5, 25)) for yy in y)}
    monkeypatch.setenv("OVR_N_JOBS", "1")
    rows1, preds1 = P.official_all_cells(cands, *y, c_grid=P.C_GRID_WIDE)
    sel1 = P.run_official(cands, *y, c_grid=P.C_GRID_WIDE)
    monkeypatch.setenv("OVR_N_JOBS", "3")
    monkeypatch.setenv("JOBLIB_TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(P, "SHARE_MIN_BYTES", 0)  # force the memmap path on small data
    rows3, preds3 = P.official_all_cells(cands, *y, c_grid=P.C_GRID_WIDE)
    sel3 = P.run_official(cands, *y, c_grid=P.C_GRID_WIDE)
    assert rows1 == rows3 and all(np.array_equal(preds1[k], preds3[k]) for k in preds1)
    assert np.array_equal(sel1[0], sel3[0]) and sel1[2] == sel3[2]
    assert not list(tmp_path.glob("ovr_X_*.npy"))  # cleaned up


def _done_setup(tmp_path, monkeypatch, rng):
    """Caches under <tmp>/caches/run, a block result recording them (wide grid)."""
    from eval import sweep as S
    from eval import wb

    monkeypatch.setattr(F, "official_spec", lambda d: {"sizes": {"train": 1, "val": 1, "test": 1}})
    cdir = tmp_path / "caches" / "run"
    cdir.mkdir(parents=True)
    feats = {}
    for s in ("train", "val", "test"):
        p = cdir / f"c_{s}.npz"
        p.write_bytes(s.encode() * 10)
        feats[s] = [wb.file_identity(str(p))]
    res = _fake_block(tmp_path / "b.npz", 28, P.C_GRID_WIDE, rng)
    d = dict(np.load(res, allow_pickle=True))
    np.savez(res, **d, config=np.array(json.dumps({"features": feats})))
    return S, wb, cdir, res, str(tmp_path / "caches" / "*" / "c_{split}.npz"), str(tmp_path / "caches")


def test_done_status_cli_exit_codes(tmp_path, monkeypatch):
    """Every exit code the sbatch `case` statements branch on, through main() (rule 6)."""
    S, wb, cdir, res, pat, cache = _done_setup(tmp_path, monkeypatch, np.random.default_rng(12))

    def code(result, *extra, c_grid="wide", cache_dir=cache):
        with pytest.raises(SystemExit) as e:
            S.main(["done-status", result, "--dataset", "m_fake", "--features", pat, "--cache-dir", cache_dir,
                    *(["--c-grid", c_grid] if c_grid else []), *extra])  # fmt: skip
        return e.value.code

    assert code(res) == 0  # current
    assert code(res, c_grid=None) == 3  # other C grid
    assert code(res, cache_dir=str(tmp_path / "elsewhere")) == 5  # not this extraction
    (cdir / "c_val.npz").write_bytes(b"changed!!")
    wb._SHA_CACHE.clear()
    assert code(res) == 3  # content changed
    for s in ("train", "val", "test"):
        (cdir / f"c_{s}.npz").unlink()
    assert code(res) == 0  # verified, inputs deleted afterwards: done
    legacy = _fake_block(tmp_path / "old.npz", 33, P.C_GRID_WIDE, np.random.default_rng(1))
    assert code(legacy) == 4  # legacy, nothing newer
    (cdir / "x_meta.json").write_text("{}")  # an extraction newer than the legacy result
    os.utime(cdir / "x_meta.json", (os.path.getmtime(legacy) + 60,) * 2)
    assert code(legacy) == 5


def test_done_status_catches_reextraction(tmp_path, monkeypatch):
    """The eighth-pass hole: recorded inputs gone but a NEW extraction (new run dir after a
    config-hash change) now under cache_dir -> stale, not done; also when only one split moved."""
    S, wb, cdir, res, pat, cache = _done_setup(tmp_path, monkeypatch, np.random.default_rng(13))
    new = cdir.parent / "run_newhash"
    new.mkdir()
    (cdir / "c_train.npz").rename(new / "c_train.npz")  # train re-extracted elsewhere, val/test old
    code, why = S.done_status(res, pat, "m_fake", P.C_GRID_WIDE, cache)
    assert code == 3 and "train" in why
    for s in ("val", "test"):
        (cdir / f"c_{s}.npz").unlink()
        (new / f"c_{s}.npz").write_bytes(b"fresh" * 3)
    assert S.done_status(res, pat, "m_fake", P.C_GRID_WIDE, cache)[0] == 3


def test_multiblock_gate_can_fail(tmp_path, monkeypatch):
    """eval.sweep.gate: slice PASSES on the real loader and FAILS when the loader serves the wrong
    block; control FAILS without a result and on a probe no better than a constant predictor."""
    from eval import sweep as S

    monkeypatch.setattr(F, "official_spec", lambda d: {"sizes": {"train": 6, "val": 6, "test": 6}})
    monkeypatch.setattr(F, "flux_pins", lambda d, e: {"dataset": "m_fake"})
    rng = np.random.default_rng(14)
    _mb_cache(str(tmp_path / "c" / "run"), "val", rng)
    _mb_cache(str(tmp_path / "c" / "run"), "train", rng)
    pat = str(tmp_path / "c" / "*" / "multistep_{split}_feats_oneshot_g1.0_k8-19-28.npz")
    out = tmp_path / "out"
    out.mkdir()
    assert S.gate("slice", "m_fake", 19, pat, str(out), "a")
    assert S.gate("slice", "m_fake", 8, pat, str(out), "a")  # a double block
    real = F._block_slice
    monkeypatch.setattr(F, "_block_slice", lambda path, meta, pins: real(path, meta, {**pins, "k": 28}))
    assert not S.gate("slice", "m_fake", 19, pat, str(out), "a")  # wrong block served
    monkeypatch.setattr(F, "_block_slice", real)
    real_mm = F.npz_member_memmap  # train memmap at a wrong row offset: only the train check sees it
    monkeypatch.setattr(
        F, "npz_member_memmap", lambda p, n: real_mm(p, n)[::-1] if "train" in p else real_mm(p, n)
    )
    assert not S.gate("slice", "m_fake", 19, pat, str(out), "a")
    monkeypatch.setattr(F, "npz_member_memmap", real_mm)
    assert not S.gate("control", "m_fake", 19, pat, str(out), "a")  # no result
    rows = [{"candidate": "t100", "C": 0.1, "val_acc": 0.40, "conv_warnings": 0}]
    np.savez(out / "sweep-block_m_fake_a-k19_x.npz", val_table=np.array(json.dumps(rows)))
    assert not S.gate("control", "m_fake", 19, pat, str(out), "a")  # 0.40 vs majority 1/3: skill 0.10 < 0.25
    rows[0]["val_acc"] = 0.95
    np.savez(out / "sweep-block_m_fake_a-k19_x.npz", val_table=np.array(json.dumps(rows)))
    assert S.gate("control", "m_fake", 19, pat, str(out), "a")


def test_shared_memmap_removed_on_sigterm(tmp_path):
    """Slurm's --time/scancel SIGTERM must not leave the multi-GB memmap behind."""
    import signal
    import subprocess
    import textwrap

    code = textwrap.dedent(f"""
        import os, signal, sys, numpy as np
        sys.path.insert(0, {repr(F.REPO_ROOT)})
        from eval import protocols as P
        P.SHARE_MIN_BYTES = 0
        os.environ["OVR_N_JOBS"] = "2"; os.environ["JOBLIB_TEMP_FOLDER"] = {repr(str(tmp_path))}
        with P._shared_for_workers(lambda: np.ones((10, 3)), np.ones((10, 2))) as A:
            print("inside", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            import time; time.sleep(5)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "inside" in r.stdout and r.returncode == 128 + signal.SIGTERM, (r.returncode, r.stderr[-500:])
    assert not list(tmp_path.glob("ovr_X_*.npy"))


def test_done_status_pattern_and_provenance_order(tmp_path, monkeypatch):
    """Ninth-pass holes: a --features pattern that matches NONE of the recorded inputs (another
    BLOCKS list, same cache_dir) is "not this extraction" (5), never "deleted -> done"; and a
    result from another sweep is 5 even when its C grid also differs (provenance before grid)."""
    S, wb, cdir, res, pat, cache = _done_setup(tmp_path, monkeypatch, np.random.default_rng(15))
    other = str(tmp_path / "caches" / "*" / "other_{split}.npz")  # matches nothing on disk
    code, why = S.done_status(res, other, "m_fake", P.C_GRID_WIDE, cache)
    assert code == 5 and "does not match" in why
    assert S.done_status(res, pat, "m_fake", P.C_GRID, str(tmp_path / "elsewhere"))[0] == 5
    assert S.done_status(res, pat, "m_fake", P.C_GRID, cache)[0] == 3  # same sweep, other grid


def test_budget_ovr_jobs(monkeypatch):
    """Worker pool sized to memory (2026-10-08 OOM: 18 workers x ~4 GB on a 64 GB job): shrinks
    with the limit, pins OVR_N_JOBS, never touches single-label or serial runs."""

    def cands(d):  # shapes only -- no allocation
        return {"x": tuple(np.broadcast_to(np.float32(0), (n, d)) for n in (20000, 1000, 1000))}

    y2 = np.zeros((20000, 43), dtype=np.int64)
    monkeypatch.setattr(P, "_rss_bytes", lambda: 10 << 30)  # the parent's measured ~10 GB
    for limit_gb, d, lo, hi in ((64, 20480, 4, 9), (64, 27648, 3, 7), (16, 27648, 1, 1)):
        monkeypatch.setenv("OVR_N_JOBS", "18")
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(limit_gb * 1024))
        n = P.budget_ovr_jobs(cands(d), y2)
        assert lo <= n <= hi and P.ovr_n_jobs() == n, (limit_gb, d, n)
        assert n * (1.3 * 20000 * d * 8) + (10 << 30) < 0.9 * (limit_gb << 30) or n == 1
    monkeypatch.setenv("OVR_N_JOBS", "18")
    assert P.budget_ovr_jobs(cands(20480), np.zeros(20000, dtype=np.int64)) == 18  # single-label: untouched
    monkeypatch.setenv("OVR_N_JOBS", "1")
    assert P.budget_ovr_jobs(cands(20480), y2) == 1


def test_mem_limit_sources(tmp_path, monkeypatch):
    """--mem-per-cpu and the job's own cgroup (v1 and v2) are honored; no readable limit with
    several multi-label workers refuses instead of running unbudgeted (review 2026-10-11)."""
    for v in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    assert P._mem_limit_bytes() == 32 << 30
    monkeypatch.delenv("SLURM_MEM_PER_CPU")
    root = tmp_path / "cg"
    job = root / "memory" / "slurm" / "uid_1" / "job_7"
    job.mkdir(parents=True)
    (job / "memory.limit_in_bytes").write_text(str(48 << 30))
    (root / "memory" / "memory.limit_in_bytes").write_text(str(1 << 62))  # v1 root: unlimited
    pc = tmp_path / "cgroup_v1"
    pc.write_text("12:blkio:/\n8:memory:/slurm/uid_1/job_7\n")
    assert P._cgroup_limit_files(str(pc), str(root))[0] == str(job / "memory.limit_in_bytes")
    v2 = tmp_path / "cgroup_v2"
    v2.write_text("0::/system.slice/slurmstepd.scope/job_7\n")
    assert P._cgroup_limit_files(str(v2), str(root))[0] == str(
        root / "system.slice" / "slurmstepd.scope" / "job_7" / "memory.max"
    )
    real = P._cgroup_limit_files
    monkeypatch.setattr(P, "_cgroup_limit_files", lambda: real(str(pc), str(root)))
    assert P._mem_limit_bytes() == 48 << 30
    monkeypatch.setattr(P, "_mem_limit_bytes", lambda: None)
    monkeypatch.setenv("OVR_N_JOBS", "8")
    y2 = np.zeros((10, 3), dtype=np.int64)
    with pytest.raises(SystemExit, match="no memory limit"):
        P.budget_ovr_jobs({"x": (np.zeros((10, 4)),)}, y2)


def test_features_artifact_digest_cached_and_identical(tmp_path, monkeypatch):
    """The cached reference md5 gives wandb's own manifest digest (producer/consumer artifacts
    still match), hashes a file once, and re-hashes after a rewrite (review 2026-10-11)."""
    from eval import wb
    from wandb.sdk.lib import hashutil

    f = tmp_path / "multistep_train_feats_oneshot_g1.0.npz"
    f.write_bytes(os.urandom(4096))
    meta = tmp_path / "multistep_train_feats_oneshot_g1.0_meta.json"
    meta.write_text("{}")

    def wandb_digest():  # what wandb itself computes, no hook
        a = wb.wandb.Artifact(wb._artifact_name("feat", str(f)), type="features")
        a.add_reference("file://" + str(f), checksum=True)
        a.add_reference("file://" + str(meta), checksum=True)
        return a.manifest.digest()

    plain = wandb_digest()
    assert wb.features_artifact(str(f)).manifest.digest() == plain
    assert json.load(open(str(f) + ".md5.json"))["md5_b64"] == hashutil.md5_file_b64(str(f))
    assert wandb_digest() == plain  # the hook is restored after use

    def boom(*a, **k):
        raise AssertionError("re-hashed despite a valid sidecar")

    monkeypatch.setattr(hashutil, "md5_file_b64", boom)
    assert wb.features_artifact(str(f)).manifest.digest() == plain  # served from the sidecar
    monkeypatch.undo()
    f.write_bytes(os.urandom(4096))  # rewrite: new size/mtime -> re-hashed, new digest
    assert wb.features_artifact(str(f)).manifest.digest() == wandb_digest() != plain


def test_backfilled_stamp_needs_verification_record(tmp_path, monkeypatch):
    """A hand-backfilled double_block_feature stamp is refused until verify_double_block.py's PASS
    record vouches for this exact file; a stale record (other sha1) or a FAIL record is refused
    too, and single blocks never need one (review 2026-10-11)."""
    from eval import wb

    root = tmp_path / "mb"
    path, _, _ = _mb_cache(str(root / "run"), "val", np.random.default_rng(21))
    mp = path.replace(".npz", "_meta.json")
    meta = json.load(open(mp))
    meta["double_block_feature_note"] = "backfilled"
    json.dump(meta, open(mp, "w"))
    pat = str(root / "*" / "multistep_{split}_feats_oneshot_g1.0_k8-19-28.npz")
    pins = {"k": 8, "dataset": "m_fake"}
    with pytest.raises(SystemExit, match="without a passing verification record"):
        F.load_flux_split(pat, pins, "val", 6)
    F.load_flux_split(pat, {**pins, "k": 19}, "val", 6)  # single-stream block: no stamp needed

    def record(ok, sha1):
        rec = {"pass": ok, "feature": "block_input+img_mod1", "caches": {path: {"sha1": sha1}}}
        json.dump(rec, open(root / F.DOUBLE_BLOCK_RECORD, "w"))

    record(True, "0" * 40)
    with pytest.raises(SystemExit, match="without a passing verification record"):
        F.load_flux_split(pat, pins, "val", 6)  # record for other content
    record(False, wb.file_identity(path)["sha1"])
    with pytest.raises(SystemExit, match="without a passing verification record"):
        F.load_flux_split(pat, pins, "val", 6)  # failed verification
    record(True, wb.file_identity(path)["sha1"])
    F.load_flux_split(pat, pins, "val", 6)


def test_ovr_fit_threads_is_a_protocol_constant(monkeypatch):
    """Every one-vs-rest label fit limits BLAS to OVR_FIT_THREADS whatever OVR_N_JOBS says: the
    operating point of a multi-label cell must not follow --cpus-per-task or the memory-budgeted
    worker count (rule 8; until 2026-10-15 the share was cpus // OVR_N_JOBS read in the worker).
    The spy only sees the serial path (workers import the real module); the pooled fits run the
    same method."""
    seen = []
    real = P.threadpool_limits

    def spy(limits=None, user_api=None):
        seen.append((limits, user_api))
        return real(limits=limits, user_api=user_api)

    monkeypatch.setattr(P, "threadpool_limits", spy)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 6))
    y = (X[:, :2] > 0).astype(int)
    for jobs in ("1", "4"):
        monkeypatch.setenv("OVR_N_JOBS", jobs)
        seen.clear()
        clf = P.official_lr(0.1, y).fit(X, y)
        assert clf.n_jobs == int(jobs)
        if jobs == "1":
            assert len(seen) == 2 and all(s == (P.OVR_FIT_THREADS, "blas") for s in seen)
    assert P.OVR_FIT_THREADS == 1
