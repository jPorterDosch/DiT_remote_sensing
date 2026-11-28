"""eval/official_tree.py: the tree check and output location the FLUX / DINO extraction scripts use.
Each guard is fed the case it exists to catch (project rule 6). Synthetic trees only."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval import export_geobench  # noqa: E402
from eval import features as F  # noqa: E402
from eval import official_tree as OT  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = tmp_path / "cxr"
    s.mkdir(mode=0o700)
    os.chmod(s, 0o700)
    monkeypatch.setenv("CXR_STORE", str(s))
    return s


def test_geobench_output_is_unchanged(store):
    assert OT.out_dir("m_eurosat", "models/dino_feats") == "models/dino_feats"
    assert OT.out_dir("m_bigearthnet", "models/x/y") == "models/x/y"


def test_cxr_output_goes_to_the_store_and_escapes_are_refused(store, tmp_path):
    assert OT.out_dir("chexpert", "models/dino_feats") == os.path.join(store, "feats", "dino_feats")
    assert OT.out_dir("chestxray14", "models/git10m_eval/run/chestxray14") == os.path.join(
        store, "feats", "git10m_eval/run/chestxray14"
    )
    inside = os.path.join(store, "elsewhere")
    assert OT.out_dir("vindr_cxr_kaggle", inside) == inside  # an override inside the store is allowed
    with pytest.raises(SystemExit, match="outside the CXR store"):
        OT.out_dir("chexpert", str(tmp_path / "public_dir"))  # absolute override outside
    with pytest.raises(SystemExit, match="outside the CXR store"):
        OT.out_dir("chexpert", "models/../../../escape")  # relative path climbing out
    os.chmod(store, 0o755)
    with pytest.raises(SystemExit, match="expected 700"):
        OT.out_dir("chexpert", "models/dino_feats")


def test_unknown_dataset_is_refused(store):
    with pytest.raises(SystemExit):
        OT.out_dir("not_a_dataset", "models/x")


def test_geobench_verify_is_the_line_the_scripts_ran(monkeypatch):
    seen = []
    monkeypatch.setattr(export_geobench, "main", lambda argv: seen.append(argv))
    OT.verify("m_brick_kiln", "data/m_brick_kiln_rgb")
    assert seen == [["--task", "m-brick-kiln", "--out", "data/m_brick_kiln_rgb"]]


def _tree(root, n=3, classes=("A", "B")):
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        os.makedirs(root / split)
        names = [f"{split}_{i}" for i in range(n)]
        for nm in names:
            Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(root / split / f"{nm}.png")
        y = rng.integers(0, 2, (n, len(classes)))
        np.savez(root / split / "labels.npz", names=np.array(names), y=y)


@pytest.mark.parametrize("key", ["chestxray14", "chexpert"])
def test_cxr_verify_passes_on_the_manifest_tree_and_fails_on_a_changed_pixel(tmp_path, monkeypatch, key):
    root = tmp_path / "tree"
    _tree(root)
    spec = {"sizes": {"train": 3, "val": 3, "test": 3}, "root": str(root), "classes": lambda: ["A", "B"],
            "multilabel": True}  # fmt: skip
    monkeypatch.setitem(F.OFFICIAL_CXR, key, spec)
    man = tmp_path / "manifest.json"
    monkeypatch.setattr(F, "cxr_manifest_path", lambda k: str(man))
    man.write_text(json.dumps(export_geobench.tree_manifest(key, str(root), ["A", "B"])))
    OT.verify(key, str(root))  # matches
    px = np.asarray(Image.open(root / "test" / "test_1.png")).copy()
    px[0, 0, 0] ^= 1
    Image.fromarray(px).save(root / "test" / "test_1.png")
    with pytest.raises(SystemExit, match="differs from the manifest"):
        OT.verify(key, str(root))


def test_flux_registry_serves_the_cxr_datasets():
    import datasets  # noqa: F401  (registers every dataset)
    from registry import DATASETS

    assert set(F.OFFICIAL_CXR) <= set(DATASETS)
    assert set(F.OFFICIAL) - {"m_eurosat"} <= set(DATASETS)
