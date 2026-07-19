# ruff: noqa: E402 — sys.path must be mutated before any local imports
from __future__ import annotations

import os
import sys
import types


def _find_project_root(start: str) -> str:
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"Could not locate project root (src/) from {start}")
        d = parent


_root = _find_project_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)                        # registry.py lives here
sys.path.insert(0, os.path.join(_root, "src"))   # datasets, data, utils, …

import pytest

import datasets  # noqa: F401  — registers "resisc45" in the registry
from registry import DATASETS
from data.resisc45_dataset import RESISC45_CLASSES, RESISC45Dataset

# Real data is optional: skip (not fail) when the dataset isn't downloaded,
# e.g. in CI. Populate with: bash download_resisc45.sh data/resisc45
DATA_PATH = os.environ.get(
    "RESISC45_PATH", os.path.join(_root, "data/resisc45/NWPU-RESISC45")
)

_have_data = os.path.isdir(DATA_PATH) and any(
    os.path.isdir(os.path.join(DATA_PATH, c)) for c in RESISC45_CLASSES
)
requires_data = pytest.mark.skipif(
    not _have_data, reason=f"RESISC45 data not found at {DATA_PATH}"
)


def test_registered():
    """Importing datasets registers the resisc45 wrapper."""
    assert "resisc45" in DATASETS


def test_class_metadata_consistent():
    """45 classes, and prompt/category tables line up with the class list."""
    from data.resisc45_dataset import RESISC45_PROMPTS, get_resisc45_categories

    assert len(RESISC45_CLASSES) == 45
    assert set(RESISC45_CLASSES) == set(RESISC45_PROMPTS)
    assert len(get_resisc45_categories()) == 45


@requires_data
def test_splits_partition_real_data():
    """Full = 31,500; train/test are a disjoint 80/20-per-class partition."""
    full = RESISC45Dataset(DATA_PATH, split=None, img_size=224)
    train = RESISC45Dataset(DATA_PATH, split="train", img_size=224)
    test = RESISC45Dataset(DATA_PATH, split="test", img_size=224)

    assert len(full) == 45 * 700
    assert len(train) == 45 * 560
    assert len(test) == 45 * 140
    assert len(train) + len(test) == len(full)

    train_paths = {p for p, _, _ in train.samples}
    test_paths = {p for p, _, _ in test.samples}
    assert train_paths.isdisjoint(test_paths)


@requires_data
def test_item_format_real_data():
    """Each item is a [-1,1] float tensor of shape (3,H,W) with aligned labels."""
    ds = RESISC45Dataset(DATA_PATH, split="test", img_size=224)
    s = ds[0]

    assert set(s.keys()) == {"img", "label", "class_name", "path"}
    img = s["img"]
    assert tuple(img.shape) == (3, 224, 224)
    assert img.dtype.is_floating_point
    assert float(img.min()) >= -1.0001 and float(img.max()) <= 1.0001

    # label index, class name, and on-disk folder all agree
    assert RESISC45_CLASSES[s["label"]] == s["class_name"]
    assert os.path.basename(os.path.dirname(s["path"])) == s["class_name"]

    labels_seen = {lbl for _, lbl, _ in ds.samples}
    assert labels_seen == set(range(45))


@requires_data
def test_wrapper_get_data_real_data():
    """The registered wrapper builds train/test loaders and honors label_fraction."""
    cfg = types.SimpleNamespace(
        img_size=224,
        dataset=types.SimpleNamespace(path=DATA_PATH),
        label_fraction=0.1,
        batch_size=8,
        num_workers=0,
    )
    wrapper = DATASETS["resisc45"](cfg)
    assert len(wrapper.category_list) == 45

    loaders = wrapper.get_data(cfg)
    assert set(loaders) == {"train", "test"}

    # label_fraction=0.1 subsamples the 25,200-sample train split
    assert len(loaders["train"].dataset) == int(45 * 560 * 0.1)

    batch = next(iter(loaders["train"]))
    assert tuple(batch["img"].shape) == (8, 3, 224, 224)
    assert tuple(batch["label"].shape) == (8,)
