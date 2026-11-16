"""src/datasets/arrow_corpus.py: the shared 64-row Arrow shard reader, exercised through a
subclass with other shard names than Git-10M's. CPU only, synthetic shards."""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import arrow_corpus as A  # noqa: E402

# Same columns the Git-10M converter writes (datasets.git10m.SCHEMA).
SCHEMA = pa.schema(
    [
        ("img_name", pa.string()),
        ("image_bytes", pa.binary()),
        ("caption", pa.string()),
        ("resolution", pa.string()),
        ("Google_location", pa.string()),
        ("img_quality_score", pa.float64()),
    ]
)


def _png(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(b, format="PNG")
    return b.getvalue()


def _write_shard(path: str, shard: int, n: int) -> None:
    with pa.OSFile(path, "wb") as sink:
        w = ipc.new_file(sink, SCHEMA)
        for off in range(0, n, A.BATCH_ROWS):
            idx = range(off, min(n, off + A.BATCH_ROWS))
            w.write_batch(
                pa.RecordBatch.from_pydict(
                    {
                        "img_name": [f"s{shard}_r{r}" for r in idx],
                        "image_bytes": [_png(shard * 1000 + r) for r in idx],
                        "caption": ["c"] * len(idx),
                        "resolution": ["1.0"] * len(idx),
                        "Google_location": ["x"] * len(idx),
                        "img_quality_score": [3.0] * len(idx),
                    },
                    schema=SCHEMA,
                )
            )
        w.close()


class Toy(A.ArrowCorpus):
    n_shards = 2

    @staticmethod
    def shard_name(i: int) -> str:
        return f"toy-{i:02d}-of-02.arrow"

    def __init__(self, arrow_dir: str, shard, row):
        shard, row = np.asarray(shard, dtype=np.int64), np.asarray(row, dtype=np.int64)
        self._set_rows(shard, row, np.arange(len(shard)), arrow_dir, 224, False, 0)
        self._check_shards()


def _all_rows(sizes):
    shard = np.concatenate([np.full(n, s) for s, n in enumerate(sizes)])
    row = np.concatenate([np.arange(n) for n in sizes])
    return shard, row


def test_subclass_with_other_shard_names_reads_rows(tmp_path):
    _write_shard(str(tmp_path / "toy-00-of-02.arrow"), 0, 70)
    _write_shard(str(tmp_path / "toy-01-of-02.arrow"), 1, 10)
    ds = Toy(str(tmp_path), *_all_rows([70, 10]))
    assert len(ds) == 80
    assert ds.raw(69)[0] == "s0_r69"
    assert ds.raw(70)[0] == "s1_r0"
    assert ds[79]["img"].shape == (3, 224, 224)
    assert ds[79]["img_name"] == "s1_r9"
    assert ds[79]["index"] == 79
    assert float(ds[79]["img"].min()) >= -1.0 and float(ds[79]["img"].max()) <= 1.0


def test_missing_second_shard_raises_at_construction(tmp_path):
    _write_shard(str(tmp_path / "toy-00-of-02.arrow"), 0, 70)
    with pytest.raises(FileNotFoundError, match="1 shards missing"):
        Toy(str(tmp_path), *_all_rows([70, 10]))


def test_too_short_shard_raises_at_construction(tmp_path):
    _write_shard(str(tmp_path / "toy-00-of-02.arrow"), 0, 70)
    _write_shard(str(tmp_path / "toy-01-of-02.arrow"), 1, 5)  # index addresses row 9
    with pytest.raises(FileNotFoundError, match="1 too short"):
        Toy(str(tmp_path), *_all_rows([70, 10]))


def test_base_class_without_shard_name_raises(tmp_path):
    ds = A.ArrowCorpus.__new__(A.ArrowCorpus)
    ds._set_rows(np.array([0]), np.array([0]), np.array([0]), str(tmp_path), 224, False, 0)
    with pytest.raises(NotImplementedError):
        ds._check_shards()
