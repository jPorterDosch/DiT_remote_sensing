"""Shared reader for Arrow shard corpora: rows (shard, row) read from 64-row record-batch Arrow
FILES under `arrow_dir`, one JPEG/PNG byte string per row, decoded to [-1, 1] at `img_size`.

Lifted unchanged out of `datasets.git10m.Git10MDataset` so a second corpus (chest X-rays) can use the
same shard format with other shard names and count. A subclass sets `n_shards` and `shard_name`,
decides which rows it serves, and calls `_set_rows(...)` then `_check_shards()` in its constructor.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor

BATCH_ROWS = 64


class ArrowCorpus(Dataset):
    """Rows (self.shard, self.row, self.index_pos) read from 64-row Arrow shards under self.arrow_dir."""

    n_shards: int  # set by the subclass (instance value is overwritten by _check_shards)
    convert_hint = "python -m datasets.git10m convert ..."  # named in the missing-shard error

    @staticmethod
    def shard_name(i: int) -> str:
        raise NotImplementedError("ArrowCorpus subclasses define shard_name(i)")

    def _set_rows(
        self,
        shard: np.ndarray,
        row: np.ndarray,
        index_pos: np.ndarray,
        arrow_dir: str,
        img_size: int,
        hflip: bool,
        flip_seed: int,
    ) -> None:
        self.shard, self.row = shard, row
        self.index_pos = index_pos
        self.arrow_dir = arrow_dir
        self.img_size = img_size
        self.hflip = bool(hflip)
        self.flip_seed = int(flip_seed)
        self._readers: dict[int, ipc.RecordBatchFileReader] = {}

    def _check_shards(self):
        """Every shard the index addresses exists and holds its highest addressed row."""
        need = {}
        for s, r in zip(self.shard, self.row):
            if r > need.get(s, -1):
                need[int(s)] = int(r)
        missing, short = [], []
        for s, rmax in sorted(need.items()):
            p = os.path.join(self.arrow_dir, self.shard_name(s))
            if not os.path.exists(p):
                missing.append(s)
                continue
            with pa.memory_map(p) as src:
                rd = ipc.open_file(src)
                nb = rd.num_record_batches
                n = (nb - 1) * BATCH_ROWS + rd.get_batch(nb - 1).num_rows if nb else 0
                if nb > 1 and rd.get_batch(0).num_rows != BATCH_ROWS:
                    raise ValueError(f"{p}: batch 0 has {rd.get_batch(0).num_rows} rows, not {BATCH_ROWS}")
            if rmax >= n:
                short.append((s, n, rmax))
        if missing or short:
            raise FileNotFoundError(
                f"{self.arrow_dir}: {len(missing)} shards missing {missing[:5]}, "
                f"{len(short)} too short {short[:3]} ({self.convert_hint})"
            )
        self.n_shards = len(need)

    def _reader(self, s: int) -> ipc.RecordBatchFileReader:
        rd = self._readers.get(s)
        if rd is None:
            rd = ipc.open_file(pa.memory_map(os.path.join(self.arrow_dir, self.shard_name(s))))
            self._readers[s] = rd
        return rd

    def __len__(self):
        return int(self.shard.shape[0])

    def raw(self, i: int) -> tuple[str, bytes]:
        s, r = int(self.shard[i]), int(self.row[i])
        b = self._reader(s).get_batch(r // BATCH_ROWS)
        j = r % BATCH_ROWS
        return b.column("img_name")[j].as_py(), b.column("image_bytes")[j].as_py()

    def __getitem__(self, i):
        name, raw = self.raw(i)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
        x = (PILToTensor()(img) / 255.0 - 0.5) * 2
        if self.hflip and np.random.default_rng([self.flip_seed, int(self.index_pos[i])]).random() < 0.5:
            x = x.flip(-1)
        # TODO: make the label configurable; -1 because the alignment corpus is label-free.
        return {"img": x, "label": -1, "index": int(self.index_pos[i]), "img_name": name}
