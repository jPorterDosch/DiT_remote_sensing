"""Git-10M (lcybuaa/Git-10M) as a label-free training set, filtered by the geographic exclusion.

The Hub ships 787 arrow STREAM shards (1000-row batches, JPEG bytes inline, ~290 MB each).
A stream file has no random access, so each shard is rewritten once into an arrow FILE with
64-row record batches (`convert_shard`): row r of shard s is then batch r // 64, slot r % 64, a
~1.3 MB memory-mapped read. Pixels are untouched (the original JPEG bytes are copied), so the
(shard, row) coordinates of eval/geo_dedup's manifests address the same images.

Which rows train is decided ONLY by an index built from a dedup manifest directory
(`build_index`): retained.csv always, unassessable.csv when --include-unassessable is given
(the 2.05 M rows whose sources publish no coordinates -- see
experiments/isaac/geo_dedup/PROTOCOL.md section 7). Excluded rows are never readable through
this module. The index records the sha256 of the manifests it was built from, and the counts
are checked against the manifest's report.json, so a run's meta.json names its data exactly.

Transform = the repo's EuroSATDataset one (bicubic to img_size, [-1, 1]) plus an optional
horizontal flip (p = 0.5, Python `random`, which utils.seed_worker seeds per worker).

    python -m datasets.git10m convert --shards 0-65 --stream-dir S --out-dir A --meta-dir M --hf-download --rm-stream
    python -m datasets.git10m build-index --manifest-dir D/dedup_v2_b1000 --out A/index_all.npz --include-unassessable
    python -m datasets.git10m check --arrow-dir A --index A/index_all.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from registry import register_dataset
from torch.utils.data import DataLoader
from utils import seed_worker

from .arrow_corpus import BATCH_ROWS, ArrowCorpus
from .utils import _strided_indices

if __name__ == "__main__":
    # `python -m datasets.git10m ...`: importing the package above already imported and
    # registered THIS module as datasets.git10m; executing the body again as __main__ would
    # register "git10m" twice (registry refuses). Delegate to the registered module's CLI.
    from datasets import git10m as _registered

    sys.exit(_registered.main())

HF_REPO = "lcybuaa/Git-10M"  # == eval.geo_dedup.HF_REPO (asserted in tests)
N_SHARDS = 787  # == eval.geo_dedup.N_SHARDS
KEEP_COLS = ["img_name", "caption", "resolution", "Google_location", "img_quality_score"]
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
MANIFESTS = {"retained": "retained.csv", "unassessable": "unassessable.csv"}


def shard_name(i: int) -> str:
    return f"data-{i:05d}-of-{N_SHARDS:05d}.arrow"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- conversion
def convert_shard(stream_path: str, out_path: str, meta_parquet: str | None = None) -> int:
    """Rewrite one Hub stream shard as a random-access arrow file (BATCH_ROWS-row batches,
    SCHEMA). With `meta_parquet` (eval/geo_dedup's meta-NNNNN.parquet for this shard) the row
    count and the img_name sequence must match it, so the manifests' (shard, row) address the
    rows written here. Written via .tmp + os.replace: present == complete."""
    if os.path.exists(out_path):
        raise FileExistsError(out_path)
    tmp = out_path + ".tmp"
    names = []
    n = 0
    with pa.memory_map(stream_path) as src, pa.OSFile(tmp, "wb") as sink:
        tables = []
        for batch in ipc.open_stream(src):
            img = batch.column("image")
            cols = [batch.column("img_name"), img.field("bytes")] + [batch.column(c) for c in KEEP_COLS[1:]]
            tables.append(pa.Table.from_arrays(cols, schema=SCHEMA))
            names.extend(batch.column("img_name").to_pylist())
        writer = ipc.new_file(sink, SCHEMA)
        for b in _rebatch(pa.concat_tables(tables)):
            writer.write_batch(b)
            n += b.num_rows
        writer.close()
    if meta_parquet is not None:
        import pandas as pd

        meta = pd.read_parquet(meta_parquet, columns=["row", "img_name"])
        if len(meta) != n or not (meta["row"].to_numpy() == np.arange(n)).all():
            os.remove(tmp)
            raise ValueError(f"{stream_path}: {n} rows, meta parquet has {len(meta)}")
        if meta["img_name"].tolist() != names:
            os.remove(tmp)
            raise ValueError(f"{stream_path}: img_name sequence differs from {meta_parquet}")
    os.replace(tmp, out_path)
    return n


def _rebatch(tbl: pa.Table):
    """BATCH_ROWS-row record batches of `tbl` regardless of its chunk layout. Table.to_batches
    splits per chunk (the 64-row grid would restart at every 1000-row Hub batch), and
    combine_chunks cannot merge a binary column past 2 GiB (shard 778 failed that way), so each
    64-row window is sliced and combined on its own."""
    for off in range(0, tbl.num_rows, BATCH_ROWS):
        win = tbl.slice(off, BATCH_ROWS).combine_chunks()
        (b,) = win.to_batches()
        if b.num_rows != BATCH_ROWS and off + BATCH_ROWS < tbl.num_rows:
            raise RuntimeError(f"batch at row {off} has {b.num_rows} rows")
        yield b


def fetch_convert(
    shards: range, stream_dir: str, out_dir: str, meta_dir: str, hf_download: bool, rm_stream: bool
) -> None:
    """Resumable: a shard whose arrow file exists is skipped."""
    os.makedirs(out_dir, exist_ok=True)
    for i in shards:
        out = os.path.join(out_dir, shard_name(i))
        if os.path.exists(out):
            print(f"shard {i}: {out} exists, skipping", flush=True)
            continue
        path = os.path.join(stream_dir, "train", shard_name(i))  # the Hub's layout
        if not os.path.exists(path):
            if not hf_download:
                raise SystemExit(f"{path} missing (pass --hf-download)")
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(
                HF_REPO, f"train/{shard_name(i)}", repo_type="dataset", local_dir=stream_dir
            )
            if os.path.realpath(got) != os.path.realpath(path):
                raise SystemExit(f"hf_hub_download landed at {got}, expected {path}")
        meta = os.path.join(meta_dir, f"meta-{i:05d}.parquet")
        if not os.path.exists(meta):
            raise SystemExit(f"{meta} missing: the conversion is checked against the dedup metadata")
        n = convert_shard(path, out, meta)
        print(f"shard {i}: {n} rows -> {out}", flush=True)
        if rm_stream:
            os.remove(path)


# ----------------------------------------------------------------------------- index
def build_index(manifest_dir: str, out: str, include_unassessable: bool, shards: range | None = None) -> dict:
    """index npz from a dedup manifest directory: ONE ROW PER IMAGE, sorted by (shard, row).
    Git-10M repeats public-dataset images on adjacent rows (one row per caption, identical
    JPEG bytes; 1,243,614 of the 10,084,854 retained + unassessable rows carry a repeated
    img_name). Training is unconditional on the image, so a repeated row is pure repetition:
    it doubles that image's sampling weight, inflates `train_size` / `unique_fraction_seen`,
    and -- because the holdout is drawn by (shard, row) -- puts a held-out image's twin row
    into the training set (125 of index_all's 1,006 holdout rows had a trained-on twin,
    verified 2026-10-13). The first (lowest shard, row) row of every img_name is kept; the
    count dropped is recorded. Twins share name, source and location, so they always fall in
    the same bucket and the kept row is the same under every manifest directory.
    Counts are checked against the directory's report.json; sha256 of every manifest read is
    recorded. `shards` restricts the index to those shards (smoke runs on a partial
    conversion); the restriction is recorded and such an index is refused for a full run
    (Git10MWrapper refuses it without --max-samples)."""
    import pandas as pd

    report = json.load(open(os.path.join(manifest_dir, "report.json")))
    parts, shas = [], {}
    for key in ["retained"] + (["unassessable"] if include_unassessable else []):
        p = os.path.join(manifest_dir, MANIFESTS[key])
        df = pd.read_csv(
            p, usecols=["shard", "row", "source", "img_name"], dtype={"shard": np.int32, "row": np.int64}
        )
        if len(df) != report[key]:
            raise ValueError(f"{p}: {len(df)} rows, report.json says {report[key]}")
        df["bucket"] = key
        parts.append(df)
        shas[MANIFESTS[key]] = _sha256(p)
    df = pd.concat(parts, ignore_index=True)
    if df.duplicated(["shard", "row"]).any():
        raise ValueError("manifests overlap in (shard, row)")
    if (df["shard"] < 0).any() or (df["shard"] >= N_SHARDS).any():
        raise ValueError("shard out of range")
    if df["img_name"].isna().any() or (df["img_name"] == "").any():
        raise ValueError("empty img_name in a manifest")
    df = df.sort_values(["shard", "row"], kind="stable").reset_index(drop=True)
    manifest_rows = int(len(df))
    dup = df["img_name"].duplicated(keep="first")  # dedup over the FULL manifests, before any shard subset
    if (df.groupby("img_name")["bucket"].nunique() > 1).any():
        raise ValueError("an img_name appears in two buckets: twins must share source and location")
    df = df[~dup].reset_index(drop=True)
    duplicate_rows_dropped = int(dup.sum())
    if shards is not None:
        df = df[df["shard"].isin(list(shards))].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no manifest rows in shards {shards}")
    sources = sorted(df["source"].unique())
    buckets = sorted(df["bucket"].unique())
    rec = {
        "shard": df["shard"].to_numpy(np.int16),
        "row": df["row"].to_numpy(np.int32),
        "source_code": np.array([sources.index(s) for s in df["source"]], dtype=np.int8),
        "sources": np.array(sources),
        "bucket_code": np.array([buckets.index(b) for b in df["bucket"]], dtype=np.int8),
        "buckets": np.array(buckets),
        "manifest_dir": np.array(os.path.abspath(manifest_dir)),
        "manifest_sha256": np.array(json.dumps(shas, sort_keys=True)),
        "include_unassessable": np.array(bool(include_unassessable)),
        "shard_subset": np.array("" if shards is None else f"{shards.start}-{shards.stop - 1}"),
        "git10m_rows": np.array(int(report["git10m_rows"])),
        "buffer_m": np.array(float(report["buffer_m"])),
        "one_row_per_image": np.array(True),
        "manifest_rows": np.array(manifest_rows),
        "duplicate_rows_dropped": np.array(duplicate_rows_dropped),
    }
    tmp = out + ".tmp.npz"
    np.savez(tmp, **rec)
    os.replace(tmp, out)
    summary = {
        "rows": int(len(df)),
        "one_row_per_image": True,
        "manifest_rows": manifest_rows,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "per_bucket": {b: int((df["bucket"] == b).sum()) for b in buckets},
        "per_source": {s: int((df["source"] == s).sum()) for s in sources},
        "manifest_sha256": shas,
        "include_unassessable": bool(include_unassessable),
        "shard_subset": "" if shards is None else f"{shards.start}-{shards.stop - 1}",
    }
    with open(out + ".report.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def index_description(index_npz: str) -> dict:
    z = np.load(index_npz)
    return {
        "index": os.path.abspath(index_npz),
        "rows": int(z["shard"].shape[0]),
        "manifest_dir": str(z["manifest_dir"]),
        "manifest_sha256": json.loads(str(z["manifest_sha256"])),
        "include_unassessable": bool(z["include_unassessable"]),
        "shard_subset": str(z["shard_subset"]) if "shard_subset" in z.files else "",
        "buffer_m": float(z["buffer_m"]),
        "git10m_rows": int(z["git10m_rows"]),
        "one_row_per_image": bool(z["one_row_per_image"]) if "one_row_per_image" in z.files else False,
        "duplicate_rows_dropped": int(z["duplicate_rows_dropped"])
        if "duplicate_rows_dropped" in z.files
        else None,
        "buckets": {str(b): int((z["bucket_code"] == i).sum()) for i, b in enumerate(z["buckets"])},
    }


# ----------------------------------------------------------------------------- dataset
ROW_ID_STRIDE = 1 << 20  # > rows per shard (max 13,357): global row id = shard * stride + row


def holdout_mask(shard: np.ndarray, row: np.ndarray, holdout_every: int) -> np.ndarray:
    """Index-independent holdout: every `holdout_every`-th global row id (shard * 2^20 + row)."""
    if int(row.max(initial=0)) >= ROW_ID_STRIDE:
        raise ValueError(f"row index {int(row.max())} >= ROW_ID_STRIDE")
    return (shard.astype(np.int64) * ROW_ID_STRIDE + row.astype(np.int64)) % holdout_every == 0


class Git10MDataset(ArrowCorpus):
    """Rows of an index npz, read from the converted arrow shards.

    which: "train" = every row except the holdout, "val" = the holdout rows of the RETAINED
    bucket, "all" = every row. Holdout membership is a function of the ROW IDENTITY (shard,
    row), not of its position in the index, and the index holds one row per image, so
    index_retained and index_all hold out the same images, a row trained on under one index is
    never another index's validation row, and no training row is a byte-identical twin of a
    validation row. "val" is restricted to the retained bucket so that every index (with or
    without the unassessable rows) validates on the SAME image set; the unassessable holdout
    rows of index_all are neither trained on nor validated (about 200 rows). Across manifest
    PASSES (v2 vs the city pass, whose retained set is smaller) the retained holdouts differ
    (758 vs 656 rows, review 2026-10-13), so `val_ref_npz` names the index whose retained
    holdout defines val for every run: its rows must all be present in this index (they are
    holdout rows here too, by the row-id rule, so never trained on), and a run's val.jsonl is
    then comparable between indexes. Pass the most restrictive index (index_retained_cities_unique).

    hflip: decided per ROW from (flip_seed, index position), not from the DataLoader worker's
    RNG, so a resumed run and every arm flip the same images; a run covers < 1 pass over the
    corpus, so a per-row flip is a per-draw flip.
    """

    n_shards = N_SHARDS
    shard_name = staticmethod(shard_name)

    def __init__(
        self,
        arrow_dir: str,
        index_npz: str,
        img_size: int = 224,
        hflip: bool = False,
        which: str = "train",
        holdout_every: int = 10_000,
        flip_seed: int = 0,
        val_ref_npz: str | None = None,
    ):
        if which not in ("train", "val", "all"):
            raise ValueError(which)
        if holdout_every < 2:
            raise ValueError("holdout_every must be >= 2")
        z = np.load(index_npz)
        if not ("one_row_per_image" in z.files and bool(z["one_row_per_image"])):
            raise ValueError(
                f"{index_npz}: built before the one-row-per-image rule; rebuild it (build-index)"
            )
        shard, row = z["shard"].astype(np.int64), z["row"].astype(np.int64)
        pos = np.arange(shard.shape[0])
        hold = holdout_mask(shard, row, holdout_every)
        buckets = [str(b) for b in z["buckets"]]
        if "retained" not in buckets:
            raise ValueError(f"{index_npz}: no retained bucket, nothing to validate on")
        retained = z["bucket_code"] == buckets.index("retained")
        val = hold & retained
        self.val_ref = None
        if val_ref_npz is not None:
            r = np.load(val_ref_npz)
            if not ("one_row_per_image" in r.files and bool(r["one_row_per_image"])):
                raise ValueError(f"{val_ref_npz}: val reference built before the one-row-per-image rule")
            rb = [str(b) for b in r["buckets"]]
            rs, rr = r["shard"].astype(np.int64), r["row"].astype(np.int64)
            ref_val = holdout_mask(rs, rr, holdout_every) & (r["bucket_code"] == rb.index("retained"))
            ref_ids = rs[ref_val] * ROW_ID_STRIDE + rr[ref_val]
            ids = shard * ROW_ID_STRIDE + row
            here = np.isin(ids, ref_ids)
            if here.sum() != len(ref_ids):
                raise ValueError(
                    f"{val_ref_npz}: {len(ref_ids) - int(here.sum())} of its {len(ref_ids)} val rows are not in {index_npz}; "
                    "the val reference must be the most restrictive index"
                )
            if not (hold[here]).all():
                raise RuntimeError("val reference rows are not holdout rows here: holdout_every differs")
            val = here
            self.val_ref = os.path.abspath(val_ref_npz)
        keep = {"train": ~hold, "val": val, "all": np.ones_like(hold)}[which]
        self._set_rows(shard[keep], row[keep], pos[keep], arrow_dir, img_size, hflip, flip_seed)
        self.which = which
        self.holdout_every = holdout_every
        self.index_npz = index_npz
        self._check_shards()

    def describe(self) -> dict:
        return {
            **index_description(self.index_npz),
            "arrow_dir": os.path.abspath(self.arrow_dir),
            "which": self.which,
            "holdout_every": self.holdout_every,
            "n": len(self),
            "n_shards": self.n_shards,
            "img_size": self.img_size,
            "hflip": self.hflip,
            "flip_seed": self.flip_seed,
            "val_ref": self.val_ref,
        }


@register_dataset("git10m")
class Git10MWrapper:
    """run.py dataset for finetune-diffusion: the Git-10M rows of cfg.git10m.index (one row per
    image, built from the geographic-exclusion manifests), read from the arrow shards under
    cfg.dataset.path. Label-free: category_list is empty, so the task builds no probe head and
    the model conditions on the null prompt only. Three views: the flipped training rows, the
    unflipped measurement source (fixed batches) and the retained holdout as validation
    (Git10MDataset docstring: never trained on under any index)."""

    def __init__(self, cfg) -> None:
        g = cfg.git10m
        # finetune-diffusion trains on it; gen-eval (tasks/gen_eval.py) reads only the val view as
        # the FID reference. Every other task (extract, probes) is refused: the corpus is unlabeled.
        if cfg.task not in ("finetune-diffusion", "gen-eval"):
            raise ValueError(
                f"dataset git10m is a training corpus for task='finetune-diffusion' (or the gen-eval "
                f"reference), got {cfg.task!r}"
            )
        if not g.index:
            raise ValueError(
                "dataset git10m needs --git10m.index <index npz> (python -m datasets.git10m build-index)"
            )
        if cfg.label_fraction != 1.0:
            raise ValueError("git10m has no labels; run with label_fraction=1.0")
        img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else int(cfg.img_size)
        if img_size % 32:
            raise ValueError(
                f"img_size {img_size}: must be a multiple of 32 (VAE /8, 2x2 packing, DINO /16 grids)"
            )
        kw = dict(
            arrow_dir=cfg.dataset.path,
            index_npz=g.index,
            img_size=img_size,
            holdout_every=g.holdout_every,
            flip_seed=cfg.seed,
            val_ref_npz=g.val_index or g.index,
        )
        if not g.val_index:
            print(
                "[git10m] WARNING: git10m.val_index not set: val losses are comparable only within this index",
                flush=True,
            )
        self.category_list: list[str] = []
        self.train_full = Git10MDataset(hflip=g.hflip, which="train", **kw)
        self.meas_src = Git10MDataset(which="train", **kw)  # no flip: measurement batches are fixed
        self.val_full = Git10MDataset(which="val", **kw)  # the holdout: never trained on
        self.describe = self.train_full.describe()
        if self.describe["shard_subset"] and cfg.max_samples is None:
            raise ValueError(
                f"{g.index} covers shards {self.describe['shard_subset']} only (a smoke index): run it with --max-samples"
            )
        self.train = self.train_full
        if cfg.max_samples is not None:
            self.train = torch.utils.data.Subset(
                self.train_full, _strided_indices(len(self.train_full), cfg.max_samples)
            )
        self.batch_size = int(cfg.batch_size)
        if len(self.train) < self.batch_size:
            raise ValueError(f"train set {len(self.train)} < batch size {self.batch_size}")

    def train_loader(self, cfg, consumed_microbatches: int = 0) -> DataLoader:
        """Seeded permutation per pass, positioned at the microbatch the run has consumed, so a
        resumed run reads exactly the rows the interrupted one would have (drop_last batches)."""
        bpe = len(self.train) // self.batch_size
        sampler = PermutationSampler(
            len(self.train),
            cfg.seed,
            epoch=consumed_microbatches // bpe,
            pos=(consumed_microbatches % bpe) * self.batch_size,
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True,
        )

    def get_data(self, cfg) -> dict:
        val = DataLoader(
            self.val_full,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        return {"train": self.train_loader(cfg, 0), "val": val}


class PermutationSampler(torch.utils.data.Sampler):
    """Seeded permutation per epoch, resumable at (epoch, pos): the loader after a resume
    yields exactly the rows the interrupted run would have."""

    def __init__(self, n: int, seed: int, epoch: int = 0, pos: int = 0):
        self.n, self.seed = n, seed
        self.epoch, self.pos = epoch, pos

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        perm = torch.randperm(self.n, generator=g)
        start = self.pos
        self.pos = 0  # a later epoch starts at 0
        for i in range(start, self.n):
            yield int(perm[i])
        self.epoch += 1

    def __len__(self):
        return self.n - self.pos


# ----------------------------------------------------------------------------- CLI
def _parse_range(s: str) -> range:
    a, b = s.split("-") if "-" in s else (s, s)
    a, b = int(a), int(b)
    if not (0 <= a <= b < N_SHARDS):
        raise argparse.ArgumentTypeError(s)
    return range(a, b + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--shards", type=_parse_range, required=True)
    c.add_argument("--stream-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--meta-dir", required=True)
    c.add_argument("--hf-download", action="store_true")
    c.add_argument("--rm-stream", action="store_true")
    b = sub.add_parser("build-index")
    b.add_argument("--manifest-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--include-unassessable", action="store_true")
    b.add_argument("--shards", type=_parse_range, default=None, help="restrict to these shards (smoke only)")
    k = sub.add_parser("check")
    k.add_argument("--arrow-dir", required=True)
    k.add_argument("--index", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        fetch_convert(a.shards, a.stream_dir, a.out_dir, a.meta_dir, a.hf_download, a.rm_stream)
    elif a.cmd == "build-index":
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists")
        print(json.dumps(build_index(a.manifest_dir, a.out, a.include_unassessable, a.shards), indent=1))
    else:
        ds = Git10MDataset(a.arrow_dir, a.index, which="all")
        print(json.dumps(ds.describe(), indent=1))
        name, raw = ds.raw(0)
        print(f"row 0: {name} {len(raw)} bytes; GIT10M_CHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
