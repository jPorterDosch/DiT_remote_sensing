"""NIH ChestX-ray14 as a label-free image corpus for domain adaptation, split by PATIENT.

The 12 `images_NNN.tar.gz` archives (112,120 PNGs, 1024 x 1024) are rewritten once into 12 arrow
FILES with 64-row record batches (`convert_archive`), PNG bytes untouched, so row r of shard s is
batch r // 64, slot r % 64. Which rows may train is decided ONLY by a manifest (`build_manifest`)
and the index built from it (`build_index`): the official train_val list is split by patient
(eval.cxr_splits.cxr14_buckets, keyed on the CSV "Patient ID" string, e.g. "1") into "align" (retained) and "validation"; the official test list
is bucket "test" and never enters the index, so no test image is reachable through
`Cxr14Dataset`. The manifest and the index are what the leakage gate reads.

Transform = the repo's ArrowCorpus one (bicubic to img_size, [-1, 1], decoded to RGB in the
reader); no horizontal flip, a flip would swap the heart side.

    python -m datasets.cxr14 convert --shard 0 --raw-dir R --out-dir A
    python -m datasets.cxr14 build-manifest --arrow-dir A --raw-dir R --out-dir M
    python -m datasets.cxr14 build-index --manifest-dir M --out A_index.npz
    python -m datasets.cxr14 check --arrow-dir A --manifest-dir M --index A_index.npz
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

from .arrow_corpus import BATCH_ROWS, ArrowCorpus

N_ARCHIVES = 12
N_IMAGES = 112120
SCHEMA = pa.schema([("img_name", pa.string()), ("image_bytes", pa.binary()), ("patient_id", pa.int32())])
ALLOWED_MODES = {"L", "RGB", "RGBA", "P", "LA"}
LABEL_FILES = ("Data_Entry_2017_v2020.csv", "train_val_list.txt", "test_list.txt")
BUCKETS = ["retained", "validation"]
# Pinned on the shipped label files (controller, 2026-11-14); `check` fails on any difference.
PINNED_N = {"align": 77780, "validation": 8744, "test": 25596}
PINNED_PATIENTS = {"align": 25207, "validation": 2801, "test": 2797}


def archive_name(i: int) -> str:
    return f"images_{i + 1:03d}.tar.gz"


def shard_name(i: int) -> str:
    return f"cxr14-{i:02d}-of-{N_ARCHIVES:02d}.arrow"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta_path(arrow_dir: str, i: int) -> str:
    return os.path.join(arrow_dir, "meta", shard_name(i) + ".json")


def _patient_prefix(name: str) -> str | None:
    p = name[:8]
    return p if len(p) == 8 and p.isascii() and p.isdigit() else None


# ----------------------------------------------------------------------------- conversion
def convert_archive(tar_path: str, out_dir: str, shard: int) -> dict:
    """One archive -> one arrow file (rows in tar member order, the member's bytes untouched) plus
    meta/<shard>.json. Every member is fully decoded (`Image.load`) and its PIL mode and size
    counted, never converted. Fails (SystemExit) on an undecodable member, a mode outside
    ALLOWED_MODES (16-bit `I;16`, `I`, `F`), a name without an eight-digit patient prefix, a
    duplicate name, or an existing output / `.partial`. A failed run leaves its `.partial` for
    the user (the next run refuses to start over it)."""
    if not 0 <= shard < N_ARCHIVES:
        raise SystemExit(f"shard {shard} outside 0..{N_ARCHIVES - 1}")
    out = os.path.join(out_dir, shard_name(shard))
    meta_path = _meta_path(out_dir, shard)
    for p in (out, meta_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists: refusing to overwrite")
    for p in (out + ".partial", meta_path + ".partial"):
        if os.path.exists(p):
            raise SystemExit(f"{p}: a partial output exists, resolve it by hand")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    names: list[str] = []
    seen: set[str] = set()
    modes: collections.Counter = collections.Counter()
    sizes: collections.Counter = collections.Counter()
    alpha_nonopaque = 0
    buf: list[tuple[str, bytes, int]] = []

    def flush(writer):
        if buf:
            cols = [pa.array([b[0] for b in buf], pa.string()), pa.array([b[1] for b in buf], pa.binary())]
            cols.append(pa.array([b[2] for b in buf], pa.int32()))
            writer.write_batch(pa.RecordBatch.from_arrays(cols, schema=SCHEMA))
            buf.clear()

    with tarfile.open(tar_path, "r|gz") as tf, pa.OSFile(out + ".partial", "wb") as sink:
        writer = ipc.new_file(sink, SCHEMA)
        ordinal = -1
        for m in tf:
            if not m.isfile():
                continue
            ordinal += 1
            name = os.path.basename(m.name)
            prefix = _patient_prefix(name)
            if prefix is None:
                raise SystemExit(f"member {ordinal} ({name}): no eight-digit patient prefix")
            if name in seen:
                raise SystemExit(f"member {ordinal} ({name}): duplicate name")
            data = tf.extractfile(m).read()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    mode, size = im.mode, im.size
                    if mode not in ALLOWED_MODES:
                        raise SystemExit(
                            f"member {ordinal} ({name}): PIL mode {mode!r} not allowed "
                            f"(16-bit or other non-8-bit mode; allowed {sorted(ALLOWED_MODES)})"
                        )
                    im.load()
                    if mode in ("RGBA", "LA"):
                        nonopaque = im.getchannel("A").getextrema()[0] < 255
                    else:
                        nonopaque = mode == "P" and "transparency" in im.info
            except SystemExit:
                raise
            except Exception as e:
                raise SystemExit(
                    f"member {ordinal} ({name}): PIL cannot decode: {type(e).__name__}: {e}"
                ) from e
            seen.add(name)
            names.append(name)
            modes[mode] += 1
            alpha_nonopaque += int(nonopaque)
            sizes[f"{size[0]}x{size[1]}"] += 1
            buf.append((name, data, int(prefix)))
            if len(buf) == BATCH_ROWS:
                flush(writer)
        flush(writer)
        writer.close()
    if not names:
        raise SystemExit(f"{tar_path}: no image members")
    os.rename(out + ".partial", out)
    meta = {
        "shard": shard,
        "rows": len(names),
        "sha256": _sha256(out),
        "pil_modes": dict(modes),
        "sizes": dict(sizes),
        "alpha_nonopaque": alpha_nonopaque,
        "names": names,
    }
    with open(meta_path + ".partial", "w") as f:
        json.dump(meta, f)
    os.rename(meta_path + ".partial", meta_path)
    return meta


# ----------------------------------------------------------------------------- manifest
def _read_list(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]


def _refuse_existing(target: str) -> None:
    if os.path.exists(target):
        raise SystemExit(f"{target} exists: refusing to overwrite")
    if os.path.exists(target + ".partial"):
        raise SystemExit(f"{target}.partial exists: resolve it by hand")


def build_manifest(arrow_dir: str, raw_dir: str, out_dir: str, shards=None) -> dict:
    """Patient-level manifest of converted shards. shards=None is the full corpus: all N_ARCHIVES
    shards must exist and their image names must equal the CSV's `Image Index` and the union of
    the two official lists (which must be disjoint). A subset is a smoke run: report.json says
    complete=false, smoke=true and build_index then needs allow_smoke. Buckets always come from the
    FULL label files, so a smoke manifest assigns the same bucket the full run would.
    Writes rows.csv (shard,row,img_name,patient_id,bucket), retained.csv (bucket "align") and
    validation.csv (shard,row,source,img_name; source = eight-digit patient id) and report.json."""
    from eval import cxr_splits as S

    smoke = shards is not None
    which = list(range(N_ARCHIVES)) if shards is None else sorted(set(int(s) for s in shards))
    if smoke and (not which or which[0] < 0 or which[-1] >= N_ARCHIVES):
        raise SystemExit(f"shards {shards} outside 0..{N_ARCHIVES - 1}")
    _refuse_existing(out_dir)
    # shards: identity first (sha256 against the meta written at conversion), then content
    metas, shas = {}, {}
    for i in which:
        mp = _meta_path(arrow_dir, i)
        if not os.path.exists(mp):
            raise SystemExit(f"shard {i}: {mp} missing (python -m datasets.cxr14 convert)")
        meta = json.load(open(mp))
        sp = os.path.join(arrow_dir, shard_name(i))
        if not os.path.exists(sp):
            raise SystemExit(f"shard {i}: {sp} missing")
        sha = _sha256(sp)
        if sha != meta["sha256"] or meta["shard"] != i or meta["rows"] != len(meta["names"]):
            raise SystemExit(f"shard {i}: sha256 or meta disagree with the file on disk")
        with pa.memory_map(sp) as src:
            rd = ipc.open_file(src)
            on_disk = [
                v
                for b in range(rd.num_record_batches)
                for v in rd.get_batch(b).column("img_name").to_pylist()
            ]
        if on_disk != meta["names"]:
            raise SystemExit(f"shard {i}: img_name sequence differs from its meta file")
        metas[i], shas[shard_name(i)] = meta, sha
    shard_names = [n for i in which for n in metas[i]["names"]]
    if len(set(shard_names)) != len(shard_names):
        raise SystemExit(f"{len(shard_names) - len(set(shard_names))} image names appear twice across shards")
    # label files: the shipped artifacts must agree with each other (CLAUDE.md rule 16)
    csv_path = os.path.join(raw_dir, LABEL_FILES[0])
    csv = pd.read_csv(csv_path, usecols=["Image Index", "Patient ID"], dtype=str)
    tv_list = _read_list(os.path.join(raw_dir, LABEL_FILES[1]))
    te_list = _read_list(os.path.join(raw_dir, LABEL_FILES[2]))
    tv, te = set(tv_list), set(te_list)
    if len(tv) != len(tv_list) or len(te) != len(te_list):
        raise SystemExit("an official list holds duplicate names")
    if tv & te:
        raise SystemExit(f"the two official lists are not disjoint: {len(tv & te)} names in both")
    csv_names = csv["Image Index"].tolist()
    if len(set(csv_names)) != len(csv_names):
        raise SystemExit(f"Image Index holds {len(csv_names) - len(set(csv_names))} duplicate names")
    if set(csv_names) != tv | te:
        raise SystemExit(
            f"CSV Image Index ({len(csv_names)}) != union of the two lists ({len(tv | te)}): "
            f"{len(set(csv_names) - (tv | te))} only in CSV, {len((tv | te) - set(csv_names))} only in lists"
        )
    if not smoke and len(csv_names) != N_IMAGES:
        raise SystemExit(f"CSV holds {len(csv_names)} images, N_IMAGES is {N_IMAGES}")
    only_shards, only_csv = set(shard_names) - set(csv_names), set(csv_names) - set(shard_names)
    if only_shards or (not smoke and only_csv):
        raise SystemExit(
            f"shard names differ from the CSV Image Index: {len(only_shards)} only in shards "
            f"{sorted(only_shards)[:3]}, {len(only_csv)} only in CSV {sorted(only_csv)[:3]}"
        )
    prefix = csv["Image Index"].str[:8]
    expected = pd.to_numeric(prefix, errors="coerce")
    actual = pd.to_numeric(csv["Patient ID"], errors="coerce")
    bad = int((expected.isna() | actual.isna() | (expected != actual)).sum())
    if bad:
        raise SystemExit(f"{bad} CSV rows have a Patient ID different from int(name[:8])")
    # The pinned split keys patients by the CSV "Patient ID" string as shipped ("1", unpadded), NOT by
    # the eight-digit name prefix ("00000001"): the validation patients are a hash of that key, so the
    # two keys pick different patients. Pin the canonical form so a re-padded CSV fails loudly.
    canon = csv["Image Index"].map(lambda n: str(int(n[:8])) if n[:8].isdigit() else None)
    noncanon = int((csv["Patient ID"] != canon).sum())
    if noncanon:
        raise SystemExit(
            f"{noncanon} CSV rows have a Patient ID that is not the canonical str(int(name[:8])) "
            "(zero-padded, whitespace or signed); the split is keyed on that exact string"
        )
    patient_of = dict(zip(csv_names, csv["Patient ID"], strict=True))
    bucket_of = S.cxr14_buckets(patient_of, tv, te)
    rows = [(i, r, n, int(n[:8]), bucket_of[n]) for i in which for r, n in enumerate(metas[i]["names"])]
    df = pd.DataFrame(rows, columns=["shard", "row", "img_name", "patient_id", "bucket"])
    n_by = {b: int((df["bucket"] == b).sum()) for b in ("align", "validation", "test")}
    p_by = {
        b: int(df.loc[df["bucket"] == b, "patient_id"].nunique()) for b in ("align", "validation", "test")
    }
    modes: collections.Counter = collections.Counter()
    for m in metas.values():
        modes.update(m["pil_modes"])
    alpha_nonopaque = sum(int(m["alpha_nonopaque"]) for m in metas.values())
    inputs = {f: _sha256(os.path.join(raw_dir, f)) for f in LABEL_FILES} | shas
    tmp = out_dir + ".partial"
    os.makedirs(tmp)
    df.to_csv(os.path.join(tmp, "rows.csv"), index=False)
    for bucket, fn in (("align", "retained.csv"), ("validation", "validation.csv")):
        sub = df[df["bucket"] == bucket]
        pd.DataFrame(
            {
                "shard": sub["shard"],
                "row": sub["row"],
                "source": sub["img_name"].str[:8],
                "img_name": sub["img_name"],
            }
        ).to_csv(os.path.join(tmp, fn), index=False)
    report = {
        "complete": not smoke,
        "smoke": smoke,
        "shards": which,
        "split_seed": S.SPLIT_SEED,
        "val_patient_fraction": S.VAL_PATIENT_FRACTION,
        "n_images": len(df),
        "n": n_by,
        "patients": p_by,
        "inputs_sha256": inputs,
        "pil_modes": dict(modes),
        "alpha_nonopaque": alpha_nonopaque,
    }
    with open(os.path.join(tmp, "report.json"), "w") as f:  # last: its presence ends the build
        json.dump(report, f, indent=1)
    os.rename(tmp, out_dir)
    return report


# ----------------------------------------------------------------------------- index
MANIFEST_FILES = ("rows.csv", "retained.csv", "validation.csv", "report.json")


def _manifest_sha(manifest_dir: str) -> dict:
    return {f: _sha256(os.path.join(manifest_dir, f)) for f in MANIFEST_FILES}


def build_index(manifest_dir: str, out: str, allow_smoke: bool = False) -> dict:
    """index npz (sorted by (shard, row)) of the retained + validation rows of a manifest. Refuses
    a manifest that is not complete (unless it is a smoke manifest and allow_smoke), whose CSV row
    counts differ from report.json, whose retained / validation rows disagree with rows.csv, or
    whose retained.csv and validation.csv share a patient. No test row is ever in the index."""
    rp = os.path.join(manifest_dir, "report.json")
    if not os.path.exists(rp):
        raise SystemExit(f"{rp} missing: the manifest build did not finish")
    report = json.load(open(rp))
    smoke = bool(report.get("smoke", False))
    if smoke and report.get("complete") is not False:
        raise SystemExit("report.json says both smoke and complete")
    if smoke and not allow_smoke:
        raise SystemExit("manifest is a smoke manifest (partial shards): pass allow_smoke / --allow-smoke")
    if not smoke and report.get("complete") is not True:
        raise SystemExit("report.json lacks complete=true: the manifest is unfinished")
    _refuse_existing(out)

    def read(f):
        return pd.read_csv(
            os.path.join(manifest_dir, f),
            dtype={"shard": np.int64, "row": np.int64, "source": str, "img_name": str},
        )

    rows = pd.read_csv(os.path.join(manifest_dir, "rows.csv"), dtype={"img_name": str, "bucket": str})
    ret, val = read("retained.csv"), read("validation.csv")
    if len(rows) != sum(report["n"].values()) or len(rows) != report["n_images"]:
        raise SystemExit(f"rows.csv has {len(rows)} rows, report.json says {report['n_images']}")
    if len(ret) != report["n"]["align"] or len(val) != report["n"]["validation"]:
        raise SystemExit(
            f"retained.csv {len(ret)} / validation.csv {len(val)} rows, report.json says "
            f"{report['n']['align']} / {report['n']['validation']}"
        )
    shared = set(ret["source"]) & set(val["source"])
    if shared:
        raise SystemExit(f"retained.csv and validation.csv share {len(shared)} patients")
    df = pd.concat([ret.assign(bucket="retained"), val.assign(bucket="validation")], ignore_index=True)
    if df.duplicated(["shard", "row"]).any() or df["img_name"].duplicated().any():
        raise SystemExit("a (shard, row) or an image appears twice in the retained + validation rows")
    if not (df["source"] == df["img_name"].str[:8]).all():
        raise SystemExit("a source is not the eight-digit prefix of its image name")
    want = rows.set_index("img_name")["bucket"]
    got = df["img_name"].map(want)
    expect = df["bucket"].map({"retained": "align", "validation": "validation"})
    if got.isna().any() or (got != expect).any():
        raise SystemExit(
            f"{int(got.isna().sum() + (got != expect).sum())} index rows disagree with rows.csv buckets"
        )
    df = df.sort_values(["shard", "row"], kind="stable").reset_index(drop=True)
    rec = {
        "shard": df["shard"].to_numpy(np.int16),
        "row": df["row"].to_numpy(np.int32),
        "bucket_code": df["bucket"].map({b: i for i, b in enumerate(BUCKETS)}).to_numpy(np.int8),
        "buckets": np.array(BUCKETS),
        "patient_id": df["source"].astype(np.int64).to_numpy(np.int32),
        "one_row_per_image": np.array(True),
        "manifest_dir": np.array(os.path.abspath(manifest_dir)),
        "manifest_sha256": np.array(json.dumps(_manifest_sha(manifest_dir), sort_keys=True)),
        "smoke": np.array(smoke),
        "split_seed": np.array(int(report["split_seed"])),
    }
    with open(out + ".partial", "wb") as f:
        np.savez(f, **rec)
    os.rename(out + ".partial", out)
    return index_description(out)


def index_description(index_npz: str) -> dict:
    z = np.load(index_npz)
    buckets = [str(b) for b in z["buckets"]]
    return {
        "index": os.path.abspath(index_npz),
        "rows": int(z["shard"].shape[0]),
        "manifest_dir": str(z["manifest_dir"]),
        "manifest_sha256": json.loads(str(z["manifest_sha256"])),
        "smoke": bool(z["smoke"]),
        "one_row_per_image": bool(z["one_row_per_image"]),
        "buckets": {b: int((z["bucket_code"] == i).sum()) for i, b in enumerate(buckets)},
        "patients": {
            b: int(np.unique(z["patient_id"][z["bucket_code"] == i]).size) for i, b in enumerate(buckets)
        },
    }


# ----------------------------------------------------------------------------- dataset
class Cxr14Dataset(ArrowCorpus):
    """Rows of an index npz, read from the converted arrow shards. which: "train" = the retained
    bucket, "val" = the validation bucket (patients disjoint from retained), "all" = both. The
    index holds no test row, so none is reachable. index_pos is the position in the npz; the
    npz order (shard, then row) is deterministic. There is no row-modulo holdout and no flip."""

    n_shards = N_ARCHIVES
    shard_name = staticmethod(shard_name)
    convert_hint = "python -m datasets.cxr14 convert ..."

    def __init__(
        self,
        arrow_dir: str,
        index_npz: str,
        img_size: int = 224,
        which: str = "train",
        hflip: bool = False,
        allow_smoke: bool = False,
    ):
        if hflip:
            raise ValueError("cxr14: horizontal flip is refused (heart side)")
        if which not in ("train", "val", "all"):
            raise ValueError(which)
        z = np.load(index_npz)
        if not bool(z["one_row_per_image"]):
            raise ValueError(f"{index_npz}: not a one-row-per-image index")
        if bool(z["smoke"]) and not allow_smoke:
            raise ValueError(f"{index_npz}: smoke index (partial shards): pass allow_smoke=True")
        buckets = [str(b) for b in z["buckets"]]
        if buckets != BUCKETS:
            raise ValueError(f"{index_npz}: buckets {buckets}, expected {BUCKETS}")
        code = z["bucket_code"]
        keep = {"train": code == 0, "val": code == 1, "all": np.ones(code.shape, bool)}[which]
        pos = np.arange(code.shape[0])
        shard, row = z["shard"].astype(np.int64), z["row"].astype(np.int64)
        self._set_rows(shard[keep], row[keep], pos[keep], arrow_dir, img_size, False, 0)
        self.which = which
        self.index_npz = index_npz
        self._check_shards()

    def describe(self) -> dict:
        return {
            **index_description(self.index_npz),
            "arrow_dir": os.path.abspath(self.arrow_dir),
            "which": self.which,
            "n": len(self),
            "n_shards": self.n_shards,
            "img_size": self.img_size,
            "hflip": self.hflip,
        }


# ----------------------------------------------------------------------------- CLI
def _parse_shards(s: str) -> list[int]:
    out: set[int] = set()
    try:
        for part in s.split(","):
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    except ValueError as e:
        raise argparse.ArgumentTypeError(s) from e
    if not out or min(out) < 0 or max(out) >= N_ARCHIVES:
        raise argparse.ArgumentTypeError(f"{s}: shards must lie in 0..{N_ARCHIVES - 1}")
    return sorted(out)


def _read_shard_columns(path: str) -> tuple[list[str], np.ndarray]:
    """img_name and patient_id of every row of a shard (memory-mapped column pass)."""
    names: list[str] = []
    pids: list[np.ndarray] = []
    with pa.memory_map(path) as src:
        rd = ipc.open_file(src)
        for b in range(rd.num_record_batches):
            batch = rd.get_batch(b)
            names.extend(batch.column("img_name").to_pylist())
            pids.append(batch.column("patient_id").to_numpy(zero_copy_only=False))
    return names, np.concatenate(pids) if pids else np.zeros(0, np.int32)


def _check_index_against_shards(arrow_dir: str, manifest_dir: str, index: str, report: dict) -> None:
    """Every index row must still be the image the manifest names: img_name, patient_id and bucket
    at (shard, row) are compared with the manifest, then every addressed shard's sha256 with the
    manifest report. Fails (SystemExit) on the first guard that fires."""
    z = np.load(index)
    idx = pd.DataFrame(
        {
            "shard": z["shard"].astype(np.int64),
            "row": z["row"].astype(np.int64),
            "bucket": np.where(z["bucket_code"] == 0, "align", "validation"),
            "pid": z["patient_id"].astype(np.int64),
        }
    )
    rows = pd.read_csv(os.path.join(manifest_dir, "rows.csv"), dtype={"img_name": str, "bucket": str})
    m = idx.merge(rows, on=["shard", "row"], how="left", suffixes=("", "_man"))
    if m["img_name"].isna().any():
        raise SystemExit(
            f"{int(m['img_name'].isna().sum())} index rows address (shard, row) pairs the manifest lacks"
        )
    m_pid = m["img_name"].str[:8].astype(np.int64)
    bad_pid = ~((m["pid"] == m_pid) & (m["patient_id"] == m_pid))
    bad_name = np.zeros(len(m), bool)
    bad_shard_pid = np.zeros(len(m), bool)
    beyond = 0
    for s, grp in m.groupby("shard"):
        names, pids = _read_shard_columns(os.path.join(arrow_dir, shard_name(int(s))))
        r = grp["row"].to_numpy()
        ok = r < len(names)
        beyond += int((~ok).sum())
        pos = grp.index.to_numpy()
        got = np.array(names, dtype=object)[np.where(ok, r, 0)]
        bad_name[pos] = ~ok | (got != grp["img_name"].to_numpy(dtype=object))
        bad_shard_pid[pos] = ~ok | (pids[np.where(ok, r, 0)] != m_pid.to_numpy()[pos])
    if beyond:
        raise SystemExit(f"{beyond} index rows address rows beyond the end of their shard")
    if bad_name.any():
        raise SystemExit(
            f"{int(bad_name.sum())} of {len(m)} index rows have a different img_name in the shard than in the manifest"
        )
    if bad_pid.any() or bad_shard_pid.any():
        raise SystemExit(
            f"{int((bad_pid | bad_shard_pid).sum())} index rows have a patient_id (index or shard column) "
            "that differs from the manifest patient"
        )
    if (m["bucket"] != m["bucket_man"]).any():
        raise SystemExit(
            f"{int((m['bucket'] != m['bucket_man']).sum())} index rows have a different bucket than the manifest"
        )
    shards = sorted(int(s) for s in m["shard"].unique())
    for s in shards:
        want = report["inputs_sha256"].get(shard_name(s))
        got = _sha256(os.path.join(arrow_dir, shard_name(s)))
        if got != want:
            raise SystemExit(
                f"shard {shard_name(s)}: sha256 {got[:12]} differs from the manifest's inputs_sha256"
            )
    print(f"index rows verified against the shards: {len(m)} rows; sha256 of {len(shards)} shards hashed")


def _check(arrow_dir: str, manifest_dir: str, index: str) -> None:
    report = json.load(open(os.path.join(manifest_dir, "report.json")))
    desc = index_description(index)
    if desc["manifest_sha256"] != _manifest_sha(manifest_dir):
        raise SystemExit("the index was built from different manifest files than those in --manifest-dir")
    counts = {
        "report_n": report["n"],
        "report_patients": report["patients"],
        "index": desc["buckets"],
        "index_patients": desc["patients"],
    }
    print(json.dumps(counts, indent=1))
    _check_index_against_shards(arrow_dir, manifest_dir, index, report)
    if report.get("smoke") or desc["smoke"]:
        print("smoke: counts not compared with the pinned constants")
    else:
        got = {
            "report n": (report["n"], PINNED_N),
            "report patients": (report["patients"], PINNED_PATIENTS),
            "index n": (
                desc["buckets"],
                {"retained": PINNED_N["align"], "validation": PINNED_N["validation"]},
            ),
            "index patients": (
                desc["patients"],
                {"retained": PINNED_PATIENTS["align"], "validation": PINNED_PATIENTS["validation"]},
            ),
        }
        diff = {k: v for k, v in got.items() if v[0] != v[1]}
        if report.get("complete") is not True or diff or report["n_images"] != N_IMAGES:
            raise SystemExit(f"counts differ from the pinned constants: {diff}")
        print("pinned counts match")
    ds = Cxr14Dataset(arrow_dir, index, which="all", allow_smoke=desc["smoke"])
    name, raw = ds.raw(0)
    print(f"row 0: {name} {len(raw)} bytes; {len(ds)} rows; CXR14_CHECK_OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--shard", type=int, required=True)
    c.add_argument("--raw-dir", required=True)
    c.add_argument("--out-dir", required=True)
    m = sub.add_parser("build-manifest")
    m.add_argument("--arrow-dir", required=True)
    m.add_argument("--raw-dir", required=True)
    m.add_argument("--out-dir", required=True)
    m.add_argument("--shards", type=_parse_shards, default=None, help="e.g. 0 or 0,2-3 (smoke only)")
    b = sub.add_parser("build-index")
    b.add_argument("--manifest-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--allow-smoke", action="store_true")
    k = sub.add_parser("check")
    k.add_argument("--arrow-dir", required=True)
    k.add_argument("--manifest-dir", required=True)
    k.add_argument("--index", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        meta = convert_archive(os.path.join(a.raw_dir, archive_name(a.shard)), a.out_dir, a.shard)
        print(
            f"shard {a.shard}: {meta['rows']} rows modes {meta['pil_modes']} sizes {meta['sizes']} alpha_nonopaque {meta['alpha_nonopaque']}"
        )
    elif a.cmd == "build-manifest":
        print(json.dumps(build_manifest(a.arrow_dir, a.raw_dir, a.out_dir, a.shards), indent=1))
    elif a.cmd == "build-index":
        print(json.dumps(build_index(a.manifest_dir, a.out, a.allow_smoke), indent=1))
    else:
        _check(a.arrow_dir, a.manifest_dir, a.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
