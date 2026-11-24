"""Write a chest X-ray probe tree: the dataset-agnostic core of the CXR export.

    python -m eval.export_cxr --dataset chestxray14 --out data/cxr/chestxray14_v<N> [--max-samples N] [--write-manifest]

Tree layout (multi-label, what eval.features.list_official_split reads):
    <root>/<split>/<name>.png      RGB uint8, greyscale replicated, <= MAX_SIDE square
    <root>/<split>/labels.npz      names (export order), y (multi-hot)
    <root>/export_table.csv        one row per image (patient ids: stays inside the tree)
    <root>/export_report.json      aggregates only

The dataset-specific readers (rows, labels, split rules) register in SPLITTERS; this module only
pads, resizes and writes. CheXpert and VinDr-CXR are patient data under data-use agreements:
for those keys nothing printed, raised or logged carries a file name, image id or per-image label
row -- messages hold counts and ordinals, the detail goes to <root>.errors.jsonl beside the tree.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import csv
import glob
import io
import json
import multiprocessing
import os
import re
import sys
import time
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable

import numpy as np
import PIL
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import export_geobench  # noqa: E402
from eval import features as F  # noqa: E402

MAX_SIDE = 448
RESTRICTED = {"chexpert", "vindr_cxr_kaggle"}
ACCEPTED_MODES = ("L", "LA", "P", "RGB", "RGBA")
# columns write_tree fills itself; patient / draw_rank / source_ref come from table_extra when given
_TABLE_BASE = ["name", "split", "patient", "draw_rank", "src_w", "src_h", "padded", "side", "source_ref"]
_FROM_EXTRA = ("patient", "draw_rank", "source_ref")
# dataset key -> splitter(args) -> (splits, table_extra, classes); Task 6-8 register theirs
SPLITTERS: dict[str, Callable] = {}


def pad_square(img: Image.Image) -> tuple[Image.Image, bool]:
    """Greyscale in: centred black padding of the short side. (square image, was_padded)."""
    w, h = img.size
    if w == h:
        return img, False
    side = max(w, h)
    out = Image.new(img.mode, (side, side), 0)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out, True


def to_tree_image(img: Image.Image) -> tuple[Image.Image, dict]:
    """The only geometry path: greyscale, pad to square, bicubic down to MAX_SIDE (never up),
    replicate to RGB uint8. Alpha (RGBA / LA / P) is discarded, not composited; the ChestX-ray14
    converter counts non-opaque sources separately. 16-bit / float sources must be windowed to 8 bit by the caller."""
    if img.mode not in ACCEPTED_MODES:
        raise ValueError(f"unsupported image mode {img.mode!r}: convert to 8-bit before to_tree_image")
    src_w, src_h = img.size
    sq, padded = pad_square(img.convert("L"))
    if sq.size[0] > MAX_SIDE:
        sq = sq.resize((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
    grey = np.asarray(sq, dtype=np.uint8)
    rgb = Image.fromarray(np.stack([grey] * 3, axis=-1), "RGB")
    return rgb, {"src_w": src_w, "src_h": src_h, "padded": padded, "side": rgb.size[0]}


def _log_error(root: str, detail: dict) -> int:
    """Append one record to <root>.errors.jsonl; return its 1-based ordinal."""
    path = f"{root}.errors.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(detail) + "\n")
    with open(path) as f:
        return sum(1 for _ in f)


def _check_names(key: str, splits: dict, restricted: bool, root: str) -> None:
    """Names become file names: unique across splits, no path separator, no leading dot."""
    seen: dict[str, str] = {}
    bad: list[tuple[str, int, str, str]] = []
    for split, rows in splits.items():
        for i, (name, _y, _load) in enumerate(rows):
            if not isinstance(name, str) or not name or "/" in name or "\\" in name or "\0" in name:
                bad.append((split, i, name if isinstance(name, str) else repr(name), "unsafe"))
            elif name.startswith("."):
                bad.append((split, i, name, "unsafe"))
            elif name in seen:
                bad.append((split, i, name, "duplicate"))
            else:
                seen[name] = split
    if not bad:
        return
    n_dup = sum(b[3] == "duplicate" for b in bad)
    n_unsafe = len(bad) - n_dup
    msg = f"{key}: {n_dup} duplicate and {n_unsafe} unsafe image names, nothing written"
    if restricted:
        ordinal = _log_error(root, {"stage": "names", "items": [list(b) for b in bad]})
        raise SystemExit(f"{msg} (detail: record {ordinal} of {root}.errors.jsonl)")
    raise SystemExit(f"{msg}: first {bad[0][2]!r} ({bad[0][3]}, {bad[0][0]} #{bad[0][1]})")


def write_tree(
    root: str,
    key: str,
    splits: dict[str, list[tuple[str, np.ndarray, Callable[[], Image.Image]]]],
    classes: list[str],
    table_extra: dict[str, dict],
    restricted: bool,
    report_extra: Callable[[dict, dict[str, list[float]]], dict] | None = None,
    before_rename: Callable[[str], None] | None = None,
) -> dict:
    """Write the tree under <root>.partial, rename to <root> on success. A failure leaves the
    .partial in place (never deleted); a retry needs a new root.

    `before_rename(partial)` (optional) runs on the complete tree while it is still <root>.partial;
    when it raises, <root> is never created (a tree that fails its manifest never gets the final name).

    `report_extra(report, means)` (optional) returns extra top-level keys for export_report.json;
    `means` holds, per split, the mean of every written 8-bit image. Without it nothing changes."""
    partial = f"{root}.partial"
    if os.path.exists(root) or os.path.exists(partial):
        raise SystemExit(
            f"{key}: {root} or {partial} already exists; pick a new --out (nothing is overwritten)"
        )
    _check_names(key, splits, restricted, root)
    extras = {k for v in table_extra.values() for k in v}
    clash = sorted((extras & set(_TABLE_BASE)) - set(_FROM_EXTRA))
    if clash:
        raise SystemExit(f"{key}: table_extra columns {clash} collide with the fixed export_table columns")
    extra_cols = sorted(extras - set(_FROM_EXTRA))
    os.makedirs(os.path.dirname(os.path.abspath(partial)), exist_ok=True)
    os.mkdir(partial)
    table: list[dict] = []
    report: dict = {"dataset": key, "classes": list(classes), "splits": {}}
    means: dict[str, list[float]] = {}
    split, i = "", -1
    try:
        for split, rows in splits.items():
            os.mkdir(os.path.join(partial, split))
            names, ys = [], []
            padded_n, src_sizes, sides = 0, Counter(), Counter()
            for i, (name, y, load) in enumerate(rows):
                y = np.asarray(y)
                if y.shape != (len(classes),):
                    raise ValueError(f"label shape {y.shape}, expected ({len(classes)},)")
                if y.dtype.kind not in "biuf" or not np.isin(y, (0, 1)).all():
                    raise ValueError("label values must be exactly 0 or 1 (map uncertain/NaN before export)")
                img, info = to_tree_image(load())
                img.save(os.path.join(partial, split, f"{name}.png"), format="PNG")
                if report_extra is not None:
                    means.setdefault(split, []).append(float(np.asarray(img).mean()))
                names.append(name)
                ys.append(y)
                padded_n += info["padded"]
                src_sizes[f"{info['src_w']}x{info['src_h']}"] += 1
                sides[str(info["side"])] += 1
                ex = table_extra.get(name, {})
                row = {
                    "name": name,
                    "split": split,
                    "patient": ex.get("patient", ""),
                    "draw_rank": ex.get("draw_rank", i),
                    "source_ref": ex.get("source_ref", ""),
                    **info,
                    **{c: ex.get(c, "") for c in extra_cols},
                }
                table.append(row)
            i = -1
            y_all = np.stack(ys).astype(np.uint8) if ys else np.zeros((0, len(classes)), np.uint8)
            np.savez(os.path.join(partial, split, "labels.npz"), names=np.array(names, dtype=str), y=y_all)
            report["splits"][split] = {
                "n": len(names),
                "counts": y_all.sum(0).tolist(),
                "padded": int(padded_n),
                "src_sizes": dict(sorted(src_sizes.items())),
                "sides": dict(sorted(sides.items())),
            }
            print(f"{key} {split}: {len(names)} images", flush=True)
        with open(os.path.join(partial, "export_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_TABLE_BASE + extra_cols)
            w.writeheader()
            w.writerows(table)
        if report_extra is not None:
            report.update(report_extra(report, means))
        with open(os.path.join(partial, "export_report.json"), "w") as f:
            json.dump(report, f, indent=1)
        if before_rename is not None:
            before_rename(partial)
        os.rename(partial, root)
    except Exception as e:
        if not restricted:
            raise
        try:
            ordinal = _log_error(root, {"stage": "write", "split": split, "ordinal": i, "error": repr(e)})
        except Exception:
            raise SystemExit(
                f"{key}: export failed in split {split!r} at item {i} ({type(e).__name__}) and the "
                f"error log {root}.errors.jsonl could not be written; {partial} left in place"
            ) from None
        raise SystemExit(
            f"{key}: export failed in split {split!r} at item {i} ({type(e).__name__}); "
            f"detail: record {ordinal} of {root}.errors.jsonl; {partial} left in place"
        ) from None
    return report


def check_cxr_manifest(key: str, manifest: dict, write: bool) -> None:
    """export_geobench.check_manifest semantics against F.cxr_manifest_path(key); the messages
    carry the key and differing top-level field names only (no counts for restricted keys)."""
    path = F.cxr_manifest_path(key)
    if os.path.exists(path):
        with open(path) as f:
            ref = json.load(f)
        if ref != manifest:
            bad = sorted(k for k in set(ref) | set(manifest) if ref.get(k) != manifest.get(k))
            what = (
                "refusing to overwrite a differing manifest" if write else "export differs from the manifest"
            )
            raise SystemExit(f"{key}: {what} {path} (fields {bad})")
        print(f"{key}: export matches the manifest {path}")
        return
    if not write:
        raise SystemExit(f"{key}: manifest {path} missing; run the export with --write-manifest")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.partial"
    if os.path.exists(tmp):
        raise SystemExit(f"{key}: {tmp} exists; remove it by hand after checking no export is running")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.rename(tmp, path)
    print(f"{key}: manifest written -> {path}")


def _tree_manifest(
    key: str, root: str, classes: list[str], restricted: bool, log_root: str | None = None
) -> dict:
    """Manifest of the tree at `root` (which may still be <out>.partial); a restricted failure goes to
    <log_root>.errors.jsonl (default: beside `root`)."""
    if not restricted:
        return export_geobench.tree_manifest(key, root, classes)
    log_root = log_root or root
    try:
        return export_geobench.tree_manifest(key, root, classes)
    except (SystemExit, Exception) as e:  # the geobench messages name image files
        ordinal = _log_error(log_root, {"stage": "tree_manifest", "error": str(e) if str(e) else repr(e)})
        raise SystemExit(
            f"{key}: tree manifest failed ({type(e).__name__}); "
            f"detail: record {ordinal} of {log_root}.errors.jsonl"
        ) from None


def _need(args, *flags: str) -> None:
    """SystemExit naming every flag that a splitter needs and the command line left out."""
    missing = [f for f in flags if getattr(args, f.replace("-", "_")) is None]
    if missing:
        raise SystemExit(f"this dataset needs {', '.join('--' + f for f in missing)}")


def chestxray14_splits(arrow_dir: str, manifest_dir: str, raw_dir: str) -> tuple[dict, dict, list[str]]:
    """(splits, table_extra, classes) for NIH ChestX-ray14 (unrestricted: names may appear in messages).

    train = draw(align images, PROBE_SIZES["train"]), val = draw(validation images, PROBE_SIZES["val"]),
    both keyed on img_name WITH the .png suffix (the pinned draw was computed on those strings);
    test = every test-bucket image sorted by name. Tree names drop the suffix. `load` decodes one
    shard row on demand: no image bytes are read while the splits are built."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from datasets import cxr14 as G

    from eval import cxr_splits as S

    rows = pd.read_csv(os.path.join(manifest_dir, "rows.csv"), dtype={"img_name": str, "bucket": str})
    retained = set(pd.read_csv(os.path.join(manifest_dir, "retained.csv"), dtype=str)["img_name"])
    validation = set(pd.read_csv(os.path.join(manifest_dir, "validation.csv"), dtype=str)["img_name"])
    meta = pd.read_csv(
        os.path.join(raw_dir, G.LABEL_FILES[0]),
        usecols=["Image Index", "Finding Labels", "Patient ID", "View Position"],
        dtype=str,
    ).set_index("Image Index")
    if meta.index.has_duplicates:
        raise SystemExit(f"chestxray14: the CSV holds {int(meta.index.duplicated().sum())} duplicate names")
    in_rows, in_csv = set(rows["img_name"]), set(meta.index)
    if in_rows != in_csv:
        raise SystemExit(
            f"chestxray14: rows.csv ({len(in_rows)} images) and the CSV ({len(in_csv)}) disagree: "
            f"{len(in_rows - in_csv)} only in rows.csv, {len(in_csv - in_rows)} only in the CSV"
        )
    by = {b: rows.loc[rows["bucket"] == b, "img_name"].tolist() for b in ("align", "validation", "test")}
    train = S.draw(by["align"], S.PROBE_SIZES["train"])
    val = S.draw(by["validation"], S.PROBE_SIZES["val"])
    test = sorted(by["test"])
    if not set(train) <= retained or not set(val) <= validation:
        raise SystemExit(
            f"chestxray14: {len(set(train) - retained)} train names are not in retained.csv, "
            f"{len(set(val) - validation)} val names are not in validation.csv"
        )
    chosen = {"train": train, "val": val, "test": test}
    odd = sum(not n.endswith(".png") for names in chosen.values() for n in names)
    if odd:  # tree names are img_name minus this suffix; any other name would be cut wrongly
        raise SystemExit(f"chestxray14: {odd} image names do not end in .png")
    patients = {s: {meta.at[n, "Patient ID"] for n in names} for s, names in chosen.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if patients[a] & patients[b]:
            raise SystemExit(
                f"chestxray14: {len(patients[a] & patients[b])} patients are in both {a} and {b}"
            )
    views = Counter(meta.at[n, "View Position"] for names in chosen.values() for n in names)
    odd = sum(v for k, v in views.items() if k not in ("PA", "AP"))
    if odd:
        raise SystemExit(f"chestxray14: {odd} selected images have a View Position other than PA/AP")

    where = {
        n: (int(s), int(r)) for n, s, r in zip(rows["img_name"], rows["shard"], rows["row"], strict=True)
    }
    readers: dict[int, ipc.RecordBatchFileReader] = {}

    def make_load(name: str):
        shard, row = where[name]

        def load():
            rd = readers.get(shard)
            if rd is None:
                rd = readers[shard] = ipc.open_file(
                    pa.memory_map(os.path.join(arrow_dir, G.shard_name(shard)))
                )
            batch = rd.get_batch(row // G.BATCH_ROWS)
            j = row % G.BATCH_ROWS
            if batch.column("img_name")[j].as_py() != name:
                raise ValueError(f"shard {shard} row {row} does not hold {name}")
            return Image.open(io.BytesIO(batch.column("image_bytes")[j].as_py()))

        return load

    splits: dict[str, list] = {}
    table_extra: dict[str, dict] = {}
    for split, names in chosen.items():
        y = S.cxr14_labels(meta.loc[names, "Finding Labels"].tolist())
        splits[split] = [(n[: -len(".png")], y[i], make_load(n)) for i, n in enumerate(names)]
        for i, n in enumerate(names):
            table_extra[n[: -len(".png")]] = {
                "patient": meta.at[n, "Patient ID"],
                "draw_rank": i,
                "source_ref": f"{where[n][0]}:{where[n][1]}",
                "view": meta.at[n, "View Position"],
            }
    return splits, table_extra, list(S.CXR14_CLASSES)


def _chestxray14(args):
    _need(args, "raw-dir", "arrow-dir", "manifest-dir")
    return chestxray14_splits(args.arrow_dir, args.manifest_dir, args.raw_dir)


SPLITTERS["chestxray14"] = _chestxray14

_PATIENT_DIR = re.compile(r"patient\d+")


class ZipMembers:
    """Index of the image members of a set of zips, by CheXpert key (restricted: messages carry
    counts and zip ordinals only, never a member name).

    The key of a member is `<split>/patientNNNNN/...` where the split is named by the components
    before the patient component (eval.cxr_count._member_split); it equals chexpert_key(<CSV Path>)
    of the matching row. Only central directories are read here; `read` opens a zip on demand."""

    def __init__(self, zip_paths: list[str]):
        from eval.cxr_count import _member_split

        self.paths = list(zip_paths)
        self._where: dict[str, list[tuple[int, str]]] = {}
        self._csv: dict[str, list[tuple[int, str]]] = {}
        self.indexed = self.unindexed = 0
        self._open: dict[int, zipfile.ZipFile] = {}
        for z, path in enumerate(self.paths):
            try:
                with zipfile.ZipFile(path) as zf:
                    infos = [i for i in zf.infolist() if not i.is_dir()]
            except Exception as e:  # noqa: BLE001 - the message must not carry the file name
                raise SystemExit(f"chexpert: zip ordinal {z} cannot be read ({type(e).__name__})") from None
            for info in infos:
                parts = [c for c in info.filename.split("/") if c]
                p = next((k for k, c in enumerate(parts) if _PATIENT_DIR.fullmatch(c)), None)
                split = _member_split(parts[:p]) if p is not None else None
                if p is not None and split is not None:
                    key = f"{split}/{'/'.join(parts[p:])}"
                    self._where.setdefault(key, []).append((z, info.filename))
                    self.indexed += 1
                else:
                    self.unindexed += 1
                    if p is None and parts and parts[-1] in ("train.csv", "valid.csv"):
                        self._csv.setdefault(parts[-1], []).append((z, info.filename))

    def csv_bytes(self, basename: str) -> bytes:
        found = self._csv.get(basename, [])
        if len(found) != 1:
            raise SystemExit(f"chexpert: {len(found)} members named {basename} across the zips, expected 1")
        z, name = found[0]
        return self._zip(z).read(name)

    def _zip(self, z: int) -> zipfile.ZipFile:
        if z not in self._open:
            try:
                self._open[z] = zipfile.ZipFile(self.paths[z])
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"chexpert: zip ordinal {z} cannot be opened ({type(e).__name__})") from None
        return self._open[z]

    def zip_ordinal(self, key: str) -> int:
        return self._where[key][0][0]

    def resolve(self, keys) -> None:
        keys = set(keys)
        missing = sum(1 for k in keys if k not in self._where)
        several = sum(1 for k in keys if len(self._where.get(k, ())) > 1)
        if missing or several:
            raise SystemExit(
                f"chexpert: {missing} unresolved keys, {several} in more than one zip "
                f"(of {len(keys)} selected); nothing read"
            )

    def _entry(self, key: str) -> tuple[int, str]:
        entries = self._where.get(key, [])
        if len(entries) != 1:
            raise KeyError(f"{key}: {len(entries)} members")
        return entries[0]

    def read(self, key: str) -> bytes:
        z, name = self._entry(key)
        return self._zip(z).read(name)

    def open(self, key: str):
        """The member as a seekable stream (header-only consumers read a few KB)."""
        z, name = self._entry(key)
        return self._zip(z).open(name)


def _mode_prepass(members: ZipMembers, keys: list[str], errors_root: str | None) -> None:
    """Header-only open of every selected member: only greyscale ("L") JPEGs are accepted."""
    tally: Counter = Counter()
    odd: list[list[str]] = []
    for key in keys:
        try:
            with members.open(key) as fh, Image.open(fh) as img:
                mode = img.mode
        except Exception:  # noqa: BLE001 - counted, never printed
            mode = "unreadable"
        tally[mode] += 1
        if mode != "L":
            odd.append([key, mode])
    if not odd:
        return
    msg = f"chexpert: image modes {dict(sorted(tally.items()))}, only L is accepted; nothing written"
    if errors_root is not None:
        ordinal = _log_error(errors_root, {"stage": "modes", "items": odd})
        msg += f" (detail: record {ordinal} of {errors_root}.errors.jsonl)"
    raise SystemExit(msg)


def chexpert_splits(raw_dir: str, errors_root: str | None = None) -> tuple[dict, dict, list[str]]:
    """(splits, table_extra, classes) for CheXpert read straight from the zips in raw_dir (restricted).

    train.csv / valid.csv are members of the zips (loose copies are never read). Selection and labels
    come from eval.cxr_splits; images are read lazily from the zips by `load`, after every selected key
    resolved to exactly one member and every selected member proved to be an "L" JPEG."""
    import pandas as pd

    from eval import cxr_splits as S

    members = ZipMembers(sorted(glob.glob(os.path.join(raw_dir, "*.zip"))))
    print(f"chexpert: {members.indexed} image members indexed, {members.unindexed} other members skipped")
    frames = {}
    for base in ("train.csv", "valid.csv"):
        try:
            frames[base] = pd.read_csv(
                io.BytesIO(members.csv_bytes(base)), dtype={"Path": str, "Frontal/Lateral": str, "AP/PA": str}
            )
            frames[base]["_key"] = [S.chexpert_key(p) for p in frames[base]["Path"]]
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - pandas errors can quote a row
            raise SystemExit(f"chexpert: {base} cannot be parsed ({type(e).__name__})") from None
    sel = S.chexpert_split(frames["train.csv"], frames["valid.csv"])
    chosen = {s: sel[sel["split"] == s].sort_values("draw_rank") for s in ("train", "val", "test")}

    labels, rows_of = {}, {}
    for split, base in (("train", "train.csv"), ("val", "train.csv"), ("test", "valid.csv")):
        f = frames[base]
        counts = f["_key"].value_counts()
        keys = list(chosen[split]["key"])
        none = sum(1 for k in keys if counts.get(k, 0) == 0)
        several = sum(1 for k in keys if counts.get(k, 0) > 1)
        if none or several:
            raise SystemExit(
                f"chexpert: {none} selected {split} keys match no CSV row, "
                f"{several} match more than one CSV row (of {len(keys)})"
            )
        rows = f.set_index("_key").loc[keys]
        rows_of[split] = rows
        labels[split] = S.chexpert_labels(rows)

    all_keys = [k for s in chosen.values() for k in s["key"]]
    members.resolve(all_keys)  # every selected key, before any image is read
    _mode_prepass(members, all_keys, errors_root)

    def make_load(key: str):
        def load():
            try:
                img = Image.open(io.BytesIO(members.read(key)))
                img.load()
            except Exception as e:  # noqa: BLE001 - the key reaches only <root>.errors.jsonl
                raise RuntimeError(f"{key}: {type(e).__name__}: {e}") from None
            return img

        return load

    splits: dict[str, list] = {}
    table_extra: dict[str, dict] = {}
    for split, frame in chosen.items():
        rows = rows_of[split]
        out = []
        for i, (key, patient, rank) in enumerate(
            zip(frame["key"], frame["patient"], frame["draw_rank"], strict=True)
        ):
            name = os.path.splitext(key)[0].replace("/", "_")
            out.append((name, labels[split][i], make_load(key)))
            ap_pa = rows["AP/PA"].iloc[i]
            table_extra[name] = {
                "patient": patient,
                "draw_rank": int(rank),
                "source_ref": str(members.zip_ordinal(key)),
                "ap_pa": "" if pd.isna(ap_pa) else str(ap_pa),
                "frontal_lateral": str(rows["Frontal/Lateral"].iloc[i]),
            }
        splits[split] = out
    return splits, table_extra, list(S.CHEXPERT_CLASSES)


def _chexpert(args):
    _need(args, "raw-dir")
    return chexpert_splits(args.raw_dir, errors_root=args.out)


SPLITTERS["chexpert"] = _chexpert


# --- VinDr-CXR, Kaggle copy (restricted: VinDr image ids are the member names and the tree file names) ---


class _VindrZip:
    """Index of the one Kaggle zip: central directory only. Messages carry counts, never a member name."""

    def __init__(self, raw_dir: str):
        paths = sorted(glob.glob(os.path.join(raw_dir, "*.zip")))
        if len(paths) != 1:
            raise SystemExit(f"vindr_cxr_kaggle: {len(paths)} zip files under the raw dir, expected 1")
        try:
            self._zf = zipfile.ZipFile(paths[0])
            infos = [i for i in self._zf.infolist() if not i.is_dir()]
        except Exception as e:  # noqa: BLE001 - the message must not carry the file name
            raise SystemExit(f"vindr_cxr_kaggle: the zip cannot be read ({type(e).__name__})") from None
        self.train: dict[str, list[tuple[int, zipfile.ZipInfo]]] = {}
        self.csv: list[zipfile.ZipInfo] = []
        self.n_test = self.n_other = 0
        for j, info in enumerate(infos):
            parts = [c for c in info.filename.split("/") if c]
            if parts and parts[-1] == "train.csv":
                self.csv.append(info)
            elif len(parts) == 2 and parts[1].endswith(".dicom") and parts[0] in ("train", "test"):
                if parts[0] == "train":
                    self.train.setdefault(parts[1][: -len(".dicom")], []).append((j, info))
                else:
                    self.n_test += 1  # unlabelled: never selected
            else:
                self.n_other += 1

    def csv_bytes(self) -> bytes:
        if len(self.csv) != 1:
            raise SystemExit(f"vindr_cxr_kaggle: {len(self.csv)} members named train.csv, expected 1")
        return self._zf.read(self.csv[0])

    def _info(self, image_id: str) -> zipfile.ZipInfo:
        found = self.train.get(image_id, [])
        if len(found) != 1:
            raise KeyError(f"{image_id}: {len(found)} members")
        return found[0][1]

    def ordinal(self, image_id: str) -> int:
        return self.train[image_id][0][0]

    def read(self, image_id: str) -> bytes:
        return self._zf.read(self._info(image_id))

    def open(self, image_id: str):
        return self._zf.open(self._info(image_id))


_VINDR_REQUIRED = ("image_id", "class_name", "rad_id")
_VINDR_CROSS_CHECKED = ("photometric", "transfer_syntax", "bits_stored", "had_voi", "n_windows")


@contextlib.contextmanager
def _counted_warnings(counter: Counter):
    """Library warnings can quote header or CSV values: record them, never show them. Only the
    warning CLASS NAME is tallied (the text goes nowhere, not even the errors log)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            counter.update(w.category.__name__ for w in caught)


class _VindrRun:
    """What one VinDr export measured: header tallies (pre-pass), per-transfer-syntax timings (load),
    library warning counts by class name, per-group maxima of the stored values."""

    def __init__(self, max_samples: int | None = None):
        self.tallies: dict[str, Counter] = {}
        self.timing: dict[str, dict] = {}
        self.warn_counts: Counter = Counter()
        self.raw_max: dict[str, list[int]] = {}
        self.raw_over: dict[str, dict] = {}  # group -> images / pixels / max_frac of clipped stored values
        self.max_samples = max_samples
        self.prefetch: _VindrPrefetch | None = None  # set when --workers > 1

    def close(self) -> None:
        """Shut the decode pool down (pending work cancelled); a no-op when there is none."""
        if self.prefetch is not None:
            self.prefetch.close()

    def add_time(self, label: str, read_s: float, decode_s: float) -> None:
        t = self.timing.setdefault(label, {"images": 0, "zip_read_s": 0.0, "decode_s": 0.0})
        t["images"] += 1
        t["zip_read_s"] += read_s
        t["decode_s"] += decode_s

    def add_raw_max(self, label: str, bits: int, value: int) -> None:
        self.raw_max.setdefault(f"{label}|{bits}", []).append(int(value))

    def add_raw_over(self, label: str, bits: int, n_over: int, n_pixels: int) -> None:
        if n_over <= 0:
            return
        g = self.raw_over.setdefault(f"{label}|{bits}", {"images": 0, "pixels": 0, "max_frac": 0.0})
        g["images"] += 1
        g["pixels"] += int(n_over)
        g["max_frac"] = max(g["max_frac"], n_over / n_pixels)

    def report_extra(self, report: dict, means: dict[str, list[float]]) -> dict:
        import pydicom  # local: only the VinDr-CXR export needs it

        positives = report["splits"].get("test", {}).get("counts", [])
        extra = {"max_samples": self.max_samples} if self.max_samples is not None else {}
        return {
            **extra,
            "warning_counts": dict(sorted(self.warn_counts.items())),
            "raw_max_quantiles": {
                k: {
                    "n": len(v),
                    "quantiles": [round(float(q), 6) for q in np.quantile(v, [0, 0.25, 0.5, 0.75, 1])],
                }
                for k, v in sorted(self.raw_max.items())
            },
            "raw_over_range": {
                k: {"images": v["images"], "pixels": v["pixels"], "max_frac": round(v["max_frac"], 8)}
                for k, v in sorted(self.raw_over.items())
            },
            "versions": {"pillow": PIL.__version__, "pydicom": pydicom.__version__, "numpy": np.__version__},
            "header_tallies": {k: dict(sorted(v.items())) for k, v in sorted(self.tallies.items())},
            "class_test_positives": dict(zip(report["classes"], positives, strict=True)),
            "transfer_syntax_timing": {
                k: {
                    "images": v["images"],
                    "zip_read_s": round(v["zip_read_s"], 6),
                    "decode_s": round(v["decode_s"], 6),
                }
                for k, v in sorted(self.timing.items())
            },
            "out_mean_quantiles": {
                s: [round(float(q), 1) for q in np.quantile(m, [0, 0.05, 0.5, 0.95, 1])] if m else []
                for s, m in means.items()
            },
        }

    def summary(self) -> str:
        n = sum(v["images"] for v in self.timing.values())
        by = ", ".join(f"{k} {v['images']}" for k, v in sorted(self.timing.items())) or "none"
        read = sum(v["zip_read_s"] for v in self.timing.values())
        dec = sum(v["decode_s"] for v in self.timing.values())
        text = (
            f"vindr_cxr_kaggle: decoded {n} images; transfer syntax {by}; "
            f"zip read {read:.6f} s, decode {dec:.6f} s"
        )
        if self.raw_over:
            n_img = sum(v["images"] for v in self.raw_over.values())
            frac = round(max(v["max_frac"] for v in self.raw_over.values()), 8)
            text += f"; {n_img} images had stored values outside the BitsStored range, clipped (max fraction {frac})"
        return text


# --- opt-in process pool for the zip read + DICOM decode (--workers N > 1; N == 1 never builds one) ---

_VINDR_WORKER: dict = {}  # per worker process: its own zip index (a shared fork would share the file offset)
_VINDR_POOL_HOOK: Callable[[str], None] | None = None  # test seam: called in a worker before the decode


def _vindr_worker_init(raw_dir: str, hook: Callable[[str], None] | None = None) -> None:
    """Pool initializer. Never raises: concurrent.futures logs an initializer exception with its
    traceback on stderr, so a failure is kept as a type name and handed back by every task."""
    try:
        _VINDR_WORKER["zip"] = _VindrZip(raw_dir)
        _VINDR_WORKER["hook"] = hook
    except BaseException as e:  # noqa: BLE001 - SystemExit from _VindrZip included
        _VINDR_WORKER["init_error"] = type(e).__name__


def _vindr_worker_decode(img: str) -> tuple:
    """(arr, info, read_s, decode_s, warning_category_names) for one image. A failure is RETURNED, not
    raised (arr None, info {"error": (exception type name, message)}): an exception crossing the
    process boundary is wrapped by concurrent.futures with a remote traceback that would carry the
    message. Warnings are captured here; only the class names leave the process."""
    from eval import cxr_dicom as DCM

    counter: Counter = Counter()
    arr, info, read_s, decode_s = None, {}, 0.0, 0.0
    try:
        if "init_error" in _VINDR_WORKER:
            raise RuntimeError(f"worker initializer failed ({_VINDR_WORKER['init_error']})")
        t0 = time.perf_counter()
        with _counted_warnings(counter):
            hook = _VINDR_WORKER.get("hook")
            if hook is not None:
                hook(img)
            data = _VINDR_WORKER["zip"].read(img)
            t1 = time.perf_counter()
            arr, info = DCM.decode_dicom_bytes(data)
            t2 = time.perf_counter()
        read_s, decode_s = t1 - t0, t2 - t1
    except Exception as e:  # noqa: BLE001 - returned as a value; the parent adds the id for the errors log
        arr, info = None, {"error": (type(e).__name__, str(e))}
    return arr, info, read_s, decode_s, sorted(counter.elements())


class _VindrPrefetch:
    """Bounded look-ahead over the ids `write_tree` will consume, in that order.

    `get(img)` returns that image's worker result and keeps at most 4 * workers results in flight or
    held, covering order[k : k + 4 * workers] for the requested position k. Ids before k that were
    never requested are dropped, so non-contiguous consumption cannot grow the window. The pool starts
    on the first `get` (start method spawn: a forked child would share the parent's zip file offset)."""

    def __init__(self, raw_dir: str, workers: int, order: list[str], hook=None):
        self.raw_dir, self.workers, self.order, self.hook = raw_dir, workers, order, hook
        self.window = 4 * workers
        self._index = {img: k for k, img in enumerate(order)}
        self._pool = None
        self._futs: dict[int, cf.Future] = {}

    def _start(self):
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_vindr_worker_init,
                initargs=(self.raw_dir, self.hook),
            )
        return self._pool

    def get(self, img: str):
        pool = self._start()
        k = self._index.get(img)
        if k is None:  # not announced: decode on its own, no look-ahead
            return pool.submit(_vindr_worker_decode, img).result()
        for j in [j for j in self._futs if j < k]:
            self._futs.pop(j).cancel()
        for j in range(k, min(k + self.window, len(self.order))):
            if j not in self._futs:
                self._futs[j] = pool.submit(_vindr_worker_decode, self.order[j])
        return self._futs.pop(k).result()

    def close(self) -> None:
        pool, self._pool = self._pool, None
        self._futs.clear()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def _vindr_header_prepass(
    members: _VindrZip, ids: list[str], errors_root: str | None, run: _VindrRun
) -> dict[str, dict]:
    """Header-only read of every selected member (no pixel is decoded): tally the allow-listed header
    facts and refuse the whole export when one member is unreadable, is not single-sample greyscale,
    has a transfer syntax outside implicit / explicit little endian and JPEG 2000 Lossless, or
    declares BitsStored outside 1..16. Returns the facts per id for `load` to cross-check."""
    import pydicom

    from eval import cxr_dicom as DCM

    facts: dict[str, dict] = {}
    tallies = {k: Counter() for k in ("photometric", "transfer_syntax", "bits_stored", "samples_per_pixel")}
    tallies.update({"had_voi": Counter(), "n_windows": Counter()})
    reasons: Counter = Counter()
    bad: list[list[str]] = []
    for img in ids:
        reason = ""
        try:
            with _counted_warnings(run.warn_counts), members.open(img) as fh:
                ds = pydicom.dcmread(fh, stop_before_pixels=True)
                f = DCM.header_facts(ds)
                spp = int(ds.get("SamplesPerPixel", 0))
        except Exception:  # noqa: BLE001 - counted, never printed
            reason = "unreadable"
        else:
            facts[img] = f
            tallies["photometric"][f["photometric"]] += 1
            tallies["transfer_syntax"][f["transfer_syntax"]] += 1
            tallies["bits_stored"][str(f["bits_stored"])] += 1
            tallies["samples_per_pixel"][str(spp)] += 1
            tallies["had_voi"][str(f["had_voi"])] += 1
            tallies["n_windows"][str(f["n_windows"])] += 1
            if f["transfer_syntax"] == "other":
                reason = "transfer_syntax"
            elif f["photometric"] == "other":
                reason = "photometric"
            elif spp != 1:
                reason = "samples_per_pixel"
            elif not 1 <= f["bits_stored"] <= 16:
                reason = "bits_stored"
        if reason:
            reasons[reason] += 1
            bad.append([img, reason])
    run.tallies = tallies
    if not bad:
        return facts
    msg = (
        f"vindr_cxr_kaggle: header pre-pass over {len(ids)} images found {len(bad)} unusable "
        f"{dict(sorted(reasons.items()))}; photometric {dict(sorted(tallies['photometric'].items()))}; "
        f"transfer_syntax {dict(sorted(tallies['transfer_syntax'].items()))}; nothing written"
    )
    if errors_root is not None:
        ordinal = _log_error(errors_root, {"stage": "header", "items": bad})
        msg += f" (detail: record {ordinal} of {errors_root}.errors.jsonl)"
    raise SystemExit(msg)


def vindr_kaggle_splits(
    raw_dir: str,
    errors_root: str | None = None,
    run: _VindrRun | None = None,
    workers: int = 1,
    limit: int | None = None,
) -> tuple[dict, dict, list[str]]:
    """(splits, table_extra, classes) for VinDr-CXR, Kaggle copy, read straight from its one zip (restricted).

    train.csv is a member of the zip. All three tree splits are drawn from the image ids of train.csv
    (eval.cxr_splits.vindr_kaggle_split); the zip's test/ members carry no labels and are never selected.
    Every selected id must resolve to exactly one train/ member and every selected member must pass the
    header pre-pass before anything is decoded; `load` then decodes the DICOM bytes lazily
    (eval.cxr_dicom) and cross-checks the decoded header facts against the pre-pass. VinDr ships no
    patient id: the image is the unit (patient column empty). `run` collects tallies and timings.

    `workers` > 1 moves the zip read + decode of `load` into a process pool (`run.prefetch`, closed by
    `run.close()`); `limit` (the caller's --max-samples) tells the look-ahead that only the first `limit`
    rows of each split will be consumed. Both default to the serial path, which is unchanged."""
    import pandas as pd

    from eval import cxr_dicom as DCM
    from eval import cxr_splits as S
    from eval.cxr_count import _CLASS_NAME_RE

    run = run if run is not None else _VindrRun()
    members = _VindrZip(raw_dir)
    print(
        f"vindr_cxr_kaggle: {len(members.train)} train and {members.n_test} test DICOM members indexed, "
        f"{members.n_other} other members skipped"
    )
    try:
        with _counted_warnings(run.warn_counts):
            rows = pd.read_csv(io.BytesIO(members.csv_bytes()), dtype=str, keep_default_na=False)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - pandas errors can quote a row
        raise SystemExit(f"vindr_cxr_kaggle: train.csv cannot be parsed ({type(e).__name__})") from None
    lacking = sum(c not in rows.columns for c in _VINDR_REQUIRED)
    if lacking:
        raise SystemExit(
            f"vindr_cxr_kaggle: train.csv lacks {lacking} of the {len(_VINDR_REQUIRED)} required columns"
        )
    rows = rows[list(_VINDR_REQUIRED)]
    sel = S.vindr_kaggle_split(list(rows["image_id"].drop_duplicates()))
    classes = S.vindr_label_set(rows, sel["test"])
    if not classes:
        raise SystemExit("vindr_cxr_kaggle: no class reaches the test-positive threshold")
    outside = sum(not _CLASS_NAME_RE.fullmatch(c) for c in classes)
    if outside:
        raise SystemExit(f"vindr_cxr_kaggle: {outside} class names fail the class-name allow-list")
    pinned = list(S.VINDR_KAGGLE_CLASSES)
    if classes != pinned:  # the label set is pinned, never re-selected from this run's test split
        differ = len(set(classes) ^ set(pinned))
        raise SystemExit(
            f"vindr_cxr_kaggle: the threshold rule gives {len(classes)} classes, the pinned "
            f"VINDR_KAGGLE_CLASSES holds {len(pinned)}, {differ} names are in one list only"
        )
    labels = {s: S.vindr_two_of_three(rows, sel[s], classes) for s in ("train", "val", "test")}

    ids = [i for s in ("train", "val", "test") for i in sel[s]]
    missing = sum(1 for i in ids if i not in members.train)
    several = sum(1 for i in ids if len(members.train.get(i, ())) > 1)
    if missing or several:
        raise SystemExit(
            f"vindr_cxr_kaggle: {missing} of {len(ids)} selected image ids have no train member, "
            f"{several} have more than one; nothing read"
        )
    facts = _vindr_header_prepass(members, ids, errors_root, run)
    if workers > 1:
        order = [img for s in ("train", "val", "test") for img in sel[s][:limit]]
        run.prefetch = _VindrPrefetch(raw_dir, workers, order, hook=_VINDR_POOL_HOOK)

    def make_load_pooled(img: str):
        # same messages as the serial `load` below: "<id>: <exception type>: <text>", id only in the errors log
        def load():
            try:
                arr, info, read_s, decode_s, warned = run.prefetch.get(img)
            except Exception as e:  # noqa: BLE001 - e.g. BrokenProcessPool: a worker died
                raise RuntimeError(f"{img}: {type(e).__name__}: {e}") from None
            run.warn_counts.update(warned)
            if arr is None:
                raise RuntimeError(f"{img}: {info['error'][0]}: {info['error'][1]}")
            for k in _VINDR_CROSS_CHECKED:
                if info[k] != facts[img][k]:
                    raise RuntimeError(f"{img}: RuntimeError: header pre-pass and decode disagree on {k}")
            run.add_time(facts[img]["transfer_syntax"], read_s, decode_s)
            run.add_raw_max(facts[img]["transfer_syntax"], facts[img]["bits_stored"], info["raw_max"])
            run.add_raw_over(
                facts[img]["transfer_syntax"], facts[img]["bits_stored"], info["raw_over"], arr.size
            )
            return Image.fromarray(arr)

        return load

    def make_load(img: str):
        if run.prefetch is not None:
            return make_load_pooled(img)

        def load():
            try:
                t0 = time.perf_counter()
                with _counted_warnings(run.warn_counts):
                    data = members.read(img)
                    t1 = time.perf_counter()
                    arr, info = DCM.decode_dicom_bytes(data)
                    t2 = time.perf_counter()
                for k in _VINDR_CROSS_CHECKED:
                    if info[k] != facts[img][k]:
                        raise RuntimeError(f"header pre-pass and decode disagree on {k}")
            except Exception as e:  # noqa: BLE001 - the id reaches only <root>.errors.jsonl
                raise RuntimeError(f"{img}: {type(e).__name__}: {e}") from None
            run.add_time(facts[img]["transfer_syntax"], t1 - t0, t2 - t1)
            run.add_raw_max(facts[img]["transfer_syntax"], facts[img]["bits_stored"], info["raw_max"])
            run.add_raw_over(
                facts[img]["transfer_syntax"], facts[img]["bits_stored"], info["raw_over"], arr.size
            )
            return Image.fromarray(arr)

        return load

    splits: dict[str, list] = {}
    table_extra: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        splits[split] = [(img, labels[split][i], make_load(img)) for i, img in enumerate(sel[split])]
        for i, img in enumerate(sel[split]):
            table_extra[img] = {
                "draw_rank": i,
                "source_ref": f"0:{members.ordinal(img)}",
                **{k: facts[img][k] for k in ("photometric", "had_voi", "n_windows", "transfer_syntax")},
                "bits_stored": facts[img]["bits_stored"],
            }
    return splits, table_extra, classes


def _vindr_kaggle(args):
    _need(args, "raw-dir")
    run = _VindrRun(max_samples=getattr(args, "max_samples", None))
    args.close = run.close  # main shuts the decode pool down on every path
    out = vindr_kaggle_splits(
        args.raw_dir,
        errors_root=args.out,
        run=run,
        workers=getattr(args, "workers", 1),
        limit=getattr(args, "max_samples", None),
    )
    args.report_extra, args.summary = run.report_extra, run.summary  # main hands these to write_tree / prints
    return out


SPLITTERS["vindr_cxr_kaggle"] = _vindr_kaggle


def _workers_arg(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(F.OFFICIAL_CXR))
    p.add_argument("--out", required=True, help="new tree root (must not exist, nor <out>.partial)")
    p.add_argument(
        "--max-samples", type=int, default=None, help="first N of each split; never writes a manifest"
    )
    p.add_argument("--write-manifest", action="store_true", help="create the manifest (refuses to overwrite)")
    p.add_argument(
        "--raw-dir",
        default=None,
        help="shipped label files (chestxray14: Data_Entry_2017_v2020.csv; chexpert: the directory of zips; "
        "vindr_cxr_kaggle: the directory holding the one Kaggle zip)",
    )
    p.add_argument("--arrow-dir", default=None, help="converted arrow shards (chestxray14)")
    p.add_argument("--manifest-dir", default=None, help="manifest with rows.csv (chestxray14)")
    p.add_argument(
        "--workers",
        type=_workers_arg,
        default=1,
        help="decode processes (vindr_cxr_kaggle only; 1 = no pool, the serial path)",
    )
    args = p.parse_args(argv)
    key = args.dataset
    if args.workers > 1 and key != "vindr_cxr_kaggle":
        raise SystemExit(f"--workers {args.workers} is honoured by vindr_cxr_kaggle only, not {key}")
    try:
        return _export(args, key)
    finally:
        close = getattr(args, "close", None)
        if close is not None:
            close()  # the decode pool (pending work cancelled) on success and on every failure path


def _export(args, key: str) -> int:
    if key not in SPLITTERS:
        raise SystemExit(f"{key}: no splitter registered in eval.export_cxr.SPLITTERS yet")
    if args.max_samples is not None and (args.write_manifest or args.max_samples < 1):
        raise SystemExit("--max-samples needs N >= 1 and cannot be combined with --write-manifest")
    if args.max_samples is None and not args.write_manifest and not os.path.exists(F.cxr_manifest_path(key)):
        raise SystemExit(
            f"{key}: manifest {F.cxr_manifest_path(key)} missing; a full export needs --write-manifest "
            "(nothing run, nothing written)"
        )
    restricted = key in RESTRICTED
    if restricted:
        try:
            splits, table_extra, classes = SPLITTERS[key](args)
        except Exception as e:  # noqa: BLE001 - SystemExit is not an Exception: count-only exits pass
            try:
                ordinal = _log_error(args.out, {"stage": "splits", "error": repr(e)})
            except Exception:  # noqa: BLE001
                raise SystemExit(
                    f"{key}: building the splits failed ({type(e).__name__}) and the error log "
                    f"{args.out}.errors.jsonl could not be written"
                ) from None
            raise SystemExit(
                f"{key}: building the splits failed ({type(e).__name__}); "
                f"detail: record {ordinal} of {args.out}.errors.jsonl"
            ) from None
    else:
        splits, table_extra, classes = SPLITTERS[key](args)
    if args.max_samples is not None:
        splits = {s: rows[: args.max_samples] for s, rows in splits.items()}
    held: dict = {}

    def verify(partial: str) -> None:
        """The complete tree, still <out>.partial, against the manifest. A tree that differs keeps
        the .partial name, so <out> never exists for a tree that failed its check."""
        try:
            held["manifest"] = _tree_manifest(key, partial, classes, restricted, log_root=args.out)
            if os.path.exists(F.cxr_manifest_path(key)) or not args.write_manifest:
                check_cxr_manifest(key, held["manifest"], write=False)
                held["checked"] = True
        except SystemExit as e:
            raise SystemExit(f"{e}; {partial} left in place, {args.out} not created") from None

    write_tree(
        args.out,
        key,
        splits,
        classes,
        table_extra,
        restricted,
        report_extra=getattr(args, "report_extra", None),
        before_rename=verify if args.max_samples is None else None,
    )
    if getattr(args, "summary", None) is not None:
        print(args.summary())
    n = {s: len(splits.get(s, [])) for s in ("train", "val", "test")}
    if args.max_samples is None:
        if not held.get("checked"):  # --write-manifest and no manifest yet: this tree defines it
            check_cxr_manifest(key, held["manifest"], write=True)
    else:
        print(f"{key}: --max-samples {args.max_samples}: subset tree, manifest not checked")
    print(f"CXR_EXPORT_OK dataset={key} train={n['train']} val={n['val']} test={n['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
