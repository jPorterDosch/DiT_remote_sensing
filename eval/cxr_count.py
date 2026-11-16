"""Rule-16 counts of the SHIPPED CheXpert and VinDr-CXR (Kaggle) artifacts.

Aggregates only: member counts, extension tallies, CSV schema, label tallies. Zip central
directories are read without decompression; only the CSVs (and, for VinDr, four technical DICOM
header fields) are decompressed. Both datasets are restricted patient data, so:

* any path component matching patient<digits> / study<digits> is COUNTED, never kept;
* exceptions carry zip / member ordinals and the exception class, never a name or a CSV row;
* `main` serializes first and refuses to print or write if the text still contains such a token.

    python -m eval.cxr_count --chexpert <dir of zips> [--vindr <dir>] --out counts.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import statistics
import sys
import zipfile
from collections import Counter

_PATIENT_RE = re.compile(r"(?i)patient(\d+)")
_SCRUB_RE = re.compile(r"(?i)(patient\d|study\d)")  # component filter and output guard
_EXT_ALLOWED = {".jpg", ".jpeg", ".png", ".csv", ".dicom", ".dcm", ".txt", ".md", ".xlsx", ".zip", ".json"}
_CATEGORY_RE = re.compile(r"[A-Za-z]{0,8}")  # AP/PA and Frontal/Lateral values reported verbatim
_CHEXPERT_NAME_RE = re.compile(r"(?i)CheXpert[-.\w ()&]{0,60}")
_DIGIT_RUN_RE = re.compile(r"\d{4,}")
_CLASS_NAME_RE = re.compile(r"[A-Za-z][A-Za-z /-]{0,39}")
_UID_RE = re.compile(r"[0-9.]{1,64}")
_PHOTOMETRIC_RE = re.compile(r"[A-Z0-9_ ]{1,16}")


def _chexpert_dir_ok(c: str) -> bool:
    if c in {"train", "valid", "test"}:
        return True
    return bool(_CHEXPERT_NAME_RE.fullmatch(c)) and not _DIGIT_RUN_RE.search(c)


def _vindr_dir_ok(c: str) -> bool:
    return c in {"train", "test"}


def _bounded(value: str, pattern: re.Pattern) -> str:
    """Technical string reported verbatim only if it fully matches `pattern`, else `<other>`."""
    return value if pattern.fullmatch(value) else "<other>"


_LABEL_VALUES = {"1": "1", "1.0": "1", "0": "0", "0.0": "0", "-1": "-1", "-1.0": "-1", "": "blank"}


class ScrubError(RuntimeError):
    """The serialized result still contains a patient or study token."""


def scrub_json(obj) -> str:
    """Serialize `obj`; raise ScrubError (no content in the message) if a token survived."""
    text = json.dumps(obj, indent=2, sort_keys=True)
    if _SCRUB_RE.search(text):
        raise ScrubError("patient or study token in serialized result")
    return text


def _ext(name: str) -> str:
    e = os.path.splitext(name)[1].lower()
    if not e:
        return "<none>"
    return e if e in _EXT_ALLOWED else "<other>"


def _category(value: str) -> str:
    """Categorical CSV value, verbatim only when short and alphabetic (fail closed otherwise)."""
    return (value or "blank") if _CATEGORY_RE.fullmatch(value) else "<other>"


def _open_zips(raw_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(raw_dir, "*.zip")))


def _fail(what: str, zip_ord: int, member_ord: int | None, exc: BaseException):
    where = f"zip ordinal {zip_ord}" + (f", member ordinal {member_ord}" if member_ord is not None else "")
    raise RuntimeError(f"{what} failed at {where}: {type(exc).__name__}") from None


def _top_dirs(parts: list[str], scrubbed: Counter, allowed) -> str:
    """First two directory levels of a member. Patient/study components are dropped (counted);
    a name is reported verbatim only if `allowed(name)`, anything else becomes `<other>`."""
    kept = []
    for c in parts[:-1][:2]:
        if _SCRUB_RE.search(c):
            scrubbed["n"] += 1
        else:
            kept.append(c if allowed(c) else "<other>")
    return "/".join(kept) if kept else "<root>"


def _member_split(prefix: list[str]) -> str | None:
    """Split named by the components before the patient component: the deepest exact name
    (`train` / `valid`) wins, else the deepest substring match; a component naming both -> None."""
    for c in reversed(prefix):
        if c.lower() in ("train", "valid"):
            return c.lower()
    for c in reversed(prefix):
        low = c.lower()
        t, v = "train" in low, "valid" in low
        if t and v:
            return None
        if t or v:
            return "train" if t else "valid"
    return None


def _read_csv_rows(zf: zipfile.ZipFile, info: zipfile.ZipInfo):
    with zf.open(info) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as fh:
        yield from csv.reader(fh)


def _scan_chexpert_zip(path: str, zip_ord: int):
    """-> (stats dict, {(split, tail): n}, {csv basename: rows}) for one zip."""
    exts, tops, splits, scrubbed = Counter(), Counter(), Counter(), Counter()
    index: Counter = Counter()
    csvs: dict[str, list] = {}
    members = dirs = 0
    j = -1
    try:
        with zipfile.ZipFile(path) as zf:
            for j, info in enumerate(zf.infolist()):
                if info.is_dir():
                    dirs += 1
                    continue
                members += 1
                parts = [p for p in info.filename.split("/") if p]
                exts[_ext(parts[-1])] += 1
                tops[_top_dirs(parts, scrubbed, _chexpert_dir_ok)] += 1
                pidx = next((k for k, c in enumerate(parts) if _PATIENT_RE.match(c)), None)
                if pidx is not None:
                    split = _member_split(parts[:pidx])
                    splits[split or "unclassified"] += 1
                    index[(split, "/".join(parts[pidx:]))] += 1
                elif parts[-1] in ("train.csv", "valid.csv"):
                    if parts[-1] in csvs:
                        raise ValueError("duplicate csv")
                    csvs[parts[-1]] = list(_read_csv_rows(zf, info))
    except Exception as e:  # noqa: BLE001 - re-raised without names
        _fail("reading zip", zip_ord, j if j >= 0 else None, e)
    stats = {
        "members": members,
        "directory_entries": dirs,
        "extensions": dict(sorted(exts.items())),
        "top_dirs": dict(sorted(tops.items())),
        "scrubbed_components": scrubbed["n"],
        "images_by_split": dict(sorted(splits.items())),
        "has_train_csv": "train.csv" in csvs,
        "has_valid_csv": "valid.csv" in csvs,
    }
    return stats, index, csvs


def _col(header: list[str], name: str) -> int:
    if name not in header:
        raise RuntimeError(f"required column {name!r} missing from the CSV header")  # public schema
    return header.index(name)


def _tally_csv(rows: list[list[str]], split: str, zip_of_key: dict):
    """Per-CSV statistics plus per-row resolution against the member index."""
    header = rows[0]
    c_path, c_fl, c_ap = (_col(header, k) for k in ("Path", "Frontal/Lateral", "AP/PA"))
    label_cols = list(range(c_ap + 1, len(header)))
    labels = {header[k]: Counter() for k in label_cols}
    patients, ap_pa, fl, referenced = set(), Counter(), Counter(), set()
    frontal = unresolved = multiple = malformed = 0
    batch_patients: dict[int, set] = {}
    n = 0
    for row in rows[1:]:
        n += 1
        if len(row) < len(header):
            malformed += 1
            continue
        parts = row[c_path].split("/")
        pidx = next((k for k, c in enumerate(parts) if _PATIENT_RE.match(c)), None)
        if pidx is None:
            malformed += 1
            continue
        if row[c_fl] == "Frontal":
            frontal += 1
        ap_pa[_category(row[c_ap])] += 1
        fl[_category(row[c_fl])] += 1
        for k in label_cols:
            labels[header[k]][_LABEL_VALUES.get(row[k].strip(), "<other>")] += 1
        pnum = int(_PATIENT_RE.match(parts[pidx]).group(1))
        patients.add(pnum)
        key = (split, "/".join(parts[pidx:]))
        referenced.add(key)
        zips = zip_of_key.get(key, [])
        if not zips:
            unresolved += 1
        elif len(zips) > 1:
            multiple += 1
        else:
            batch_patients.setdefault(zips[0], set()).add(pnum)
    stats = {
        "columns": header,
        "rows": n,
        "patients": len(patients),
        "frontal": frontal,
        "ap_pa": dict(sorted(ap_pa.items())),
        "frontal_lateral": dict(sorted(fl.items())),
        "n_label_columns": len(label_cols),
        "labels": {k: dict(sorted(v.items())) for k, v in labels.items()},
        "malformed_rows": malformed,
    }
    return stats, patients, unresolved, multiple, batch_patients, referenced


def count_chexpert(raw_dir: str) -> dict:
    paths = _open_zips(raw_dir)
    zip_stats, csvs, zip_of_key = [], {}, {}
    for z, path in enumerate(paths):
        stats, index, found = _scan_chexpert_zip(path, z)
        zip_stats.append(stats)
        for key in index:
            zip_of_key.setdefault(key, []).extend([z] * index[key])
        for name, rows in found.items():
            if name in csvs:
                raise RuntimeError(f"{name} found in more than one zip (second at zip ordinal {z})")
            csvs[name] = rows
    out: dict = {"zips": zip_stats}
    pats: dict[str, set] = {}
    unresolved = multiple = 0
    referenced: set = set()
    batches: dict[int, set] = {}
    for name in ("train.csv", "valid.csv"):
        if name not in csvs:
            raise RuntimeError(f"{name} not found in any zip")
        try:
            stats, p, u, m, bp, ref = _tally_csv(csvs[name], name.split(".")[0], zip_of_key)
        except RuntimeError:
            raise  # our own name-free messages
        except Exception as e:  # noqa: BLE001 - never echo a CSV row
            raise RuntimeError(f"tallying {name} failed: {type(e).__name__}") from None
        referenced |= ref
        out[name] = stats
        pats[name] = p
        unresolved += u
        multiple += m
        if name == "train.csv":
            batches = bp
    orphans: Counter = Counter()
    for (split, _tail), zs in zip_of_key.items():
        if (split, _tail) not in referenced:
            orphans[split or "unclassified"] += len(zs)
    out["orphan_members"] = dict(sorted(orphans.items()))
    out["patients_in_both"] = len(pats["train.csv"] & pats["valid.csv"])
    out["unresolved_rows"] = unresolved
    out["multiply_resolved_rows"] = multiple
    out["train_batches"] = [
        {
            "zip_ordinal": z,
            "patients": len(s),
            "min_patient_number": min(s),
            "max_patient_number": max(s),
        }
        for z, s in sorted(batches.items())
    ]
    return out


def _read_dicom_header(fobj) -> dict:
    """Exactly four technical tags (plus Rows/Columns). pydicom is imported lazily."""
    import pydicom

    ds = pydicom.dcmread(
        fobj,
        stop_before_pixels=True,
        specific_tags=[
            "PhotometricInterpretation",
            "BitsStored",
            "WindowCenter",
            "VOILUTSequence",
            "Rows",
            "Columns",
        ],
    )
    return {
        "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
        "photometric": str(ds.get("PhotometricInterpretation", "missing")),
        "bits_stored": int(ds.get("BitsStored", -1)),
        "has_window_center": "WindowCenter" in ds,
        "has_voi_lut": "VOILUTSequence" in ds,
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
    }


def _min_med_max(xs: list[float]) -> dict:
    if not xs:
        return {}
    return {"min": float(min(xs)), "median": float(statistics.median(xs)), "max": float(max(xs))}


def count_vindr_kaggle(raw_dir: str) -> dict:
    paths = _open_zips(raw_dir) if os.path.isdir(raw_dir) else []
    if not paths:
        print("VINDR_KAGGLE_ABSENT")
        return {"status": "VINDR_KAGGLE_ABSENT"}
    exts, tops, scrubbed = Counter(), Counter(), Counter()
    where: dict[str, tuple[int, int]] = {}  # image_id -> (zip ordinal, member ordinal), train dir
    unlabelled = 0
    csv_rows = None
    for z, path in enumerate(paths):
        j = -1
        try:
            with zipfile.ZipFile(path) as zf:
                infos = zf.infolist()
                for j, info in enumerate(infos):
                    if info.is_dir():
                        continue
                    parts = [p for p in info.filename.split("/") if p]
                    exts[_ext(parts[-1])] += 1
                    tops[_top_dirs(parts, scrubbed, _vindr_dir_ok)] += 1
                    if parts[-1] == "train.csv":
                        if csv_rows is not None:
                            raise ValueError("more than one train.csv")
                        csv_rows = list(_read_csv_rows(zf, info))
                    elif parts[-1].endswith(".dicom") and len(parts) == 2:
                        if parts[0] == "train":
                            where[parts[1][: -len(".dicom")]] = (z, j)
                        else:
                            unlabelled += 1
        except Exception as e:  # noqa: BLE001
            _fail("reading zip", z, j if j >= 0 else None, e)
    if csv_rows is None:
        raise RuntimeError("train.csv not found in any zip")
    rads: dict[str, set] = {}
    classes = Counter()
    try:
        header = csv_rows[0]
        c_img, c_cls, c_rad = (_col(header, k) for k in ("image_id", "class_name", "rad_id"))
        for row in csv_rows[1:]:
            classes[_bounded(row[c_cls], _CLASS_NAME_RE)] += 1
            rads.setdefault(row[c_img], set()).add(row[c_rad])
    except RuntimeError:
        raise  # missing column: name-free apart from the public column name
    except Exception as e:  # noqa: BLE001 - never echo a CSV row
        raise RuntimeError(f"parsing train.csv failed: {type(e).__name__}") from None
    hist = Counter(len(v) for v in rads.values())

    hdr = {
        "transfer_syntax": Counter(),
        "photometric": Counter(),
        "bits_stored": Counter(),
        "window": Counter(),
    }
    rows_, cols_, ratio = [], [], []
    errors: Counter = Counter()
    missing = n_read = 0
    for z, path in enumerate(paths):
        wanted = {(zi, j): img for img, (zi, j) in where.items() if zi == z and img in rads}
        if not wanted:
            continue
        j = -1
        try:
            with zipfile.ZipFile(path) as zf:
                infos = zf.infolist()
                for j in sorted(k[1] for k in wanted):
                    try:
                        with zf.open(infos[j]) as fh:
                            h = _read_dicom_header(fh)
                    except Exception as e:  # noqa: BLE001 - class name only, never the member
                        errors[type(e).__name__] += 1
                        continue
                    n_read += 1
                    hdr["transfer_syntax"][_bounded(h["transfer_syntax"], _UID_RE)] += 1
                    hdr["photometric"][_bounded(h["photometric"], _PHOTOMETRIC_RE)] += 1
                    hdr["bits_stored"][str(h["bits_stored"])] += 1
                    w, v = h["has_window_center"], h["has_voi_lut"]
                    hdr["window"][
                        "both"
                        if w and v
                        else "window_center_only"
                        if w
                        else "voi_lut_only"
                        if v
                        else "neither"
                    ] += 1
                    rows_.append(h["rows"])
                    cols_.append(h["columns"])
                    ratio.append(round(h["rows"] / h["columns"], 6))
        except Exception as e:  # noqa: BLE001 - ordinals and class only
            _fail("reading DICOM headers", z, j if j >= 0 else None, e)
    missing = sum(1 for img in rads if img not in where)
    return {
        "status": "present",
        "zips": len(paths),
        "members": sum(tops.values()),
        "by_top_dir": dict(sorted(tops.items())),
        "by_extension": dict(sorted(exts.items())),
        "scrubbed_components": scrubbed["n"],
        "train.csv": {
            "columns": header,
            "rows": len(csv_rows) - 1,
            "images": len(rads),
            "class_name": dict(sorted(classes.items())),
            "rad_ids_per_image": {str(k): v for k, v in sorted(hist.items())},
        },
        "unlabelled_dicoms": unlabelled,
        "labelled_without_dicom": missing,
        "dicom_headers": {
            "read": n_read,
            "errors": dict(sorted(errors.items())),
            **{k: dict(sorted(v.items())) for k, v in hdr.items()},
            "rows": _min_med_max(rows_),
            "columns": _min_med_max(cols_),
            "ratio": _min_med_max(ratio),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chexpert", help="directory holding the CheXpert zips")
    ap.add_argument("--vindr", help="directory holding the VinDr-CXR Kaggle zip")
    ap.add_argument("--out", required=True, help="JSON output; must not exist")
    a = ap.parse_args(argv)
    if not (a.chexpert or a.vindr):
        ap.error("give at least one of --chexpert / --vindr")
    part = a.out + ".partial"
    if os.path.exists(a.out) or os.path.exists(part):
        ap.error("--out or --out.partial already exists; refusing to overwrite")
    result: dict = {}
    if a.chexpert:
        result["chexpert"] = count_chexpert(a.chexpert)
    if a.vindr:
        result["vindr_kaggle"] = count_vindr_kaggle(a.vindr)
    try:
        text = scrub_json(result)
    except ScrubError:
        print("CXR_COUNT_SCRUB_FAIL")
        return 3
    with open(part, "x") as fh:
        fh.write(text + "\n")
    os.rename(part, a.out)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
