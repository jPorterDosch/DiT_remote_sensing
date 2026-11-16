"""Rule-16 counting tool for the CheXpert / VinDr-CXR archives, tested on SYNTHETIC zips only.

The fixture mirrors the documented CheXpert full-resolution layout: several zips in one directory;
the one holding train.csv / valid.csv also holds the validation images and the first batch of
training images; later zips carry training images under a DIFFERENT leading prefix than the CSV
`Path` column uses. CPU only, no pydicom.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import zipfile

import pytest

from eval import cxr_count

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
HEADER = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", *LABELS]
VALS = ["1.0", "0.0", "-1.0", ""]


def _row(split, patient, study, view, frontal, i):
    path = f"CheXpert-v1.0/{split}/patient{patient:05d}/study{study}/view{view}_{'frontal' if frontal else 'lateral'}.jpg"
    labels = [VALS[(i + j) % 4] for j in range(len(LABELS))]
    return [path, "Female", "63", "Frontal" if frontal else "Lateral", "AP" if frontal else "", *labels]


def _csv_bytes(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows(rows)
    return buf.getvalue().encode()


def make_fake_chexpert(tmp_path, n_train_patients, n_valid_patients, drop_members=0, dup_members=0):
    """Two zips. Names are chosen so the CSV zip sorts SECOND (no reliance on order or names)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    train_rows, valid_rows = [], []
    for p in range(1, n_train_patients + 1):
        train_rows.append(_row("train", p, 1, 1, True, len(train_rows)))
        train_rows.append(_row("train", p, 1, 2, False, len(train_rows)))
    for p in (1, 2):  # two extra frontal studies -> 14 rows for 6 patients
        train_rows.append(_row("train", p, 2, 1, True, len(train_rows)))
    for k in range(n_valid_patients):
        valid_rows.append(_row("valid", n_train_patients + 1 + k, 1, 1, True, k))

    half = n_train_patients // 2
    # zip A (CSV zip): train patients <= half under the CSV's own prefix, plus valid images
    # zip B: remaining train patients under a different prefix
    csv_zip, other_zip = [], []
    for r in train_rows:
        tail = r[0].split("/", 2)[2]
        pnum = int(tail.split("/")[0][len("patient") :])
        if pnum <= half:
            csv_zip.append(r[0])
        else:
            other_zip.append("CheXpert-v1.0 batch 2 (train 1)/" + tail)
    csv_zip += [r[0] for r in valid_rows]
    if drop_members:
        del other_zip[-drop_members:]
    dups = other_zip[:dup_members]
    with zipfile.ZipFile(raw / "z_batch1.zip", "w") as z:
        z.writestr("CheXpert-v1.0/train.csv", _csv_bytes(train_rows))
        z.writestr("CheXpert-v1.0/valid.csv", _csv_bytes(valid_rows))
        for m in csv_zip:
            z.writestr(m, b"x")
    with zipfile.ZipFile(raw / "a_batch2.zip", "w") as z:
        for m in other_zip:
            z.writestr(m, b"x")
    if dups:  # same train images present in a third zip -> resolve to more than one member
        with zipfile.ZipFile(raw / "m_batch3.zip", "w") as z:
            for m in dups:
                z.writestr(
                    m.replace("CheXpert-v1.0 batch 2 (train 1)", "CheXpert-v1.0 batch 3 (train 2)"), b"x"
                )
    return str(raw)


def test_count_chexpert_on_synthetic_zips(tmp_path):
    raw = make_fake_chexpert(tmp_path, n_train_patients=6, n_valid_patients=2)
    c = cxr_count.count_chexpert(raw)
    assert c["train.csv"]["rows"] == 14 and c["train.csv"]["patients"] == 6
    assert c["valid.csv"]["frontal"] == 2 and c["patients_in_both"] == 0
    assert c["unresolved_rows"] == 0 and c["multiply_resolved_rows"] == 0
    assert set(c["train.csv"]["labels"]["Edema"]) <= {"1", "0", "-1", "blank"}
    assert c["train.csv"]["columns"] == HEADER  # header reported as found, Sex/Age by name only
    assert c["train.csv"]["frontal"] == 8 and c["train.csv"]["ap_pa"] == {"AP": 8, "blank": 6}
    assert sum(c["train.csv"]["labels"]["Edema"].values()) == 14
    # zips are reported by sorted ordinal, never by name: 0 = a_batch2 (6 train images),
    # 1 = z_batch1 (2 CSVs + 8 train + 2 valid images)
    assert [z["members"] for z in c["zips"]] == [6, 12]
    assert [z["has_train_csv"] for z in c["zips"]] == [False, True]
    assert c["zips"][1]["extensions"] == {".csv": 2, ".jpg": 10}
    assert c["zips"][1]["images_by_split"] == {"train": 8, "valid": 2}
    assert c["zips"][0]["images_by_split"] == {"train": 6}
    # batch membership follows patient order in the fixture: 4-6 in zip 0, 1-3 in zip 1
    assert c["train_batches"] == [
        {"zip_ordinal": 0, "patients": 3, "min_patient_number": 4, "max_patient_number": 6},
        {"zip_ordinal": 1, "patients": 3, "min_patient_number": 1, "max_patient_number": 3},
    ]


def test_top_dirs_drop_patient_directories(tmp_path):
    c = cxr_count.count_chexpert(make_fake_chexpert(tmp_path, 6, 2))
    # the differing leading prefix is visible as a name, patient/study levels are not
    assert c["zips"][0]["top_dirs"] == {"CheXpert-v1.0 batch 2 (train 1)": 6}
    assert c["zips"][1]["top_dirs"] == {
        "CheXpert-v1.0": 2,
        "CheXpert-v1.0/train": 8,
        "CheXpert-v1.0/valid": 2,
    }
    assert c["zips"][0]["scrubbed_components"] > 0


def test_count_output_has_no_patient_token(tmp_path, capsys):
    raw = make_fake_chexpert(tmp_path, n_train_patients=6, n_valid_patients=2)
    cxr_count.main(["--chexpert", raw, "--out", str(tmp_path / "counts.json")])
    text = capsys.readouterr().out + open(tmp_path / "counts.json").read()
    assert "patient0" not in text and "study" not in text.lower()
    assert json.loads(open(tmp_path / "counts.json").read())["chexpert"]["train.csv"]["rows"] == 14


def test_count_reports_unresolved_rows(tmp_path):
    raw = make_fake_chexpert(tmp_path, n_train_patients=6, n_valid_patients=2, drop_members=1)
    assert cxr_count.count_chexpert(raw)["unresolved_rows"] == 1


def test_count_reports_multiply_resolved_rows(tmp_path):
    raw = make_fake_chexpert(tmp_path, n_train_patients=6, n_valid_patients=2, dup_members=2)
    c = cxr_count.count_chexpert(raw)
    assert c["multiply_resolved_rows"] == 2 and c["unresolved_rows"] == 0


def test_zip_error_carries_ordinal_not_name(tmp_path):
    raw = make_fake_chexpert(tmp_path, 6, 2)
    (tmp_path / "raw" / "zz_patient00099_study1.zip").write_bytes(b"not a zip")
    with pytest.raises(RuntimeError) as ei:
        cxr_count.count_chexpert(raw)
    msg = str(ei.value)
    assert "zip ordinal 2" in msg and "BadZipFile" in msg
    assert "patient" not in msg.lower() and "study" not in msg.lower()
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_scrub_fires_on_patient_token():
    # rule 6: the guard is fed the leak it exists to catch
    with pytest.raises(cxr_count.ScrubError):
        cxr_count.scrub_json({"a": {"b": "CheXpert-v1.0/train/patient00001/x"}})
    with pytest.raises(cxr_count.ScrubError):
        cxr_count.scrub_json({"Study7": 1})
    assert json.loads(cxr_count.scrub_json({"patients": 3})) == {"patients": 3}


def test_main_fails_closed_on_leak(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cxr_count, "count_chexpert", lambda raw: {"leak": "patient00001"})
    out = tmp_path / "counts.json"
    rc = cxr_count.main(["--chexpert", str(tmp_path), "--out", str(out)])
    assert rc != 0
    assert capsys.readouterr().out == "CXR_COUNT_SCRUB_FAIL\n"
    assert not out.exists() and not os.path.exists(str(out) + ".partial")


def test_main_refuses_existing_out(tmp_path):
    raw = make_fake_chexpert(tmp_path, 6, 2)
    out = tmp_path / "counts.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        cxr_count.main(["--chexpert", raw, "--out", str(out)])
    assert out.read_text() == "{}"


def test_vindr_absent_without_zip(tmp_path, capsys):
    d = tmp_path / "vindr"
    d.mkdir()
    assert cxr_count.count_vindr_kaggle(str(d))["status"] == "VINDR_KAGGLE_ABSENT"
    assert capsys.readouterr().out.strip() == "VINDR_KAGGLE_ABSENT"
    assert cxr_count.count_vindr_kaggle(str(tmp_path / "missing"))["status"] == "VINDR_KAGGLE_ABSENT"


def _fake_vindr(tmp_path):
    d = tmp_path / "vindr"
    d.mkdir()
    cols = [
        "image_id",
        "class_name",
        "class_id",
        "rad_id",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
    ]
    rows = []
    for img, rads in (("aaa", ["R1", "R2", "R3"]), ("bbb", ["R1", "R2", "R3", "R3"]), ("ccc", ["R1", "R2"])):
        for r in rads:
            rows.append(
                [img, "Cardiomegaly" if r == "R1" else "No finding", 3, r, "", "", "", "", 2000, 2500]
            )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(rows)
    with zipfile.ZipFile(d / "vin.zip", "w") as z:
        z.writestr("train.csv", buf.getvalue())
        for img in ("aaa", "bbb", "ccc"):
            z.writestr(f"train/{img}.dicom", b"d")
        for img in ("t1", "t2"):
            z.writestr(f"test/{img}.dicom", b"d")
    return str(d)


def test_vindr_csv_tallies_and_dicom_header_stats(tmp_path, monkeypatch):
    sizes = iter([(2000, 2500), (2000, 2000), (3000, 2400)])

    def fake_header(fobj):
        r, c = next(sizes)
        return {
            "transfer_syntax": "1.2.840.10008.1.2.1",
            "photometric": "MONOCHROME2",
            "bits_stored": 12,
            "has_window_center": True,
            "has_voi_lut": False,
            "rows": r,
            "columns": c,
        }

    monkeypatch.setattr(cxr_count, "_read_dicom_header", fake_header)
    c = cxr_count.count_vindr_kaggle(_fake_vindr(tmp_path))
    assert c["members"] == 6 and c["by_top_dir"] == {"<root>": 1, "test": 2, "train": 3}
    assert c["train.csv"]["rows"] == 9 and c["train.csv"]["images"] == 3
    assert c["train.csv"]["class_name"] == {"Cardiomegaly": 3, "No finding": 6}
    assert c["train.csv"]["rad_ids_per_image"] == {"2": 1, "3": 2}
    assert c["dicom_headers"]["read"] == 3 and c["dicom_headers"]["errors"] == {}
    assert c["dicom_headers"]["bits_stored"] == {"12": 3}
    assert c["dicom_headers"]["window"] == {"window_center_only": 3}
    assert c["dicom_headers"]["rows"] == {"min": 2000.0, "median": 2000.0, "max": 3000.0}
    assert c["dicom_headers"]["ratio"]["min"] == pytest.approx(0.8)
    assert c["unlabelled_dicoms"] == 2 and c["labelled_without_dicom"] == 0
    assert "aaa" not in cxr_count.scrub_json(c)  # image ids never reach the output


def _load_quota_guard():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "quota_guard", os.path.join(here, "experiments", "isaac", "cxr", "quota_guard.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


quota_guard = _load_quota_guard()

# `lfs quota -u jdosch1 /lustre/isaac24/scratch`, captured 2026-11-14 on a login node
SAMPLE = """Disk quotas for usr jdosch1 (uid 12473):
      Filesystem  kbytes   bquota  blimit  bgrace   files   iquota  ilimit  igrace
/lustre/isaac24/scratch 8637073092  10737418240 10737418240       - 5461145  10485760 10485760       -
"""
EXPECTED = (10737418240 - 8637073092) / 1e6


def test_free_gb_parses_lfs_quota():
    assert quota_guard.free_gb(SAMPLE) == pytest.approx(EXPECTED, abs=0.5)


def test_free_gb_wrapped_and_over_quota():
    wrapped = SAMPLE.replace("scratch 8637", "scratch\n 8637")
    assert quota_guard.free_gb(wrapped) == pytest.approx(EXPECTED, abs=0.5)
    over = SAMPLE.replace("8637073092", "10737418999*")
    assert quota_guard.free_gb(over) < 0
    with pytest.raises(ValueError):
        quota_guard.free_gb("lfs: no such filesystem")


# ---- fix round 1 -------------------------------------------------------------------------


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members:
            z.writestr(name, data)


def _raw(tmp_path, train_rows, valid_rows, members, extra_zips=()):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_zip(
        raw / "b.zip",
        [
            ("CheXpert-v1.0/train.csv", _csv_bytes(train_rows)),
            ("CheXpert-v1.0/valid.csv", _csv_bytes(valid_rows)),
        ]
        + [(m, b"x") for m in members],
    )
    for name, ms in extra_zips:
        _write_zip(raw / name, ms)
    return str(raw)


def test_member_split_prefers_deepest_exact_name():
    f = cxr_count._member_split
    assert f(["CheXpert-v1.0 (validate & csv)", "train"]) == "train"
    assert f(["CheXpert-v1.0 (validate & csv)", "valid"]) == "valid"
    assert f(["CheXpert-v1.0 batch 2 (train 1)"]) == "train"
    assert f(["CheXpert-v1.0 (validate & csv)"]) == "valid"  # substring only
    assert f(["CheXpert-v1.0 (train valid)"]) is None  # one component names both
    assert f(["CheXpert-v1.0"]) is None and f([]) is None


def test_identical_tails_in_train_and_valid_stay_separate(tmp_path):
    tr = [_row("train", 1, 1, 1, True, 0)]
    va = [_row("valid", 1, 1, 1, True, 1)]
    tail = "patient00001/study1/view1_frontal.jpg"
    members = [f"CheXpert-v1.0 (validate & csv)/train/{tail}", f"CheXpert-v1.0 (validate & csv)/valid/{tail}"]
    c = cxr_count.count_chexpert(_raw(tmp_path, tr, va, members))
    assert c["unresolved_rows"] == 0 and c["multiply_resolved_rows"] == 0
    assert c["orphan_members"] == {}
    assert c["zips"][0]["images_by_split"] == {"train": 1, "valid": 1}


def test_orphans_malformed_rows_and_label_column_count(tmp_path):
    tr = [_row("train", 1, 1, 1, True, 0), ["short"], ["CheXpert-v1.0/train/nopatient/x.jpg"] + [""] * 18]
    va = [_row("valid", 2, 1, 1, True, 1)]
    members = [
        "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
        "CheXpert-v1.0/valid/patient00002/study1/view1_frontal.jpg",
        "CheXpert-v1.0/train/patient00009/study1/view1_frontal.jpg",  # no CSV row
        "CheXpert-v1.0/valid/patient00008/study1/view1_frontal.jpg",  # no CSV row
        "Unknown/patient00007/study1/view1_frontal.jpg",  # no split, no CSV row
    ]
    c = cxr_count.count_chexpert(_raw(tmp_path, tr, va, members))
    assert c["train.csv"]["malformed_rows"] == 2 and c["valid.csv"]["malformed_rows"] == 0
    assert c["train.csv"]["rows"] == 3 and c["unresolved_rows"] == 0
    assert c["orphan_members"] == {"train": 1, "unclassified": 1, "valid": 1}
    assert c["train.csv"]["n_label_columns"] == 14


def test_duplicate_csv_raises_name_free_error(tmp_path):
    tr, va = [_row("train", 1, 1, 1, True, 0)], [_row("valid", 2, 1, 1, True, 1)]
    within = _raw(tmp_path, tr, va, [])
    # same zip: a second train.csv in another directory
    with zipfile.ZipFile(os.path.join(within, "b.zip"), "a") as z:
        z.writestr("Other/train.csv", _csv_bytes(tr))
    with pytest.raises(RuntimeError) as ei:
        cxr_count.count_chexpert(within)
    assert "ValueError" in str(ei.value) and "zip ordinal 0" in str(ei.value)
    # across zips
    across = tmp_path / "across"
    across.mkdir()
    _write_zip(across / "a.zip", [("x/valid.csv", _csv_bytes(va))])
    _write_zip(across / "b.zip", [("x/valid.csv", _csv_bytes(va)), ("x/train.csv", _csv_bytes(tr))])
    with pytest.raises(RuntimeError, match="more than one zip"):
        cxr_count.count_chexpert(str(across))


def test_unlisted_names_and_extensions_are_bucketed(tmp_path):
    tr, va = [_row("train", 1, 1, 1, True, 0)], [_row("valid", 2, 1, 1, True, 1)]
    extra = [
        ("CheXpert-v1.0/patient_00001/a.JPG", b"x"),
        ("12345/b.dat", b"x"),
        ("CheXpert12345/d.jpg", b"x"),
        ("12345/README", b"x"),
        ("train/c.csv", b"x"),
    ]
    raw = _raw(tmp_path, tr, va, [], extra_zips=[("zzz_extra.zip", extra)])
    c = cxr_count.count_chexpert(raw)
    z = c["zips"][1]
    assert z["top_dirs"] == {"<other>": 3, "CheXpert-v1.0/<other>": 1, "train": 1}
    assert z["extensions"] == {".csv": 1, ".jpg": 2, "<none>": 1, "<other>": 1}
    text = cxr_count.scrub_json(c)
    assert "patient_00001" not in text and "12345" not in text


def test_vindr_dirs_allow_list(tmp_path, monkeypatch):
    d = tmp_path / "v"
    d.mkdir()
    cols = "image_id,class_name,class_id,rad_id,x_min,y_min,x_max,y_max,width,height\n"
    _write_zip(
        d / "v.zip",
        [("train.csv", cols + "a,c,1,R1,,,,,1,1\n"), ("extra/train/a.dicom", b"d"), ("train/a.dicom", b"d")],
    )
    monkeypatch.setattr(
        cxr_count,
        "_read_dicom_header",
        lambda f: {
            "transfer_syntax": "u",
            "photometric": "p",
            "bits_stored": 12,
            "has_window_center": False,
            "has_voi_lut": False,
            "rows": 2,
            "columns": 1,
        },
    )
    c = cxr_count.count_vindr_kaggle(str(d))
    assert c["by_top_dir"] == {"<other>/train": 1, "<root>": 1, "train": 1}


def test_category_values_never_printed_raw(tmp_path):
    row = _row("train", 1, 1, 1, True, 0)
    row[3], row[4], row[5 + 5] = "Frontal 9999", "AP-weird-777", "cell-secret-31"  # F/L, AP/PA, Edema
    row2 = _row("train", 2, 1, 1, True, 1)
    row2[10] = "1.0"
    va = [_row("valid", 3, 1, 1, True, 2)]
    members = [f"CheXpert-v1.0/train/patient0000{p}/study1/view1_frontal.jpg" for p in (1, 2)]
    members.append("CheXpert-v1.0/valid/patient00003/study1/view1_frontal.jpg")
    c = cxr_count.count_chexpert(_raw(tmp_path, [row, row2], va, members))
    t = c["train.csv"]
    assert t["ap_pa"]["<other>"] == 1 and t["frontal_lateral"]["<other>"] == 1
    assert t["labels"]["Edema"]["<other>"] == 1 and t["labels"]["Edema"]["1"] == 1
    text = cxr_count.scrub_json(c)
    for secret in ("9999", "weird", "secret"):
        assert secret not in text


def test_main_requires_a_dataset(tmp_path):
    with pytest.raises(SystemExit):
        cxr_count.main(["--out", str(tmp_path / "o.json")])


def test_missing_columns_raise_name_free_errors(tmp_path):
    # CheXpert: header without AP/PA
    bad = _csv_bytes([]).replace(b"AP/PA", b"APPA")
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_zip(raw / "b.zip", [("CheXpert-v1.0/train.csv", bad), ("CheXpert-v1.0/valid.csv", bad)])
    with pytest.raises(RuntimeError, match="AP/PA"):
        cxr_count.count_chexpert(str(raw))
    # VinDr: no rad_id column, then a short row
    d = tmp_path / "v1"
    d.mkdir()
    _write_zip(d / "v.zip", [("train.csv", "image_id,class_name\nsecretimg,c\n")])
    with pytest.raises(RuntimeError) as ei:
        cxr_count.count_vindr_kaggle(str(d))
    assert "rad_id" in str(ei.value) and "secretimg" not in str(ei.value)
    d2 = tmp_path / "v2"
    d2.mkdir()
    _write_zip(d2 / "v.zip", [("train.csv", "image_id,class_name,rad_id\nsecretimg\n")])
    with pytest.raises(RuntimeError) as ei:
        cxr_count.count_vindr_kaggle(str(d2))
    assert "IndexError" in str(ei.value) and "secretimg" not in str(ei.value)
    assert ei.value.__suppress_context__


class _Run:
    def __init__(self, stdout):
        self.stdout = stdout


def test_quota_guard_main(monkeypatch, capsys):
    free = EXPECTED  # ~2100 GB
    monkeypatch.setattr(quota_guard.subprocess, "run", lambda *a, **k: _Run(SAMPLE))
    assert quota_guard.main(["--min-free-gb", "900"]) == 0
    assert quota_guard.main(["--min-free-gb", str(free + 100)]) == 1  # below margin fires
    assert "BELOW MARGIN" in capsys.readouterr().out
    monkeypatch.setattr(quota_guard.subprocess, "run", lambda *a, **k: _Run("garbage"))
    assert quota_guard.main(["--min-free-gb", "900"]) != 0  # unparseable output fires


# ---- fix round 2 -------------------------------------------------------------------------


def test_chexpert_dir_allow_list():
    ok = cxr_count._chexpert_dir_ok
    assert ok("CheXpert-v1.0 batch 2 (train 1)") and ok("CheXpert-v1.0") and ok("train")
    assert (
        not ok("CheXpert12345")
        and not ok("CheXpert-v1.0\nx")
        and not ok("other")
        and not ok("CheXpert" + "a" * 70)
    )


def test_category_rejects_trailing_newline():
    assert cxr_count._category("AP\n") == "<other>"
    assert cxr_count._category("AP") == "AP" and cxr_count._category("") == "blank"


def test_malformed_rows_stay_out_of_every_tally(tmp_path):
    good = [_row("train", p, 1, 1, True, p) for p in (1, 2)]
    bad = _row("train", 3, 1, 1, True, 3)
    bad[0] = "CheXpert-v1.0/train/nopatient/x.jpg"
    va = [_row("valid", 4, 1, 1, True, 4)]
    members = [f"CheXpert-v1.0/train/patient0000{p}/study1/view1_frontal.jpg" for p in (1, 2)]
    members.append("CheXpert-v1.0/valid/patient00004/study1/view1_frontal.jpg")
    t = cxr_count.count_chexpert(_raw(tmp_path, [*good, bad, ["short"]], va, members))["train.csv"]
    assert t["rows"] == 4 and t["malformed_rows"] == 2
    ok_rows = t["rows"] - t["malformed_rows"]
    assert sum(t["ap_pa"].values()) == ok_rows and sum(t["frontal_lateral"].values()) == ok_rows
    assert t["frontal"] == ok_rows and sum(t["labels"]["Edema"].values()) == ok_rows


def test_vindr_class_names_and_dicom_strings_are_bounded(tmp_path, monkeypatch):
    d = tmp_path / "v"
    d.mkdir()
    cols = "image_id,class_name,class_id,rad_id,x_min,y_min,x_max,y_max,width,height\n"
    rows = f"a,Cardiomegaly,1,R1,,,,,1,1\nb,12345,1,R1,,,,,1,1\nc,{'x' * 50},1,R1,,,,,1,1\n"
    _write_zip(d / "v.zip", [("train.csv", cols + rows)] + [(f"train/{i}.dicom", b"d") for i in "abc"])
    monkeypatch.setattr(
        cxr_count,
        "_read_dicom_header",
        lambda f: {
            "transfer_syntax": "id-777",
            "photometric": "monochrome2",
            "bits_stored": 12,
            "has_window_center": True,
            "has_voi_lut": False,
            "rows": 2,
            "columns": 1,
        },
    )
    c = cxr_count.count_vindr_kaggle(str(d))
    assert c["train.csv"]["class_name"] == {"<other>": 2, "Cardiomegaly": 1}
    assert c["dicom_headers"]["transfer_syntax"] == {"<other>": 3}
    assert c["dicom_headers"]["photometric"] == {"<other>": 3}
    assert cxr_count._bounded("1.2.840.10008.1.2.1", cxr_count._UID_RE) == "1.2.840.10008.1.2.1"
    assert cxr_count._bounded("MONOCHROME2", cxr_count._PHOTOMETRIC_RE) == "MONOCHROME2"


def test_quota_guard_timeout_returns_2(monkeypatch, capsys):
    seen = {}

    def boom(*a, **k):
        seen.update(k)
        raise quota_guard.subprocess.TimeoutExpired("lfs", 60)

    monkeypatch.setattr(quota_guard.subprocess, "run", boom)
    assert quota_guard.main(["--min-free-gb", "900"]) == 2
    assert seen["timeout"] == 60
