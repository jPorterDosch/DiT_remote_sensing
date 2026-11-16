"""Pinned split and label rules for the chest X-ray datasets (pure functions, no file I/O).

One rule for all three datasets: a patient pool is fixed first, then images are drawn inside it
by image hash. ``draw_rank`` is stored so that a smaller sizing is a prefix of the export.

CheXpert and VinDr-CXR are patient data under personal agreements: no message raised here carries
a path, patient id, image id or label row, only counts. ChestX-ray14 is unrestricted, but its
messages follow the same counts-only habit except where a token name helps (unknown label).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # annotations only; pandas is imported inside the functions that use it
    import pandas as pd

SPLIT_SEED = 20261114
VAL_PATIENT_FRACTION = 0.10
PROBE_SIZES = {"train": 16000, "val": 2000}
CHEXPERT_TEST_N = 202  # frontal rows of valid.csv, counted on the shipped file by job 6599933 (2026-11-14)
VINDR_KAGGLE_SIZES = {"test": 3000, "val": 1500, "train": 10500}
VINDR_EXCLUDED = ("No finding", "Other lesion")
VINDR_MIN_TEST_POSITIVES = 10
CXR14_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
CHEXPERT_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def hash_rank(role: str, key: str, seed: int = SPLIT_SEED) -> int:
    return int(hashlib.sha256(f"{seed}|{role}|{key}".encode()).hexdigest()[:16], 16)


def ranked(role: str, keys: Iterable[str], seed: int = SPLIT_SEED) -> list[str]:
    keys = list(keys)
    non_str = sum(not isinstance(k, str) for k in keys)
    if non_str:
        raise ValueError(f"{non_str} keys are not strings")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{len(keys) - len(set(keys))} duplicate keys")
    return sorted(keys, key=lambda k: (hash_rank(role, k, seed), k))


def pick_val_patients(
    patients: Iterable[str], fraction: float = VAL_PATIENT_FRACTION, seed: int = SPLIT_SEED
) -> set[str]:
    order = ranked("patient", patients, seed)
    return set(order[: int(round(fraction * len(order)))])


def draw(keys: Iterable[str], n: int, seed: int = SPLIT_SEED) -> list[str]:
    order = ranked("image", keys, seed)
    if len(order) < n:
        raise ValueError(f"cannot draw {n} keys from a pool of {len(order)}")
    return order[:n]


# --- ChestX-ray14 (unrestricted) ---------------------------------------------------------------


def cxr14_buckets(patient_of: dict[str, str], train_val: set[str], test: set[str]) -> dict[str, str]:
    bad = sum(not isinstance(v, str) or not v for v in patient_of.values())
    if bad:
        example = next(k for k, v in patient_of.items() if not isinstance(v, str) or not v)
        raise ValueError(f"{bad} patient ids are not non-empty strings, e.g. for {example}")
    both = train_val & test
    if both:
        raise ValueError(f"{len(both)} images are in both official lists")
    missing = set(patient_of) - train_val - test
    if missing:
        raise ValueError(f"{len(missing)} images are in neither official list, e.g. {sorted(missing)[0]}")
    unknown = (train_val | test) - set(patient_of)
    if unknown:
        raise ValueError(f"{len(unknown)} listed images have no patient, e.g. {sorted(unknown)[0]}")
    tv_patients = {patient_of[i] for i in train_val}
    te_patients = {patient_of[i] for i in test}
    if tv_patients & te_patients:
        raise ValueError(
            f"{len(tv_patients & te_patients)} patients have images on both sides of the official split"
        )
    val = pick_val_patients(tv_patients)
    out = {}
    for name in patient_of:
        if name in test:
            out[name] = "test"
        else:
            out[name] = "validation" if patient_of[name] in val else "align"
    return out


def cxr14_labels(finding_labels: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(CXR14_CLASSES)}
    y = np.zeros((len(finding_labels), len(CXR14_CLASSES)), dtype=np.uint8)
    for row, cell in enumerate(finding_labels):
        for token in str(cell).split("|"):
            if token == "No Finding":
                continue
            if token not in index:
                raise ValueError(f"unknown ChestX-ray14 label {token!r}")
            y[row, index[token]] = 1
    return y


# --- CheXpert (restricted: counts only in messages) --------------------------------------------

_PATIENT = re.compile(r"patient\d+")


def _chexpert_parts(path: str) -> list[str]:
    parts = str(path).replace("\\", "/").split("/")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in ("train", "valid") and _PATIENT.fullmatch(parts[i + 1]):
            return parts[i:]
    raise ValueError("a path has no train|valid/patientNNNN component")


def chexpert_key(path: str) -> str:
    return "/".join(_chexpert_parts(path))


def chexpert_patient(path: str) -> str:
    return _chexpert_parts(path)[1]


def chexpert_labels(frame: pd.DataFrame) -> np.ndarray:
    import pandas as pd  # local: eval.features imports this module in envs without pandas

    cells = frame[CHEXPERT_CLASSES]
    non_numeric = sum(
        not pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t) for t in cells.dtypes
    )
    if non_numeric:
        raise ValueError(f"{non_numeric} label columns are not numeric")
    num = cells.apply(pd.to_numeric, errors="coerce")
    bad = (num.isna() & cells.notna()) | (~num.isin([-1, 0, 1]) & num.notna())
    if bad.to_numpy().any():
        raise ValueError(f"{int(bad.to_numpy().sum())} label cells are not in {{-1, 0, 1, blank}}")
    return (num == 1).to_numpy().astype(np.uint8)


def chexpert_split(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Frontal rows only. Validation patients are picked among all patients of `train`."""
    import pandas as pd

    shared = {chexpert_patient(p) for p in train["Path"]} & {chexpert_patient(p) for p in valid["Path"]}
    if shared:
        raise ValueError(f"{len(shared)} patients appear in both CSVs")

    def frontal(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame[frame["Frontal/Lateral"] == "Frontal"]
        return pd.DataFrame(
            {"key": [chexpert_key(p) for p in f["Path"]], "patient": [chexpert_patient(p) for p in f["Path"]]}
        )

    tr, te = frontal(train), frontal(valid)
    val_pool = pick_val_patients({chexpert_patient(p) for p in train["Path"]})
    patient_of = dict(zip(tr.key, tr.patient, strict=True))
    in_val = tr.patient.isin(val_pool)
    val_keys = draw(tr.key[in_val], PROBE_SIZES["val"])
    train_keys = draw(tr.key[~in_val], PROBE_SIZES["train"])
    test_keys = ranked("image", te.key)
    test_patient = dict(zip(te.key, te.patient, strict=True))

    rows = []
    for split, keys, lookup in (
        ("train", train_keys, patient_of),
        ("val", val_keys, patient_of),
        ("test", test_keys, test_patient),
    ):
        rows += [(k, lookup[k], split, r) for r, k in enumerate(keys)]
    return pd.DataFrame(rows, columns=["key", "patient", "split", "draw_rank"])


# --- VinDr-CXR, Kaggle copy (restricted: counts only in messages) ------------------------------


def vindr_kaggle_split(image_ids: Iterable[str]) -> dict[str, list[str]]:
    ids = list(image_ids)
    total = sum(VINDR_KAGGLE_SIZES.values())
    if len(ids) != total or len(set(ids)) != total:
        raise ValueError(
            f"expected exactly {total} distinct image ids, got {len(ids)} ({len(set(ids))} distinct)"
        )
    order = ranked("image", ids)
    n_test, n_val = VINDR_KAGGLE_SIZES["test"], VINDR_KAGGLE_SIZES["val"]
    return {"test": order[:n_test], "val": order[n_test : n_test + n_val], "train": order[n_test + n_val :]}


def vindr_two_of_three(rows: pd.DataFrame, image_ids: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    if len(image_ids) == 0:
        return np.zeros((0, len(classes)), dtype=np.uint8)
    sub = rows[rows["image_id"].isin(set(image_ids))]
    n_rad = sub.groupby("image_id")["rad_id"].nunique().reindex(list(image_ids)).fillna(0)
    wrong = int((n_rad != 3).sum())
    if wrong:
        raise ValueError(f"{wrong} images do not have exactly 3 radiologists")
    votes = sub.groupby(["image_id", "class_name"])["rad_id"].nunique().unstack(fill_value=0)
    votes = votes.reindex(index=list(image_ids), columns=list(classes), fill_value=0)
    return (votes.to_numpy() >= 2).astype(np.uint8)


def vindr_label_set(rows: pd.DataFrame, test_ids: Sequence[str]) -> list[str]:
    classes = sorted(set(rows[rows["image_id"].isin(set(test_ids))]["class_name"]) - set(VINDR_EXCLUDED))
    positives = vindr_two_of_three(rows, test_ids, classes).sum(axis=0)
    return [c for c, p in zip(classes, positives, strict=True) if p >= VINDR_MIN_TEST_POSITIVES]
