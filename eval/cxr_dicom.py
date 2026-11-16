"""DICOM -> 8-bit greyscale for the VinDr-CXR (Kaggle copy) probe tree.

    dicom_to_uint8(ds)       one pydicom Dataset -> (uint8 array, info)
    decode_dicom_bytes(b)    the same from the bytes of a .dicom file (module level, picklable)
    header_facts(ds)         the header facts the export pre-pass tallies (works header-only)

VinDr-CXR is restricted patient data: every message here carries counts, header numbers and
allow-list labels only (photometric in {MONOCHROME1, MONOCHROME2} else "other"; transfer syntax
in {implicit_le, explicit_le, jpeg2000_lossless} else "other"), never a file name, id or free
DICOM string. The function refuses (ValueError) rather than produce a silently wrong picture.

Order: modality LUT (rescale slope / intercept), VOI LUT or the FIRST window when present
(pydicom's own apply_modality_lut / apply_voi_lut), MONOCHROME1 -> max - x, min-max to 0..255.
"""

from __future__ import annotations

import io

import numpy as np
import pydicom
from pydicom.multival import MultiValue
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, JPEG2000Lossless

PHOTOMETRIC = ("MONOCHROME1", "MONOCHROME2")
TRANSFER_SYNTAX = {
    str(ImplicitVRLittleEndian): "implicit_le",
    str(ExplicitVRLittleEndian): "explicit_le",
    str(JPEG2000Lossless): "jpeg2000_lossless",
}
TRANSFER_SYNTAX_LABELS = tuple(TRANSFER_SYNTAX.values())
RAW_OVER_MAX_FRAC = 0.01  # above this share of out-of-range stored values the image is refused


def photometric_label(ds) -> str:
    value = str(ds.get("PhotometricInterpretation", ""))
    return value if value in PHOTOMETRIC else "other"


def transfer_syntax_label(ds) -> str:
    meta = getattr(ds, "file_meta", None)
    uid = str(meta.get("TransferSyntaxUID", "")) if meta is not None else ""
    return TRANSFER_SYNTAX.get(uid, "other")


def _count(ds, keyword: str) -> int:
    v = ds.get(keyword)
    if v is None:
        return 0
    if isinstance(v, MultiValue):
        return len(v)
    return 0 if v == "" else 1


def _n_windows(ds) -> int:
    """Windows that pydicom can apply: a centre AND a width are both needed (one alone is no window)."""
    return min(_count(ds, "WindowCenter"), _count(ds, "WindowWidth"))


def header_facts(ds) -> dict:
    """Allow-listed header facts (no pixel data needed). `had_voi`: a window (centre and width) or a VOI LUT sequence
    is present, i.e. something the VOI step actually applies."""
    n_windows = _n_windows(ds)
    return {
        "photometric": photometric_label(ds),
        "had_voi": bool(n_windows > 0 or "VOILUTSequence" in ds),
        "n_windows": int(n_windows),
        "bits_stored": int(ds.get("BitsStored", -1)),
        "transfer_syntax": transfer_syntax_label(ds),
    }


def dicom_to_uint8(ds: pydicom.Dataset) -> tuple[np.ndarray, dict]:
    """(2-D uint8 array, info). info = header_facts(ds) plus raw_min / raw_max (stored values before any clip) and raw_over
    (count of stored values outside the BitsStored range, clipped to it)."""
    facts = header_facts(ds)
    if facts["photometric"] == "other":
        raise ValueError("unsupported photometric interpretation (label other)")
    samples = int(ds.get("SamplesPerPixel", 0))
    if samples != 1:
        raise ValueError(f"samples per pixel is {samples}, expected 1")
    bits = facts["bits_stored"]
    if not 1 <= bits <= 16:
        raise ValueError(f"BitsStored {bits} is outside 1..16")
    arr = ds.pixel_array
    if arr.ndim != 2:
        raise ValueError(f"pixel array has {arr.ndim} dimensions, expected 2")
    raw_min, raw_max = int(arr.min()), int(arr.max())
    # Range guard, for ENCAPSULATED transfer syntaxes only (JPEG 2000 goes through a plugin that does not
    # mask; pydicom masks native pixel data to BitsStored before this line, so there it cannot fire). A
    # measured VinDr-CXR image (JPEG 2000, 14 bits stored) had 93 of 6.9 million values just above 16383, all
    # in a burned-in marker box: a few over-range values are clipped and counted; a wholesale bit shift would
    # put far more than RAW_OVER_MAX_FRAC of the pixels out of range, and that still raises.
    if int(ds.get("PixelRepresentation", 0)) == 0:
        lo_lim, hi_lim, kind = 0, 2**bits - 1, f"{bits}-bit"
    else:
        lo_lim, hi_lim, kind = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1, f"{bits}-bit signed"
    n_over = (
        int(np.count_nonzero((arr > hi_lim) | (arr < lo_lim)))
        if (raw_max > hi_lim or raw_min < lo_lim)
        else 0
    )
    if n_over / arr.size > RAW_OVER_MAX_FRAC:
        raise ValueError(f"{n_over} of {arr.size} stored values outside the {kind} range")
    if n_over:
        arr = np.clip(arr, lo_lim, hi_lim)
    x = apply_modality_lut(arr, ds)
    if facts["had_voi"]:
        x = apply_voi_lut(x, ds, index=0)
    x = np.asarray(x, dtype=np.float64)
    if facts["photometric"] == "MONOCHROME1":
        x = x.max() - x
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        raise ValueError("constant image after the LUTs and window")
    out = np.rint((x - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return out, {**facts, "raw_min": raw_min, "raw_max": raw_max, "raw_over": n_over}


def decode_dicom_bytes(data: bytes) -> tuple[np.ndarray, dict]:
    """The bytes of one DICOM file -> dicom_to_uint8. Pure function of its argument."""
    return dicom_to_uint8(pydicom.dcmread(io.BytesIO(data)))
