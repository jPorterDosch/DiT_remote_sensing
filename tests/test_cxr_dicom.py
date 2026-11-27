"""eval/cxr_dicom.py on synthetic DICOM only (restricted VinDr-CXR data is never touched).

Every dataset goes through bytes: pydicom writes it to a BytesIO and the code decodes those bytes,
for Implicit VR Little Endian, Explicit VR Little Endian and (raw J2K codestream) JPEG 2000 Lossless.
Every refusal is fed the input it exists to catch (rule 6) and its message is checked for the
fixture string it must not repeat."""

from __future__ import annotations

import io
import pickle

import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, JPEG2000Lossless, generate_uid

from eval import cxr_dicom as D

EVR, IVR, J2K = ExplicitVRLittleEndian, ImplicitVRLittleEndian, JPEG2000Lossless
SECRET = "XSECRETX"


def scan(seed=0, h=16, w=24, bits=12):
    """A bright square on a dark noisy ground, unsigned `bits`-bit values."""
    rng = np.random.default_rng(seed)
    top = 2 ** (bits - 1)
    a = (200 + rng.integers(0, 100, (h, w))).astype(np.uint16)
    a[4:12, 6:18] = (top + 900 + rng.integers(0, 100, (8, 12))).astype(np.uint16)
    return a


def j2k_codestream(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16)).save(buf, format="JPEG2000", irreversible=False, no_jp2=True)
    return buf.getvalue()


def make_dicom(
    arr,
    photometric="MONOCHROME2",
    bits=12,
    rep=0,
    ts=EVR,
    samples=1,
    window=None,
    rescale=None,
    frames=1,
    voi_lut_sequence=False,
) -> bytes:
    """The bytes of a DICOM file; `arr` holds the stored values (h, w) or (h, w, 3)."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ts
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = arr.shape[:2]
    ds.SamplesPerPixel = samples
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, bits, bits - 1, rep
    if samples > 1:
        ds.PlanarConfiguration = 0
    if frames > 1:
        ds.NumberOfFrames = str(frames)
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = list(window[0]), list(window[1])
    if rescale is not None:
        ds.RescaleSlope, ds.RescaleIntercept = rescale
    if voi_lut_sequence:
        item = Dataset()
        item.add_new(0x00283002, "US", [4, 0, 16])  # LUTDescriptor
        item.add_new(0x00283006, "US", [0, 100, 200, 300])  # LUTData
        ds.VOILUTSequence = [item]
    if ts == J2K:
        ds.add_new(0x7FE00010, "OB", encapsulate([j2k_codestream(arr)]))
        ds["PixelData"].is_undefined_length = True
    else:
        ds.PixelData = (arr.astype("<i2" if rep else "<u2").tobytes()) * frames
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def decode(**kw):
    return D.decode_dicom_bytes(make_dicom(**kw))


TS = [pytest.param(IVR, id="implicit_le"), pytest.param(EVR, id="explicit_le")]


@pytest.mark.parametrize("ts", TS)
def test_dicom_monochrome2_bright_square_stays_bright(ts):
    out, info = decode(arr=scan(), ts=ts)
    assert out.dtype == np.uint8 and out.shape == (16, 24)
    assert out[4:12, 6:18].mean() > out[:4].mean() + 100
    assert out.min() == 0 and out.max() == 255
    assert info["photometric"] == "MONOCHROME2" and info["bits_stored"] == 12 and info["had_voi"] is False


@pytest.mark.parametrize("ts", TS)
def test_dicom_monochrome1_is_inverted_to_match(ts):
    a = scan(seed=3)
    two, _ = decode(arr=a, ts=ts)
    one, info = decode(arr=(4095 - a).astype(np.uint16), photometric="MONOCHROME1", ts=ts)
    assert info["photometric"] == "MONOCHROME1"
    assert np.abs(one.astype(int) - two.astype(int)).max() <= 1
    assert one[4:12, 6:18].mean() > one[:4].mean() + 100  # inversion applied: still a bright square


def test_dicom_window_is_applied_and_first_of_several_is_used():
    a = scan(seed=1)
    a[0, 0], a[1, 1] = 1500, 700
    out, info = decode(arr=a, window=([1000, 3000], [200, 200]))
    assert info["n_windows"] == 2 and info["had_voi"] is True
    assert out[0, 0] == 255 and out[1, 1] == 0  # 1500 above 1000+100, 700 below 1000-100
    out2, _ = decode(arr=a, window=([3000, 1000], [200, 200]))  # first window moves: the result differs
    assert out2[0, 0] == 0 and not np.array_equal(out, out2)


def test_dicom_without_window_uses_full_range():
    a = scan()
    a[0, 0], a[0, 1] = 5, 4000
    out, info = decode(arr=a)
    assert info["had_voi"] is False and info["n_windows"] == 0
    assert out[0, 0] == 0 and out[0, 1] == 255
    mid = int(a[10, 10])
    assert abs(int(out[10, 10]) - round((mid - 5) / (4000 - 5) * 255)) <= 1  # linear in the stored value


def test_dicom_modality_lut_is_applied_before_the_window():
    a = np.full((16, 24), 500, np.uint16)
    a[2:8, 2:8], a[10, 10] = 1500, 1050
    out, info = decode(arr=a, window=([0], [200]), rescale=(1, -1000))
    assert out[0, 0] == 0 and out[2, 2] == 255
    assert 128 < out[10, 10] < 255  # raw 1050 = 50 after the intercept; unrescaled it would saturate to 255
    assert info["n_windows"] == 1


def test_voi_lut_sequence_counts_as_voi_in_the_header_facts():
    ds = pydicom.dcmread(io.BytesIO(make_dicom(arr=scan(), voi_lut_sequence=True)), stop_before_pixels=True)
    facts = D.header_facts(ds)
    assert facts["had_voi"] is True and facts["n_windows"] == 0


def test_dicom_constant_image_raises_without_the_value():
    with pytest.raises(ValueError, match="constant image") as ei:
        decode(arr=np.full((16, 24), 777, np.uint16))
    assert "777" not in str(ei.value)
    with pytest.raises(ValueError, match="constant image"):  # a window that clips everything
        decode(arr=np.full((16, 24), 100, np.uint16), window=([3000], [100]))


def test_dicom_unsupported_photometric_is_refused_by_label_only():
    with pytest.raises(ValueError, match=r"unsupported photometric interpretation \(label other\)") as ei:
        decode(arr=scan(), photometric=SECRET)
    assert SECRET not in str(ei.value)
    out, _ = decode(arr=scan(), photometric="MONOCHROME2")  # control: the guard passes the good value
    assert out.shape == (16, 24)


def test_dicom_samples_per_pixel_is_refused():
    rgb = np.stack([scan()] * 3, axis=-1)
    with pytest.raises(ValueError, match="samples per pixel is 3, expected 1"):
        decode(arr=rgb, samples=3)


def test_dicom_pixel_array_that_is_not_2d_is_refused():
    with pytest.raises(ValueError, match="pixel array has 3 dimensions, expected 2"):
        decode(arr=scan(), frames=2)


class _DecoderReturned(pydicom.Dataset):
    """A dataset whose decoder hands back `forced`: pydicom 3 masks native pixel data to BitsStored, so
    the "decoder returned bits the header does not declare" guard (the JPEG 2000 path goes through a
    plugin) is fed through the decoded array itself."""

    forced = None

    @property
    def pixel_array(self):
        return self.forced


def _decoded_as(arr, **kw):
    ds = pydicom.dcmread(io.BytesIO(make_dicom(arr=np.zeros((16, 24), np.uint16), **kw)))
    bad = _DecoderReturned(ds)
    bad.file_meta = ds.file_meta
    bad.forced = arr
    return D.dicom_to_uint8(bad)


def test_dicom_value_above_bits_stored_is_refused():
    a = scan(bits=12)  # values up to ~3100: not representable in 10 bits
    with pytest.raises(ValueError) as ei:
        _decoded_as(a, bits=10)
    n_over = int((a > 1023).sum())
    assert str(ei.value) == f"{n_over} of {a.size} stored values outside the 10-bit range"  # counts only
    ok = np.minimum(a, 1023).astype(np.uint16)  # control: exactly at the limit passes
    out, info = _decoded_as(ok, bits=10)
    assert info["bits_stored"] == 10 and info["raw_max"] == 1023 and out.max() == 255


def test_dicom_signed_range_is_checked_against_the_signed_limits():
    a = np.full((16, 24), -100, np.int16)
    a[4:8, 4:8] = 2047
    out, info = decode(arr=a, rep=1)  # a real signed fixture decodes
    assert out.max() == 255 and info["raw_min"] == -100 and info["raw_max"] == 2047 and info["raw_over"] == 0
    for bad in (2048, -2049):  # above the 1 % fraction: 4 of 384 pixels is 1.04 %
        b = a.copy()
        b.ravel()[:4] = bad
        with pytest.raises(ValueError) as ei:
            _decoded_as(b, bits=12, rep=1)
        assert str(ei.value) == f"4 of {b.size} stored values outside the 12-bit signed range"
    edge = a.copy()
    edge[0, 0] = -2048  # control: the signed limits themselves pass
    assert _decoded_as(edge, bits=12, rep=1)[1]["raw_min"] == -2048


def test_signed_few_over_range_pixels_are_clipped_not_refused():
    a = np.full((16, 24), -100, np.int16)
    a[4:8, 4:8] = 2047
    for bad, lim in ((2100, 2047), (-2100, -2048)):
        b = a.copy()
        b[0, 0] = bad  # 1 of 384 pixels, below the fraction
        twin = a.copy()
        twin[0, 0] = lim
        out, info = _decoded_as(b, bits=12, rep=1)
        ref, ref_info = _decoded_as(twin, bits=12, rep=1)
        assert np.array_equal(out, ref) and info["raw_over"] == 1 and ref_info["raw_over"] == 0
        assert (info["raw_max"] if bad > 0 else info["raw_min"]) == bad  # pre-clip extreme


def _with_over(arr, n, value, bits=12):
    b = arr.copy()
    b.ravel()[:n] = value
    return b


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({}, id="plain"),
        pytest.param({"window": ([2500.0], [1500.0])}, id="window"),
        pytest.param({"photometric": "MONOCHROME1"}, id="monochrome1"),
    ],
)
def test_few_over_range_pixels_are_clipped_to_the_range_maximum(kw):
    a = scan(bits=12)
    over = _with_over(a, 3, 4200)  # 3 of 384 = 0.78 %, range max is 4095
    twin = _with_over(a, 3, 4095)
    out, info = _decoded_as(over, bits=12, **kw)
    ref, ref_info = _decoded_as(twin, bits=12, **kw)
    assert np.array_equal(out, ref)
    assert info["raw_over"] == 3 and ref_info["raw_over"] == 0
    assert info["raw_max"] == 4200 and ref_info["raw_max"] == 4095  # raw_max is the pre-clip value
    assert info["raw_min"] == int(over.min())


def test_more_than_one_percent_over_range_pixels_still_refuse():
    a = scan(bits=12)
    over = _with_over(a, 4, 4200)  # 4 of 384 = 1.04 %
    with pytest.raises(ValueError) as ei:
        _decoded_as(over, bits=12)
    assert str(ei.value) == f"4 of {a.size} stored values outside the 12-bit range"
    assert D.RAW_OVER_MAX_FRAC == 0.01


def test_clean_image_has_raw_over_zero():
    assert decode(arr=scan())[1]["raw_over"] == 0


def test_info_carries_the_header_facts_and_is_the_allow_list_label():
    a = scan()
    for ts, label in ((IVR, "implicit_le"), (EVR, "explicit_le"), (J2K, "jpeg2000_lossless")):
        _, info = decode(arr=a, ts=ts)
        assert info["transfer_syntax"] == label
        assert info["raw_min"] == int(a.min()) and info["raw_max"] == int(a.max())
        assert set(info) == {
            "photometric",
            "had_voi",
            "n_windows",
            "bits_stored",
            "transfer_syntax",
            "raw_min",
            "raw_max",
            "raw_over",
        }


def test_unknown_transfer_syntax_has_the_label_other():
    data = make_dicom(arr=scan(), ts=EVR)
    assert b"1.2.840.10008.1.2.1\x00" in data
    rle = data.replace(b"1.2.840.10008.1.2.1\x00", b"1.2.840.10008.1.2.5\x00")  # same length: RLE Lossless
    ds = pydicom.dcmread(io.BytesIO(rle), stop_before_pixels=True)
    assert D.header_facts(ds)["transfer_syntax"] == "other"


def test_jpeg2000_lossless_decodes_exactly_and_matches_the_uncompressed_twin():
    a = scan(seed=5)
    data = make_dicom(arr=a, ts=J2K)
    ds = pydicom.dcmread(io.BytesIO(data))
    assert ds.file_meta.TransferSyntaxUID == J2K
    np.testing.assert_array_equal(ds.pixel_array, a)  # lossless: the stored values are identical
    j2k_out, j2k_info = D.decode_dicom_bytes(data)
    raw_out, _ = D.decode_dicom_bytes(make_dicom(arr=a, ts=EVR))
    np.testing.assert_array_equal(j2k_out, raw_out)
    assert j2k_info["transfer_syntax"] == "jpeg2000_lossless" and j2k_info["raw_max"] == int(a.max())


def test_decode_dicom_bytes_is_a_picklable_module_level_function():
    f = pickle.loads(pickle.dumps(D.decode_dicom_bytes))
    out, info = f(make_dicom(arr=scan()))
    assert out.dtype == np.uint8 and info["bits_stored"] == 12


def _linear_window(x, c, w, ymin=0.0, ymax=4095.0):
    """DICOM C.11.2.1.2.1 LINEAR window, written out in plain numpy (not pydicom's apply_windowing)."""
    x = x.astype(np.float64)
    y = ((x - (c - 0.5)) / (w - 1) + 0.5) * (ymax - ymin) + ymin
    y[x <= c - 0.5 - (w - 1) / 2] = ymin
    y[x > c - 0.5 + (w - 1) / 2] = ymax
    return y


def test_monochrome1_with_an_asymmetric_window_windows_the_stored_values_then_inverts():
    a = np.linspace(0, 4095, 16 * 24).astype(np.uint16).reshape(16, 24)
    c, w = 1000, 400  # far from the middle of 0..4095: windowing the inverted values gives another picture
    windowed = _linear_window(a, c, w)
    x = windowed.max() - windowed  # window first, then MONOCHROME1 inversion
    want = np.rint((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    out, info = decode(arr=a, photometric="MONOCHROME1", window=([c], [w]))
    assert info["photometric"] == "MONOCHROME1" and info["had_voi"] is True
    assert np.abs(out.astype(int) - want.astype(int)).max() <= 1
    swapped = _linear_window(
        (4095 - a), c, w
    )  # the wrong order: the guard against it needs a different array
    other = np.rint((swapped - swapped.min()) / (swapped.max() - swapped.min()) * 255).astype(np.uint8)
    assert np.abs(want.astype(int) - other.astype(int)).max() > 50


@pytest.mark.parametrize("drop", ["WindowWidth", "WindowCenter"])
def test_a_window_centre_without_a_width_is_no_window(drop):
    data = make_dicom(arr=scan(seed=2), window=([1000], [200]))
    full = pydicom.dcmread(io.BytesIO(data))
    ds = pydicom.dcmread(io.BytesIO(data))
    delattr(ds, drop)
    header_only = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
    delattr(header_only, drop)
    for facts in (D.header_facts(header_only), D.dicom_to_uint8(ds)[1]):
        assert facts["had_voi"] is False and facts["n_windows"] == 0
    assert D.header_facts(full)["had_voi"] is True  # control: both present
    plain, _ = decode(arr=scan(seed=2))
    np.testing.assert_array_equal(D.dicom_to_uint8(ds)[0], plain)  # and no window was applied
