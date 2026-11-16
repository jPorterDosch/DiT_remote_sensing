"""src/datasets/cxr14.py on synthetic ChestX-ray14 archives: conversion layout and guards, the
patient-level manifest, the index, the dataset views. CPU only; real files in tmp_path."""

from __future__ import annotations

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
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import cxr14 as G  # noqa: E402
from eval import cxr_splits as S  # noqa: E402

N_TV_PATIENTS, N_TEST_PATIENTS, PER_PATIENT, N_ARCH = 20, 5, 4, 3


def _png(seed: int, mode: str = "L", size=(20, 16)) -> bytes:
    rng = np.random.default_rng(seed)
    ch = {"L": 1, "RGBA": 4, "RGB": 3}.get(mode)
    if mode == "I;16":
        im = Image.fromarray(rng.integers(0, 60000, (size[1], size[0]), dtype=np.uint16))
    elif ch == 1:
        im = Image.fromarray(rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8))
    else:
        im = Image.fromarray(rng.integers(0, 255, (size[1], size[0], ch), dtype=np.uint8), mode)
    assert im.mode == mode, im.mode
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _tar(path, members):
    """members: [(name, bytes)] written under images/ in this order."""
    with tarfile.open(path, "w:gz") as t:
        for name, data in members:
            ti = tarfile.TarInfo(f"images/{name}")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return str(path)


def _names(n, patient0=1):
    return [f"{patient0 + i // 7:08d}_{i % 7:03d}.png" for i in range(n)]


def test_convert_layout_and_bytes_untouched(tmp_path):
    names = _names(70)
    tar = _tar(tmp_path / "a.tar.gz", [(n, _png(i)) for i, n in enumerate(names)])
    out = tmp_path / "arrow"
    meta = G.convert_archive(tar, str(out), 0)
    rd = ipc.open_file(pa.memory_map(str(out / G.shard_name(0))))
    assert [rd.get_batch(i).num_rows for i in range(rd.num_record_batches)] == [64, 6]
    assert rd.schema.equals(G.SCHEMA)
    for r in (0, 63, 64, 69):
        b = rd.get_batch(r // 64)
        j = r % 64
        assert b.column("img_name")[j].as_py() == names[r]
        assert b.column("image_bytes")[j].as_py() == _png(r)
        assert b.column("patient_id")[j].as_py() == int(names[r][:8])
    on_disk = json.load(open(out / "meta" / (G.shard_name(0) + ".json")))
    assert on_disk == meta
    assert meta["rows"] == 70 and meta["names"] == names and meta["shard"] == 0
    assert meta["sha256"] == hashlib.sha256(open(out / G.shard_name(0), "rb").read()).hexdigest()
    assert meta["sizes"] == {"20x16": 70}
    assert not list(out.glob("*.partial"))


def test_convert_counts_modes_and_refuses_16bit(tmp_path):
    names = _names(6)
    modes = ["L", "L", "RGBA", "L", "RGBA", "I;16"]
    members = [(n, _png(i, m)) for i, (n, m) in enumerate(zip(names, modes, strict=True))]
    with pytest.raises(SystemExit, match="16-bit|mode"):
        G.convert_archive(_tar(tmp_path / "bad.tar.gz", members), str(tmp_path / "o1"), 0)
    assert not (tmp_path / "o1" / G.shard_name(0)).exists()
    good = _tar(tmp_path / "ok.tar.gz", members[:5])
    meta = G.convert_archive(good, str(tmp_path / "o2"), 0)
    assert meta["pil_modes"] == {"L": 3, "RGBA": 2}


def test_convert_fails_on_corrupt_member_and_leaves_no_shard(tmp_path):
    names = _names(60)
    members = [(n, _png(i)) for i, n in enumerate(names)]
    members[40] = (names[40], _png(40)[: len(_png(40)) // 2])  # truncated: open() succeeds, load() fails
    out = tmp_path / "arrow"
    with pytest.raises(SystemExit, match=r"member 40\b"):
        G.convert_archive(_tar(tmp_path / "c.tar.gz", members), str(out), 0)
    assert not (out / G.shard_name(0)).exists()
    assert not (out / "meta" / (G.shard_name(0) + ".json")).exists()


def test_convert_refuses_bad_name_and_duplicate(tmp_path):
    m = [(n, _png(i)) for i, n in enumerate(_names(3))]
    with pytest.raises(SystemExit, match="patient|prefix"):
        G.convert_archive(
            _tar(tmp_path / "x.tar.gz", [("abcdefgh_000.png", _png(0)), *m]), str(tmp_path / "o1"), 0
        )
    with pytest.raises(SystemExit, match="duplicate"):
        G.convert_archive(_tar(tmp_path / "y.tar.gz", [*m, m[0]]), str(tmp_path / "o2"), 0)


def test_convert_refuses_existing_and_partial(tmp_path):
    tar = _tar(tmp_path / "a.tar.gz", [(n, _png(i)) for i, n in enumerate(_names(5))])
    out = tmp_path / "arrow"
    G.convert_archive(tar, str(out), 0)
    with pytest.raises(SystemExit, match="exists"):
        G.convert_archive(tar, str(out), 0)
    out2 = tmp_path / "arrow2"
    out2.mkdir()
    (out2 / (G.shard_name(0) + ".partial")).write_bytes(b"x")
    with pytest.raises(SystemExit, match="partial"):
        G.convert_archive(tar, str(out2), 0)
    assert (out2 / (G.shard_name(0) + ".partial")).read_bytes() == b"x"  # left for the user


# ----------------------------------------------------------------------------- corpus fixture
@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """3 archives, 20 train/val patients + 5 test patients x 4 images; modes alternate L / RGBA."""
    monkeypatch.setattr(G, "N_ARCHIVES", N_ARCH)
    rng = np.random.default_rng(0)
    tv = [f"{p:08d}" for p in range(1, N_TV_PATIENTS + 1)]
    te = [f"{p:08d}" for p in range(100, 100 + N_TEST_PATIENTS)]
    rows = [(f"{p}_{k:03d}.png", p) for p in tv + te for k in range(PER_PATIENT)]
    monkeypatch.setattr(G, "N_IMAGES", len(rows))
    order = rng.permutation(len(rows))
    raw = tmp_path / "raw"
    raw.mkdir()
    per = -(-len(rows) // N_ARCH)
    for a in range(N_ARCH):
        idx = order[a * per : (a + 1) * per]
        _tar(raw / G.archive_name(a), [(rows[i][0], _png(i, "L" if i % 2 == 0 else "RGBA")) for i in idx])
    pd.DataFrame(
        {
            "Image Index": [r[0] for r in rows],
            "Finding Labels": "No Finding",
            "Follow-up #": 0,
            "Patient ID": [int(r[1]) for r in rows],
        }
    ).to_csv(raw / "Data_Entry_2017_v2020.csv", index=False)
    (raw / "train_val_list.txt").write_text("\n".join(n for n, p in rows if p in tv))  # no final newline
    (raw / "test_list.txt").write_text("\n".join(n for n, p in rows if p in te))
    arrow = tmp_path / "arrow"
    for a in range(N_ARCH):
        G.convert_archive(str(raw / G.archive_name(a)), str(arrow), a)
    return {"tmp": tmp_path, "raw": str(raw), "arrow": str(arrow), "rows": rows, "te": set(te), "tv": set(tv)}


def _manifest(c, name="man", shards=None):
    return G.build_manifest(c["arrow"], c["raw"], str(c["tmp"] / name), shards)


def test_archive_and_shard_names():
    assert [G.archive_name(0), G.archive_name(11)] == ["images_001.tar.gz", "images_012.tar.gz"]
    assert G.shard_name(0) == f"cxr14-00-of-{G.N_ARCHIVES:02d}.arrow"
    assert (G.N_ARCHIVES, G.N_IMAGES) == (12, 112120)


def test_manifest_counts_and_complete_flag(corpus):
    rep = _manifest(corpus)
    n = len(corpus["rows"])
    assert (
        rep["alpha_nonopaque"]
        == sum(
            json.load(open(os.path.join(corpus["arrow"], "meta", G.shard_name(i) + ".json")))[
                "alpha_nonopaque"
            ]
            for i in range(N_ARCH)
        )
        > 0
    )
    assert rep["complete"] is True and rep["n"]["align"] + rep["n"]["validation"] + rep["n"]["test"] == n
    assert rep["n"]["test"] == N_TEST_PATIENTS * PER_PATIENT
    assert rep["patients"]["test"] == N_TEST_PATIENTS
    assert rep["patients"]["align"] + rep["patients"]["validation"] == N_TV_PATIENTS
    d = corpus["tmp"] / "man"
    assert json.load(open(d / "report.json")) == rep
    rows = pd.read_csv(d / "rows.csv", dtype={"patient_id": str})
    assert len(rows) == n and rows["img_name"].is_unique
    ret = pd.read_csv(d / "retained.csv", dtype={"source": str})
    val = pd.read_csv(d / "validation.csv", dtype={"source": str})
    assert list(ret.columns) == ["shard", "row", "source", "img_name"] and len(ret) == rep["n"]["align"]
    banned = set(corpus["te"]) | set(val["source"])
    assert not (set(ret["source"]) & banned)
    assert not (set(val["source"]) & (set(corpus["te"]) | set(ret["source"])))
    assert set(val["source"]) <= corpus["tv"]
    assert len(rep["inputs_sha256"]) == 3 + N_ARCH and set(rep["pil_modes"]) == {"L", "RGBA"}
    with pytest.raises(SystemExit, match="exists"):
        _manifest(corpus)


def _redo(c, tag, edit_csv=None, edit_lists=None):
    raw = c["tmp"] / f"raw_{tag}"
    raw.mkdir()
    csv = pd.read_csv(os.path.join(c["raw"], "Data_Entry_2017_v2020.csv"))
    if edit_csv:
        csv = edit_csv(csv)
    csv.to_csv(raw / "Data_Entry_2017_v2020.csv", index=False)
    lists = {f: open(os.path.join(c["raw"], f)).read() for f in ("train_val_list.txt", "test_list.txt")}
    if edit_lists:
        lists = edit_lists(lists)
    for f, t in lists.items():
        (raw / f).write_text(t)
    return G.build_manifest(c["arrow"], str(raw), str(c["tmp"] / f"man_{tag}"), None)


def test_build_manifest_rejects_noncanonical_patient_id(corpus):
    """A zero-padded Patient ID is numerically equal to the name prefix but would re-key the split."""

    def padded(df):
        df["Patient ID"] = df["Patient ID"].astype(str).str.zfill(8)
        return df

    with pytest.raises(SystemExit, match="canonical|zero|padded"):
        _redo(corpus, "pad", edit_csv=padded)


def test_build_manifest_keys_patients_by_csv_patient_id(tmp_path, monkeypatch):
    """The split is pinned on the CSV Patient ID string ("1"), not the 8-digit name prefix ("00000001")."""
    n_tv, n_te = 220, 20
    monkeypatch.setattr(G, "N_ARCHIVES", 1)
    tv = list(range(1, n_tv + 1))
    te = list(range(1000, 1000 + n_te))
    rows = [(f"{p:08d}_{k:03d}.png", p) for p in tv + te for k in range(2)]
    monkeypatch.setattr(G, "N_IMAGES", len(rows))
    raw = tmp_path / "raw"
    raw.mkdir()
    _tar(raw / G.archive_name(0), [(n, _png(i, size=(4, 4))) for i, (n, _) in enumerate(rows)])
    pd.DataFrame(
        {
            "Image Index": [n for n, _ in rows],
            "Finding Labels": "No Finding",
            "Follow-up #": 0,
            "Patient ID": [str(p) for _, p in rows],
        }
    ).to_csv(raw / "Data_Entry_2017_v2020.csv", index=False)
    tv_names = {n for n, p in rows if p in tv}
    te_names = {n for n, p in rows if p in te}
    (raw / "train_val_list.txt").write_text("\n".join(sorted(tv_names)))
    (raw / "test_list.txt").write_text("\n".join(sorted(te_names)))
    arrow = tmp_path / "arrow"
    G.convert_archive(str(raw / G.archive_name(0)), str(arrow), 0)
    out = str(tmp_path / "man")
    G.build_manifest(str(arrow), str(raw), out, None)
    got = pd.read_csv(os.path.join(out, "rows.csv")).set_index("img_name")["bucket"].to_dict()
    unpadded = S.cxr14_buckets({n: str(p) for n, p in rows}, tv_names, te_names)
    prefix = S.cxr14_buckets({n: n[:8] for n, _ in rows}, tv_names, te_names)
    assert got == unpadded
    # the instrument can tell the two keys apart on this fixture
    assert sum(unpadded[n] != prefix[n] for n in unpadded) > 0


def test_manifest_guards_fire(corpus):
    def extra_row(df):
        return pd.concat(
            [df, df.iloc[[0]].assign(**{"Image Index": "99999999_000.png", "Patient ID": 99999999})]
        )

    def wrong_patient(df):
        df.loc[3, "Patient ID"] += 1
        return df

    def overlap(lists):
        first = lists["train_val_list.txt"].splitlines()[0]
        lists["test_list.txt"] += "\n" + first
        return lists

    def drop_from_list(lists):
        lists["test_list.txt"] = "\n".join(lists["test_list.txt"].splitlines()[1:])
        return lists

    with pytest.raises(SystemExit, match="Image Index"):
        _redo(corpus, "a", edit_csv=extra_row)
    with pytest.raises(SystemExit, match="Patient ID"):
        _redo(corpus, "b", edit_csv=wrong_patient)
    with pytest.raises(SystemExit, match="disjoint|both"):
        _redo(corpus, "c", edit_lists=overlap)
    with pytest.raises(SystemExit, match="lists"):
        _redo(corpus, "d", edit_lists=drop_from_list)


def test_manifest_fails_on_missing_shard_and_tampered_shard(corpus):
    os.remove(os.path.join(corpus["arrow"], "meta", G.shard_name(1) + ".json"))
    with pytest.raises(SystemExit, match="missing"):
        _manifest(corpus)


def test_manifest_detects_modified_shard(corpus):
    shard = os.path.join(corpus["arrow"], G.shard_name(2))
    with open(shard, "r+b") as f:
        f.seek(os.path.getsize(shard) // 2)
        f.write(b"\x00\x01\x02\x03")
    with pytest.raises(SystemExit, match="sha256"):
        _manifest(corpus)


def _edit_csv(path, fn):
    df = pd.read_csv(path, dtype=str)
    fn(df)
    df.to_csv(path, index=False)


def test_build_index_refuses_incomplete_or_tampered_manifest(corpus):
    import shutil

    _manifest(corpus)
    src = corpus["tmp"] / "man"

    def variant(tag, fn, match):
        d = corpus["tmp"] / tag
        shutil.copytree(src, d)
        fn(d)
        with pytest.raises(SystemExit, match=match):
            G.build_index(str(d), str(corpus["tmp"] / f"{tag}.npz"))

    def no_complete(d):
        r = json.load(open(d / "report.json"))
        del r["complete"]
        json.dump(r, open(d / "report.json", "w"))

    def drop_row(d):
        _edit_csv(d / "retained.csv", lambda df: df.drop(index=[0], inplace=True))

    def val_into_retained(d):  # counts differ AND a patient is shared
        val = pd.read_csv(d / "validation.csv", dtype=str)
        ret = pd.read_csv(d / "retained.csv", dtype=str)
        pd.concat([ret, val.iloc[[0]]]).to_csv(d / "retained.csv", index=False)

    def shared_patient_keeping_counts(d):  # same counts; a retained row now names a validation patient
        val = pd.read_csv(d / "validation.csv", dtype=str)
        ret = pd.read_csv(d / "retained.csv", dtype=str)
        ret.loc[0, "source"] = val.loc[0, "source"]
        ret.to_csv(d / "retained.csv", index=False)

    def test_row_into_retained(d):  # same counts; a TEST-bucket image replaces a retained row
        rows = pd.read_csv(d / "rows.csv", dtype=str)
        t = rows[rows["bucket"] == "test"].iloc[0]
        ret = pd.read_csv(d / "retained.csv", dtype=str)
        ret.loc[0] = [t["shard"], t["row"], t["img_name"][:8], t["img_name"]]
        ret.to_csv(d / "retained.csv", index=False)

    variant("a", no_complete, "complete=true")
    variant("b", drop_row, "report.json says")
    variant("c", val_into_retained, "report.json says")
    variant("d", shared_patient_keeping_counts, "share 1 patients")
    variant("e", test_row_into_retained, "disagree with rows.csv buckets")
    G.build_index(str(src), str(corpus["tmp"] / "ok.npz"))  # the untampered manifest still passes


def test_smoke_manifest_needs_allow_smoke_and_is_stamped(corpus):
    rep = _manifest(corpus, "sm", shards=[0])
    assert rep["complete"] is False and rep["smoke"] is True
    assert (
        rep["n"]["align"] + rep["n"]["validation"] + rep["n"]["test"]
        == json.load(open(os.path.join(corpus["arrow"], "meta", G.shard_name(0) + ".json")))["rows"]
    )
    d = str(corpus["tmp"] / "sm")
    with pytest.raises(SystemExit, match="smoke"):
        G.build_index(d, str(corpus["tmp"] / "s.npz"))
    G.build_index(d, str(corpus["tmp"] / "s.npz"), allow_smoke=True)
    assert G.index_description(str(corpus["tmp"] / "s.npz"))["smoke"] is True
    with pytest.raises(ValueError, match="smoke index"):
        G.Cxr14Dataset(corpus["arrow"], str(corpus["tmp"] / "s.npz"), which="all")
    ds = G.Cxr14Dataset(corpus["arrow"], str(corpus["tmp"] / "s.npz"), which="all", allow_smoke=True)
    assert ds.describe()["smoke"] is True and len(ds) > 0
    with pytest.raises(SystemExit, match="exists"):
        G.build_index(d, str(corpus["tmp"] / "s.npz"), allow_smoke=True)


@pytest.fixture
def indexed(corpus):
    _manifest(corpus)
    npz = str(corpus["tmp"] / "index.npz")
    G.build_index(str(corpus["tmp"] / "man"), npz)
    return corpus, npz


def test_index_fields_and_description(indexed):
    c, npz = indexed
    z = np.load(npz)
    assert [str(b) for b in z["buckets"]] == ["retained", "validation"]
    assert bool(z["one_row_per_image"]) and not bool(z["smoke"])
    order = np.lexsort((z["row"], z["shard"]))
    assert (order == np.arange(len(order))).all()  # sorted by (shard, row)
    d = G.index_description(npz)
    assert d["smoke"] is False and d["rows"] == N_TV_PATIENTS * PER_PATIENT
    assert d["patients"]["retained"] + d["patients"]["validation"] == N_TV_PATIENTS


def test_dataset_buckets_and_no_test_rows(indexed):
    c, npz = indexed
    tr, va, al = (G.Cxr14Dataset(c["arrow"], npz, img_size=32, which=w) for w in ("train", "val", "all"))
    assert len(tr) + len(va) == len(al) == N_TV_PATIENTS * PER_PATIENT and len(va) > 0
    test_names = {n for n, p in c["rows"] if p in c["te"]}
    for ds in (tr, va, al):
        reachable = {ds.raw(i)[0] for i in range(len(ds))}
        assert not (reachable & test_names)
    assert not ({tr.raw(i)[0] for i in range(len(tr))} & {va.raw(i)[0] for i in range(len(va))})
    with pytest.raises(ValueError, match="horizontal flip is refused"):
        G.Cxr14Dataset(c["arrow"], npz, hflip=True)
    with pytest.raises(ValueError):
        G.Cxr14Dataset(c["arrow"], npz, which="test")
    assert al.describe()["which"] == "all" and al.describe()["smoke"] is False
    os.remove(os.path.join(c["arrow"], G.shard_name(1)))
    with pytest.raises(FileNotFoundError, match="datasets.cxr14 convert"):
        G.Cxr14Dataset(c["arrow"], npz, which="all")


def test_item_is_rgb_224_in_unit_range(indexed):
    c, npz = indexed
    ds = G.Cxr14Dataset(c["arrow"], npz, which="all")
    seen = set()
    for i in range(len(ds)):
        mode = Image.open(io.BytesIO(ds.raw(i)[1])).mode
        if mode in seen:
            continue
        seen.add(mode)
        item = ds[i]
        x = item["img"]
        assert tuple(x.shape) == (3, 224, 224) and float(x.min()) >= -1 and float(x.max()) <= 1
        assert item["index"] == i and item["label"] == -1
        if mode == "L":
            assert (x[0] == x[1]).all() and (x[1] == x[2]).all()
    assert seen == {"L", "RGBA"}


def test_cli_check_pins_counts_and_reports_smoke(corpus, capsys):
    _manifest(corpus)
    npz = str(corpus["tmp"] / "index.npz")
    G.build_index(str(corpus["tmp"] / "man"), npz)
    argv = [
        "check",
        "--arrow-dir",
        corpus["arrow"],
        "--manifest-dir",
        str(corpus["tmp"] / "man"),
        "--index",
        npz,
    ]
    with pytest.raises(SystemExit, match="pinned"):  # synthetic counts are not the real corpus's
        G.main(argv)
    _manifest(corpus, "sm", shards=[0])
    snpz = str(corpus["tmp"] / "s.npz")
    G.build_index(str(corpus["tmp"] / "sm"), snpz, allow_smoke=True)
    assert (
        G.main(
            [
                "check",
                "--arrow-dir",
                corpus["arrow"],
                "--manifest-dir",
                str(corpus["tmp"] / "sm"),
                "--index",
                snpz,
            ]
        )
        == 0
    )
    assert "smoke" in capsys.readouterr().out


def test_cli_convert_and_parse(corpus, tmp_path):
    out = tmp_path / "cli_arrow"
    assert G.main(["convert", "--shard", "1", "--raw-dir", corpus["raw"], "--out-dir", str(out)]) == 0
    assert (out / G.shard_name(1)).exists()
    assert G._parse_shards("0,1-2") == [0, 1, 2]
    with pytest.raises(Exception):
        G._parse_shards("3")


# ----------------------------------------------------------------------------- check: index vs shards
def _check_argv(c, man="man", npz="index.npz"):
    return [
        "check",
        "--arrow-dir",
        c["arrow"],
        "--manifest-dir",
        str(c["tmp"] / man),
        "--index",
        str(c["tmp"] / npz),
    ]


def _full(c):
    _manifest(c)
    G.build_index(str(c["tmp"] / "man"), str(c["tmp"] / "index.npz"))


def _addressed(c, k=0):
    z = np.load(c["tmp"] / "index.npz")
    return int(z["shard"][k]), int(z["row"][k])


def _rewrite_shard(c, shard, fn):
    """Rewrite one shard file with fn(dict of column lists) -> dict, in 64-row batches."""
    path = os.path.join(c["arrow"], G.shard_name(shard))
    with pa.memory_map(path) as src:
        cols = ipc.open_file(src).read_all().to_pydict()
    cols = fn(cols)
    tbl = pa.table(cols, schema=G.SCHEMA)
    with pa.OSFile(path + ".new", "wb") as sink:
        w = ipc.new_file(sink, G.SCHEMA)
        for b in tbl.to_batches(max_chunksize=64):
            w.write_batch(b)
        w.close()
    os.replace(path + ".new", path)


def test_check_passes_and_hashes_shards_when_pinned_counts_match(corpus, monkeypatch, capsys):
    _full(corpus)
    rep = json.load(open(corpus["tmp"] / "man" / "report.json"))
    monkeypatch.setattr(G, "PINNED_N", rep["n"])
    monkeypatch.setattr(G, "PINNED_PATIENTS", rep["patients"])
    assert G.main(_check_argv(corpus)) == 0
    out = capsys.readouterr().out
    assert "pinned counts match" in out and "sha256 of 3 shards hashed" in out and "CXR14_CHECK_OK" in out


def test_check_fails_when_shards_are_swapped(corpus):
    _full(corpus)
    a, b = (os.path.join(corpus["arrow"], G.shard_name(i)) for i in (0, 1))
    os.rename(a, a + ".t")
    os.rename(b, a)
    os.rename(a + ".t", b)
    with pytest.raises(SystemExit, match="different img_name"):
        G.main(_check_argv(corpus))


def test_check_fails_when_a_shard_is_reversed(corpus):
    _full(corpus)
    _rewrite_shard(corpus, 2, lambda c: {k: v[::-1] for k, v in c.items()})
    with pytest.raises(SystemExit, match="different img_name"):
        G.main(_check_argv(corpus))


def test_check_fails_on_one_different_image_name(corpus):
    _full(corpus)
    s, r = _addressed(corpus)

    def rename(c):
        c["img_name"][r] = "99999999_000.png"
        return c

    _rewrite_shard(corpus, s, rename)
    with pytest.raises(SystemExit, match=r"\b1 of \d+ index rows have a different img_name"):
        G.main(_check_argv(corpus))


def test_check_fails_on_shard_patient_column(corpus):
    _full(corpus)
    s, r = _addressed(corpus)

    def bump(c):
        c["patient_id"][r] += 1
        return c

    _rewrite_shard(corpus, s, bump)
    with pytest.raises(SystemExit, match="patient_id"):
        G.main(_check_argv(corpus))


def test_check_fails_on_same_names_different_bytes(corpus):
    _full(corpus)
    s, r = _addressed(corpus)

    def swap_bytes(c):
        c["image_bytes"][r] = _png(12345)
        return c

    _rewrite_shard(corpus, s, swap_bytes)
    with pytest.raises(SystemExit, match="sha256 .* differs"):
        G.main(_check_argv(corpus))


def test_check_fails_on_index_bucket_disagreeing_with_manifest(corpus):
    _full(corpus)
    z = dict(np.load(corpus["tmp"] / "index.npz"))
    z["bucket_code"] = z["bucket_code"].copy()
    z["bucket_code"][0] = 1 - z["bucket_code"][0]
    with open(corpus["tmp"] / "flipped.npz", "wb") as f:
        np.savez(f, **z)
    with pytest.raises(SystemExit, match="different bucket"):
        G.main(_check_argv(corpus, npz="flipped.npz"))


def test_convert_counts_nonopaque_alpha(tmp_path):
    def rgba(transparent):
        im = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
        if transparent:
            im.putpixel((0, 0), (10, 20, 30, 0))
        b = io.BytesIO()
        im.save(b, format="PNG")
        return b.getvalue()

    names = _names(3)
    members = [(names[0], rgba(False)), (names[1], rgba(True)), (names[2], _png(2))]
    meta = G.convert_archive(_tar(tmp_path / "a.tar.gz", members), str(tmp_path / "o"), 0)
    assert meta["alpha_nonopaque"] == 1 and meta["pil_modes"] == {"RGBA": 2, "L": 1}
    ipc_rows = ipc.open_file(pa.memory_map(str(tmp_path / "o" / G.shard_name(0)))).read_all()
    assert ipc_rows.column("image_bytes")[1].as_py() == rgba(True)  # stored bytes untouched
