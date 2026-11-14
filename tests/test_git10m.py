"""src/datasets/git10m.py on a synthetic Hub-style stream shard: conversion layout, the
metadata guards, index construction from dedup manifests, the dataset transform / holdout,
and the resumable sampler. CPU only, no network."""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import git10m as G  # noqa: E402

N_ROWS = 150  # > 2 Hub batches of 70 -> the 64-row grid must not restart at a source batch


def _jpeg(seed: int, size=(40, 30)) -> bytes:
    rng = np.random.default_rng(seed)
    im = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=95)
    return b.getvalue()


def _stream_shard(path: str, n: int = N_ROWS, hub_batch: int = 70, names=None):
    names = names or [f"Google-10-10_{i}_5.jpg" for i in range(n)]
    schema = pa.schema(
        [
            ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("text", pa.string()),
            ("img_name", pa.string()),
            ("img_quality_score", pa.float64()),
            ("caption", pa.string()),
            ("caption_length", pa.int64()),
            ("resolution", pa.string()),
            ("Google_location", pa.string()),
        ]
    )
    with pa.OSFile(path, "wb") as sink:
        w = ipc.new_stream(sink, schema)
        for s in range(0, n, hub_batch):
            idx = list(range(s, min(n, s + hub_batch)))
            w.write_batch(
                pa.RecordBatch.from_pydict(
                    {
                        "image": [{"bytes": _jpeg(i), "path": None} for i in idx],
                        "text": ["t"] * len(idx),
                        "img_name": [names[i] for i in idx],
                        "img_quality_score": [3.0] * len(idx),
                        "caption": [f"caption {i}" for i in idx],
                        "caption_length": [2] * len(idx),
                        "resolution": ["1.0"] * len(idx),
                        "Google_location": [f"10_{i}_5" for i in idx],
                    },
                    schema=schema,
                )
            )
        w.close()
    return names


def _meta(path: str, names):
    pd.DataFrame({"row": np.arange(len(names)), "img_name": names}).to_parquet(path, index=False)


@pytest.fixture
def shard0(tmp_path):
    stream = tmp_path / "stream" / "train"
    stream.mkdir(parents=True)
    names = _stream_shard(str(stream / G.shard_name(0)))
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    _meta(str(meta_dir / "meta-00000.parquet"), names)
    arrow = tmp_path / "arrow"
    G.fetch_convert(range(0, 1), str(tmp_path / "stream"), str(arrow), str(meta_dir), False, True)
    assert not (stream / G.shard_name(0)).exists()  # --rm-stream
    return tmp_path, arrow, names


def test_constants_match_geo_dedup():
    from eval import geo_dedup

    assert (G.HF_REPO, G.N_SHARDS) == (geo_dedup.HF_REPO, geo_dedup.N_SHARDS)


def test_convert_layout_is_a_uniform_64_row_grid(shard0):
    _, arrow, names = shard0
    rd = ipc.open_file(pa.memory_map(str(arrow / G.shard_name(0))))
    sizes = [rd.get_batch(i).num_rows for i in range(rd.num_record_batches)]
    assert sizes[:-1] == [G.BATCH_ROWS] * (len(sizes) - 1) and sum(sizes) == N_ROWS
    assert rd.schema.equals(G.SCHEMA)
    # row r is batch r // 64, slot r % 64, and the JPEG bytes are byte-identical to the source
    for r in (0, 63, 64, 69, 70, 100, N_ROWS - 1):
        b = rd.get_batch(r // G.BATCH_ROWS)
        j = r % G.BATCH_ROWS
        assert b.column("img_name")[j].as_py() == names[r]
        assert b.column("image_bytes")[j].as_py() == _jpeg(r)


def test_rebatch_ignores_source_chunking():
    """70-row chunks (Hub batches) and a chunk split that combine_chunks would not merge."""
    parts = [pa.table({"a": list(range(s, min(s + 70, N_ROWS)))}) for s in range(0, N_ROWS, 70)]
    sizes = [b.num_rows for b in G._rebatch(pa.concat_tables(parts))]
    assert sizes[:-1] == [G.BATCH_ROWS] * (len(sizes) - 1) and sum(sizes) == N_ROWS
    rows = [v for b in G._rebatch(pa.concat_tables(parts)) for v in b.column("a").to_pylist()]
    assert rows == list(range(N_ROWS))


def test_convert_guards(tmp_path):
    stream = tmp_path / "s.arrow"
    names = _stream_shard(str(stream), n=10)
    out = tmp_path / "out.arrow"
    bad = tmp_path / "bad.parquet"
    _meta(str(bad), names[:9])
    with pytest.raises(ValueError, match="rows"):
        G.convert_shard(str(stream), str(out), str(bad))
    assert not out.exists() and not (tmp_path / "out.arrow.tmp").exists()
    _meta(str(bad), names[::-1])
    with pytest.raises(ValueError, match="img_name sequence"):
        G.convert_shard(str(stream), str(out), str(bad))
    assert G.convert_shard(str(stream), str(out), None) == 10
    with pytest.raises(FileExistsError):
        G.convert_shard(str(stream), str(out), None)
    with pytest.raises(SystemExit, match=r"missing \(pass --hf-download\)"):
        G.fetch_convert(
            range(0, 1), str(tmp_path), str(tmp_path / "a"), str(tmp_path / "nometa"), False, False
        )
    (tmp_path / "train").mkdir()
    os.replace(str(stream), str(tmp_path / "train" / G.shard_name(0)))
    with pytest.raises(SystemExit, match="meta-00000.parquet missing"):
        G.fetch_convert(
            range(0, 1), str(tmp_path), str(tmp_path / "a"), str(tmp_path / "nometa"), False, False
        )


def _manifests(d, retained, unassessable, extra_report=None):
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(retained, columns=["shard", "row", "img_name", "google_location", "source"]).to_csv(
        os.path.join(d, "retained.csv"), index=False
    )
    pd.DataFrame(unassessable, columns=["shard", "row", "img_name", "source", "reason"]).to_csv(
        os.path.join(d, "unassessable.csv"), index=False
    )
    rep = {
        "git10m_rows": 999,
        "buffer_m": 1000.0,
        "retained": len(retained),
        "unassessable": len(unassessable),
    }
    rep.update(extra_report or {})
    json.dump(rep, open(os.path.join(d, "report.json"), "w"))


def test_build_index_counts_sources_and_guards(tmp_path):
    d = str(tmp_path / "m")
    ret = [(0, r, f"n{r}", "10_1_1", "google_tile") for r in range(0, N_ROWS, 2)]
    una = [(0, r, f"n{r}", "million_aid", "no coords") for r in range(1, 20, 2)]
    _manifests(d, ret, una)
    rep = G.build_index(d, str(tmp_path / "i_ret.npz"), include_unassessable=False)
    assert rep["rows"] == len(ret) and rep["per_bucket"] == {"retained": len(ret)}
    rep = G.build_index(d, str(tmp_path / "i_all.npz"), include_unassessable=True)
    assert rep["per_bucket"] == {"retained": len(ret), "unassessable": len(una)}
    assert rep["per_source"] == {"google_tile": len(ret), "million_aid": len(una)}
    z = np.load(str(tmp_path / "i_all.npz"))
    rows = z["row"]
    assert (np.diff(rows) > 0).all()  # sorted by (shard, row)
    assert set(json.loads(str(z["manifest_sha256"]))) == {"retained.csv", "unassessable.csv"}
    assert bool(z["include_unassessable"]) and float(z["buffer_m"]) == 1000.0
    # counts must agree with report.json
    _manifests(str(tmp_path / "m2"), ret, una, {"retained": len(ret) + 1})
    with pytest.raises(ValueError, match="report.json"):
        G.build_index(str(tmp_path / "m2"), str(tmp_path / "i2.npz"), False)
    # excluded rows are never indexed: a row absent from both manifests is absent from the index
    assert 1 not in set(np.load(str(tmp_path / "i_ret.npz"))["row"].tolist())
    # overlap guard
    _manifests(str(tmp_path / "m3"), ret, [(0, 0, "n0", "million_aid", "x")])
    with pytest.raises(ValueError, match="overlap"):
        G.build_index(str(tmp_path / "m3"), str(tmp_path / "i3.npz"), True)


def test_build_index_one_row_per_image_and_val_never_leaks(shard0):
    """Git-10M repeats an image on adjacent rows (same bytes, another caption). The index keeps
    one row per img_name, counts the drop, and the val split never shares an image with train
    (the guard this exists for: a holdout row whose twin is a training row)."""
    tmp_path, arrow, names = shard0
    # rows 0..N-1 exist in the shard; rows 2 and 12 are twins of 1 and 9 (same img_name, same
    # bucket: 1, 2 retained; 9, 12 unassessable)
    twin = {2: names[1], 12: names[9]}
    ret = [(0, r, twin.get(r, names[r]), "10_1_1", "google_tile") for r in range(N_ROWS) if r % 3]
    una = [(0, r, twin.get(r, names[r]), "million_aid", "no coords") for r in range(N_ROWS) if r % 3 == 0]
    d = str(tmp_path / "m_twins")
    _manifests(d, ret, una)
    out = str(tmp_path / "idx_twins.npz")
    rep = G.build_index(d, out, include_unassessable=True)
    assert rep["manifest_rows"] == N_ROWS and rep["duplicate_rows_dropped"] == 2 and rep["rows"] == N_ROWS - 2
    z = np.load(out)
    rows = z["row"].tolist()
    assert 2 not in rows and 12 not in rows and 1 in rows and 9 in rows  # first occurrence kept
    assert bool(z["one_row_per_image"]) and int(z["duplicate_rows_dropped"]) == 2
    tr = G.Git10MDataset(str(arrow), out, img_size=32, which="train", holdout_every=10)
    va = G.Git10MDataset(str(arrow), out, img_size=32, which="val", holdout_every=10)
    tr_names = {tr[i]["img_name"] for i in range(len(tr))}
    va_names = {va[i]["img_name"] for i in range(len(va))}
    assert va_names and not (tr_names & va_names)
    # val = holdout rows of the RETAINED bucket only: row 0 (unassessable twin owner) is dropped from val
    assert all(r % 3 for r in va.row.tolist()) and 10 in va.row.tolist()
    # the twin that straddles buckets is refused: row 1 (retained) named like row 0 (unassessable)
    bad = [(0, 1, names[0], "10_1_1", "google_tile")], [(0, 0, names[0], "million_aid", "x")]
    _manifests(str(tmp_path / "m_cross"), *bad)
    with pytest.raises(ValueError, match="two buckets"):
        G.build_index(str(tmp_path / "m_cross"), str(tmp_path / "idx_cross.npz"), True)
    # an index built without the rule is refused by the dataset
    legacy = str(tmp_path / "legacy.npz")
    np.savez(
        legacy,
        **{
            k: z[k]
            for k in z.files
            if k not in ("one_row_per_image", "duplicate_rows_dropped", "manifest_rows")
        },
    )
    with pytest.raises(ValueError, match="one-row-per-image"):
        G.Git10MDataset(str(arrow), legacy, img_size=32, which="train")


def test_val_reference_fixes_the_val_set_and_flips_are_per_row(shard0):
    """val_ref_npz: the val rows of a run are the retained holdout of the reference index (the
    most restrictive one) whatever index the run trains on; a reference with rows missing from
    the index is refused. hflip: a function of (flip_seed, row), identical across DataLoader
    workers and resumes."""
    tmp_path, arrow, names = shard0
    big = _index(tmp_path, list(range(N_ROWS)))  # rows 0..149, holdout (every 10th) = 15 rows
    small = _index(tmp_path, list(range(0, N_ROWS, 3)))  # every 3rd row: holdout 0,30,...,120 = 5 rows
    va_big = G.Git10MDataset(str(arrow), big, img_size=32, which="val", holdout_every=10)
    va_small = G.Git10MDataset(str(arrow), small, img_size=32, which="val", holdout_every=10)
    assert len(va_big) == 15 and len(va_small) == 5
    va_ref = G.Git10MDataset(str(arrow), big, img_size=32, which="val", holdout_every=10, val_ref_npz=small)
    assert va_ref.row.tolist() == va_small.row.tolist() and va_ref.describe()["val_ref"].endswith(
        os.path.basename(small)
    )
    tr_ref = G.Git10MDataset(str(arrow), big, img_size=32, which="train", holdout_every=10, val_ref_npz=small)
    assert not set(tr_ref.row.tolist()) & set(va_ref.row.tolist())
    with pytest.raises(ValueError, match="not in"):
        G.Git10MDataset(str(arrow), small, img_size=32, which="val", holdout_every=10, val_ref_npz=big)
    a = G.Git10MDataset(
        str(arrow), big, img_size=32, hflip=True, which="train", holdout_every=10, flip_seed=3
    )
    b = G.Git10MDataset(
        str(arrow), big, img_size=32, hflip=True, which="train", holdout_every=10, flip_seed=3
    )
    c = G.Git10MDataset(
        str(arrow), big, img_size=32, hflip=True, which="train", holdout_every=10, flip_seed=4
    )
    xa = [a[i]["img"] for i in range(20)]
    assert all(torch.equal(x, b[i]["img"]) for i, x in enumerate(xa))  # same seed: same flips, any order
    plain = G.Git10MDataset(str(arrow), big, img_size=32, hflip=False, which="train", holdout_every=10)
    flipped = [not torch.equal(x, plain[i]["img"]) for i, x in enumerate(xa)]
    assert any(flipped) and not all(flipped)
    assert any(not torch.equal(x, c[i]["img"]) for i, x in enumerate(xa))  # another seed: other flips


def _index(tmp_path, rows, shard=0):
    tag = f"{shard}_{len(rows)}_{sum(rows)}"
    d = str(tmp_path / f"m_{tag}")
    _manifests(d, [(shard, r, f"n{r}", "10_1_1", "google_tile") for r in rows], [])
    out = str(tmp_path / f"idx_{tag}.npz")
    G.build_index(d, out, False)
    return out


def test_holdout_mask_is_index_independent():
    shard = np.array([0, 0, 0, 3, 3, 5])
    row = np.array([0, 10, 11, 0, 20, 10])
    full = G.holdout_mask(shard, row, 10)
    gid = shard * G.ROW_ID_STRIDE + row
    assert full.tolist() == (gid % 10 == 0).tolist() and full[:3].tolist() == [True, True, False]
    assert 0 < full.sum() < len(full)
    sub = G.holdout_mask(shard[[1, 2, 4]], row[[1, 2, 4]], 10)  # a subset index: same rows held out
    assert sub.tolist() == [full[1], full[2], full[4]]
    # a permuted index holds out the same rows (membership is a property of the row, not its position)
    perm = np.array([5, 3, 1, 4, 0, 2])
    assert G.holdout_mask(shard[perm], row[perm], 10).tolist() == full[perm].tolist()
    with pytest.raises(ValueError):
        G.holdout_mask(np.array([0]), np.array([G.ROW_ID_STRIDE]), 10)


def test_dataset_transform_holdout_and_flip(shard0):
    tmp_path, arrow, names = shard0
    idx = _index(tmp_path, list(range(N_ROWS)))
    tr = G.Git10MDataset(str(arrow), idx, img_size=32, hflip=False, which="train", holdout_every=10)
    va = G.Git10MDataset(str(arrow), idx, img_size=32, which="val", holdout_every=10)
    al = G.Git10MDataset(str(arrow), idx, img_size=32, which="all")
    assert len(va) == 15 and len(tr) == N_ROWS - 15 and len(al) == N_ROWS
    assert not set(tr.index_pos.tolist()) & set(va.index_pos.tolist())
    x = tr[0]
    assert x["img"].shape == (3, 32, 32) and x["img"].dtype == torch.float32
    assert -1.0 <= float(x["img"].min()) and float(x["img"].max()) <= 1.0
    assert x["img_name"] == names[1] and x["index"] == 1  # row 0 is the holdout
    # the transform is EuroSATDataset's: bicubic resize, (x/255 - 0.5) * 2
    ref = Image.open(io.BytesIO(_jpeg(1))).convert("RGB").resize((32, 32), Image.Resampling.BICUBIC)
    ref = (torch.from_numpy(np.asarray(ref)).permute(2, 0, 1).float() / 255.0 - 0.5) * 2
    assert torch.allclose(x["img"], ref)
    fl = G.Git10MDataset(str(arrow), idx, img_size=32, hflip=True, which="train", holdout_every=10)
    # the flip is a per-row decision: repeated reads of one row agree, rows differ, a flipped
    # row is the exact mirror of the unflipped one
    assert all(torch.equal(fl[0]["img"], fl[0]["img"]) for _ in range(5))
    seen = set()
    for i in range(20):
        same = torch.equal(fl[i]["img"], tr[i]["img"])
        assert same or torch.equal(fl[i]["img"], tr[i]["img"].flip(-1))
        seen.add(same)
    assert seen == {True, False}  # both orientations occur across rows
    d = tr.describe()
    assert d["n"] == len(tr) and d["which"] == "train" and d["buckets"] == {"retained": N_ROWS}


def test_dataset_refuses_missing_or_short_shards(shard0, tmp_path):
    _, arrow, _ = shard0
    with pytest.raises(FileNotFoundError, match="missing"):
        G.Git10MDataset(str(arrow), _index(tmp_path, [0, 1], shard=3), which="all")
    with pytest.raises(FileNotFoundError, match="too short"):
        G.Git10MDataset(str(arrow), _index(tmp_path, [0, N_ROWS + 5]), which="all")
    with pytest.raises(ValueError):
        G.Git10MDataset(str(arrow), _index(tmp_path, [0, 1]), which="test")


def test_permutation_sampler_resumes_exactly():
    full = list(G.PermutationSampler(20, seed=7))
    assert sorted(full) == list(range(20))
    s = G.PermutationSampler(20, seed=7, epoch=0, pos=13)
    assert list(s) == full[13:]
    assert s.epoch == 1 and s.pos == 0
    assert list(s) == list(G.PermutationSampler(20, seed=7, epoch=1))  # next pass, from 0
    assert list(G.PermutationSampler(20, seed=8)) != full


def test_cli_check(shard0, capsys):
    tmp_path, arrow, _ = shard0
    idx = _index(tmp_path, [0, 5, 64, N_ROWS - 1])
    assert G.main(["check", "--arrow-dir", str(arrow), "--index", idx]) == 0
    assert "GIT10M_CHECK_OK" in capsys.readouterr().out


def test_module_cli_runs_as_main():
    """`python -m datasets.git10m ...` (train.sbatch's index check, convert, build-index): the
    package import registers the module once and the __main__ copy must delegate, not register
    "git10m" a second time (smoke job 6405496, 2026-10-14: "Dataset 'git10m' is already registered")."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-m", "datasets.git10m", "check", "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": os.path.join(root, "src")},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0 and "already registered" not in r.stderr, r.stderr[-800:]
    assert "--index" in r.stdout
