"""Residual content overlap (eval/content_overlap.py) end to end on a synthetic shard, a fake
backbone and a fake GEO-Bench feature cache: planted copies are found, random images are not,
the null-calibrated thresholds behave, and every gate is fed the fault it exists to catch
(rule 6): a reference cache from a different embedding path, overlapping parts, a misaddressed
row. No GPU, no downloads."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import git10m as G10  # noqa: E402
from eval import content_overlap as C  # noqa: E402
from eval import features as F  # noqa: E402


# ----------------------------------------------------------------------------- fixtures
class FakeBackbone:
    """Deterministic 'embedding': the 6 x 6 mean-pooled grey image (36 dims) plus a colour
    histogram (12 dims). Identical pixels -> cos 1; flips / crops stay close; random -> low."""

    def forward_features(self, x):
        g = x.mean(1, keepdim=True)
        pooled = torch.nn.functional.adaptive_avg_pool2d(g, 6).flatten(1)
        hist = torch.cat(
            [
                torch.histc(x[:, k].flatten(1)[i], bins=4, min=-3, max=3)
                for i in range(len(x))
                for k in range(3)
            ]
        ).view(len(x), 12)
        return {"x_norm_clstoken": torch.cat([pooled, hist / x[0, 0].numel()], 1)}


SPEC = {"norm": ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), "dtype": torch.float32, "batch": 8}


def _img(seed: int, size=(48, 40)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (3, 3, 3), dtype=np.uint8)  # smooth-ish: 3x3 upsampled + noise
    im = Image.fromarray(base).resize((size[0], size[1]), Image.Resampling.BICUBIC)
    arr = np.clip(np.array(im).astype(int) + rng.integers(-20, 20, (size[1], size[0], 3)), 0, 255).astype(
        np.uint8
    )
    return Image.fromarray(arr)


def _jpeg(im: Image.Image, q=95) -> bytes:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return b.getvalue()


def _stream_shard(path: str, items: list[tuple[str, bytes]]):
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
        for s in range(0, len(items), 70):
            chunk = items[s : s + 70]
            w.write_batch(
                pa.RecordBatch.from_pydict(
                    {
                        "image": [{"bytes": b, "path": None} for _, b in chunk],
                        "text": ["t"] * len(chunk),
                        "img_name": [n for n, _ in chunk],
                        "img_quality_score": [3.0] * len(chunk),
                        "caption": ["c"] * len(chunk),
                        "caption_length": [1] * len(chunk),
                        "resolution": ["1.0"] * len(chunk),
                        "Google_location": [None] * len(chunk),
                    },
                    schema=schema,
                )
            )
        w.close()


def _embed_files(files):
    return C.embed_images(FakeBackbone(), SPEC, (Image.open(f) for f in files), "cpu", 8)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two fake GEO-Bench tasks (train/val/test PNG trees + feature cache from the SAME fake
    backbone), one converted Git-10M shard holding: 1200 random 'retained' images (the null),
    40 'excluded' images, and an 'unassessable' set = 300 random images + 12 planted copies of
    reference images (6 identical JPEG re-encodes, 6 flipped) with 3 names repeated."""
    monkeypatch.setattr(C, "load_backbone", lambda preset, device: (FakeBackbone(), SPEC))
    official = {}
    ref_files = {}
    data = tmp_path / "data"
    for t, (tr, va, te) in {"t1": (24, 8, 8), "t2": (16, 6, 6)}.items():
        official[t] = {
            "sizes": {"train": tr, "val": va, "test": te},
            "root": str(data / f"{t}_rgb"),
            "classes": lambda: ["a", "b"],
        }
        k = 0
        for split, n in zip(("train", "val", "test"), (tr, va, te)):
            for cls in ("a", "b"):
                d = data / f"{t}_rgb" / split / cls
                d.mkdir(parents=True)
                for i in range(n // 2):
                    _img(hash((t, k)) % 10_000).save(d / f"{t}_{split}_{k:03d}.png")
                    k += 1
    # one all-white (no-data) reference image: a degenerate row the filter must drop
    Image.new("RGB", (48, 40), (255, 255, 255)).save(data / "t1_rgb" / "val" / "a" / "t1_val_024.png")
    monkeypatch.setattr(F, "OFFICIAL", official)
    ref_dir = tmp_path / "dino_feats"
    ref_dir.mkdir()
    for t in official:
        for split in ("train", "val", "test"):
            files, y = F.list_official_split(t, split)
            ref_files[(t, split)] = files
            np.savez(
                ref_dir / f"fake_{t}_{split}.npz",
                cls=_embed_files(files),
                paths=np.array(files),
                labels=y,
                preset=np.array("fake"),
            )
    # the shard
    items, man = [], {"retained": [], "excluded": [], "unassessable": []}
    for i in range(1200):
        items.append((f"Google-ret-{i}.jpg", _jpeg(_img(100_000 + i))))
        man["retained"].append(("google_tile", items[-1][0]))
    for i in range(3):  # all-white retained tiles: degenerate, must not set tau
        items.append((f"Sky-white-{i}.jpg", _jpeg(Image.new("RGB", (64, 48), (255, 255, 255)))))
        man["retained"].append(("skyscript", items[-1][0]))
    for i in range(40):
        items.append((f"Google-exc-{i}.jpg", _jpeg(_img(200_000 + i))))
        man["excluded"].append(("google_tile", items[-1][0]))
    for i in range(10):  # excluded by a So2Sat city ROI only: not a same-location reference row
        items.append((f"Google-city-{i}.jpg", _jpeg(_img(250_000 + i))))
        man["excluded"].append(("google_tile", items[-1][0]))
    for i in range(300):
        items.append((f"Million-AID-{i}.jpg", _jpeg(_img(300_000 + i))))
        man["unassessable"].append(("million_aid", items[-1][0]))
    for i in range(2):  # all-white queries: they would 'hit' the white reference image at cos 1.0
        items.append((f"Million-AID-white-{i}.jpg", _jpeg(Image.new("RGB", (32, 32), (255, 255, 255)))))
        man["unassessable"].append(("million_aid", items[-1][0]))
    planted = ref_files[("t1", "test")][:3] + ref_files[("t2", "train")][:3]
    for j, f in enumerate(planted):
        items.append((f"GeoPileV0-copy-{j}.png", _jpeg(Image.open(f))))
        man["unassessable"].append(("geopile", items[-1][0]))
        items.append(
            (f"GeoPileV0-flip-{j}.png", _jpeg(Image.open(f).transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
        )
        man["unassessable"].append(("geopile", items[-1][0]))
    for j in range(3):  # repeated names on adjacent rows (one row per caption)
        items.append((f"GeoPileV0-copy-{j}.png", _jpeg(Image.open(planted[j]))))
        man["unassessable"].append(("geopile", items[-1][0]))
    stream = tmp_path / "stream"
    stream.mkdir()
    _stream_shard(str(stream / G10.shard_name(0)), items)
    arrow = tmp_path / "arrow"
    arrow.mkdir()
    G10.convert_shard(str(stream / G10.shard_name(0)), str(arrow / G10.shard_name(0)))
    mdir = tmp_path / "manifest"
    mdir.mkdir()
    row_of = {}
    for r, (n, _) in enumerate(items):
        row_of.setdefault(n, []).append(r)
    for key, rows in man.items():
        used = {}
        recs = []
        for src, name in rows:
            r = row_of[name][used.get(name, 0)]
            used[name] = used.get(name, 0) + 1
            ids = ("city:Sydney" if "city" in name else "id_1;city:Sydney") if key == "excluded" else ""
            recs.append((0, r, name, src, ids))
        pd.DataFrame(recs, columns=["shard", "row", "img_name", "source", "matched_ids"]).to_csv(
            mdir / f"{key}.csv", index=False
        )
    return dict(
        tmp=tmp_path,
        arrow=str(arrow),
        manifest=str(mdir),
        ref_dir=str(ref_dir),
        planted=planted,
        ref_files=ref_files,
    )


# ----------------------------------------------------------------------------- units
def test_variants_and_tensor_shape():
    im = _img(1)
    for k in C.VARIANTS:
        v = C.variant(im, k)
        assert isinstance(v, Image.Image)
    assert C.variant(im, "crop80").size == (38, 32) and C.variant(im, "half").size == (24, 20)
    x = C.to_tensor(im, *SPEC["norm"])
    assert x.shape == (3, C.IMG, C.IMG)
    with pytest.raises(ValueError):
        C.variant(im, "rot")


def test_select_rows_is_deterministic_and_parts_tile():
    df = pd.DataFrame(
        {"shard": 0, "row": np.arange(1000), "source": "s", "img_name": [f"n{i}" for i in range(1000)]}
    )
    a = C.select_rows(df, 100, 0, None)
    b = C.select_rows(df, 100, 0, None)
    assert a.equals(b) and len(a) == 100 and (np.diff(a["row"]) > 0).all()
    assert not C.select_rows(df, 100, 1, None).equals(a)
    parts = [C.select_rows(df, None, 0, f"{i}/3") for i in range(3)]
    assert sum(len(p) for p in parts) == 1000 and pd.concat(parts)["row"].tolist() == list(range(1000))
    with pytest.raises(SystemExit, match="--sample"):
        C.select_rows(df, 1001, 0, None)
    with pytest.raises(SystemExit, match="--part"):
        C.select_rows(df, None, 0, "3/3")


def test_max_cosine_and_thresholds():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(50, 8)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    q = np.concatenate([ref[[3, 7]] * 5.0, rng.normal(size=(3, 8))])
    m, arg = C.max_cosine(q, ref, chunk=2)
    assert arg[:2].tolist() == [3, 7] and np.allclose(m[:2], 1.0, atol=1e-6) and (m[2:] < 0.999).all()
    null = rng.uniform(0, 0.8, 5000)
    taus = C.thresholds(null, (1e-2, 1e-3))
    assert 0.78 < taus["0.01"] < taus["0.001"] <= 0.8
    with pytest.raises(SystemExit, match="need >= 1000"):
        C.thresholds(null[:999], (1e-3,))


# ----------------------------------------------------------------------------- end to end
def _run(world, tmp, fprs="0.01 0.001", redo_ref=None):
    man, arrow = world["manifest"], world["arrow"]
    out = tmp / "emb"
    out.mkdir(exist_ok=True)
    C.main(
        [
            "embed",
            "--preset",
            "fake",
            "--arrow-dir",
            arrow,
            "--manifest",
            f"{man}/retained.csv",
            "--sample",
            "1100",
            "--seed",
            "0",
            "--workers",
            "0",
            "--out",
            str(out / "null.npz"),
        ]
    )
    C.main(
        [
            "embed",
            "--preset",
            "fake",
            "--arrow-dir",
            arrow,
            "--manifest",
            f"{man}/excluded.csv",
            "--drop-city-only",
            "--workers",
            "0",
            "--out",
            str(out / "excl.npz"),
        ]
    )
    for i in range(2):
        C.main(
            [
                "embed",
                "--preset",
                "fake",
                "--arrow-dir",
                arrow,
                "--manifest",
                f"{man}/unassessable.csv",
                "--unique-names",
                "--part",
                f"{i}/2",
                "--workers",
                "0",
                "--out",
                str(out / f"q_p{i}of2.npz"),
            ]
        )
    C.main(["controls", "--preset", "fake", "--per-task", "4", "--out", str(out / "ctrl.npz")])
    rep_dir = tmp / "rep"
    C.main(
        [
            "match",
            "--preset",
            "fake",
            "--ref-dir",
            redo_ref or world["ref_dir"],
            "--null",
            str(out / "null.npz"),
            "--controls",
            str(out / "ctrl.npz"),
            "--excluded",
            str(out / "excl.npz"),
            "--query",
            str(out / "q_p*.npz"),
            "--fprs",
            *fprs.split(),
            "--out-dir",
            str(rep_dir),
        ]
    )
    return out, rep_dir


def test_end_to_end_finds_planted_copies_and_only_those(world):
    out, rep_dir = _run(world, world["tmp"])
    rep = json.load(open(rep_dir / "report.json"))
    assert rep["status"] == "OK" and all(g["pass"] for g in rep["gates"].values())
    assert rep["queries"]["n"] == 300 + 12  # repeated names collapsed, parts disjoint, 2 white dropped
    d = rep["degenerate"]
    assert d["reference"] == 1 and d["reference_rows"] == ["t1/val/t1_val_024.png"] and d["controls"] == 0
    assert d["queries"] == 2 and d["queries_per_source"] == {"million_aid": 2} and d["excluded"] == 0
    assert 1 <= d["null"] <= 3 and d["null_per_source"] == {"google_tile": 0, "skyscript": d["null"]}
    assert all(t < 0.999 for t in rep["taus"].values())  # the white pairs no longer saturate tau
    assert rep["queries"]["per_source_n"] == {"geopile": 12, "million_aid": 300}
    g = rep["gates"]["identity_reproduces_cache"]
    assert g["n"] == 8 and g["missing_reference"] == 0 and g["min_cos"] > 0.999
    for f, h in rep["hits"].items():
        assert (
            h["expected_false_hits"] == pytest.approx(float(f) * 312)
            and h["excess_hits"] == h["hits"] - h["expected_false_hits"]
        )
        assert h["null_rows_above_tau"] >= 1 and h["null_rows_above_tau"] <= max(2, int(float(f) * 1100) + 1)
        assert h["per_source"]["geopile"] >= 6, (f, h)  # every identical copy is found at every tau
        # random images score at about the null's false-positive rate: 300 x fpr expected, Poisson-wide
        assert h["per_source"]["million_aid"] <= {"0.01": 10, "0.001": 4}[f], (f, h)
        assert h["per_nearest_task"]["t1"] >= 3 and h["per_nearest_task"]["t2"] >= 3
    assert rep["controls"]["per_variant_recall"]["identity"] == {f: 1.0 for f in rep["taus"]}
    hits = pd.read_csv(rep_dir / "hits.csv")  # at the loosest tau: every identical copy listed, best first
    assert hits["img_name"].str.contains("copy").sum() == 6 and hits["max_sim"].is_monotonic_decreasing
    z = np.load(rep_dir / "query_max_sim.npz")
    assert len(z["max_sim"]) == 312 and len(set(z["img_name"])) == 312
    assert (
        len(z["null_max_sim"]) == 1100 - d["null"]
    )  # the degenerate null rows are gone from the vectors too
    assert set(rep["controls"]["per_variant_recall"]) == set(
        C.VARIANTS
    )  # recorded per variant; values are instrument-specific
    # reports are re-readable prose
    md = (rep_dir / "report.md").read_text()
    assert "Gates: identity_reproduces_cache PASS" in md and "| 0.001 |" in md and "expected false" in md
    assert "Degenerate (uniform, no-data) images dropped" in md and "queries 2 (million_aid 2)" in md


def test_match_refuses_controls_without_blank_variants(world, monkeypatch):
    """A controls file from before the degenerate filter has no synthetic blanks; match must refuse
    rather than silently run with no filter (the white pairs would saturate tau again)."""
    out, rep_dir = _run(world, world["tmp"])
    old = out / "ctrl_old.npz"
    with monkeypatch.context() as m:  # a controls file written by the pre-filter code
        m.setattr(C, "BLANK_VARIANTS", {})
        C.main(["controls", "--preset", "fake", "--per-task", "2", "--out", str(old)])
    assert not np.isin(np.load(old)["variant"], list(C.BLANK_VARIANTS)).any()
    with pytest.raises(SystemExit, match="rerun `controls`"):
        C.main(
            [
                "match",
                "--preset",
                "fake",
                "--ref-dir",
                world["ref_dir"],
                "--null",
                str(out / "null.npz"),
                "--controls",
                str(old),
                "--query",
                str(out / "q_p*.npz"),
                "--out-dir",
                str(world["tmp"] / "rep_old"),
            ]
        )


def test_gate_fires_when_reference_cache_comes_from_another_path(world, tmp_path):
    """A cache whose embeddings do not match this path (here: shuffled rows) must FAIL the
    identity gate and refuse to report counts."""
    bad = tmp_path / "bad_ref"
    bad.mkdir()
    for f in os.listdir(world["ref_dir"]):
        z = np.load(os.path.join(world["ref_dir"], f))
        d = dict(z)
        d["cls"] = d["cls"][::-1].copy()
        np.savez(bad / f, **d)
    with pytest.raises(SystemExit, match="instrument gate FAILED"):
        _run(world, world["tmp"], redo_ref=str(bad))
    rep = json.load(open(world["tmp"] / "rep" / "report.json"))
    assert rep["status"] == "FAIL" and not rep["gates"]["identity_reproduces_cache"]["pass"]


def test_embed_refuses_overlapping_parts_and_misaddressed_rows(world, tmp_path):
    man, arrow = world["manifest"], world["arrow"]
    a = tmp_path / "a.npz"
    C.main(
        [
            "embed",
            "--preset",
            "fake",
            "--arrow-dir",
            arrow,
            "--manifest",
            f"{man}/unassessable.csv",
            "--unique-names",
            "--part",
            "0/2",
            "--workers",
            "0",
            "--out",
            str(a),
        ]
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        C.main(
            [
                "embed",
                "--preset",
                "fake",
                "--arrow-dir",
                arrow,
                "--manifest",
                f"{man}/unassessable.csv",
                "--part",
                "0/2",
                "--workers",
                "0",
                "--out",
                str(a),
            ]
        )
    # a manifest whose img_name does not match the shard row -> fatal before any embedding is written
    m = pd.read_csv(f"{man}/unassessable.csv")
    m.loc[0, "img_name"] = "not-this-image.jpg"
    m.to_csv(tmp_path / "wrong.csv", index=False)
    with pytest.raises(SystemExit, match="!= manifest"):
        C.main(
            [
                "embed",
                "--preset",
                "fake",
                "--arrow-dir",
                arrow,
                "--manifest",
                str(tmp_path / "wrong.csv"),
                "--max-rows",
                "4",
                "--workers",
                "0",
                "--out",
                str(tmp_path / "w.npz"),
            ]
        )
    assert not (tmp_path / "w.npz").exists()
    # query parts that repeat an image are refused by match
    ref_args = ["match", "--preset", "fake", "--ref-dir", world["ref_dir"]]
    nul = tmp_path / "null.npz"
    C.main(
        [
            "embed",
            "--preset",
            "fake",
            "--arrow-dir",
            arrow,
            "--manifest",
            f"{man}/retained.csv",
            "--sample",
            "1010",  # >= 1000 null rows remain after the 3 white rows are dropped
            "--workers",
            "0",
            "--out",
            str(nul),
        ]
    )
    ctl = tmp_path / "ctrl.npz"
    C.main(["controls", "--preset", "fake", "--per-task", "2", "--out", str(ctl)])
    with pytest.raises(SystemExit, match="parts overlap|already loaded"):
        C.main(
            ref_args
            + [
                "--null",
                str(nul),
                "--controls",
                str(ctl),
                "--query",
                str(a),
                str(a),
                "--out-dir",
                str(tmp_path / "r"),
            ]
        )


def test_reference_loader_validates_identity(world, tmp_path):
    with pytest.raises(SystemExit, match="missing: extract it first"):
        C.load_reference(world["ref_dir"], "other", ["t1"])
    renamed = tmp_path / "renamed"  # files named for one preset holding another's features
    renamed.mkdir()
    for f in os.listdir(world["ref_dir"]):
        np.savez(renamed / f.replace("fake_", "other_"), **dict(np.load(os.path.join(world["ref_dir"], f))))
    with pytest.raises(SystemExit, match="preset fake != other"):
        C.load_reference(str(renamed), "other", ["t1"])
    bad = tmp_path / "short"
    bad.mkdir()
    for f in os.listdir(world["ref_dir"]):
        z = dict(np.load(os.path.join(world["ref_dir"], f)))
        if f == "fake_t1_val.npz":
            z["cls"], z["paths"], z["labels"] = z["cls"][:-1], z["paths"][:-1], z["labels"][:-1]
        np.savez(bad / f, **z)
    with pytest.raises(SystemExit, match="OFFICIAL says"):
        C.load_reference(str(bad), "fake", ["t1"])
    ref, task, split, path, sha = C.load_reference(world["ref_dir"], "fake", ["t1", "t2"])
    assert ref.shape == (24 + 8 + 8 + 16 + 6 + 6, ref.shape[1]) and np.allclose(
        np.linalg.norm(ref, axis=1), 1, atol=1e-5
    )
    assert len(sha) == 16 and (task == "t2").sum() == 28


def test_reference_drops_city_only_rows_and_match_checks_origins(world, tmp_path):
    """excluded.csv of a city pass: rows matched by a city ROI alone are not same-location rows
    (71 % of dedup_v3_b1000_cities/excluded.csv) and are dropped by --drop-city-only; match
    refuses a reference embedded without it, a null from another manifest dir, a missing query
    part, and a part from another manifest (every file's `origin` is checked)."""
    man = world["manifest"]
    df = C.rows_from_manifest(f"{man}/excluded.csv", False, drop_city_only=True)
    assert len(df) == 40 and not df["img_name"].str.contains("city").any()
    assert len(C.rows_from_manifest(f"{man}/excluded.csv", False)) == 50
    out, rep_dir = _run(world, tmp_path)
    rep = json.load(open(rep_dir / "report.json"))
    assert rep["excluded_same_location"]["n"] == 40 and rep["manifest_dir"] == man
    assert rep["queries"]["origin"]["n_all"] == len(C.rows_from_manifest(f"{man}/unassessable.csv", True))
    base = ["match", "--preset", "fake", "--ref-dir", world["ref_dir"], "--controls", str(out / "ctrl.npz")]
    # reference without --drop-city-only
    C.main(
        [
            "embed",
            "--preset",
            "fake",
            "--arrow-dir",
            world["arrow"],
            "--manifest",
            f"{man}/excluded.csv",
            "--workers",
            "0",
            "--out",
            str(tmp_path / "excl_raw.npz"),
        ]
    )
    with pytest.raises(SystemExit, match="drop-city-only"):
        C.main(
            base
            + [
                "--null",
                str(out / "null.npz"),
                "--excluded",
                str(tmp_path / "excl_raw.npz"),
                "--query",
                str(out / "q_p*.npz"),
                "--out-dir",
                str(tmp_path / "r1"),
            ]
        )
    # a missing part
    with pytest.raises(SystemExit, match="do not tile"):
        C.main(
            base
            + [
                "--null",
                str(out / "null.npz"),
                "--query",
                str(out / "q_p0of2.npz"),
                "--out-dir",
                str(tmp_path / "r2"),
            ]
        )
    # null from another manifest dir (a copy of the manifests elsewhere)
    other = tmp_path / "other_manifest"
    shutil.copytree(man, other)
    C.main(
        [
            "embed",
            "--preset",
            "fake",
            "--arrow-dir",
            world["arrow"],
            "--manifest",
            f"{other}/retained.csv",
            "--sample",
            "1010",  # >= 1000 null rows remain after the 3 white rows are dropped
            "--seed",
            "0",
            "--workers",
            "0",
            "--out",
            str(tmp_path / "null_other.npz"),
        ]
    )
    with pytest.raises(SystemExit, match="queries from .* but null from"):
        C.main(
            base
            + [
                "--null",
                str(tmp_path / "null_other.npz"),
                "--query",
                str(out / "q_p*.npz"),
                "--out-dir",
                str(tmp_path / "r3"),
            ]
        )
    # a null embedded from the wrong manifest file
    with pytest.raises(SystemExit, match="must be embedded from retained.csv"):
        C.main(
            base
            + [
                "--null",
                str(out / "excl.npz"),
                "--query",
                str(out / "q_p*.npz"),
                "--out-dir",
                str(tmp_path / "r4"),
            ]
        )
