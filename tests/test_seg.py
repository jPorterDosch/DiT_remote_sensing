"""Segmentation harness, CPU only: the exporter (fake geobench), the token-extraction task
(fake model), the per-token DiTF, the protocol pieces of eval.seg_probe and a full synthetic
run where the tokens carry the mask (mIoU must reach ~1 and selection must be val-only)."""

from __future__ import annotations

import json
import os
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

from eval import export_geobench as E  # noqa: E402
from eval import features as F  # noqa: E402
from eval import seg_probe as SP  # noqa: E402

GRID = (4, 4)  # token grid of the fake model
C = 8
MASK_HW = (8, 8)


# ------------------------------------------------------------------------ registry fixture
@pytest.fixture
def seg_registry(monkeypatch, tmp_path):
    root = tmp_path / "tree"
    monkeypatch.setitem(
        F.OFFICIAL_SEG,
        "m_fakeseg",
        {
            "sizes": {"train": 8, "val": 4, "test": 4},
            "root": str(root),
            "classes": ["a", "b", "c"],
            "mask_hw": MASK_HW,
        },
    )
    return root


def _write_tree(root, sizes, seed=0):
    """images whose top-left 2x2 token block color encodes the mask class layout (see _fake_tokens)."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in sizes.items():
        os.makedirs(root / split / "images", exist_ok=True)
        os.makedirs(root / split / "masks", exist_ok=True)
        masks = []
        for i in range(n):
            px = rng.integers(0, 255, (*MASK_HW, 3), dtype=np.uint8)
            Image.fromarray(px).save(root / split / "images" / f"{split}_{i:03d}.png")
            m = rng.integers(0, 3, (2, 2)).astype(np.uint8)  # 2 x 2 blocks of 4 x 4 pixels
            mask = np.kron(m, np.ones((4, 4), np.uint8))
            if i == 0:
                mask[0, 0] = F.SEG_IGNORE
            Image.fromarray(mask).save(root / split / "masks" / f"{split}_{i:03d}.png")
            masks.append(mask)
        out[split] = np.stack(masks)
    return out


# ------------------------------------------------------------------------------ exporter
class _Band:
    def __init__(self, data):
        self.data = data


class _Sample:
    def __init__(self, i, image, mask):
        self.sample_name = f"s{i:03d}"
        self._image = image
        self.label = _Band(mask)

    def pack_to_3d(self, band_names):
        return self._image, band_names


def test_export_seg_roundtrip_and_manifest(tmp_path, monkeypatch, seg_registry):  # names shipped
    names = ("a", "b", "c", "nodata")
    lt = type("SegmentationClasses", (), {})()
    lt.class_names = list(names)
    lt.n_classes = len(names)

    def dataset(_dir, split, partition_name):
        n = {"train": 8, "valid": 4, "test": 4}[split]
        rng = np.random.default_rng(len(split))
        return [
            _Sample(i, np.full((*MASK_HW, 3), 50.0, np.float32), rng.integers(0, 4, MASK_HW).astype(np.uint8))
            for i in range(n)
        ]

    mod = types.SimpleNamespace(
        load_task_specs=lambda d: types.SimpleNamespace(label_type=lt), GeobenchDataset=dataset
    )
    monkeypatch.setitem(sys.modules, "geobench", mod)
    monkeypatch.setitem(
        E.SEG_SPECS,
        "m-fakeseg",
        E.SegSpec(
            zenodo=0,
            rgb_bands=("r", "g", "b"),
            scale=100.0,
            img_hw=MASK_HW,
            classes=names,
            ignore=("nodata",),
        ),
    )
    man = E.export_seg("m-fakeseg", "unused", str(seg_registry))
    images, masks = F.list_official_seg_split("m_fakeseg", "train")
    assert len(images) == 8 and all(os.path.exists(m) for m in masks)
    mk = np.asarray(Image.open(masks[0]))
    assert set(np.unique(mk)) <= {0, 1, 2, F.SEG_IGNORE} and (mk == F.SEG_IGNORE).any()  # class 3 -> ignore
    assert sum(man["splits"]["train"]["pixel_counts"]) == 8 * 64
    monkeypatch.setattr(E, "manifest_path", lambda t: str(tmp_path / "man.json"))
    E.check_manifest("m-fakeseg", man, write=True)
    E.check_manifest("m-fakeseg", E.seg_tree_manifest("m_fakeseg", str(seg_registry)))
    # a wrong class list in the spec is refused before anything is written
    monkeypatch.setitem(
        E.SEG_SPECS,
        "m-fakeseg",
        E.SegSpec(
            zenodo=0,
            rgb_bands=("r", "g", "b"),
            scale=100.0,
            img_hw=MASK_HW,
            classes=("a", "b", "c"),
            ignore=(),
        ),
    )
    with pytest.raises(SystemExit, match="shipped class names"):
        E.export_seg("m-fakeseg", "unused", str(tmp_path / "again"))
    # an artifact without names (m-chesapeake-landcover ships class_names=None): only the count is checkable
    lt.class_names = None
    with pytest.raises(SystemExit, match="n_classes"):
        E.export_seg("m-fakeseg", "unused", str(tmp_path / "again"))
    monkeypatch.setitem(
        E.SEG_SPECS,
        "m-fakeseg",
        E.SegSpec(
            zenodo=0,
            rgb_bands=("r", "g", "b"),
            scale=100.0,
            img_hw=MASK_HW,
            classes=names,
            ignore=("nodata",),
        ),
    )
    assert E.export_seg("m-fakeseg", "unused", str(tmp_path / "noname"))["splits"]["train"]["n"] == 8


# ------------------------------------------------------------------------ token extraction
class FakeModel:
    """FluxModel.extract_raw stand-in: multi-block (B, C, h, w) features from (latents, eps, t)."""

    def eval(self):  # the adapter interface the task calls (FluxModel.eval / GeoCoreModel.eval)
        return self

    @torch.no_grad()
    def extract_raw(
        self, img, timestep, block_idx, ensemble_size, latents=None, noise=None, generator=None, **kw
    ):
        blocks = [block_idx] if isinstance(block_idx, int) else list(block_idx)
        shape = (ensemble_size, C, *GRID)
        if latents is None:
            latents = img.flatten()[0] + torch.randn(shape)
        if noise is None:
            noise = torch.randn(shape, generator=generator)
        t = timestep / 1000
        base = (t * noise + (1.0 - t) * latents).mean(0, keepdim=True)
        feat = torch.cat([base + k for k in blocks], dim=0)  # B, C, h, w
        ada = torch.stack([torch.full((3, C), t + k) for k in blocks])  # B, 3, C
        return feat, ada, latents, noise


def _run_cfg(tmp_path, split="train", t=(100, 260), k=(12, 28), max_images=None):
    return SimpleNamespace(
        t=list(t), k=list(k), img_size=[64, 64], extraction_mode=__import__("config_types").ExtractionMode.ONESHOT,
        label_fraction=1.0, model=SimpleNamespace(ensemble_size=2), extract_split=split, device="cpu",
        guidance_scale=1.0, max_images=max_images, num_shards=1, batch_size=2, num_workers=0, eps_seed=None,
        seed={"train": 42, "test": 43, "val": 44}[split], save_dir=str(tmp_path / "tok"), lora_checkpoint="",
        dataset=SimpleNamespace(name="m_fakeseg", path=None), max_samples=None, subset_size=None,
    )  # fmt: skip


def test_seg_tokens_task_writes_one_memmap_per_t_k_in_listing_order(tmp_path, seg_registry, monkeypatch):
    from datasets import geobench_seg as GS
    from tasks import seg_tokens as ST

    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    monkeypatch.setenv("FLUX_RANDOM_INIT", "")
    cfg = _run_cfg(tmp_path)
    # the module registers one wrapper per OFFICIAL_SEG entry at import time, which may predate
    # the fixture's m_fakeseg entry when another test imported `datasets` first
    ds = GS._wrapper("m_fakeseg")(cfg)
    meta = ST.SegTokenExtractionTask().run(cfg, FakeModel(), ds)
    assert meta["n"] == 8 and meta["grid_hw"] == list(GRID) and meta["weights"] == "flux-dev"
    tag = meta["tag"]
    assert tag == "oneshot_g1.0_ens2_img64"
    for t in (100, 260):
        for k in (12, 28):
            arr = np.load(ST.token_path(cfg.save_dir, "train", t, k, tag), mmap_mode="r")
            assert arr.shape == (8, 16, C) and arr.dtype == np.float16
    d = np.load(ST.meta_path(cfg.save_dir, "train", tag))
    images, masks = F.list_official_seg_split("m_fakeseg", "train")
    assert [str(p) for p in d["paths"]] == images and [str(m) for m in d["masks"]] == masks
    assert d["mods"].shape == (2, 2, 3, C) and float(d["mods"][1, 1, 0, 0]) == pytest.approx(0.26 + 28)
    # block 28 - block 12 = 16 everywhere (the fake model's block offset survives the (B, C, h, w) -> (L, C) layout)
    a = np.load(ST.token_path(cfg.save_dir, "train", 100, 12, tag), mmap_mode="r").astype(np.float32)
    b = np.load(ST.token_path(cfg.save_dir, "train", 100, 28, tag), mmap_mode="r").astype(np.float32)
    assert np.allclose(b - a, 16.0, atol=0.05)
    with pytest.raises(FileExistsError):
        ST.SegTokenExtractionTask().run(cfg, FakeModel(), ds)
    smoke = ST.SegTokenExtractionTask().run(_run_cfg(tmp_path / "s", max_images=3), FakeModel(), ds)
    assert smoke["n"] == 3 and smoke["tag"].endswith("_smoke3") and smoke["max_images"] == 3


def test_flux_token_source_finds_the_run_dir_per_split(tmp_path, seg_registry, monkeypatch):
    """run.py writes each split's cache into save_dir/<run_name>/; TokenSource globs one level down
    and refuses a second run dir carrying the same tag."""
    from datasets import geobench_seg as GS
    from tasks import seg_tokens as ST

    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    tok = tmp_path / "flux"
    for split in ("train", "val", "test"):
        cfg = _run_cfg(tmp_path, split=split)
        cfg.save_dir = str(tok / f"m_fakeseg_flux_abc+{cfg.seed}")
        tag = ST.SegTokenExtractionTask().run(cfg, FakeModel(), GS._wrapper("m_fakeseg")(cfg))["tag"]
    model = {"name": "f", "kind": "flux", "tokens": str(tok), "tag": tag, "timesteps": [100, 260], "blocks": [12, 28],
             "expect": {"weights": "flux-dev", "ensemble_size": 2}}  # fmt: skip
    src = SP.TokenSource(model, "m_fakeseg")
    assert sorted(src.files) == ["t100_k12", "t100_k28", "t260_k12", "t260_k28"]
    assert src.files["t100_k12"]["val"].startswith(str(tok / "m_fakeseg_flux_abc+44"))
    assert src._paths("test") == F.list_official_seg_split("m_fakeseg", "test")[0]
    assert src.mod_for("t260_k28", "train").shape == (3, C)
    assert float(src.mod_for("t260_k28", "train")[0, 0]) == pytest.approx(
        0.26 + 28
    )  # (t, k) row order, not (k, t)
    assert float(src.mod_for("t100_k12", "val")[2, 3]) == pytest.approx(0.1 + 12)
    assert sorted(src.input_roles()) == sorted(
        [f"{c}/{s}" for c in src.files for s in ("train", "val", "test")]
        + [f"meta/{s}" for s in ("train", "val", "test")]
    )
    monkeypatch.chdir(tmp_path)  # a repo-relative tokens dir (the real specs) is stored absolute
    rel = SP.TokenSource({**model, "tokens": "flux"}, "m_fakeseg")
    assert all(os.path.isabs(q) for q in rel.input_files()) and rel.input_files() == src.input_files()
    with pytest.raises(SystemExit, match="0 matches"):
        SP.TokenSource({**model, "tokens": str(tmp_path / "nowhere")}, "m_fakeseg")
    dup = tok / "stale_run"
    dup.mkdir()
    (dup / f"seg_tokens_val_{tag}_meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="2 matches"):
        SP.TokenSource(model, "m_fakeseg")


def test_flux_cell_reuse_binds_each_candidate_file_and_the_mods(tmp_path, seg_registry, monkeypatch):
    """Role-keyed identity: swapping two candidate files (same bytes, permuted) or rewriting the
    adaLN mods must refuse the cells; a json t-list that no longer matches the npz is refused."""
    from datasets import geobench_seg as GS
    from tasks import seg_tokens as ST

    orig = SP.ditf_tokens  # the fake model has C = 8 channels: no DiTF massive-activation channels to zero
    monkeypatch.setattr(SP, "ditf_tokens", lambda x, mod, discard=None: orig(x, mod, discard=[]))
    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    tok = tmp_path / "flux"
    for split in ("train", "val", "test"):
        cfg = _run_cfg(tmp_path, split=split)
        cfg.save_dir = str(tok / f"m_fakeseg_flux_abc+{cfg.seed}")
        tag = ST.SegTokenExtractionTask().run(cfg, FakeModel(), GS._wrapper("m_fakeseg")(cfg))["tag"]
    model = {"name": "f", "kind": "flux", "tokens": str(tok), "tag": tag, "timesteps": [100], "blocks": [12, 28],
             "expect": {"weights": "flux-dev"}, "teacher": None}  # fmt: skip
    spec = {
        "fractions": [1.0],
        "seeds": [0],
        "lr_grid": [1e-2],
        "steps": 5,
        "batch_images": 4,
        "datasets": ["m_fakeseg"],
        "models": [model],
    }
    out = str(tmp_path / "res")
    run_dir = tok / "m_fakeseg_flux_abc+42"
    a, b = run_dir / f"seg_tokens_train_t100_k12_{tag}.npy", run_dir / f"seg_tokens_train_t100_k28_{tag}.npy"
    T = time.time_ns() - 10**9
    os.utime(a, ns=(T, T)), os.utime(b, ns=(T, T))  # same size and mtime: only the inode can tell the swap
    with pytest.raises(SystemExit, match="steps=0"):
        SP.run({**spec, "steps": 0}, [model], ["m_fakeseg"], out, redo=False, device="cpu")
    assert len(SP.run(spec, [model], ["m_fakeseg"], out, redo=False, device="cpu")) == 1
    assert SP.run(spec, [model], ["m_fakeseg"], out, redo=False, device="cpu") == []

    def swap():  # the two files permuted by rename: same bytes, same size, same mtime (to the second on Lustre)
        os.rename(a, run_dir / "x"), os.rename(b, a), os.rename(run_dir / "x", b)

    swap()
    with pytest.raises(SystemExit, match="re-extraction"):
        SP.run(spec, [model], ["m_fakeseg"], out, redo=False, device="cpu")
    swap()
    assert SP.run(spec, [model], ["m_fakeseg"], out, redo=False, device="cpu") == []
    npz = run_dir / f"seg_tokens_train_{tag}_meta.npz"
    with np.load(npz) as z:
        d = dict(z)
    np.savez(npz, **{**d, "mods": d["mods"] * 2})
    os.utime(
        npz, ns=(time.time_ns() + 5 * 10**9,) * 2
    )  # same size and inode: an in-place rewrite, a later second
    with pytest.raises(SystemExit, match="re-extraction"):
        SP.run(spec, [model], ["m_fakeseg"], out, redo=False, device="cpu")
    mp = run_dir / f"seg_tokens_train_{tag}_meta.json"
    meta = json.load(open(mp))
    json.dump({**meta, "t": list(reversed(meta["t"]))}, open(mp, "w"))
    with pytest.raises(SystemExit, match="differ from its npz"):
        SP.TokenSource(model, "m_fakeseg")
    json.dump(meta, open(mp, "w"))
    # an inputs record that is empty or not role-keyed (an older cell format) is refused
    chk = SP.CellInputCheck(model, "m_fakeseg")
    for bad in ({}, {str(a): "abc"}):
        with pytest.raises(SystemExit, match="old inputs format"):
            chk.check("cell", bad, {})


def test_seg_dataset_refuses_a_root_other_than_the_registered_one(tmp_path, seg_registry):
    from datasets import geobench_seg as GS

    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    cfg = _run_cfg(tmp_path)
    cfg.dataset.path = str(seg_registry) + "/"  # the same root, spelled differently, is accepted
    assert len(GS._wrapper("m_fakeseg")(cfg).get_data(cfg)["train"].dataset) == 8
    cfg.dataset.path = "/lustre/somewhere/EuroSAT"
    with pytest.raises(ValueError, match="registered root"):
        GS._wrapper("m_fakeseg")(cfg).get_data(cfg)


class OverflowModel(FakeModel):
    def extract_raw(self, *a, **kw):
        feat, ada, latents, noise = super().extract_raw(*a, **kw)
        feat[:, 3] = 1e5  # beyond fp16 in a kept channel
        return feat, ada, latents, noise


def test_seg_tokens_refuses_fp16_overflow(tmp_path, seg_registry):
    from datasets import geobench_seg as GS
    from tasks import seg_tokens as ST

    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    cfg = _run_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite fp16"):
        ST.SegTokenExtractionTask().run(cfg, OverflowModel(), GS._wrapper("m_fakeseg")(cfg))


def test_ditf_tokens_matches_the_offline_ditf():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3072)).astype(np.float32)
    mods = rng.standard_normal((1, 3, 3072)).astype(np.float32)
    want = F.ditf(x[:, None, :], mods)[:, 0, :]
    got = SP.ditf_tokens(torch.from_numpy(x), torch.from_numpy(mods[0])).numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)
    # another backbone's stamped list: the per-token DiTF follows the offline one for that list too
    want = F.ditf(x[:, None, :], mods, discard=[7, 9])[:, 0, :]
    got = SP.ditf_tokens(torch.from_numpy(x), torch.from_numpy(mods[0]), discard=[7, 9]).numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)
    assert not np.allclose(got, SP.ditf_tokens(torch.from_numpy(x), torch.from_numpy(mods[0])).numpy())


class GeoFakeModel(FakeModel):
    """A non-FLUX adapter (GeoCore-style class attributes): own weights tag, 8 double blocks,
    no discard list at extraction time, provenance_meta() stamped into the cache meta."""

    weights_tag = "geocore-9b"
    hidden_size = C
    num_double_blocks = 8
    num_blocks = 32
    discard_channels = None

    def provenance_meta(self):
        return {"hidden_size": C, "num_double_blocks": 8, "num_blocks": 32, "geocore_commit": "deadbeef"}


def test_seg_tokens_non_flux_backbone_stamps_its_own_provenance(tmp_path, seg_registry, monkeypatch):
    from datasets import geobench_seg as GS
    from tasks import seg_tokens as ST

    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    monkeypatch.delenv("FLUX_RANDOM_INIT", raising=False)
    cfg = _run_cfg(tmp_path, k=(4, 12))
    cfg.guidance_scale = 0.0
    ds = GS._wrapper("m_fakeseg")(cfg)
    meta = ST.SegTokenExtractionTask().run(cfg, GeoFakeModel(), ds)
    assert meta["weights"] == "geocore-9b" and meta["tag"] == "oneshot_g0.0_ens2_img64"
    assert (
        meta["hidden_size"] == C and meta["num_double_blocks"] == 8 and meta["geocore_commit"] == "deadbeef"
    )
    assert "random_init" not in meta and "discard_channels" not in meta  # stamped post hoc, never guessed
    # the double-block stamp is against the MODEL's count: k=4 is a double block of an 8-double model
    from config_types import double_block_stamp

    assert {k: meta[k] for k in double_block_stamp([4, 12], num_double=8)} == double_block_stamp(
        [4, 12], num_double=8
    )
    assert str(np.load(ST.meta_path(cfg.save_dir, "train", meta["tag"]))["weights"]) == "geocore-9b"
    # the DiTF seg view needs the stamped list: the probe's source refuses a non-FLUX meta without it
    with pytest.raises(SystemExit, match="discard_channels"):
        F.meta_discard_channels(meta, "x")
    assert F.meta_discard_channels({**meta, "discard_channels": [3]}, "x") == [3]


@pytest.mark.parametrize("flag", ["FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO"])
def test_seg_tokens_non_flux_backbone_rejects_control_env(tmp_path, seg_registry, monkeypatch, flag):
    from datasets import geobench_seg as GS
    from tasks import seg_tokens as ST

    _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    monkeypatch.setenv(flag, "1" if flag != "FIXED_COND_T" else "500")
    cfg = _run_cfg(tmp_path)
    cfg.guidance_scale = 0.0
    ds = GS._wrapper("m_fakeseg")(cfg)
    with pytest.raises(ValueError, match="honors"):
        ST.SegTokenExtractionTask().run(cfg, GeoFakeModel(), ds)


# ------------------------------------------------------------------------------- protocol
def test_channel_stats_match_the_full_fp32_reduction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((37, 16, 8)).astype(np.float32)
    x[..., 0] = 1000.0 + 1e-3 * rng.standard_normal(
        (37, 16)
    )  # large offset, tiny spread: no cancellation allowed
    x[..., 1] = 0.3  # constant: var 0 -> std exactly 1e-6
    mean, std = SP.channel_stats(torch.from_numpy(x[i : i + 5]) for i in range(0, 37, 5))
    full = torch.from_numpy(x.reshape(-1, 8)).double()
    ref_mean, ref_std = full.mean(0), full.std(0)
    assert torch.allclose(mean.double(), ref_mean, rtol=1e-7, atol=0)
    assert torch.allclose(std.double()[2:], ref_std[2:] + 1e-6, rtol=1e-6, atol=0)
    assert abs(float(std[0]) - float(ref_std[0] + 1e-6)) <= 2 * np.spacing(
        np.float32(ref_std[0])
    )  # the two fp32 roundings
    assert float(std[1]) == np.float32(1e-6)


def test_prep_all_standardizes_before_the_fp16_cast(tmp_path):
    """The head's fp16 input error must be 2^-11 in STANDARDIZED units on every channel. A channel
    that is a large constant plus a tiny spread tells the two orders apart: casting first would
    round the offset (error ~ 2^-11 * 1000 / 1e-3 standardized units), standardizing first does not."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 16, 8)).astype(np.float32)
    x[..., 0] = 1000.0 + 1e-3 * rng.standard_normal((30, 16))
    np.save(tmp_path / "t.npy", x)  # fp32 on disk here; the real caches are fp16 but the channel is < 65504
    mm = np.load(tmp_path / "t.npy", mmap_mode="r")
    idx = np.arange(30)
    mean, std = SP.channel_stats(SP._chunks(mm, idx, None, "cpu"))
    a = SP.prep_all(mm, idx, None, "cpu", "cpu", torch.float16, mean, std).float()
    b = SP.prep_all(mm, idx, None, "cpu", "cpu", torch.float32, mean, std)
    assert (a - b).abs().max() <= 2**-11 * b.abs().max() + 1e-6
    assert b[..., 0].std() == pytest.approx(
        1.0, abs=0.05
    )  # the constant channel is standardized, not swamped


def test_image_subsets_are_nested_and_seeded():
    a = SP.image_subsets(100, (0.01, 0.1, 0.25, 1.0), 0)
    b = SP.image_subsets(100, (0.01, 0.1, 0.25, 1.0), 0)
    assert all(np.array_equal(a[f], b[f]) for f in a)
    assert len(a[0.01]) == 1 and len(a[0.1]) == 10 and len(a[1.0]) == 100
    assert set(a[0.01]) <= set(a[0.1]) <= set(a[0.25])
    assert not np.array_equal(SP.image_subsets(100, (0.1,), 1)[0.1], a[0.1])


def test_confusion_miou_and_bootstrap():
    ref = np.array([[[0, 0], [1, F.SEG_IGNORE]]], np.uint8)
    pred = np.array([[[0, 1], [1, 2]]], np.uint8)
    conf = SP.confusion_per_image(pred, ref, 3)
    assert conf.shape == (1, 3, 3) and conf.sum() == 3  # the ignore pixel is dropped
    assert conf[0, 0, 0] == 1 and conf[0, 0, 1] == 1 and conf[0, 1, 1] == 1
    # IoU: class 0 = 1/2, class 1 = 1/2, class 2 absent -> skipped
    assert SP.miou(conf.sum(0)) == pytest.approx(0.5)
    lo, hi = SP.bootstrap_miou(np.repeat(conf, 20, axis=0), n=50)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def _fake_tokens(masks, grid, c=C, informative=True, seed=0):
    """(N, L, C) fp16 tokens: channel j of a token = 1 if the mask block under it is class j."""
    rng = np.random.default_rng(seed)
    n = masks.shape[0]
    h, w = grid
    H, W = masks.shape[1:]
    out = rng.standard_normal((n, h * w, c)).astype(np.float32) * 0.1
    for i in range(n):
        for yy in range(h):
            for xx in range(w):
                blk = masks[i, yy * H // h : (yy + 1) * H // h, xx * W // w : (xx + 1) * W // w]
                vals = blk[blk != F.SEG_IGNORE]
                if vals.size and informative:
                    out[i, yy * w + xx, int(np.bincount(vals).argmax())] += 3.0
    return out.astype(np.float16)


def _dino_cache(dirpath, dataset, split, images, masks, tokens, img=64):
    os.makedirs(dirpath, exist_ok=True)
    base = os.path.join(dirpath, f"fake_{dataset}_{split}_tokens{img}")
    np.save(base + ".npy", tokens)
    meta = {"dataset": dataset, "split": split, "n": len(images), "max_images": None, "preset": "fake", "weights": "w", "img": img,
            "grid_hw": list(GRID), "file": os.path.basename(base + ".npy"), "paths": images, "masks": masks}  # fmt: skip
    json.dump(meta, open(base + "_meta.json", "w"))


def test_seg_probe_run_selects_on_val_and_reaches_the_planted_miou(tmp_path, seg_registry):
    masks = _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    good, bad = tmp_path / "good", tmp_path / "bad"
    for split in ("train", "val", "test"):
        images, mfiles = F.list_official_seg_split("m_fakeseg", split)
        _dino_cache(str(good), "m_fakeseg", split, images, mfiles, _fake_tokens(masks[split], GRID))
        _dino_cache(
            str(bad), "m_fakeseg", split, images, mfiles, _fake_tokens(masks[split], GRID, informative=False)
        )
    spec = {
        "fractions": [0.25, 1.0], "seeds": [0, 1], "lr_grid": [1e-2, 1e-1], "steps": 150, "batch_images": 4,
        "datasets": ["m_fakeseg"],
        "models": [
            {"name": "teacher", "kind": "dino", "tokens": str(good), "preset": "fake", "img": 64, "expect": {"weights": "w"}, "teacher": None},
            {"name": "noise", "kind": "dino", "tokens": str(bad), "preset": "fake", "img": 64, "expect": {"weights": "w"}, "teacher": "teacher"},
        ],
    }  # fmt: skip
    out = str(tmp_path / "res")
    done = SP.run(spec, spec["models"], ["m_fakeseg"], out, redo=False, device="cpu")
    assert len(done) == 2 * 3  # 2 models x (0.25 x 2 seeds + 1.0 once)
    # the fp32 train path (the old one) and the fp16 default agree to fp16 rounding on the synthetic cells
    SP.run(
        spec,
        spec["models"][:1],
        ["m_fakeseg"],
        str(tmp_path / "res32"),
        redo=False,
        device="cpu",
        train_dtype="float32",
    )
    c16, _ = SP.collect(spec, out)
    c32, _ = SP.collect(spec, str(tmp_path / "res32"))
    for key, v in c32.items():
        assert abs(v["test"] - c16[key]["test"]) < 0.02 and abs(v["val"] - c16[key]["val"]) < 0.02
    cells, missing = SP.collect(spec, out)
    assert not missing
    assert cells[("teacher", "m_fakeseg", 1.0, 0)]["test"] > 0.9
    assert cells[("noise", "m_fakeseg", 1.0, 0)]["test"] < 0.7
    d = np.load(SP.cell_path(out, "teacher", "m_fakeseg", 1.0, 0), allow_pickle=True)
    sel, table = json.loads(str(d["selected"])), json.loads(str(d["val_table"]))
    assert sel["val_miou"] == max(r["val_miou"] for r in table)  # selected on val, first max
    assert d["test_conf_per_image"].shape == (4, 3, 3) and np.isfinite(d["ci95"]).all()
    summary = SP.summarize(spec, cells)
    assert summary[("noise", "m_fakeseg")]["aulc"] < summary[("teacher", "m_fakeseg")]["aulc"]
    md = SP.tables_markdown(spec, summary, missing)
    assert "teacher" in md and "Δ AULC" in md
    # a second run reuses every cell; a changed protocol is refused without --redo
    assert SP.run(spec, spec["models"], ["m_fakeseg"], out, redo=False, device="cpu") == []
    with pytest.raises(SystemExit, match="config differs"):
        SP.run({**spec, "steps": 151}, spec["models"], ["m_fakeseg"], out, redo=False, device="cpu")
    # cell reuse (CellInputCheck): the spec pointing at an empty dir while the files exist
    # elsewhere is a mis-pointed spec; a partial delete is refused; the README's delete of the
    # token dir once the cells exist -> cells kept, said aloud; a relocated cache with the same
    # bytes is kept; a re-extraction with the same tag but different bytes is refused whether
    # or not the original files still exist
    import shutil

    teacher = spec["models"][:1]
    with pytest.raises(SystemExit, match="exist elsewhere"):
        SP.run(
            spec,
            [{**teacher[0], "tokens": str(tmp_path / "empty")}],
            ["m_fakeseg"],
            out,
            redo=False,
            device="cpu",
        )
    shutil.copytree(good, tmp_path / "moved")
    shutil.copytree(good, tmp_path / "again")
    tr = tmp_path / "again" / "fake_m_fakeseg_train_tokens64.npy"
    np.save(tr, np.load(tr) * np.float16(2))
    again = [{**teacher[0], "tokens": str(tmp_path / "again")}]
    with pytest.raises(SystemExit, match="re-extraction"):
        SP.run(spec, again, ["m_fakeseg"], out, redo=False, device="cpu")
    os.remove(good / "fake_m_fakeseg_test_tokens64.npy")
    with pytest.raises(SystemExit, match="token file missing"):
        SP.run(spec, teacher, ["m_fakeseg"], out, redo=False, device="cpu")
    shutil.rmtree(good)
    assert SP.run(spec, teacher, ["m_fakeseg"], out, redo=False, device="cpu") == []
    pinned = [
        {**teacher[0], "expect": {"weights": "w", "preset": "fake"}}
    ]  # a tightened pin that holds: kept
    assert SP.run(spec, pinned, ["m_fakeseg"], out, redo=False, device="cpu") == []
    with pytest.raises(SystemExit, match="stored train meta"):
        SP.run(
            spec,
            [{**teacher[0], "expect": {"weights": "other"}}],
            ["m_fakeseg"],
            out,
            redo=False,
            device="cpu",
        )
    with pytest.raises(SystemExit, match="stored train meta"):  # report re-enforces the pins too
        SP.collect({**spec, "models": [{**teacher[0], "expect": {"weights": "other"}}]}, out)
    moved = [{**teacher[0], "tokens": str(tmp_path / "moved")}]
    assert SP.run(spec, moved, ["m_fakeseg"], out, redo=False, device="cpu") == []
    with pytest.raises(SystemExit, match="re-extraction"):
        SP.run(spec, again, ["m_fakeseg"], out, redo=False, device="cpu")
    # a token file shorter than the listing is refused before any head is trained
    for split in ("train", "val", "test"):
        images, mfiles = F.list_official_seg_split("m_fakeseg", split)
        _dino_cache(
            str(tmp_path / "short"), "m_fakeseg", split, images, mfiles, _fake_tokens(masks[split], GRID)
        )
    va = tmp_path / "short" / "fake_m_fakeseg_val_tokens64.npy"
    np.save(va, np.load(va)[:-1])
    with pytest.raises(SystemExit, match="rows x tokens"):
        SP.run(spec, [{**teacher[0], "tokens": str(tmp_path / "short")}], ["m_fakeseg"], str(tmp_path / "res_short"), redo=False, device="cpu")  # fmt: skip
    # a NaN val mIoU is refused, never selected around (rule 6: the guard must fire in a test)
    images, mfiles = F.list_official_seg_split(
        "m_fakeseg", "val"
    )  # restore the row count: the refusal must come from the NaN
    _dino_cache(str(tmp_path / "short"), "m_fakeseg", "val", images, mfiles, _fake_tokens(masks["val"], GRID))
    with pytest.raises(SystemExit, match="NaN val mIoU"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SP, "miou", lambda conf: float("nan"))
            SP.run(spec, [{**teacher[0], "tokens": str(tmp_path / "short")}], ["m_fakeseg"], str(tmp_path / "res_nan"), redo=False, device="cpu")  # fmt: skip


def test_token_source_refuses_smoke_caches_wrong_pins_and_reordered_images(tmp_path, seg_registry):
    masks = _write_tree(seg_registry, {"train": 8, "val": 4, "test": 4})
    d = tmp_path / "c"
    for split in ("train", "val", "test"):
        images, mfiles = F.list_official_seg_split("m_fakeseg", split)
        _dino_cache(str(d), "m_fakeseg", split, images, mfiles, _fake_tokens(masks[split], GRID))
    model = {
        "name": "m",
        "kind": "dino",
        "tokens": str(d),
        "preset": "fake",
        "img": 64,
        "expect": {"weights": "w"},
    }
    src = SP.TokenSource(model, "m_fakeseg")
    assert list(src.files) == ["patch"] and src.grid() == GRID
    with pytest.raises(SystemExit, match="weights"):
        SP.TokenSource({**model, "expect": {"weights": "other"}}, "m_fakeseg")
    mp = d / "fake_m_fakeseg_val_tokens64_meta.json"
    meta = json.load(open(mp))
    json.dump({**meta, "max_images": 4}, open(mp, "w"))
    with pytest.raises(SystemExit, match="smoke"):
        SP.TokenSource(model, "m_fakeseg")
    json.dump({**meta, "paths": list(reversed(meta["paths"]))}, open(mp, "w"))
    src = SP.TokenSource(model, "m_fakeseg")
    images, _ = F.list_official_seg_split("m_fakeseg", "val")
    assert src._paths("val") != images  # run() refuses this: the cache's image list is not the official one


# ------------------------------------------------------------------------- run.py wiring
def test_runconfig_guards_for_extract_seg_tokens():
    """max_images is honored by extract_seg_tokens only and stays out of the config hash (rule 9);
    segmentation datasets are served to that task only, and the task refuses other datasets."""
    import importlib

    run_mod = importlib.import_module("run")

    def cfg(**kw):
        base = dict(task="extract_seg_tokens", dataset=run_mod.DatasetConfig(name="m_pv4ger_seg", path=None),
                    t=[100, 260, 420], k=[12, 28, 38], img_size=[512, 512], guidance_scale=1.0, extract_split="val")  # fmt: skip
        base.update(kw)
        return run_mod.RunConfig(**base)

    a, b = cfg(), cfg(max_images=24)
    assert a.config_hash() == b.config_hash()
    assert cfg(extract_split="test").config_hash() == a.config_hash()
    with pytest.raises(ValueError, match="max_images"):
        cfg(task="extract", max_images=24, dataset=run_mod.DatasetConfig(name="m_eurosat", path=None))
    with pytest.raises(ValueError, match="extract_seg_tokens"):
        cfg(task="extract")  # a segmentation dataset under the classification extractor
    with pytest.raises(ValueError, match="OFFICIAL_SEG"):
        cfg(dataset=run_mod.DatasetConfig(name="m_eurosat", path=None))
