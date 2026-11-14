"""eval.satdifuser: the pooled-pyramid dump format, its candidate views, and the pins the matrix
checks. CPU only; their extractor is never imported (torch tensors stand in for its output)."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import features as F  # noqa: E402
from eval import matrix as M  # noqa: E402
from eval import probe as PR  # noqa: E402
from eval import satdifuser as SD  # noqa: E402

TS = [47, 45, 40]


def _feats(b=2, widths=(32, 16, 8, 4), chans=(6, 4, 3, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    return {t: [torch.randn(b, c, w, w, generator=g) for w, c in zip(widths, chans)] for t in TS}


def test_pool_pyramid_is_the_spatial_mean_in_fuser_order():
    feats = _feats()
    widths, arrays = SD.pool_pyramid(feats, TS)
    assert widths == [32, 16, 8, 4]
    assert [a.shape for a in arrays] == [(2, 3, 6), (2, 3, 4), (2, 3, 3), (2, 3, 2)]
    # (image 1, t=45 -> index 1, scale 16) is that tensor's mean over h, w
    np.testing.assert_allclose(arrays[1][1, 1], feats[45][1][1].mean(dim=(1, 2)).numpy(), rtol=1e-5)
    with pytest.raises(RuntimeError, match="save_timesteps"):
        SD.pool_pyramid({47: feats[47]}, TS)
    bad = {t: list(reversed(v)) for t, v in feats.items()}  # ascending widths
    with pytest.raises(RuntimeError, match="descending"):
        SD.pool_pyramid(bad, TS)


def test_to_model_input_matches_their_transform():
    img = np.zeros((64, 64, 3), np.float32)
    img[:, :, 0] = 1.0  # red
    x = SD.to_model_input(img, 256)
    assert x.shape == (3, 256, 256)
    assert torch.allclose(x[0], torch.ones(256, 256)) and torch.allclose(x[1], -torch.ones(256, 256))


def _write_cache(
    tmp_path, dataset, split, n, seed=None, max_images=None, extra_meta=None, scales=(32, 16, 8, 4)
):
    rng = np.random.default_rng(
        {"train": 0, "val": 1, "test": 2}[split] * 1000 + n
    )  # reproducible across runs
    files = [os.path.join("data", f"{dataset}_rgb", split, "c", f"{split}_{i:04d}.png") for i in range(n)]
    labels = np.arange(n) % 3
    arrays = {
        f"{SD.SCALE_PREFIX}{w}": rng.standard_normal((n, len(TS), 2 + w // 8)).astype(np.float16)
        for w in scales
    }
    path = tmp_path / f"satdifuser_{dataset}_{split}.npz"
    np.savez(path, **arrays, timesteps=np.array(TS), t_ddpm=np.array([21, 61, 161]), scales=np.array(scales),
             paths=np.array(files), labels=labels, weights=np.array(SD.WEIGHTS), pooling=np.array(SD.POOLING))  # fmt: skip
    meta = {
        "dataset": dataset, "split": split, "n": n, "weights": SD.WEIGHTS, "img_size": 256, "num_timesteps": 50,
        "save_timesteps": TS, "diffusion_mode": "inversion", "prompt": "A satellite image", "pooling": SD.POOLING,
        "vae_seed": SD.SPLIT_SEED[split] if seed is None else seed, "max_images": max_images,
        "model_root": "/x/resolution256", "unet_config_sha1": "abc", "repo_commit": "deadbeef",
        "extractor_yaml_sha1": "def", "source": "png", **(extra_meta or {}),
    }  # fmt: skip
    with open(str(path).replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f)
    return str(path), files, labels


@pytest.fixture
def caches(tmp_path, monkeypatch):
    sizes = {"train": 12, "val": 6, "test": 6}
    monkeypatch.setitem(
        F.OFFICIAL, "m_fake", {"sizes": sizes, "root": "data/m_fake_rgb", "classes": lambda: ["a", "b", "c"]}
    )
    written = {s: _write_cache(tmp_path, "m_fake", s, n) for s, n in sizes.items()}
    monkeypatch.setattr(
        F, "list_official_split", lambda ds, s, root=None, classes=None: (written[s][1], written[s][2])
    )
    return tmp_path, written


def _args(tmp_path, scale, view=None):
    return SimpleNamespace(dataset="m_fake", features=str(tmp_path / "satdifuser_m_fake_{split}.npz"), view=view,
                           sizes=None, expect={"scale": scale})  # fmt: skip


def test_official_satdifuser_builds_the_flux_candidate_set_at_the_pinned_scale(caches):
    tmp_path, _ = caches
    cands, ytr, yva, yte, paths, inputs, extra = PR.official_satdifuser(_args(tmp_path, "s16"))
    assert list(cands) == ["t47", "t45", "t40", "concat3t"]
    assert cands["t47"][0].shape == (12, 4) and cands["concat3t"][2].shape == (6, 12)
    assert cands["t47"][0].dtype == np.float64
    assert len(paths) == 6 and len(inputs) == 3 and extra["scale"] == "s16"
    allc, *_ = PR.official_satdifuser(_args(tmp_path, "all"))
    assert allc["t47"][0].shape == (12, 6 + 4 + 3 + 2)
    one, *_ = PR.official_satdifuser(_args(tmp_path, "s8", view="t40"))
    assert list(one) == ["t40"]
    with pytest.raises(SystemExit, match="scale"):
        PR.official_satdifuser(_args(tmp_path, "s64"))
    with pytest.raises(SystemExit, match="expect.scale"):
        PR.official_satdifuser(
            SimpleNamespace(dataset="m_fake", features="x", view=None, sizes=None, expect={})
        )


def test_official_satdifuser_refuses_wrong_pins_shared_seeds_and_smoke_caches(caches, monkeypatch):
    tmp_path, written = caches
    _write_cache(tmp_path, "m_fake", "val", 6, seed=SD.SPLIT_SEED["train"])  # same VAE seed as train
    with pytest.raises(SystemExit, match="vae_seed shared"):
        PR.official_satdifuser(_args(tmp_path, "s16"))
    _write_cache(tmp_path, "m_fake", "val", 6, max_images=6)
    with pytest.raises(SystemExit, match="smoke"):
        PR.official_satdifuser(_args(tmp_path, "s16"))
    _write_cache(tmp_path, "m_fake", "val", 6, extra_meta={"prompt": "A photo"})
    with pytest.raises(SystemExit, match="prompt"):
        PR.official_satdifuser(_args(tmp_path, "s16"))
    _write_cache(tmp_path, "m_fake", "val", 6, extra_meta={"weights": "other"})
    with pytest.raises(SystemExit, match="weights"):
        PR.official_satdifuser(_args(tmp_path, "s16"))
    _write_cache(tmp_path, "m_fake", "val", 6)  # restore
    monkeypatch.setattr(
        F,
        "list_official_split",
        lambda ds, s, root=None, classes=None: (list(reversed(written[s][1])), written[s][2]),
    )
    with pytest.raises(SystemExit, match="paths differ"):
        PR.official_satdifuser(_args(tmp_path, "s16"))


def test_matrix_dispatches_kind_satdifuser(caches):
    tmp_path, _ = caches
    model = {"name": "satdifuser", "kind": "satdifuser", "features": str(tmp_path / "satdifuser_{dataset}_{split}.npz"),
             "view": None, "expect": {"scale": "s32"}, "per_dataset": {"m_fake": {"expect": {"scale": "s4"}}}}  # fmt: skip
    cands, ytr, yva, yte, paths, inputs = M.load_cands(model, "m_fake")
    assert cands["t47"][0].shape == (12, 2)  # s4 from per_dataset, not the top-level s32
    with pytest.raises(SystemExit, match="not supported"):
        M.load_cands({"name": "x", "kind": "vae", "features": "x"}, "m_fake")


def test_compare_passes_on_near_identical_dumps_and_fails_otherwise(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _, _ = _write_cache(tmp_path / "a", "m_fake", "val", 4)
    b, _, _ = _write_cache(tmp_path / "b", "m_fake", "val", 4)
    da = dict(np.load(a))
    np.savez(b, **{k: (v + np.float16(1e-3) if k.startswith(SD.SCALE_PREFIX) else v) for k, v in da.items()})
    SD.compare(SimpleNamespace(a=a, b=b, min_cosine=0.99))
    assert "COMPARE PASS" in capsys.readouterr().out
    np.savez(b, **{k: (-v if k.startswith(SD.SCALE_PREFIX) else v) for k, v in da.items()})
    with pytest.raises(SystemExit):
        SD.compare(SimpleNamespace(a=a, b=b, min_cosine=0.99))


def test_select_is_val_only_and_never_opens_the_test_cache(caches):
    tmp_path, _ = caches
    os.remove(tmp_path / "satdifuser_m_fake_test.npz")
    os.remove(tmp_path / "satdifuser_m_fake_test_meta.json")
    args = SimpleNamespace(features=str(tmp_path / "satdifuser_{dataset}_{split}.npz"), datasets=["m_fake"],
                           out_dir=str(tmp_path / "sel"), c_grid=["0.1", "1"])  # fmt: skip
    res = SD.select(args)["m_fake"]
    assert res["selected"]["val_acc"] == max(r["val_acc"] for r in res["rows"])
    assert res["scales"] == ["s32", "s16", "s8", "s4", "all"] and len(res["rows"]) == 5 * 4 * 2
    # a smoke dump or a differently pinned dump cannot drive the selected scale
    mp = tmp_path / "satdifuser_m_fake_val_meta.json"
    meta = json.load(open(mp))
    json.dump({**meta, "max_images": 3}, open(mp, "w"))
    with pytest.raises(SystemExit, match="smoke"):
        SD.select(args)
    json.dump({**meta, "pooling": "max"}, open(mp, "w"))
    with pytest.raises(SystemExit, match="pooling"):
        SD.select(args)
