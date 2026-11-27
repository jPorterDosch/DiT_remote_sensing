"""MedSigLIP target and probe-feature extractor (design section 7), on CPU with a tiny randomly
initialised SiglipVisionModel at the real 448 px / patch-14 geometry (32 x 32 tokens). No download,
no patient data: images are synthetic PNGs in tmp_path."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as TF
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

transformers = pytest.importorskip("transformers")

from eval import extract_medsiglip as EM  # noqa: E402
from eval import features as F  # noqa: E402
from models import align_targets as AT  # noqa: E402
from models.medsiglip import MedSigLIP, preprocess_info  # noqa: E402

INFO = {
    "mean": [0.5] * 3,
    "std": [0.5] * 3,
    "side": 448,
    "resample": 2,
    "model_id": "test",
    "revision": "test-rev",
}


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    cfg = transformers.SiglipVisionConfig(
        image_size=448,
        patch_size=14,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    return MedSigLIP("cpu", model=transformers.SiglipVisionModel(cfg), info=dict(INFO))


def processor(mean=0.5):
    return transformers.SiglipImageProcessor(
        do_resize=True,
        size={"height": 448, "width": 448},
        resample=2,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[mean] * 3,
        image_std=[0.5] * 3,
    )


def write_pngs(d, n, side):
    rng = np.random.default_rng(0)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(d, f"{i:03d}.png")
        Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def fake_dataset(monkeypatch, paths_by_split):
    sizes = {s: len(p) for s, p in paths_by_split.items()}
    monkeypatch.setattr(F, "official_spec", lambda ds: {"sizes": sizes, "multilabel": True})
    monkeypatch.setattr(
        F,
        "list_official_split",
        lambda ds, s: (paths_by_split[s], np.zeros((len(paths_by_split[s]), 3), np.int64)),
    )


def test_preprocess_info_reads_the_card_values_and_rejects_others(tmp_path):
    good = {
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "size": {"height": 448, "width": 448},
        "resample": 2,
    }
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(good))
    assert preprocess_info(str(tmp_path))["side"] == 448
    bad = {**good, "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="does not map"):
        preprocess_info(str(tmp_path))


def test_prepare_upsamples_224_bilinear_without_antialias_and_guards_range(enc):
    x = torch.rand(2, 3, 224, 224) * 2 - 1
    want = TF.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False, antialias=False)
    assert torch.allclose(enc.prepare(x), want, atol=1e-6)  # mean = std = 0.5 is the identity on [-1, 1]
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        enc.prepare(x * 3)
    with pytest.raises(ValueError, match="square"):
        enc.prepare(torch.zeros(1, 3, 224, 200))


def test_encode_gives_a_square_32x32_grid_and_is_deterministic(enc):
    x = torch.rand(2, 3, 224, 224) * 2 - 1
    tok, pooled = enc.encode(x)
    assert tok.shape == (2, 32 * 32, 16) and pooled.shape == (2, 16)
    tok2, _ = enc.encode(x)
    assert torch.equal(tok, tok2)


def test_target_registry_parse_and_grid_pooling(enc):
    assert "medsiglip" in AT.families()
    assert AT.parse_medsiglip_arg("448") is None and AT.parse_medsiglip_arg("448@abc") == "abc"
    with pytest.raises(SystemExit, match="medsiglip:448"):
        AT.parse_medsiglip_arg("224")
    t = AT.MedSigLIPTarget("cpu", encoder=enc)
    assert (t.name, t.dim) == ("medsiglip448", 16)
    d = t.describe()
    assert (
        d["revision"] == "test-rev"
        and d["grid"] == 32
        and d["patch"] == 14
        and "antialias=False" in d["resize"]
    )
    z = AT.match_grid(t.patches(torch.zeros(2, 3, 224, 224)), 196)  # 32 x 32 -> FLUX's 14 x 14
    assert z.shape == (2, 196, 16)


def test_extract_writes_the_dino_layout_and_the_probe_views_read_it(enc, tmp_path, monkeypatch):
    paths = {s: write_pngs(tmp_path / s, 3, 448) for s in ("train", "val", "test")}
    fake_dataset(monkeypatch, paths)
    out = tmp_path / "feats"
    outs = EM.main(
        [
            "--dataset",
            "chestxray14",
            "--view",
            "448",
            "--out-dir",
            str(out),
            "--batch",
            "2",
            "--device",
            "cpu",
        ],
        encoder=enc,
        processor=processor(),
    )
    assert len(outs) == 3
    d = np.load(outs[0])
    assert d["cls"].shape == (3, 16) and d["mp"].shape == (3, 16) and "cls4" not in d.files
    for v in F.MEDSIGLIP_OFFICIAL_VIEWS.split(","):
        kind, X = F.view("dino", d, v)
        assert kind == "plain" and X.shape[0] == 3
    x = torch.stack([EM.load_image(p, "448") for p in paths["train"]])
    tok, pooled = enc.encode(x)
    assert np.allclose(d["cls"], pooled.numpy(), atol=1e-6) and np.allclose(
        d["mp"], tok.mean(1).numpy(), atol=1e-6
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        EM.main(
            ["--dataset", "chestxray14", "--view", "448", "--out-dir", str(out), "--device", "cpu"],
            encoder=enc,
            processor=processor(),
        )


def test_224up_view_passes_the_processor_gate(enc, tmp_path, monkeypatch):
    paths = {s: write_pngs(tmp_path / s, 2, 448) for s in ("train", "val", "test")}
    fake_dataset(monkeypatch, paths)
    outs = EM.main(
        ["--dataset", "chestxray14", "--view", "224up", "--out-dir", str(tmp_path / "f"), "--device", "cpu"],
        encoder=enc,
        processor=processor(),
    )
    assert os.path.basename(outs[0]).startswith("medsiglip448_224up_chestxray14_")


def test_448_view_refuses_images_stored_below_448(enc, tmp_path, monkeypatch):
    paths = {
        "train": write_pngs(tmp_path / "a", 1, 300),
        "val": write_pngs(tmp_path / "b", 1, 448),
        "test": write_pngs(tmp_path / "c", 1, 448),
    }
    fake_dataset(monkeypatch, paths)
    with pytest.raises(SystemExit, match="below 448"):
        EM.main(
            [
                "--dataset",
                "chestxray14",
                "--view",
                "448",
                "--out-dir",
                str(tmp_path / "f"),
                "--device",
                "cpu",
            ],
            encoder=enc,
            processor=processor(),
        )


def test_gate_fails_when_the_processor_normalizes_differently(enc):
    x = torch.rand(2, 3, 448, 448) * 2 - 1
    x = (((x + 1) * 127.5).round() / 127.5) - 1  # uint8-representable, as stored PNGs are
    assert EM.gate(enc, x, "448", processor()) < 1e-5
    with pytest.raises(SystemExit, match="refusing"):
        EM.gate(enc, x, "448", processor(mean=0.4))
