"""RunConfig fields for the GeoCore backbone: required paths, hash participation, honors.

pytest tests/test_geocore_config.py -v
"""

import os
import sys

import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from run import ModelConfig, RunConfig, _task_honors  # noqa: E402

T = [100, 180, 260, 340, 420, 500, 580, 740, 900]


def _geo(**over):
    kw = dict(task="extract", model=ModelConfig(name="geocore"), geocore_ckpt="/a/GeoCore-9B",
              geocore_vae="/a/GeoCore-9B/vae/diffusion_pytorch_model.safetensors", t=T,
              k=[4, 8, 12, 16, 20, 24, 28, 31], guidance_scale=0.0)  # fmt: skip
    kw.update(over)
    return RunConfig(**kw)


def test_geocore_requires_paths():
    with pytest.raises(ValueError, match="geocore-ckpt"):
        RunConfig(task="extract", model=ModelConfig(name="geocore"), t=T)


def test_flux_rejects_geocore_paths():
    with pytest.raises(ValueError, match="only valid with model.name='geocore'"):
        RunConfig(task="extract", geocore_ckpt="/x", t=T)


def test_paths_change_hash():
    a, b, c = _geo(), _geo(geocore_ckpt="/b/GeoCore-9B"), _geo(geocore_vae="/b/vae.safetensors")
    assert len({a.config_hash(), b.config_hash(), c.config_hash()}) == 3


def test_geocore_run_name_has_model_name():
    name = _geo(dataset=__import__("run").DatasetConfig(name="m_eurosat", path="/d")).make_run_name()
    assert name.startswith("m_eurosat_geocore_") and name.endswith("+42")


def test_flux_hash_unchanged():
    """Pins the hash of an EXISTING cache directory so the new fields cannot repoint FLUX runs:
    models/m_bigearthnet_multiblock_ens8/m_bigearthnet_flux_9c2b5c5a+42 (frozen multi-block train
    extraction, experiments/isaac/sweep/multiblock.sbatch: 9 t, 14 blocks, ens 8, g1.0, 224 px,
    no --subset-size (the meta's subset_size=20000 is the EFFECTIVE size, not the request),
    --dataset.path data/m_bigearthnet_rgb). Verified 2026-11-02 (job 6532057): the untouched main
    checkout and this branch both hash this request to 9c2b5c5a; with --subset-size 20000 both give
    57ec2684, so a mismatch here means the hash moved, not the reconstruction."""
    from run import DatasetConfig

    cfg = RunConfig(
        task="extract", model=ModelConfig(name="flux", ensemble_size=8),
        dataset=DatasetConfig(name="m_bigearthnet", path="data/m_bigearthnet_rgb"),
        img_size=[224, 224], t=T, k=[4, 8, 12, 18, 19, 24, 28, 33, 38, 43, 48, 54, 55, 56],
        guidance_scale=1.0, seed=42, eps_seed=42, subset_seed=42, extract_split="train",
        label_fraction=1.0, batch_size=1,
    )  # fmt: skip
    assert cfg.config_hash() == "9c2b5c5a", (
        "FLUX config hash changed -- every frozen cache dir would move (rule 9)"
    )


def test_honors():
    assert _task_honors("extract", "geocore") == ()
    assert _task_honors("extract") == ("FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO")
    assert _task_honors("finetune-diffusion", "flux") == ("FLUX_RANDOM_INIT", "DEGRADE_TO")


def test_geocore_control_env_rejected(monkeypatch):
    from tasks.extraction import env_provenance

    monkeypatch.setenv("FLUX_RANDOM_INIT", "1")
    with pytest.raises(ValueError, match="honors"):
        env_provenance(_geo(), honors=_task_honors("extract", "geocore"))


def test_guidance_pin():
    with pytest.raises(ValueError, match="guidance_scale=0.0"):
        _geo(guidance_scale=1.0)
    with pytest.raises(ValueError, match="positive"):
        RunConfig(task="extract", t=T, guidance_scale=0.0)
    assert _geo().guidance_scale == 0.0
