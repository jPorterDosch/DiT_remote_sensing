"""Model-aware provenance in tasks/extraction (model_provenance, block_stamp_for).

pytest tests/test_geocore_extraction_meta.py -v
"""

import os
import sys

import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from config_types import DOUBLE_BLOCK_FEATURE  # noqa: E402
from run import ModelConfig, RunConfig  # noqa: E402
from tasks.extraction import block_stamp_for, env_provenance, model_provenance  # noqa: E402

T = [100, 180, 260]


def _flux_cfg():
    return RunConfig(task="extract", t=T, k=[4, 28], guidance_scale=1.0)


def _geo_cfg():
    return RunConfig(task="extract", model=ModelConfig(name="geocore"), geocore_ckpt="/a", geocore_vae="/v",
                     t=T, k=[4, 8, 12], guidance_scale=0.0)  # fmt: skip


class FakeFlux:
    pass  # no weights_tag: the pre-attribute adapter shape


class FakeGeo:
    weights_tag = "geocore-9b"
    num_double_blocks = 8
    discard_channels = None

    def provenance_meta(self):
        return {"hidden_size": 4096, "num_double_blocks": 8, "num_blocks": 32, "geocore_commit": "abc"}


def test_flux_provenance_unchanged(monkeypatch):
    for v in ("FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO"):
        monkeypatch.delenv(v, raising=False)
    cfg = _flux_cfg()
    suffix, meta = model_provenance(FakeFlux(), cfg)
    ref_suffix, ref_meta, _ = env_provenance(cfg)
    assert (suffix, meta) == (ref_suffix, ref_meta) and meta["weights"] == "flux-dev" and suffix == ""


def test_flux_randinit_still_honored(monkeypatch):
    monkeypatch.setenv("FLUX_RANDOM_INIT", "1")
    suffix, meta = model_provenance(FakeFlux(), _flux_cfg())
    assert suffix == "_RANDINIT" and meta["weights"] == "random_init"


def test_geocore_provenance_meta(monkeypatch):
    for v in ("FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO"):
        monkeypatch.delenv(v, raising=False)
    suffix, meta = model_provenance(FakeGeo(), _geo_cfg())
    assert suffix == ""
    assert meta["weights"] == "geocore-9b" and meta["hidden_size"] == 4096 and meta["num_double_blocks"] == 8
    assert meta["geocore_commit"] == "abc" and "discard_channels" not in meta
    assert meta["degrade_to"] is None and meta["fixed_cond_t"] is None


def test_geocore_discard_channels_stamped():
    m = FakeGeo()
    m.discard_channels = [3, 1000]
    assert model_provenance(m, _geo_cfg())[1]["discard_channels"] == [3, 1000]


@pytest.mark.parametrize("flag", ["FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO"])
def test_geocore_rejects_control_env(monkeypatch, flag):
    monkeypatch.setenv(flag, "1" if flag != "FIXED_COND_T" else "500")
    with pytest.raises(ValueError, match="honors"):
        model_provenance(FakeGeo(), _geo_cfg())


def test_geocore_rejects_lora():
    cfg = _geo_cfg()
    cfg.lora_checkpoint = "/some/lora.pt"
    with pytest.raises(ValueError, match="lora"):
        model_provenance(FakeGeo(), cfg)


def test_stamp_uses_model_double_count():
    assert block_stamp_for(FakeGeo(), [4, 8]) == {"double_block_feature": DOUBLE_BLOCK_FEATURE}
    assert block_stamp_for(FakeGeo(), [8, 12]) == {}
    assert block_stamp_for(FakeFlux(), [8, 12]) == {
        "double_block_feature": DOUBLE_BLOCK_FEATURE
    }  # FLUX: 19 doubles
    assert block_stamp_for(FakeFlux(), 28) == {}
