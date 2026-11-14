"""Pre-registered massive-channel rule (eval/discard_channels.py).

pytest tests/test_discard_channels.py -v
"""

import os
import sys

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from eval.discard_channels import massive_channels  # noqa: E402


def test_rule_finds_planted_channels():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(50, 3, 32))
    f[:, :, [5, 17]] *= 200
    assert massive_channels(f, n_images=50) == [5, 17]


def test_rule_empty_when_flat():
    assert massive_channels(np.random.default_rng(1).normal(size=(50, 3, 32))) == []


def test_rule_uses_only_first_n_images():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(60, 2, 16))
    f[50:, :, 3] *= 1e4  # massive only in images the rule must ignore
    assert massive_channels(f, n_images=50) == []
    assert massive_channels(f, n_images=60) == [3]


def test_rule_rejects_bad_shape():
    with pytest.raises(ValueError):
        massive_channels(np.zeros((10, 16)))


def test_stamp_meta_idempotent_and_refuses_conflict(tmp_path):
    import json

    from eval.discard_channels import stamp_meta

    cache = tmp_path / "multistep_train_feats_oneshot_g0.0_k4-8.npz"
    meta = tmp_path / "multistep_train_feats_oneshot_g0.0_k4-8_meta.json"
    cache.write_bytes(b"")
    meta.write_text(json.dumps({"weights": "geocore-9b"}))
    stamp_meta(str(cache), [3, 9], {"factor": 50.0})
    assert json.loads(meta.read_text())["discard_channels"] == [3, 9]
    stamp_meta(str(cache), [3, 9], {"factor": 50.0})  # same list: fine
    with pytest.raises(SystemExit, match="refusing"):
        stamp_meta(str(cache), [4], {"factor": 50.0})


# ------------------------------------------------------------------ the FLUX check is part of stamping
def _cache(tmp_path, name, weights, k, massive, n=40, c=64):
    """A single-block cache (N, K, C) + meta whose `massive` channels the rule finds."""
    import json

    rng = np.random.default_rng(len(name))
    f = rng.normal(size=(n, 2, c)).astype(np.float32)
    f[:, :, massive] *= 500
    p = tmp_path / f"{name}.npz"
    np.savez(p, feats=f, labels=np.zeros(n, int))
    (tmp_path / f"{name}_meta.json").write_text(json.dumps({"weights": weights, "k": k}))
    return str(p)


def test_stamp_refused_without_flux_reference(tmp_path):
    from eval import discard_channels as D

    geo = _cache(tmp_path, "geo_train", "geocore-9b", 8, [3, 9])
    with pytest.raises(SystemExit, match="--stamp needs --flux-ref"):
        D.main(["--features", geo, "--k", "8", "--stamp", geo])


def test_stamp_refused_when_rule_misses_flux_list(tmp_path, monkeypatch):
    """The mismatch the gate exists to catch: a rule that does not reproduce FLUX's list stamps nothing."""
    import json

    from eval import discard_channels as D
    from eval import features as F

    monkeypatch.setattr(F, "DISCARD", [5, 17])
    geo = _cache(tmp_path, "geo_train", "geocore-9b", 8, [3, 9])
    bad_flux = _cache(tmp_path, "flux_bad", "flux-dev", D.FLUX_K, [5])  # one of the two channels only
    with pytest.raises(SystemExit, match=r"DISCARD_FAIL rule .* gives \[5\], expected \[5, 17\]"):
        D.main(["--features", geo, "--k", "8", "--flux-ref", bad_flux, "--stamp", geo])
    not_flux = _cache(tmp_path, "notflux", "geocore-9b", D.FLUX_K, [5, 17])  # right list, wrong backbone
    with pytest.raises(SystemExit, match="is not a FLUX cache"):
        D.main(["--features", geo, "--k", "8", "--flux-ref", not_flux, "--stamp", geo])
    for i, w in enumerate(
        ["flux-dev+lora:run42", "random_init", "flux-dev-other"]
    ):  # FLUX, but not the frozen model
        adapted = _cache(tmp_path, f"flux_adapted{i}", w, D.FLUX_K, [5, 17])
        assert F.is_flux_meta({"weights": w})  # the looser check this gate applied before
        with pytest.raises(SystemExit, match="is not a FLUX cache with the pretrained weights"):
            D.main(["--features", geo, "--k", "8", "--flux-ref", adapted, "--stamp", geo])
    wrong_k = _cache(tmp_path, "flux_k8", "flux-dev", 8, [5, 17])  # not the block the list refers to
    with pytest.raises(SystemExit, match="cache block k=8 != --k 28"):
        D.main(["--features", geo, "--k", "8", "--flux-ref", wrong_k, "--stamp", geo])
    assert "discard_channels" not in json.loads((tmp_path / "geo_train_meta.json").read_text())


def test_stamp_records_the_flux_check(tmp_path, monkeypatch, capsys):
    import json

    from eval import discard_channels as D
    from eval import features as F

    monkeypatch.setattr(F, "DISCARD", [5, 17])
    geo = _cache(tmp_path, "geo_train", "geocore-9b", 8, [3, 9])
    flux = _cache(tmp_path, "flux_ok", "flux-dev", D.FLUX_K, [5, 17])
    assert D.main(["--features", geo, "--k", "8", "--flux-ref", flux, "--stamp", geo]) == 0
    assert "DISCARD_OK FLUX reference k=28: [5, 17]" in capsys.readouterr().out
    meta = json.loads((tmp_path / "geo_train_meta.json").read_text())
    assert meta["discard_channels"] == [3, 9]
    check = meta["discard_rule"]["flux_check"]
    assert check["found"] == [5, 17] and check["k"] == D.FLUX_K and len(check["sha1"]) == 40


def test_expect_mode_returns_one_on_mismatch(tmp_path, capsys):
    from eval import discard_channels as D

    flux = _cache(tmp_path, "flux_ok", "flux-dev", 28, [5, 17])
    assert D.main(["--features", flux, "--k", "28", "--expect", "5", "17"]) == 0
    assert D.main(["--features", flux, "--k", "28", "--expect", "154", "1446"]) == 1
    assert "DISCARD_FAIL expected [154, 1446] got [5, 17]" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="separate modes"):
        D.main(["--features", flux, "--k", "28", "--flux-ref", flux, "--expect", "5", "--stamp", flux])
