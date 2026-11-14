"""eval/object_diag.py: token labels, the swap null, the per-image statistics, the guards (each fed
the mismatch it exists to catch, rule 6) and one end-to-end run / report / figure on a fake tree."""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval import features as F  # noqa: E402
from eval import object_diag as OD  # noqa: E402

GRID = (8, 8)
L = GRID[0] * GRID[1]
MASK_HW = (16, 16)
C = 16
DS = "m_fakeobj"
SIZES = {"train": 2, "val": 2, "test": 24}
TS = (100, 260)
TAG = "oneshot_fake"


# ----------------------------------------------------------------------------- units
def test_token_labels_majority_ignore_and_refusals():
    m = np.zeros((1, 4, 4), np.uint8)
    m[0, :2, 2:] = 1  # top-right block all class 1
    m[0, 2:, :2] = [[1, 1], [1, 0]]  # bottom-left: 3 of 4 class 1
    m[0, 2:, 2:] = [[F.SEG_IGNORE, F.SEG_IGNORE], [F.SEG_IGNORE, 1]]  # 1 of 4 labelled: below the 50 % floor
    lab = OD.token_labels(m, (2, 2), 2)
    assert lab.tolist() == [[0, 1, 1, -1]]
    m[0, 2, 2] = 1  # now 2 of 4 labelled: exactly the floor counts
    assert OD.token_labels(m, (2, 2), 2)[0, 3] == 1
    with pytest.raises(SystemExit, match="not a multiple"):
        OD.token_labels(np.zeros((1, 5, 4), np.uint8), (2, 2), 2)
    with pytest.raises(SystemExit, match="outside"):
        OD.token_labels(np.full((1, 4, 4), 7, np.uint8), (2, 2), 2)


def test_eligible_labels_drops_small_classes_and_needs_two():
    lab = np.zeros((3, 40), np.int16)
    lab[0, :10] = 1  # 10 + 30: eligible
    lab[1, :3] = 1  # class 1 has 3 tokens: dropped, one class left
    lab[2, :10] = 1
    lab[2, 10:14] = 2  # class 2 dropped, still two classes
    out, el = OD.eligible_labels(lab, min_tok=8)
    assert el.tolist() == [True, False, True]
    assert (out[1, :3] == -1).all() and (out[2, 10:14] == -1).all() and (out[2, :10] == 1).all()
    assert (lab[1, :3] == 1).all()  # the input is not modified


def test_swap_partners_are_other_distinct_members():
    members = np.array([1, 3, 4, 6, 8, 9])
    p = OD.swap_partners(members, 10, n_swap=4)
    assert (p[[0, 2, 5, 7]] == -1).all()
    for i in members:
        assert i not in p[i] and len(set(p[i])) == 4 and set(p[i]) <= set(members.tolist())
    assert np.array_equal(p, OD.swap_partners(members, 10, n_swap=4))  # seeded
    # every member is used as a partner exactly n_swap times: the null sees the same masks as the truth
    assert np.bincount(p[members].ravel(), minlength=10)[members].tolist() == [4] * 6
    with pytest.raises(SystemExit, match="need more than"):
        OD.swap_partners(np.arange(4), 10, n_swap=4)


def _two_class_tokens(y, signal, seed=0, dim=12):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((2, dim))
    return torch.from_numpy(signal * emb[y] + rng.standard_normal((len(y), dim)))


def test_mask_stats_separates_structured_tokens_from_noise():
    y = np.zeros(L, np.int16)
    y.reshape(GRID)[:, 4:] = 1
    eta, gap, nn = OD.mask_stats(OD.gram(_two_class_tokens(y, 50.0)), y, GRID)
    assert eta > 0.99 and gap > 1.5 and nn == 1.0
    eta0, gap0, nn0 = OD.mask_stats(OD.gram(_two_class_tokens(y, 0.0)), y, GRID)
    assert eta0 < 0.1 and abs(gap0) < 0.05 and 0.2 < nn0 < 0.8
    # tokens outside the scored set do not move the statistics
    y2 = y.copy()
    y2[:8] = -1
    x = _two_class_tokens(y, 50.0)
    x2 = x.clone()
    x2[:8] = 1e3
    assert OD.mask_stats(OD.gram(x), y2, GRID) == pytest.approx(
        OD.mask_stats(OD.gram(x2), y2, GRID), rel=1e-6
    )
    with pytest.raises(ValueError, match="two classes"):
        OD.mask_stats(OD.gram(x), np.zeros(L, np.int16), GRID)


def test_eta2_equals_the_direct_variance_decomposition():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, L).astype(np.int16)
    x = rng.standard_normal((L, 7)) + y[:, None]
    xc = x - x.mean(0)
    between = sum((y == k).sum() * (xc[y == k].mean(0) ** 2).sum() for k in range(3))
    assert OD.mask_stats(OD.gram(torch.from_numpy(x)), y, GRID)[0] == pytest.approx(
        between / (xc**2).sum(), rel=1e-9
    )


def test_nn_agreement_ignores_adjacent_tokens():
    """Tokens that copy a neighbour (shared receptive field) must not score: every token's most
    similar token is its horizontal neighbour, which is inside the excluded radius."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((GRID[0], GRID[1] // 2, 6))
    x = np.repeat(base, 2, axis=1).reshape(L, 6) + 1e-3 * rng.standard_normal((L, 6))
    y = np.zeros(L, np.int16)
    y.reshape(GRID)[:, ::2] = 1  # the neighbour pair always straddles the two classes
    y.reshape(GRID)[:, 1::2] = 0
    s = OD.mask_stats(OD.gram(torch.from_numpy(x)), y, GRID)
    assert 0.2 < s[2] < 0.8  # an unrestricted nearest neighbour would give 0.0 here


def test_debiased_cka_identity_invariance_and_independence():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((L, 10)))
    q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    K = OD.gram(x)
    assert OD.cka_u(K, K) == pytest.approx(1.0)
    assert OD.cka_u(K, OD.gram(3.0 * x @ torch.from_numpy(q))) == pytest.approx(1.0)  # rotation + scale
    vals = [
        OD.cka_u(
            OD.gram(torch.from_numpy(rng.standard_normal((L, 200)))),
            OD.gram(torch.from_numpy(rng.standard_normal((L, 200)))),
        )
        for _ in range(40)
    ]
    assert abs(np.mean(vals)) < 0.02  # the biased estimator sits near 200 / 64 > 0.7 here


def test_null_controls_are_unbiased_and_the_gate_is_calibrated():
    """The swap-corrected statistics of tokens that carry no image-specific object structure have
    mean 0, and the gate's Bonferroni intervals exclude 0 about as rarely as they claim (rule 6: the
    gate must not fire on a true null, and rule 2: a null control reading nonzero is a broken
    instrument). 40 replicate draws of 24 images, split masks as in the fake tree."""
    rng = np.random.default_rng(5)
    n = 24
    lab = np.zeros((n, *GRID), np.int16)
    for i in range(n):
        cut = int(rng.integers(2, 7))
        if i % 2:
            lab[i, :, cut:] = 1
        else:
            lab[i, cut:, :] = 1
    lab, el = OD.eligible_labels(lab.reshape(n, L))
    assert el.all()
    part = OD.swap_partners(np.arange(n), n)
    ctx = {"eligible": el, "lab": lab, "grid": GRID, "partners": part}
    teacher = [OD.gram(_two_class_tokens(lab[i], 2.0, seed=100 + i)) for i in range(n)]
    level = 1 - 0.05 / 8
    for name in ("ctrl-noise", "ctrl-position"):
        pooled, fails = [], 0
        for s in range(40):
            d = np.zeros((n, 4))
            for i in range(n):
                G = OD.gram(OD.control_tokens(name, i, lab[i], GRID, seed=s))
                true, null = OD.score_masks(G, i, ctx)
                d[i, :3] = true - null
                d[i, 3] = OD.cka_u(G, teacher[i]) - np.mean([OD.cka_u(G, teacher[p]) for p in part[i]])
            pooled.append(d)
            for k in range(4):
                _, lo, hi = OD.boot_ci(d[:, k], level, n_boot=2000)
                fails += not (lo <= 0 <= hi)
        pooled = np.concatenate(pooled)
        z = pooled.mean(0) / (pooled.std(0, ddof=1) / np.sqrt(len(pooled)))
        assert (np.abs(z) < 3.5).all(), (name, z)  # 960 images: a bias of 0.11 sd would fail this
        assert fails <= 8, (
            name,
            fails,
        )  # 160 checks at a nominal 0.6 %; <= 5 % allowed for n = 24 percentile intervals


def test_check_same_images_fires_on_a_reordered_or_short_list():
    ref = ["a/test/1.png", "a/test/2.png", "a/test/3.png"]
    OD.check_same_images(list(ref), ref, "x")
    with pytest.raises(SystemExit, match="first difference at row 1"):
        OD.check_same_images([ref[0], ref[2], ref[1]], ref, "x")
    with pytest.raises(SystemExit, match="2 vs 3 images"):
        OD.check_same_images(ref[:2], ref, "x")


def _arm(n=6, smoke=False, **over):
    a = {
        "paths": np.array([f"t/test/{i}.png" for i in range(n)]),
        "rows": np.arange(n),
        "eligible": np.ones(n, bool),
        "config": {"smoke": smoke},
    }
    a.update(over)
    return a


def test_check_arms_paired_fires_on_images_eligibility_and_smoke():
    OD.check_arms_paired({"a": _arm(), "b": _arm()}, smoke=False)
    with pytest.raises(SystemExit, match="different images"):
        OD.check_arms_paired({"a": _arm(), "b": _arm(paths=_arm()["paths"][::-1])}, smoke=False)
    with pytest.raises(SystemExit, match="different images"):
        OD.check_arms_paired({"a": _arm(), "b": _arm(rows=np.arange(6) * 2)}, smoke=False)
    with pytest.raises(SystemExit, match="different eligible"):
        OD.check_arms_paired(
            {"a": _arm(), "b": _arm(eligible=np.array([1, 1, 1, 1, 1, 0], bool))}, smoke=False
        )
    with pytest.raises(SystemExit, match="smoke=True cache"):
        OD.check_arms_paired({"a": _arm(), "b": _arm(smoke=True)}, smoke=False)


def test_delta_shares_recover_a_planted_split():
    rng = np.random.default_rng(0)
    n, dim = 30, 5
    const = np.ones(dim) * 2.0
    glob = rng.standard_normal((n, 1, dim))
    local = rng.standard_normal((n, L, dim))
    local -= local.mean(1, keepdims=True)
    d = const + glob + local
    rec = {
        "d_mean": d.mean(1),
        "d_total": (d**2).sum((1, 2)),
        "d_local": ((d - d.mean(1, keepdims=True)) ** 2).sum((1, 2)),
        "d_cos": np.ones(n),
    }
    s = OD.delta_shares(rec, L)
    assert s["constant"] + s["image_global"] + s["local"] == pytest.approx(1.0)
    total = (d**2).sum()
    assert s["local"] == pytest.approx((local**2).sum() / total)
    assert s["constant"] == pytest.approx(n * L * (d.mean((0, 1)) ** 2).sum() / total)
    only_const = {
        "d_mean": np.tile(const, (n, 1)),
        "d_total": np.full(n, L * (const**2).sum()),
        "d_local": np.zeros(n),
        "d_cos": np.ones(n),
    }
    assert OD.delta_shares(only_const, L)["constant"] == pytest.approx(1.0)


def test_global_share_matches_the_direct_decomposition():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((12, L, 6)) + 2.0 * rng.standard_normal((12, 1, 6))
    m = x.mean(1)
    within = ((x - m[:, None]) ** 2).sum((1, 2))
    between = L * ((m - m.mean(0)) ** 2).sum()
    share, draws = OD.global_share({"img_mean": m.astype(np.float32), "ss_within": within}, L, n_boot=50)
    assert share == pytest.approx(between / (between + within.sum()), rel=1e-5)
    assert draws.shape == (50,) and (draws > 0).all() and (draws < 1).all()


def test_boot_ci_uses_finite_entries_and_shares_resamples():
    d = np.r_[np.arange(50.0), np.nan]
    v, lo, hi = OD.boot_ci(d, n_boot=500)
    assert v == pytest.approx(24.5) and lo < v < hi
    assert OD.boot_ci(d + 1.0, n_boot=500)[1] == pytest.approx(lo + 1.0)  # same index stream
    assert np.isnan(OD.boot_ci(np.array([1.0, np.nan]))[0])


def test_strided_covers_the_range():
    assert OD.strided(10, None).tolist() == list(range(10))
    s = OD.strided(403, 48)
    assert s[0] == 0 and s[-1] == 402 and len(s) == 48 and (np.diff(s) > 0).all()


# ----------------------------------------------------------------------------- end to end
def _write_tree(root):
    """Two-class masks: a vertical or horizontal split at a random position (so position alone is
    partly informative, which the swap null must cancel), one image with a single class."""
    rng = np.random.default_rng(0)
    masks = {}
    for split, n in SIZES.items():
        os.makedirs(root / split / "images", exist_ok=True)
        os.makedirs(root / split / "masks", exist_ok=True)
        out = []
        for i in range(n):
            m = np.zeros(GRID, np.uint8)
            cut = int(rng.integers(2, 7))
            if i % 2:
                m[:, cut:] = 1
            else:
                m[cut:, :] = 1
            if i == 3:
                m[:] = 0  # one class only: not eligible
            mask = np.kron(m, np.ones((2, 2), np.uint8))
            if i == 0:
                mask[0, 0] = F.SEG_IGNORE
            Image.fromarray(rng.integers(0, 255, (*MASK_HW, 3), dtype=np.uint8)).save(
                root / split / "images" / f"{split}_{i:03d}.png"
            )
            Image.fromarray(mask).save(root / split / "masks" / f"{split}_{i:03d}.png")
            out.append(m.reshape(-1))
        masks[split] = np.stack(out)
    return masks


def _tokens(lab, signal, seed, pos_scale=1.0):
    """(n, L, C) fp16: per-image class embedding * signal + shared position code + noise."""
    rng = np.random.default_rng(seed)
    pos = OD.position_features(GRID, dim=C, seed=99)
    out = np.zeros((len(lab), L, C), np.float32)
    for i, y in enumerate(lab):
        out[i] = signal * rng.standard_normal((2, C))[y] + pos_scale * pos + 0.5 * rng.standard_normal((L, C))
    return out


def _dino_cache(dirpath, preset, split, images, mfiles, tokens):
    os.makedirs(dirpath, exist_ok=True)
    base = os.path.join(dirpath, f"{preset}_{DS}_{split}_tokens64")
    np.save(base + ".npy", tokens.astype(np.float16))
    meta = {"dataset": DS, "split": split, "n": len(images), "max_images": None, "preset": preset, "weights": "w", "img": 64,
            "grid_hw": list(GRID), "file": os.path.basename(base + ".npy"), "paths": images, "masks": mfiles}  # fmt: skip
    json.dump(meta, open(base + "_meta.json", "w"))


def _flux_cache(dirpath, split, images, tokens_by_t, paths=None):
    run = os.path.join(dirpath, f"run+{split}")
    os.makedirs(run, exist_ok=True)
    files = {}
    for t, tok in tokens_by_t.items():
        files[f"t{t}_k{OD.BLOCK}"] = f"seg_tokens_{split}_t{t}_k{OD.BLOCK}_{TAG}.npy"
        np.save(os.path.join(run, files[f"t{t}_k{OD.BLOCK}"]), tok.astype(np.float16))
    np.savez(os.path.join(run, f"seg_tokens_{split}_{TAG}_meta.npz"), mods=np.zeros((len(tokens_by_t), 1, 3, C), np.float32),
             timesteps=np.array(list(tokens_by_t)), blocks=np.array([OD.BLOCK]), paths=np.array(paths or images))  # fmt: skip
    meta = {"dataset": DS, "split": split, "max_images": None, "weights": "fake-backbone", "discard_channels": [0], "t": list(tokens_by_t),
            "k": [OD.BLOCK], "files": files, "grid_hw": list(GRID), "eps_seed": {"train": 42, "val": 44, "test": 43}[split]}  # fmt: skip
    json.dump(meta, open(os.path.join(run, f"seg_tokens_{split}_{TAG}_meta.json"), "w"))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    root = tmp_path / "tree"
    monkeypatch.setitem(
        F.OFFICIAL_SEG,
        DS,
        {"sizes": dict(SIZES), "root": str(root), "classes": ["bg", "obj"], "mask_hw": MASK_HW},
    )
    monkeypatch.setattr(OD, "DATASETS", (DS,))
    monkeypatch.setattr(OD, "FIG_T", 260)
    monkeypatch.setattr(OD, "N_BOOT", 2000)
    lab = _write_tree(root)
    tok = tmp_path / "tok"
    for split in SIZES:
        images, mfiles = F.list_official_seg_split(DS, split)
        _dino_cache(str(tok / "sat"), "sat", split, images, mfiles, _tokens(lab[split], 2.0, seed=1))
        _dino_cache(str(tok / "web"), "web", split, images, mfiles, _tokens(lab[split], 1.0, seed=2))
        frozen = {t: _tokens(lab[split], 0.1, seed=10 + t) for t in TS}
        # "naive": the frozen tokens plus an object-aligned local change; "proj": plus a constant shift only
        naive = {t: frozen[t] + _tokens(lab[split], 1.0, seed=20 + t, pos_scale=0.0) for t in TS}
        proj = {t: frozen[t] + np.linspace(-1, 1, C, dtype=np.float32) for t in TS}
        _flux_cache(str(tok / "frozen"), split, images, frozen)
        _flux_cache(str(tok / "naive"), split, images, naive)
        _flux_cache(str(tok / "proj"), split, images, proj)

    def flux(name, d):
        return {
            "name": name,
            "kind": "flux",
            "tokens": str(tok / d),
            "tag": TAG,
            "timesteps": list(TS),
            "blocks": [OD.BLOCK],
        }

    def dino(name, d):
        return {"name": name, "kind": "dino", "tokens": str(tok / d), "preset": d, "img": 64, "teacher": None}

    spec = {"datasets": [DS], "models": [dino("dinov3-sat", "sat"), dino("dinov3-web", "web"), flux("flux-pretrained", "frozen"),
                                           flux("flux-naive", "naive"), flux("flux-proj", "proj")]}  # fmt: skip
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return {"spec": str(path), "out": str(tmp_path / "res"), "tok": tok, "lab": lab}


def _run(fake, *extra):
    return OD.main(["run", "--spec", fake["spec"], "--out-dir", fake["out"], "--dataset", DS, *extra])


def test_run_report_figure_end_to_end(fake, capsys):
    assert _run(fake) == 0
    assert "arms=13 computed=13" in capsys.readouterr().out  # 2 dino + 2 timesteps x 3 flux + 5 controls
    arms = sorted(os.listdir(os.path.join(fake["out"], "arms", DS)))
    assert len(arms) == 13 and "flux-naive__t260_k28.npz" in arms and "ctrl-object__ctrl.npz" in arms
    assert _run(fake) == 0
    assert "computed=0" in capsys.readouterr().out  # resumable: nothing recomputed

    a = OD.load_arm(fake["out"], DS, "flux-naive/t100_k28")
    assert (
        a["eligible"].sum() == 23
        and not a["eligible"][3]
        and np.isnan(a["eta2"][3])
        and np.isfinite(a["cka_t"][3])
    )
    assert a["paths"][0] == F.path_key(F.list_official_seg_split(DS, "test")[0][0])
    assert a["img_mean"].shape == (24, C) and a["pca_rgb"].shape == (OD.PCA_N, L, 3)
    t = OD.load_arm(fake["out"], DS, "dinov3-sat/patch")
    assert np.isnan(t["cka_t"]).all() and np.nanmean(t["eta2"] - t["eta2_null"]) > 0.2

    assert OD.main(["report", "--spec", fake["spec"], "--out-dir", fake["out"]]) == 0
    out = capsys.readouterr().out
    res = json.load(open(os.path.join(fake["out"], "report.json")))[DS]
    # The positive controls must be detected. The null controls are NOT asserted on this one seeded
    # draw of 23 images (the gate has a family-wise false-fail rate by design; its calibration is
    # test_null_controls_are_unbiased_and_the_gate_is_calibrated).
    lines = res["gate"]["lines"]
    power = [ln for ln in lines if ln.startswith(("ctrl-object ", "ctrl-teacherproj "))]
    assert len(power) == 4 and all(ln.endswith("PASS") for ln in power), "\n".join(lines)
    assert len([ln for ln in lines if ln.startswith(("ctrl-noise ", "ctrl-position "))]) == 8
    assert ("GATE PASS" in out) == res["gate"]["pass"] and ("do NOT quote any cell above" in out) != res[
        "gate"
    ]["pass"]
    assert "missing_arms=0" in out
    assert res["n_images"] == 24 and res["n_eligible"] == 23
    for t_ in TS:
        naive, proj = res["contrasts"][f"flux-naive/t{t_}_k28"], res["contrasts"][f"flux-proj/t{t_}_k28"]
        assert (
            naive["eta2"][1] > 0 and naive["gap"][1] > 0 and naive["cka_t"][1] > 0
        )  # the planted object-aligned change is detected
        assert (
            proj["eta2"][1] < 0.05 and abs(proj["eta2"][0]) < 0.05
        )  # a constant shift does not make tokens object-aligned
        d_naive, d_proj = res["delta"][f"flux-naive/t{t_}_k28"], res["delta"][f"flux-proj/t{t_}_k28"]
        assert d_naive["d_eta2_diff"][1] > 0  # the local change follows the masks
        assert d_naive["local"] > d_proj["local"]
        assert d_naive["constant"] + d_naive["image_global"] + d_naive["local"] == pytest.approx(1.0)

    assert OD.main(["figure", "--spec", fake["spec"], "--out-dir", fake["out"], "--dataset", DS]) == 0
    assert os.path.getsize(os.path.join(fake["out"], f"{DS}_patch_pca.png")) > 1000


def test_smoke_runs_are_segregated_and_stale_configs_refused(fake, capsys, monkeypatch):
    assert _run(fake, "--max-images", "12") == 0
    assert os.path.isdir(os.path.join(fake["out"], "smoke", "arms", DS)) and not os.path.isdir(
        os.path.join(fake["out"], "arms")
    )
    with pytest.raises(SystemExit, match="no arms under"):
        OD.main(["report", "--spec", fake["spec"], "--out-dir", fake["out"]])
    assert OD.main(["report", "--spec", fake["spec"], "--out-dir", fake["out"], "--smoke"]) == 0
    assert "SMOKE: never quotable" in capsys.readouterr().out
    a = OD.load_arm(os.path.join(fake["out"], "smoke"), DS, "ctrl-object/ctrl")
    assert a["config"]["smoke"] and len(a["rows"]) == 12 and a["rows"][-1] == 23
    assert (
        np.nanmean(a["eta2"] - a["eta2_null"]) > 0.02
    )  # the control read the labels of ITS images (strided rows)
    monkeypatch.setattr(OD, "MIN_TOK", 9)  # a changed constant must not silently reuse the cached arms
    with pytest.raises(SystemExit, match="stored config differs"):
        _run(fake, "--max-images", "12")


def test_run_refuses_a_token_cache_with_reordered_images(fake):
    images = F.list_official_seg_split(DS, "test")[0]
    frozen = {t: np.zeros((24, L, C), np.float32) for t in TS}
    shutil.rmtree(fake["tok"] / "naive" / "run+test")
    _flux_cache(str(fake["tok"] / "naive"), "test", images, frozen, paths=[images[1], images[0], *images[2:]])
    with pytest.raises(SystemExit, match="token cache image list != the official listing"):
        _run(fake)
