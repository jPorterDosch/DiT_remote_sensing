"""Reproduction gates for the eval pipeline (rule 13: executed, PASS/FAIL/SKIP printed).

    pytest tests/test_eval_gates.py -v -rs            # CPU gates (~1 h)
    pytest tests/test_eval_gates.py -v -rs --gpu      # + GPU gates (DINO extraction, MLP head)
    pytest tests/test_eval_gates.py -v -rs -k budget  # one gate

Each gate drives a real entry point (eval.probe / compare / extract_dino /
export_geobench) and demands it reproduce the banked per-image vectors of the prototype
it replaced. W&B is disabled: gates are instrument checks, not results. A gate whose
inputs are absent on this machine SKIPs (with the missing paths as the reason); it never
passes vacuously. Run after any change under eval/.
"""

from __future__ import annotations

import filecmp
import json
import os

os.environ["WANDB_MODE"] = "disabled"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from eval import compare, probe  # noqa: E402
from eval import features as F  # noqa: E402
from eval import protocols as P  # noqa: E402

R45_INV = F.CV_IDENTITY["resisc45"]
R45_ENS8 = (
    "models/n5000_resisc45_oneshot_ens8/resisc45_flux_4118f153+42/multistep_train_feats_oneshot_g1.0.npz"
)
DINO_R45 = "results/dinov2_resisc45_feats_n5000.npz"
VAE_R45 = "results/vae_latents_resisc45_n5000.npz"
REF_6AC = "results/dinov2_resisc45_paired.npz"
EXACT = 1e-9


def need(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"inputs absent: {missing}")


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("eval_gates"))


def run_probe(out_dir, *argv) -> dict:
    out = probe.main([*argv, "--out-dir", out_dir])
    d = np.load(out, allow_pickle=True)
    return {str(c): (d[f"correct__{c}"], d[f"ev__{c}"]) for c in d["cells"]} | {"_path": out}


def maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


# --------------------------------------------------------------------------------- cv
CV_ARMS = {
    "A1": [
        "--arm",
        "flux-inv-sec13",
        "--kind",
        "flux",
        "--features",
        R45_INV,
        "--expect",
        "extraction_mode=INVERSION",
        "num_inversion_steps=50",
        "--view",
        "sec13:1",
    ],
    "A2": [
        "--arm",
        "flux-ens8-t260",
        "--kind",
        "flux",
        "--features",
        R45_ENS8,
        "--expect",
        "extraction_mode=ONESHOT",
        "ensemble_size=8",
        "--view",
        "t:2",
    ],
    "B1": ["--arm", "dinov2-clsmp", "--kind", "dino", "--features", DINO_R45, "--view", "clsmp"],
    "B2": ["--arm", "dinov2-cls", "--kind", "dino", "--features", DINO_R45, "--view", "cls"],
    "B3": [
        "--arm",
        "dinov2-clsmp-nestedC",
        "--kind",
        "dino",
        "--features",
        DINO_R45,
        "--view",
        "clsmp",
        "--nested-c",
        "0.01",
        "0.1",
        "1.0",
    ],
    "B1p": [
        "--arm",
        "dinov2-clsmp-pca512",
        "--kind",
        "dino",
        "--features",
        DINO_R45,
        "--view",
        "clsmp",
        "--pca",
        "512",
    ],
}


@pytest.fixture(scope="module")
def cv_runs(out_dir):
    """The six 6ac arms through eval.probe --protocol cv (shared by the cv and compare gates)."""
    need(R45_INV, R45_ENS8, DINO_R45, REF_6AC)
    return {
        k: run_probe(out_dir, "--protocol", "cv", "--dataset", "resisc45", *a) for k, a in CV_ARMS.items()
    }


def test_fold_identity():
    """Pairing premise: CV folds depend only on (n, y, seed); the check fires on permuted y."""
    need(R45_INV)
    _, y = F.load_identity(R45_INV)
    f = [tuple(va) for _, va in P.cv_folds(y, 0)]
    fx = [
        tuple(va)
        for _, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.random.rand(len(y), 7), y)
    ]
    fp = [tuple(va) for _, va in P.cv_folds(y[np.random.default_rng(0).permutation(len(y))], 0)]
    assert f == fx, "folds depend on X"
    assert f != fp, "guard cannot fire: permuted labels gave identical folds"


def test_cv_linear(cv_runs):
    """6ac arms A1/A2/B1/B2/B3/B1p reproduce results/dinov2_resisc45_paired.npz bit-for-bit."""
    ref = np.load(REF_6AC, allow_pickle=True)
    for k, r in cv_runs.items():
        dk = maxdiff(r["cv"][0], ref[k])
        assert dk < EXACT, f"{k}: max |new - banked| = {dk:.3g}"


def test_compare(cv_runs, out_dir):
    """eval.compare reproduces the 6ac PRIMARY statistic, and its pairing guard fires."""
    (r,) = compare.compare(cv_runs["A1"]["_path"], cv_runs["B1"]["_path"])
    ref = np.load(REF_6AC, allow_pickle=True)
    d = ref["B1"] - ref["A1"]
    lo, hi = P.ci(d)
    assert (
        abs(r["delta"] - d.mean()) < EXACT and abs(r["ci_lo"] - lo) < EXACT and abs(r["ci_hi"] - hi) < EXACT
    )
    bad = os.path.join(out_dir, "permuted.npz")
    z = dict(np.load(cv_runs["B1"]["_path"], allow_pickle=True))
    z["paths"] = z["paths"][::-1]
    np.savez(bad, **z)
    with pytest.raises(SystemExit, match="UNPAIRED"):
        compare.compare(cv_runs["A1"]["_path"], bad)


def test_cv_vae(out_dir):
    """Q1 VAE poolings reproduce results/vae_fullwidth_probe_resisc45.npz bit-for-bit."""
    ref_path = "results/vae_fullwidth_probe_resisc45.npz"
    need(VAE_R45, ref_path, R45_INV)
    ref = np.load(ref_path, allow_pickle=True)
    views = {
        "full 32x32 (16384d)": "full",
        "pool 4x4 (256d)": "pool8",  # prototype label is wrong: 4x4 WINDOWS = 8x8 grid, 1024-d
        "pool 2x2 (64d)": "pool2",
        "pool 1x1 (16d)": "pool1",
    }
    for i, n in enumerate(str(x) for x in ref["names"]):
        r = run_probe(
            out_dir,
            "--protocol",
            "cv",
            "--dataset",
            "resisc45",
            "--arm",
            f"vae-{views[n]}",
            "--kind",
            "vae",
            "--features",
            VAE_R45,
            "--view",
            views[n],
        )
        dk = maxdiff(r["cv"][0], ref[f"vae_{i}"])
        assert dk < EXACT, f"{n}: max |diff| {dk:.3g}"


def test_budget(out_dir):
    """Q3 label-budget cells (FLUX sec13 + DINOv2 clsmp) reproduce bit-for-bit."""
    ref_path = "results/label_budget_curves_resisc45.npz"
    need(ref_path, R45_INV, DINO_R45)
    ref = np.load(ref_path, allow_pickle=True)
    common = ["--protocol", "budget", "--dataset", "resisc45"]
    rf = run_probe(
        out_dir,
        *common,
        "--arm",
        "flux-inv-sec13",
        "--kind",
        "flux",
        "--features",
        R45_INV,
        "--expect",
        "extraction_mode=INVERSION",
        "num_inversion_steps=50",
        "--view",
        "sec13:1",
    )
    rd = run_probe(
        out_dir, *common, "--arm", "dinov2-clsmp", "--kind", "dino", "--features", DINO_R45, "--view", "clsmp"
    )
    for b in P.BUDGETS:
        for s in P.SEEDS:
            c = f"b{b}_s{s}"
            assert np.array_equal(rf[c][1], ref[f"ev_{c}"]), f"{c}: eval rows differ"
            assert np.array_equal(rf[c][0], ref[f"flux_{c}"]), f"{c}: FLUX vector differs"
            assert np.array_equal(rd[c][0], ref[f"dino_{c}"]), f"{c}: DINO vector differs"


def test_export_m_eurosat(tmp_path):
    """export_geobench --task m-eurosat is byte-identical to the existing data/m_eurosat_rgb."""
    src, ref = "data/m_eurosat_meta", F.OFFICIAL["m_eurosat"]["root"]
    need(src, ref)
    pytest.importorskip("geobench")
    from eval import export_geobench

    out = str(tmp_path / "m_eurosat_rgb")
    export_geobench.export("m-eurosat", src, out)
    n, bad = 0, []
    for root, _, files in os.walk(ref):
        for f in files:
            a = os.path.join(root, f)
            b = os.path.join(out, os.path.relpath(a, ref))
            n += 1
            if not (os.path.exists(b) and filecmp.cmp(a, b, shallow=False)):
                bad.append(os.path.relpath(a, ref))
    extra = sum(len(fs) for _, _, fs in os.walk(out)) - n
    assert not bad and extra == 0, f"{len(bad)} differing/missing, {extra} extra (first: {bad[:3]})"


def test_official_flux_m_eurosat(out_dir):
    """(ISAAC) probe --protocol official on the FLUX m-eurosat caches reproduces m_eurosat_probe.npz."""
    ref_path = "results/m_eurosat_probe.npz"
    arms = {
        "oneshot_ens8": (
            "models/m_eurosat_oneshot_ens8/*/multistep_{split}_feats_oneshot_g1.0.npz",
            ["extraction_mode=ONESHOT", "ensemble_size=8"],
        ),
        "inversion": (
            "models/m_eurosat_inversion/*/multistep_{split}_feats_inversion_g1.0_n50.npz",
            ["extraction_mode=INVERSION", "num_inversion_steps=50"],
        ),
    }
    need(ref_path, "models/m_eurosat_oneshot_ens8", "models/m_eurosat_inversion")
    ref = np.load(ref_path, allow_pickle=True)
    for arm, (pat, expect) in arms.items():
        r = run_probe(
            out_dir,
            "--protocol",
            "official",
            "--dataset",
            "m_eurosat",
            "--arm",
            f"flux-{arm}",
            "--kind",
            "flux",
            "--features",
            pat,
            "--expect",
            *expect,
        )
        assert np.array_equal(r["test"][0], ref[f"{arm}_correct_test"]), f"{arm}: test vector differs"


# -------------------------------------------------------------------------------- GPU
@pytest.mark.gpu
def test_extract_dino_resisc45(out_dir):
    """Fresh DINOv2 extraction reproduces the 6ac feature cache (atol 1e-4)."""
    need(DINO_R45, R45_INV)
    from eval import extract_dino

    (out,) = extract_dino.main(["--preset", "dinov2_vitl14", "--dataset", "resisc45", "--out-dir", out_dir])
    new, ref = np.load(out, allow_pickle=True), np.load(DINO_R45, allow_pickle=True)
    assert list(new["paths"]) == list(ref["paths"]), "path order differs"
    assert np.allclose(new["cls"], ref["cls"], atol=1e-4), maxdiff(new["cls"], ref["cls"])
    assert np.allclose(new["mp"], ref["mp"], atol=1e-4), maxdiff(new["mp"], ref["mp"])


@pytest.mark.gpu
def test_official_dino_m_eurosat(out_dir):
    """extract_dino + probe --protocol official reproduce results/dinov2_m_eurosat.npz exactly."""
    ref_path = "results/dinov2_m_eurosat.npz"
    need(ref_path, F.OFFICIAL["m_eurosat"]["root"])
    from eval import extract_dino

    extract_dino.main(["--preset", "dinov2_vitl14", "--dataset", "m_eurosat", "--out-dir", out_dir])
    pat = os.path.join(out_dir, "dinov2_vitl14_m_eurosat_{split}.npz")
    r = run_probe(
        out_dir,
        "--protocol",
        "official",
        "--dataset",
        "m_eurosat",
        "--arm",
        "dinov2",
        "--kind",
        "dino",
        "--features",
        pat,
    )
    ref = np.load(ref_path, allow_pickle=True)
    sel = json.loads(str(np.load(r["_path"], allow_pickle=True)["info"]))["selected"]
    want = str(ref["selected"]).split(",")  # "variant,C=..,val=..,test=..,f1=.."
    assert sel["candidate"] == want[0] and f"C={sel['C']}" == want[1], (sel, want)
    assert np.array_equal(r["test"][0], ref["correct_test"]), "test vector differs"


@pytest.mark.gpu
def test_mlp(out_dir):
    """Q2 matched MLP head: per-seed accuracy within 0.01 and >=97% per-image agreement."""
    ref_path = "results/matched_head_resisc45.npz"
    need(ref_path, R45_INV, DINO_R45)
    ref = np.load(ref_path, allow_pickle=True)
    common = ["--protocol", "mlp", "--dataset", "resisc45"]
    runs = {
        "flux_7t": run_probe(
            out_dir,
            *common,
            "--arm",
            "flux-inv-concat",
            "--kind",
            "flux",
            "--features",
            R45_INV,
            "--expect",
            "extraction_mode=INVERSION",
            "num_inversion_steps=50",
            "--view",
            "concat",
        ),
        "dino_clsmp": run_probe(
            out_dir,
            *common,
            "--arm",
            "dinov2-clsmp",
            "--kind",
            "dino",
            "--features",
            DINO_R45,
            "--view",
            "clsmp",
        ),
    }
    for name, r in runs.items():
        for s in P.SEEDS:
            c, ev = r[f"s{s}"]
            rc, rev = ref[f"{name}_s{s}_correct"], ref[f"{name}_s{s}_eval_idx"]
            assert np.array_equal(ev, rev), f"{name} s{s}: eval split differs"
            dacc, agree = abs(c.mean() - rc.mean()), float((c == rc).mean())
            assert dacc < 0.01 and agree >= 0.97, f"{name}/s{s} dacc {dacc:.4f} agree {agree:.3f}"


def test_sweep_k28_reproduces_official(out_dir):
    """(ISAAC) eval.sweep at k=28 alone selects the same cell and gives the same test vector
    as the banked m_eurosat_probe oneshot_ens8 arm (official_all_cells == run_official)."""
    ref_path = "results/m_eurosat_probe.npz"
    need(ref_path, "models/m_eurosat_oneshot_ens8")
    from eval import sweep

    blk = sweep.main(
        [
            "block",
            "--dataset",
            "m_eurosat",
            "--k",
            "28",
            "--arm",
            "flux-oneshot-ens8",
            "--features",
            "models/m_eurosat_oneshot_ens8/*/multistep_{split}_feats_oneshot_g1.0.npz",
            "--expect",
            "extraction_mode=ONESHOT",
            "ensemble_size=8",
            "--out-dir",
            out_dir,
        ]
    )
    res = np.load(
        sweep.main(
            [
                "select",
                "--dataset",
                "m_eurosat",
                "--arm",
                "flux-oneshot-ens8",
                "--blocks",
                blk,
                "--out-dir",
                out_dir,
            ]
        ),
        allow_pickle=True,
    )
    ref = np.load(ref_path, allow_pickle=True)
    sel = json.loads(str(res["info"]))["selected"]
    want = str(ref["oneshot_ens8_selected"]).split(",")  # "cand,C=..,val=..,test=..,..."
    assert sel["candidate"] == want[0] and f"C={sel['C']}" == want[1], (sel, want)
    assert np.array_equal(res["correct__test"], ref["oneshot_ens8_correct_test"]), "test vector differs"
