"""Reproduction gates for the eval pipeline (rule 13: executed code, PASS/FAIL printed).

    python -m eval.gates                  # all CPU gates
    python -m eval.gates --gpu            # + GPU gates (DINO extraction, MLP head)
    python -m eval.gates cv_linear budget # just these

Each gate drives the real CLI entry point (probe/compare/extract_dino/export_geobench)
and demands it reproduce the banked per-image vectors of the prototype it replaced.
W&B is disabled here: gates are instrument checks, not results. SKIP (never PASS) when a
gate's inputs are absent on this machine; exit status is nonzero on any FAIL.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import sys
import tempfile
import traceback

os.environ["WANDB_MODE"] = "disabled"

import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
TMP = tempfile.mkdtemp(prefix="eval_gates_")
EXACT = 1e-9


class Skip(Exception):
    pass


def need(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise Skip(f"inputs absent: {missing}")


def run_probe(*argv) -> dict:
    out = probe.main([*argv, "--out-dir", TMP])
    d = np.load(out, allow_pickle=True)
    return {str(c): (d[f"correct__{c}"], d[f"ev__{c}"]) for c in d["cells"]} | {"_path": out}


def maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


# ------------------------------------------------------------------------------ gates
def g_fold_identity():
    """Pairing premise: CV folds depend only on (n, y, seed); the check must fire on permuted y."""
    need(R45_INV)
    _, y = F.load_identity(R45_INV)
    f = [tuple(va) for _, va in P.cv_folds(y, 0)]
    from sklearn.model_selection import StratifiedKFold

    fx = [
        tuple(va)
        for _, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.random.rand(len(y), 7), y)
    ]
    fp = [tuple(va) for _, va in P.cv_folds(y[np.random.default_rng(0).permutation(len(y))], 0)]
    assert f == fx, "folds depend on X"
    assert f != fp, "guard cannot fire: permuted labels gave identical folds"
    return "folds X-independent; FIRED on permuted labels"


def g_cv_linear():
    """6ac arms A1/A2/B1/B2/B3/B1p through eval.probe --protocol cv, bit-identical."""
    need(R45_INV, R45_ENS8, DINO_R45, REF_6AC)
    ref = np.load(REF_6AC, allow_pickle=True)
    common = ["--protocol", "cv", "--dataset", "resisc45"]
    arms = {
        "A1": ["--arm", "flux-inv-sec13", "--kind", "flux", "--features", R45_INV, "--view", "sec13:1"],
        "A2": ["--arm", "flux-ens8-t260", "--kind", "flux", "--features", R45_ENS8, "--view", "t:2"],
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
    worst, outs = 0.0, {}
    for k, a in arms.items():
        r = run_probe(*common, *a)
        outs[k] = r["_path"]
        dk = maxdiff(r["cv"][0], ref[k])
        worst = max(worst, dk)
        assert dk < EXACT, f"{k}: max |new - banked| = {dk:.3g}"
    g_cv_linear.outputs = outs
    return f"6 arms reproduce results/dinov2_resisc45_paired.npz (max |diff| {worst:.1e})"


def g_compare():
    """eval.compare reproduces the 6ac PRIMARY statistic, and its pairing guard fires."""
    need(REF_6AC)
    outs = getattr(g_cv_linear, "outputs", None)
    if not outs:
        raise Skip("needs cv_linear outputs (run cv_linear in the same invocation)")
    rows = compare.compare(outs["A1"], outs["B1"])
    ref = np.load(REF_6AC, allow_pickle=True)
    d = ref["B1"] - ref["A1"]
    lo, hi = P.ci(d)
    (r,) = rows
    assert (
        abs(r["delta"] - d.mean()) < EXACT and abs(r["ci_lo"] - lo) < EXACT and abs(r["ci_hi"] - hi) < EXACT
    )
    # fire-test: a result over a permuted image order must be refused
    bad = os.path.join(TMP, "permuted.npz")
    z = dict(np.load(outs["B1"], allow_pickle=True))
    z["paths"] = z["paths"][::-1]
    np.savez(bad, **z)
    try:
        compare.compare(outs["A1"], bad)
    except SystemExit as e:
        fired = str(e)
    else:
        raise AssertionError("pairing guard did NOT fire on a reversed identity list")
    return f"B1-A1 {r['delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] matches 6ac; guard FIRED ({fired[:40]}...)"


def g_cv_vae():
    """Q1 VAE poolings, bit-identical to results/vae_fullwidth_probe_resisc45.npz."""
    ref_path = "results/vae_fullwidth_probe_resisc45.npz"
    need(VAE_R45, ref_path, R45_INV)
    ref = np.load(ref_path, allow_pickle=True)
    names = [str(n) for n in ref["names"]]
    views = {
        "full 32x32 (16384d)": "full",
        "pool 4x4 (256d)": "pool4",
        "pool 2x2 (64d)": "pool2",
        "pool 1x1 (16d)": "pool1",
    }
    worst = 0.0
    for i, n in enumerate(names):
        r = run_probe(
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
        worst = max(worst, dk)
        assert dk < EXACT, f"{n}: max |diff| {dk:.3g}"
    return f"4 VAE views reproduce Q1 (max |diff| {worst:.1e})"


def g_budget():
    """Q3 label-budget cells (FLUX sec13 + DINOv2 clsmp), bit-identical."""
    ref_path = "results/label_budget_curves_resisc45.npz"
    need(ref_path, R45_INV, DINO_R45)
    ref = np.load(ref_path, allow_pickle=True)
    common = ["--protocol", "budget", "--dataset", "resisc45"]
    rf = run_probe(
        *common, "--arm", "flux-inv-sec13", "--kind", "flux", "--features", R45_INV, "--view", "sec13:1"
    )
    rd = run_probe(
        *common, "--arm", "dinov2-clsmp", "--kind", "dino", "--features", DINO_R45, "--view", "clsmp"
    )
    n = 0
    for b in P.BUDGETS:
        for s in P.SEEDS:
            c = f"b{b}_s{s}"
            assert np.array_equal(rf[c][1], ref[f"ev_{c}"]), f"{c}: eval rows differ"
            assert np.array_equal(rf[c][0], ref[f"flux_{c}"]), f"{c}: FLUX vector differs"
            assert np.array_equal(rd[c][0], ref[f"dino_{c}"]), f"{c}: DINO vector differs"
            n += 1
    return f"{n} budget cells x 2 arms bit-identical to Q3"


def g_mlp():
    """Q2 matched MLP head (GPU): per-seed accuracy within 0.01 and >=97% per-image agreement."""
    ref_path = "results/matched_head_resisc45.npz"
    need(ref_path, R45_INV, DINO_R45)
    ref = np.load(ref_path, allow_pickle=True)
    common = ["--protocol", "mlp", "--dataset", "resisc45"]
    runs = {
        "flux_7t": run_probe(
            *common, "--arm", "flux-inv-concat", "--kind", "flux", "--features", R45_INV, "--view", "concat"
        ),
        "dino_clsmp": run_probe(
            *common, "--arm", "dinov2-clsmp", "--kind", "dino", "--features", DINO_R45, "--view", "clsmp"
        ),
    }
    msgs = []
    for name, r in runs.items():
        for s in P.SEEDS:
            c, ev = r[f"s{s}"]
            rc, rev = ref[f"{name}_s{s}_correct"], ref[f"{name}_s{s}_eval_idx"]
            assert np.array_equal(ev, rev), f"{name} s{s}: eval split differs"
            dacc, agree = abs(c.mean() - rc.mean()), float((c == rc).mean())
            msgs.append(f"{name}/s{s} dacc {dacc:.4f} agree {agree:.3f}")
            assert dacc < 0.01 and agree >= 0.97, msgs[-1]
    return "; ".join(msgs)


def g_extract_dino_resisc45():
    """GPU: fresh DINOv2 extraction reproduces the 6ac feature cache (atol 1e-4)."""
    need(DINO_R45, R45_INV)
    from eval import extract_dino

    (out,) = extract_dino.main(["--preset", "dinov2_vitl14", "--dataset", "resisc45", "--out-dir", TMP])
    new, ref = np.load(out, allow_pickle=True), np.load(DINO_R45, allow_pickle=True)
    assert list(new["paths"]) == list(ref["paths"]), "path order differs"
    dc, dm = maxdiff(new["cls"], ref["cls"]), maxdiff(new["mp"], ref["mp"])
    assert np.allclose(new["cls"], ref["cls"], atol=1e-4) and np.allclose(new["mp"], ref["mp"], atol=1e-4), (
        dc,
        dm,
    )
    return f"cls max|diff| {dc:.1e}, mp {dm:.1e}"


def g_official_dino_m_eurosat():
    """GPU: extract_dino + probe --protocol official reproduce results/dinov2_m_eurosat.npz exactly."""
    ref_path = "results/dinov2_m_eurosat.npz"
    need(ref_path, F.OFFICIAL["m_eurosat"]["root"])
    from eval import extract_dino

    extract_dino.main(["--preset", "dinov2_vitl14", "--dataset", "m_eurosat", "--out-dir", TMP])
    pat = os.path.join(TMP, "dinov2_vitl14_m_eurosat_{split}.npz")
    r = run_probe(
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
    info = __import__("json").loads(str(np.load(r["_path"], allow_pickle=True)["info"]))
    sel = info["selected"]
    want = str(ref["selected"]).split(",")  # "variant,C=..,val=..,test=..,f1=.."
    assert sel["candidate"] == want[0] and f"C={sel['C']}" == want[1], (sel, want)
    assert np.array_equal(r["test"][0], ref["correct_test"]), "test vector differs"
    return f"selected {sel['candidate']} C={sel['C']}; test vector identical ({r['test'][0].mean():.4f})"


def g_official_flux_m_eurosat():
    """ISAAC: probe --protocol official on the FLUX m-eurosat caches reproduces m_eurosat_probe.npz."""
    ref_path = "results/m_eurosat_probe.npz"
    arms = {
        "oneshot_ens8": "models/m_eurosat_oneshot_ens8/*/multistep_{split}_feats_oneshot_g1.0.npz",
        "inversion": "models/m_eurosat_inversion/*/multistep_{split}_feats_inversion_g1.0_n50.npz",
    }
    need(ref_path, "models/m_eurosat_oneshot_ens8", "models/m_eurosat_inversion")
    ref = np.load(ref_path, allow_pickle=True)
    msgs = []
    for arm, pat in arms.items():
        r = run_probe(
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
        )
        assert np.array_equal(r["test"][0], ref[f"{arm}_correct_test"]), f"{arm}: test vector differs"
        msgs.append(f"{arm} {r['test'][0].mean():.4f}")
    return "identical: " + ", ".join(msgs)


def g_export_m_eurosat():
    """export_geobench --task m-eurosat is byte-identical to the existing data/m_eurosat_rgb."""
    src, ref = "data/m_eurosat_meta", F.OFFICIAL["m_eurosat"]["root"]
    need(src, ref)
    try:
        import geobench  # noqa: F401
    except ImportError as e:
        raise Skip(f"geobench not importable: {e}") from None
    from eval import export_geobench

    out = os.path.join(TMP, "m_eurosat_rgb")
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
    return f"{n} PNGs byte-identical"


CPU = [
    "fold_identity",
    "cv_linear",
    "compare",
    "cv_vae",
    "budget",
    "export_m_eurosat",
    "official_flux_m_eurosat",
]
GPU = ["extract_dino_resisc45", "official_dino_m_eurosat", "mlp"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help=f"subset of {CPU + GPU}")
    p.add_argument("--gpu", action="store_true", help="also run the GPU gates")
    args = p.parse_args(argv)
    names = args.names or (CPU + GPU if args.gpu else CPU)
    results = {}
    for n in names:
        fn = globals().get(f"g_{n}")
        if fn is None:
            raise SystemExit(f"unknown gate {n}")
        print(f"\n########## gate {n}: {fn.__doc__.splitlines()[0]}", flush=True)
        try:
            results[n] = ("PASS", fn())
        except Skip as e:
            results[n] = ("SKIP", str(e))
        except (AssertionError, SystemExit) as e:
            traceback.print_exc()
            results[n] = ("FAIL", str(e))
    print("\n================ GATES ================")
    for n, (st, msg) in results.items():
        print(f"{st:<4}  {n:<26} {msg}")
    return 1 if any(st == "FAIL" for st, _ in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
