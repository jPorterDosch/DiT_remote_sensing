# ruff: noqa: E402
"""Verify the paired 500-image extraction caches are shape-correct and image-paired.

For every cache produced by the paired-extraction jobs (default glob:
models/paired_500/*/multistep_train_feats_*.npz):
  - asserts feats shape (N, K, C) == (--expect-n, --expect-k, --expect-c)
  - asserts mods shape (K, 3, C) and labels length N
  - cross-checks subset_indices / paths / timesteps / block_idx across ALL caches,
    so a mismatch in any pair fails loudly (paired comparison requires it)
  - prints the subset image IDs once so the pairing can be eyeballed

Optionally, --legacy <old.npz> compares a pre-existing one-shot cache (produced by
the original extraction.py) against the new subset to decide whether the g=3.5
arm (job 3) can be skipped. The original cache format recorded neither
guidance_scale nor extraction_mode, so guidance can only be presumed 3.5 — the
script says so explicitly rather than pretending to verify it.

Usage:
    python experiments/verify_paired_caches.py
    python experiments/verify_paired_caches.py models/paired_500/*/multistep_train_feats_*.npz
    python experiments/verify_paired_caches.py --legacy /path/to/old_multistep_train_feats.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

DEFAULT_GLOB = "models/paired_500/*/multistep_train_feats_*.npz"


def load_cache(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["__path__"] = path
    return out


def describe(c: dict, expect_n: int, expect_k: int, expect_c: int) -> list[str]:
    """Assert one cache's shapes; return a list of problems (empty = OK).

    Missing required keys are reported as problems (not raised) so an old/partial npz
    yields a clean FAIL line rather than a KeyError traceback — this is a pre-flight tool.
    """
    problems = []
    if "feats" not in c:
        problems.append("missing 'feats' array")
    elif c["feats"].shape != (expect_n, expect_k, expect_c):
        problems.append(f"feats shape {c['feats'].shape} != ({expect_n}, {expect_k}, {expect_c})")
    if "mods" in c and c["mods"].shape != (expect_k, 3, expect_c):
        problems.append(f"mods shape {c['mods'].shape} != ({expect_k}, 3, {expect_c})")
    if "labels" not in c:
        problems.append("missing 'labels' array")
    elif len(c["labels"]) != expect_n:
        problems.append(f"labels length {len(c['labels'])} != {expect_n}")
    if "subset_indices" in c and len(c["subset_indices"]) != expect_n:
        problems.append(f"subset_indices length {len(c['subset_indices'])} != {expect_n}")
    return problems


def tag_of(c: dict) -> str:
    mode = str(c["extraction_mode"]) if "extraction_mode" in c else "ONESHOT(untagged legacy)"
    g = str(c["guidance_scale"]) if "guidance_scale" in c else "3.5?(unrecorded)"
    extras = []
    if "num_inversion_steps" in c:
        extras.append(f"n={c['num_inversion_steps']}")
    if "eps_seed" in c:
        extras.append(f"eps_seed={c['eps_seed']}")
    if "ensemble_size" in c:
        extras.append(f"ensemble={c['ensemble_size']}")
    return f"{mode} g={g}" + (f" ({', '.join(extras)})" if extras else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="*", help=f"cache npz paths (default: glob {DEFAULT_GLOB})")
    ap.add_argument("--expect-n", type=int, default=500)
    ap.add_argument("--expect-k", type=int, default=7)
    ap.add_argument("--expect-c", type=int, default=3072)
    ap.add_argument(
        "--legacy", default=None, help="pre-existing one-shot cache to compare (job-3 skip decision)"
    )
    ap.add_argument("--quiet-ids", action="store_true", help="print only the first/last 5 image IDs")
    args = ap.parse_args()

    paths = args.caches or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print(f"ERROR: no caches found (glob: {DEFAULT_GLOB}) — nothing extracted yet?")
        return 2

    caches = [load_cache(p) for p in paths]
    failed = False

    print(
        f"checking {len(caches)} cache(s), expecting feats ({args.expect_n}, {args.expect_k}, {args.expect_c})\n"
    )
    for c in caches:
        problems = describe(c, args.expect_n, args.expect_k, args.expect_c)
        status = "OK " if not problems else "FAIL"
        print(f"[{status}] {c['__path__']}")
        t_str = c["timesteps"].tolist() if "timesteps" in c else "MISSING"
        k_str = c["block_idx"] if "block_idx" in c else "MISSING"
        print(f"       {tag_of(c)}; t={t_str}; k={k_str}; subset_seed={c.get('subset_seed', 'MISSING')}")
        for p in problems:
            print(f"       PROBLEM: {p}")
            failed = True

    # --- cross-cache pairing: every cache must cover the identical images/settings
    ref = caches[0]
    print("\ncross-cache pairing vs", os.path.basename(os.path.dirname(ref["__path__"])) or ref["__path__"])
    for c in caches[1:]:
        for key in ("subset_indices", "paths", "timesteps", "block_idx", "labels"):
            if key not in ref or key not in c:
                print(f"  WARN: {key} missing in one of the caches — cannot assert pairing on it")
                continue
            if not np.array_equal(ref[key], c[key]):
                print(f"  FAIL: {key} differs between {ref['__path__']} and {c['__path__']}")
                failed = True
    if len(caches) > 1 and not failed:
        print("  all caches are IMAGE-PAIRED (identical subset_indices, paths, t, k, labels)")

    # --- legacy cache comparison (job-3 skip decision)
    if args.legacy:
        legacy = load_cache(args.legacy)
        print(f"\nlegacy cache: {args.legacy}\n  {tag_of(legacy)}")
        legacy_ok = True
        for key in ("subset_indices", "paths", "timesteps", "block_idx"):
            if key not in legacy:
                print(f"  WARN: legacy cache has no '{key}' — cannot verify, treat as NOT reusable")
                legacy_ok = False
            elif not np.array_equal(ref[key], legacy[key]):
                print(f"  MISMATCH on {key} — legacy cache covers different images/settings")
                legacy_ok = False
        if "guidance_scale" not in legacy:
            print(
                "  NOTE: legacy format records no guidance — presumed 3.5 (the old hard-coded default), UNVERIFIED"
            )
        if "ensemble_size" in legacy and int(np.atleast_1d(legacy["ensemble_size"])[0]) != 1:
            print(
                f"  MISMATCH: legacy ensemble_size={legacy['ensemble_size']} != 1 (the new jobs' setting) — "
                "NOT the same object as the paired arms"
            )
            legacy_ok = False
        print(
            "  => LEGACY MATCHES — job 3 (oneshot g3.5) can be SKIPPED"
            if legacy_ok
            else "  => legacy NOT confirmed reusable — submit job 3 (extract_500_oneshot_g3.5.sbatch)"
        )

    # --- image IDs (shared across all paired caches)
    if "paths" in ref:
        ids = [str(p) for p in ref["paths"]]
        print(f"\nsubset image IDs ({len(ids)} images, identical across paired caches):")
        shown = ids if not args.quiet_ids else ids[:5] + ["..."] + ids[-5:]
        for s in shown:
            print("  ", s)

    if failed:
        print("\nVERIFICATION FAILED — see problems above")
        return 1
    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
