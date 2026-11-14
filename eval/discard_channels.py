"""Massive-activation ("discard") channels for the offline DiTF view, by a fixed rule.

DiTF zeroes a backbone's massive-activation channels before its per-token LayerNorm. For
FLUX.1-dev those are the published [154, 1446] (eval/features.DISCARD). Another backbone needs
its own list, so this module defines ONE pre-registered rule and checks it against FLUX:

    channel c is massive  <=>  mean |feats[:n_images, :, c]|  >  factor x median over channels

with n_images = 1000 (subset order), all cached timesteps, factor = 50, at the arm's aligned
block. The rule is only trusted if it reproduces [154, 1446] on a FLUX k=28 cache (spec §4.5):

    python -m eval.discard_channels --features <flux multi-block train cache> --k 28 --expect 154 1446

prints DISCARD_OK. For a GeoCore cache the extraction job runs the rule on the finished TRAIN
cache at k=8 and stamps the list into every split's meta (`--stamp`, with the rule and the source
cache's sha1 recorded as "discard_rule"); the probes refuse a non-FLUX cache without the stamp.

The FLUX check is part of stamping, not a separate step someone has to remember (review
2026-11-12, item 10: it had run once, by hand, as job 6532058, and no job script called it).
`--stamp` requires `--flux-ref <FLUX multi-block train cache>`: the same factor / n_images are run
on that cache at k=28 first, and a list other than [154, 1446] refuses the stamp (DISCARD_FAIL,
exit 1). The reference cache and its sha1 are recorded in "discard_rule" as "flux_check".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FACTOR = 50.0
N_IMAGES = 1000
FLUX_K = 28  # FLUX.1-dev's aligned block: the block the published [154, 1446] refers to


def massive_channels(feats: np.ndarray, factor: float = FACTOR, n_images: int = N_IMAGES) -> list[int]:
    """feats (N, K, C) pooled pre-normalization features; returns the sorted massive channels."""
    if feats.ndim != 3:
        raise ValueError(f"feats must be (N, K, C), got {feats.shape}")
    a = np.abs(np.asarray(feats[:n_images], dtype=np.float64)).mean(axis=(0, 1))  # (C,)
    med = float(np.median(a))
    if med <= 0:
        raise ValueError("median channel magnitude is zero; the features are degenerate")
    return [int(c) for c in np.flatnonzero(a > factor * med)]


def rule_on_cache(path: str, k: int, factor: float, n_images: int) -> tuple[list[int], dict]:
    """The rule on block k of one cache -> (massive channels, the cache's meta)."""
    from eval import features as F

    meta = F.cache_meta(path)
    if meta.get("layout") == "N,K,B,C":
        d = F._block_slice(path, meta, {"k": k})
    else:
        if meta.get("k") != k:
            raise SystemExit(f"{path}: cache block k={meta.get('k')} != --k {k}")
        d = np.load(path)
    return massive_channels(d["feats"], factor, n_images), meta


def flux_reference_check(path: str, factor: float, n_images: int) -> dict:
    """The rule (same factor / n_images as the stamp) must reproduce FLUX's published list on a FLUX
    cache at k=28. -> the record stored in "discard_rule"; SystemExit (DISCARD_FAIL) otherwise."""
    from eval import features as F
    from eval import wb

    found, meta = rule_on_cache(path, FLUX_K, factor, n_images)
    # The PRETRAINED weights only. F.is_flux_meta also accepts "flux-dev+lora:*" and "random_init"
    # (right for block counting), but the published list describes frozen FLUX.1-dev: an adapted or
    # randomly initialised model that happens to give it validates nothing. Absent = historical flux-dev.
    weights = meta.get("weights", "flux-dev") or "flux-dev"
    if weights != "flux-dev":
        raise SystemExit(
            f"DISCARD_FAIL {path}: weights={meta.get('weights')!r} is not a FLUX cache with the pretrained "
            f"weights ('flux-dev'); the rule is validated against frozen FLUX's published {sorted(F.DISCARD)} only"
        )
    if found != sorted(F.DISCARD):
        raise SystemExit(
            f"DISCARD_FAIL rule (factor {factor}, n_images {n_images}) on the FLUX reference {path} at k={FLUX_K} "
            f"gives {found}, expected {sorted(F.DISCARD)}: the rule is not validated, nothing is stamped"
        )
    print(f"DISCARD_OK FLUX reference k={FLUX_K}: {found} (factor {factor}, n_images {n_images}) {path}")
    return {"cache": os.path.relpath(path, REPO_ROOT), "sha1": wb.file_identity(path)["sha1"], "k": FLUX_K, "found": found}  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="one cache .npz (single- or multi-block)")
    ap.add_argument("--k", type=int, required=True, help="block to analyse (multi-block caches are sliced)")
    ap.add_argument("--factor", type=float, default=FACTOR)
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument(
        "--expect", type=int, nargs="*", default=None, help="expected list; prints DISCARD_OK / DISCARD_FAIL"
    )
    ap.add_argument("--stamp", nargs="*", default=None, metavar="CACHE.npz",
                    help="write the TRAIN-derived list into these caches' _meta.json (discard_channels + discard_rule); "
                         "refuses a meta that already carries a different list; needs --flux-ref")  # fmt: skip
    ap.add_argument("--flux-ref", default=None, metavar="FLUX_CACHE.npz",
                    help=f"FLUX multi-block train cache: the rule must give FLUX's published list on it at k={FLUX_K} "
                         "before anything is stamped (required with --stamp)")  # fmt: skip
    args = ap.parse_args(argv)
    if args.stamp is not None and not args.flux_ref:
        raise SystemExit(
            "--stamp needs --flux-ref: a list is stamped only by a rule that reproduces FLUX's own list"
        )
    if args.stamp is not None and args.expect is not None:
        raise SystemExit("--stamp and --expect are separate modes (with --expect nothing would be stamped)")

    flux_check = flux_reference_check(args.flux_ref, args.factor, args.n_images) if args.flux_ref else None
    found, meta = rule_on_cache(args.features, args.k, args.factor, args.n_images)
    print(json.dumps({"features": args.features, "k": args.k, "weights": meta.get("weights"), "factor": args.factor,
                      "n_images": args.n_images, "discard_channels": found}))  # fmt: skip
    if args.expect is not None:
        ok = found == sorted(args.expect)
        print("DISCARD_OK" if ok else f"DISCARD_FAIL expected {sorted(args.expect)} got {found}")
        return 0 if ok else 1
    if args.stamp:
        from eval import wb

        rule = {"factor": args.factor, "n_images": args.n_images, "k": args.k,
                "source": os.path.relpath(args.features, REPO_ROOT), "source_sha1": wb.file_identity(args.features)["sha1"],
                "flux_check": flux_check}  # fmt: skip
        for cache in args.stamp:
            stamp_meta(cache, found, rule)
    return 0


def stamp_meta(cache: str, channels: list[int], rule: dict) -> None:
    """Add discard_channels / discard_rule to <cache>_meta.json. Idempotent for an identical list;
    a different existing list is refused (a stamp is provenance, never silently rewritten)."""
    meta_path = cache[: -len(".npz")] + "_meta.json" if cache.endswith(".npz") else cache
    with open(meta_path) as f:
        meta = json.load(f)
    if "discard_channels" in meta and list(meta["discard_channels"]) != list(channels):
        raise SystemExit(
            f"{meta_path}: already stamped discard_channels={meta['discard_channels']} != {channels}; refusing"
        )
    meta["discard_channels"] = list(channels)
    meta["discard_rule"] = rule
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=4)
    os.replace(tmp, meta_path)
    print(f"stamped {meta_path}: discard_channels={channels}")


if __name__ == "__main__":
    raise SystemExit(main())
