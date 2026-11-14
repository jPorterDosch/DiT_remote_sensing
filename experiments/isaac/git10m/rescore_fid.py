"""Re-score saved gen-eval samples against another reference (no regeneration).

Every gen-eval run dir (models/git10m_gen_eval/n<N>_h<H>_g<G>/<run>/) holds its samples/*.png and
gen_metrics.json. For each dir this
  1. checks the PNGs ARE the scored samples: their uint8 identity (sha1) equals gen_metrics.json
     "fake" (fatal otherwise);
  2. GATE (first dir, or --gate-all): re-scores against the dir's own fid_ref/ and must reproduce the
     recorded FID / KID (|dFID| <= --gate-fid, default 0.1 absolute; |dKID| <= --gate-kid-std x the
     recorded KID subset std, default 0.1; relative errors are printed only): proves this path
     equals the original;
  3. scores the samples against --ref (Inception features of the reference computed once) and writes
     <dir>/gen_metrics_<tag>.json (never overwrites).
Prints a table and RESCORE_FID_OK; writes --summary json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from eval import gen_metrics as GM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen-root",
        default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/models/git10m_gen_eval/n5000_h100_g1.0",
    )
    ap.add_argument("--ref", required=True, help="directory of reference PNGs (+ manifest.json)")
    ap.add_argument("--tag", required=True, help="output file suffix, e.g. geoheldout")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--gate-all", action="store_true")
    ap.add_argument("--subsets", type=int, default=100)
    ap.add_argument("--gate-fid", type=float, default=0.1)
    ap.add_argument("--gate-kid-std", type=float, default=0.1)
    a = ap.parse_args()
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = GM.InceptionFeatures(dev)
    ref = GM._load_pngs(a.ref, None)
    f_ref = ext(ref)
    ref_id = GM.image_identity(ref)
    man = os.path.join(a.ref, "manifest.json")
    if os.path.exists(man) and json.load(open(man))["identity"]["sha1"] != ref_id["sha1"]:
        raise SystemExit(f"{a.ref}: PNG identity differs from its manifest")
    rows, gated, gates = [], False, []
    for d in sorted(glob.glob(os.path.join(a.gen_root, "*", "gen_metrics.json"))):
        run = os.path.dirname(d)
        out = os.path.join(run, f"gen_metrics_{a.tag}.json")
        g = json.load(open(d))
        fake = GM._load_pngs(os.path.join(run, "samples"), None)
        if GM.image_identity(fake)["sha1"] != g["fake"]["sha1"]:
            raise SystemExit(f"{run}: samples/*.png are not the scored samples (sha1 mismatch)")
        f_fake = ext(fake)
        if a.gate_all or not gated:
            own = GM._load_pngs(os.path.join(run, "fid_ref"), None)
            if GM.image_identity(own)["sha1"] != g["real"]["sha1"]:
                raise SystemExit(f"{run}: fid_ref/*.png are not the recorded reference")
            f_own = ext(own)
            fid0 = GM.fid(f_own, f_fake)
            k0, _ = GM.kid(f_own, f_fake, subsets=a.subsets)
            # Tolerance at the metric's noise floor, not at float precision: the Inception features
            # differ between devices (recorded on an A40; the same PNGs re-scored on CPU moved FID by
            # 3.6e-4 relative, on an H100 by 3.5e-5 for the pretrained model but 0.014 absolute for proj
            # at FID 24.7, 2026-10-24). |dFID| <= --gate-fid (0.1; the training-seed FID spread is
            # 4.6-10.3) and |dKID| <= --gate-kid-std x the recorded KID subset std.
            rf, rk = abs(fid0 - g["fid"]) / g["fid"], abs(k0 - g["kid_mean"]) / max(abs(g["kid_mean"]), 1e-12)
            ok = (
                abs(fid0 - g["fid"]) <= a.gate_fid
                and abs(k0 - g["kid_mean"]) <= a.gate_kid_std * g["kid_std"]
            )
            gates.append(
                {
                    "run": os.path.basename(run),
                    "fid_recorded": g["fid"],
                    "fid_rescored": fid0,
                    "rel_fid": rf,
                    "kid_recorded": g["kid_mean"],
                    "kid_rescored": k0,
                    "rel_kid": rk,
                }
            )
            print(
                f"GATE {os.path.basename(run)} [{dev}]: recorded FID {g['fid']:.4f} KID {g['kid_mean']:.6f} -> re-scored {fid0:.4f} {k0:.6f} (rel {rf:.1e} / {rk:.1e}) {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
            if not ok:
                raise SystemExit("RESCORE GATE FAIL")
            gated = True
        if len(f_ref) != len(f_fake):
            raise SystemExit(f"n differs: reference {len(f_ref)}, samples {len(f_fake)}")
        fid = GM.fid(f_ref, f_fake)
        km, ks = GM.kid(f_ref, f_fake, subsets=a.subsets)
        res = {
            "run": os.path.basename(run),
            "checkpoint": g.get("checkpoint"),
            "n": len(f_fake),
            "fid": fid,
            "kid_mean": km,
            "kid_std": ks,
            "kid_subsets": a.subsets,
            "extractor": ext.name,
            "reference_dir": os.path.abspath(a.ref),
            "real": ref_id,
            "fake": g["fake"],
            "device": dev,
            "fid_train_overlap_ref": g["fid"],
            "kid_train_overlap_ref": g["kid_mean"],
        }
        rescored = None
        if os.path.exists(out):
            prev = json.load(open(out))
            # Same reference, same samples: a re-run may differ only by device / kernel numerics.
            if (
                prev["real"]["sha1"] != ref_id["sha1"]
                or prev["fake"]["sha1"] != res["fake"]["sha1"]
                or abs(prev["fid"] - fid) > a.gate_fid
            ):
                raise SystemExit(f"{out} exists with a different result: refusing to overwrite")
            # The per-run record is the canonical number (review 2026-10-27): the summary row carries
            # it, so the two artifacts never disagree; this pass's value is kept alongside as evidence.
            rescored = {"fid": fid, "kid_mean": km, "device": dev}
            res = prev
        else:
            with open(out, "w") as f:
                json.dump(res, f, indent=1)
        ck = (g.get("checkpoint") or {}).get("lora_checkpoint") or "pretrained"
        row = {
            k: res[k]
            for k in ("run", "fid", "kid_mean", "kid_std", "fid_train_overlap_ref", "kid_train_overlap_ref")
        } | {"checkpoint": ck, "device": res.get("device")}
        if rescored is not None:
            row["rescored"] = rescored
        rows.append(row)
        print(
            f"{os.path.basename(run):32s} {ck.split('/models/git10m/')[-1][:60]:60s} FID {res['fid']:7.2f} (overlap ref {g['fid']:7.2f}) KID {res['kid_mean']:.4f}"
            + (f" [this pass {fid:.4f} / {km:.6f} on {dev}, record kept]" if rescored else ""),
            flush=True,
        )
    with open(a.summary, "w") as f:
        json.dump(
            {
                "reference": ref_id,
                "reference_dir": os.path.abspath(a.ref),
                "device": dev,
                "gates": gates,
                "rows": rows,
            },
            f,
            indent=1,
        )
    print("RESCORE_FID_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
