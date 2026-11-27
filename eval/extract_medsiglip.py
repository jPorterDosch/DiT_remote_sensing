"""MedSigLIP-448 probe-baseline features for the chest X-ray probe trees (design section 7.2).

    python -m eval.extract_medsiglip --dataset chestxray14 --view 224up
    python -m eval.extract_medsiglip --dataset chexpert --view 448 --max-images 64   # smoke

Two input conditions, both reported:
  224up  the stored PNG resized to 224 px (bicubic, the probe view every FLUX / DINO row reads),
         then upsampled to 448 inside the encoder -- exactly what the student was aligned to;
  448    the stored 448 px PNG as is -- the teacher's native strength. Design 7.2: only images whose
         SOURCE has a short side of at least 448 px qualify (read from the export's
         export_table.csv, src_w / src_h); an image that does not would have to be excluded from
         this row, which is not implemented, so the run refuses and prints the count per split
         instead (the design expects none).

Writes <out-dir>/medsiglip448-<revision[:12]>_<view>_<dataset>_<split>.npz (the snapshot revision is
in the name, so caches of two snapshots cannot be mixed) in the DINO cache layout, so the probe
reads it as kind "dino": `cls` = the attention-pooled head embedding, `mp` = the mean of the
last-layer patch tokens. eval.probe.official_dino picks MEDSIGLIP_OFFICIAL_VIEWS (cls | mp | clsmp)
for these caches by their `preset`; there is no cls4 (SigLIP has no class token per block).

GATE (executed, prints PASS/FAIL): batch 0 of every split is pushed through the shipped
SiglipImageProcessor as well, with its resample set to bilinear (the card's resize; the shipped
config says bicubic, see models/medsiglip.py); the model inputs must agree (448: to float precision;
224up: within 2/255, the processor quantizes its resized image to uint8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from eval import features as F  # noqa: E402

VIEWS = ("224up", "448")
GATE_TOL = {"448": 1e-5, "224up": 2.0 / 255 * 2}  # in normalized units ([-1, 1] scale)


def load_image(path: str, view: str) -> torch.Tensor | None:
    """(3, H, W) in [-1, 1]; None for a 448-view image stored below 448 px."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w != h:
        raise SystemExit(f"{path}: probe images are square (padded by the export), got {w}x{h}")
    if view == "448":
        if w != 448:
            return None
    else:
        im = im.resize((224, 224), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1


def native_448_shortfall(dataset: str, jobs: dict) -> dict:
    """Per split, the number of images that do NOT qualify for the native 448 row: source short side
    below 448 px (export_table.csv src_w / src_h) or stored side other than 448. Refuses when the
    export table is missing or lacks an image."""
    import csv

    table = os.path.join(F.official_spec(dataset)["root"], "export_table.csv")
    if not os.path.exists(table):
        raise SystemExit(f"{table} missing: the 448 row needs the export's per-image source sizes")
    rows = {}
    with open(table, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["split"], r["name"])] = (int(r["src_w"]), int(r["src_h"]), int(r["side"]))
    out = {}
    for split, (paths, _) in jobs.items():
        bad = 0
        for p in paths:
            key = (split, os.path.splitext(os.path.basename(p))[0])
            if key not in rows:
                raise SystemExit(f"{p}: not in {table}")
            w, h, side = rows[key]
            bad += min(w, h) < 448 or side != 448
        out[split] = bad
    return out


def gate(enc, x: torch.Tensor, view: str, processor) -> float:
    """Max |our model input - the shipped processor's| on one batch; raises above GATE_TOL[view]."""
    ours = enc.prepare(x).float().cpu()
    pil = [
        Image.fromarray(((xi.permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8))
        for xi in x
    ]
    ref = processor(images=pil, return_tensors="pt")["pixel_values"].float()
    d = float((ours - ref).abs().max())
    ok = d <= GATE_TOL[view]
    print(
        f"GATE processor parity ({view}): max |diff| {d:.2e} (tol {GATE_TOL[view]:.1e}) {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not ok:
        raise SystemExit("model input differs from the shipped SiglipImageProcessor -- refusing")
    return d


def extract(enc, paths: list[str], view: str, batch: int, processor=None):
    cls_all, mp_all = [], []
    for i in range(0, len(paths), batch):
        xs = [load_image(p, view) for p in paths[i : i + batch]]
        x = torch.stack(xs)
        if i == 0 and processor is not None:
            gate(enc, x, view, processor)
        tok, pooled = enc.encode(x)
        cls_all.append(pooled.float().cpu().numpy())
        mp_all.append(tok.float().mean(dim=1).cpu().numpy())
        if i % (batch * 50) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.concatenate(cls_all), np.concatenate(mp_all)


def main(argv=None, encoder=None, processor=None) -> list[str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(F.OFFICIAL_CXR))
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--revision", default=None, help="pin a Hugging Face snapshot revision")
    p.add_argument("--max-images", type=int, default=None, help="SMOKE: strided cap per split")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out-dir", default="results/eval_feats")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args(argv)

    jobs = {s: F.list_official_split(args.dataset, s) for s in F.official_spec(args.dataset)["sizes"]}
    if args.max_images:
        jobs = {
            k: ([ps[i] for i in keep], ys[keep])
            for k, (ps, ys) in jobs.items()
            for keep in [np.linspace(0, len(ps) - 1, args.max_images).astype(int)]
        }
    if args.view == "448":  # count before any GPU work
        small = native_448_shortfall(args.dataset, jobs)
        if any(small.values()):
            raise SystemExit(
                f"{args.dataset}: images whose source short side is below 448 px (or not stored at 448) "
                f"per split {small}; the 448 row would have to exclude them, which is not implemented -- refusing"
            )

    if encoder is None:
        from models.medsiglip import MedSigLIP, snapshot

        encoder = MedSigLIP(args.device, revision=args.revision)
        if processor is None:
            from transformers import SiglipImageProcessor

            from models.medsiglip import BILINEAR

            # the card's resize, not the shipped bicubic: the gate checks the inputs prepare() builds
            processor = SiglipImageProcessor.from_pretrained(
                snapshot(revision=encoder.info["revision"]), resample=BILINEAR
            )
    preset = f"medsiglip{encoder.side}-{str(encoder.info['revision'])[:12]}_{args.view}"
    tag = "_smoke" if args.max_images else ""
    os.makedirs(args.out_dir, exist_ok=True)
    outs = []
    for split, (paths, y) in jobs.items():
        out = os.path.join(args.out_dir, f"{preset}_{args.dataset}_{split}{tag}.npz")
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")
        print(f"{split}: {len(paths)} images", flush=True)
        cls_f, mp_f = extract(encoder, paths, args.view, args.batch, processor)
        np.savez(
            out,
            cls=cls_f,
            mp=mp_f,
            paths=np.array(paths),
            labels=y,
            preset=np.array(preset),
            view=np.array(args.view),
            info=np.array(json.dumps(encoder.info, sort_keys=True, default=str)),  # parseable provenance
            images=np.array(hashlib.sha1("\n".join(paths).encode()).hexdigest()[:16] + f"/{len(paths)}"),
            device=np.array(str(encoder.device)),
        )
        print(f"cached cls {cls_f.shape} + mp {mp_f.shape} to {out}", flush=True)
        outs.append(out)
    print("MEDSIGLIP_EXTRACT_OK", args.dataset, args.view)
    return outs


if __name__ == "__main__":
    main()
