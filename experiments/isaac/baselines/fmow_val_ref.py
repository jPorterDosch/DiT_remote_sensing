"""fMoW-RGB validation reference for the two-way FID table (OUTLINE §4.8).

DiffusionSat was fine-tuned on fMoW (+ Satlas, SpaceNet); our arms on Git-10M. Scoring each
model on the other's training distribution needs a second reference set decoded EXACTLY like the
geo-held-out Git-10M reference (geo_heldout_ref.py): RGB, PIL bicubic resize to --img-size x
--img-size (no crop), uint8 PNG. Rows: every k-th line of val_62classes.csv (sorted as shipped),
k = floor(n_val / n), so all 62 classes appear.

Writes <out>/{00000..}.png, <out>/manifest.json (rows, class histogram, image identity from
eval.gen_metrics.image_identity) and <out>/rows.csv. Refuses to overwrite. Prints FMOW_REF_OK.

    python experiments/isaac/baselines/fmow_val_ref.py --n 5000 \
        --out $STORE/DiT_remote_sensing/git10m/fid_ref_fmowval_n5000
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from eval import gen_metrics as GM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmow", default="/lustre/isaac24/scratch/jdosch1/datasets/fmow-rgb-preproc")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    with open(os.path.join(a.fmow, "val_62classes.csv")) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < a.n:
        raise SystemExit(f"only {len(rows)} val rows < n={a.n}")
    k = len(rows) // a.n
    picked = rows[::k][: a.n]
    os.makedirs(a.out)
    imgs = np.empty((a.n, a.img_size, a.img_size, 3), dtype=np.uint8)
    hist: collections.Counter = collections.Counter()
    with open(os.path.join(a.out, "rows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "label", "path"])
        for i, r in enumerate(picked):
            img = Image.open(os.path.join(a.fmow, r["path"])).convert("RGB")
            img = img.resize((a.img_size, a.img_size), Image.Resampling.BICUBIC)
            imgs[i] = np.asarray(img)
            img.save(os.path.join(a.out, f"{i:05d}.png"))
            hist[r["label"]] += 1
            w.writerow([i, r["label"], r["path"]])
    man = {
        "kind": "fMoW-RGB val reference (val_62classes.csv, every k-th row)",
        "fmow": a.fmow,
        "n": a.n,
        "stride": k,
        "n_val_rows": len(rows),
        "img_size": a.img_size,
        "n_classes_present": len(hist),
        "class_hist": dict(sorted(hist.items())),
        "decode": "PIL RGB, bicubic resize to img_size x img_size, no crop",
        "identity": GM.image_identity(imgs),
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"FMOW_REF_OK n={a.n} stride={k} classes={len(hist)}/62 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
