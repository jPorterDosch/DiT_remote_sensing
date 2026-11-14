"""COCO val2017 prompts + paired reference images for the text-conditioned generation check
(reviewer objection 2026-11-03: the satellite FID numbers say nothing about whether the adapted
model still generates general text-conditioned images).

Takes the first --n image ids of val2017 (sorted), one caption each (the first annotation of
that image in captions_val2017.json), and writes the real image center-cropped to a square and
resized to --size (bicubic) as <out>/{00000..}.png, the prompts as <out>/prompts.json and a
manifest with the uint8 identity (eval.gen_metrics.image_identity). The generated set is scored
against THESE images (same n, same captions: the usual COCO FID-k protocol), so the reference
is paired with the prompts by construction. Prints COCO_REF_OK.
    python experiments/isaac/git10m/coco_ref.py --n 1000 --size 512 --out $STORE/git10m/fid_ref_coco_val2017_n1000_512
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from eval.gen_metrics import image_identity  # noqa: E402

COCO = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/coco2017"


def square_resize(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=COCO)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; never overwritten")
    ann = json.load(open(os.path.join(a.coco, "annotations", "captions_val2017.json")))
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    caption = {}
    for c in sorted(ann["annotations"], key=lambda c: c["id"]):  # first annotation id per image
        caption.setdefault(c["image_id"], c["caption"].strip())
    ids = sorted(files)[: a.n]
    if len(ids) != a.n:
        raise SystemExit(f"only {len(ids)} val2017 images, asked for {a.n}")
    os.makedirs(a.out, exist_ok=True)
    arrs, prompts = [], []
    for i, iid in enumerate(ids):
        im = Image.open(os.path.join(a.coco, "val2017", files[iid])).convert("RGB")
        sq = square_resize(im, a.size)
        sq.save(os.path.join(a.out, f"{i:05d}.png"))
        arrs.append(np.asarray(sq))
        prompts.append({"idx": i, "image_id": int(iid), "file_name": files[iid], "caption": caption[iid]})
    arr = np.stack(arrs)
    json.dump(prompts, open(os.path.join(a.out, "prompts.json"), "w"), indent=1)
    man = {"kind": "COCO val2017 paired reference: first n image ids (sorted), first caption id per image, center square crop + bicubic resize",
           "n": len(ids), "size": a.size, "identity": image_identity(arr), "coco": a.coco}  # fmt: skip
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"COCO_REF_OK n={len(ids)} size={a.size} -> {a.out} ({man['identity']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
