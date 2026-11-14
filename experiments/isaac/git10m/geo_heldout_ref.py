"""Geographically held-out FID/KID reference (mentor P0, user decision 2026-10-23).

The post-hoc gen-eval reference (every 100th row of index_retained_cities_unique) overlaps the
training rows. This builds a reference from the Git-10M rows the geo-dedup EXCLUDED (excluded.csv:
tiles within 1 km of a GEO-Bench evaluation sample or inside a So2Sat city ROI): no training index
contains them, so the reference is disjoint from training by construction, not only by row.

  * Twin guard: an excluded row whose img_name also occurs in retained.csv or unassessable.csv (a
    repeated-caption twin that could sit in a training index) is dropped and counted.
  * Stratified by GEO-Bench dataset: 71 % of the excluded rows are So2Sat city tiles, so a plain
    strided draw would be a city reference. Each of the six datasets gets an equal quota of rows
    that matched it (a row matching several counts for the first pool it is drawn into); a pool
    smaller than its quota is taken whole and the remainder is spread over the others.
  * Decoded exactly like the training-time reference (datasets.git10m.Git10M.__getitem__ ->
    train_diffusion._reference_images): RGB, bicubic resize to --img-size, [-1, 1], then
    ((x + 1) * 127.5).round() as uint8. Unflipped.

Writes <out>/{00000..}.png, <out>/manifest.json (rows, per-dataset composition, guard counts,
image identity) and <out>/rows.csv. Prints GEO_HELDOUT_REF_OK.

    python experiments/isaac/git10m/geo_heldout_ref.py --n 5000 --out $STORE/git10m/fid_ref_geoheldout_n5000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from PIL import Image
from torchvision.transforms import PILToTensor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from datasets.git10m import BATCH_ROWS, shard_name  # noqa: E402
from eval.gen_metrics import image_identity  # noqa: E402

DATASETS = ["m_so2sat", "m_eurosat", "m_bigearthnet", "m_brick_kiln", "m_forestnet", "m_pv4ger"]


def quotas(sizes: dict, n: int) -> dict:
    q = {k: 0 for k in sizes}
    left, open_ = n, [k for k in sizes if sizes[k] > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for k in list(open_):
            take = min(share, sizes[k] - q[k], left)
            q[k] += take
            left -= take
            if q[k] >= sizes[k]:
                open_.remove(k)
            if left == 0:
                break
    if left:
        raise SystemExit(f"only {n - left} distinct excluded rows available, need {n}")
    return q


def assert_disjoint(sel: pd.DataFrame, train_index: str) -> dict:
    """Executable guard (rule 6): no chosen (shard, row) may be a row of the training index. Raises on
    the first overlap; returns the record written to manifest.json (index path, its sha256, counts)."""
    import hashlib

    z = np.load(train_index)
    train = set(zip(z["shard"].astype(np.int64).tolist(), z["row"].astype(np.int64).tolist()))
    chosen = list(zip(sel["shard"].astype(np.int64).tolist(), sel["row"].astype(np.int64).tolist()))
    overlap = [c for c in chosen if c in train]
    if overlap:
        raise SystemExit(
            f"GEO_HELDOUT_REF_FAIL: {len(overlap)} reference rows are training rows of {train_index}, e.g. {overlap[:5]}"
        )
    h = hashlib.sha256()
    with open(train_index, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    rec = {
        "train_index": train_index,
        "train_index_sha256": h.hexdigest(),
        "train_rows": len(train),
        "reference_rows": len(chosen),
        "overlap": 0,
    }
    print(
        f"disjoint check PASS: {len(chosen)} reference rows, 0 in the {len(train)}-row training index",
        flush=True,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--manifest-dir",
        default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/geo_dedup/dedup_v3_b1000_cities",
    )
    ap.add_argument("--arrow-dir", default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/git10m/arrow")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--train-index",
        default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/git10m/index_all_cities_unique.npz",
        help="training index npz; every chosen (shard, row) must be absent from it (executable disjointness check)",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="do not build: load <out>/rows.csv and run the disjointness check against --train-index, write <out>/disjoint_check.json",
    )
    a = ap.parse_args()
    if a.verify_only:
        sel = pd.read_csv(os.path.join(a.out, "rows.csv"), usecols=["shard", "row"])
        rec = assert_disjoint(sel, a.train_index)
        with open(os.path.join(a.out, "disjoint_check.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print("GEO_HELDOUT_REF_OK (verify-only)")
        return 0
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty: references are never overwritten")

    ex = pd.read_csv(
        os.path.join(a.manifest_dir, "excluded.csv"), usecols=["shard", "row", "img_name", "datasets"]
    )
    used = set()
    for f in ("retained.csv", "unassessable.csv"):
        used.update(pd.read_csv(os.path.join(a.manifest_dir, f), usecols=["img_name"])["img_name"].tolist())
    twin = ex["img_name"].isin(used)
    n_twin = int(twin.sum())
    ex = ex[~twin].reset_index(drop=True)
    print(f"excluded rows {len(ex) + n_twin}, twin-guard dropped {n_twin}, pool {len(ex)}", flush=True)

    rng = np.random.default_rng(a.seed)
    pools = {
        d: rng.permutation(np.flatnonzero(ex["datasets"].str.contains(d, regex=False).to_numpy()))
        for d in DATASETS
    }
    q = quotas({d: len(p) for d, p in pools.items()}, a.n)
    chosen, comp = [], {}
    taken = set()
    for d in DATASETS:  # sequential draw; a row already taken by an earlier pool is skipped
        got = 0
        for i in pools[d]:
            if got >= q[d]:
                break
            if i in taken:
                continue
            taken.add(int(i))
            chosen.append(int(i))
            got += 1
        comp[d] = got
    if len(chosen) < a.n:  # overlap between pools left a shortfall: fill from the largest pool
        for i in pools["m_so2sat"]:
            if len(chosen) >= a.n:
                break
            if int(i) not in taken:
                taken.add(int(i))
                chosen.append(int(i))
                comp["m_so2sat"] += 1
    sel = ex.iloc[sorted(chosen)].reset_index(drop=True)  # shard / row order: deterministic, reader-friendly
    print("composition", comp, flush=True)
    disjoint = assert_disjoint(sel, a.train_index)

    os.makedirs(a.out, exist_ok=True)
    readers, imgs = {}, []
    for k, (s, r) in enumerate(zip(sel["shard"].to_numpy(), sel["row"].to_numpy())):
        s, r = int(s), int(r)
        rd = readers.get(s)
        if rd is None:
            rd = readers[s] = ipc.open_file(pa.memory_map(os.path.join(a.arrow_dir, shard_name(s))))
        b = rd.get_batch(r // BATCH_ROWS)
        j = r % BATCH_ROWS
        name, raw = b.column("img_name")[j].as_py(), b.column("image_bytes")[j].as_py()
        if name != sel["img_name"][k]:
            raise SystemExit(f"shard {s} row {r}: arrow img_name {name!r} != manifest {sel['img_name'][k]!r}")
        img = (
            Image.open(io.BytesIO(raw))
            .convert("RGB")
            .resize((a.img_size, a.img_size), Image.Resampling.BICUBIC)
        )
        x = (PILToTensor()(img) / 255.0 - 0.5) * 2  # == Git10M.__getitem__
        u8 = (
            ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        )  # == _reference_images
        imgs.append(u8)
        Image.fromarray(u8).save(os.path.join(a.out, f"{k:05d}.png"))
        if k % 1000 == 0:
            print(f"  {k}/{len(sel)}", flush=True)
    arr = np.stack(imgs)
    sel.to_csv(os.path.join(a.out, "rows.csv"), index=False)
    man = {
        "kind": "geo-heldout reference: Git-10M rows excluded by the geo-dedup (never in a training index)",
        "manifest_dir": a.manifest_dir,
        "n": int(len(sel)),
        "seed": a.seed,
        "img_size": a.img_size,
        "twin_guard_dropped": n_twin,
        "pool_rows": int(len(ex)),
        "composition": comp,
        "pool_sizes": {d: int(len(p)) for d, p in pools.items()},
        "identity": image_identity(arr),
        "disjoint_check": disjoint,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man, indent=1))
    print("GEO_HELDOUT_REF_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
