"""Direct image check of the Git-10M SSL4EO-S12 name -> location rule (eval/geo_dedup.ssl4eo_boxes).

The rule maps `SSL4EO-S12-rgb-<4i+1>.jpg` to the i-th centre of the authors' centre file in
ascending id order (void ids skipped). geo_dedup.ssl4eo_verify tests it statistically (captions
vs land-cover labels). This script tests it on pixels: SSL4EO-S12 v1.1 (HF
embed2scale/SSL4EO-S12-v1.1) keeps the v1.0 sample ids and ships an S2RGB modality (4 seasons,
264 x 264, uint8 B04/B03/B02) in tar shards of ~512 zarr zips, so for a Git-10M row the patch
at the MAPPED id can be compared with the Git-10M image itself.

Score per row = max over the 4 seasons and two geometric fits (bicubic resize 264 -> 256, centre
crop 256) of the Pearson correlation between greyscale images. Null = the same Git-10M image
scored against the patches of a DIFFERENT sampled location in the same shard. v1.1 was
re-downloaded, so acquisition dates may differ from the image Git-10M carries; a correct
mapping still gives a high correlation at the same location (fields, roads, water do not move),
a wrong mapping gives the null distribution.

Executed gate (rule 13), thresholds fixed before the first run:
  median(mapped) >= 0.6, median(null) <= 0.3, mapped > null for >= 95 % of rows, and that
  last criterion separately for rows below / above the first void id (the two halves
  distinguish "skip the voids" from "do not").

    python -m eval.ssl4eo_pixel_check --join J --arrow-dir A --v11-dir V --ssl4eo-void void_ids.csv \\
        --n-shards 20 --per-shard 64 --out ssl4eo_pixel_check.json [--hf-download]
`--join` is the parquet written by experiments/isaac/geo_dedup/ssl4eo_pixel_check.sbatch's
join step: (shard, row, img_name, source_id, sid, tar, zarr) for every Git-10M SSL4EO row whose
mapped id exists in v1.1 (243,968 of 251,079 on 2026-10-13).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

V11_REPO = "embed2scale/SSL4EO-S12-v1.1"
GIT10M_SIDE = 256
V11_SIDE = 264
GATE = {"median_mapped_min": 0.6, "median_null_max": 0.3, "frac_mapped_gt_null_min": 0.95}


# ----------------------------------------------------------------------------- readers
def read_zarr_rgb(blob: bytes) -> tuple[np.ndarray, str]:
    """(4, 264, 264, 3) uint8 RGB of one v1.1 S2RGB zarr zip + its sample id. Zarr v2, one chunk,
    blosc-compressed (numcodecs); band order checked against the stored band names."""
    import numcodecs

    z = zipfile.ZipFile(io.BytesIO(blob))
    za = json.loads(z.read("bands/.zarray"))
    if za["chunks"] != za["shape"] or za["shape"][1:] != [3, V11_SIDE, V11_SIDE]:
        raise ValueError(f"unexpected bands layout {za['shape']} chunks {za['chunks']}")
    raw = numcodecs.get_codec(za["compressor"]).decode(z.read("bands/0.0.0.0"))
    arr = np.frombuffer(raw, dtype=np.dtype(za["dtype"])).reshape(za["shape"])
    bz = json.loads(z.read("band/.zarray"))
    bands = np.frombuffer(numcodecs.get_codec(bz["compressor"]).decode(z.read("band/0")), dtype=bz["dtype"])
    if list(bands[:3]) != ["B04", "B03", "B02"]:
        raise ValueError(f"band order {bands[:3]} is not B04,B03,B02")
    sid = str(np.frombuffer(z.read("sample/0"), dtype=json.loads(z.read("sample/.zarray"))["dtype"])[0])
    return np.ascontiguousarray(arr.transpose(0, 2, 3, 1)), sid


class ArrowRows:
    """Git-10M images by (shard, row) from the converted arrow shards (src/datasets/git10m)."""

    def __init__(self, arrow_dir: str):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
        from datasets.git10m import BATCH_ROWS, shard_name

        self.dir, self.bs, self.name = arrow_dir, BATCH_ROWS, shard_name
        self._rd = {}

    def image(self, shard: int, row: int) -> tuple[str, np.ndarray]:
        rd = self._rd.get(shard)
        if rd is None:
            rd = self._rd[shard] = ipc.open_file(pa.memory_map(os.path.join(self.dir, self.name(shard))))
        b = rd.get_batch(row // self.bs)
        j = row % self.bs
        name = b.column("img_name")[j].as_py()
        img = Image.open(io.BytesIO(b.column("image_bytes")[j].as_py())).convert("RGB")
        return name, np.asarray(img)


# ----------------------------------------------------------------------------- scoring
def _grey(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64).mean(-1).ravel()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def fits(patch: np.ndarray, side: int = GIT10M_SIDE) -> list[np.ndarray]:
    """The two geometric fits of a 264 px patch to a 256 px image."""
    im = Image.fromarray(patch)
    off = (patch.shape[0] - side) // 2
    return [
        np.asarray(im.resize((side, side), Image.Resampling.BICUBIC)),
        patch[off : off + side, off : off + side],
    ]


def score(img: np.ndarray, patches: np.ndarray) -> tuple[float, int, int]:
    """max correlation over seasons x fits; returns (score, season, fit)."""
    if img.shape[:2] != (GIT10M_SIDE, GIT10M_SIDE):
        img = np.asarray(Image.fromarray(img).resize((GIT10M_SIDE, GIT10M_SIDE), Image.Resampling.BICUBIC))
    g = _grey(img)
    best = (-2.0, -1, -1)
    for s in range(patches.shape[0]):
        for f, p in enumerate(fits(patches[s])):
            c = _corr(g, _grey(p))
            if c > best[0]:
                best = (c, s, f)
    return best


def gate(rows: pd.DataFrame, first_void: int) -> dict:
    """PASS/FAIL on the fixed thresholds, overall and per half of the id range."""
    out = {"thresholds": GATE, "n": int(len(rows))}
    gt = rows["mapped"] > rows["null"]
    out["median_mapped"] = float(rows["mapped"].median())
    out["median_null"] = float(rows["null"].median())
    out["frac_mapped_gt_null"] = float(gt.mean())
    out["frac_mapped_ge_0.9"] = float((rows["mapped"] >= 0.9).mean())
    halves = {}
    for name, m in (
        ("below_first_void", rows["id"] < first_void),
        ("above_first_void", rows["id"] >= first_void),
    ):
        halves[name] = {
            "n": int(m.sum()),
            "frac_mapped_gt_null": float(gt[m].mean()) if m.any() else None,
            "median_mapped": float(rows.loc[m, "mapped"].median()) if m.any() else None,
        }
    out["halves"] = halves
    ok = (
        out["median_mapped"] >= GATE["median_mapped_min"]
        and out["median_null"] <= GATE["median_null_max"]
        and out["frac_mapped_gt_null"] >= GATE["frac_mapped_gt_null_min"]
        and all(
            h["n"] > 0 and h["frac_mapped_gt_null"] >= GATE["frac_mapped_gt_null_min"]
            for h in halves.values()
        )
    )
    out["pass"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- join
def build_join(meta_dir: str, source_boxes: str, v11_metadata: str, out: str) -> dict:
    """Every Git-10M SSL4EO row (shard, row, img_name) with its mapped id (source_boxes.parquet,
    eval/geo_dedup source-boxes) and the v1.1 shard holding that id (train_metadata.parquet)."""
    import glob

    sb = pd.read_parquet(source_boxes, columns=["img_name", "source", "source_id"])
    sb = sb[sb["source"] == "ssl4eo"]
    metas = sorted(glob.glob(os.path.join(meta_dir, "meta-*.parquet")))
    if len(metas) != 787:
        raise SystemExit(f"{meta_dir}: {len(metas)} meta parquets, expected 787")
    parts = []
    for f in metas:
        m = pd.read_parquet(f, columns=["shard", "row", "img_name"])
        m = m[m["img_name"].str.startswith("SSL4EO-S12-rgb-")]
        if len(m):
            parts.append(m)
    meta = pd.concat(parts, ignore_index=True)
    j = meta.merge(sb[["img_name", "source_id"]], on="img_name", how="left")
    if j["source_id"].isna().any():
        raise SystemExit(
            f"{int(j['source_id'].isna().sum())} SSL4EO rows have no mapped id in {source_boxes}"
        )
    j["sid"] = j["source_id"].astype(int).map(lambda i: f"{i:07d}")
    v = pd.read_parquet(v11_metadata, columns=["tar", "zarr", "sample_id", "center_lon", "center_lat"])
    if v["sample_id"].duplicated().any():
        raise SystemExit(f"{v11_metadata}: duplicate sample_id")
    jj = j.merge(v, left_on="sid", right_on="sample_id", how="inner")
    cols = ["shard", "row", "img_name", "source_id", "sid", "tar", "zarr", "center_lon", "center_lat"]
    jj[cols].to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    rep = {
        "git10m_ssl4eo_rows": int(len(j)),
        "in_v11": int(len(jj)),
        "tars": int(jj["tar"].nunique()),
        "out": out,
    }
    print(json.dumps(rep))
    return rep


# ----------------------------------------------------------------------------- run
def run(join, arrow: ArrowRows, v11_dir: str, n_shards: int, per_shard: int, seed: int, hf: bool, tars=None):
    rng = np.random.default_rng(seed)
    pool = sorted(join["tar"].unique())
    if tars:
        bad = sorted(set(tars) - set(pool))
        if bad:
            raise SystemExit(f"--tars not in the join: {bad}")
        chosen = sorted(tars)
    else:
        chosen = sorted(rng.choice(pool, size=min(n_shards, len(pool)), replace=False).tolist())
    rows = []
    for tar in chosen:
        path = os.path.join(v11_dir, "train", "S2RGB", tar)
        if not os.path.exists(path):
            if not hf:
                raise SystemExit(f"{path} missing (pass --hf-download)")
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(V11_REPO, f"train/S2RGB/{tar}", repo_type="dataset", local_dir=v11_dir)
            if os.path.realpath(got) != os.path.realpath(path):
                raise SystemExit(f"download landed at {got}, expected {path}")
        sub = join[join["tar"] == tar]
        take = sub.sample(n=min(per_shard, len(sub)), random_state=int(rng.integers(0, 2**31)))
        want = {r.zarr: r for r in take.itertuples()}
        patches = {}
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                if m.name in want:
                    arr, sid = read_zarr_rgb(tf.extractfile(m).read())
                    if sid != want[m.name].sid:
                        raise SystemExit(f"{tar}/{m.name}: zarr sample {sid} != metadata {want[m.name].sid}")
                    patches[m.name] = arr
        missing = set(want) - set(patches)
        if missing:
            raise SystemExit(f"{tar}: {len(missing)} zarr members missing, e.g. {sorted(missing)[:3]}")
        names = list(patches)
        if len(names) < 2:
            raise SystemExit(f"{tar}: need >= 2 sampled rows for a null")
        for i, zn in enumerate(names):
            r = want[zn]
            name, img = arrow.image(int(r.shard), int(r.row))
            if name != r.img_name:
                raise SystemExit(f"shard {r.shard} row {r.row}: {name} != {r.img_name}")
            other = names[(i + 1 + int(rng.integers(0, len(names) - 1))) % len(names)]
            if other == zn:
                raise SystemExit("null drew the mapped location")
            sc, season, fit = score(img, patches[zn])
            nl, _, _ = score(img, patches[other])
            rows.append(
                {
                    "img_name": name,
                    "id": int(r.source_id),
                    "tar": tar,
                    "mapped": sc,
                    "season": season,
                    "fit": fit,
                    "null": nl,
                    "null_id": int(want[other].source_id),
                }
            )
        print(
            f"{tar}: {len(names)} rows, median mapped {np.median([x['mapped'] for x in rows[-len(names) :]]):.3f}",
            flush=True,
        )
    return pd.DataFrame(rows), chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--build-join",
        nargs=3,
        metavar=("META_DIR", "SOURCE_BOXES", "V11_METADATA"),
        default=None,
        help="write --join from the dedup metadata, source_boxes.parquet and v1.1 train_metadata.parquet, then exit",
    )
    ap.add_argument("--join", required=True)
    ap.add_argument("--arrow-dir", default=None)
    ap.add_argument("--v11-dir", default=None)
    ap.add_argument(
        "--ssl4eo-void", default=None, help="authors' void_ids.csv (first void id splits the halves)"
    )
    ap.add_argument("--n-shards", type=int, default=20)
    ap.add_argument(
        "--tars", nargs="*", default=None, help="explicit shard names instead of a seeded draw (smoke)"
    )
    ap.add_argument("--per-shard", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-download", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.build_join:
        if os.path.exists(a.join):
            raise SystemExit(f"{a.join} exists")
        build_join(*a.build_join, a.join)
        return 0
    missing = [f for f in ("arrow_dir", "v11_dir", "ssl4eo_void", "out") if not getattr(a, f)]
    if missing:
        ap.error("required for the check: " + " ".join("--" + f.replace("_", "-") for f in missing))
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    join = pd.read_parquet(a.join)
    need = {"shard", "row", "img_name", "source_id", "sid", "tar", "zarr"}
    if not need <= set(join.columns):
        raise SystemExit(f"{a.join}: columns {sorted(join.columns)}, need {sorted(need)}")
    voids = pd.read_csv(a.ssl4eo_void, header=None).iloc[:, 0].astype(int)
    first_void = int(voids.min())
    rows, chosen = run(
        join, ArrowRows(a.arrow_dir), a.v11_dir, a.n_shards, a.per_shard, a.seed, a.hf_download, a.tars
    )
    g = gate(rows, first_void)
    rec = {
        "gate": g,
        "first_void_id": first_void,
        "shards": chosen,
        "per_shard": a.per_shard,
        "seed": a.seed,
        "join": os.path.abspath(a.join),
        "join_rows": int(len(join)),
        "season_hist": rows["season"].value_counts().sort_index().to_dict(),
        "fit_hist": rows["fit"].value_counts().sort_index().to_dict(),
        "rows": rows.to_dict(orient="records"),
    }
    with open(a.out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({k: v for k, v in g.items()}, indent=1))
    print(
        f"SSL4EO_PIXEL_CHECK: {'PASS' if g['pass'] else 'FAIL'} ({len(rows)} rows, {len(chosen)} shards) -> {a.out}"
    )
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
