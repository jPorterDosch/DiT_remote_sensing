"""Fetch the per-image fMoW-rgb metadata JSON of the fMoW-val FID reference rows (OUTLINE §4.8).

DiffusionSat's native fMoW conditioning needs gsd, cloud_cover, country_code and timestamp of each
image. The preprocessed fMoW copy on scratch has pixels and labels only; the JSON sidecars are in
the public SpaceNet bucket (Hosted-Datasets/fmow/fmow-rgb/<path>.json). This downloads one small
JSON per reference row (network I/O only), keeps the files under <out>/json/, and writes
<out>/cond.csv: i,path,img_filename,category,location_id,image_id,gsd,cloud_cover,country_code,
timestamp. Refuses to write cond.csv unless every row was fetched and its img_filename matches.
Already-downloaded JSON files are reused. Prints FMOW_VAL_META_OK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsat_cond as DC  # noqa: E402


def fetch(path: str, dst: str, tries: int = 4) -> dict:
    if os.path.exists(dst):
        with open(dst) as f:
            return json.load(f)
    last = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(DC.fmow_json_url(path), timeout=30) as r:
                raw = r.read()
            d = json.loads(raw)
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(raw)
            os.replace(tmp, dst)
            return d
        except Exception as e:  # noqa: BLE001  (retried, then re-raised)
            last = e
            time.sleep(1.5 * (t + 1))
    raise RuntimeError(f"{path}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows",
        default="/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/git10m/fid_ref_fmowval_n5000/rows.csv",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    csv_out = os.path.join(a.out, "cond.csv")
    if os.path.exists(csv_out):
        raise SystemExit(f"{csv_out} exists; refusing to overwrite")
    with open(a.rows) as f:
        rows = list(csv.DictReader(f))
    if [int(r["i"]) for r in rows] != list(range(len(rows))):
        raise SystemExit(f"{a.rows}: column i is not 0..n-1 in order")
    os.makedirs(os.path.join(a.out, "json"), exist_ok=True)

    def one(r):
        cat, loc, img, fname = DC.parse_fmow_path(r["path"])
        d = fetch(r["path"], os.path.join(a.out, "json", fname.replace(".jpg", ".json")))
        if d.get("img_filename") != fname:
            raise RuntimeError(f"{r['path']}: JSON img_filename {d.get('img_filename')!r}")
        return {
            "i": int(r["i"]),
            "path": r["path"],
            "img_filename": fname,
            "category": cat,
            "location_id": loc,
            "image_id": img,
            "gsd": repr(float(d["gsd"])),
            "cloud_cover": repr(float(d["cloud_cover"])),
            "country_code": d["country_code"],
            "timestamp": d["timestamp"],
        }

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        recs = list(ex.map(one, rows))
    if len(recs) != len(rows):
        raise SystemExit("row count changed")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    gs = sorted(float(r["gsd"]) for r in recs)
    cc = [float(r["cloud_cover"]) for r in recs]
    print(
        f"gsd min / median / max: {gs[0]:.3f} / {gs[len(gs) // 2]:.3f} / {gs[-1]:.3f}; cloud_cover max {max(cc):.0f}, "
        f"nonzero {sum(c > 0 for c in cc)}; countries {len({r['country_code'] for r in recs})}"
    )
    print(f"FMOW_VAL_META_OK n={len(recs)} ({time.time() - t0:.0f}s) -> {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
