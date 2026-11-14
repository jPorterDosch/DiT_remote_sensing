"""Measure (do not change) how many Git-10M training rows lie near the three GEO-Bench
segmentation tasks, which the geographic exclusion of eval.geo_dedup never covered
(review finding 5, 2026-11-12). Nothing here rebuilds a manifest or a training index.

What each task's SHIPPED hdf5 files allow (decided per file, CLAUDE.md rule 16):
  m_pv4ger_seg            sample names are "lon,lat" pairs -> centre box, half side taken from
                          the artifact (rows x spatial_resolution / 2), exact gap test
  m_chesapeake_landcover  names id_N, no transform / crs in any band -> not locatable
  m_SA_crop_type          names ref_south_africa_crops_competition_v1_train_labels_<id>_<k>,
                          no transform / crs -> not locatable from the artifact

For a task that cannot be located the only geographic statement available is a REGION count: the
number of located training rows inside a declared superset of where its samples can be
(REGIONS). That is a loose upper bound on rows that could be near a sample, not a hit count; it
is reported under `region_upper_bound` and must be quoted with that label.

The training rows tested are the located ones (Google tiles, SSL4EO-S12, SkyScript); each is
labelled retained / excluded from an existing dedup directory, so "retained rows within the
buffer" is the number that would have to leave the training set. The ~2.05M rows without
coordinates are outside this measurement, as they are outside the exclusion itself.

Gates (rule 13): eval.geo_dedup.control on the evaluation boxes used here (on-sample tile hit,
antipode miss, 0.99 x / 1.01 x buffer); the retained + excluded manifests must partition the
located rows exactly.

    python -m eval.seg_overlap --seg-root <.../segmentation_v1.0> --meta <git10m_meta> \
        --source-boxes <source_boxes.parquet> --dedup <dedup_v3_b1000_cities> --points <points_v3_cities.csv> \
        --buffer-m 1000 --out <prefix>
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os

import numpy as np
import pandas as pd

from eval import geo_dedup as G

SEG_TASKS = {  # task -> directory under --seg-root, and the name parser when names carry a centre
    "m_pv4ger_seg": {"dir": "m-pv4ger-seg", "name_parser": "m_pv4ger"},
    "m_chesapeake_landcover": {"dir": "m-chesapeake-landcover", "name_parser": None},
    "m_SA_crop_type": {"dir": "m-SA-crop-type", "name_parser": None},
}
# Declared supersets for the tasks without coordinates: lon_min, lat_min, lon_max, lat_max, each
# rounded OUTWARD. Chesapeake: the source dataset (Robinson et al. 2019, "Large Scale High-
# Resolution Land Cover Mapping with Multi-Resolution Data") covers the Chesapeake Bay watershed
# in six states; the state bounding boxes are a superset of the watershed. South Africa: the
# source competition tiles are in the Western Cape; the whole-country box is used because the
# shipped files give no tighter extent.
REGIONS = {
    "m_chesapeake_landcover": {
        "state:DE": (-75.9, 38.4, -75.0, 39.9),
        "state:MD": (-79.5, 37.8, -75.0, 39.8),
        "state:NY": (-79.8, 40.4, -71.8, 45.1),
        "state:PA": (-80.6, 39.7, -74.6, 42.3),
        "state:VA": (-83.7, 36.5, -75.2, 39.5),
        "state:WV": (-82.7, 37.2, -77.7, 40.7),
    },
    "m_SA_crop_type": {"country:ZA": (16.0, -35.0, 33.0, -22.0)},
}


def sample_geometry(blob: bytes) -> tuple[int, int, float]:
    """rows, cols and spatial resolution (m) of the first band of one shipped sample."""
    import geobench  # noqa: F401  registers the ccb.* aliases the pickles reference
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as h:
        attrs = G._Unpickler(io.BytesIO(ast.literal_eval(h.attrs["pickle"]))).load()
        band = attrs["bands_order"][0]
        rows, cols = h[band].shape[:2]
        return int(rows), int(cols), float(attrs[band]["spatial_resolution"])


def task_boxes(task: str, task_dir: str, read=None) -> tuple[pd.DataFrame, dict]:
    """Evaluation boxes of one segmentation task (every split of default_partition.json) and what
    was found. EVERY sample file is opened: `footprint` if all carry transform + crs, the name
    parser if none does and the names parse, otherwise an empty frame with assessable False."""
    with open(os.path.join(task_dir, "default_partition.json")) as f:
        partition = json.load(f)
    names = [(split, n) for split in partition for n in partition[split]]
    if len({n for _, n in names}) != len(names):
        raise SystemExit(f"{task}: duplicate sample names across splits")
    if read is None:

        def read(name):
            with open(os.path.join(task_dir, f"{name}.hdf5"), "rb") as fh:
                return fh.read()

    parser = SEG_TASKS[task]["name_parser"]
    rows, n_foot, geom = [], 0, set()
    for split, name in names:
        blob = read(name)
        box = G.hdf5_footprint(blob, name)
        if box is not None:
            n_foot += 1
            rows.append((task, split, name, "footprint", *box))
        elif parser is not None:
            r, c, res = sample_geometry(blob)
            geom.add((r, c, res))
            lat, lon = G.parse_eval_name(parser, name)
            rows.append((task, split, name, "centre+nominal", *G.centre_box(lat, lon, max(r, c) * res / 2)))
    rep = {"n_samples": len(names), "splits": {s: len(v) for s, v in partition.items()}, "files_opened": len(names),
           "files_with_transform_and_crs": n_foot}  # fmt: skip
    if 0 < n_foot < len(names):
        raise SystemExit(f"{task}: {n_foot} of {len(names)} samples carry georeferencing -- mixed, refusing")
    if n_foot == len(names):
        rep.update(assessable=True, source="footprint")
    elif parser is not None:
        if len(geom) != 1:
            raise SystemExit(f"{task}: samples differ in size / resolution: {sorted(geom)}")
        r, c, res = next(iter(geom))
        rep.update(assessable=True, source="centre+nominal", name_parser=parser, rows=r, cols=c,
                   spatial_resolution_m=res, half_extent_m=max(r, c) * res / 2)  # fmt: skip
    else:
        rep.update(assessable=False, reason=f"no transform / crs in any of {len(names)} shipped hdf5 files and no "
                   f"coordinates in the sample names (e.g. {names[0][1]!r})")  # fmt: skip
    df = pd.DataFrame(rows, columns=["dataset", "split", "sample_name", "source", *G.BOX])
    return df, rep


def region_boxes(task: str) -> pd.DataFrame:
    rows = [
        (f"region:{task}", "region", name, "declared_region", *box) for name, box in REGIONS[task].items()
    ]
    return pd.DataFrame(rows, columns=["dataset", "split", "sample_name", "source", *G.BOX])


def label_rows(git: pd.DataFrame, retained: pd.DataFrame, excluded: pd.DataFrame) -> pd.Series:
    """'retained' / 'excluded' for every located Git-10M row; the two manifests must partition the
    located rows exactly (a manifest from other metadata fails here)."""
    key = ["shard", "row"]
    located = git[G.check_boxes(git, "Git-10M rows", allow_unlocated=True)]
    idx = pd.MultiIndex.from_frame(located[key])
    in_r = idx.isin(pd.MultiIndex.from_frame(retained[key]))
    in_e = idx.isin(pd.MultiIndex.from_frame(excluded[key]))
    if (in_r & in_e).any() or not (in_r | in_e).all() or len(retained) + len(excluded) != len(located):
        raise SystemExit(
            f"dedup manifests do not partition the located rows: located {len(located)}, retained {len(retained)}, "
            f"excluded {len(excluded)}, in both {int((in_r & in_e).sum())}, in neither {int((~(in_r | in_e)).sum())}"
        )
    return pd.Series(np.where(in_r, "retained", "excluded"), index=located.index)


def summarize(
    matches: pd.DataFrame, git: pd.DataFrame, status: pd.Series, dataset: str, buffer_m: float
) -> dict:
    """Counts of DISTINCT training rows within the buffer of any sample of `dataset`."""
    m = matches[matches["dataset"] == dataset]
    per_row = m.groupby(["shard", "row"], sort=False)["distance_m"].min().reset_index()
    info = git.loc[status.index, ["shard", "row", "source"]].assign(status=status.to_numpy())
    per_row = per_row.merge(info, on=["shard", "row"], how="left", validate="one_to_one")
    if per_row["status"].isna().any():
        raise SystemExit(f"{dataset}: a matched row is not a located Git-10M row")
    ret = per_row[per_row["status"] == "retained"]
    return {
        "buffer_m": buffer_m,
        "eval_samples_hit": int(m["sample_name"].nunique()),
        "training_rows_within_buffer": int(len(per_row)),
        "of_which_already_excluded": int((per_row["status"] == "excluded").sum()),
        "of_which_retained": int(len(ret)),
        "retained_rows_overlapping_a_sample": int((ret["distance_m"] == 0).sum()),
        "retained_by_source": {k: int(v) for k, v in ret["source"].value_counts().sort_index().items()},
    }


def measure(seg_root: str, git: pd.DataFrame, retained: pd.DataFrame, excluded: pd.DataFrame,
            points: pd.DataFrame | None, buffer_m: float, tasks=None, readers=None) -> tuple[dict, pd.DataFrame]:  # fmt: skip
    boxes, rep = [], {"tasks": {}}
    for task in tasks or SEG_TASKS:
        df, r = task_boxes(task, os.path.join(seg_root, SEG_TASKS[task]["dir"]), (readers or {}).get(task))
        rep["tasks"][task] = r
        if r["assessable"]:
            boxes.append(df)
        else:
            boxes.append(region_boxes(task))
            r["region"] = {k: list(v) for k, v in REGIONS[task].items()}
    ev = pd.concat(boxes, ignore_index=True)
    gate = G.control(ev, buffer_m)
    rep["control"] = gate
    if not gate["pass"]:
        return rep, pd.DataFrame()
    status = label_rows(git, retained, excluded)
    matches = G.nearby(git.loc[status.index], ev, buffer_m)
    # an empty result has object columns, which the merges below would refuse to join on
    matches = matches.astype({"shard": "int64", "row": "int64", "distance_m": "float64"})
    for task, r in rep["tasks"].items():
        if r["assessable"]:
            r["overlap"] = summarize(matches, git, status, task, buffer_m)
            if points is not None and SEG_TASKS[task]["name_parser"]:
                cls = set(points.loc[points["dataset"] == SEG_TASKS[task]["name_parser"], "sample_name"])
                names = set(ev.loc[ev["dataset"] == task, "sample_name"])
                r["names_shared_with_" + SEG_TASKS[task]["name_parser"]] = len(names & cls)
        else:
            s = summarize(matches, git, status, f"region:{task}", buffer_m)
            r["region_upper_bound"] = {
                "meaning": "located training rows inside the declared region; NOT a hit count",
                "training_rows_in_region": s["training_rows_within_buffer"],
                "of_which_already_excluded": s["of_which_already_excluded"],
                "of_which_retained": s["of_which_retained"],
                "retained_by_source": s["retained_by_source"],
            }
    rep["git10m"] = {"rows": int(len(git)), "located": int(len(status)), "retained": int(len(retained)),
                     "excluded": int(len(excluded)), "without_coordinates": int(len(git) - len(status))}  # fmt: skip
    keep = matches.merge(git.loc[status.index, ["shard", "row", "img_name"]].assign(status=status.to_numpy()),
                         on=["shard", "row"], how="left")  # fmt: skip
    keep = keep[(keep["status"] == "retained") & ~keep["dataset"].str.startswith("region:")]
    return rep, keep.reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--seg-root", required=True, help="dir holding m-pv4ger-seg/ m-chesapeake-landcover/ m-SA-crop-type/"
    )
    ap.add_argument("--meta", required=True, help="git10m_meta dir (787 parquets)")
    ap.add_argument("--expect-shards", type=int, default=G.N_SHARDS)
    ap.add_argument("--source-boxes", required=True)
    ap.add_argument(
        "--dedup", required=True, help="existing dedup dir (retained.csv, excluded.csv); read only"
    )
    ap.add_argument(
        "--points", default=None, help="classification eval points csv (for the shared-name count)"
    )
    ap.add_argument("--buffer-m", type=float, required=True)
    ap.add_argument("--out", required=True, help="prefix; writes <out>.json and <out>.retained_matches.csv")
    a = ap.parse_args(argv)
    for ext in (".json", ".retained_matches.csv"):
        if os.path.exists(a.out + ext):
            raise SystemExit(f"{a.out + ext} exists: refusing to overwrite")
    git, missing = G.load_meta(a.meta, a.expect_shards, False)
    git = G.apply_source_boxes(git, pd.read_parquet(a.source_boxes))
    key = ["shard", "row"]
    retained = pd.read_csv(os.path.join(a.dedup, "retained.csv"), usecols=key)
    excluded = pd.read_csv(os.path.join(a.dedup, "excluded.csv"), usecols=key)
    points = pd.read_csv(a.points) if a.points else None
    rep, keep = measure(a.seg_root, git, retained, excluded, points, a.buffer_m)
    rep.update(seg_root=a.seg_root, dedup=a.dedup, source_boxes=a.source_boxes, buffer_m=a.buffer_m)
    print("\n".join(rep["control"]["detail"]), flush=True)
    print(f"GATE instrument control {'PASS' if rep['control']['pass'] else 'FAIL'}", flush=True)
    if not rep["control"]["pass"]:
        print("GATE FAIL: diagnose nearby()/box_gap_m before trusting any count", flush=True)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    keep.to_csv(a.out + ".retained_matches.csv", index=False)
    print(json.dumps(rep["tasks"], indent=1), flush=True)
    print(f"SEG_OVERLAP_OK -> {a.out}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
