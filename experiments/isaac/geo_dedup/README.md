# Geographic exclusion: Git-10M vs GEO-Bench

`eval/geo_dedup.py` removes Git-10M (lcybuaa/Git-10M) samples that lie at or near GEO-Bench
classification evaluation locations. Inputs, the rule and every threshold are explicit; the
outputs are plain CSV manifests plus `report.json`.

**Second pass (2026-10-11):** the public-source Git-10M rows and the `id_N` GEO-Bench tasks are
located too, where their upstream artifacts allow it. The as-executed protocol, its executed
controls and its explicit limits are in `PROTOCOL.md` (this is an attempt at deduplication,
not a proof of zero overlap). Quick map of the additions:

| Side | Rows / samples | How located | Subcommand |
|---|---|---|---|
| Git-10M SSL4EO-S12 | 251,079 | authors' centre file; name rule `n = 4i + 1` verified by `ssl4eo-verify` | `source-boxes` |
| Git-10M SkyScript | 954,641 images | authors' `meta6.zip` bbox per image (restricted unpickler) | `source-boxes` |
| Git-10M Million-AID / GeoPile / DIOR / AID / RSI-CB / VGoogle | 2.05 M | not locatable (no coordinates published) | reported |
| m-eurosat | 18,192 | pixel hash join to the 27,000 EuroSAT MS GeoTIFFs | `eval-points --originals-root` |
| m-bigearthnet | 22,000 | pixel hash join to the 590,326 BigEarthNet-S2 v1.0 patches (`bigearthnet-index`) | `eval-points --originals-root` |
| m-so2sat | 21,964 | not locatable | reported |

Inputs on scratch (`$STORE/geo_dedup/source_meta/`): `ssl4eo/ssl4eo_centers.bin` (authors'
Google Drive file, sha256 e9e5e5d3...), `ssl4eo/void_ids.csv`, `ssl4eo/hf/.../ssl4eo_multilabel.json`
(HF wangyi111/SSL4EO-S12), `skyscript/meta6.zip` (S3 opendatasharing, sha256 48d0da99...),
`eurosat/EuroSAT_MS.zip` (Zenodo 7711810, md5 091174ad...), `bigearthnet/BigEarthNet-S2-v1.0.tar.gz`
(Zenodo 12687186, md5 5a64e9ce...), `geobench_converters/` (GEO-Bench's own converter sources,
the evidence that pixels were stored unchanged). Jobs: `git10m_text.sbatch`, `fetch_bigearthnet.sbatch`,
`bigearthnet_index.sbatch`, `dedup_v2.sbatch` (see PROTOCOL.md §6).

Second-pass result (`$STORE/geo_dedup/dedup_v2_b1000/`, 2026-10-11, all controls PASS):
8,039,068 retained / 418,713 excluded / 2,045,786 unassessable of 10,503,567 rows; m-eurosat
and m-bigearthnet matched 100 % of their samples with 0 multi-matches. Full table in PROTOCOL.md §7.

**Third pass (2026-10-13, `dedup_v3_b1000_cities/`):** m-so2sat covered at CITY level. So2Sat
LCZ42's 52 cities (42 training + 10 validation/test, paper appendix B) are `so2sat_cities.csv`
here (geocoded with OSM Nominatim on 2026-10-13, osm id + display name per row), each a
centre +- 25 km `city_roi` box (the paper's "approximately 50 x 50 kilometers" ROI);
`python -m eval.geo_dedup city-points` adds them to the points file, `dedup_v2.sbatch` with
`CITIES=experiments/isaac/geo_dedup/so2sat_cities.csv` runs the pass. Result: 1,455,321 excluded
(1,222,564 inside a city box) / 7,002,460 retained. Adopted (user decision 2026-10-14): training
uses the pass-3 unique-image index `$STORE/git10m/index_all_cities_unique.npz` (8,417,991 rows,
retained + unassessable), validation `index_retained_cities_unique.npz` (6,711,575); PROTOCOL.md
§7b and the index table there.

**Residual content overlap (instrument built 2026-10-13, run 2026-10-14, job 6406657):**
`eval/content_overlap.py` + `content_overlap.sbatch` embed every unique unlocatable image with
DINOv3-SAT, match against the cached GEO-Bench features, and count hits above a null-calibrated
threshold with planted positives and executed gates. Result: no near-copy overlap detectable above
chance at any false-positive rate (hits below the null's expectation while every planted identical
copy is recalled). Upper bound: 597 of 1,706,393 rows (0.035 %) score above the 1e-3 threshold,
26 of them nearest to a val / test image; the raw hit count is the bound, the (negative) excess
is not. PROTOCOL.md §8 holds the table and the instrument's limits.

## What can and cannot be located (first pass)

Decided from the shipped artifact, per task, by `eval-points` (first sample of the partition):

| GEO-Bench task | location source | status |
|---|---|---|
| m-brick-kiln | hdf5 band attrs carry transform + crs (EPSG:4326, 64 px at 10 m) | assessable, **footprint** box |
| m-forestnet | no georef in hdf5; sample name `lat_lon_YYYY_MM_DD` (nominal centre) | assessable, centre + nominal box |
| m-pv4ger | no georef in hdf5; sample name `lat,lon` (some written `lon,lat`; decided by Germany's disjoint lat/lon ranges) | assessable, centre + nominal box |
| m-eurosat | `id_N`; hdf5 has no crs/transform (verified id_0000) | **unassessable** |
| m-so2sat | `id_N`; hdf5 has no crs/transform (verified id_0005); So2Sat LCZ42 publishes no coordinates | **unassessable** |
| m-bigearthnet | `id_N`; hdf5 has no crs/transform (verified id_0034) | **unassessable** |

Git-10M rows: Google Earth tiles carry `Google_location = "z_x_y"` (XYZ Web Mercator tile, the
dataset card's `XYZToLonLat`), an exact footprint. Rows taken from public datasets have
`Google_location = None` and are **unassessable** (listed in `unassessable.csv`, never in
`retained.csv`). Shard 0 is all tiles (13,347 rows), shard 786 is all None (13,346 rows).

Rule: a tile is excluded when the great-circle gap between its lon/lat box and an evaluation
sample's box is `<= --buffer-m` (0 m = the boxes overlap). Evaluation boxes are the reprojected
corner bbox where a footprint exists (a conservative superset), else the nominal centre box
`centre +- half_extent_m` with `half_extent_m` = export image px x coarsest band resolution / 2
(m-forestnet 3840 m, m-pv4ger 12.8 m). A nominal box is never reported as a footprint
(`source` column in the points csv). Candidates come from a KD-tree over the evaluation box
centres (radius = tile half-diagonal + largest evaluation half-diagonal + buffer), then the
exact gap test; antimeridian-crossing pairs are handled by trying +-360 deg.

## Commands

```bash
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
STORE=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing
# 1. Git-10M metadata (787 shards, 390 GB as converted arrow, nothing kept but the parquet files):
mkdir -p logs && sbatch --array=0-11 experiments/isaac/geo_dedup/git10m_meta.sbatch
# 2. Evaluation points (all six tasks; the four unassessable ones are reported, not skipped):
python -m eval.geo_dedup eval-points --geobench-root $STORE/datasets --raw-root $STORE/geo_dedup/geobench_raw \
    --out $STORE/geo_dedup/eval_points.csv   # raw-root: <task>/data.zip from Zenodo (md5-check it first)
# 3. Dry run, then the manifests (refuses to overwrite; refuses < 787 shards unless --allow-partial):
python -m eval.geo_dedup dedup --meta $STORE/geo_dedup/git10m_meta --expect-shards 787 \
    --points $STORE/geo_dedup/eval_points.csv --buffer-m 1000 --out $STORE/geo_dedup/dedup_b1000 --dry-run
python -m eval.geo_dedup dedup --meta $STORE/geo_dedup/git10m_meta --expect-shards 787 \
    --points $STORE/geo_dedup/eval_points.csv --buffer-m 1000 --out $STORE/geo_dedup/dedup_b1000
# or as a CPU job (same gates): sbatch --export=ALL,BUFFER_M=1000 experiments/isaac/geo_dedup/dedup.sbatch
```

`dedup` first runs an executed instrument control (rule 13): one synthetic z=17 tile on the
first sample of every assessable dataset must be excluded at distance 0 and one at its
antipode must be retained; a FAIL stops the run. Outputs in `--out`: `retained.csv`,
`excluded.csv` (datasets, matched ids, min distance), `unassessable.csv` (reason),
`matches.csv` (every pair), `report.json` (`complete` is false whenever shards are missing;
`control` holds the gate result). Step 1 is resumable: finished shards are skipped. Identity of a Git-10M row is `(shard, row)`. A Google tile `img_name`
must be unique or the run stops. Public-dataset images repeat on adjacent rows (one row per
caption, 1,287,310 rows in pass 1; 602,844 in passes 2-3 after SSL4EO / SkyScript rows were located); they are all unassessable and counted in
`unassessable_rows_with_repeated_img_name`.

## Limitations (first pass; see PROTOCOL.md §5 for the second)

- Three of six tasks are unassessable in the first pass; the second pass locates m-eurosat and
  m-bigearthnet by pixel join. m-so2sat stays unassessable.
- This covers overlap between Git-10M (our adaptation data) and GEO-Bench only. It does not
  show that DINOv3-SAT, CLIP, RemoteCLIP or GeoRSCLIP pretraining was independent of GEO-Bench.
- It is geographic only; there is no image-similarity deduplication in this repository.
- The nominal half extents assume the sample name is the image centre (ForestNet and PV4GER
  convention); `--buffer-m` is the slack for that and for the clamp approximation.

## Locating the id_N tasks from the upstream originals (implemented in the second pass)

Originally sketched here; now `join_originals` / `eurosat_index` / `bigearthnet_index`. The shipped `id_N` names carry no
link to the originals, so every route is a pixel-level join (hash the raw band values of each
GEO-Bench sample, hash the same bands of every original patch, match exactly; rule 16: verify
on the artifact that GEO-Bench did not rescale).

- m-eurosat: EuroSAT MS (27,000 GeoTIFFs, ~2 GB, georeferenced) -> feasible.
- m-bigearthnet: BigEarthNet-S2 (590,326 patches, ~66 GB; patch json carries UTM bounds) ->
  feasible but heavy; hash only the 10 m bands.
- m-so2sat: So2Sat LCZ42 ships no coordinates in any version -> not locatable.
