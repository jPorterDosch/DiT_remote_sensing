# Git-10M vs GEO-Bench deduplication protocol (as executed, 2026-10-11)

This is an **attempt** at removing GEO-Bench evaluation locations from the Git-10M adaptation
set. It is not a proof of zero overlap: two Git-10M sources and one GEO-Bench task carry no
coordinates anywhere, and all other rows are matched on coordinates, not on image content.
Every number below comes from an executed script with a PASS/FAIL control; the files named are
the evidence. Nothing in this protocol touches the DINOv3-SAT / CLIP / RemoteCLIP / GeoRSCLIP
pretraining corpora, whose independence from GEO-Bench is unknown.

## 1. Rule

A Git-10M row is **excluded** when the great-circle gap between its lon/lat box and any
GEO-Bench evaluation sample's box (train, valid and test splits of all six classification tasks)
is `<= 1000 m` (0 m = the boxes overlap). The 1000 m buffer absorbs the nominal-centre
approximation of two GEO-Bench tasks and the corner-bbox approximation of reprojected
footprints. A row is **retained** when it has a box and matches nothing. A row is
**unassessable** when neither Git-10M nor its upstream source publishes a location; it is
listed with its source and reason, never silently retained.

The buffer is measured from the **box edge**, not from the centre: a sample's own footprint
(EuroSAT 640 m, BigEarthNet 1,200 m, SSL4EO-S12 2,640 m, brick-kiln and SkyScript whatever their
georeferencing says) IS the box, and the 1000 m is added outside it. Patch size therefore never
has to be folded into the buffer and one value covers every task; only the two nominal-centre
tasks (m-forestnet +-3,840 m, m-pv4ger +-12.8 m) and SSL4EO (+-1,320 m) use a documented nominal
extent instead of a measured footprint.

Matching: KD-tree over the evaluation box centres (radius = half-diagonals + buffer), then the
exact box-gap test; antimeridian pairs handled by trying +-360 deg; evaluation boxes wider than
10 km (the city ROIs of section 7b) skip the tree and are tested against every row directly.
Code: `eval/geo_dedup.py` (`nearby`, `_nearby_big`, `box_gap_m`), tests in `tests/test_geo_dedup.py`.

Geometry model: a sphere of IUGG mean radius (6,371,008.8 m). Against the WGS84 geodesic
(Vincenty, `experiments/isaac/geo_dedup/verify_bruteforce/ellipsoid_check.py`, 640 m boxes at
0-70 deg latitude, both separations) a gap that the rule reads as exactly 1,000 m is truly
994.4-1,004.1 m: north-south pairs below 45 deg are over-read (sphere says 1,000 m for a true
994.4 m at the equator, i.e. a pair up to 5.6 m INSIDE the buffer can be retained), east-west
pairs and high latitudes are under-read (a true gap up to 4.1 m outside the buffer is still
excluded). The buffer is therefore 1,000 m +- 6 m, which is why sub-metre boundary effects
(section 7c, the 2026-10-14 latitude fix) are not worth a rebuild.

## 2. Git-10M side: where each row's box comes from

Git-10M (lcybuaa/Git-10M, 787 arrow shards, 10,503,567 rows) ships `Google_location` for
Google Earth tiles only. The rest are images copied from public datasets, identified by their
`img_name` prefix. Decided per source from the upstream artifact (CLAUDE.md rule 16):

| Source (`img_name` prefix) | Rows | Location | Box |
|---|---|---|---|
| Google Earth tile (`Google_location = z_x_y`) | 6,895,978 | XYZ Web Mercator tile, exact | tile bounds |
| SSL4EO-S12 (`SSL4EO-S12-rgb-<n>`) | 251,079 | authors' centre file (251,079 ids; 97 void ids); `n = 4 i + 1`, i = index into the centres in ascending id order | centre +- 1320 m (264 px at 10 m) |
| SkyScript (`Sky-images6-<stem>`) | 1,310,724 rows, 954,641 unique images | authors' `meta6.zip`: `bbox = (lon_min, lat_min, lon_max, lat_max)` per image | exact footprint (10 cm orthophotos: 10-700 m; S2/Landsat scenes: 1-40 km, p99.9 = 13.7 km) |
| Million-AID, GeoPileV0, DIOR, AID, RSICB, VGoogle | 2,045,786 | none published | unassessable |

The SSL4EO name rule was found by inspection (all 251,079 names satisfy `n mod 4 = 1`,
`n div 4` is a permutation of 0..251078) and **verified by an executed gate** (`ssl4eo-verify`,
`source_meta/ssl4eo/verify.json`): Git-10M's own captions against the authors' per-location
land-cover labels, versus a permuted mapping, separately below and above the first void id
(a mapping that ignored the voids passes the first half and fails the second). Result:
keyword classes trees / crops / built / bare / snow gain +0.15 to +0.44 over the null,
identically in both halves; water gains nothing (71% base rate). PASS. The gate's criteria
were fixed after the first real run and the code comment says so.

SkyScript pickles are read with an unpickler that refuses every global except numpy's float
scalar (a planted pickle cannot run code); the bbox order was checked on one Brandenburg
image against its `DE` (Brandenburg 20 cm orthophoto) source alias.

## 3. GEO-Bench side: where each sample's box comes from

| Task | Samples | Location source | Box |
|---|---|---|---|
| m-brick-kiln | 17,061 | shipped hdf5 transform + crs (EPSG:4326) | footprint (reprojected corner bbox) |
| m-forestnet | 8,446 | sample name `lat_lon_YYYY_MM_DD` | centre +- 3840 m (256 px x 30 m) |
| m-pv4ger | 13,812 | sample name `lat,lon` (some `lon,lat`; decided by Germany's disjoint ranges) | centre +- 12.8 m (256 px x 0.1 m) |
| m-eurosat | 18,192 | **pixel join**: GEO-Bench's converter stored EuroSAT MS pixels unchanged (uint16 -> int16) with no georeferencing; each sample's B02+B04 pixels are hashed and matched to the one GeoTIFF among the 27,000 in `EuroSAT_MS.zip` (Zenodo 7711810, md5-checked) with identical pixels; the GeoTIFF's crs + transform give the footprint | footprint |
| m-bigearthnet | 22,000 | **pixel join**: converter stored the 120 x 120 10 m bands unchanged (uint16 -> int32); B02 pixels hashed and matched to the 590,326 patches of `BigEarthNet-S2-v1.0.tar.gz` (Zenodo 12687186, md5-checked, streamed, never extracted); the patch json's corner coordinates + projection give the footprint | footprint |
| m-so2sat | 21,964 | **none**: So2Sat LCZ42 publishes no coordinates in any version | unassessable |

A pixel join that leaves any sample unmatched is fatal (`join_originals`), so a located task is
located completely or not at all. Samples that match several originals (duplicated patches in
the source dataset) take the union box and are counted in the points report.

## 4. Executed controls (every run prints PASS/FAIL and stops on FAIL)

- `dedup` plants, per assessable GEO-Bench task, a synthetic z=17 tile on its first sample
  (must be excluded at 0 m), one at the antipode (must be retained), the same sample as a
  non-tile row located through source boxes (must be excluded), and that box shifted east to a
  gap of 0.99 x buffer (must match the planted sample at 0.97-1.0 x buffer) and 1.01 x buffer
  (must not match the planted sample; neighbouring samples may, real footprints have adjacent
  patches): a wrong buffer unit or a gap error beyond 1-3 % fails here. Verified 2026-10-14 to
  PASS on `points_v3_cities.csv` and to FAIL under km units, +-5 %, +-2 %, a sign flip and a
  dead instrument.
- `ssl4eo-verify` as above.
- `source-boxes` refuses a name that resolves twice; `dedup --source-boxes` refuses a located
  name that is a Google tile or absent from the metadata, and a source label that disagrees
  with the name prefix.
- Every downloaded artifact is md5/sha256-checked before use (EuroSAT MS, BigEarthNet-S2,
  GEO-Bench raw zips; SkyScript meta6.zip by our own recorded sha256, the authors publish none).

## 5. Protocol yes, zero overlap no: the residual channels (say so when quoting the numbers)

The exclusion is a protocol with executed controls, not a proof of disjointness. Three channels
remain, each with what was done about it:

1. **Unassessable sources -- 2,045,786 Git-10M rows (19.5 %, 1,706,416 unique images)**:
   Million-AID, GeoPile, DIOR, AID, RSI-CB and VGoogle publish no coordinates. These are mostly
   0.3-1 m Google Earth / aerial scene patches, so content overlap with the 10-30 m
   Sentinel/Landsat GEO-Bench tasks is implausible; overlap with m-pv4ger (10 cm aerial, Germany)
   and m-brick-kiln is possible. Decision (mentor, 2026-10-13): **keep** them (dropping costs
   17.4 % of the unique images and all of the class-curated, object-centric content) and
   **measure** the residual overlap by content instead -- section 8 turns it into a number
   with a null-calibrated threshold, planted positives and executed gates.
2. **m-so2sat cannot be located** (So2Sat LCZ42 publishes no coordinates in any version). The
   42 + 10 cities are published, and SSL4EO-S12 -- locatable and city-sampled -- is the riskiest
   Git-10M source for it. Section 7b covers m-so2sat at **city level** (~50 x 50 km ROIs around
   the 52 city centres): coarse, but a measured count rather than nothing. Whether the training
   index uses it is a flag (`index_*_cities.npz`), see section 7b.
3. **Geographic only.** Two images of the same place from different sensors or dates are
   caught; a re-tiled or re-cropped copy of the same scene whose published coordinates are
   wrong is not. Coordinates are trusted as published. Section 8's content pass is the only
   instrument that sees this channel, and it runs on the unassessable rows, not on all 10.5 M.

Plus two approximations: **nominal centres** (m-forestnet, m-pv4ger, SSL4EO-S12) assume the
published point is the image centre, with the 1000 m buffer as slack; and **nothing is said
about the pretraining data of the frozen backbones** -- DINOv3-SAT (SAT-493M), CLIP, RemoteCLIP
and GeoRSCLIP were never deduplicated against GEO-Bench by anyone. That cuts the other way in
the write-up: the Git-10M adaptation set is deduplicated against the evaluation tasks more
rigorously than any baseline's pretraining corpus, which is a property to state, not to
apologise for.

## 6. Reproduction (ISAAC, worktree `DiT_remote_sensing-geodedup`, env DiTF)

```bash
STORE=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing; G=$STORE/geo_dedup
sbatch --array=0-11 experiments/isaac/geo_dedup/git10m_meta.sbatch      # Git-10M metadata (787 shards)
sbatch experiments/isaac/geo_dedup/git10m_text.sbatch                   # SSL4EO captions (for the gate)
python -m eval.geo_dedup ssl4eo-verify --text $G/git10m_text_ssl4eo \
    --multilabel $G/source_meta/ssl4eo/hf/ssl4eo-s12-ml/ssl4eo_multilabel.json \
    --ssl4eo-centres $G/source_meta/ssl4eo/ssl4eo_centers.bin --ssl4eo-void $G/source_meta/ssl4eo/void_ids.csv \
    --out $G/source_meta/ssl4eo/verify.json
sbatch experiments/isaac/geo_dedup/fetch_bigearthnet.sbatch             # 66 GB, md5-checked
sbatch --dependency=afterok:<fetch> experiments/isaac/geo_dedup/bigearthnet_index.sbatch
sbatch --export=ALL,BUFFER_M=1000,STEP=source-boxes experiments/isaac/geo_dedup/dedup_v2.sbatch
sbatch --export=ALL,BUFFER_M=1000 --dependency=afterok:<index> experiments/isaac/geo_dedup/dedup_v2.sbatch
```
Inputs on scratch: `$G/source_meta/{ssl4eo,skyscript,eurosat,bigearthnet}` (download provenance in
`experiments/isaac/geo_dedup/README.md`). Outputs: `$G/source_boxes.parquet(.report.json)`,
`$G/points_v2.csv(.report.json)`, `$G/dedup_v2_b1000/{retained,excluded,unassessable,matches}.csv`
and `report.json` (per-source and per-task counts, control results).

## 6b. SSL4EO-S12 name rule: pixel-level verification (2026-10-13)

The statistical gate (§4) was followed by a direct image check, `eval/ssl4eo_pixel_check.py`
(job `ssl4eo-pixel-check` 6384912, `experiments/isaac/geo_dedup/ssl4eo_pixel_check.sbatch`).
SSL4EO-S12 v1.1 (HF `embed2scale/SSL4EO-S12-v1.1`) keeps the v1.0 sample ids (243,968 of the
251,079 mapped ids are present) and ships an S2RGB modality (4 seasons, 264 px). For 1,280
Git-10M SSL4EO rows (20 random v1.1 shards x 64 rows, seed 0) the Git-10M image was correlated
(greyscale Pearson, max over 4 seasons x {bicubic resize, centre crop}) with the patch at the
MAPPED id and with the patch of another sampled location (null):

| | mapped | null |
|---|---|---|
| median correlation | 0.891 | 0.055 |
| rows with mapped > null | 98.9 % (below first void 99.1 %, above 98.7 %) | |
| rows with correlation >= 0.9 | 46.7 % | |

Gate thresholds (fixed before the run: median mapped >= 0.6, median null <= 0.3, mapped > null
for >= 95 % overall and in each half of the id range): **PASS**. 1,215 of 1,280 best matches
were season 0, i.e. Git-10M carries the first seasonal image, consistent with `n = 4i + 1`.
The 1.1 % of rows not above their null (0.9 % strictly below; 3 of the 14 are constant images scoring 0.0 on both) and the 8 % below 0.5 are consistent with clouds / snow in
one of the two acquisitions (v1.1 was re-downloaded; dates can differ from v1.0); they were
not inspected individually. Result file:
`$STORE/geo_dedup/source_meta/ssl4eo_v11/ssl4eo_pixel_check_s0_n20x64.json` (per-row scores).

## 7. Results

First pass (tiles only, `dedup_b1000/`, 2026-10-11): 6,890,925 retained, 5,053 excluded
(brick-kiln 4,032; forestnet 733; pv4ger 288 tile hits), 3,607,589 unassessable.

Second pass (`dedup_v2_b1000/`, 2026-10-11, jobs 6371991 -> 6372081 -> 6372226, plus 6372227 for
`source_boxes.parquet`; every step printed its `_OK` line; all 15 executed controls PASS):

| | rows |
|---|---|
| Git-10M rows | 10,503,567 |
| retained | 8,039,068 |
| excluded (box gap <= 1000 m to an evaluation sample) | 418,713 (5.0 % of the assessable rows) |
| unassessable (Million-AID, GeoPile, DIOR, AID, RSI-CB, VGoogle) | 2,045,786 (19.5 %; 1,706,416 unique images) |

Per source: Google tiles 6,895,978 -> 333,248 excluded; SkyScript 1,310,724 rows (954,641 images,
0 missing in `meta6.zip`) -> 80,814 excluded; SSL4EO-S12 251,079 -> 4,651 excluded.

Per task (`eval_samples_hit` = evaluation samples with at least one Git-10M row within the
buffer; a Git-10M row can hit several tasks):

| task | samples | located by | Git-10M rows excluded | samples hit |
|---|---|---|---|---|
| m-eurosat | 18,192 | pixel join, 18,192/18,192 matched, 0 multi-match | 284,426 | 15,077 |
| m-bigearthnet | 22,000 | pixel join, 22,000/22,000 matched, 0 multi-match | 137,166 | 17,571 |
| m-brick-kiln | 17,061 | hdf5 transform + crs | 4,229 | 10,374 |
| m-pv4ger | 13,812 | centre in name, +-12.8 m | 894 | 11,690 |
| m-forestnet | 8,446 | centre in name, +-3,840 m | 895 | 6,199 |
| m-so2sat | 21,964 | not locatable | 0 | -- |

The two European Sentinel-2 tasks drive the count: Git-10M is dense over Europe at zoom 17-18
(212,666 of the 249,002 EuroSAT-excluded tiles are z17-18; only 2.4 % of all tile exclusions
are z <= 13, i.e. coarse tiles that merely contain a sample). First-pass numbers above are
SUPERSEDED by this pass (same rule and buffer; the difference is the added locations).

The 418,713 by task, exclusively and by split and source (`dedup_v2_b1000/matches.csv`,
883,503 (row, sample) pairs; 8,895 rows hit more than one task, 2026-10-13):

| task | rows hit | rows hit ONLY by this task | train / valid / test samples' rows | Google tile / SkyScript / SSL4EO | rows at 0 m (overlap) |
|---|---|---|---|---|---|
| m-eurosat | 284,426 | 275,532 | 263,827 / 19,156 / 21,221 | 249,002 / 32,038 / 3,386 | 42,227 |
| m-bigearthnet | 137,166 | 128,563 | 132,590 / 23,086 / 18,236 | 83,602 / 52,563 / 1,001 | 52,417 |
| m-brick-kiln | 4,229 | 4,229 | 3,977 / 756 / 797 | 4,032 / 195 / 2 | 1,615 |
| m-forestnet | 895 | 895 | 710 / 246 / 335 | 733 / 162 / 0 | 833 |
| m-pv4ger | 894 | 599 | 690 / 399 / 444 | 288 / 163 / 443 | 531 |

(A row can sit near samples of several splits; the split columns count rows per split and do
not sum to the row total.) Most exclusions are within the 1000 m buffer rather than on top of a
sample: 96,801 distinct rows overlap an evaluation footprint at 0 m (the per-task 0 m column sums to 97,623 because 822 rows touch two tasks), the rest sit 0-1000 m away.

### 7b. Third pass: m-so2sat at city level (`dedup_v3_b1000_cities/`, 2026-10-13, job 6388296)

So2Sat LCZ42 (Zhu et al. 2020, sec. III.B.2, appendix B) was labelled inside "a rectangle of
approximately 50 x 50 kilometers around the city center" of 42 training cities plus 10 cities
whose western / eastern halves are its validation / test sets (torchgeo So2Sat version 2, which
GEO-Bench's converter consumed: training.h5 -> train, validation.h5 -> valid, testing.h5 ->
test). GEO-Bench's `id_N` names keep no city, so every So2Sat city is treated as evaluation
territory: `experiments/isaac/geo_dedup/so2sat_cities.csv` holds the 52 names from the paper's
appendix B, geocoded with OSM Nominatim (2026-10-13; the OSM relation / node id and display
name are recorded per row), and `eval.geo_dedup city-points` adds one `city_roi` box of
centre +- 25 km per city (split `train` for the 42, `test` for the 10) to `points_v2.csv` ->
`points_v3_cities.csv`. Same rule, same 1000 m buffer, same executed controls (18 PASS). Known
coarseness: an OSM administrative centroid can sit well off the labelled centre (Moscow's is
~14 km south of the Kremlin), the ROIs were hand-drawn, and GEO-Bench's subset of So2Sat may not
touch every city.

| | rows |
|---|---|
| excluded, pass 2 (coordinates) | 418,713 |
| excluded, pass 3 (coordinates + 52 city boxes) | 1,455,321 (13.9 % of all rows) |
| of which inside a So2Sat city box | 1,222,564 (1,036,608 hit by no other task) |
| retained | 7,002,460 |
| unassessable | 2,045,786 (unchanged) |

Inside the city boxes: Google tiles 1,019,551, SkyScript 199,023, SSL4EO-S12 3,990 rows
(SSL4EO is city-sampled, but only 3,990 of its 251,079 rows fall in these 52 cities -- its
sampling spreads over ~10,000 cities). Rows per city are dominated by Europe: London 256,193,
Paris 201,668, Berlin 162,477, Melbourne 96,179, Zurich 70,666, Moscow 47,026, Chicago 45,877;
the 10 held-out cities account for 83,126 rows. Salvador (15) and Karachi (47) are the smallest.

Cost of adopting the city pass: 1,036,608 more rows (9.9 % of Git-10M, almost all Google
Earth tiles of major cities) for a task whose 32 x 32 px, 10 m Sentinel-2 patches share
nothing visual with 0.3 m urban tiles -- the channel it closes is "same city, same sensor",
i.e. SSL4EO-S12 and Sentinel-2 SkyScript scenes, which are 3,990 + a fraction of 199,023 rows.
Both choices are built as indexes so the decision is a flag, not a rerun:

| index (`$STORE/git10m/`) | manifests | unique images (manifest rows) |
|---|---|---|
| `index_retained_unique.npz` | pass 2, retained only | 7,705,674 (8,039,068) |
| `index_all_unique.npz` | pass 2, retained + unassessable | 9,412,090 (10,084,854) |
| `index_retained_cities_unique.npz` | pass 3, retained only | 6,711,575 (7,002,460) |
| `index_all_cities_unique.npz` | pass 3, retained + unassessable | 8,417,991 (9,048,246) |

Indexes keep one row per image (`experiments/isaac/git10m/README.md`), hence fewer rows than
the manifests. Decision (user, 2026-10-14): **train on pass 3**, `INDEX=$STORE/git10m/index_all_cities_unique.npz`
(keep the unassessable rows per the mentor, plus the city boxes, so the m-so2sat result is
defensible at the only resolution its samples can be located at). The pass-2 indexes stay
available for an ablation of the city cost.

Decision on the 2,045,786 unassessable rows (2026-10-11, pending the user's confirmation):
their sources are provenance-disjoint from every GEO-Bench source dataset and are sub-metre
to few-metre RGB crops, so the only leak channel is a coincidental same-location, cross-sensor
match; at the tile exclusion rate of this pass (4.8 %) that is at most ~1e5 rows that cannot
be identified, against 17.4 % of the unique images (and all of the class-curated, object-centric
content) if they are dropped. Whatever is chosen, the result is an attempt at deduplication,
not a complete one (section 5).

## 7c. Independent brute-force check of the artifacts (2026-10-13)

A second implementation of the rule (own tile-bounds formula, exact minimum box-to-box
great-circle gap; `experiments/isaac/geo_dedup/verify_bruteforce/`) re-decided random samples
of the v2 and v3 manifests: 2,000 excluded rows all within 1000 m, 20,000 retained rows all
beyond (closest 1000.89 m), 2,000 city-only exclusions all within 1000 m of a city box, 2,000
v3-retained rows all outside (nearest 1074.5 m), 500 `matches.csv` distances within 0.104 m;
tile bounds and the SSL4EO centre mapping agree exactly. Zero disagreements. Details and the
table in that directory's README.

**`box_gap_m` latitude fix (2026-10-14).** The manifests above were produced by a `box_gap_m`
that converted the longitude gap at the equator-nearest shared latitude; the staged code uses
the poleward one (the exact spherical minimum). The old value read up to 0.2 m HIGH and never
low, so the only possible error is a row with a true gap in (999.8, 1000] m being retained.
Measured by rerunning both passes with the staged code into new directories
(`dedup_v2_b1000_gapfix`, job 6402266; `dedup_v3_b1000_cities_gapfix`, job 6402267) and
diffing row by row (`experiments/isaac/geo_dedup/diff_manifests.py`, prints
`MANIFEST_DIFF_PASS` only when every moved row is retained -> excluded with the new gap inside
(999.8, 1000] m):

| pass | moved rows | rows (shard, row, new gap) |
|---|---|---|
| v2 | 2 of 10,503,567 | (336, 213, 999.988 m), (353, 5630, 999.991 m) |
| v3 | 1 of 10,503,567 | (353, 5630, 999.991 m) |

Both PASS. Decision (user, 2026-10-14): keep the original manifests and the indexes built from
them. The two rows are 2e-7 of the corpus and sit 9-12 mm inside the buffer, far below the
+-6 m spherical-vs-WGS84 model error at 1,000 m (section 1); re-deriving the content-overlap
null and reference samples (seeded draws over the retained row count) would cost a GPU
re-embed to move one training row. Anyone rebuilding the manifests from the staged code
will therefore see 7,002,459 retained / 1,455,322 excluded for v3 (one row fewer / more than
section 7b) and 8,039,066 / 418,715 for v2.

## 8. Residual content overlap of the unassessable rows (instrument built 2026-10-13; run 2026-10-14: gates PASS, no excess over the null -- result at the end of this section)

`eval/content_overlap.py` turns channel 1 of section 5 into a number. Every unique unassessable
image (1,706,416) is embedded with the SAME frozen DINOv3-SAT ViT-7B/16 and the same 224 px /
normalisation path that produced the cached GEO-Bench features (`models/dino_feats/
dinov3_vit7b16_sat_<task>_<split>.npz`, CLS of the last block, all six tasks x train/val/test =
79,511 + m-so2sat's 21,964 reference rows); its nearest reference by cosine is recorded.

- **Null** (sets the threshold, not the eye): 200,000 rows RETAINED by the city-level pass
  (`dedup_v3_b1000_cities`: located, > 1000 m from every evaluation box AND outside every
  So2Sat city ROI, so known to be a different place for all six tasks) through the identical
  path. The v2 retained set is NOT a valid null: m-so2sat is unlocated there, so 1.2 M of its
  retained rows sit inside So2Sat cities and could be the same place as a so2sat sample; the
  top tail of the null is what sets tau, and at fpr 1e-4 that tail is the top 20 rows of 200k
  (it would have been the top 2 of the first draft's 20k, hence the size). tau(fpr) = the
  (1 - fpr) quantile of the null's max-similarity; reported at fpr 1e-2, 1e-3, 1e-4. This null
  is deliberately hard: it contains European tiles a few km from EuroSAT / BigEarthNet samples,
  i.e. "a different place that looks alike".
- **What a count means**: at tau(fpr), fpr x 1,706,416 queries exceed tau by chance alone
  (1,706 at 1e-3, 171 at 1e-4). The report prints that expected-false count and the EXCESS
  (hits - expected false) next to the raw hits. The excess is evidence of overlap only when it
  is positive, and it is NOT a bound (corrected 2026-11-12; this paragraph called it an upper
  bound until then): the null's sources (Google tiles, SSL4EO-S12, SkyScript) are not the
  queries' (Million-AID, DIOR, AID, GeoPile, RSI-CB, VGoogle), no retained rows of those sources
  exist to calibrate per source, and a negative excess shows the null over-predicts the queries'
  false positives, so the excess under-states overlap. The upper bound on near-copy overlap
  detectable at tau is the RAW hit count (divided by the planted-positive recall of the
  transformation in question). The per-source hit table and the same-location reference below
  are what to read it against.
- **Planted positives**: 200 test images per task x {identical, h-flip, 80 % centre crop,
  JPEG q60, half resolution} pushed through the Git-10M path. Recall at each tau is reported per
  variant.
- **Same-location reference**: 20,000 EXCLUDED rows matched by a sample box (same place by
  coordinates, usually another sensor and date) -- what a true geographic duplicate scores
  under this instrument. Rows a So2Sat city ROI alone excluded (71 % of the pass-3 excluded
  set: somewhere inside a ~50 km city, not at a sample) are dropped (`embed --drop-city-only`,
  file `<preset>_excluded_loc_n20000.npz`); `match` refuses a reference embedded without it,
  and refuses a null, reference or query part that did not come from one manifest directory,
  or query parts that do not tile the unassessable set (every file's `origin` is checked).
- **Gates (executed, PASS/FAIL, the run refuses to report counts on FAIL)**: the identical
  positives must reproduce their own cached reference (cos >= 0.99 for >= 99 %), i.e. this
  embedding path and the cache's are the same instrument; every identical positive must be
  recalled at every tau; no tau may saturate (>= 0.999). `tests/test_content_overlap.py` feeds
  each gate the fault it exists to catch (a cache from a different path, overlapping parts, a
  misaddressed shard row) and checks that planted copies -- and only those -- are found on a
  synthetic shard.
- **Degenerate (no-data) images**: a uniform image has ONE DINO CLS vector whatever its source,
  so all-white tiles in Git-10M and all-white patches in GEO-Bench pair at cosine 1.0 and
  saturated tau at fpr 1e-4 on the first run (below). The controls now include three synthetic
  uniform images (white, black, mid-grey); `match` drops every row within cosine 0.999 of one of
  them from the reference, the null, the same-location reference, the controls and the queries
  BEFORE any statistic, and reports the counts per set and source (`degenerate` in report.json).
  A controls file without the synthetic blanks is refused.
- **Output**: `$STORE/geo_dedup/content/report_dinov3_vit7b16_sat/{report.md,report.json}` (hits
  per fpr, by source, by nearest task and split; null quantiles; per-variant recall), `hits.csv`
  (every image above the loosest tau with its nearest sample, best first -- the list to eyeball
  and, if adopted, to exclude by (shard, row)), `query_max_sim.npz` (per-image vectors, rule 12).

Run: `experiments/isaac/geo_dedup/content_overlap.sbatch` with
`--export=ALL,MANIFEST_DIR=$STORE/geo_dedup/dedup_v3_b1000_cities,NULL_N=200000` (GPU array of
8 parts + null + excluded + controls, then `STEP=match,NULL_N=200000` on CPU; the unassessable
set is byte-identical in v2 and v3, only the null and the same-location reference depend on the
pass). Sizing is an estimate until the first part
reports: ~3 TFLOP per image through the 7B backbone, 214 k images per part.

**First run (2026-10-14, job 6404537, 3 min 37 s on 16 CPUs).** `identity_reproduces_cache` PASS
(1,200 identity controls, min cos 0.99996 to their cached rows: the embedding path is the
cache's). `null_not_saturated` FAIL: tau = 0.660 / 0.748 / 1.000 at fpr 1e-2 / 1e-3 / 1e-4,
hence `identity_recalled_at_tau` FAIL at 1e-4 (recall 0.36 at a tau of 1.0). Diagnosis
(rule 2: the instrument, not the number): exactly 21 of the 200,000 null rows score >= 0.9 at
all, all 21 at cosine 1.000 against ONE reference row, m-pv4ger val
`8.178118859515267, 50.73970161447261.png`; the 21 are 12 SkyScript and 9 SSL4EO-S12 tiles with
byte-identical embeddings, and decoding them gives pure white images (all pixels 255, std 0);
the reference image is pure white too, as are 3 more m-pv4ger val/test patches (4 of 101,475
reference rows). Nothing else in the null exceeds 0.75 at the 1e-3 quantile, so the null is
otherwise clean. Outcome: the degenerate-image filter above; the controls are regenerated
with the synthetic blanks and `match` rerun. Note for training: Git-10M contains all-white
no-data tiles (SkyScript, SSL4EO-S12; 21 per 200k retained rows, 3 per 20k same-location
excluded rows); they are harmless for flow matching but worthless, and are not removed.

**Result (second run, 2026-10-14, controls regenerated with the blanks: job 6405497, 1 h 52 min
CPU; match: job 6406657, 13 min 55 s on 16 CPUs). All three gates PASS** (identity min cos
0.99996 to the cache; identity recall 1.000 at every tau; tau = 0.659 / 0.743 / 0.830 at fpr
1e-2 / 1e-3 / 1e-4, set by 2,000 / 200 / 20 null rows). Degenerate rows dropped before any
statistic: reference 4 (m-pv4ger val/test), null 21 (12 SkyScript, 9 SSL4EO-S12), same-location
excluded 3, queries 23 (all GeoPile), controls 0. Null (199,979 retained rows): median max-sim
0.395, q0.9 0.552, q0.99 0.659, q0.999 0.743, q0.9999 0.830.

| fpr | tau | hits (of 1,706,393) | expected false | excess | recall identity / hflip / crop80 / jpeg60 / half | same-location excluded >= tau |
|---|---|---|---|---|---|---|
| 1e-2 | 0.659 | 15,233 | 17,064 | -1,831 | 1.000 / 0.940 / 0.981 / 0.378 / 0.901 | 2.8 % |
| 1e-3 | 0.743 | 597 | 1,706 | -1,109 | 1.000 / 0.863 / 0.941 / 0.159 / 0.768 | 0.1 % |
| 1e-4 | 0.830 | 10 | 171 | -161 | 1.000 / 0.718 / 0.795 / 0.033 / 0.509 | 0.0 % |

Reading (the instrument's bounds, not a certificate):

- **Near-copy overlap: none detectable above chance.** At every fpr the unassessable rows
  score BELOW the null's expectation (hits < fpr x queries), while every planted identical copy
  and 72-94 % of flipped copies are recalled. Whatever the queries' sources share with GEO-Bench,
  it is not at the rate of one near-copy per 10,000 images, and the hits that do occur (597 at
  1e-3: GeoPile 510, Million-AID 42, RSI-CB 24, VGoogle 20, AID 1, DIOR 0; nearest task
  m-bigearthnet 287, m-pv4ger 255, m-so2sat 33, m-eurosat 19; nearest split train 571, val 20,
  test 6) are consistent with the null rate. `hits.csv` lists them for eyeballing; none are
  excluded from training on this evidence. The bound this supports (added 2026-11-12): at most
  597 of 1,706,393 unlocatable rows (0.035 %) are near-copies of a GEO-Bench image at the 1e-3
  threshold (15,233 = 0.89 % at 1e-2, 10 at 1e-4), 26 of the 597 nearest to a val / test image.
  Quote these raw counts, not the negative excess, as the upper bound.
- **What this does NOT measure: the same place under another sensor or date.** Only 2.8 % of
  the 20,000 EXCLUDED rows (same place by coordinates) exceed tau at fpr 1e-2, 0.1 % at 1e-3:
  a geographic duplicate that is not a near-copy is invisible to CLS cosine at these thresholds.
  So the number above bounds the near-copy channel of section 5, not the geographic one; for
  the 1.7 M unassessable rows the geographic channel stays unmeasured, exactly as section 5
  states. Recall also falls to 0.16 / 0.03 at 1e-3 / 1e-4 for JPEG q60 re-encodes: a heavily
  recompressed copy is missed at the tight taus (crop and half-resolution copies are mostly kept).
- Outputs: `$STORE/geo_dedup/content/report_dinov3_vit7b16_sat/{report.md,report.json,hits.csv,
  query_max_sim.npz}`; controls `dinov3_vit7b16_sat_controls_v2.npz` (6,003 rows = 6 tasks x 200
  x 5 variants + 3 blanks).
