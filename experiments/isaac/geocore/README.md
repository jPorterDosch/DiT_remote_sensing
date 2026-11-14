# GeoCore-9B on the GEO-Bench harness

Spec: `docs/superpowers/specs/2026-11-02-geocore-eval-design.md`. Plan:
`docs/superpowers/plans/2026-11-02-geocore-eval.md`. Model code: `src/models/geocore/`.

Order of operations, each gated on the previous one's OK line:

1. `sbatch experiments/isaac/geocore/download.sbatch` (short, ~1 h): snapshot to `$STORE/GeoCore-9B`,
   `sha256sum -c SHA256SUMS`. Prints `GEOCORE_DOWNLOAD_OK`.
2. `bash experiments/isaac/geocore/dryrun.sh` (login, seconds): stub-python flag check of
   `extract.sbatch`. Prints `DRYRUN_OK`.
3. `sbatch experiments/isaac/geocore/smoke.sbatch` (campus-gpu-bigmem, 1 A40, ~1.5 h): null-text
   test, 24-image extraction, parity gate, FLUX byte-identity regression. Prints `GEOCORE_SMOKE_OK`.
4. Extraction. The campus-gpu QOS caps a job at 24 h (3 running / 6 submitted per user) and an
   A40 needs 7.2 s per image at ensemble 8 (13-40 h per train split), so on campus-gpu-bigmem the
   work runs as UNITs of `extract.sbatch` (`UNIT=train:<i>` with `TRAIN_SHARDS=<S>`, `UNIT=valtest`,
   `UNIT=stamp` on short), fed by `/lustre/isaac24/scratch/jdosch1/tmp/geocore_orchestrator.sh`
   (`RES=224 bash geocore_orchestrator.sh`; shard counts 3 3 2 3 1 3 at 224 px, per-shard seeds
   100+i; state in `tmp/geocore_orch_<res>.tsv`). The orchestrator also submits the stamp unit,
   the 8-block probe array and the select once a dataset's caches are complete. On ai-tenn (3-day
   wall) one job per dataset suffices: `sbatch --array=0-5 -p ai-tenn -q ai-tenn --time=3-00:00:00
   experiments/isaac/geocore/extract.sbatch` (UNIT=all, unsharded, seed 42). The ens-1 anchor is one
   job: `sbatch --array=0 --export=ALL,ANCHOR=1 experiments/isaac/geocore/extract.sbatch`. The
   256 px check: `RES=256 ONLY="0 3" bash geocore_orchestrator.sh`. Every stamp / all unit prints
   `GEOCORE_EXTRACT_OK <dataset> <res>`; shards are merged by `eval.features.load_flux_split`.
5. Block selection (val only), cells, report: plan Task 12 / spec §4.4.

Environment variables of `extract.sbatch`: `RES` (224|256), `ANCHOR` (ensemble 1, t 260 580),
`SMOKE` (24 images, ensemble 2, private `models/geocore_eval_smoke/`), `CKPT_DIR`, `VAE`,
`DATA`, `BLOCKS`, `UNIT`, `TRAIN_SHARDS`. Caches: `models/geocore_eval[_img256|_anchor]/<dataset>/<run dir>/
multistep_<split>_feats_oneshot_g0.0_k4-8-12-16-20-24-28-31.npz` + `_meta.json` (weights
`geocore-9b`, pinned digests, `discard_channels` stamped from the train cache).
