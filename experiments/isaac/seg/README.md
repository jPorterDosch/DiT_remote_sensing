# Segmentation harness (GEO-Bench, linear per-token probe)

Three GEO-Bench segmentation tasks -- m-pv4ger-seg (binary, 320 px aerial), m-chesapeake-landcover
(6 land-cover classes + ignored no-data, 256 px NAIP) and m-SA-crop-type (9 crop classes + ignored
no-data, 256 px Sentinel-2 RGB; the hard one) -- probed the way the classification matrix probes:
frozen features, a linear head, selection on the official val split only, label fractions of the
train images, image bootstrap. Metric mIoU over the registered classes (eval/features.OFFICIAL_SEG).

Pipeline (each step gated, smoke first):

1. `export.sbatch --export=ALL,COUNT=1 --array=0-2` prints what the SHIPPED artifact says
   (partition sizes, class names, bands, ranges). Fill / verify OFFICIAL_SEG and SEG_SPECS from
   that output (rule 16; m-pv4ger-seg: 3221 / 403 / 403 train/val/test, not the paper's 3000).
   Then `export.sbatch --export=ALL,WRITE_MANIFEST=1 --array=0-2` writes `data/<key>/<split>/
   {images,masks}/*.png` and `eval/manifests/<key>.json` (commit it). Every later job re-verifies
   the tree against the manifest before reading it.
2. Tokens, 512 px (FLUX: 64 x 64 latent -> 32 x 32 tokens of 3072; DINOv3/16: 32 x 32 of 4096), so
   one token covers 8 px of a 256 px mask. `extract_flux.sbatch` (run.py --task extract_seg_tokens,
   one float16 (N, L, C) memmap per (t, k) and split; frozen: t 100 / 260 / 420 x k 12 / 28 / 38,
   ensemble 2, eps seeds 42 / 43 / 44 per split, guidance 1.0, DiTF applied later per token with
   the saved adaLN mods) and `extract_dino.sbatch` (last-block normed patch tokens). `SMOKE=1` on
   both first (24 images, models/seg_tokens_smoke/).
3. `probe.sbatch --array=0-2` (eval/seg_probe.py run): per cell (model, dataset, fraction, seed)
   every (t, k) file x lr {1e-3, 1e-2, 1e-1}: channel standardization fitted on the cell's
   probe-training tokens, linear head, logits bilinearly upsampled to the mask, cross-entropy with
   ignore, Adam, 2000 steps of 16 images for EVERY fraction (matched budget). Selection by val
   mIoU; the selected candidate's test confusion is the only one opened. Cells under
   results/seg/cells/ carry the per-image test confusions (rule 12).
4. `python -m eval.seg_probe report --spec eval/seg_specs/phase1.json` -> results/seg/summary.md.

Adapted arms (phase 2): `extract_flux.sbatch` with `CKPT=<run>/checkpoints/lora_last_*.pt`
(blocks 28 only, same t grid), then a flux entry per arm in a phase2 spec with
`"tag": "oneshot_g1.0_lora-<arm>_ens2_img512"` and `"expect": {"weights": "flux-dev+lora:<arm>"}`.

Disk: a frozen dataset is ~280 GB of tokens (9 files x 3 splits), an arm ~95 GB, DINO ~40 GB. Run
one dataset at a time; once its results/seg cells exist and are reported, delete
`models/seg_tokens/<dataset>/` (the cells hold everything the tables need; re-extraction costs GPU
hours, not correctness: the caches are deterministic given the seeds).
