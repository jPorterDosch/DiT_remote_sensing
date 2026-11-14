# GeoCore-9B on the GEO-Bench evaluation harness — design

Date: 2026-11-02. Status: approved in conversation, awaiting written review.
Owner: J. Dosch. Worktree: `feat/geo-dedup`.

## 1. Goal and success criteria

Add one external comparator row, **GeoCore-9B (frozen)**, to Table A (six GEO-Bench
classification tasks, label-efficiency matrix) under exactly the harness every other row
uses: cached multi-timestep block features, StandardScaler + logistic-regression probe,
val-selected candidates, image-level bootstrap. GeoCore-9B (Do et al., arXiv 2608.01896,
NeurIPS 2026) is a 9.24B flow-matching DiT trained from scratch on Git-10M with a
DINOv3-Sat alignment loss; it is the closest prior work to our contribution 2 and must be
cited and compared.

Success:

1. `results/matrix/cells/geocore-9b/<dataset>/f*_s*.npz` complete for all six datasets at
   the phase-2 fractions and seeds, produced by the unchanged `eval.matrix` pipeline.
2. Every cache carries verifiable provenance (checkpoint and VAE sha256, vendored commit,
   conditioning constants) and cannot be confused with any FLUX cache.
3. A harness-validation anchor: GeoCore's own EuroSAT readout rebuilt from our cache lands
   within about one point of their reported 97.5 (256 px). If it does not, the row does not
   ship until the discrepancy is explained.
4. FLUX extraction is byte-identical before and after the shared-code changes.
5. Paper: Table A row, related-work paragraph, contribution-2 wording, supplement entries.

Stretch: segmentation (three GEO-Bench seg tasks at 512 px) through the same adapter.

Deadline context: paper due 2026-11-16; freeze moved later to accommodate this row.

## 2. Decisions taken (with the user, 2026-11-01/02)

| Decision | Choice |
|---|---|
| Grid | Our full grid: 9 timesteps t in {100,180,260,340,420,500,580,740,900}, ensemble 8, frozen weights |
| Blocks | k in {4, 8, 12, 16, 20, 24, 28, 31} (block-input convention, 32 blocks total) |
| Resolution | 224 px for the table (harness parity); 256 px check on m_eurosat and m_so2sat for the supplement |
| Conditioning | Text `""` through T5-XXL and CLIP-L (their probe convention); GSD/lat/lon all −999.0 (learned null) |
| VAE encode | As shipped: posterior mean, frozen BatchNorm; recorded in meta |
| Guidance | None (model has no guidance embedding); cache tag `g0.0`; non-zero refused |
| Readouts | (a) Table A raw protocol; (b) DiTF view with model-specific discard channels (supplement); (c) their-protocol anchor on EuroSAT (supplement) |
| Identity | `geocore_ckpt`, `geocore_vae` are required config fields hashed into `config_hash`; sha256 pins verified at load |
| Approach | Adapter in our model registry (approach A); vendored GeoCore model code; no standalone extractor |

## 3. GeoCore facts the design relies on

Source: `KAIST-VICLab/GeoCore-9B` at commit `17406c03a9bd090ff1570809bf6bd7f4a35e2359`
(2026-10-06), files `models/flux2.py`, `models/vae_flux2.py`, `models/text_encoder.py`,
`eval/frozen_probe.py`, `inference.py`, `huggingface/SHA256SUMS`, `huggingface/config.json`.

- `Flux2(GeoCore9BParams)`: in_channels 128, hidden 4096, 32 heads, 8 double + 24 single
  blocks, mlp_ratio 3.0, RoPE axes_dim [32, 48, 48], theta 2000, context_in_dim 4096
  (T5-XXL), y_in_dim 768 (CLIP pooled). No guidance embedding.
- Modulation is global, not per block: `double_stream_modulation_img(vec)` yields two
  (shift, scale, gate) triples shared by all 8 double blocks; `single_stream_modulation(vec)`
  yields one triple shared by all 24 single blocks. Blocks apply
  `(1 + scale) * LayerNorm(x) + shift` and gate the residual.
- `vec = time_in(t_emb) + res_in(res) + lon_in(lon) + lat_in(lat) + y_in(y)`, where
  `timestep_embedding(t, 256)` multiplies by 1000 internally, so t is passed as a fraction.
  Values of −999.0 for res / lon / lat select learned null embeddings.
- Forward order: `img = img_in(x.flatten(2).transpose(1, 2))`, `txt = txt_in(ctx)`, double
  blocks on (img, txt), then `cat([txt, img], dim=1)` through single blocks, text tokens
  stripped, `final_layer(img, vec)`.
- Position ids: image `[0, row, col]` per latent position; text `[seq_idx, 0, 0]` over 256
  tokens.
- VAE (`vae_flux2.py`): FLUX.2 autoencoder, z_channels 32, 8x spatial, then 2x2 patchify to
  128 channels (16x total). `encode` returns the posterior mean followed by a frozen
  `BatchNorm2d(affine=False)` using running statistics. 224 px input gives 14x14 tokens;
  256 px gives 16x16.
- Their frozen probe: 256 px bicubic, pixels in [−1, 1], `z_t = (1 − tau) z0 + tau eps`,
  tau = 0.25, one eps per image, empty prompt through both encoders, all geo fields null,
  block outputs of double 3, double 7, single 7, single 15 (their "g4/g8/g16/g24"),
  text tokens stripped for single blocks, mean-pooled, concatenated (16384-d), linear head
  with Adam lr 1e-3 for 40 epochs, EuroSAT random 80/20 split, best tau reported.
- Weights: HF `JeonghyeokDo/GeoCore-9B`, four safetensors shards (18.5 GB bf16, final
  non-EMA), `vae/diffusion_pytorch_model.safetensors`, `SHA256SUMS`.
- Requirements are unpinned (`torch>=2.4`, `transformers`, `einops`, `safetensors`, ...).
  The DiTF environment already has torch 2.8.0, transformers 5.5.4, einops 0.8.2,
  safetensors 0.7.0, sentencepiece; T5-XXL and CLIP-L weights are local under `$STORE`.
  peft and deepspeed are training-only and not needed.

## 4. Components

### 4.1 Vendored model code — `src/models/geocore/`

Copied verbatim from the pinned commit: `flux2.py`, `vae_flux2.py`, `text_encoder.py`,
`LICENSE` (Apache-2.0), `NOTICE`. Only import paths change. `VENDORED.md` records the
commit sha, date, file sha1s and the import-path diff. Nothing else in these files is
edited; feature tapping lives in the adapter.

### 4.2 Adapter — `src/models/geocore/adapter.py`

`@register_model("geocore") class GeoCoreModel`, mirroring the `FluxModel` contract that
`tasks/utils.extract_features` and `tasks/extraction.ExtractionTask` already consume:

- `__init__(cfg, category_list)`: builds `Flux2(GeoCore9BParams())` in bf16, loads the
  sharded state dict with `strict=True` after verifying every shard against the pinned
  sha256; loads the VAE the same way; encodes `""` once through T5-XXL and CLIP-L, keeps
  `(ctx, ctx_ids, y)` and frees both encoders; precomputes nothing else.
- Attributes: `hidden_size = 4096`, `num_double_blocks = 8`, `num_blocks = 32`,
  `weights_tag = "geocore-9b"`, `vae_encode = "mean"`, `text = ""`, `geo_null = -999.0`.
- `extract_raw(img, timestep, block_idx, ensemble_size, guidance, latents, noise,
  generator) -> (feat_raw, ada, latents_used, noise_used)`, same shapes as FLUX:
  `feat_raw (B, C, h, w)` for a block list (or `(1, C, h, w)` for an int), `ada (B, 3, C)`
  (or `(1, 3, C)`), latents/noise `(ensemble, 128, h, w)`. Per ensemble member: encode
  (or reuse the passed latents), draw eps from `generator` (or reuse), mix
  `z_t = (1 − t/1000) z0 + (t/1000) eps`, run `_forward_feat`, mean over ensemble members
  exactly where FLUX averages.
- `_forward_feat(z_t, t, ft_indices)`: reimplements the trunk loop over the vendored
  blocks with early exit after `max(ft_indices)`. Taps the image tokens **entering** block
  k (for k >= 8 the `[txt, img]` concatenation is sliced from `num_txt_tokens`), returns
  `[feat, mod]` pairs in ascending k. `mod` is the first (shift, scale, gate) triple of
  `double_stream_modulation_img(vec)` for k < 8 and `single_stream_modulation(vec)` for
  k >= 8. Calls the vendored `block.forward_kv_extract(...)` with `num_ref_tokens=0` for the
  actual block computation, so block arithmetic is theirs.
- `extract_inversion`, `roundtrip`, `extract` (single-timestep classification path): raise
  `NotImplementedError("geocore: one-shot multi-timestep extraction only")`.
- Token path for segmentation (stretch): `extract_raw` already returns the `(C, h, w)`
  grid; `tasks/seg_tokens.py` consumes it unchanged apart from the width and block-count
  attributes.

### 4.3 Shared-code changes (minimal, model-parameterised)

- `src/config_types.py`: `double_block_stamp(k, num_double=NUM_DOUBLE_BLOCKS)`; FLUX
  callers keep the default. `ExtractionTask` passes `model.num_double_blocks`.
- `src/tasks/extraction.py`: provenance meta reads `weights` from
  `getattr(model, "weights_tag", "flux-dev")`; for non-FLUX models `env_provenance` is
  called with `honors=()` so any control env var raises; refuses `extraction_mode !=
  ONESHOT`, `lora_checkpoint`, and `guidance_scale != 0.0` for geocore; writes extra meta:
  `hidden_size`, `num_double_blocks`, `num_blocks`, `vae_encode`, `text`, `geo_null`,
  `img_tokens`, `geocore_commit`, `geocore_ckpt_sha256`, `geocore_vae_sha256`,
  `discard_channels` (see 4.5).
- `src/tasks/utils.extract_features`: no change in logic; the `model._inner.model.eval()`
  line becomes `model.eval()` on the adapter (FLUX adapter gains a one-line `eval()`).
- `run.py`: `ModelConfig.name` accepts `"geocore"`; new `RunConfig` fields `geocore_ckpt:
  str` and `geocore_vae: str` with no default (required when name is geocore, must be
  unset otherwise); both are part of `config_hash`.
- `eval/features.py`, `eval/probe.py`: `check_double_block_stamp` and the DiTF view read
  `num_double_blocks` and `discard_channels` from the cache meta when `weights` is not
  `flux-dev`, and refuse when absent. FLUX caches keep the hard-coded [154, 1446]. These two
  files carry older unreviewed edits; the GeoCore edits stay unstaged until reviewed.

### 4.4 Cache identity

Run dir `models/geocore_eval/m_<dataset>_geocore_<hash>+42[+val|+test]/` (one per split, as the FLUX multi-block caches), cache
`multistep_<split>_feats_oneshot_g0.0_k4-8-12-16-20-24-28-31.npz` plus `_meta.json`, the
same layout as the FLUX multi-block caches: `feats (N, 9, 8, 4096)`, `mods (9, 8, 3, 4096)`,
`subset_indices`, `labels`, `eps_seed`, `ensemble_size`. No FLUX cache shares a path.

Matrix spec `eval/matrix_specs/phase8_geocore.json`: arm `geocore-9b`, kind `flux` (same
npz layout), `expect` pins `extraction_mode=ONESHOT`, `ensemble_size=8`,
`weights=geocore-9b`, `hidden_size=4096`, and a `per_dataset` block pin `k` chosen the same
way the frozen FLUX arm's blocks were chosen (phase 1): a val-only sweep at 100 % labels over
the eight cached blocks and the candidate views, best val accuracy per dataset, test never
read (rule 1). The sweep output and the chosen k per dataset are recorded in the spec's
`_comment`. Arm `geocore-9b-256` for the two-dataset resolution check, same procedure.

### 4.5 Discard channels (DiTF view)

Rule, pre-registered: at the aligned block for the arm, over the first 1,000 train images
(subset order, all timesteps), a channel is "massive" when its mean absolute activation
exceeds 50x the median of the per-channel means. The rule is computed by
`eval/discard_channels.py` from the cache and written into meta as `discard_channels`.
Validation: the same rule on an existing FLUX k=28 cache must return exactly [154, 1446];
if it does not, the threshold is wrong and the DiTF view is not reported for GeoCore.

### 4.6 Their-protocol anchor

`eval/geocore_anchor.py`: from the m_eurosat cache (224 and 256 runs), take k in {4, 8, 16,
24} (block inputs = their block outputs), t = 260 (nearest grid point to their 0.25; stated
in the output), ensemble member 0 only (requires the per-member cache below), concatenate,
train `nn.Linear(16384, 10)` with Adam lr 1e-3, 40 epochs, batch 1024, on an 80/20 split
drawn with `np.random.default_rng(0)` over the union of our official splits. Prints the
accuracy next to their 97.5. Population caveat, stated in the output: GEO-Bench m_eurosat is
a 20,200-image subset of the 27,000-image EuroSAT they split; the anchor tests the readout,
not the dataset. Their protocol draws one eps per image, so the anchor reads a separate small
extraction of m_eurosat with `--model.ensemble-size 1 --t 260 580` (same blocks, seeds and
resolution; `extract.sbatch ANCHOR=1`, stored under `models/geocore_eval_anchor/`), not the
ensemble-8 cache: an ensemble average is not their feature. (Amended 2026-11-02 from an earlier
per-member side-file design.)

## 5. Data flow and jobs

1. `experiments/isaac/geocore/download.sbatch` (`-p short -q short`, 2 h):
   `huggingface_hub.snapshot_download("JeonghyeokDo/GeoCore-9B", local_dir=$STORE/GeoCore-9B)`,
   then `sha256sum -c SHA256SUMS`. Fails the job on any mismatch.
2. `experiments/isaac/geocore/smoke.sbatch` (campus-gpu-bigmem, 1 A40, 1 h):
   24 strided images of m_eurosat, t in {260, 580}, all 8 blocks, ensemble 2; asserts
   shapes, finite values, eps constancy, and the parity gate: the k=8 feature of member 0
   equals (allclose, bf16 tolerance) the double-block-7 output captured by wrapping
   `double_blocks[7].forward_kv_extract` (a plain method, so `nn.Module` forward hooks do not
   fire) during a vendored `Flux2.forward` run on the same z_t. Also runs the FLUX regression smoke
   (4.3): 20 images of an existing `flux-proj` m_eurosat cache re-extracted and compared
   byte-for-byte. Prints `GEOCORE_SMOKE_OK`.
3. `experiments/isaac/geocore/extract.sbatch` (array 0-5 over datasets; `RES=224|256`):
   train, val and test splits, 100 % labels, grid from section 2; writes caches and the
   EuroSAT anchor file. Target campus-gpu-bigmem first (one A40 per task, 48 GB; model
   18.5 GB + activations), ai-tenn from 2026-11-08 for tasks still pending. Prints
   `GEOCORE_EXTRACT_OK <dataset>`.
4a. Block selection: val-only sweep (100 % labels, eight blocks x candidate views) per
   dataset, as in section 4.4; writes `results/eval/sweep_mb/geocore_*` and the k pins.
4b. Cells: existing `experiments/isaac/eval/matrix.sbatch` with `phase8_geocore.json`, low
   fractions on short, 100 % on campus-gpu-bigmem (the long partition queues for days).
5. Report: `python -m eval.matrix report --spec eval/matrix_specs/phase8_geocore.json` into
   `results/matrix/phase8/`.

Estimates (label them estimates): per large dataset (so2sat, bigearthnet, ~22k images)
about 13 h on H100 or 35 h on A40, scaled from the semt extractions (12B FLUX, 512 text
tokens, same grid: 21.6 h H100). Six datasets about 60-80 H100-hours. Disk about 150 GB for
224 plus 40 GB for the 256 check. Scratch has about 1.4 TB free after the 2026-11-02
cache deletions.

## 6. Fail-fast guards (all raise before any GPU work where possible)

- Checkpoint or VAE sha256 mismatch against the pinned constants; `strict=True` load.
- `FLUX_RANDOM_INIT`, `FIXED_COND_T` or `DEGRADE_TO` set (geocore honors none).
- `guidance_scale != 0.0`, `lora_checkpoint` set, `extraction_mode != ONESHOT`.
- `cfg.k` not within 0..31 or not strictly increasing.
- `img_size` not a multiple of 16; latent token count != (H/16)*(W/16).
- T5 context length != 256, CLIP pooled width != 768, dtype not bf16.
- `geocore_ckpt` / `geocore_vae` set while `model.name != "geocore"`, or unset while it is.
- Probe side: GeoCore cache without `num_double_blocks` or `discard_channels` in meta;
  `weights` pin mismatch; k-stamp mismatch.

## 7. Tests (`tests/test_geocore_adapter.py`, CPU, pytest)

- Tiny `Flux2` (hidden 64, 2 double + 3 single blocks, 2 heads) with seeded random
  weights; a reference pass records each block's input image tokens by wrapping the vendored
  blocks' `forward_kv_extract` methods; `_forward_feat` must return the same tensors for every k, with the
  text tokens stripped for single blocks.
- Mod triple equals the global modulation output for the block family.
- Early exit: no block beyond `max(ft_indices)` runs (hook counter).
- Shape contract of `extract_raw` for int and list k; eps / latent reuse across timesteps.
- Guards in section 6 raise with the expected messages.
- Discard-channel rule reproduces [154, 1446] on a stored FLUX fixture (small slice).

## 8. Paper integration

- Table A: `GeoCore-9B (frozen)` row next to the DINOv3 rows; caption states 224 px, empty
  prompt, null geo metadata, our probe; their reported EuroSAT number and our anchor in the
  supplement.
- Related work: GeoCore as the closest prior (from-scratch 9B EO generator with a DINOv3-Sat
  alignment loss from block 8, mu = 0.5); contribution 2 reworded as adapting a pretrained
  natural-image generator with a small adapter versus training from scratch.
- Supplement: 256 px check on m_eurosat and m_so2sat; block x timestep curves; DiTF view
  row; anchor table.
- No FID comparison outside fMoW val (their training set contains our geo-held-out set).

## 9. Out of scope

Geo-conditioned extraction (true lat/lon/GSD), LoRA adaptation of GeoCore, inversion-chain
extraction, FID against GeoCore samples, GeoCore-4B.
