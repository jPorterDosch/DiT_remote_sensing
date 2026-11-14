# GeoCore-9B Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a frozen GeoCore-9B row through the unchanged GEO-Bench label-efficiency harness (extraction cache, matrix probe, report), with provenance that cannot be confused with any FLUX cache.

**Architecture:** A new `@register_model("geocore")` adapter wraps vendored GeoCore model code behind the same `extract_raw` contract the FLUX adapter exposes, so `tasks/extraction.py`, `tasks/utils.extract_features` and the matrix never see the model class. Three FLUX-specific constants (hidden width, double-block count, DiTF discard channels) become model attributes carried in cache meta. Jobs follow the existing sbatch templates.

**Tech Stack:** PyTorch 2.8.0 (bf16), transformers 5.5.4, safetensors 0.7.0, einops 0.8.2, huggingface_hub 1.12.0, numpy, scikit-learn, pytest 9.1.1; SLURM on ISAAC (campus-gpu-bigmem A40, ai-tenn H100, short).

**Spec:** `docs/superpowers/specs/2026-11-02-geocore-eval-design.md`

## Global Constraints

- Worktree `/nfs/home/jdosch1/COSC424/DiT_remote_sensing-geodedup`, branch `feat/geo-dedup`. Stage changes; the user commits (never commit or push). Do not stage `eval/features.py`, `eval/probe.py`, `eval/matrix.py`, `src/tasks/extraction.py` — they hold older unreviewed edits; leave the GeoCore hunks unstaged and list them in the handoff.
- Never run compute on a login node; seconds-long checks only via `timeout 60 $HOME/.conda/envs/DiTF/bin/python`. pytest runs as a `-p short -q short -A isaac-utk0256` job (template: `/lustre/isaac24/scratch/jdosch1/tmp/pytest_align.sh`). GPU smokes on `-p campus-gpu-bigmem -q campus-gpu`, `-n 1 --gpus-per-task=1`.
- Weights: `$STORE/GeoCore-9B/` (`$STORE` = `/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing`; the shell does not export it, use the absolute path). Text encoders: `$STORE/t5-v1_1-xxl`, `$STORE/clip-vit-large-patch14`. Never overwrite an existing cache, manifest or index.
- Pinned constants (spec §3): commit `17406c03a9bd090ff1570809bf6bd7f4a35e2359`; shard sha256s (from `huggingface/SHA256SUMS`, copied verbatim into `adapter.py`): `model-00001-of-00004.safetensors` 37867b56a02748c67fe64f60118b6b6cd6966653026ba101abd1bf849e4f4b37, `model-00002-of-00004.safetensors` 561831477a146da51524beacbd28e4e4db9c3d66318c98388823e641e920a956, `model-00003-of-00004.safetensors` 60107ab28114906426b4187bbca9fda59af121318cb39543a90697b7a7ecef9a, `model-00004-of-00004.safetensors` 7cef9bd84c652cb25d0868d0f06d3cc4279d8a066db219f9c5f3fb4c7ac7288a, `model.safetensors.index.json` 6af0afb014d208f28e04804cb0bbe8bdf3b52a8490cf3610e001b85c17fa8b65; VAE sha256 `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04`; `PINNED_TEXT = ""`; `PINNED_GEO_NULL = -999.0`; `PINNED_VAE_ENCODE = "mean"`; weights tag `geocore-9b`; hidden 4096; 8 double + 24 single blocks.
- Grid: t = 100 180 260 340 420 500 580 740 900; k = 4 8 12 16 20 24 28 31; ensemble 8; img 224 (and 256 for the m_eurosat / m_so2sat check); guidance 0.0; split seeds train 42 / test 43 / val 44 (as `extract_ckpt.sbatch`).
- Fail fast: every guard in spec §6 raises before GPU work where possible; no silent fallbacks.
- Normal prose in files, commit messages and docs.

## Review Focus

1. A 224 px image gives a 14x14 latent grid; `x_ids` must be built from the latent height/width after the VAE's 2x2 patchify, not from 256/16. Test in Task 5 (`test_extract_raw_token_grid_224`).
2. The global modulation triples are identical for every double block and for every single block; the DiTF view must still receive one `(3, C)` per cached block so `features.ditf` indexes `m[:, 1, :]` unchanged. Test in Task 4 (`test_mods_per_block_shape_and_sharing`).
3. The `""` prompt must tokenise to `[1] + [0] * 255` for T5 (pad id 0, eos id 1) with `T5TokenizerFast` built from the local `spiece.model`; a conversion failure would change `ctx` silently. Test in Task 2.
4. A FLUX cache must still pass every gate after the shared-code edits: `check_double_block_stamp` on a FLUX meta without the new keys must behave exactly as before. Test in Task 8 (`test_flux_meta_without_new_keys_unchanged`).
5. The multi-block early exit must not run blocks beyond `max(k)`; with k = 31 the final layer must never run (no `final_layer` call). Test in Task 4 (`test_early_exit_counts`).

---

### Task 1: Vendor the GeoCore model code

**Files:**
- Create: `src/models/geocore/__init__.py`, `src/models/geocore/flux2.py`, `src/models/geocore/vae_flux2.py`, `src/models/geocore/LICENSE`, `src/models/geocore/NOTICE`, `src/models/geocore/VENDORED.md`
- Modify: `src/models/__init__.py` (one line: `from . import geocore  # noqa: F401`)
- Test: `tests/test_geocore_vendored.py`

**Interfaces:**
- Produces: `models.geocore.flux2.{Flux2, GeoCore9BParams, GeoCore4BParams, Modulation, DoubleStreamBlock, SingleStreamBlock, timestep_embedding}`, `models.geocore.vae_flux2.{AutoEncoder, AutoEncoderParams, load_autoencoder, load_vae_state_dict}`; `GEOCORE_COMMIT = "17406c03a9bd090ff1570809bf6bd7f4a35e2359"` in `src/models/geocore/__init__.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geocore_vendored.py
import hashlib, os
from models.geocore import GEOCORE_COMMIT
from models.geocore import flux2, vae_flux2

def test_commit_pin():
    assert GEOCORE_COMMIT == "17406c03a9bd090ff1570809bf6bd7f4a35e2359"

def test_vendored_file_sha1s_match_vendored_md():
    root = os.path.dirname(flux2.__file__)
    md = open(os.path.join(root, "VENDORED.md")).read()
    for name in ("flux2.py", "vae_flux2.py"):
        sha = hashlib.sha1(open(os.path.join(root, name), "rb").read()).hexdigest()
        assert sha in md, f"{name} sha1 {sha} not recorded in VENDORED.md"

def test_params_match_hf_config():
    p = flux2.GeoCore9BParams()
    assert (p.in_channels, p.hidden_size, p.num_heads, p.depth, p.depth_single_blocks) == (128, 4096, 32, 8, 24)
    assert p.axes_dim == [32, 48, 48] and p.theta == 2000 and p.mlp_ratio == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run (short job): `pytest tests/test_geocore_vendored.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.geocore'`

- [ ] **Step 3: Vendor the files**

Download `models/flux2.py`, `models/vae_flux2.py`, `LICENSE`, `NOTICE` from `https://raw.githubusercontent.com/KAIST-VICLab/GeoCore-9B/17406c03a9bd090ff1570809bf6bd7f4a35e2359/...` into `src/models/geocore/` (a verified copy already sits in the session scratchpad `geocore_src/`). Do not edit the two `.py` files (their imports are absolute third-party only: torch, einops, safetensors; nothing to rewrite). `text_encoder.py` is NOT vendored (hard-coded HF ids; Task 2 replaces it). Write `VENDORED.md`: source URL, commit, date 2026-10-06, sha1 of each vendored file, the statement that no line was changed, and why `text_encoder.py` was not taken. `__init__.py` defines `GEOCORE_COMMIT` and imports `adapter` lazily at the end (`from . import adapter  # noqa: E402,F401`, added in Task 5; until then leave a comment).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_geocore_vendored.py -v` — Expected: 3 PASS

- [ ] **Step 5: Stage**

```bash
git add src/models/geocore tests/test_geocore_vendored.py src/models/__init__.py
```

---

### Task 2: Null text embeddings from local encoders

**Files:**
- Create: `src/models/geocore/null_text.py`
- Test: `tests/test_geocore_null_text.py`

**Interfaces:**
- Produces: `null_text_embeddings(t5_dir: str, clip_dir: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> tuple[Tensor, Tensor, Tensor]` returning `ctx (1, 256, 4096)`, `ctx_ids (1, 256, 3)` with `ctx_ids[0, :, 0] = arange(256)`, `y (1, 768)`; loads `T5TokenizerFast` + `T5EncoderModel` and `CLIPTokenizer` + `CLIPTextModel` from the two local directories, encodes `""` exactly as GeoCore's `TextEncoder.__call__` does (`padding="max_length"`, `max_length=256` / 77, `truncation=True`, `input_ids` only, T5 output `[0]`, CLIP `pooler_output`), then deletes both encoders and empties the CUDA cache. Also `null_token_ids(t5_dir, clip_dir) -> tuple[list[int], list[int]]` (tokeniser only, for the test).
- Constants: `T5_DIR_DEFAULT = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/t5-v1_1-xxl"`, `CLIP_DIR_DEFAULT = ".../clip-vit-large-patch14"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geocore_null_text.py
import os, pytest
from models.geocore import null_text as NT

needs_local = pytest.mark.skipif(not os.path.isdir(NT.T5_DIR_DEFAULT), reason="local encoders absent")

@needs_local
def test_empty_prompt_token_ids():
    t5_ids, clip_ids = NT.null_token_ids(NT.T5_DIR_DEFAULT, NT.CLIP_DIR_DEFAULT)
    assert t5_ids == [1] + [0] * 255          # </s> then pad, as GeoCore's TextEncoder
    assert len(clip_ids) == 77 and clip_ids[0] == 49406 and clip_ids[1] == 49407  # BOS, EOS, then pad(=EOS)

@pytest.mark.gpu
def test_null_embeddings_shapes():
    import torch
    ctx, ctx_ids, y = NT.null_text_embeddings(NT.T5_DIR_DEFAULT, NT.CLIP_DIR_DEFAULT, torch.device("cuda"))
    assert ctx.shape == (1, 256, 4096) and y.shape == (1, 768) and ctx_ids.shape == (1, 256, 3)
    assert ctx.dtype == torch.bfloat16 and torch.equal(ctx_ids[0, :, 0].long(), torch.arange(256, device=ctx.device))
    assert torch.isfinite(ctx.float()).all() and torch.isfinite(y.float()).all()
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_geocore_null_text.py -v` → FAIL `No module named 'models.geocore.null_text'`
- [ ] **Step 3: Implement `null_text.py`** (signatures above; tokenisers from the local dirs, `torch_dtype=dtype`, `.to(device)`, `@torch.no_grad()`). The tokeniser-only path must not load encoder weights.
- [ ] **Step 4: Run tests** — CPU test PASS; GPU test deferred to the Task 11 smoke job (`pytest --gpu`).
- [ ] **Step 5: Stage** — `git add src/models/geocore/null_text.py tests/test_geocore_null_text.py`

---

### Task 3: Model-parameterised double-block stamp

**Files:**
- Modify: `src/config_types.py:23-27`
- Test: `tests/test_flux_blocks.py` (append)

**Interfaces:**
- Produces: `double_block_stamp(k, num_double: int = NUM_DOUBLE_BLOCKS) -> dict` (same return as today when `num_double` is omitted).

- [ ] **Step 1: Write the failing tests**

```python
def test_double_block_stamp_default_unchanged():
    from config_types import double_block_stamp, DOUBLE_BLOCK_FEATURE
    assert double_block_stamp(28) == {} and double_block_stamp([4, 28]) == {"double_block_feature": DOUBLE_BLOCK_FEATURE}

def test_double_block_stamp_num_double():
    from config_types import double_block_stamp, DOUBLE_BLOCK_FEATURE
    assert double_block_stamp([8, 12], num_double=8) == {}
    assert double_block_stamp([4, 8], num_double=8) == {"double_block_feature": DOUBLE_BLOCK_FEATURE}
```

- [ ] **Step 2: Run** → second test FAILS `TypeError: unexpected keyword argument 'num_double'`
- [ ] **Step 3: Implement** the keyword in `config_types.double_block_stamp`; comparison `b < num_double`.
- [ ] **Step 4: Run** `pytest tests/test_flux_blocks.py -v` → all PASS (existing tests unchanged).
- [ ] **Step 5: Stage** — `git add src/config_types.py tests/test_flux_blocks.py`

---

### Task 4: Trunk feature tap (`_forward_feat`) on the vendored blocks

**Files:**
- Create: `src/models/geocore/trunk.py`
- Test: `tests/test_geocore_trunk.py`

**Interfaces:**
- Produces: `forward_feat(model: Flux2, x: Tensor, x_ids: Tensor, timesteps: Tensor, ctx: Tensor, ctx_ids: Tensor, y: Tensor, res: Tensor, lon: Tensor, lat: Tensor, ft_indices: list[int]) -> list[Tensor]` returning `[feat_0, mod_0, feat_1, mod_1, ...]` in ascending `ft_indices`; `feat_j` = image tokens **entering** block k_j, shape `(B, L_img, C)`; `mod_j` = `(B, 3, C)` stacked `[shift, scale, gate]` from `double_stream_modulation_img(vec)[0]` for k < depth, else `single_stream_modulation(vec)[0]`. Early exit after `max(ft_indices)`; `final_layer` never called. Raises `ValueError` for empty, non-increasing or out-of-range (`0..depth+depth_single-1`) indices.
- Also `tiny_params(hidden=64, heads=2, depth=2, depth_single=3, mlp_ratio=2.0, context_in_dim=16, y_in_dim=8, z_out_dim=16, in_channels=8, axes_dim=[8, 12, 12], theta=2000) -> GeoCore9BParams` (test helper in `trunk.py`, used by Tasks 5 and 7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_geocore_trunk.py  (CPU, float32, seeded)
import torch, pytest
from models.geocore.flux2 import Flux2
from models.geocore.trunk import forward_feat, tiny_params

L_TXT, H_LAT, W_LAT = 5, 3, 4

def _model_and_inputs(seed=0):
    torch.manual_seed(seed); p = tiny_params(); m = Flux2(p).eval()
    B = 2
    x = torch.randn(B, p.in_channels, H_LAT, W_LAT)
    ids = torch.zeros(H_LAT, W_LAT, 3); ids[..., 1] = torch.arange(H_LAT)[:, None]; ids[..., 2] = torch.arange(W_LAT)[None, :]
    x_ids = ids.reshape(1, -1, 3).repeat(B, 1, 1)
    ctx = torch.randn(B, L_TXT, p.context_in_dim); ctx_ids = torch.zeros(B, L_TXT, 3); ctx_ids[:, :, 0] = torch.arange(L_TXT)
    y = torch.randn(B, p.y_in_dim); t = torch.full((B,), 0.25)
    null = torch.full((B,), -999.0)
    return m, dict(x=x, x_ids=x_ids, timesteps=t, ctx=ctx, ctx_ids=ctx_ids, y=y, res=null, lon=null, lat=null)

def _reference_inputs(m, kw):
    """Record the image tokens entering every block by wrapping forward_kv_extract."""
    seen = {}
    for i, blk in enumerate(m.double_blocks):
        orig = blk.forward_kv_extract
        def w(img, txt, *a, _i=i, _o=orig, **k): seen[_i] = img.clone(); return _o(img, txt, *a, **k)
        blk.forward_kv_extract = w
    for j, blk in enumerate(m.single_blocks):
        orig = blk.forward_kv_extract
        def w(xx, pe, mod, num_txt, *a, _j=j, _o=orig, **k): seen[len(m.double_blocks) + _j] = xx[:, num_txt:].clone(); return _o(xx, pe, mod, num_txt, *a, **k)
        blk.forward_kv_extract = w
    with torch.no_grad(): m.forward(**kw)
    return seen

def test_block_input_matches_reference_every_k():
    m, kw = _model_and_inputs(); ref = _reference_inputs(*_model_and_inputs())
    ks = list(range(5))
    with torch.no_grad(): out = forward_feat(m, ft_indices=ks, **kw)
    assert len(out) == 2 * len(ks)
    for j, k in enumerate(ks):
        assert torch.allclose(out[2 * j], ref[k], atol=1e-6), k

def test_mods_per_block_shape_and_sharing():
    m, kw = _model_and_inputs()
    with torch.no_grad(): out = forward_feat(m, ft_indices=[0, 1, 2, 4], **kw)
    mods = out[1::2]
    assert all(md.shape == (2, 3, 64) for md in mods)
    assert torch.equal(mods[0], mods[1]) and torch.equal(mods[2], mods[3]) and not torch.equal(mods[1], mods[2])

def test_early_exit_counts():
    m, kw = _model_and_inputs(); calls = {"single": 0, "final": 0}
    for blk in m.single_blocks:
        o = blk.forward_kv_extract
        blk.forward_kv_extract = (lambda *a, _o=o, **k: (calls.__setitem__("single", calls["single"] + 1), _o(*a, **k))[1])
    fo = m.final_layer.forward; m.final_layer.forward = lambda *a, **k: (calls.__setitem__("final", 1), fo(*a, **k))[1]
    with torch.no_grad(): forward_feat(m, ft_indices=[3], **kw)   # input of single block 1 -> runs single block 0 only
    assert calls == {"single": 1, "final": 0}
    with torch.no_grad(): forward_feat(m, ft_indices=[4], **kw)   # last block input: all but the last single block
    assert calls["final"] == 0

@pytest.mark.parametrize("bad", [[], [2, 1], [5], [-1]])
def test_bad_indices_raise(bad):
    m, kw = _model_and_inputs()
    with pytest.raises(ValueError): forward_feat(m, ft_indices=bad, **kw)
```

- [ ] **Step 2: Run** → FAIL `No module named 'models.geocore.trunk'`
- [ ] **Step 3: Implement `trunk.py`**. Mirror `Flux2.forward` lines (flatten, `make_vec`, the three modulations, `img_in`, `txt_in`, `pe_embedder` on both id sets) but loop with early exit: for double block i, if `i in ft_indices` append `(img.clone(), stack(mod_img[0]))` before calling `block.forward_kv_extract(img, txt, pe_x, pe_ctx, mod_img, mod_txt, num_ref_tokens=0)`; stop when `i == max_ft`. Then `img = cat([txt, img], 1)`, `pe = cat([pe_ctx, pe_x], 2)`; for single block j with global index `d + j`, if in indices append `(img[:, num_txt:].clone(), stack(single_mod))` before `block.forward_kv_extract(img, pe, single_mod, num_txt_tokens, num_ref_tokens=0)`; stop at `max_ft`. Stack mod as `torch.stack([shift, scale, gate], dim=1)` → `(B, 3, C)`.
- [ ] **Step 4: Run** `pytest tests/test_geocore_trunk.py -v` → PASS
- [ ] **Step 5: Stage** — `git add src/models/geocore/trunk.py tests/test_geocore_trunk.py`

---

### Task 5: `GeoCoreModel` adapter (`extract_raw`, loading, pins)

**Files:**
- Create: `src/models/geocore/adapter.py`
- Modify: `src/models/geocore/__init__.py` (import adapter at the end)
- Test: `tests/test_geocore_adapter.py`

**Interfaces:**
- Consumes: Task 4 `forward_feat`, `tiny_params`; Task 2 `null_text_embeddings`; vendored `Flux2`, `load_autoencoder`.
- Produces: `@register_model("geocore") class GeoCoreModel` with
  - `__init__(cfg, category_list)`: guards (spec §6) first: `cfg.lora_checkpoint` set → `ValueError`; `cfg.extraction_mode != ONESHOT` → `ValueError`; `cfg.guidance_scale != 0.0` → `ValueError`; `cfg.img_size` entries not multiples of 16 → `ValueError`; any k outside 0..31 → `ValueError`. Then `_verify_sha256(path, expected)` for every shard listed in `model.safetensors.index.json` under `cfg.geocore_ckpt` and for `cfg.geocore_vae` (`PINNED_SHARD_SHA256: dict[str, str]`, `PINNED_VAE_SHA256`); build `Flux2(GeoCore9BParams())` in bf16 on `cfg.device`, `load_state_dict(strict=True)` from the shards; `load_autoencoder(cfg.geocore_vae, device, dtype=torch.bfloat16)`; `self.ctx, self.ctx_ids, self.y = null_text_embeddings(T5_DIR_DEFAULT, CLIP_DIR_DEFAULT, device)`; `self.eval()`.
  - Class attributes: `hidden_size = 4096`, `num_double_blocks = 8`, `num_blocks = 32`, `weights_tag = "geocore-9b"`, `vae_encode = "mean"`, `text = ""`, `geo_null = -999.0`, `commit = GEOCORE_COMMIT`.
  - `provenance_meta() -> dict` with keys `hidden_size, num_double_blocks, num_blocks, vae_encode, text, geo_null, geocore_commit, geocore_ckpt_sha256 (index.json sha), geocore_vae_sha256`.
  - `extract_raw(img, timestep, block_idx, ensemble_size, caption="", category="image", guidance=0.0, latents=None, noise=None, generator=None) -> (feat_raw, ada, latents_used, noise_used)`: `img (3, H, W)` in [−1, 1] → `(1, 3, H, W)` bf16; per member: `z0 = vae.encode(img)` (or `latents[i:i+1]`), `eps` from `generator` (or `noise[i:i+1]`), `z_t = (1 − t/1000) z0 + (t/1000) eps`; build `x_ids` from `z_t.shape[-2:]`; `out = forward_feat(...)` with `timesteps = full((1,), t/1000)`, ctx/ctx_ids/y expanded to the member batch, `res = lon = lat = full((1,), -999.0)`; per block `rearrange(feat, "b (h w) c -> b c h w")`; stack over members, mean over members → `feat_raw (B, C, h, w)`, `ada (B, 3, C)` (int k → `(1, C, h, w)`, `(1, 3, C)`); return the concatenated `latents_used (E, 128, h, w)` and `noise_used`. `guidance` must be 0.0 (`ValueError` otherwise).
  - `eval()`, `extract(...)`, `extract_inversion(...)`, `roundtrip(...)` → `NotImplementedError`.
  - `GeoCoreModel.from_parts(flux2, vae, ctx, ctx_ids, y, device)` classmethod: builds an instance without loading anything (tests and smokes).

- [ ] **Step 1: Write the failing tests** (CPU; a stub VAE with `encode(x) -> (1, in_channels, H/16, W/16)` deterministic from x; tiny `Flux2` from Task 4; `ctx (1, 5, 16)`, `y (1, 8)`)

```python
def test_extract_raw_token_grid_224():   # Review Focus 1
    m = _adapter(); f, ada, lat, eps = m.extract_raw(torch.rand(3, 224, 224) * 2 - 1, timestep=260, block_idx=[0, 3], ensemble_size=2, generator=torch.Generator().manual_seed(0))
    assert f.shape == (2, 64, 14, 14) and ada.shape == (2, 3, 64) and lat.shape == (2, 8, 14, 14) and eps.shape == lat.shape

def test_extract_raw_int_k_shapes():
    f, ada, *_ = _adapter().extract_raw(torch.rand(3, 32, 32) * 2 - 1, timestep=100, block_idx=2, ensemble_size=1)
    assert f.shape == (1, 64, 2, 2) and ada.shape == (1, 3, 64)

def test_latent_and_noise_reuse_across_timesteps():
    m = _adapter(); img = torch.rand(3, 32, 32) * 2 - 1; g = torch.Generator().manual_seed(1)
    f1, _, lat, eps = m.extract_raw(img, 100, [1], 2, generator=g)
    f2, _, lat2, eps2 = m.extract_raw(img, 500, [1], 2, latents=lat, noise=eps)
    assert torch.equal(lat, lat2) and torch.equal(eps, eps2) and not torch.allclose(f1, f2)

def test_member_mean_equals_manual():
    # ensemble_size=2 result equals the mean of two ensemble_size=1 calls with the same latents/noise slices
    ...

@pytest.mark.parametrize("field,value", [("guidance_scale", 1.0), ("lora_checkpoint", "x.pt"), ("extraction_mode", "INVERSION"), ("img_size", [200, 200]), ("k", [4, 40])])
def test_init_guards(field, value):   # cfg stub; must raise BEFORE any file is opened (monkeypatch _verify_sha256 to fail the test if called)
    ...

def test_sha256_mismatch_raises(tmp_path): ...
def test_provenance_meta_keys(): ...
```

- [ ] **Step 2: Run** → FAIL (`No module named 'models.geocore.adapter'`)
- [ ] **Step 3: Implement `adapter.py`** per the interface. Verification reads files in 1 MiB chunks; message names the file and both digests.
- [ ] **Step 4: Run** `pytest tests/test_geocore_adapter.py -v` → PASS; `pytest tests/test_flux_blocks.py tests/test_geocore_*.py` → PASS
- [ ] **Step 5: Stage** — `git add src/models/geocore tests/test_geocore_adapter.py`

---

### Task 6: `run.py` config fields and honors

**Files:**
- Modify: `run.py:43` (`ModelConfig` docstring: names `flux | geocore`), `run.py:120-128` (new fields next to `k`), `run.py:316` (`__post_init__` checks), `run.py:578` (`_task_honors`)
- Test: `tests/test_geocore_config.py`

**Interfaces:**
- Produces: `RunConfig.geocore_ckpt: str = ""`, `RunConfig.geocore_vae: str = ""` (hashed: NOT in the config-hash blacklist); `__post_init__` raises `ValueError` when `model.name == "geocore"` and either is empty, or when `model.name != "geocore"` and either is set; `_task_honors(task, model_name="flux")` returns `()` for geocore; `main()` passes `cfg.model.name`.

- [ ] **Step 1: Write the failing tests**

```python
from run import RunConfig, ModelConfig, _task_honors
def test_geocore_requires_paths():
    with pytest.raises(ValueError): RunConfig(model=ModelConfig(name="geocore"), task="extract", t=[100, 180])
def test_flux_rejects_geocore_paths():
    with pytest.raises(ValueError): RunConfig(geocore_ckpt="/x", task="extract", t=[100, 180])
def test_paths_change_hash():
    a = RunConfig(model=ModelConfig(name="geocore"), geocore_ckpt="/a", geocore_vae="/v", task="extract", t=[100, 180], guidance_scale=0.0)
    b = RunConfig(model=ModelConfig(name="geocore"), geocore_ckpt="/b", geocore_vae="/v", task="extract", t=[100, 180], guidance_scale=0.0)
    assert a.config_hash() != b.config_hash()
def test_flux_hash_unchanged():   # value copied from an existing cache dir name: m_eurosat_flux_9c2b5c5a+42 (multiblock_ens8 train)
    ...  # rebuild that exact config and assert config_hash() == "9c2b5c5a"
def test_honors():
    assert _task_honors("extract", "geocore") == () and _task_honors("extract") == ("FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO")
```

- [ ] **Step 2: Run** → FAIL (`unexpected keyword argument 'geocore_ckpt'`)
- [ ] **Step 3: Implement.** Adding two defaulted fields does not change existing hashes only if `config_hash` drops empty-string fields or blacklists them when empty: implement as "blacklist when `model.name != 'geocore'`" so FLUX hashes are byte-identical (test 4 pins that). Document in the field comment (rule 9).
- [ ] **Step 4: Run** → PASS, including the pinned FLUX hash.
- [ ] **Step 5: Stage** — `git add run.py tests/test_geocore_config.py`

---

### Task 7: Model-aware provenance in the extraction task

**Files:**
- Modify: `src/tasks/extraction.py:227-240` (weights from model), `:318-345` (stamp + meta), `src/tasks/utils.py` (`model.eval()` at the `_inner.model.eval()` line), `src/tasks/seg_tokens.py:99` (same), `src/models/flux/adapter.py` (add `eval()`, `weights_tag = "flux-dev"`, `num_double_blocks = 19`, `hidden_size = 3072`, `provenance_meta() -> {}`)
- Test: `tests/test_geocore_extraction_meta.py`

**Interfaces:**
- Produces: `model_provenance(model, cfg) -> tuple[str, dict]` in `src/tasks/extraction.py`: `(suffix, meta)` where for FLUX it returns exactly today's `env_provenance` + LoRA result (byte-identical tag and meta), and for a model with `weights_tag != "flux-dev"` it calls `env_provenance(cfg, honors=())`, sets `meta["weights"] = model.weights_tag`, merges `model.provenance_meta()`, and raises on `cfg.lora_checkpoint`. `ExtractionTask.run` uses it, stamps `double_block_stamp(cfg.k, num_double=model.num_double_blocks)`, and writes the merged meta into `_meta.json` (plus `discard_channels` when present on the model, else omitted).

- [ ] **Step 1: Write the failing tests** (fake model objects; `cfg` from `RunConfig`)

```python
def test_flux_provenance_unchanged(monkeypatch):   # suffix "" and meta weights "flux-dev" for a frozen FLUX cfg; "+lora" path untouched (covered by existing tests)
def test_geocore_provenance_meta():   # weights "geocore-9b", hidden_size 4096, num_double_blocks 8, commit, shas present; suffix ""
def test_geocore_rejects_control_env(monkeypatch):  # FLUX_RANDOM_INIT=1 -> ValueError mentioning honors
def test_geocore_rejects_lora():   # cfg.lora_checkpoint set -> ValueError
def test_stamp_uses_model_double_count():  # k=[4, 8] with num_double_blocks 8 -> stamp present; k=[8, 12] -> absent
```

- [ ] **Step 2: Run** → FAIL (`ImportError: cannot import name 'model_provenance'`)
- [ ] **Step 3: Implement**, keeping the FLUX path's existing lines intact inside the helper (move, do not rewrite).
- [ ] **Step 4: Run** new tests + `tests/test_multistep_extraction.py` (untracked but present) → PASS
- [ ] **Step 5: Stage** `src/tasks/utils.py src/tasks/seg_tokens.py src/models/flux/adapter.py tests/test_geocore_extraction_meta.py` only. Leave `src/tasks/extraction.py` unstaged (unreviewed older hunks); record the hunk in the handoff.

---

### Task 8: Probe side reads width/depth/discard from meta

**Files:**
- Modify: `eval/features.py:161-170` (`ditf(f, m, discard=DISCARD)`), `:316-320` (`check_double_block_stamp(meta, k, src)` uses `meta.get("num_double_blocks", NUM_DOUBLE_BLOCKS)` only when `meta.get("weights", "flux-dev")` does not start with `"flux-dev"`, else today's behaviour; for non-FLUX meta both `num_double_blocks` and `discard_channels` must be present or `SystemExit`), `eval/probe.py:184-190` (pass `discard=m_tr["discard_channels"]` for non-FLUX meta)
- Test: `tests/test_eval_gates.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def test_flux_meta_without_new_keys_unchanged():   # Review Focus 4: meta {"weights": "flux-dev", "double_block_feature": ...}, k=4 passes; k=28 passes without stamp
def test_geocore_meta_requires_keys():   # weights geocore-9b, k=4, no num_double_blocks -> SystemExit
def test_geocore_stamp_uses_meta_count():   # num_double_blocks 8: k=8 needs no stamp, k=4 does
def test_ditf_discard_param():   # ditf(f, m, discard=[0]) zeroes channel 0 only; default still [154, 1446]
```

- [ ] **Step 2: Run** → FAIL on the two new signatures
- [ ] **Step 3: Implement** (3 small edits).
- [ ] **Step 4: Run** `pytest tests/test_eval_gates.py tests/test_matrix.py -v` → PASS
- [ ] **Step 5: Stage** `tests/test_eval_gates.py` only; `eval/features.py` and `eval/probe.py` stay unstaged (older unreviewed hunks). List both hunks in the handoff.

---

### Task 9: Discard-channel rule

**Files:**
- Create: `eval/discard_channels.py`
- Test: `tests/test_discard_channels.py`

**Interfaces:**
- Produces: `massive_channels(feats: np.ndarray, factor: float = 50.0, n_images: int = 1000) -> list[int]` for `feats (N, K, C)`: `a = |feats[:n_images]|.mean(axis=(0, 1))`; return sorted channels with `a > factor * median(a)`. CLI: `python -m eval.discard_channels --features <cache.npz> --k <block>` (multi-block caches sliced via `features._block_slice`), prints the list and `DISCARD_OK` when a `--expect 154 1446` list matches. Writing into meta is done by the extraction job (Task 11) through `GeoCoreModel.discard_channels` computed at the end of the train-split extraction from the in-memory `feats` at the aligned block (k = 8, their GSA block), and verified identical on val and test extractions (mismatch raises).

- [ ] **Step 1: Write the failing tests**

```python
def test_rule_finds_planted_channels():
    rng = np.random.default_rng(0); f = rng.normal(size=(50, 3, 32)); f[:, :, [5, 17]] *= 200
    assert massive_channels(f, n_images=50) == [5, 17]
def test_rule_empty_when_flat():
    assert massive_channels(np.random.default_rng(1).normal(size=(50, 3, 32))) == []
```

- [ ] **Step 2: Run** → FAIL; **Step 3: Implement**; **Step 4: Run** → PASS
- [ ] **Step 5: Cluster gate** (short job, CPU): `python -m eval.discard_channels --features models/m_eurosat_multiblock_ens8/m_eurosat_flux_9c2b5c5a+42/multistep_train_feats_oneshot_g1.0_k4-8-12-18-19-24-28-33-38-43-48-54-55-56.npz --k 28 --expect 154 1446` → `DISCARD_OK`. If not OK, stop: record the output in the handoff and do not report a DiTF row for GeoCore (spec §4.5).
- [ ] **Step 6: Stage** — `git add eval/discard_channels.py tests/test_discard_channels.py`

---

### Task 10: Their-protocol anchor script

**Files:**
- Create: `eval/geocore_anchor.py`
- Modify: spec §4.6 (replace the per-member side file with a separate ensemble-1 extraction; see Step 3)
- Test: `tests/test_geocore_anchor.py`

**Interfaces:**
- Produces: `python -m eval.geocore_anchor --features '<ens1 run dir>/multistep_{split}_feats_oneshot_g0.0_k4-8-12-16-20-24-28-31.npz' --blocks 4 8 16 24 --t 260 --epochs 40 --lr 1e-3 --batch 1024 --seed 0`: loads train+val+test caches (ensemble_size must be 1 in meta, else `SystemExit`), concatenates the four block slices at t = 260 → `(N, 16384)`, random 80/20 split with `np.random.default_rng(0).permutation`, trains `torch.nn.Linear(16384, 10)` with Adam (CPU is fine: 20k x 16k), prints `ANCHOR eurosat acc=<x> (GeoCore reports 97.5 at 256 px, tau 0.25; here t=260, <res> px)`. Function `anchor_accuracy(X, y, seed=0, epochs=40, lr=1e-3, batch=1024) -> float`.

- [ ] **Step 1: Write the failing test** — `anchor_accuracy` on a linearly separable synthetic set (200 x 32, 3 classes) returns > 0.95; `--features` with `ensemble_size: 8` meta exits with the ensemble message.
- [ ] **Step 2: Run** → FAIL; **Step 3: Implement**; edit spec §4.6 to say: anchor cache = a separate m_eurosat extraction with `--model.ensemble-size 1 --t 260 580` (same blocks, same seeds; `extract.sbatch ANCHOR=1`), stored under `models/geocore_eval_anchor/`; one eps per image as in their probe.
- [ ] **Step 4: Run** → PASS; **Step 5: Stage** `eval/geocore_anchor.py tests/test_geocore_anchor.py docs/superpowers/specs/2026-11-02-geocore-eval-design.md`

---

### Task 11: SLURM jobs — download, smoke, extract

**Files:**
- Create: `experiments/isaac/geocore/download.sbatch`, `experiments/isaac/geocore/smoke.sbatch`, `experiments/isaac/geocore/extract.sbatch`, `experiments/isaac/geocore/README.md`
- Test: shell dry-run with a stub `python3` on `PATH` that prints its argv (CLAUDE.md "validate before submitting")

**Interfaces:**
- `download.sbatch` (`-p short -q short -A isaac-utk0256 -t 2:00:00 -c 4`): `huggingface_hub.snapshot_download("JeonghyeokDo/GeoCore-9B", local_dir=<STORE>/GeoCore-9B, allow_patterns=["model-*.safetensors", "model.safetensors.index.json", "config.json", "vae/*", "SHA256SUMS", "LICENSE-FLUX2-VAE.md"])`, then `cd <STORE>/GeoCore-9B && sha256sum -c SHA256SUMS`; refuses to run if the directory already exists; prints `GEOCORE_DOWNLOAD_OK`.
- `extract.sbatch`: copy `experiments/isaac/git10m/extract_ckpt.sbatch` structure (bootstrap, env check, DATASETS array, `official_spec` splits, `seed_for`, reuse check, incomplete-dir cleanup) with: `--model.name geocore --geocore-ckpt <STORE>/GeoCore-9B --geocore-vae <STORE>/GeoCore-9B/vae/diffusion_pytorch_model.safetensors --guidance-scale 0.0 --k 4 8 12 16 20 24 28 31 --img-size $RES $RES`, `ENS=8`, `SAVE_ROOT=models/geocore_eval${RES:+_img$RES}/$DATASET` (224 → `models/geocore_eval/`, 256 → `models/geocore_eval_img256/`); `ANCHOR=1` → `ENS=1`, `TS=(260 580)`, `SAVE_ROOT=models/geocore_eval_anchor/$DATASET`; `SMOKE=1` → `ENS=2 --subset-size 24`, `SAVE_ROOT=models/geocore_eval_smoke/...` (wiped first). Default partition campus-gpu-bigmem (`-n 1 --gpus-per-task=1 -c 8 --mem=96G -t 2-00:00:00`); `PART=ai-tenn` switches `-p/-q` and `-t 30:00:00`. Prints the reuse-check meta comparison (fields: t, k, ensemble_size, seed, eps_seed, split, dataset, subset_size, extraction_mode, guidance_scale 0.0, weights geocore-9b, img_size, geocore_ckpt_sha256) and `GEOCORE_EXTRACT_OK <dataset> <res>`.
- `smoke.sbatch` (campus-gpu-bigmem, 1 GPU, 2 h): (1) `pytest --gpu tests/test_geocore_null_text.py`; (2) `extract.sbatch` logic inline with `SMOKE=1 DATASET=m_eurosat`, then a python check: feats `(24, 9, 8, 4096)`, mods `(9, 8, 3, 4096)`, finite, meta keys from Task 7, `discard_channels` present; (3) parity gate: for image 0, member 0, rebuild `z_t` from the cache's `eps_seed` path (re-run `extract_raw` with ensemble 1 and the same generator seed) and compare its k=8 feature with `double_blocks[7].forward_kv_extract` output captured by wrapping the method during `Flux2.forward` on the same `z_t` (`allclose`, atol 1e-2 in bf16 → compare in float32); (4) FLUX regression: `run.py --task extract --model.name flux` on 20 m_eurosat images with `--k 4 28 --t 100 180 --model.ensemble-size 2 --subset-size 20` into `models/geocore_smoke_fluxreg/`, compared byte-for-byte (`np.array_equal` on feats and mods) against a cache produced by the SAME command run from the main checkout `/nfs/home/jdosch1/COSC424/DiT_remote_sensing` (pre-change code) in the same job, both with `--seed 42 --eps-seed 42`. Prints `GEOCORE_SMOKE_OK`.

- [ ] **Step 1: Write the dry-run check**: `experiments/isaac/geocore/dryrun.sh` puts a stub `python3` first on PATH that echoes `"$@"` and exits 0, runs `bash extract.sbatch` with `SLURM_ARRAY_TASK_ID=0 RES=224 SMOKE=1`, and greps the echoed argv for every flag above; exits non-zero on a missing flag. Also runs `python3 run.py --help` with the real interpreter (seconds) to confirm `--geocore-ckpt`, `--geocore-vae`, `--model.name` accept the values (tyro choices).
- [ ] **Step 2: Run the dry-run** → FAIL (files absent)
- [ ] **Step 3: Write the three sbatch files and README** (what each env var does, the order download → smoke → extract, expected log lines).
- [ ] **Step 4: Run the dry-run** → PASS. `sbatch --test-only` each file from the worktree (prints a start estimate, submits nothing).
- [ ] **Step 5: Stage** — `git add experiments/isaac/geocore`
- [ ] **Step 6: Submit download** (short), wait for `GEOCORE_DOWNLOAD_OK`; check `lfs quota -h -u jdosch1 /lustre/isaac24/scratch` before and after (+18.6 GB).
- [ ] **Step 7: Submit smoke**; wait for `GEOCORE_SMOKE_OK`. Any failure stops the plan here; fix, re-run smoke, never proceed on a partial pass.

---

### Task 12: Production extraction, block selection, cells, report

**Files:**
- Create: `eval/matrix_specs/phase8_geocore.json`
- Create: `/lustre/isaac24/scratch/jdosch1/tmp/geocore_chain.sh` (watcher that submits cells as extractions land, modelled on `auto_cells7.sh` + `phase7_chain.sh`; moves 100 %-label cell jobs to campus-gpu-bigmem instead of `long`)

- [ ] **Step 1: Submit extraction** `--array=0-5` on campus-gpu-bigmem with `RES=224`; when ai-tenn opens (2026-11-08) resubmit still-pending tasks with `PART=ai-tenn` and cancel the pending bigmem copies (own jobs only). Then `RES=256 --array=0,3` and `ANCHOR=1 --array=0`.
- [ ] **Step 2: Block selection** per dataset once its three splits exist: `python -m eval.sweep block` for each k in the eight blocks with `--expect extraction_mode=ONESHOT ensemble_size=8 weights=geocore-9b guidance_scale=0.0 k=<k>` and `--features 'models/geocore_eval/<ds>/*/multistep_{split}_feats_oneshot_g0.0_k4-8-12-16-20-24-28-31.npz'`, then `python -m eval.sweep select` (val only; the one test use it prints is not quoted). Array on short (3 h) with `probe_multiblock.sbatch`'s packing; `long` only for the forestnet 7-C block as before.
- [ ] **Step 3: Write `phase8_geocore.json`**: copy `phase7_semt.json` fractions / seeds / c_grid / datasets (all six); models: `dinov3-sat` (cached cells), `flux-proj` and `flux-pretrained` (cached cells, for the report's columns), `geocore-9b` with `kind: flux`, `features` pattern above, `expect {extraction_mode: ONESHOT, ensemble_size: 8, weights: geocore-9b, guidance_scale: 0.0}`, `per_dataset.<ds>.expect.k` from Step 2, `_comment` naming the sweep files; `geocore-9b-256` for m_eurosat / m_so2sat (pattern under `models/geocore_eval_img256/`, `expect.img_size [256, 256]`); DiTF row `geocore-9b-ditf` = same with `"transform": "ditf"` only if Task 9 Step 5 printed `DISCARD_OK`.
- [ ] **Step 4: Cells** via `experiments/isaac/eval/matrix.sbatch` (`PART=low` on short with `SEEDS` split, `PART=full` on campus-gpu-bigmem `-c 35 --mem 128G -t 8:00:00`), driven by the chain script.
- [ ] **Step 5: Report** `python -m eval.matrix report --spec eval/matrix_specs/phase8_geocore.json`, outputs moved to `results/matrix/phase8/` (snapshot/restore `results/matrix/summary.*` as `phase7_report.sh` does); anchor: `python -m eval.geocore_anchor ...` for 224 and 256; paste both into `results/matrix/phase8/README.md` with the harness-validation verdict (spec §1 item 3).
- [ ] **Step 6: Stage** `eval/matrix_specs/phase8_geocore.json results/matrix/phase8 results/matrix/cells/geocore-9b*` and report job ids + logs in the handoff.

---

### Task 13: Paper integration

**Files:**
- Modify: `/nfs/home/jdosch1/COSC424/DiT_remote_sensing/OUTLINE.md` (Table A row, related-work paragraph, contribution-2 sentence, supplement list), `/nfs/home/jdosch1/COSC424/DiT_remote_sensing/SUPPLEMENTARY.md` (256 check, anchor table, DiTF row, block x t curves pointer)

- [ ] (placeholder row + caption sentence in place 2026-11-02; numbers pending) **Step 1:** Add the Table A row `GeoCore-9B (frozen)` with mean5 / mean6 and per-dataset numbers from `results/matrix/phase8/summary.md`, caption sentence: "GeoCore-9B: frozen, 224 px, empty prompt, null geo metadata, our probe; their own EuroSAT readout rebuilt from our cache is in the supplement."
- [x] (2026-11-02, text in place) **Step 2:** Related work: one paragraph on GeoCore (from-scratch 9.24B EO DiT on Git-10M; GSA loss aligns block 8 to frozen DINOv3-Sat with mu = 0.5; NeurIPS 2026; arXiv 2608.01896). Contribution 2 sentence rewritten to "adapting a pretrained natural-image generator with a small adapter, against training a 9B EO generator from scratch (GeoCore)".
- [x] (2026-11-02, B1 entry + FID caveat in §2; numbers pending) **Step 3:** Supplement entries; the FID caveat sentence (no FID vs GeoCore outside fMoW val).
- [ ] **Step 4:** `git add OUTLINE.md SUPPLEMENTARY.md` in the main checkout.

---

### Task 14 (stretch): Segmentation tokens through the adapter

- [x] **Step 1:** `tasks/seg_tokens.py` already calls `model.extract_raw` and `model.eval()` (Task 7); add `C = model.hidden_size` where 3072 is assumed (grep `3072` in `src/tasks/seg_tokens.py`, `eval/seg_probe.py`) and read it from meta in the probe. Done 2026-11-02: C was never hardcoded (it is read from `feat_raw`); the real gaps were the inline FLUX provenance (now `model_provenance`), the FLUX-count double-block stamp (now `block_stamp_for`), the fp16 overflow check's FLUX discard list (now the model's own, none for GeoCore until stamped) and the seg probe's DiTF discard list (now per split from meta via `discard_for`). A GeoCore seg cache needs its `discard_channels` stamped into `seg_tokens_<split>_<tag>_meta.json` from the classification train cache's list (`eval.discard_channels --stamp <that json>`) before the DiTF seg view can read it.
- [ ] **Step 2:** Smoke `--max-images 8` at 512 px on campus-gpu-bigmem for `m_pv4ger_seg`; token grid must be `(32, 32)`.
- [ ] **Step 3:** Only after Task 12 lands: one dataset at a time (disk), `experiments/isaac/seg/` template with `--model.name geocore`.

## Self-review notes

- Spec coverage: §4.1 → T1; §4.2 → T2, T4, T5; §4.3 → T3, T6, T7, T8; §4.4 → T12; §4.5 → T9; §4.6 → T10 (spec amended: separate ens-1 extraction instead of a per-member side file); §5 → T11, T12; §6 → T5, T6, T7, T8 tests; §7 → T4, T5; §8 → T13; stretch → T14.
- Names used across tasks: `forward_feat`, `tiny_params`, `GeoCoreModel.from_parts`, `provenance_meta`, `model_provenance`, `weights_tag`, `num_double_blocks`, `hidden_size`, `discard_channels`, `massive_channels`, `anchor_accuracy` — each defined once in its Interfaces block.
- Staging rule: Tasks 7 and 8 leave `src/tasks/extraction.py`, `eval/features.py`, `eval/probe.py` unstaged.
