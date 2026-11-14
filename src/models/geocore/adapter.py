"""GeoCore-9B as a frozen feature extractor behind the registry contract the FLUX adapter
exposes (`extract_raw`), so `tasks/extraction.py`, `tasks/utils.extract_features`,
`tasks/seg_tokens.py` and the matrix probes run unchanged.

Protocol constants are PINNED here (spec §2/§3, CLAUDE.md rule 9): the weights are verified
against the upstream SHA256SUMS before anything is loaded, the prompt is the empty string, the
geospatial fields take GeoCore's learned null embedding (-999.0), the VAE encode is the posterior
mean behind the frozen BatchNorm exactly as shipped, and there is no guidance (the model has no
guidance embedding). Each is stamped into every cache's meta through `provenance_meta()`.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
from einops import rearrange
from torch import Tensor

from registry import register_model

from . import GEOCORE_COMMIT
from .flux2 import Flux2, GeoCore9BParams
from .null_text import CLIP_DIR_DEFAULT, T5_DIR_DEFAULT, null_text_embeddings
from .trunk import forward_feat
from .vae_flux2 import load_autoencoder

# huggingface/SHA256SUMS at GEOCORE_COMMIT (the HF repo JeonghyeokDo/GeoCore-9B ships the same file).
PINNED_SHARD_SHA256 = {
    "model-00001-of-00004.safetensors": "37867b56a02748c67fe64f60118b6b6cd6966653026ba101abd1bf849e4f4b37",
    "model-00002-of-00004.safetensors": "561831477a146da51524beacbd28e4e4db9c3d66318c98388823e641e920a956",
    "model-00003-of-00004.safetensors": "60107ab28114906426b4187bbca9fda59af121318cb39543a90697b7a7ecef9a",
    "model-00004-of-00004.safetensors": "7cef9bd84c652cb25d0868d0f06d3cc4279d8a066db219f9c5f3fb4c7ac7288a",
}
PINNED_INDEX_SHA256 = "6af0afb014d208f28e04804cb0bbe8bdf3b52a8490cf3610e001b85c17fa8b65"
PINNED_VAE_SHA256 = "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"
PINNED_TEXT = ""
PINNED_GEO_NULL = -999.0
PINNED_VAE_ENCODE = "mean"
PINNED_GUIDANCE = 0.0
WEIGHTS_TAG = "geocore-9b"
HIDDEN_SIZE = 4096
NUM_DOUBLE_BLOCKS = 8
NUM_BLOCKS = 32
LATENT_DOWNSAMPLE = 16  # 8x VAE then 2x2 patchify -> one token per 16x16 pixels
INDEX_FILE = "model.safetensors.index.json"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: str, expected: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"geocore: {path} missing")
    got = sha256_of(path)
    if got != expected:
        raise ValueError(f"geocore: sha256 mismatch for {path}: got {got}, pinned {expected}")
    return got


def _check_cfg(cfg) -> None:
    """Protocol guards that need no file access (spec §6). Raised before any weight is read."""
    if getattr(cfg, "lora_checkpoint", ""):
        raise ValueError("geocore: lora_checkpoint is not supported (frozen weights only)")
    mode = getattr(cfg, "extraction_mode", None)
    if mode is not None and getattr(mode, "value", mode) != "ONESHOT":
        raise ValueError(f"geocore: one-shot extraction only, got extraction_mode={mode}")
    if float(cfg.guidance_scale) != PINNED_GUIDANCE:
        raise ValueError(
            f"geocore: guidance_scale must be {PINNED_GUIDANCE} (the model has no guidance embedding), got {cfg.guidance_scale}"
        )
    sizes = list(cfg.img_size) if isinstance(cfg.img_size, (list, tuple)) else [cfg.img_size, cfg.img_size]
    if any(int(s) % LATENT_DOWNSAMPLE for s in sizes):
        raise ValueError(f"geocore: img_size must be a multiple of {LATENT_DOWNSAMPLE}, got {cfg.img_size}")
    ks = [cfg.k] if isinstance(cfg.k, int) else list(cfg.k)
    if any(not 0 <= int(k) < NUM_BLOCKS for k in ks):
        raise ValueError(f"geocore: k must lie in [0, {NUM_BLOCKS - 1}], got {cfg.k}")
    if not getattr(cfg, "geocore_ckpt", "") or not getattr(cfg, "geocore_vae", ""):
        raise ValueError(
            "geocore: --geocore-ckpt (snapshot directory) and --geocore-vae (safetensors file) are required"
        )


def _latent_ids(h: int, w: int, batch: int, device, dtype) -> Tensor:
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(1, h * w, 3).expand(batch, -1, -1)


@register_model("geocore")
class GeoCoreModel:
    hidden_size = HIDDEN_SIZE
    num_double_blocks = NUM_DOUBLE_BLOCKS
    num_blocks = NUM_BLOCKS
    weights_tag = WEIGHTS_TAG
    vae_encode = PINNED_VAE_ENCODE
    text = PINNED_TEXT
    geo_null = PINNED_GEO_NULL
    commit = GEOCORE_COMMIT
    discard_channels: list[int] | None = None  # set by the extraction job from the train cache (spec §4.5)

    def __init__(self, cfg, category_list: list[str]) -> None:
        _check_cfg(cfg)
        device = torch.device(cfg.device)
        ckpt_dir = cfg.geocore_ckpt
        index_path = os.path.join(ckpt_dir, INDEX_FILE)
        self._index_sha256 = _verify_sha256(index_path, PINNED_INDEX_SHA256)
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        if set(shards) != set(PINNED_SHARD_SHA256):
            raise ValueError(f"geocore: index lists shards {shards}, pinned {sorted(PINNED_SHARD_SHA256)}")
        for name in shards:
            _verify_sha256(os.path.join(ckpt_dir, name), PINNED_SHARD_SHA256[name])
        self._vae_sha256 = _verify_sha256(cfg.geocore_vae, PINNED_VAE_SHA256)

        from safetensors.torch import load_file

        model = Flux2(GeoCore9BParams()).to(device=device, dtype=torch.bfloat16)
        state = {}
        for name in shards:
            state.update(load_file(os.path.join(ckpt_dir, name), device=str(device)))
        model.load_state_dict(state, strict=True)
        del state
        vae = load_autoencoder(cfg.geocore_vae, device=device, dtype=torch.bfloat16)
        ctx, ctx_ids, y = null_text_embeddings(
            T5_DIR_DEFAULT, CLIP_DIR_DEFAULT, device=device, dtype=torch.bfloat16
        )
        self._init_parts(model, vae, ctx, ctx_ids, y, device)
        print(
            f"geocore: loaded {len(shards)} shards from {ckpt_dir} (commit {GEOCORE_COMMIT[:8]}), vae {cfg.geocore_vae}"
        )

    @classmethod
    def from_parts(cls, model, vae, ctx: Tensor, ctx_ids: Tensor, y: Tensor, device=None) -> "GeoCoreModel":
        """Build without loading anything (tests, smokes). `vae` needs `.encode(x) -> (1, C_in, h, w)`."""
        self = cls.__new__(cls)
        self._index_sha256 = self._vae_sha256 = None
        self._init_parts(
            model, vae, ctx, ctx_ids, y, torch.device(device) if device is not None else ctx.device
        )
        return self

    def _init_parts(self, model, vae, ctx, ctx_ids, y, device) -> None:
        if (
            ctx.ndim != 3
            or ctx.shape[0] != 1
            or y.ndim != 2
            or y.shape[0] != 1
            or ctx_ids.shape != (1, ctx.shape[1], 3)
        ):
            raise ValueError(
                f"geocore: conditioning shapes ctx={tuple(ctx.shape)} ctx_ids={tuple(ctx_ids.shape)} y={tuple(y.shape)}"
            )
        self.model = model
        self.vae = vae
        self.ctx, self.ctx_ids, self.y = ctx, ctx_ids, y
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.n_blocks = len(model.double_blocks) + len(model.single_blocks)
        self.eval()

    def eval(self) -> "GeoCoreModel":
        self.model.eval()
        if hasattr(self.vae, "eval"):
            self.vae.eval()
        return self

    def provenance_meta(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_double_blocks": self.num_double_blocks,
            "num_blocks": self.num_blocks,
            "vae_encode": self.vae_encode,
            "text": self.text,
            "geo_null": self.geo_null,
            "geocore_commit": self.commit,
            "geocore_ckpt_sha256": self._index_sha256,
            "geocore_vae_sha256": self._vae_sha256,
        }

    @torch.no_grad()
    def extract_raw(
        self,
        img: Tensor,
        timestep: int,
        block_idx,
        ensemble_size: int,
        caption: str = PINNED_TEXT,
        category: str = "image",
        guidance: float = PINNED_GUIDANCE,
        latents: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One-shot noising to `timestep` (1..1000 index; t = timestep / 1000) and one trunk pass per
        ensemble member, averaged over members. Returns (feat_raw, ada, latents_used, noise_used):
        feat_raw (B, C, h, w) for a block list / (1, C, h, w) for an int, ada (B, 3, C) / (1, 3, C),
        latents_used and noise_used (ensemble_size, C_in, h, w) -- the tensors actually mixed, so
        callers can assert constancy across timesteps (tasks/utils.extract_features)."""
        if float(guidance) != PINNED_GUIDANCE:
            raise ValueError(f"geocore: guidance must be {PINNED_GUIDANCE}, got {guidance}")
        if caption != PINNED_TEXT:
            raise ValueError(f"geocore: the prompt is pinned to {PINNED_TEXT!r}, got {caption!r}")
        if img.ndim != 3:
            raise ValueError(f"geocore: img must be (3, H, W), got {tuple(img.shape)}")
        if img.shape[-1] % LATENT_DOWNSAMPLE or img.shape[-2] % LATENT_DOWNSAMPLE:
            raise ValueError(
                f"geocore: image sides must be multiples of {LATENT_DOWNSAMPLE}, got {tuple(img.shape)}"
            )
        multi_block = not isinstance(block_idx, int)
        ks = list(block_idx) if multi_block else [block_idx]
        t = float(timestep) / 1000.0
        x_img = img.unsqueeze(0).to(self.device, self.dtype)

        feats, mods, latents_used, noise_used = [], [], [], []
        for i in range(int(ensemble_size)):
            z0 = (
                latents[i : i + 1].to(self.device, self.dtype)
                if latents is not None
                else self.vae.encode(x_img).to(self.dtype)
            )
            if noise is not None:
                eps = noise[i : i + 1].to(self.device, self.dtype)
            elif generator is not None:
                eps = torch.randn(z0.shape, generator=generator, device=self.device, dtype=z0.dtype)
            else:
                eps = torch.randn_like(z0)
            zt = t * eps + (1.0 - t) * z0
            _, _, h, w = zt.shape
            null = torch.full((1,), PINNED_GEO_NULL, device=self.device, dtype=self.dtype)
            out = forward_feat(
                self.model,
                x=zt,
                x_ids=_latent_ids(h, w, 1, self.device, self.dtype),
                timesteps=torch.full((1,), t, device=self.device, dtype=self.dtype),
                ctx=self.ctx,
                ctx_ids=self.ctx_ids,
                y=self.y,
                res=null,
                lon=null,
                lat=null,
                ft_indices=ks,
            )
            feat = torch.cat(
                [rearrange(out[2 * j], "b (h w) c -> b c h w", h=h, w=w) for j in range(len(ks))], dim=0
            )
            mod = torch.cat([out[2 * j + 1] for j in range(len(ks))], dim=0)
            feats.append(feat)  # (B, C, h, w)
            mods.append(mod)  # (B, 3, C)
            latents_used.append(z0)
            noise_used.append(eps)
        feat_raw = torch.stack(feats, dim=0).mean(0)
        ada = torch.stack(mods, dim=0).mean(0)
        return feat_raw, ada, torch.cat(latents_used, dim=0), torch.cat(noise_used, dim=0)

    # The registry's single-timestep / inversion / roundtrip entry points are FLUX-only paths.
    def extract(self, *args, **kwargs):
        raise NotImplementedError(
            "geocore: one-shot multi-timestep extraction only (task=extract with --t list)"
        )

    def extract_inversion(self, *args, **kwargs):
        raise NotImplementedError("geocore: the inversion chain is not implemented")

    def roundtrip(self, *args, **kwargs):
        raise NotImplementedError("geocore: roundtrip is not implemented")
