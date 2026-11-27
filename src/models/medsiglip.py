"""MedSigLIP-448 vision tower: the chest X-ray alignment target and the probe-baseline encoder.

Design: docs/superpowers/specs/2026-11-14-cxr-domain-design.md, section 7 (Part B).

- Weights: `google/medsiglip-448` on Hugging Face (gated; accept the model's terms first). Only the
  vision tower is loaded (SiglipVisionModel from the full checkpoint), and loading REFUSES if any
  vision weight is missing or mis-shaped (transformers would otherwise only warn and keep random
  weights). The resolved snapshot revision is recorded in every describe() / cache; the alignment
  target requires it pinned (`medsiglip:448@<revision>`, models/align_targets.py), so it is part of
  the run identity. `python -m models.medsiglip` downloads the snapshot and prints its revision.
- Input: images in [-1, 1], (B, 3, H, W). A non-native side is resized to the model's side with
  bilinear interpolation WITHOUT antialiasing (the model card's resize). The 224 px training view
  is therefore upsampled to 448 -- the accepted design call, with its named failure mode (the
  `@224up` vs `@448` probe rows of eval/extract_medsiglip.py measure it).
- Normalisation: the shipped preprocessor_config.json's mean / std, read from the snapshot, never
  typed in. The card states they map [0, 1] to [-1, 1]; preprocess_info() asserts exactly that, and
  that the shipped resample is bilinear (2), the method prepare() implements.
- Outputs: `last_hidden_state` (B, grid^2, D), the post-layernorm patch tokens (SigLIP has no class
  token), and `pooler_output` (B, D), the attention-pooled head embedding.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

MODEL_ID = "google/medsiglip-448"
RESIZE = "bilinear, align_corners=False, antialias=False"
BILINEAR = 2  # PIL.Image.Resampling.BILINEAR, as stored in preprocessor_config.json


def snapshot(model_id: str = MODEL_ID, revision: str | None = None) -> str:
    """Local snapshot directory (downloads config, preprocessor config and weights on first use;
    set HF_HUB_OFFLINE=1 on nodes without network once the cache is filled)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors"],
    )


def preprocess_info(snapshot_dir: str) -> dict:
    """mean / std / side / resample from the shipped preprocessor_config.json, with the card's
    [0, 1] -> [-1, 1] mapping asserted (a guard that must be able to fail: tests feed it a bad file)."""
    with open(os.path.join(snapshot_dir, "preprocessor_config.json")) as f:
        pc = json.load(f)
    mean, std = [float(x) for x in pc["image_mean"]], [float(x) for x in pc["image_std"]]
    if len(mean) != 3 or len(std) != 3:
        raise SystemExit(f"preprocessor_config.json: expected 3-channel mean/std, got {mean} / {std}")
    for m, s in zip(mean, std):
        lo, hi = (0.0 - m) / s, (1.0 - m) / s
        if abs(lo + 1.0) > 1e-6 or abs(hi - 1.0) > 1e-6:
            raise SystemExit(f"preprocessor_config.json mean {mean} std {std} does not map [0, 1] to [-1, 1]")
    size = pc.get("size", {})
    side = int(size.get("height", size.get("shortest_edge", 0)))
    if side <= 0 or int(size.get("width", side)) != side:
        raise SystemExit(f"preprocessor_config.json: expected a square size, got {size}")
    if pc.get("resample") != BILINEAR:
        raise SystemExit(
            f"preprocessor_config.json resample {pc.get('resample')!r} is not bilinear ({BILINEAR}); prepare() "
            "implements the card's bilinear resize, so the model input would not match the shipped processor"
        )
    return {"mean": mean, "std": std, "side": side, "resample": pc.get("resample")}


def load_vision_tower(snap: str, dtype=torch.float32):
    """SiglipVisionModel from a (full SigLIP) snapshot; refuses when any vision weight was not loaded.
    Unexpected keys are expected (the text tower of the full checkpoint) and ignored."""
    from transformers import SiglipVisionModel

    model, li = SiglipVisionModel.from_pretrained(snap, dtype=dtype, output_loading_info=True)
    missing, mismatched = sorted(li.get("missing_keys") or []), sorted(li.get("mismatched_keys") or [])
    if missing or mismatched:
        raise SystemExit(
            f"{snap}: vision weights not loaded from the checkpoint (missing {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}, mismatched {mismatched[:5]}) -- refusing a partly "
            "random teacher"
        )
    return model


class MedSigLIP:
    """Frozen vision tower + the input transform. `model` / `info` may be injected (tests)."""

    def __init__(
        self,
        device,
        dtype=torch.float32,
        model_id: str = MODEL_ID,
        revision: str | None = None,
        model=None,
        info: dict | None = None,
        snapshot_dir: str | None = None,
    ):
        if model is None:
            snap = snapshot_dir or snapshot(model_id, revision)
            rev = os.path.basename(os.path.normpath(snap))
            if revision is not None and not rev.startswith(revision):
                raise SystemExit(f"pinned revision {revision!r} but the snapshot resolved to {rev!r}")
            info = {**preprocess_info(snap), "model_id": model_id, "revision": rev}
            model = load_vision_tower(snap, dtype)
        if info is None:
            raise ValueError("an injected model needs its info dict")
        cfg = model.config
        side, patch = int(cfg.image_size), int(cfg.patch_size)
        if side != info["side"]:
            raise SystemExit(f"model image_size {side} != preprocessor side {info['side']}")
        if side % patch:
            raise SystemExit(f"image size {side} is not a multiple of the patch size {patch}")
        self.model = model.to(device=device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device, self.dtype = torch.device(device), dtype
        self.side, self.patch, self.grid, self.dim = side, patch, side // patch, int(cfg.hidden_size)
        self.mean = torch.tensor(info["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(info["std"], device=device).view(1, 3, 1, 1)
        self.info = {
            **info,
            "patch": patch,
            "grid": self.grid,
            "dim": self.dim,
            "resize": RESIZE,
            "dtype": str(dtype),
        }

    def prepare(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> normalized (B, 3, side, side) model input."""
        if imgs.ndim != 4 or imgs.shape[1] != 3 or imgs.shape[2] != imgs.shape[3]:
            raise ValueError(f"expected square (B, 3, H, H) images, got {tuple(imgs.shape)}")
        lo, hi = float(imgs.min()), float(imgs.max())
        if lo < -1.001 or hi > 1.001:
            raise ValueError(f"images must be in [-1, 1], got [{lo:.3f}, {hi:.3f}]")
        x = imgs.to(self.device, torch.float32)
        if x.shape[-1] != self.side:
            x = F.interpolate(
                x, size=(self.side, self.side), mode="bilinear", align_corners=False, antialias=False
            )
        x = ((x + 1) / 2 - self.mean) / self.std
        return x.to(self.dtype)

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (patch tokens (B, grid^2, D), pooled head embedding (B, D))."""
        out = self.model(pixel_values=self.prepare(imgs))
        tok = out.last_hidden_state
        n = tok.shape[1]
        if n != self.grid * self.grid or math.isqrt(n) ** 2 != n:
            raise RuntimeError(f"expected a square {self.grid}x{self.grid} token grid, got {n} tokens")
        return tok, out.pooler_output


if __name__ == "__main__":  # login node: download (gated; `huggingface-cli login` first) and print the pin
    snap = snapshot()
    print(snap)
    print("align target pin: medsiglip:448@" + os.path.basename(os.path.normpath(snap)))
