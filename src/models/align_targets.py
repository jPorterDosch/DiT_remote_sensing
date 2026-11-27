"""Alignment targets for the semantic-alignment (REPA-style) loss -- modular by design.

finetune-diffusion (tasks/train_diffusion.py, tasks/align.py) aligns FLUX block-k INPUT tokens
(B, L, C) through a trained projector to the patch tokens (B, L', D) of a FROZEN target encoder. Which encoder that is should be a
swappable choice, not wired in: this module defines the interface and a registry.

    target = build("dinov3_vit7b16_sat", device)          # any eval.extract_dino preset
    target = build("dino:dinov2_vitl14", device)          # same, explicit family prefix
    target = build("import:my_pkg.my_mod:MyTarget", device)  # your own class, see AlignTarget
    target = build("medsiglip:448@<revision>", device)     # MedSigLIP-448 (chest X-ray); pin required
    # TODO(spec s5/s6): shuffled / random-teacher control at matched compute -- a target whose
    # patches are a fixed random permutation of the images' real targets (or a frozen random
    # encoder); not implemented, no family registered for it yet.

    z = target.patches(imgs)                 # imgs (B, 3, H, W) in [-1, 1] -> (B, L', D)
    z = match_grid(z, n_tokens=feat.shape[1])  # L' -> L when the token grids differ

Adding a target: subclass AlignTarget (implement .patches, set .name / .dim), then either
@register("myname") a factory here, or point --align-target at it with the import: form.
Targets must be frozen (no grad) and deterministic in eval mode.
"""

from __future__ import annotations

import importlib
import math
from typing import Callable

import torch
import torch.nn.functional as F


class AlignTarget:
    """Interface. .name: stable id (run names / meta). .dim: patch-token width D."""

    name: str = "abstract"
    dim: int = 0

    @torch.no_grad()
    def patches(self, imgs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        """(B, 3, H, W) images in [-1, 1] -> (B, L', D) patch tokens, row-major square grid."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "dim": self.dim, "class": type(self).__name__}


_REGISTRY: dict[str, Callable[[str, torch.device], AlignTarget]] = {}


def register(family: str):
    """Register a factory(spec_suffix, device) -> AlignTarget under a family prefix."""

    def deco(fn):
        if family in _REGISTRY:
            raise ValueError(f"target family {family!r} already registered")
        _REGISTRY[family] = fn
        return fn

    return deco


def families() -> list[str]:
    return sorted(_REGISTRY)


def build(spec: str, device) -> AlignTarget:
    """'family:arg' -> registered factory; 'import:mod.path:Attr' -> user class/factory; a bare
    name is tried as a DINO preset (the default family)."""
    if spec.startswith("import:"):
        _, mod, attr = spec.split(":", 2)
        obj = getattr(importlib.import_module(mod), attr)
        t = obj(device) if callable(obj) else obj
    else:
        family, _, arg = spec.partition(":") if ":" in spec else ("dino", ":", spec)
        if family not in _REGISTRY:
            raise SystemExit(f"unknown alignment-target family {family!r} in {spec!r}; have {families()}")
        t = _REGISTRY[family](arg, device)
    if not isinstance(t, AlignTarget):
        raise SystemExit(f"{spec}: built {type(t).__name__}, not an AlignTarget")
    if t.dim <= 0:
        raise SystemExit(f"{spec}: target dim must be set (got {t.dim})")
    return t


def match_grid(z: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Resample (B, L', D) square-grid patch tokens to n_tokens (square) by adaptive average
    pooling over the 2-D grid. Identity when L' == n_tokens (the DINOv3/16 at 224 px case,
    14x14 = FLUX's packed grid). Refuses non-square grids rather than guessing a layout."""
    B, L, D = z.shape
    if L == n_tokens:
        return z
    s_in, s_out = math.isqrt(L), math.isqrt(n_tokens)
    if s_in * s_in != L or s_out * s_out != n_tokens:
        raise RuntimeError(f"token grids must be square to resample: target {L} -> features {n_tokens}")
    g = z.transpose(1, 2).reshape(B, D, s_in, s_in).float()
    g = F.adaptive_avg_pool2d(g, s_out)
    return g.reshape(B, D, n_tokens).transpose(1, 2).to(z.dtype)


class DinoTarget(AlignTarget):
    """Last-block, normed patch tokens of an eval.extract_dino preset (DINOv2 / DINOv3)."""

    def __init__(self, preset: str, device):
        from eval.extract_dino import load_model

        self.model, self.spec = load_model(preset, None, device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        mean, std = self.spec["norm"]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.name, self.dim, self.preset = preset, int(self.model.embed_dim), preset

    @torch.no_grad()
    def patches(self, imgs):
        x = ((imgs.float() / 2 + 0.5) - self.mean) / self.std
        x = x.to(self.spec["dtype"])
        patch, _cls = self.model.get_intermediate_layers(x, n=1, return_class_token=True, norm=True)[-1]
        return patch

    def describe(self):
        return {**super().describe(), "preset": self.preset, "norm": list(map(list, self.spec["norm"]))}


class MedSigLIPTarget(AlignTarget):
    """MedSigLIP-448 last-layer patch tokens (models/medsiglip.py; design section 7.1). The training
    image (224 px, [-1, 1]) is upsampled to 448 inside the encoder; the 32 x 32 tokens are pooled to
    the student's grid by match_grid."""

    def __init__(self, device, revision: str | None = None, encoder=None):
        from models.medsiglip import MedSigLIP

        self.enc = encoder if encoder is not None else MedSigLIP(device, revision=revision)
        self.dim = self.enc.dim
        self.name = f"medsiglip{self.enc.side}"

    @torch.no_grad()
    def patches(self, imgs):
        tok, _pooled = self.enc.encode(imgs)
        return tok

    def describe(self):
        return {**super().describe(), **self.enc.info}


def parse_medsiglip_arg(arg: str) -> str:
    """'448@<revision>' -> the revision. The pin is required: the target string is the run identity
    (AlignConfig.identity -> run name and resume check), so an unpinned `medsiglip:448` would name
    the same run whichever snapshot the Hub serves (project rule 9: a default is not a pin).
    `python -m models.medsiglip` prints the pin of the cached snapshot."""
    side, _, rev = arg.partition("@")
    if side != "448" or not rev:
        raise SystemExit(
            f"medsiglip target: expected 'medsiglip:448@<revision>' (pinned; `python -m models.medsiglip` "
            f"prints it), got {arg!r}"
        )
    return rev


@register("medsiglip")
def _medsiglip(arg: str, device) -> AlignTarget:
    return MedSigLIPTarget(device, revision=parse_medsiglip_arg(arg))


@register("dino")
def _dino(arg: str, device) -> AlignTarget:
    from eval.extract_dino import PRESETS

    if arg not in PRESETS:
        raise SystemExit(f"unknown DINO preset {arg!r}; have {sorted(PRESETS)}")
    return DinoTarget(arg, device)
