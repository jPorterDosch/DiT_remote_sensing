"""Per-block gradient bookkeeping for FM-vs-semantic-alignment training (tasks/align.py).

Pure tensor functions (CPU-testable, no model): group LoRA parameters by DiT block, flatten
per-block gradients, cosine/dot statistics between two gradient sets, the ONE-SIDED
projection of a conflicting semantic gradient onto the half-space of the flow-matching (FM)
gradient, and an executed verifier for the projected result. The FM gradient is never
modified by anything here.

Conventions
    blocks: global DiT indices, 0..18 double-stream, 19..56 single-stream (config_types).
    A "gradient set" is {param_name: Tensor | None}; None = parameter unused by the loss
    (torch.autograd.grad(..., allow_unused=True)), treated as zeros.
    All arithmetic is done in float32 regardless of the parameters' dtype (bf16 LoRA).
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch


def _dot(x: torch.Tensor, y: torch.Tensor) -> float:
    """Inner product accumulated in float64. fp32 accumulation over the ~7.9M LoRA elements errs by
    ~1e-3 relative (measured: cosine 1.0008 for nearly parallel vectors, 2026-10-10 smoke showed
    cos_post 1.003) -- far above the verifier's 1e-5 tolerance, so every dot/norm here is fp64."""
    return float(x.double() @ y.double())


def _norm(x: torch.Tensor) -> float:
    return float(x.double().norm())


_BLOCK_RE = re.compile(r"(double|single)_blocks\.(\d+)\.")


def block_of(param_name: str, n_double: int) -> int:
    """Global block index of a LoRA parameter name, e.g. single_blocks.9.linear1.A -> 28."""
    m = _BLOCK_RE.search(param_name)
    if m is None:
        raise ValueError(f"not a block parameter: {param_name}")
    return int(m.group(2)) + (0 if m.group(1) == "double" else n_double)


def lora_groups(model: torch.nn.Module) -> OrderedDict[int, list[tuple[str, torch.nn.Parameter]]]:
    """{global block index: [(name, param), ...]} over every trainable LoRA A/B tensor, blocks
    ascending, names in model order. Raises if the model has no LoRA parameters."""
    n_double = len(model.double_blocks)
    groups: dict[int, list] = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name.endswith((".A", ".B")):
            groups.setdefault(block_of(name, n_double), []).append((name, p))
    if not groups:
        raise ValueError("model has no trainable LoRA parameters (.A/.B with requires_grad)")
    return OrderedDict(sorted(groups.items()))


def flatten(groups: OrderedDict, grads: dict[str, torch.Tensor | None]) -> OrderedDict[int, torch.Tensor]:
    """Per-block flat float32 gradient vectors (zeros for parameters the loss did not reach)."""
    out = OrderedDict()
    for b, params in groups.items():
        parts = []
        for name, p in params:
            g = grads[name]  # KeyError on a missing name: None (allow_unused) is the only zero case
            parts.append(
                torch.zeros(p.numel(), dtype=torch.float32)
                if g is None
                else g.detach().float().reshape(-1).cpu()
            )
        out[b] = torch.cat(parts)
    return out


def unflatten(groups: OrderedDict, flat: OrderedDict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of flatten: {name: tensor shaped like the parameter}, float32, on CPU."""
    out = {}
    for b, params in groups.items():
        v, off = flat[b], 0
        for name, p in params:
            n = p.numel()
            out[name] = v[off : off + n].view_as(p)
            off += n
        if off != v.numel():
            raise ValueError(f"block {b}: flat vector has {v.numel()} elements, parameters {off}")
    return out


def stats(g_a: OrderedDict, g_b: OrderedDict, eps: float = 1e-30) -> dict[str, torch.Tensor]:
    """Per-block (and global, last row) dot / norms / cosine between two flat gradient sets.
    cosine is NaN where either side is all-zero (e.g. a block the alignment loss cannot reach)."""
    if list(g_a) != list(g_b):
        raise ValueError("gradient sets cover different blocks")
    rows = list(g_a) + ["global"]
    a_all = torch.cat([g_a[b] for b in g_a])
    b_all = torch.cat([g_b[b] for b in g_b])
    pairs = [(g_a[b], g_b[b]) for b in g_a] + [(a_all, b_all)]
    dot = torch.tensor([_dot(x, y) for x, y in pairs], dtype=torch.float64)
    na = torch.tensor([_norm(x) for x, _ in pairs], dtype=torch.float64)
    nb = torch.tensor([_norm(y) for _, y in pairs], dtype=torch.float64)
    cos = dot / (na * nb)
    cos[(na < eps) | (nb < eps)] = float("nan")
    # computed in fp64, returned as fp32 (the established API; a cosine <= 1 survives the cast)
    return {
        "blocks": rows,
        "dot": dot.float(),
        "norm_a": na.float(),
        "norm_b": nb.float(),
        "cos": cos.float(),
    }


def project_one_sided(
    g_sem: OrderedDict, g_fm: OrderedDict, scope: str = "block"
) -> tuple[OrderedDict[int, torch.Tensor], torch.Tensor]:
    """Remove the component of the semantic gradient that opposes the FM gradient.

    For each unit (a block, or the whole parameter vector when scope="global"):
        if <g_sem, g_fm> < 0:  g_sem <- g_sem - (<g_sem, g_fm> / ||g_fm||^2) g_fm
        else:                  unchanged
    g_fm is returned untouched by construction (the caller keeps its own reference; nothing
    here writes to it). A unit whose FM gradient is zero has nothing to conflict with and is
    left unchanged. Returns (projected set, per-unit bool mask of where projection happened).
    """
    if scope not in ("block", "global"):
        raise ValueError(f"scope must be 'block' or 'global', got {scope!r}")
    if list(g_sem) != list(g_fm):
        raise ValueError("gradient sets cover different blocks")
    if scope == "global":
        s = torch.cat([g_sem[b] for b in g_sem])
        f = torch.cat([g_fm[b] for b in g_fm])
        s2, hit = _project_vec(s, f)
        out, off = OrderedDict(), 0
        for b in g_sem:
            n = g_sem[b].numel()
            out[b] = s2[off : off + n].clone()
            off += n
        return out, torch.tensor([hit])
    out, hits = OrderedDict(), []
    for b in g_sem:
        out[b], hit = _project_vec(g_sem[b], g_fm[b])
        hits.append(hit)
    return out, torch.tensor(hits)


def _project_vec(s: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, bool]:
    f2 = _dot(f, f)
    if f2 == 0.0:
        return s.clone(), False
    d = _dot(s, f)
    if d >= 0.0:
        return s.clone(), False
    # coefficient AND subtraction in fp64, one rounding to fp32 at the end
    return (s.double() - (d / f2) * f.double()).to(s.dtype), True


def verify_nonneg(
    g_proj: OrderedDict, g_fm: OrderedDict, tol: float = 1e-5, scope: str = "block"
) -> tuple[bool, torch.Tensor, list]:
    """Executed gate (rule 13): every unit satisfies <g_proj, g_fm> >= -tol * ||g_proj|| ||g_fm||.
    Returns (all_ok, per-unit normalized dot (cosine; 0 where a side is zero), unit labels).
    The check is RELATIVE so it is scale-free; dots/norms are fp64 (_dot), so an exact
    projection leaves |cos| at the fp32 storage rounding of the projected vector (~1e-8), not
    at fp32 accumulation noise (~1e-3 over 7.9M elements)."""
    if scope == "global":
        units = [("global", torch.cat(list(g_proj.values())), torch.cat(list(g_fm.values())))]
    else:
        units = [(b, g_proj[b], g_fm[b]) for b in g_proj]
    cos = []
    for _, p, f in units:
        denom = _norm(p) * _norm(f)
        cos.append(0.0 if denom == 0.0 else _dot(p, f) / denom)
    cos_t = torch.tensor(cos, dtype=torch.float64)
    ok = bool((cos_t >= -tol).all())  # decided in fp64; returned as fp32 below
    cos_t = cos_t.float()
    return ok, cos_t, [u[0] for u in units]


def check_fp32_params(params, opt, check_grads: bool = False, where: str = "") -> None:
    """Executed fp32 gate (the semantic gradient is 0.5-3 % of the FM one per block -- at bf16's
    ~0.4 % resolution -- so a bf16 slip anywhere on the update path rounds it away, smoke
    2026-10-10). The LoRA A/B are fp32 parameters cast to bf16 inside LoRALinear.forward, so
    the update path is: fp32 .grad -> fp32 AdamW state -> fp32 parameter. Raises RuntimeError
    naming the first violation; returns None on PASS.

    params      : the trainable LoRA tensors; each must be an fp32 leaf
    opt         : the optimizer; it must step EXACTLY these tensors
    check_grads : also require every present .grad to be fp32 (call before step)
    Also checks AdamW-style state (exp_avg, exp_avg_sq, ...) is fp32 once it exists.
    """
    tag = f" [{where}]" if where else ""
    for i, p in enumerate(params):
        if p.dtype != torch.float32:
            raise RuntimeError(f"fp32 gate{tag}: param {i} is {p.dtype}, not float32")
        if not (p.is_leaf and p.requires_grad):
            raise RuntimeError(f"fp32 gate{tag}: param {i} is not a trainable leaf")
        if check_grads and p.grad is not None and p.grad.dtype != torch.float32:
            raise RuntimeError(f"fp32 gate{tag}: param {i} .grad is {p.grad.dtype}, not float32")
    owned = [p for g in opt.param_groups for p in g["params"]]
    if len(owned) != len(params) or any(a is not b for a, b in zip(owned, params)):
        raise RuntimeError(
            f"fp32 gate{tag}: optimizer does not step exactly the fp32 LoRA params "
            f"({len(owned)} owned vs {len(params)} params, or different tensors)"
        )
    for i, p in enumerate(params):
        for k, v in opt.state.get(p, {}).items():
            if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1 and v.dtype != torch.float32:
                raise RuntimeError(f"fp32 gate{tag}: optimizer state {k!r} of param {i} is {v.dtype}")
