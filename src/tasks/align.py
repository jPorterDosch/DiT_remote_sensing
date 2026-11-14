"""Semantic alignment for finetune-diffusion (tasks/train_diffusion.py): the projector, the
frozen target, the two-gradient bookkeeping per DiT block, the one-sided gradient surgery with
its executed verifiers, the per-step statistics, the measurement passes and the projector warmup.

Two losses share one LoRA (every double + single block, models/lora.lora_block_indices):
  FM    flow-matching MSE on the velocity, v(x_t, t) vs (eps - x0) (the task's own loss).
  SEM   REPA-style alignment: block-k INPUT tokens in their DiTF form (AlignConfig.feat: massive
        channels zeroed, LayerNorm, the block's adaLN = the block's modulated input; "raw" = the
        plain tokens; the matrix probes read the RAW pooled tokens, see AlignConfig) -> projector
        -> negative cosine to the frozen target's patch tokens of the clean image. At 224 px FLUX
        packs 28x28 latents into 14x14 = 196 tokens and DINOv3/16 gives 14x14 patches, so the
        grids coincide position by position (align_targets.match_grid resamples otherwise).
Both gradients are taken separately w.r.t. the LoRA from ONE graph (torch.autograd.grad), so
every arm records their per-block relation and only the UPDATE RULE differs (AlignConfig.arm).

Why not a loss subclass with a custom backward: the surgery acts on the two gradient VECTORS of
each block after they are separated, and a scalar loss's backward only sees grad_output = 1 for
the sum -- the separation has to happen at the parameter level, which is what accumulate() does
(the same thing as flow_loss.backward(retain_graph) + snapshot .grad + align_loss.backward(),
at the same cost, without touching .grad until apply()).

The LoRA tensors are fp32 (models/lora.py param_dtype): the semantic gradient is 0.5-3 % of the
FM one per block, below bf16's ~0.4 % relative resolution, so a bf16 .grad rounds it away
(measured 2026-10-10). Forward/backward run in bf16; .grad, the combination, AdamW state and
the parameters are fp32, and gradproj.check_fp32_params is executed around every step.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from config_types import ALIGN_ARMS, NUM_BLOCKS, AlignConfig
from models import gradproj as GP
from models.align_targets import AlignTarget, build as build_target, match_grid


class Projector(torch.nn.Module):
    """REPA projector from the DiT hidden size to the target dim."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, arch: str = "mlp"):
        super().__init__()
        if arch == "mlp":  # 3-layer SiLU MLP (REPA)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(d_hidden, d_hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(d_hidden, d_out),
            )
        else:
            # TODO(spec s9): arch="linear" (one nn.Linear) is the second required projector; refuse
            # rather than silently fall back to the MLP.
            raise ValueError(f"align.proj_arch={arch!r}: only 'mlp' is implemented")
        self.arch = arch

    def forward(self, x):
        return self.net(x)


class AlignStats:
    """Per-step per-block arrays (S rows x blocks + 1 global row) and per-step scalars; plain
    numpy so a checkpoint can carry them and stats.npz can be written from them."""

    BLOCK_KEYS = ("cos_pre", "cos_post", "norm_fm", "norm_sem", "dot", "cos_used")
    BOOL_KEYS = ("conflict", "projected")
    SCALAR_KEYS = (
        "flow_loss",
        "align_loss",
        "t_mean",
        "grad_norm",
        "lr",
        "applied_cos_min",
        "step_time",
        "w_eff",
        "cos_update",
    )

    def __init__(self, steps: int, n_rows: int, accum: int = 0, batch: int = 0):
        self.st = {k: np.full((steps, n_rows), np.nan, dtype=np.float32) for k in self.BLOCK_KEYS}
        self.st.update({k: np.zeros((steps, n_rows), dtype=bool) for k in self.BOOL_KEYS})
        self.sc = {k: np.full(steps, np.nan, dtype=np.float32) for k in self.SCALAR_KEYS}
        # Per-microbatch log (2026-10-23, burst forensics): flow / align loss of each microbatch and
        # the sampled t + index position of each row. accum = 0 keeps the pre-2026-10-23 layout.
        self.mb: dict[str, np.ndarray] = {}
        self.batch = int(batch)
        if accum:
            self.mb = {
                "micro_flow": np.full((steps, accum), np.nan, dtype=np.float32),
                "micro_align": np.full((steps, accum), np.nan, dtype=np.float32),
                "micro_t": np.full((steps, accum * batch), np.nan, dtype=np.float32),
                "micro_row": np.full((steps, accum * batch), -1, dtype=np.int64),
            }

    def state(self) -> dict:
        return {"st": self.st, "sc": self.sc, "mb": self.mb}

    @classmethod
    def from_state(cls, state: dict, steps: int, n_rows: int, accum: int = 0, batch: int = 0) -> AlignStats:
        s = cls(steps, n_rows, accum, batch)
        for k in s.st:
            if state["st"][k].shape != s.st[k].shape:
                raise ValueError(f"checkpoint stats {k}: shape {state['st'][k].shape} != {s.st[k].shape}")
        s.st, s.sc = state["st"], state["sc"]
        old_mb = state.get("mb") or {}
        for k in s.mb:  # a checkpoint from before the micro log keeps the fresh (empty) arrays
            if k in old_mb:
                if old_mb[k].shape != s.mb[k].shape:
                    raise ValueError(f"checkpoint stats {k}: shape {old_mb[k].shape} != {s.mb[k].shape}")
                s.mb[k] = old_mb[k]
        return s

    def log_micro(self, step: int, j: int, flow: float, align: float, t, rows) -> None:
        if not self.mb:
            return
        self.mb["micro_flow"][step, j] = flow
        self.mb["micro_align"][step, j] = align
        t = np.asarray(t, dtype=np.float32).ravel()
        rows = np.asarray(rows, dtype=np.int64).ravel()
        o = (
            j * self.batch
        )  # slot of microbatch j at the CONFIGURED batch size (a short last batch leaves NaN / -1)
        if t.shape[0] > self.batch or rows.shape[0] > self.batch:
            raise ValueError(f"microbatch of {t.shape[0]} rows > batch size {self.batch}")
        self.mb["micro_t"][step, o : o + t.shape[0]] = t
        self.mb["micro_row"][step, o : o + rows.shape[0]] = rows

    def npz_fields(self) -> dict:
        return {
            **{f"step/{k}": v for k, v in self.st.items()},
            **{f"step/{k}": v for k, v in self.sc.items()},
            **{f"step/{k}": v for k, v in self.mb.items()},
        }


class AlignState:
    """Everything the alignment adds to the training loop, owned by the task:
    accumulate(flow_loss, align_loss) per microbatch -> apply(stats, step) at the optimizer boundary
    (writes the combined fp32 gradient into the LoRA .grad) -> after_step(...) once the optimizer
    stepped (update cosine, post-step gate, projector step)."""

    def __init__(
        self,
        flux: torch.nn.Module,
        cfg: AlignConfig,
        k: int,
        device,
        target: AlignTarget | None = None,
        discard: list[int] = (154, 1446),
    ):
        if cfg.arm not in ALIGN_ARMS:
            raise ValueError(f"align.arm must be one of {ALIGN_ARMS}, got {cfg.arm!r}")
        if cfg.feat not in ("ditf", "ln", "raw"):
            raise ValueError(f"align.feat must be 'ditf', 'ln' or 'raw', got {cfg.feat!r}")
        if cfg.target_shuffle not in ("", "batch"):
            raise ValueError(f"align.target_shuffle must be '' or 'batch', got {cfg.target_shuffle!r}")
        self.discard = [int(c) for c in discard]
        if cfg.proj_scope not in ("block", "global"):
            raise ValueError(f"align.proj_scope must be 'block' or 'global', got {cfg.proj_scope!r}")
        self.cfg, self.k, self.device = cfg, k, torch.device(device)
        self.groups = GP.lora_groups(flux)
        self.names = [n for ps in self.groups.values() for n, _ in ps]
        self.params = [p for ps in self.groups.values() for _, p in ps]
        self.blocks = list(self.groups)
        if len(self.blocks) != NUM_BLOCKS:
            raise RuntimeError(f"LoRA covers {len(self.blocks)} blocks, expected {NUM_BLOCKS}")
        self.target = target if target is not None else build_target(cfg.target, self.device)
        self.projector = Projector(flux.hidden_size, cfg.proj_hidden, self.target.dim, cfg.proj_arch).to(
            self.device
        )
        self.proj_names = [n for n, _ in self.projector.named_parameters()]
        self.proj_params = list(self.projector.parameters())
        self.opt_proj = torch.optim.AdamW(self.proj_params, lr=cfg.proj_lr, weight_decay=0.0)
        self.w = 0.0 if cfg.arm == "fm" else cfg.weight
        self.w_fm = 0.0 if cfg.arm == "sem" else 1.0  # the sem arm: g_FM measured, never applied
        self.n_rows = len(self.blocks) + 1  # + global row
        self._clear()
        self._acc_fm_applied: OrderedDict | None = None

    # ---- losses
    @torch.no_grad()
    def target_patches(self, imgs: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """(B, n_tokens, D) frozen target tokens of the clean images (in [-1, 1])."""
        return match_grid(self.target.patches(imgs), n_tokens)

    def features(self, feat_tokens: torch.Tensor, mod=None) -> torch.Tensor:
        """What the projector reads (AlignConfig.feat). ditf: per token, the massive-activation
        channels zeroed, LayerNorm (eps 1e-6, no affine), then the block's adaLN
        (1 + scale) * x + shift -- FluxModel.extract's definition. ln: the same without the
        scale/shift (ablation: massive-channel removal alone). mod = the block's ModulationOut
        (up_ft[1]; shift/scale (B, 1, C)). raw: the block input tokens unchanged."""
        x = feat_tokens.float()
        if self.cfg.feat == "raw":
            return x
        if self.discard:
            x = x.index_fill(-1, torch.tensor(self.discard, device=x.device), 0.0)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        if self.cfg.feat == "ln":
            return x
        if mod is None:
            raise ValueError("align.feat='ditf' needs the block's modulation (up_ft[1])")
        return (1 + mod.scale.float()) * x + mod.shift.float()

    def align_loss(
        self,
        feat_tokens: torch.Tensor,
        imgs: torch.Tensor,
        mod=None,
        w_img: torch.Tensor | None = None,
        return_unweighted: bool = False,
    ):
        """Negative mean cosine between projector(features(block-k tokens)) and the target tokens.
        w_img (B,) weights each image's mean cosine (align.sem_t_weights in the training
        microbatch, already normalised by the caller so that E[w] = 1 over the t distribution:
        0 drops the image from the semantic gradient; the batch mean keeps the divisor B, so a
        batch of ones equals the unweighted loss). None = unweighted (every measurement /
        validation caller). return_unweighted=True also returns the UNWEIGHTED -cos.mean() (the
        quantity logged as train/align_loss, comparable across arms and with val/align_loss_mean;
        review 2026-10-26).
        align.target_shuffle='batch' (the shuffled-teacher control): the targets of the batch
        rolled by one, so row b is aligned to the target of row b-1 -- an unrelated scene with
        the same target statistics. Every caller (training, projector warmup, validation, the
        gradient measurement) goes through here, so the control is shuffled everywhere."""
        z = self.target_patches(imgs, feat_tokens.shape[1])
        if self.cfg.target_shuffle == "batch":
            if z.shape[0] < 2:
                raise ValueError(
                    "align.target_shuffle='batch' needs batches of >= 2 images (a 1-row batch would align to itself)"
                )
            z = z.roll(1, dims=0)
        cos = F.cosine_similarity(
            self.projector(self.features(feat_tokens, mod)), z.float(), dim=-1
        )  # (B, T)
        unweighted = -cos.mean()
        if w_img is None:
            return (unweighted, unweighted) if return_unweighted else unweighted
        if w_img.shape != (cos.shape[0],):
            raise ValueError(f"w_img must be (B,) = ({cos.shape[0]},), got {tuple(w_img.shape)}")
        weighted = -(cos.mean(dim=1) * w_img.to(cos.dtype)).sum() / cos.shape[0]
        return (weighted, unweighted) if return_unweighted else weighted

    # ---- gradients
    def two_grads(self, flow_loss: torch.Tensor, align_loss: torch.Tensor):
        """Separate LoRA gradients of each loss from ONE graph; projector grads of the SEM loss."""
        g_fm = torch.autograd.grad(flow_loss, self.params, retain_graph=True, allow_unused=True)
        g_al = torch.autograd.grad(align_loss, self.params + self.proj_params, allow_unused=True)
        d_fm = dict(zip(self.names, g_fm))
        d_al = dict(zip(self.names, g_al[: len(self.params)]))
        d_pr = dict(zip(self.proj_names, g_al[len(self.params) :]))
        return d_fm, d_al, d_pr

    def _clear(self):
        self.acc_fm: OrderedDict | None = None
        self.acc_al: OrderedDict | None = None
        self.acc_pr: dict[str, torch.Tensor | None] = {n: None for n in self.proj_names}
        self.n_micro = 0

    def accumulate(self, flow_loss: torch.Tensor, align_loss: torch.Tensor) -> None:
        """One microbatch: the caller has already divided both losses by the accumulation count."""
        d_fm, d_al, d_pr = self.two_grads(flow_loss, align_loss)
        f_fm, f_al = GP.flatten(self.groups, d_fm), GP.flatten(self.groups, d_al)
        if self.acc_fm is None:
            self.acc_fm, self.acc_al = f_fm, f_al
        else:
            for b in self.blocks:
                self.acc_fm[b] += f_fm[b]
                self.acc_al[b] += f_al[b]
        for n in self.proj_names:
            g = d_pr[n]
            if g is not None:
                g = g.detach().float()
                self.acc_pr[n] = g if self.acc_pr[n] is None else self.acc_pr[n] + g
        self.n_micro += 1

    def apply(self, stats: AlignStats, step: int) -> dict:
        """Combine the accumulated gradients by the arm's rule, verify, and write the result into
        the fp32 LoRA .grad (no dtype rounding between the gate and the optimizer). Raises
        RuntimeError on a non-finite gradient or a verifier failure; the task turns that into
        the FAIL file. Returns the printable summary of this step."""
        if self.acc_fm is None:
            raise RuntimeError("apply() before any accumulate()")
        a = self.cfg
        for b in self.blocks:
            if not (torch.isfinite(self.acc_fm[b]).all() and torch.isfinite(self.acc_al[b]).all()):
                raise RuntimeError(f"non-finite gradient in block {b} at step {step + 1}")
        s = GP.stats(self.acc_fm, self.acc_al)
        st = stats.st
        st["cos_pre"][step], st["norm_fm"][step] = s["cos"].numpy(), s["norm_a"].numpy()
        st["norm_sem"][step], st["dot"][step] = s["norm_b"].numpy(), s["dot"].numpy()
        st["conflict"][step] = (s["dot"] < 0).numpy()
        nb = len(self.blocks)
        if a.arm == "proj":
            g_use, hits = GP.project_one_sided(self.acc_al, self.acc_fm, scope=a.proj_scope)
            ok, vcos, units = GP.verify_nonneg(g_use, self.acc_fm, tol=a.proj_tol, scope=a.proj_scope)
            if a.proj_scope == "block":
                st["projected"][step, :nb] = hits.numpy()
                st["cos_used"][step, :nb] = vcos.numpy()
            else:
                st["projected"][step, -1] = bool(hits[0])
                st["cos_used"][step, -1] = float(vcos[0])
            if not ok:
                bad = [(u, float(c)) for u, c in zip(units, vcos) if c < -a.proj_tol]
                raise RuntimeError(f"PROJECTION VERIFY FAIL at step {step + 1}: {bad[:5]} (tol {a.proj_tol})")
        else:
            g_use = self.acc_al
            st["cos_used"][step] = s["cos"].numpy()
        # align.stop_step (branching): from that optimizer step on the semantic part is dropped.
        w = 0.0 if (a.stop_step and step >= a.stop_step) else self.w
        stats.sc["w_eff"][step] = w
        combined = OrderedDict((b, self.w_fm * self.acc_fm[b] + w * g_use[b]) for b in self.blocks)
        st["cos_post"][step] = GP.stats(combined, self.acc_fm)["cos"].numpy()
        upd = GP.unflatten(self.groups, combined)
        for n, p in zip(self.names, self.params):
            p.grad = upd[n].to(device=p.device, dtype=torch.float32)
        applied = GP.flatten(self.groups, {n: p.grad for n, p in zip(self.names, self.params)})
        for b in self.blocks:
            applied[b] -= self.w_fm * self.acc_fm[b]
        if w > 0:
            # Executed re-check on the gradient the optimizer consumes (fp32, so this equals the
            # fp32 verifier up to the subtraction's roundoff); recorded, fatal in the proj arm.
            # A unit whose applied semantic part is below applied_tol x |g_FM| is roundoff (a fully
            # projected block leaves a ~1e-7 residual of random direction): counted as zero.
            for b in self.blocks:
                if float(applied[b].norm()) <= a.applied_tol * float(self.acc_fm[b].norm()):
                    applied[b] = torch.zeros_like(applied[b])
            ok_app, acos, _ = GP.verify_nonneg(applied, self.acc_fm, tol=a.applied_tol, scope=a.proj_scope)
            stats.sc["applied_cos_min"][step] = float(acos.min())
            if a.arm == "proj" and not ok_app:
                raise RuntimeError(
                    f"APPLIED VERIFY FAIL at step {step + 1}: min cos {float(acos.min()):.3e} < -{a.applied_tol}"
                )
        self._acc_fm_applied = self.acc_fm
        reach = s["cos"][: self.k]
        r = reach[~torch.isnan(reach)]
        return {
            "cos_lt_k_mean": float(r.mean()) if r.numel() else float("nan"),
            "frac_neg": float((r < 0).float().mean()) if r.numel() else float("nan"),
            "global_cos": float(s["cos"][-1]),
            "nproj": int(st["projected"][step].sum()),
        }

    def pre_step_gate(self, opt: torch.optim.Optimizer, step: int) -> None:
        GP.check_fp32_params(self.params, opt, check_grads=True, where=f"step {step + 1} pre-step")

    def after_step(
        self, stats: AlignStats, step: int, before: list[torch.Tensor], opt: torch.optim.Optimizer
    ) -> None:
        """After the LoRA optimizer stepped: the honest 'did the update oppose FM' number (AdamW's
        momentum and scaling make the step a different vector from the raw gradient; its cosine
        with g_FM over all blocks is recorded), the post-step fp32 gate, and the projector step
        on its accumulated (unprojected) gradient."""
        GP.check_fp32_params(self.params, opt, where=f"step {step + 1} post-step")
        delta = GP.flatten(
            self.groups, {n: p.detach() - b0 for n, p, b0 in zip(self.names, self.params, before)}
        )
        stats.sc["cos_update"][step] = float(GP.stats(delta, self._acc_fm_applied)["cos"][-1])
        for n, p in self.projector.named_parameters():
            p.grad = self.acc_pr[n]
        self.opt_proj.step()
        self.opt_proj.zero_grad(set_to_none=True)
        self._clear()
        self._acc_fm_applied = None

    # ---- passes
    def measure(self, batches, losses_fn, meas_t: list[int], seed: int, tag: str) -> dict:
        """Per-block cos(g_FM, g_SEM) at fixed timesteps on fixed batches. `batches` is a list of
        (imgs, latents) with posterior-MEAN latents; losses_fn(imgs, latents, t, noise) ->
        (flow_loss, align_loss) with gradients. Returns {t: {"cos", "norm_fm", "norm_sem": (n_batches,
        rows) arrays, "agg_cos": (rows,)}}."""
        t0 = time.time()
        out = {}
        for t_nom in meas_t:
            cos_rows, nfm_rows, nal_rows = [], [], []
            acc_fm, acc_al = None, None
            e_gen = torch.Generator(device=self.device).manual_seed(seed * 1000 + t_nom)
            for imgs, latents in batches:
                t = torch.full((imgs.shape[0],), t_nom / 1000.0, device=self.device, dtype=torch.float32)
                noise = torch.randn(latents.shape, generator=e_gen, device=self.device, dtype=latents.dtype)
                fl, al = losses_fn(imgs, latents, t, noise)
                d_fm, d_al, _ = self.two_grads(fl, al)
                f_fm, f_al = GP.flatten(self.groups, d_fm), GP.flatten(self.groups, d_al)
                s = GP.stats(f_fm, f_al)
                cos_rows.append(s["cos"])
                nfm_rows.append(s["norm_a"])
                nal_rows.append(s["norm_b"])
                if acc_fm is None:
                    acc_fm, acc_al = f_fm, f_al
                else:
                    for b in self.blocks:
                        acc_fm[b] += f_fm[b]
                        acc_al[b] += f_al[b]
            if acc_fm is None:
                raise RuntimeError("measure(): no batches")
            s_agg = GP.stats(acc_fm, acc_al)
            out[t_nom] = {
                "cos": torch.stack(cos_rows).numpy(),
                "norm_fm": torch.stack(nfm_rows).numpy(),
                "norm_sem": torch.stack(nal_rows).numpy(),
                "agg_cos": s_agg["cos"].numpy(),
            }
            reach = s_agg["cos"][: self.k]
            r = reach[~torch.isnan(reach)]
            print(
                f"[measure:{tag}] t={t_nom}: aggregated cos over blocks<k: mean={float(r.mean()):.3f} "
                f"min={float(r.min()):.3f} frac<0={float((r < 0).float().mean()):.2f} "
                f"global={float(s_agg['cos'][-1]):.3f}",
                flush=True,
            )
        print(f"[measure:{tag}] done in {time.time() - t0:.0f}s", flush=True)
        return out

    def warmup(self, microbatches) -> None:
        """Projector-only phase, LoRA frozen, identical in every arm. `microbatches` yields
        (feat_tokens, imgs) with feat_tokens computed under no_grad."""
        self.projector.train()
        t0 = time.time()
        n = 0
        for i, item in enumerate(microbatches):
            feat, imgs, *rest = item  # (feat, imgs) or (feat, imgs, mod)
            loss = self.align_loss(feat, imgs, rest[0] if rest else None)
            self.opt_proj.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_proj.step()
            n = i + 1
            if i % 20 == 0:
                print(f"[proj-warmup] {n} align_loss={loss.item():.4f}", flush=True)
        self.opt_proj.zero_grad(set_to_none=True)
        print(f"[proj-warmup] {n} microbatches in {time.time() - t0:.0f}s", flush=True)

    # ---- state
    def state_dict(self) -> dict:
        return {"projector": self.projector.state_dict(), "opt_proj": self.opt_proj.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.projector.load_state_dict(d["projector"])
        self.opt_proj.load_state_dict(d["opt_proj"])

    def describe(self) -> dict:
        return {
            "arm": self.cfg.arm,
            "feat": self.cfg.feat,
            "discard_channels": self.discard if self.cfg.feat in ("ditf", "ln") else [],
            "t_weights": self.cfg.t_weights,
            "sem_t_weights": self.cfg.sem_t_weights,
            "weight": self.w,
            "fm_weight": self.w_fm,
            "stop_step": self.cfg.stop_step,
            "blocks": self.blocks,
            "k": self.k,
            "target": self.target.describe(),
            "target_shuffle": self.cfg.target_shuffle,
            "projector": {"arch": self.projector.arch, "params": sum(p.numel() for p in self.proj_params)},
            "lora_params": sum(p.numel() for p in self.params),
        }


def resume_identity(cfg, train_data: dict | None) -> dict:
    """Everything that fixes WHICH rows each step consumes and what the optimizer state means.
    A resume with any of these changed would silently re-position the sampler (consumed =
    step * accumulation microbatches of batch_size rows) or load an optimizer state for another
    schedule, so the checkpoint stores this record and a resume must match it exactly."""
    accum = cfg.gradient_accumulation_steps if cfg.use_gradient_accumulation else 1
    return {
        "run_name": cfg.make_run_name(),
        "dataset": {"name": cfg.dataset.name, "path": cfg.dataset.path},
        "git10m": cfg.git10m.identity(),
        "align": cfg.align.identity(),
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "grad_accum": accum,
        "max_train_steps": cfg.max_train_steps,
        "warmup_steps": cfg.warmup_steps,
        "finetune_lr": cfg.finetune_lr,
        "lora": {
            "rank": cfg.lora_rank,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "wrap_output": cfg.wrap_output,
        },
        "k": cfg.k,
        "t": cfg.t,
        "img_size": list(cfg.img_size),
        "max_samples": cfg.max_samples,
        "train_data": train_data,
    }
