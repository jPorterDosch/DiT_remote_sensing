from utils import env_value
import gc
from abc import ABC, abstractmethod

import torch
from einops import rearrange, repeat

from config_types import (
    NUM_DOUBLE_BLOCKS,
    validate_inversion_block,
)
from config_types import map_timesteps_to_grid as _map_timesteps_to_grid

from .util import load_ae, load_flow_model


def prepare_txt(bs, t5, clip, prompt, device="cuda"):
    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return txt.to(device), txt_ids.to(device), vec.to(device)


def prepare(img):
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return img, img_ids.to(img.device)


# Refactor this as base class for Featurizer4Eval. Code copied from old repo deleted to avoid confusion.
class Featurizer(ABC):
    def __init__(
        self,
        name: str = "flux-dev",
        null_prompt: str = "",
        device: str = "cuda",
        null_embed_path: str
        | None = "./src/models/flux/null_embeddings.pt",  # TODO: cwd should be project root, but this could be made more robust.
    ):
        model = load_flow_model(name, device=device)
        ae = load_ae(name, device=device)
        self.model = model
        self.ae = ae

        if null_embed_path is None:
            raise ValueError(
                "null_embed_path must be provided -- for memory savings, t5 and clip are not supported currently."
            )
            # t5 = load_t5(device, max_length=512)
            # clip = load_clip(device)
            # self.t5 = t5
            # self.clip = clip
            # self.null_prompt_embeds = None
        else:
            self.t5 = None
            self.clip = None
            null_embed = torch.load(null_embed_path, weights_only=True)
            self.null_prompt_embeds = null_embed["prompt_embeds"].to(device)
            self.text_ids = null_embed["text_ids"].to(device)
            self.vec = null_embed["vec"].to(device)

        self.null_prompt = null_prompt

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Featurizer4Eval(Featurizer):
    def __init__(
        self,
        flux_id="flux-dev",
        null_prompt="",
        cat_list=None,
        ensemble_size=1,
    ):
        super().__init__(name=flux_id, null_prompt=null_prompt)

        if cat_list is None:
            cat_list = []

        with torch.no_grad():
            cat2prompt_embeds = {}
            for cat in cat_list:
                prompt = f"a photo of a {cat}"

                # Only run text encoder if null embeddings are not provided.
                if self.null_prompt_embeds is not None:
                    prompt_embeds = self.null_prompt_embeds
                    text_ids = self.text_ids
                    vec = self.vec
                else:
                    prompt_embeds, text_ids, vec = prepare_txt(
                        bs=ensemble_size, t5=self.t5, clip=self.clip, prompt=prompt
                    )
                cat2prompt_embeds[cat] = (prompt_embeds, text_ids, vec)
            self.cat2prompt_embeds = cat2prompt_embeds

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
        self,
        args,
        img_tensor,
        caption="",
        category="image",
        timestep=261,
        block_idx=1,
        ensemble_size=1,
        guidance=3.5,
        latents=None,
        noise=None,
        generator=None,
    ):
        """One-shot noising to `timestep` and a single DiT forward pass (per ensemble member).

        `latents`/`noise` (each (ensemble_size, c, h, w)) let a caller reuse the exact clean
        latents and eps across multiple calls at different timesteps for the same image, so
        only t varies between calls. When None, they are computed/drawn here (from `generator`
        if given, else the global RNG). Returns (feat, mod, latents, noise) where latents/noise
        are the tensors actually mixed into the noisy input, so callers can assert constancy.
        """
        if img_tensor.dim() != 3:
            raise ValueError(
                f"Expected img_tensor to have 3 dimensions (C, H, W), but got {img_tensor.shape}. If passing batched images, refactor this check, and make sure that this does not cause OOM."
            )

        img_tensor = img_tensor.unsqueeze(0).cuda()  # 1, c, h, w

        ############ caption "a photo of a {cat}" #####################
        # prompt_embeds, text_ids, vec = self.cat2prompt_embeds[category]
        if self.null_prompt_embeds is None:
            # detailed caption generated by pretrained MLLM. Bring about 0.3% gain for flux
            prompt_embeds, text_ids, vec = prepare_txt(
                bs=ensemble_size, t5=self.t5, clip=self.clip, prompt=caption
            )
        # Should save significant memory
        else:
            prompt_embeds, text_ids, vec = (
                self.null_prompt_embeds,
                self.text_ids,
                self.vec,
            )

        if prompt_embeds.shape[0] == 1 and ensemble_size > 1:
            prompt_embeds = repeat(prompt_embeds, "1 ... -> bs ...", bs=ensemble_size)
        if text_ids.shape[0] == 1 and ensemble_size > 1:
            text_ids = repeat(text_ids, "1 ... -> bs ...", bs=ensemble_size)
        if vec.shape[0] == 1 and ensemble_size > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=ensemble_size)

        prompt_embeds = prompt_embeds.to(img_tensor.device)
        text_ids = text_ids.to(img_tensor.device)
        vec = vec.to(img_tensor.device)

        device = img_tensor.device
        t = timestep / 1000

        dit_feats = []
        mods = []
        latents_used = []
        noise_used = []

        block_indices = [block_idx] if isinstance(block_idx, int) else block_idx

        # Sequential to avoid OOM.
        for i in range(ensemble_size):
            if latents is not None:
                latents_i = latents[i : i + 1]
            else:
                # Encode inside the loop (not batched upfront) to preserve the exact global-RNG
                # draw order of the original single-timestep implementation.
                latents_i = self.ae.encode(img_tensor).to(torch.bfloat16)

            if noise is not None:
                noise_i = noise[i : i + 1]
            elif generator is not None:
                noise_i = torch.randn(
                    latents_i.shape, generator=generator, device=device, dtype=latents_i.dtype
                )
            else:
                noise_i = torch.randn_like(latents_i).to(device)

            # add noise
            latents_noisy = t * noise_i + (1.0 - t) * latents_i
            _, c, h, w = latents_noisy.shape

            img, img_ids = prepare(img=latents_noisy)

            # FIXED-CONDITIONING CONTROL. Along the t axis the one-shot arm varies TWO
            # things at once: the input x_t AND the extractor itself (this t_vec conditions
            # the adaLN shift/scale/gate of every block). The raw-x_t baseline varies only
            # the input, so "features decay more than the raw baseline" cannot distinguish
            # "the protocol destroyed information" from "the t-conditioned features are
            # simply worse at high t". FIXED_COND_T=<timestep in [1,1000]> pins the
            # conditioning at one value while the noising t still sweeps, separating the
            # two for the first time. Env override (not a config field) mirrors
            # FLUX_RANDOM_INIT; provenance is stamped by the extraction task.
            _fc = env_value("FIXED_COND_T")
            t_cond = (int(_fc) / 1000) if _fc else t
            t_vec = torch.full((img.shape[0],), t_cond, dtype=img.dtype, device=img.device)
            guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

            prompt_embeds_i = prompt_embeds[i : i + 1]
            text_ids_i = text_ids[i : i + 1]
            vec_i = vec[i : i + 1]

            model_output = self.model.forward_feat(
                img=img,
                img_ids=img_ids,
                txt=prompt_embeds_i,
                txt_ids=text_ids_i,
                y=vec_i,
                timesteps=t_vec,
                ft_indices=block_indices,
                cat=category,
                guidance=guidance_vec,
            )

            mod = model_output[1]
            dit_feat = model_output[0]
            dit_feat = rearrange(dit_feat, "b (h w) c -> b h w c", h=h // 2, w=w // 2)
            dit_feat = dit_feat.permute(0, 3, 1, 2)

            mod = torch.cat([mod.shift, mod.scale, mod.gate], dim=1)
            dit_feats.append(dit_feat)
            mods.append(mod)
            # Record the tensors actually mixed into latents_noisy (not the caller's inputs),
            # so a downstream constancy assertion catches any reintroduced per-call resampling.
            latents_used.append(latents_i)
            noise_used.append(noise_i)

            del (
                latents_noisy,
                img,
                img_ids,
                t_vec,
                guidance_vec,
                model_output,
                mod,
                dit_feat,
            )
            torch.cuda.empty_cache()

        dit_feat = torch.cat(dit_feats, dim=0).mean(0, keepdim=True)  # 1, c, h, w
        mod = torch.cat(mods, dim=0).mean(0, keepdim=True)  # 1, c, h, w

        return dit_feat, mod, torch.cat(latents_used, dim=0), torch.cat(noise_used, dim=0)

    def _null_text_inputs(self, device):
        """Null-prompt conditioning tensors (the extractor's standard conditioning)."""
        return (
            self.null_prompt_embeds.to(device),
            self.text_ids.to(device),
            self.vec.to(device),
        )

    def _velocity(self, x, img_ids, t, guidance, txt, txt_ids, vec):
        """One velocity evaluation v(x_t, t) in packed-latent space."""
        t_vec = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
        guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)
        return self.model(
            img=x,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

    def _ode_step(self, x, img_ids, pred, t_curr, h_step, order, guidance, txt, txt_ids, vec, capture=None):
        """One integrator step from t_curr to t_curr + h_step, given pred = v(x, t_curr).

        order 2 is the RF-Solver second-order step (Wang et al., arXiv:2411.04746);
        order 1 is naive Euler. Shared by invert_chain (h_step > 0) and generate_chain
        (h_step < 0) so the two directions cannot drift numerically apart — the
        round-trip validation depends on them using the identical update.
        """
        if order == 2:
            # RF-Solver's 2nd-order Taylor term 0.5*h^2*(pred_mid-pred)/(h/2) cancels
            # algebraically to h*(pred_mid-pred), so the whole step is the explicit
            # midpoint update x + h*v(x_mid, t+h/2) (cf. REPORT / arXiv:2411.04746). We
            # apply that directly: identical in exact arithmetic, and free of the bf16
            # cancellation from differencing two near-equal velocities then rescaling.
            x_mid = x + (h_step / 2) * pred
            pred_mid = self._velocity(x_mid, img_ids, t_curr + h_step / 2, guidance, txt, txt_ids, vec)
            if capture is not None:
                # SOLVER CURVATURE. (pred_mid - pred)/(h/2) is the local dv/dt along the
                # path -- the deviation from rectification, i.e. where crossing training
                # paths pull the field. Unlike state increments/curvature, pred_mid is a
                # NONLINEAR function of the state through the full network, so it lies
                # OUTSIDE the cached-state linear span -- the first admissible instrument
                # for the wall hypothesis (see RESEARCH_NOTES 6g). Both velocities are
                # stashed in fp32 and differenced OFFLINE, per the standing bf16 rule.
                capture["pred"] = pred.detach().float().clone()
                capture["pred_mid"] = pred_mid.detach().float().clone()
            return x + h_step * pred_mid
        return x + h_step * pred

    @staticmethod
    def map_timesteps_to_grid(cache_timesteps, num_inversion_steps):
        """Map nominal timesteps ([1, 1000], the one-shot convention t/1000) onto the
        uniform integration grid t_i = i / num_inversion_steps.

        The cached feature timesteps are a SUBSAMPLE of the integration grid (integration
        granularity and cache timesteps are separate knobs — cf. Diffusion Hyperfeatures,
        which integrates 50 steps and subsamples 11 for features). Each requested timestep
        must land exactly on a grid point; a mismatch means features would silently be
        extracted at a different t than the label claims, corrupting timestep sweeps.

        Returns {nominal_timestep: grid_index}, in ascending nominal order. Thin wrapper
        over config_types.map_timesteps_to_grid, the single source of truth shared with
        run.py's config validation.
        """
        return _map_timesteps_to_grid(cache_timesteps, num_inversion_steps)

    @torch.no_grad()
    def invert_chain(
        self,
        img_tensor,
        cache_timesteps,
        num_inversion_steps,
        block_idx,
        guidance=3.5,
        order=2,
        t_stop=None,
        want_velocity: bool = False,
        want_states: bool = False,
        want_curvature: bool = False,
    ):
        """Invert a clean image toward noise along the reverse generative ODE, caching
        block hidden states at the requested timesteps.

        Follows RF-Solver (Wang et al., ICML 2025, arXiv:2411.04746): a second-order
        Taylor expansion of the rectified-flow ODE, with the velocity derivative
        estimated by a half-step finite difference (two model evaluations per step).
        Unlike one-shot noising, each state z_{t_{i+1}} depends on z_{t_i}, producing a
        chained trajectory: z_{t_{i+1}} = z_{t_i} + h*v(z_{t_i}, t_i)
                                          + 0.5*h^2 * (v(z_mid, t_i + h/2) - v(z_{t_i}, t_i)) / (h/2).

        The chain draws no eps of its own, so ensemble_size does not apply. NOTE:
        ae.encode SAMPLES the VAE posterior from the global RNG (repo-wide convention),
        so the chain — and every cached feature — is deterministic only given the
        sampled clean latents, i.e. reproducible under identical global RNG state,
        not unconditionally "deterministic given the image".

        Args:
            img_tensor: (C, H, W) image in [-1, 1] (dataset convention).
            cache_timesteps: nominal timesteps ([1, 1000]) at which to cache features.
                Must lie on the integration grid (see map_timesteps_to_grid).
            num_inversion_steps: integration granularity of the reverse ODE — a separate
                knob from cache_timesteps.
            block_idx: single-stream DiT block to cache hidden states from ([19, 56]
                for FLUX). Double blocks (0-18) do not expose the adaLN mod triple the
                cache stores, and multi-block caching is not implemented.
            guidance: guidance strength embedded by the guidance-distilled model; also
                used for every velocity evaluation of the chain.
            order: 2 = RF-Solver second-order step (default), 1 = naive Euler (for
                ablation only; known-unreliable for rectified flows).
            t_stop: nominal timestep to integrate up to (defaults to max(cache_timesteps)).
                Pass 1000 to invert fully to t=1 (e.g. for round-trip validation).

        Returns dict with:
            feats: {nominal_t: (1, C, h, w)} block hidden states, pre-normalization.
            mods:  {nominal_t: (1, 3, C)} adaLN [shift, scale, gate] rows.
            latents_clean: (1, c, h, w) unpacked clean VAE latents (t=0).
            z_final: (1, T, d) packed latent state at the stop timestep.
            img_ids: positional ids for z_final (needed to continue/reverse the chain).
            grid: {nominal_t: grid_index} mapping actually used.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 (Euler) or 2 (RF-Solver), got {order}")
        if img_tensor.dim() != 3:
            raise ValueError(f"Expected img_tensor of shape (C, H, W), got {tuple(img_tensor.shape)}")

        # Single source of truth for the block-range rule (shared with run.py). Check the
        # loaded model matches the assumed layout so a divergent architecture fails loudly
        # rather than silently validating against wrong bounds.
        if len(self.model.double_blocks) != NUM_DOUBLE_BLOCKS:
            raise RuntimeError(
                f"model has {len(self.model.double_blocks)} double blocks, config_types assumes "
                f"{NUM_DOUBLE_BLOCKS}"
            )
        block_indices = [validate_inversion_block(block_idx)]

        device = img_tensor.device if img_tensor.is_cuda else torch.device("cuda")
        img_tensor = img_tensor.unsqueeze(0).to(device)

        grid = self.map_timesteps_to_grid(cache_timesteps, num_inversion_steps)
        if t_stop is None:
            if not grid:
                raise ValueError("cache_timesteps is empty and t_stop is None — nothing to do")
            stop_idx = max(grid.values())
        else:
            stop_map = self.map_timesteps_to_grid([t_stop], num_inversion_steps)
            stop_idx = stop_map[t_stop]
            if grid and max(grid.values()) > stop_idx:
                raise ValueError("t_stop is below the largest cache timestep")

        txt, txt_ids, vec = self._null_text_inputs(device)

        latents_clean = self.ae.encode(img_tensor).to(torch.bfloat16)
        _, c, h, w = latents_clean.shape
        x, img_ids = prepare(img=latents_clean)

        idx2t = {gi: ct for ct, gi in grid.items()}

        feats: dict[int, torch.Tensor] = {}
        mods: dict[int, torch.Tensor] = {}
        vels: dict[int, torch.Tensor] = {}
        states: dict[int, torch.Tensor] = {}
        curvs: dict[int, dict] = {}
        n = num_inversion_steps
        h_step = 1.0 / n

        for i in range(stop_idx + 1):
            t_curr = i / n
            pred = None

            if i in idx2t:
                # Features live at exactly (z_i, t_i) — the first evaluation of the
                # RF-Solver step — so mid-chain caching adds no extra forward passes.
                t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
                guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)
                model_kwargs = dict(
                    img=x,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    ft_indices=block_indices,
                    guidance=guidance_vec,
                )
                if i < stop_idx or want_velocity:
                    # The velocity `pred` is the MODEL'S OUTPUT at (z_i, t_i) — where the
                    # learned prior says the sample should go — as opposed to the hidden
                    # state, which describes where it currently is.
                    #
                    # NOTE (measured, 2026-08-03): this does NOT escape the redundancy that
                    # makes trajectory features uninformative. v(x, t) = model(x, t) is a
                    # deterministic function of the state, and up to the order-2 correction
                    # it is just the latent difference (x_{t+h} = x_t + h*(v + corr)). So it
                    # carries no information the state lacks — only a different encoding.
                    # Empirically it adds nothing: (states + velocity) - states was a dead
                    # heat, 0/4 comparisons cleared. Cache it for diagnostics (path
                    # curvature, kinetic energy), not as a source of new signal.
                    #
                    # It is computed anyway at every non-terminal step, so caching is free
                    # except at the terminal step (see below).
                    pred, up_ft = self.model.forward_velocity_feat(**model_kwargs)
                else:
                    # Terminal step: no ODE step follows, so the velocity would be
                    # discarded — use the early-exiting feature pass instead (identical
                    # features; the blocks after block_idx are skipped).
                    up_ft = self.model.forward_feat(**model_kwargs)
                dit_feat = rearrange(up_ft[0], "b (h w) c -> b h w c", h=h // 2, w=w // 2)
                dit_feat = dit_feat.permute(0, 3, 1, 2)  # 1, C, h/2, w/2
                mod = up_ft[1]
                mod = torch.cat([mod.shift, mod.scale, mod.gate], dim=1)  # 1, 3, C
                feats[idx2t[i]] = dit_feat
                mods[idx2t[i]] = mod
                if want_states:
                    # The STATE z_i itself, (1, T, d) packed — the inversion analogue of the
                    # one-shot x_t. Free: it is the tensor already being fed to the model, so
                    # this is a clone, not a forward. Unlike one-shot x_t it has NO closed
                    # form and no eps — it is a deterministic function of x0 alone.
                    states[idx2t[i]] = x.detach().clone()
                if want_velocity and pred is not None:
                    # (1, T, d) packed latent velocity. T matches the feature token grid
                    # (both are h/2 x w/2), so velocity tokens and feature tokens index the
                    # same patches and the alignment result carries over unchanged.
                    vels[idx2t[i]] = pred.detach().clone()

            if i >= stop_idx:
                break

            if pred is None:
                pred = self._velocity(x, img_ids, t_curr, guidance, txt, txt_ids, vec)
            if want_curvature and order != 2:
                # Curvature is the 2nd-order midpoint term; order=1 never computes pred_mid,
                # so the capture dict would stay empty and the consumer would np.stack([]).
                raise ValueError("want_curvature requires order=2 (RF-Solver); got order=%d" % order)
            cap = {} if (want_curvature and i in idx2t) else None
            x = self._ode_step(
                x, img_ids, pred, t_curr, h_step, order, guidance, txt, txt_ids, vec, capture=cap
            )
            if cap:
                curvs[idx2t[i]] = cap

        return {
            "feats": feats,
            "mods": mods,
            "vels": vels,
            "states": states,
            "curvs": curvs,
            "latents_clean": latents_clean,
            "z_final": x,
            "img_ids": img_ids,
            "grid": grid,
        }

    @torch.no_grad()
    def generate_chain(
        self,
        z_packed,
        img_ids,
        t_start,
        num_inversion_steps,
        guidance=3.5,
        order=2,
    ):
        """Integrate the generative ODE from t_start back to t=0 (the reverse of
        invert_chain), with the same RF-Solver step and the same conditioning, so that
        invert_chain -> generate_chain forms a round trip.

        Args:
            z_packed: (1, T, d) packed latent state at nominal timestep t_start.
            img_ids: positional ids matching z_packed.
            t_start: nominal timestep in [1, 1000] the state currently sits at.
            num_inversion_steps / guidance / order: must match the inversion run.

        Returns the packed latent state at t=0 (unpack + ae.decode to get pixels).
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 (Euler) or 2 (RF-Solver), got {order}")

        device = z_packed.device
        txt, txt_ids, vec = self._null_text_inputs(device)

        start_idx = self.map_timesteps_to_grid([t_start], num_inversion_steps)[t_start]
        n = num_inversion_steps
        h_step = -1.0 / n  # stepping toward t=0

        x = z_packed
        for i in range(start_idx, 0, -1):
            t_curr = i / n
            pred = self._velocity(x, img_ids, t_curr, guidance, txt, txt_ids, vec)
            x = self._ode_step(x, img_ids, pred, t_curr, h_step, order, guidance, txt, txt_ids, vec)
        return x
