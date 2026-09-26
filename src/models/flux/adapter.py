from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from registry import register_model

from .feat_flux import Featurizer4Eval, prepare
from models.lora import LORA_SCOPE, lora_block_indices, lora_wrap_flux


@register_model("flux")
class FluxModel:
    def __init__(self, cfg, category_list: list[str]) -> None:
        self._cd = cfg.cd
        self._discard_channels = list(cfg.discard_channels)
        self._pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)
        self._inner = Featurizer4Eval(
            cat_list=list(category_list),
            ensemble_size=cfg.model.ensemble_size,
        )

        if getattr(cfg, "lora_checkpoint", ""):
            flux = self._inner.model

            # load all tensors onto cpu first regardless of where they were saved from
            ckpt = torch.load(cfg.lora_checkpoint, map_location="cpu", weights_only=False)
            sd = ckpt["lora_state_dict"]
            if not sd:
                # Freeze-mode runs save checkpoints with an EMPTY lora_state_dict (only
                # the probe head trains); without this check the key-set comparison below
                # fails with a misleading "wrong k/wrap_output at training?" message.
                raise ValueError(
                    f"{cfg.lora_checkpoint} has an empty lora_state_dict — this is a "
                    "freeze_backbone (probe-only) checkpoint; there is no adapter to load."
                )

            # Hyperparameters that leave NO trace in key names or tensor shapes must be
            # verified against the checkpoint's recorded training config: lora_alpha exists
            # only as LoRALinear.scaling, so an alpha mismatch loads cleanly and silently
            # applies the adapter at the wrong strength (2026-09-11 review, finding 5).
            # k/wrap_output/rank are also compared here for a clearer message than the
            # key-set/shape errors below would give.
            if ckpt.get("lora_scope") != LORA_SCOPE:
                raise ValueError(
                    f"{cfg.lora_checkpoint} has lora_scope={ckpt.get('lora_scope')!r}, expected "
                    f"{LORA_SCOPE!r}. Checkpoints trained before 2026-09-24 adapted only block k, "
                    "whose adapter cannot reach the probed features (block k's INPUT); retrain."
                )
            tcfg = ckpt.get("cfg", {}) or {}
            for field in ("lora_alpha", "lora_rank", "k", "wrap_output", "lora_dropout"):
                if field in tcfg and getattr(cfg, field) != tcfg[field]:
                    raise ValueError(
                        f"LoRA checkpoint was trained with {field}={tcfg[field]!r} but this "
                        f"config has {field}={getattr(cfg, field)!r} -- the adapter would "
                        f"load cleanly and behave differently from the trained model. "
                        f"Match the training config ({cfg.lora_checkpoint})."
                    )

            # Fence the RNG around wrapping: kaiming init inside LoRALinear consumes global
            # (CUDA) RNG draws that a frozen extraction never makes, desynchronizing every
            # later global-RNG consumer -- most importantly ae.encode's VAE posterior
            # sampling, which would give adapted-vs-frozen re-extractions DIFFERENT clean
            # latents per image (2026-09-11 review, finding 8). The init values are
            # irrelevant here (immediately overwritten by the checkpoint), so a fenced fork
            # keeps the downstream stream byte-identical to the frozen arm's.
            _dev = next(flux.parameters()).device
            with torch.random.fork_rng(devices=[_dev] if _dev.type == "cuda" else []):
                lora_wrap_flux(
                    flux,
                    lora_block_indices(flux),
                    cfg.lora_rank,
                    cfg.lora_alpha,
                    cfg.lora_dropout,
                    wrap_o=cfg.wrap_output,
                )
            # HARD verification (2026-09-11 review): strict=False only PRINTED mismatches,
            # so a wrap_output/k mismatch between the training and extraction configs
            # yielded a silently un- or partly-adapted model carrying an "adapted" run
            # name -- directly load-bearing for the frozen-vs-adapted comparison. Every
            # checkpoint LoRA key must land on exactly one wrapped parameter, and every
            # wrapped parameter must be covered by the checkpoint.
            lora_params = {n for n, _ in flux.named_parameters() if n.endswith((".A", ".B"))}
            ckpt_keys = set(sd.keys())
            if ckpt_keys != lora_params:
                raise ValueError(
                    f"LoRA checkpoint does not match the wrapped model.\n"
                    f"  in ckpt but not wrapped (wrong k/wrap_output at extraction?): "
                    f"{sorted(ckpt_keys - lora_params)[:4]}\n"
                    f"  wrapped but not in ckpt (wrong k/wrap_output at training?): "
                    f"{sorted(lora_params - ckpt_keys)[:4]}\n"
                    f"  ckpt={len(ckpt_keys)} keys, model={len(lora_params)} wrapped params. "
                    f"Check --k / --wrap-output / --lora-rank against the training config "
                    f"recorded in the checkpoint's 'cfg' field."
                )
            missing, unexpected = flux.load_state_dict(sd, strict=False)
            bad = [k for k in missing if k.endswith((".A", ".B"))] + list(unexpected)
            if bad:
                raise ValueError(f"LoRA load failed for keys: {bad[:6]}")
            print(f"Loaded LoRA checkpoint ({len(ckpt_keys)} adapter tensors): {cfg.lora_checkpoint}")

    @torch.no_grad()
    def extract_raw(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: int,
        ensemble_size: int,
        caption: str = "a photo of a image",
        category: str = "image",
        guidance: float = 3.5,
        latents: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-shot raw features, PRE-normalization (no channel discard / LayerNorm /
        adaLN / L2) — the multi-timestep extraction path applies those offline.

        Returns (feat_raw (1, C, h, w), ada (1, 3, C), latents, noise) where latents/
        noise are the tensors actually mixed into the noisy input (see Featurizer4Eval.
        forward), so callers can assert eps-constancy across timesteps.

        NOTE: called by tasks/utils.extract_features but missing from every branch —
        reconstructed as a thin delegation to Featurizer4Eval.forward, which already
        returns exactly this tuple.
        """
        return self._inner.forward(
            None,
            img,
            caption=caption,
            category=category,
            timestep=timestep,
            block_idx=block_idx,
            ensemble_size=ensemble_size,
            guidance=guidance,
            latents=latents,
            noise=noise,
            generator=generator,
        )

    @torch.no_grad()
    def extract_inversion(
        self,
        img: torch.Tensor,
        timesteps: list[int],
        num_inversion_steps: int,
        block_idx: int,
        guidance: float = 3.5,
        want_velocity: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Inversion-chain features: one RF-Solver reverse-ODE trajectory per image,
        caching pre-normalization block hidden states at each requested timestep.

        The chain draws no eps of its own (no ensemble), but ae.encode samples the
        VAE posterior from the global RNG, so features are reproducible only under
        identical RNG state (see Featurizer4Eval.invert_chain). Returns
        (feats (K, C, h, w), mods (K, 3, C), vels) in the order of `timesteps`, where
        vels is (K, T, d) packed-latent velocities when want_velocity else None.

        Velocity is the model's OUTPUT at each cached point rather than the state itself.
        It is already computed at every non-terminal step, so requesting it costs one extra
        forward only at the terminal timestep (which otherwise early-exits at block_idx).
        """
        out = self._inner.invert_chain(
            img,
            cache_timesteps=list(timesteps),
            num_inversion_steps=num_inversion_steps,
            block_idx=block_idx,
            guidance=guidance,
            want_velocity=want_velocity,
        )
        feats = torch.cat([out["feats"][t] for t in timesteps], dim=0)  # K, C, h, w
        mods = torch.cat([out["mods"][t] for t in timesteps], dim=0)  # K, 3, C
        vels = None
        if want_velocity:
            missing = [t for t in timesteps if t not in out["vels"]]
            if missing:
                raise RuntimeError(f"velocity missing at timesteps {missing} — chain did not cache it")
            vels = torch.cat([out["vels"][t] for t in timesteps], dim=0)  # K, T, d
        return feats, mods, vels

    @torch.no_grad()
    def roundtrip(
        self,
        img: torch.Tensor,
        t_stop: int,
        num_inversion_steps: int,
        block_idx: int,
        guidance: float = 3.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert to `t_stop` then integrate back to t=0. Returns (recovered, clean) packed
        latents, both (1, T, d).

        This measures how faithfully the chain can represent an image: the ODE is
        analytically reversible, so any discrepancy is accumulated DISCRETIZATION error
        from integrating a finite number of RF-Solver steps. Images the model finds hard
        (high-frequency content, out-of-distribution structure) should invert worse, and
        that error is what would degrade inversion features relative to one-shot noising —
        which is exact at every t by construction.
        """
        out = self._inner.invert_chain(
            img,
            cache_timesteps=[],
            num_inversion_steps=num_inversion_steps,
            block_idx=block_idx,
            guidance=guidance,
            t_stop=t_stop,
        )
        recovered = self._inner.generate_chain(
            out["z_final"],
            out["img_ids"],
            t_start=t_stop,
            num_inversion_steps=num_inversion_steps,
            guidance=guidance,
        )
        clean, _ = prepare(img=out["latents_clean"])
        return recovered, clean

    @torch.no_grad()
    def extract(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: int,
        ensemble_size: int,
        caption: str = "a photo of a image",
        category: str = "image",
        guidance: float = 3.5,
    ) -> torch.Tensor:
        # Featurizer4Eval.forward returns (feat, mod, latents, noise); the trailing
        # latents/noise are only needed by the multi-timestep eps-constancy checks.
        feat_raw, ada, _latents, _noise = self._inner.forward(
            None,
            img,
            caption=caption,
            category=category,
            timestep=timestep,
            block_idx=block_idx,
            ensemble_size=ensemble_size,
            guidance=guidance,
        )

        B, C, H, W = feat_raw.shape

        if self._cd:
            for ch in self._discard_channels:
                feat_raw[:, ch, :, :] = 0.0

        feat = rearrange(feat_raw, "b c h w -> b (h w) c")
        feat = self._pre_norm(feat)
        feat = rearrange(feat, "b (h w) c -> b c h w", h=H, w=W)

        shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return (1 + scale) * feat + shift  # B, C, H, W
