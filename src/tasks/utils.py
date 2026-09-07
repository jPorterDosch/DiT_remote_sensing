import time

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import seed_worker
from classifier_heads import FourierKANProbe, KANProbe, LinearProbe, MLPProbe
from config_types import ExtractionMode, ProbeType


def _flatten_scalars_into(prefix, values, out):
    for key, value in values.items():
        tag = f"{prefix}/{key}"
        if isinstance(value, dict):
            _flatten_scalars_into(tag, value, out)
        else:
            out[tag] = value


def log_scalars_recursive(prefix, values):
    flat = {}
    _flatten_scalars_into(prefix, values, flat)
    wandb.log(flat)


def _sink_tokens(sink: dict, timesteps: list[int], feats_bchw: torch.Tensor) -> None:
    """Stash PRE-POOL tokens for the subset of `timesteps` listed in sink["timesteps"].

    feats_bchw is (S, C, h, w) covering `timesteps` in order. Stored as (1, S_sel, L, C)
    with L = h*w, i.e. the same layout the pooled cache would have had before
    `.mean(dim=[2,3])` collapsed the spatial axes. Token order is row-major over (h, w)
    and is identical across timesteps, which is exactly the correspondence the
    alignment check exists to verify rather than assume.

    MEMORY CEILING: every image is retained here and concatenated only at the end, so host
    RAM peaks at ~2x the final array (the list plus the cat). At N=500, S=3, L=256, C=3072
    that is ~4.4 GB -> ~8.8 GB peak. It scales linearly in N*S*L, and the failure would
    land AFTER the full extraction, so raising any of those materially (a 7-timestep rider
    is ~10 GB -> ~20 GB) wants preallocating one array and filling it by index instead.
    """
    want = sink["timesteps"]
    sel = [i for i, t in enumerate(timesteps) if t in want]
    if not sel:
        return
    toks = feats_bchw[sel]  # S_sel, C, h, w
    s, c, h, w = toks.shape
    if sink.setdefault("hw", (h, w)) != (h, w):
        # Every image must land on the same token grid or the stacked (N, S, L, C) array is
        # meaningless. Cannot happen at fixed img_size, but failing here names the cause
        # instead of surfacing as a confusing reshape error at save time.
        raise RuntimeError(f"token grid changed mid-extraction: {sink['hw']} -> {(h, w)}")
    # (S_sel, C, h, w) -> (S_sel, L, C)
    toks = toks.reshape(s, c, h * w).transpose(1, 2).contiguous()
    # .cpu() BEFORE .float(): features are bf16 on device, so converting first would double
    # the bytes crossing PCIe on a per-image hot path.
    sink["feats"].append(toks.cpu().float().unsqueeze(0))  # 1, S_sel, L, C


@torch.inference_mode()
def extract_features(
    cfg,
    model,
    dataloader,
    split_name: str,
    token_sink: dict | None = None,
    vel_sink: list | None = None,
):
    """Extract and return features and labels for all images in dataloader.

    cfg.t is an int: single-timestep mode (unchanged behavior) — returns
    (feats (N, C), labels (N,)) with DiTF normalization applied and L2-normalized
    global-average-pooled vectors.

    cfg.t is a list of K timesteps: multi-timestep mode — one-shot noising to each t
    independently (no denoising chain), with the clean latents and eps drawn ONCE per
    image and reused verbatim across all K forward passes (only t varies). Features are
    global-average-pooled but PRE-normalization (no channel discard, LayerNorm, adaLN
    modulation, or L2 norm — those are applied offline in the probe so they can be
    toggled). Returns (feats (N, K, C), labels (N,), mods (K, 3, C)) where mods hold the
    per-timestep adaLN [shift, scale, gate] vectors needed to apply DiTF normalization
    offline; they depend only on t (null prompt embeds are constant), not on the image.

    cfg.extraction_mode == INVERSION (multi-timestep only): instead of independent
    one-shot noising, a single RF-Solver reverse-ODE chain per image (each state depends
    on the previous one; see Featurizer4Eval.invert_chain), with features cached at the
    K requested timesteps — which must lie on the cfg.num_inversion_steps integration
    grid. Output shapes are identical to one-shot multi-timestep ((N, K, C) pre-norm +
    (K, 3, C) mods), so cached features are drop-in interchangeable. The chain has no
    eps stream and ensemble_size does not apply; note ae.encode samples the VAE
    posterior from the global RNG, so inversion features are reproducible only under
    identical RNG state (unlike the oneshot eps stream, which is generator-isolated).

    cfg.guidance_scale feeds every model evaluation in ALL modes (oneshot passes it
    through extract_raw/extract; inversion through the chain), so oneshot-vs-inversion
    comparisons at a given guidance are apples-to-apples.
    """
    # TODO: change call sites to pass in underlying Flux model directly instead of wrapper,
    # which adds extra unnecessary calls to access underlying model.
    model._inner.model.eval()
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    device = torch.device(cfg.device)

    multi_timestep = isinstance(cfg.t, list)
    inversion = getattr(cfg, "extraction_mode", ExtractionMode.ONESHOT) == ExtractionMode.INVERSION
    if inversion and not multi_timestep:
        raise ValueError("extraction_mode=inversion requires a list of timesteps (cfg.t)")
    eps_generator = None
    mods: list[torch.Tensor] = []
    if multi_timestep and not inversion:
        # Dedicated, seeded RNG stream for eps so extraction is reproducible independent of
        # global RNG consumption. Per-image eps values differ (drawn sequentially from this
        # stream); requires a deterministic dataloader order (shuffle=False).
        eps_seed = cfg.eps_seed if cfg.eps_seed is not None else cfg.seed
        eps_generator = torch.Generator(device=device).manual_seed(eps_seed)

    print("saving %s images' features..." % split_name)
    for batch in tqdm(dataloader):
        img = batch["img"].to(device)  # B, 3, H, W
        label = batch["label"]  # B

        for single_img, single_label in zip(img, label, strict=True):
            # TODO: if GPU can tolerate higher batch sizes, we can extract features for the whole batch at once instead of looping through images one by one.
            if inversion:
                feats_k, mods_k, vels_k = model.extract_inversion(
                    single_img,
                    timesteps=cfg.t,
                    num_inversion_steps=cfg.num_inversion_steps,
                    block_idx=cfg.k,
                    guidance=cfg.guidance_scale,
                    want_velocity=vel_sink is not None,
                )  # feats_k: K, C, h, w; mods_k: K, 3, C; vels_k: K, T, d or None
                if vel_sink is not None:
                    if vels_k is None:
                        raise RuntimeError("velocity requested but the chain returned none")
                    vel_sink.append(vels_k.unsqueeze(0).cpu().float())  # 1, K, T, d
                if token_sink is not None:
                    _sink_tokens(token_sink, list(cfg.t), feats_k)
                per_t = feats_k.mean(dim=[2, 3])  # K, C — global average pool, pre-normalization
                all_feats.append(per_t.unsqueeze(0).cpu())  # 1, K, C
                if not mods:
                    # adaLN mods depend only on (t, null embeds), not the image.
                    mods = [m.unsqueeze(0).cpu() for m in mods_k]
            elif multi_timestep:
                latents = None
                noise = None
                per_t_feats: list[torch.Tensor] = []
                # Pre-pool tokens for this image, gathered across timesteps and sunk ONCE
                # after the loop so the oneshot arm yields (1, S, L, C) per image — the
                # same layout the inversion arm produces in a single call.
                per_t_tokens: list[torch.Tensor] = []
                for t_idx, timestep in enumerate(cfg.t):
                    feat_raw, ada, latents_used, noise_used = model.extract_raw(
                        single_img,
                        timestep=timestep,
                        block_idx=cfg.k,
                        ensemble_size=cfg.model.ensemble_size,
                        guidance=cfg.guidance_scale,
                        latents=latents,
                        noise=noise,
                        generator=eps_generator,
                    )  # feat_raw: 1, C, H, W
                    if t_idx == 0:
                        latents, noise = latents_used, noise_used
                    else:
                        # eps (and clean latents) must be IDENTICAL across the K forward
                        # passes for a given image — resampling per timestep corrupts
                        # increments/curvature along the timestep axis.
                        if not torch.equal(noise_used, noise):
                            raise RuntimeError(
                                "eps-consistency violation: noise at timestep %s differs from "
                                "the eps drawn at timestep %s for the same image." % (timestep, cfg.t[0])
                            )
                        if not torch.equal(latents_used, latents):
                            raise RuntimeError(
                                "latent-consistency violation: clean latents at timestep %s differ "
                                "from those encoded at timestep %s for the same image." % (timestep, cfg.t[0])
                            )
                    if token_sink is not None and timestep in token_sink["timesteps"]:
                        # clone: this reference is held until after the timestep loop, and
                        # extract_raw may hand back a reused buffer that later iterations
                        # overwrite in place. The pooled path is immune because .mean()
                        # below materialises a new tensor immediately.
                        per_t_tokens.append(feat_raw.detach().clone())  # 1, C, h, w
                    per_t_feats.append(
                        feat_raw.mean(dim=[2, 3])
                    )  # 1, C — global average pool, pre-normalization
                    if len(mods) < len(cfg.t):
                        mods.append(ada[0].unsqueeze(0).cpu())  # 1, 3, C — image-independent
                if token_sink is not None and per_t_tokens:
                    # cat over the wanted timesteps -> (S, C, h, w), matching the layout
                    # _sink_tokens expects from the inversion arm.
                    wanted = [t for t in cfg.t if t in token_sink["timesteps"]]
                    _sink_tokens(token_sink, wanted, torch.cat(per_t_tokens, dim=0))
                all_feats.append(torch.cat(per_t_feats, dim=0).unsqueeze(0).cpu())  # 1, K, C
            else:
                feat = model.extract(
                    single_img,
                    timestep=cfg.t,
                    block_idx=cfg.k,
                    ensemble_size=cfg.model.ensemble_size,
                    guidance=cfg.guidance_scale,
                )  # 1, C, H, W

                feat_vec = feat.mean(dim=[2, 3])  # 1, C  — global average pool
                feat_vec = F.normalize(feat_vec, dim=1)

                all_feats.append(feat_vec.cpu())
            all_labels.append(single_label.cpu())

    feats = torch.cat(all_feats, dim=0).float().numpy()  # N, C or N, K, C
    labels = torch.stack(all_labels, dim=0).numpy()  # N
    if multi_timestep:
        return feats, labels, torch.cat(mods, dim=0).float().numpy()  # mods: K, 3, C
    return feats, labels


def train_probe(
    probe_type: ProbeType,
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    num_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    num_classes: int,
    grid_size: int = 5,
    polynomial_order: int = 3,
) -> tuple[nn.Module, int, float]:
    X = torch.from_numpy(train_feats).float().to(device)
    y = torch.from_numpy(train_labels).long().to(device)

    ds = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    # Define probe model
    match probe_type:
        case ProbeType.LINEAR:
            probe = LinearProbe(X.shape[1], num_classes).to(device)
        case ProbeType.MLP:
            probe = MLPProbe(X.shape[1], num_classes).to(device)
        case ProbeType.KAN:
            probe = KANProbe(X.shape[1], num_classes, grid_size=grid_size, k=polynomial_order).to(device)
        case ProbeType.FOURIER_KAN:
            probe = FourierKANProbe(X.shape[1], num_classes, grid_size=grid_size, add_bias=True).to(device)
        case _:
            raise ValueError(
                f"Unsupported probe type: {probe_type}. Expected one of {list(ProbeType)}, got {probe_type}."
            )

    # TODO: make optimizer and loss configurable (e.g. SGD, label smoothing)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    t0 = time.perf_counter()
    probe.train()
    for _ in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            total_steps += 1
            epoch_loss += loss.item()
        wandb.log({"probe/train_loss": epoch_loss / len(loader)})

    return probe, total_steps, time.perf_counter() - t0


@torch.no_grad()
def evaluate_probe(probe, test_feats, test_labels, device):
    probe.eval()
    X = torch.from_numpy(test_feats).float().to(device)
    preds = probe(X).cpu().numpy().argmax(axis=1)
    top1 = (preds == test_labels).mean() * 100.0
    macro_f1 = f1_score(test_labels, preds, average="macro") * 100.0
    weighted_f1 = f1_score(test_labels, preds, average="weighted") * 100.0
    per_class_f1 = f1_score(test_labels, preds, average=None) * 100.0
    return top1, macro_f1, weighted_f1, per_class_f1
