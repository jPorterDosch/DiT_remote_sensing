"""Extract SOLVER CURVATURE along the inversion chain: pred and pred_mid at each cached
timestep, fp32, differenced offline.

WHY THIS IS THE ADMISSIBLE WALL INSTRUMENT (RESEARCH_NOTES 6g). Increments/curvature of the
cached STATES are linear functions of the state concat, so a linear probe on the states
already spans them -- "beyond the states" was untestable there by construction. pred_mid is
the network's velocity at a midpoint the cache never stores, a NONLINEAR function of the
state: (pred_mid - pred) lies outside the cached-state span. It is dv/dt along the path --
deviation from rectification, the model's density map read out where crossing training paths
pull the field.

GATE (print, do not assume): rms(pred_mid - pred)/rms(pred) per t, against bf16 noise
(velocities computed in bf16, rel eps ~0.004; stored fp32 so no further loss).

Cost: the second velocity call already happens inside every order-2 step -- capture is a
clone, not a forward. Same 11-16 s/image as any chain run.

FINDINGS. Magnitude gate passes at every t on both datasets at n=500 AND n=5000 (ratios
0.03-0.07, ~10x the bf16 floor) -- the residual un-rectifiedness at g=1.0 is real and the
downstream probes are readable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "models"))

import datasets  # noqa: F401,E402
import models  # noqa: F401,E402
from registry import DATASETS, MODELS  # noqa: E402
from run import DatasetConfig, ModelConfig, RunConfig  # noqa: E402
from tasks.extraction import _stratified_indices  # noqa: E402
from utils import seed_all  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

T_GRID = [100, 180, 260, 340, 420, 500, 580]


def _env_suffix_and_meta():
    """Provenance stamp for the extraction-control env vars, mirroring
    tasks.extraction.env_provenance: these standalone extractors honor FLUX_RANDOM_INIT
    (via load_flow_model) and DEGRADE_TO (via the dataset hook) but previously wrote
    UN-suffixed cache names -- a stale export would poison the exact filenames the
    downstream probes glob (2026-09-04 review)."""
    from utils import env_value

    parts = []
    if env_value("FLUX_RANDOM_INIT"):
        parts.append("RANDINIT")
    if env_value("FIXED_COND_T"):
        parts.append(f"FIXEDCOND{env_value('FIXED_COND_T')}")
    if env_value("DEGRADE_TO"):
        parts.append(f"DEG{env_value('DEGRADE_TO')}")
    suffix = "".join("_" + p for p in parts)
    meta = {
        "weights": "random_init" if env_value("FLUX_RANDOM_INIT") else "flux-dev",
        "degrade_to": int(env_value("DEGRADE_TO")) if env_value("DEGRADE_TO") else None,
        "fixed_cond_t": int(env_value("FIXED_COND_T")) if env_value("FIXED_COND_T") else None,
    }
    return suffix, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--img-size", type=int, nargs=2, required=True)
    p.add_argument("--subset-size", type=int, default=500)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--out-dir", default="models/solver_curvature")
    args = p.parse_args()

    cfg = RunConfig(
        task="extract",
        model=ModelConfig(name="flux", ensemble_size=1),
        dataset=DatasetConfig(name=args.dataset, path=args.path),
        img_size=list(args.img_size),
        subset_size=args.subset_size,
        subset_seed=42,
        batch_size=1,
        num_workers=4,
        label_fraction=1.0,
    )
    seed_all(cfg.seed)
    dataset = DATASETS[cfg.dataset.name](cfg)
    train_ds = dataset.get_data(cfg)["train"].dataset
    all_labels = np.array([c for _, c, _ in train_ds.samples])
    indices = _stratified_indices(all_labels, cfg.subset_size, cfg.subset_seed)
    model = MODELS["flux"](cfg, dataset.category_list)
    loader = DataLoader(Subset(train_ds, indices.tolist()), batch_size=1, shuffle=False, num_workers=4)

    preds, mids, states, labels = [], [], [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            img = batch["img"].to("cuda").squeeze(0)
            out = model._inner.invert_chain(
                img,
                cache_timesteps=T_GRID,
                num_inversion_steps=args.num_steps,
                block_idx=28,
                guidance=1.0,
                want_curvature=True,
                want_states=True,
            )
            # curvs: {t: {"pred": (1,T,64), "pred_mid": (1,T,64)}} fp32. The TERMINAL
            # cached t has no step leaving it (the chain breaks before its _ode_step), so
            # curvature exists only at the non-terminal cached timesteps.
            got = sorted(out["curvs"].keys())
            preds.append(np.stack([out["curvs"][t]["pred"].cpu().numpy()[0] for t in got]))
            mids.append(np.stack([out["curvs"][t]["pred_mid"].cpu().numpy()[0] for t in got]))
            t_used = got
            states.append(np.stack([out["states"][t].cpu().float().numpy()[0] for t in T_GRID]))
            labels.append(int(batch["label"][0]))
            if n % 25 == 0:
                print(f"  {n}/{len(indices)}", flush=True)

    P, M = np.stack(preds), np.stack(mids)  # (N, 7, T, 64)
    y = np.array(labels)
    os.makedirs(args.out_dir, exist_ok=True)
    base = f"{args.dataset}_solvercurv_n{args.num_steps}" + _env_suffix_and_meta()[0]
    S_ = np.stack(states)  # (N, 7, T, 64) packed raw chain states, fp32 -- paired by
    # construction with the curvature (same chain pass), so the n=5000 probe needs no
    # separate invstate extraction and no cross-cache pairing.
    np.savez(
        os.path.join(args.out_dir, base + ".npz"),
        pred=P.astype(np.float32),
        pred_mid=M.astype(np.float32),
        states=S_.astype(np.float32),
        labels=y,
        timesteps=np.array(t_used),  # curvature axis (6, non-terminal)
        state_timesteps=np.array(T_GRID),  # states axis (7, incl. terminal)
        subset_indices=indices,
        subset_seed=np.array(42),
    )
    D = M - P
    print(f"\nwrote {base}.npz  pred={P.shape}")
    print("GATE  (rms(pred_mid - pred) / rms(pred) per t; bf16 floor ~0.004):")
    for i, t in enumerate(t_used):
        r = float(np.sqrt((D[:, i] ** 2).mean()) / np.sqrt((P[:, i] ** 2).mean()))
        print(f"  t={t:<4} ratio={r:.5f}  {'PASS' if r > 0.012 else 'AT FLOOR -- null unreadable'}")
    meta = {
        **_env_suffix_and_meta()[1],
        "dataset": args.dataset,
        "img_size": list(args.img_size),
        "t": "non-terminal cached timesteps (terminal has no departing step)",
        "num_inversion_steps": args.num_steps,
        "subset_size": int(len(indices)),
        "subset_seed": 42,
        "guidance": 1.0,
        "k": 28,
        "content": "pred and pred_mid at cached t, fp32, packed (N,K,T,64)",
        "extraction_mode": "SOLVER_CURVATURE",
    }
    with open(os.path.join(args.out_dir, base + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
