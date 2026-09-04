"""Per-image roundtrip reconstruction error: a flow-native, within-dataset OOD score.

Invert a clean image to t_stop along the RF-Solver chain, integrate back to t=0, and measure
the relative L2 discrepancy in latent space. The ODE is analytically reversible, so the error
is accumulated discretization error -- larger where the learned field is more curved, i.e.
where the model handles the image worse. Unlike every cross-dataset comparison in this project,
this is PER-IMAGE and WITHIN-dataset, so the resolution confound does not touch it.

Two uses:
  1. Dataset-level: does RESISC45 invert worse than EuroSAT? (A direct check on the OOD story.)
  2. Image-level: does an image's roundtrip error predict its inversion-vs-oneshot advantage?
     (Prediction 2 of section 7. The per-image correctness comes from
     experiments/paired_image_bootstrap.py's caches -- correlate offline.)

Also runs the MATCHED-NFE integrator ablation (audit finding O3): REPORT.md compared Euler and
RF-Solver at the same step count, but the order-2 step evaluates the velocity twice per step,
so it had 2x the NFE. Euler@2N vs RF-Solver@N is the fair comparison.
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


def rel_err(rec: torch.Tensor, clean: torch.Tensor) -> float:
    return float((rec - clean).float().norm() / clean.float().norm())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--img-size", type=int, nargs=2, required=True)
    p.add_argument("--n", type=int, default=200, help="images (subsampled from the n=5000 subset)")
    p.add_argument("--t-stop", type=int, default=580)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--nfe-ablation", action="store_true",
                   help="also run Euler@2N vs RF-Solver@N on the first 10 images")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = RunConfig(task="extract", model=ModelConfig(name="flux", ensemble_size=1),
                    dataset=DatasetConfig(name=args.dataset, path=args.path),
                    img_size=list(args.img_size), subset_size=5000, subset_seed=42,
                    batch_size=1, num_workers=4, label_fraction=1.0)
    seed_all(cfg.seed)
    dataset = DATASETS[cfg.dataset.name](cfg)
    train_ds = dataset.get_data(cfg)["train"].dataset
    all_labels = np.array([c for _, c, _ in train_ds.samples])
    idx5000 = _stratified_indices(all_labels, 5000, 42)
    # Deterministic subsample of the SAME 5000-image subset the probes use, so per-image
    # roundtrip error can be joined against per-image probe correctness by subset position.
    rng = np.random.default_rng(7)
    pos = np.sort(rng.choice(len(idx5000), args.n, replace=False))
    indices = idx5000[pos]

    model = MODELS["flux"](cfg, dataset.category_list)
    loader = DataLoader(Subset(train_ds, indices.tolist()), batch_size=1, shuffle=False,
                        num_workers=2)

    from models.flux.feat_flux import prepare  # packed clean latent for the error metric

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["img"].to("cuda").squeeze(0)  # invert_chain expects (C, H, W)
            rec, clean = model.roundtrip(img, t_stop=args.t_stop,
                                         num_inversion_steps=args.num_steps,
                                         block_idx=28, guidance=1.0)
            rows.append({"subset_pos": int(pos[i]), "train_index": int(indices[i]),
                         "label": int(batch["label"][0]),
                         "rel_err": rel_err(rec, clean)})
            if i % 20 == 0:
                print(f"  {i}/{args.n}  rel_err={rows[-1]['rel_err']:.5f}", flush=True)

    errs = np.array([r["rel_err"] for r in rows])
    print(f"\n{args.dataset}: roundtrip rel-L2 @ t_stop={args.t_stop}, N={args.num_steps}: "
          f"mean {errs.mean():.5f}  median {np.median(errs):.5f}  p90 {np.quantile(errs, .9):.5f}",
          flush=True)
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "t_stop": args.t_stop,
                   "num_steps": args.num_steps, "rows": rows}, f)
    print(f"wrote {args.out}", flush=True)

    if args.nfe_ablation:
        print("\nMATCHED-NFE integrator ablation (10 images):", flush=True)
        loader10 = DataLoader(Subset(train_ds, indices[:10].tolist()), batch_size=1,
                              shuffle=False, num_workers=2)
        for order, steps in [(2, args.num_steps), (1, args.num_steps), (1, 2 * args.num_steps)]:
            es = []
            with torch.no_grad():
                for batch in loader10:
                    img = batch["img"].to("cuda").squeeze(0)  # invert_chain expects (C, H, W)
                    out = model._inner.invert_chain(
                        img, cache_timesteps=[], num_inversion_steps=steps, block_idx=28,
                        guidance=1.0, order=order, t_stop=args.t_stop)
                    gen = model._inner.generate_chain(
                        out["z_final"], out["img_ids"], t_start=args.t_stop,
                        num_inversion_steps=steps, guidance=1.0, order=order)
                    clean_packed, _ = prepare(img=out["latents_clean"])
                    es.append(rel_err(gen, clean_packed))
            nfe = steps * (2 if order == 2 else 1)
            print(f"  order={order} steps={steps:<4} NFE/leg={nfe:<4} "
                  f"rel_err mean {np.mean(es):.5f}", flush=True)


if __name__ == "__main__":
    main()
