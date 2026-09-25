"""6ad: DINO-family frozen probes on the 6ac paired RESISC45 protocol (model-agnostic).

Generalizes dinov2_resisc45.py so DINOv3 Web-7B and Sat-7B (gated weights, downloaded by
JD) drop into the SAME instrument: identical 5,000 images (cache-order paths), identical
folds, section-13 probe (3 seeds x 5-fold, in-fold scaler, LR C=0.1), paired per-image
deltas against the CACHED 6ac vectors (A1 FLUX inversion base, B1 DINOv2-L clsmp).

Presets:
  dinov2_vitl14  — torch.hub, weights auto-download. Doubles as the REPRODUCTION GATE:
                   its extracted features must match results/dinov2_resisc45_feats_n5000.npz
                   and its probe must land on the cached B1 accuracy (0.9175).
  dinov3_web7b   — torch.hub facebookresearch/dinov3, entry dinov3_vit7b16,
                   --weights /path/to/dinov3_vit7b16_pretrain_lvd1689m*.pth
  dinov3_sat7b   — same entry, --weights /path/to/dinov3_vit7b16_pretrain_sat493m*.pth
                   NORMALIZATION GATE: sat preset REFUSES to run until --mean/--std are
                   passed explicitly, copied from the shipped dinov3 repo README
                   (rule 16: read the artifact, do not trust memory or web summaries).

Usage once weights are downloaded:
  python experiments/prototypes/dino_family_resisc45.py --preset dinov3_web7b --weights <pth>
  python experiments/prototypes/dino_family_resisc45.py --preset dinov3_sat7b --weights <pth> \
      --mean M M M --std S S S
See RESEARCH_NOTES 6ad for the pre-registration.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from experiments.prototypes._absorption_harness import C as HARNESS_C  # noqa: E402
from experiments.prototypes.dinov2_resisc45 import _fold_plain, paired, run_arm  # noqa: E402

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

PRESETS = {
    "dinov2_vitl14": dict(repo="facebookresearch/dinov2", entry="dinov2_vitl14",
                          needs_weights=False, dtype=torch.float32, batch=64),
    "dinov3_web7b": dict(repo="facebookresearch/dinov3", entry="dinov3_vit7b16",
                         needs_weights=True, dtype=torch.bfloat16, batch=8),
    "dinov3_sat7b": dict(repo="facebookresearch/dinov3", entry="dinov3_vit7b16",
                         needs_weights=True, dtype=torch.bfloat16, batch=8,
                         needs_explicit_norm=True),
}


def load_model(preset, weights, device):
    spec = PRESETS[preset]
    kwargs = {}
    if spec["needs_weights"]:
        assert weights and os.path.exists(weights), f"--weights required and must exist for {preset}"
        kwargs["weights"] = weights
    model = torch.hub.load(spec["repo"], spec["entry"], **kwargs)
    model = model.to(device=device, dtype=spec["dtype"]).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{preset}: {n_par/1e9:.2f} B params, dtype {spec['dtype']}, batch {spec['batch']}")
    return model, spec


@torch.no_grad()
def extract(model, spec, paths, device, mean, std):
    """CLS + mean-patch features. Handles dict (dinov2/v3 forward_features) output."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    batch = spec["batch"]
    cls_all, mp_all = [], []
    for i in range(0, len(paths), batch):
        imgs = []
        for f in paths[i : i + batch]:
            im = Image.open(f).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        x = ((torch.stack(imgs) - mean_t) / std_t).to(device=device, dtype=spec["dtype"])
        out = model.forward_features(x)
        assert isinstance(out, dict) and "x_norm_clstoken" in out and "x_norm_patchtokens" in out, \
            f"unexpected forward_features output: {type(out)} {list(out) if isinstance(out, dict) else ''}"
        cls_all.append(out["x_norm_clstoken"].float().cpu().numpy())
        mp_all.append(out["x_norm_patchtokens"].mean(dim=1).float().cpu().numpy())
        if i % (batch * 20) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.concatenate(cls_all), np.concatenate(mp_all)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--weights", default=None, help="local checkpoint path (gated presets)")
    p.add_argument("--mean", type=float, nargs=3, default=None)
    p.add_argument("--std", type=float, nargs=3, default=None)
    p.add_argument("--max-images", type=int, default=None, help="smoke cap (strided)")
    p.add_argument("--n-jobs", type=int, default=7)
    args = p.parse_args()

    spec = PRESETS[args.preset]
    if spec.get("needs_explicit_norm") and (args.mean is None or args.std is None):
        raise SystemExit(
            f"{args.preset}: pass --mean/--std copied from the SHIPPED dinov3 repo README "
            "(~/.cache/torch/hub/facebookresearch_dinov3_main/README* after first hub load). "
            "Sat-493M does not use ImageNet statistics; running with the wrong norm would "
            "silently handicap the sat arm (rule 16: verify the artifact).")
    mean, std = (args.mean or IMAGENET[0]), (args.std or IMAGENET[1])
    print(f"normalization: mean {mean} std {std}")

    # identity: the 6ac cache is the single source of image order and labels
    r = np.load("results/dinov2_resisc45_paired.npz", allow_pickle=True)
    paths = [str(x) for x in r["paths"]]
    y = r["labels"]
    assert len(paths) == 5000 and all(os.path.exists(f) for f in paths), "6ac path set broken"
    if args.max_images:
        keep = np.linspace(0, len(paths) - 1, args.max_images).astype(int)
        paths = [paths[i] for i in keep]
        y = y[keep]
        print(f"SMOKE: {len(paths)} images — accuracies meaningless")

    device = "cuda"
    model, spec = load_model(args.preset, args.weights, device)
    cache = f"results/{args.preset}_resisc45_feats_n5000.npz"
    if os.path.exists(cache) and not args.max_images:
        d = np.load(cache, allow_pickle=True)
        assert list(d["paths"]) == paths
        cls_f, mp_f = d["cls"], d["mp"]
        print(f"reusing {cache}")
    else:
        cls_f, mp_f = extract(model, spec, paths, device, mean, std)
        if not args.max_images:
            np.savez(cache, cls=cls_f, mp=mp_f, paths=np.array(paths),
                     mean=np.array(mean), std=np.array(std))
    del model
    torch.cuda.empty_cache()

    # REPRODUCTION GATE (dinov2 preset only): features must match the 6ac extraction
    if args.preset == "dinov2_vitl14" and not args.max_images:
        ref = np.load("results/dinov2_resisc45_feats_n5000.npz", allow_pickle=True)
        ok = np.allclose(ref["cls"], cls_f, atol=1e-4) and np.allclose(ref["mp"], mp_f, atol=1e-4)
        print(f"feature-reproduction gate vs 6ac cache: {'PASS' if ok else 'FAIL'}")
        assert ok, "dinov2 features do not reproduce the 6ac extraction — instrument audit first"

    clsmp = np.ascontiguousarray(np.concatenate([cls_f, mp_f], axis=1)).astype(np.float32)
    mp_only = np.ascontiguousarray(mp_f).astype(np.float32)
    print(f"\nprobes (section-13 instrument, C={HARNESS_C}):")
    cv, _, _ = run_arm(f"{args.preset} clsmp {clsmp.shape[1]}d", y, _fold_plain,
                       (clsmp, y, HARNESS_C, None, None), args.n_jobs)
    cv_mp, _, _ = run_arm(f"{args.preset} mp-only {mp_only.shape[1]}d", y, _fold_plain,
                          (mp_only, y, HARNESS_C, None, None), args.n_jobs)

    if not args.max_images:
        A1, B1 = r["A1"], r["B1"]
        print("\nPAIRED deltas (identical folds/images as 6ac):")
        paired(f"{args.preset} clsmp - A1 FLUXinv  ", A1, cv)
        paired(f"{args.preset} clsmp - B1 DINOv2-L ", B1, cv)
        if args.preset == "dinov2_vitl14":
            same = float(np.abs(cv - B1).mean())
            print(f"probe-reproduction gate: mean |cv - B1| = {same:.6f} "
                  f"({'PASS' if same < 1e-9 else 'CHECK'})")
        out = f"results/{args.preset}_resisc45_paired.npz"
        np.savez(out, cv=cv, cv_mp=cv_mp, labels=y,
                 protocol=np.array(f"6ad: {args.preset}, 6ac instrument, paired on identical 5000 images, "
                                   f"norm mean={mean} std={std}"))
        print(f"cached to {out}")


if __name__ == "__main__":
    main()
