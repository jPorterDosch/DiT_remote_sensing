"""Gradient-reach gate for finetune-diffusion (2026-09-24 review, finding 1).

Run the default and supervised arms for ONE optimizer step (same seed; see the
smoke-finetune skill), then:

    python tests/check_lora_grad_reach.py models/smoke_grad_default models/smoke_grad_sup --k 28

Before step 1 every LoRA B is zero, so both arms run the same forward pass; blocks >= k get
flow-loss gradient only (identical across arms), blocks < k additionally get probe CE in
the supervised arm. Adam's first step is ~lr*sign(g), so per-block SIGN agreement of B
separates the two cleanly (exact equality does not: global-norm clipping scales the arms
differently and moves near-eps elements). Measured 2026-09-24: 0.40-0.50 below k,
>= 0.996 at/after k. Under the old block-k-only wrapping there is nothing below k and the
CE arm is indistinguishable from label-free -> FAIL.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
N_DOUBLE = 19


def step1(run_root: str) -> dict:
    hits = sorted(glob.glob(os.path.join(run_root, "*", "checkpoints", "lora_best_step00000001.pt")))
    if len(hits) != 1:
        raise SystemExit(f"{run_root}: expected exactly one step-1 checkpoint, found {hits}")
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)  # our own checkpoint
    if ck["global_step"] != 1:
        raise SystemExit(f"{hits[0]}: global_step={ck['global_step']}, need 1")
    return ck["lora_state_dict"]


def sign_agreement_by_block(sa: dict, sb: dict) -> dict[int, float]:
    if sa.keys() != sb.keys():
        raise SystemExit("arms wrapped different parameter sets")
    xs: dict[int, list] = {}
    for key in sa:
        if not key.endswith(".B"):
            continue
        m = re.search(r"(double|single)_blocks\.(\d+)\.", key)
        g = int(m.group(2)) + (0 if m.group(1) == "double" else N_DOUBLE)
        xs.setdefault(g, [[], []])
        xs[g][0].append(sa[key].float().flatten())
        xs[g][1].append(sb[key].float().flatten())
    out = {}
    for g, (a, b) in xs.items():
        a, b = torch.cat(a), torch.cat(b)
        nz = (a != 0) | (b != 0)
        out[g] = float((torch.sign(a[nz]) == torch.sign(b[nz])).float().mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("default_root")
    p.add_argument("supervised_root")
    p.add_argument("--k", type=int, default=28)
    args = p.parse_args()
    agree = sign_agreement_by_block(step1(args.default_root), step1(args.supervised_root))
    up = [v for g, v in agree.items() if g < args.k]
    dn = [v for g, v in agree.items() if g >= args.k]
    print(f"adapted blocks: {len(agree)}  (below k: {len(up)}, at/after k: {len(dn)})")
    if not up:
        print(
            "GRADIENT-REACH: FAIL -- no adapters upstream of the readout; CE cannot reach the probed features"
        )
        raise SystemExit(1)
    print(f"sign agreement below k: max {max(up):.4f}   at/after k: min {min(dn):.4f}")
    ok = max(up) < 0.9 and min(dn) > 0.99
    print("GRADIENT-REACH:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
