"""Link the located, gsd-limited subset of a DiffusionSat geo run for a matched FID / KID score.

    python experiments/isaac/baselines/dsat_subset.py --run <geo run dir> --gsd-max 10 --expect-n 2886

Reads <run>/conditioning.csv, keeps the located rows whose tile gsd is <= --gsd-max metres
(dsat_cond.gsd_subset) and writes two directories of symlinks, sample i next to real image i:
    <run>/samples224_gsdle<G>/   -> ../samples224/<i>.png
    <run>/ref224_gsdle<G>/       -> the target of <run>/ref224_located/<i>.png
Refuses existing output directories. `--expect-n` is the count declared before the score was
computed; a different count is a FAIL (the subset is not the declared one).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsat_cond as DC  # noqa: E402


def tag(gsd_max: float) -> str:
    return f"gsdle{gsd_max:g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run", required=True, help="geo run directory (holds conditioning.csv, samples224, ref224_located)"
    )
    ap.add_argument(
        "--gsd-max", type=float, required=True, help="keep located rows with tile gsd <= this many metres"
    )
    ap.add_argument(
        "--expect-n", type=int, required=True, help="declared subset size; mismatch stops the run"
    )
    a = ap.parse_args(argv)
    with open(os.path.join(a.run, "conditioning.csv")) as f:
        rows = list(csv.DictReader(f))
    idx = DC.gsd_subset(rows, a.gsd_max)
    if len(idx) != a.expect_n:
        raise SystemExit(
            f"GATE gsd subset FAIL: {len(idx)} located rows with gsd <= {a.gsd_max:g} m in {a.run}, declared {a.expect_n}"
        )
    fake, real = (
        os.path.join(a.run, f"samples224_{tag(a.gsd_max)}"),
        os.path.join(a.run, f"ref224_{tag(a.gsd_max)}"),
    )
    for d in (fake, real):
        if os.path.exists(d):
            raise SystemExit(f"{d} exists; refusing to overwrite")
    links = []
    for i in idx:
        s = os.path.join(a.run, "samples224", f"{i:05d}.png")
        r = os.path.join(a.run, "ref224_located", f"{i:05d}.png")
        if not os.path.exists(s) or not os.path.exists(r):
            raise SystemExit(f"row {i}: missing sample or located reference ({s}, {r})")
        links.append((i, os.path.realpath(r)))
    os.makedirs(fake)
    os.makedirs(real)
    for i, r in links:
        os.symlink(os.path.join("..", "samples224", f"{i:05d}.png"), os.path.join(fake, f"{i:05d}.png"))
        os.symlink(r, os.path.join(real, f"{i:05d}.png"))
    print(
        f"GATE gsd subset PASS: {len(idx)} located rows with gsd <= {a.gsd_max:g} m (declared {a.expect_n}) of {len(rows)} -> {fake}, {real}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
