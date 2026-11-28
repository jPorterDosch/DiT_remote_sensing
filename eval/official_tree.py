"""Tree check and output location for the FLUX and DINO extraction scripts, for both registries.

    python -m eval.official_tree verify <dataset> <tree root>
    python -m eval.official_tree out <dataset> <default output dir>

`verify` checks an exported probe tree against its manifest before any feature is extracted:
GEO-Bench tasks (eval.features.OFFICIAL) through export_geobench (the line the scripts ran
before), chest X-ray datasets (OFFICIAL_CXR) through export_cxr's own manifest check, with its
count-only messages for the restricted keys.

`out` prints the directory a script writes features to. A GEO-Bench task keeps the script's
default unchanged. A chest X-ray dataset writes under the mode-700 CXR store instead
(<store>/feats/<default without a leading models/>): CheXpert and VinDr-CXR features are
derived from patient data under data-use agreements and stay with the images (design 6.4,
the MedSigLIP extractor does the same); ChestX-ray14 goes there too so the rows share one root.
A chest X-ray output outside the store, or a store that is not mode 700, is refused, so a
SAVE_ROOT / OUT override cannot move restricted features elsewhere.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import features as F  # noqa: E402


def cxr_store() -> str:
    """The CXR store: the parent of the trees directory that data/cxr links to (CXR_STORE overrides,
    for tests)."""
    env = os.environ.get("CXR_STORE")
    if env:
        return os.path.realpath(env)
    return os.path.dirname(os.path.realpath(os.path.join(F.REPO_ROOT, "data", "cxr")))


def is_cxr(dataset: str) -> bool:
    F.official_spec(dataset)  # unknown dataset -> SystemExit
    return dataset in F.OFFICIAL_CXR


def verify(dataset: str, root: str) -> None:
    if not is_cxr(dataset):
        from eval import export_geobench

        export_geobench.main(["--task", dataset.replace("_", "-"), "--out", root])
        return
    from eval import export_cxr as X

    restricted = dataset in X.RESTRICTED
    manifest = X._tree_manifest(dataset, root, F.official_classes(dataset), restricted)
    X.check_cxr_manifest(dataset, manifest, write=False)


def out_dir(dataset: str, default: str) -> str:
    if not is_cxr(dataset):
        return default
    store = cxr_store()
    mode = stat.S_IMODE(os.stat(store).st_mode)
    if mode != 0o700:
        raise SystemExit(f"{store}: mode {mode:o}, expected 700 -- refusing to write chest X-ray features")
    rel = default[len("models/") :] if default.startswith("models/") else default
    if os.path.isabs(rel):  # an explicit override: allowed only inside the store
        path = rel
    else:
        path = os.path.join(store, "feats", rel)
    real = os.path.realpath(path)
    if os.path.commonpath([real, store]) != store:
        raise SystemExit(f"{dataset}: output {path} is outside the CXR store {store} -- refusing")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("dataset")
    v.add_argument("root")
    o = sub.add_parser("out")
    o.add_argument("dataset")
    o.add_argument("default")
    a = p.parse_args(argv)
    if a.cmd == "verify":
        verify(a.dataset, a.root)
    else:
        print(out_dir(a.dataset, a.default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
