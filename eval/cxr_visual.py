"""Visual check of the chest X-ray data stage: the figures a human looks at before training starts.

    python -m eval.cxr_visual --dataset chestxray14 --out <new dir> --arrow-dir <shards> --index <npz> [--root <tree>]
    python -m eval.cxr_visual --dataset chexpert --out <new dir> --raw-dir <zips> [--root <tree>]
    python -m eval.cxr_visual --dataset vindr_cxr_kaggle --out <new dir> --raw-dir <dir of the zip> [--root <tree>]

Every function takes a dataset `key` (chestxray14 | chexpert | vindr_cxr_kaggle); the tree root comes
from eval.features.OFFICIAL_CXR[key]["root"] (repo-relative paths are joined to the repository root)
unless the keyword `root` overrides it. Functions return a matplotlib Figure or a pandas DataFrame.

Source of a tree row (read again from the archive, never from the tree): chestxray14 `source_ref`
"shard:row" in the Arrow shards (`raw_dir` is the shard directory), chexpert the zip member whose key
spells the tree name (zip ordinal checked against `source_ref`), vindr_cxr_kaggle the train/<name>.dicom
member decoded by eval.cxr_dicom (ordinal checked against `source_ref`).

CheXpert and VinDr-CXR are patient data. The figures show their images (that is their purpose) and are
written only under a private directory (owned by you, no group/other bits). Everything else is
count-only for those keys: stdout, stderr and exception messages carry counts, ordinals, split names,
class names and exception type names, never an image name, patient id, source_ref, member name, CSV row
or DICOM string; titles and tick labels carry class names and the ordinal within the sample (`#3`).
Library warnings are tallied by category name only.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import stat as _stat_mod
import sys
import textwrap
import zlib
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import export_cxr as X  # noqa: E402
from eval import features as F  # noqa: E402

KEYS = ("chestxray14", "chexpert", "vindr_cxr_kaggle")
RESTRICTED = X.RESTRICTED
SPLITS = ("train", "val", "test")
WARN_COUNTS: Counter = Counter()  # library warning classes seen by the DICOM / CSV / figure steps


def _stat(path: str) -> os.stat_result:
    """The one stat lookup of the private-directory guard (tests substitute it)."""
    return os.stat(path)


# ------------------------------------------------------------------------------------ guards
@contextlib.contextmanager
def _count_only(key: str, what: str):
    """Restricted keys: any exception becomes `<key>: <what> failed (<TypeName>)`; the text of the
    caught exception (it can quote a name, a CSV row or a DICOM string) goes nowhere."""
    try:
        yield
    except Exception as e:  # noqa: BLE001 - SystemExit is not an Exception: count-only exits pass through
        if key not in RESTRICTED:
            raise
        raise SystemExit(f"{key}: {what} failed ({type(e).__name__})") from None


def _is_private_dir(path: str) -> bool:
    try:
        st = _stat(path)
    except OSError:
        return False
    return _stat_mod.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and st.st_mode & 0o077 == 0


def _resolve_out(key: str, out: str) -> str:
    """`--out` resolved once: realpath of its parent (symlinks and `..` are followed by the OS, as the
    writes would) joined to its leaf name. Everything after this uses the returned path only."""
    base = os.path.basename(out)
    if base in ("", ".", ".."):
        raise SystemExit(
            f"{key}: --out names no new directory (it ends in a slash, '.' or '..'); nothing written"
        )
    return os.path.join(os.path.realpath(os.path.dirname(out) or "."), base)


def _require_private(key: str, out: str, private_root: str | None) -> None:
    """Restricted keys: `--private-root` is required; it is private itself and holds the resolved `out`
    (from _resolve_out). Messages carry the dataset key only."""
    if private_root is None:
        raise SystemExit(f"{key}: --private-root is required for this dataset; nothing written")
    if not _is_private_dir(private_root):
        raise SystemExit(
            f"{key}: --private-root is not a directory owned by you with mode 700 (no group/other bits); "
            "nothing written"
        )
    _require_under(key, out, private_root)


def _require_under(key: str, out: str, private_root: str) -> None:
    root, parent = os.path.realpath(private_root), os.path.dirname(out)
    if parent != root and not parent.startswith(root.rstrip(os.sep) + os.sep):
        raise SystemExit(f"{key}: --out is not under --private-root; nothing written")


def tree_root(key: str, root: str | None = None) -> str:
    """The tree root: `root` when given, else the registry's, joined to the repository root if relative."""
    if root is not None:
        return root
    if key not in F.OFFICIAL_CXR:
        raise SystemExit(f"unknown chest X-ray dataset key (expected one of {list(KEYS)})")
    r = F.OFFICIAL_CXR[key]["root"]
    return r if os.path.isabs(r) else os.path.join(F.REPO_ROOT, r)


# ------------------------------------------------------------------------------------ the tree
class _Tree:
    """One probe tree: export_table.csv, export_report.json, labels.npz, PNGs."""

    def __init__(self, key: str, root: str | None):
        self.key, self.restricted = key, key in RESTRICTED
        self.root = tree_root(key, root)
        with _count_only(key, "reading the tree tables"):
            self.table = pd.read_csv(
                os.path.join(self.root, "export_table.csv"), dtype=str, keep_default_na=False
            )
            with open(os.path.join(self.root, "export_report.json")) as f:
                self.report = json.load(f)
            self.classes = [str(c) for c in self.report["classes"]]
            self.table["src_w"] = self.table["src_w"].astype(int)
            self.table["src_h"] = self.table["src_h"].astype(int)
        self._labels: dict[str, tuple[list[str], np.ndarray]] = {}

    def splits(self) -> list[str]:
        return [s for s in SPLITS if (self.table["split"] == s).any()]

    def labels(self, split: str) -> tuple[list[str], np.ndarray]:
        if split not in self._labels:
            with _count_only(self.key, "reading labels.npz"):
                z = np.load(os.path.join(self.root, split, "labels.npz"))
                self._labels[split] = ([str(n) for n in z["names"]], np.asarray(z["y"]))
        return self._labels[split]

    def y_of(self, split: str, name: str) -> np.ndarray:
        names, y = self.labels(split)
        with _count_only(self.key, "matching a label row"):  # list.index quotes the name it misses
            return y[names.index(name)]

    def png(self, split: str, name: str) -> np.ndarray:
        with _count_only(self.key, "reading a tree image"):
            with Image.open(os.path.join(self.root, split, f"{name}.png")) as im:
                return np.asarray(im.convert("RGB"))

    def sample_rows(self, n: int, seed: int, salt: str, split: str | None = None) -> pd.DataFrame:
        t = self.table if split is None else self.table[self.table["split"] == split]
        k = min(n, len(t))
        rng = np.random.default_rng([seed, zlib.crc32(salt.encode())])
        return t.iloc[rng.choice(len(t), size=k, replace=False)].reset_index(drop=True)


def _class_text(classes: list[str], y: np.ndarray, width: int = 22) -> str:
    pos = [c for c, v in zip(classes, y, strict=True) if v]
    return textwrap.fill(", ".join(pos) if pos else "no positive class", width=width)


def _grid(n: int, ncols: int, cell: float = 2.0, extra_h: float = 0.0):
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows + extra_h), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")  # no tick labels anywhere
    return fig, axes.ravel()


def _show(ax, arr: np.ndarray, title: str) -> None:
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(arr)
    ax.set_title(title, fontsize=6)
    ax.axis("off")


def _message_figure(text: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9)
    return fig


# ------------------------------------------------------------------------------------ source readers
def _open_source(key: str, raw_dir: str, table: pd.DataFrame):
    """read(row) -> uint8 greyscale source array of one table row, checked against its source_ref."""
    if key == "chestxray14":
        import pyarrow as pa
        import pyarrow.ipc as ipc
        from datasets import cxr14 as G

        readers: dict[int, ipc.RecordBatchFileReader] = {}

        def read(row) -> np.ndarray:
            shard, r = (int(v) for v in row["source_ref"].split(":"))
            if shard not in readers:
                readers[shard] = ipc.open_file(pa.memory_map(os.path.join(raw_dir, G.shard_name(shard))))
            batch = readers[shard].get_batch(r // G.BATCH_ROWS)
            j = r % G.BATCH_ROWS
            if batch.column("img_name")[j].as_py() != f"{row['name']}.png":
                raise ValueError("shard row does not hold the tree image")
            return np.asarray(Image.open(io.BytesIO(batch.column("image_bytes")[j].as_py())).convert("L"))

        return read
    if key == "chexpert":
        with _count_only(key, "indexing the zips"):
            members = X.ZipMembers(sorted(glob.glob(os.path.join(raw_dir, "*.zip"))))
            by_name: dict[str, list[str]] = {}
            for k in members._where:  # noqa: SLF001 - the exporter's own index, reused as is
                by_name.setdefault(os.path.splitext(k)[0].replace("/", "_"), []).append(k)

        def read(row) -> np.ndarray:
            found = by_name.get(row["name"], [])
            if len(found) != 1:
                raise ValueError(f"{len(found)} members for one tree row")
            if str(members.zip_ordinal(found[0])) != row["source_ref"]:
                raise ValueError("zip ordinal differs from source_ref")
            with Image.open(io.BytesIO(members.read(found[0]))) as im:
                return np.asarray(im.convert("L"))

        return read
    if key == "vindr_cxr_kaggle":
        from eval import cxr_dicom as DCM

        with _count_only(key, "indexing the zip"):
            zipped = X._VindrZip(raw_dir)  # noqa: SLF001

        def read(row) -> np.ndarray:
            if f"0:{zipped.ordinal(row['name'])}" != row["source_ref"]:
                raise ValueError("member ordinal differs from source_ref")
            with X._counted_warnings(WARN_COUNTS):  # noqa: SLF001
                arr, _info = DCM.decode_dicom_bytes(zipped.read(row["name"]))
            return np.asarray(arr)

        return read
    raise SystemExit(f"unknown chest X-ray dataset key (expected one of {list(KEYS)})")


def _title(tree: _Tree, k: int, row, extra: str = "") -> str:
    """Restricted: the ordinal only. ChestX-ray14: the image name too."""
    head = f"#{k}" if tree.restricted else f"#{k} {row['name']}"
    return f"{head} {extra}".strip()


def _pairs_figure(tree: _Tree, entries: list[tuple[str, pd.Series, np.ndarray]], pairs_per_row: int):
    """entries: (caption, table row, source array); source panel left of its exported panel."""
    fig, axes = _grid(len(entries) * 2, pairs_per_row * 2, cell=2.6)
    for i, (cap, row, src) in enumerate(entries):
        exp = tree.png(row["split"], row["name"])
        _show(axes[2 * i], src, f"{cap} source {src.shape[1]}x{src.shape[0]}")
        _show(axes[2 * i + 1], exp, f"exported {exp.shape[1]}x{exp.shape[0]} padded={row['padded']}")
    fig.suptitle(f"{tree.key}: source beside export", fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------ figures
def corpus_grid(arrow_dir, index_npz, which="train", n=24, seed=0):
    """Training view: n images of the index exactly as Cxr14Dataset hands them to the model (224 px)."""
    from datasets.cxr14 import Cxr14Dataset

    smoke = bool(np.load(index_npz)["smoke"])
    ds = Cxr14Dataset(arrow_dir, index_npz, which=which, allow_smoke=smoke)
    rng = np.random.default_rng([seed, zlib.crc32(which.encode())])
    pick = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    fig, axes = _grid(len(pick), 6)
    for k, i in enumerate(pick):
        item = ds[int(i)]
        arr = np.rint((item["img"].numpy().transpose(1, 2, 0) / 2 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        _show(axes[k], arr, f"#{k + 1} {item['img_name']}")
    fig.suptitle(
        f"chestxray14 training view ({which}, {ds.img_size} px)" + (" SMOKE INDEX" if smoke else ""),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def tree_grid(key, split, n=24, seed=0, *, root=None):
    """Exported images with their positive class names (restricted keys: ordinal and classes only)."""
    tree = _Tree(key, root)
    rows = tree.sample_rows(n, seed, f"tree_grid_{split}", split=split)
    fig, axes = _grid(len(rows), 6)
    for k, (_, row) in enumerate(rows.iterrows(), start=1):
        y = tree.y_of(split, row["name"])
        _show(
            axes[k - 1],
            tree.png(split, row["name"]),
            f"{_title(tree, k, row)}\n{_class_text(tree.classes, y)}",
        )
    fig.suptitle(
        f"{key} tree, split {split}: {len(rows)} of {int((tree.table['split'] == split).sum())}", fontsize=8
    )
    fig.tight_layout()
    return fig


def raw_vs_exported(key, raw_dir, n=6, seed=0, *, root=None):
    """n random tree rows: the source (read again from the archive) beside the PNG in the tree."""
    tree = _Tree(key, root)
    rows = tree.sample_rows(n, seed, "raw_vs_exported")
    with _count_only(key, "reading the sources"):
        read = _open_source(key, raw_dir, tree.table)
        entries = [(f"#{k}", row, read(row)) for k, (_, row) in enumerate(rows.iterrows(), start=1)]
    return _pairs_figure(tree, entries, pairs_per_row=2)


def extreme_aspect(key, raw_dir, k=6, *, root=None):
    """The k widest and the k tallest sources beside their exports."""
    tree = _Tree(key, root)
    t = tree.table.assign(_aspect=tree.table["src_w"] / tree.table["src_h"]).sort_values(["_aspect", "name"])
    tall = t.head(k).reset_index(drop=True)
    wide = t.tail(k).iloc[::-1].reset_index(drop=True)
    with _count_only(key, "reading the sources"):
        read = _open_source(key, raw_dir, tree.table)
        entries = []
        for i in range(max(len(wide), len(tall))):
            for tag, frame in (("widest", wide), ("tallest", tall)):
                if i < len(frame):
                    row = frame.iloc[i]
                    entries.append((f"{tag} #{i + 1} {row['_aspect']:.6f}", row, read(row)))
    return _pairs_figure(tree, entries, pairs_per_row=2)


def photometric_pairs(root, n=6, seed=0):
    """VinDr-CXR: MONOCHROME1 exports (top row) beside MONOCHROME2 exports (bottom row)."""
    tree = _Tree("vindr_cxr_kaggle", root)
    rng = np.random.default_rng([seed, zlib.crc32(b"photometric_pairs")])
    picked = {}
    for ph in ("MONOCHROME1", "MONOCHROME2"):
        t = tree.table[tree.table["photometric"] == ph].reset_index(drop=True)
        picked[ph] = t.iloc[rng.choice(len(t), size=min(n, len(t)), replace=False)] if len(t) else t
    empty = [ph for ph, t in picked.items() if len(t) == 0]
    if empty:
        counts = {ph: int((tree.table["photometric"] == ph).sum()) for ph in picked}
        return _message_figure(f"no rows of {', '.join(empty)} in this tree; rows per class {counts}")
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, ph in enumerate(("MONOCHROME1", "MONOCHROME2")):
        for k, (_, row) in enumerate(picked[ph].iterrows()):
            _show(axes[r, k], tree.png(row["split"], row["name"]), f"{ph} #{k + 1}")
    fig.suptitle("vindr_cxr_kaggle exports by source photometric interpretation", fontsize=8)
    fig.tight_layout()
    return fig


def prevalence(key, *, root=None) -> pd.DataFrame:
    """class x split: images, positives, rate (counted from labels.npz)."""
    tree = _Tree(key, root)
    out = []
    for split in tree.splits():
        _names, y = tree.labels(split)
        for j, c in enumerate(tree.classes):
            pos = int(y[:, j].sum())
            out.append(
                {"class": c, "split": split, "n": int(y.shape[0]), "positives": pos, "rate": pos / y.shape[0]}
            )
    return pd.DataFrame(out, columns=["class", "split", "n", "positives", "rate"])


def size_histograms(key, *, root=None):
    """Source widths and heights, and the stored square sides, counted over the export table."""
    tree = _Tree(key, root)
    t = tree.table
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3))
    a.hist(t["src_w"], bins=30)
    a.set_title("source width (px)", fontsize=8)
    b.hist(t["src_h"], bins=30)
    b.set_title("source height (px)", fontsize=8)
    sides = t["side"].astype(int).value_counts().sort_index()
    c.bar([str(s) for s in sides.index], sides.to_numpy())
    c.set_title(f"stored side (px); padded {int((t['padded'] == 'True').sum())} of {len(t)}", fontsize=8)
    fig.suptitle(f"{key}: {len(t)} images", fontsize=8)
    fig.tight_layout()
    return fig


def gate_summary(gate_json) -> pd.DataFrame:
    """The leakage gate JSON (eval.cxr_leakage.main --out) as a table: check, status, detail."""
    with open(gate_json) as f:
        rec = json.load(f)
    rows = [
        {"check": c["name"], "status": "PASS" if c["passed"] else "FAIL", "detail": c["detail"]}
        for c in rec["checks"]
    ]
    rows += [{"check": f"tree {k}", "status": "SKIPPED", "detail": ""} for k in rec.get("skipped", [])]
    rows.append(
        {"check": "verdict", "status": str(rec["verdict"]).removeprefix("CXR_LEAKAGE_GATE "), "detail": ""}
    )
    return pd.DataFrame(rows, columns=["check", "status", "detail"])


def resample_gap(arrow_dir, index_npz, tree_root, n=256, seed=0) -> dict:
    """Mean and max abs difference (0-255) between the training view (Cxr14Dataset: source -> 224) and
    the probe view (the tree PNG: pad, <= 448, then -> 224 bicubic) of n tree images of the index."""
    from datasets.cxr14 import Cxr14Dataset

    tree = _Tree("chestxray14", tree_root)
    smoke = bool(np.load(index_npz)["smoke"])
    ds = Cxr14Dataset(arrow_dir, index_npz, which="all", allow_smoke=smoke)
    pos = {(int(s), int(r)): i for i, (s, r) in enumerate(zip(ds.shard, ds.row, strict=True))}
    t = tree.table[tree.table["split"].isin(["train", "val"])]
    rng = np.random.default_rng([seed, zlib.crc32(b"resample_gap")])
    rows = t.iloc[rng.choice(len(t), size=min(n, len(t)), replace=False)]
    total, n_px, worst, per_image = 0.0, 0, 0, []
    for _, row in rows.iterrows():
        shard, r = (int(v) for v in row["source_ref"].split(":"))
        if (shard, r) not in pos:
            raise ValueError("tree row is not a row of the index")
        item = ds[pos[(shard, r)]]
        if item["img_name"] != f"{row['name']}.png":
            raise ValueError("index row and tree row hold different images")
        train = np.rint((item["img"].numpy().transpose(1, 2, 0) / 2 + 0.5) * 255).astype(np.int64)
        probe = np.asarray(
            Image.fromarray(tree.png(row["split"], row["name"])).resize(
                (ds.img_size, ds.img_size), Image.BICUBIC
            ),
            dtype=np.int64,
        )
        d = np.abs(train - probe)
        total, n_px, worst = total + float(d.sum()), n_px + d.size, max(worst, int(d.max()))
        per_image.append(float(d.mean()))
    return {
        "n": len(rows),
        "mean_abs_diff": round(total / n_px, 6) if n_px else float("nan"),
        "max_abs_diff": int(worst),
        "worst_image_mean_abs_diff": round(max(per_image), 6) if per_image else float("nan"),
    }


# ------------------------------------------------------------------------------------ main
def _plan(args, key: str) -> list[tuple[str, str, object]]:
    """(kind, file name, thunk) for every figure / table that applies to the dataset."""
    root, n = args.root, args.n
    steps: list[tuple[str, str, object]] = []
    raw = args.raw_dir or args.arrow_dir
    if key == "chestxray14":
        for which in ("train", "val"):
            steps.append(
                ("figure", f"corpus_grid_{which}.png",
                 lambda w=which: corpus_grid(args.arrow_dir, args.index, w, n))
            )  # fmt: skip
    for split in SPLITS:
        steps.append(("figure", f"tree_grid_{split}.png", lambda s=split: tree_grid(key, s, n, root=root)))
    steps.append(("figure", "raw_vs_exported.png", lambda: raw_vs_exported(key, raw, root=root)))
    steps.append(("figure", "extreme_aspect.png", lambda: extreme_aspect(key, raw, root=root)))
    if key == "vindr_cxr_kaggle":
        steps.append(("figure", "photometric_pairs.png", lambda: photometric_pairs(tree_root(key, root))))
    steps.append(("figure", "size_histograms.png", lambda: size_histograms(key, root=root)))
    steps.append(("table", "prevalence.csv", lambda: prevalence(key, root=root)))
    if key == "chestxray14":
        steps.append(
            ("table", "resample_gap.csv",
             lambda: pd.DataFrame([resample_gap(args.arrow_dir, args.index, tree_root(key, root))]))
        )  # fmt: skip
    if args.gate_json:
        steps.append(("table", "gate_summary.csv", lambda: gate_summary(args.gate_json)))
    return steps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=KEYS)
    p.add_argument("--out", required=True, help="new directory (must not exist, nor <out>.partial)")
    p.add_argument("--root", default=None, help="tree root; default: the registry's")
    p.add_argument(
        "--raw-dir", default=None, help="chexpert: the zips; vindr_cxr_kaggle: the dir of the Kaggle zip"
    )
    p.add_argument(
        "--arrow-dir", default=None, help="chestxray14: the Arrow shards (also the source of raw_vs_exported)"
    )
    p.add_argument("--index", default=None, help="chestxray14: the alignment index npz")
    p.add_argument("--gate-json", default=None, help="leakage gate JSON for gate_summary")
    p.add_argument("--n", type=int, default=24, help="images per grid")
    p.add_argument(
        "--private-root",
        default=None,
        help="mode-700 directory (yours) that must hold --out; required for chexpert and vindr_cxr_kaggle",
    )
    args = p.parse_args(argv)
    key = args.dataset
    plt.switch_backend("Agg")
    WARN_COUNTS.clear()
    if key == "chestxray14" and (not args.arrow_dir or not args.index):
        raise SystemExit("chestxray14 needs --arrow-dir and --index")
    if key != "chestxray14" and not args.raw_dir:
        raise SystemExit(f"{key} needs --raw-dir")
    if args.n < 1:
        raise SystemExit("--n needs N >= 1")
    out = _resolve_out(key, args.out)  # the path checked below is the path written
    partial = f"{out}.partial"
    if key in RESTRICTED:
        _require_private(key, out, args.private_root)
    elif args.private_root is not None:
        if not _is_private_dir(args.private_root):
            raise SystemExit(f"{key}: --private-root is not a directory owned by you with mode 700")
        _require_under(key, out, args.private_root)
    if os.path.lexists(out) or os.path.lexists(partial):  # a dangling symlink at either counts
        raise SystemExit(
            f"{key}: --out or --out.partial already exists; pick a new --out (nothing is overwritten)"
        )
    os.makedirs(os.path.dirname(out), mode=0o700, exist_ok=True)
    os.mkdir(partial, 0o700)
    os.chmod(partial, 0o700)
    n_fig = n_tab = 0
    for kind, fname, thunk in _plan(args, key):
        with _count_only(key, f"{fname}"), X._counted_warnings(WARN_COUNTS):  # noqa: SLF001
            obj = thunk()
            if kind == "figure":
                panels = sum(1 for a in obj.axes if a.images)
                obj.savefig(os.path.join(partial, fname), dpi=110)
                plt.close(obj)
                n_fig += 1
                print(f"figure {fname} panels={panels}", flush=True)
            else:
                obj.to_csv(os.path.join(partial, fname), index=False)
                n_tab += 1
                print(f"table {fname} rows={len(obj)}", flush=True)
    os.rename(partial, out)
    if WARN_COUNTS:
        print(f"library warnings by category: {dict(sorted(WARN_COUNTS.items()))}")
    print(f"CXR_VISUAL_OK dataset={key} figures={n_fig} tables={n_tab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
