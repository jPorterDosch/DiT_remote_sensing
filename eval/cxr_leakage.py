"""Leakage gate of the chest X-ray data stage (spec 6.3).

An executed check, independent of the code that built the manifest and the trees: it re-reads the
shipped ChestX-ray14 CSV and list files, the alignment index npz and each probe tree, and prints one
PASS / FAIL line per check. It never imports datasets.cxr14 or calls a function of eval.cxr_splits; the
pinned counts of check 5 are literals below. OFFICIAL_CXR comes from eval.features (check 8 is defined
against it), and eval.features imports the split-size constants from eval.cxr_splits, so the split sizes of
check 8 are the same constants the exporter draws with: for CheXpert and VinDr-CXR check 8 confirms
"tree size equals the constant", not an independent count.

What the checks can and cannot see. Checks 1-5 are anchored on the shipped ChestX-ray14 files, so they are
independent of the exporter; check 0 first proves that --manifest-dir is the manifest the index was built
from (its own sha256 over the four manifest files against the hashes the index recorded), because the
index names of every later check come from that directory's rows.csv. Checks 6, 7 and 8 read columns the
exporter itself wrote (export_table.csv, labels.npz): they prove each CheXpert / VinDr-CXR tree is
consistent with itself, not that the exporter read the source archive correctly.

    python -m eval.cxr_leakage --index <npz> --manifest-dir <dir> --raw-dir <dir> \\
        --tree chestxray14=<root> --tree chexpert=<root> --tree vindr_cxr_kaggle=<root> --out <json>

Output is counts only, for every dataset: no image name, patient id, tree-row path component or label
row is ever printed, written to the JSON or put in an exception message. A tree that cannot be read
becomes `<check>: cannot read <file kind> of <key> (<ExceptionType>)`, never the exception text.

A tree that is absent prints `SKIPPED <key>`; the checks that need it are not run and the last line is
CXR_LEAKAGE_GATE INCOMPLETE (non-zero exit). FAIL outranks INCOMPLETE. There is no smoke mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from eval import features as F

# Counted from the shipped files (rule 16); deliberately not read from datasets.cxr14 / eval.cxr_splits.
PINNED = {
    "index_retained": 77_780,
    "index_validation": 8_744,
    "train_val_list": 86_524,
    "test_list": 25_596,
    "csv_rows": 112_120,
}
CSV_NAME = "Data_Entry_2017_v2020.csv"
TRAIN_VAL_LIST = "train_val_list.txt"
TEST_LIST = "test_list.txt"
SPLITS = ("train", "val", "test")
# The files an index is bound to. A literal: deliberately not read from datasets.cxr14.
MANIFEST_FILES = ("rows.csv", "retained.csv", "validation.csv", "report.json")

NOTES = (
    "published 86,523 / 25,595 vs shipped 86,524 / 25,596 (list files lack a final newline)",
    "no patient identifier in this release: patient overlap cannot be checked",
    "the gate cannot see which split a training run loads: alignment must load split train (bucket retained) only",
)

C0 = "0 index_bound_to_manifest"
C1 = "1 index_patients_disjoint_from_tree_eval"
C1B = "1b index_patient_id_matches_csv"
C1C = "1c tree_patient_column_matches_csv"
C2 = "2 index_names_disjoint_from_tree_val_test"
C2B = "2b index_retained_validation_patients_disjoint"
C3 = "3 official_lists_patient_disjoint"
C4 = "4 tree_splits_inside_index_buckets"
C5 = "5 pinned_counts"
C6 = "6 chexpert_patient_disjoint_and_frontal"
C7 = "7 vindr_no_shared_image_id"


def C8(key: str) -> str:
    return f"8 tree_sizes_match_official[{key}]"


class _Unreadable(Exception):
    """An input could not be read. Carries only the file kind, the key and the exception type."""

    def __init__(self, kind: str, key: str, exc: BaseException):
        super().__init__(f"cannot read {kind} of {key} ({type(exc).__name__})")


def _canon(x) -> str | None:
    """Canonical patient form str(int(x)); None when x is not an integer string."""
    try:
        return str(int(str(x)))
    except (TypeError, ValueError):
        return None


class _Inputs:
    """Lazy, cached readers. A failed read is cached too and re-raised as _Unreadable."""

    def __init__(self, index_npz, manifest_dir, raw_dir, trees):
        self.index_npz, self.manifest_dir, self.raw_dir = index_npz, manifest_dir, raw_dir
        self.trees = trees
        self._cache: dict = {}

    def _get(self, tag, kind, key, fn):
        if tag not in self._cache:
            try:
                self._cache[tag] = (True, fn())
            except Exception as e:  # noqa: BLE001 - the text of e is never shown
                self._cache[tag] = (False, _Unreadable(kind, key, e))
        ok, val = self._cache[tag]
        if not ok:
            raise val
        return val

    # ---- shipped label files
    def csv_map(self) -> tuple[dict, int]:
        """(name -> canonical patient, None when ambiguous or not an integer; number of rows)."""

        def read():
            df = pd.read_csv(
                os.path.join(self.raw_dir, CSV_NAME),
                dtype=str,
                keep_default_na=False,
                usecols=["Image Index", "Patient ID"],
            )
            dup = df["Image Index"].duplicated(keep=False)
            m = {n: _canon(p) for n, p in zip(df["Image Index"], df["Patient ID"], strict=True)}
            for n in df.loc[dup, "Image Index"].unique():
                m[n] = None  # ambiguous: never resolves
            return m, len(df)

        return self._get("csv", "label CSV", "raw dir", read)

    def list_names(self, fname: str) -> list[str]:
        def read():
            with open(os.path.join(self.raw_dir, fname)) as f:
                return [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        return self._get(("list", fname), "list file", "raw dir", read)

    # ---- index
    def manifest_binding(self) -> tuple[dict, dict]:
        """(sha256 per manifest file as the index recorded it, sha256 of the files in --manifest-dir now)."""

        def read():
            z = np.load(self.index_npz, allow_pickle=False)
            recorded = json.loads(str(z["manifest_sha256"]))
            if not isinstance(recorded, dict):
                raise TypeError("manifest_sha256 is not a mapping")
            now = {}
            for name in MANIFEST_FILES:
                h = hashlib.sha256()
                with open(os.path.join(self.manifest_dir, name), "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 24), b""):
                        h.update(chunk)
                now[name] = h.hexdigest()
            return recorded, now

        return self._get("binding", "index npz or manifest files", "index", read)

    def index(self) -> dict:
        def read():
            z = np.load(self.index_npz, allow_pickle=False)
            rec = {k: z[k] for k in ("shard", "row", "bucket_code", "buckets", "patient_id")}
            rows = pd.read_csv(
                os.path.join(self.manifest_dir, "rows.csv"),
                dtype={"img_name": str},
                usecols=["shard", "row", "img_name"],
            )
            return rec, rows

        rec, rows = self._get("index", "index npz or rows.csv", "index", read)
        return self._index_names(rec, rows)

    def _index_names(self, rec, rows) -> dict:
        key = self._cache.get("index_names")
        if key is not None:
            return key
        pos = rows.groupby(["shard", "row"]).size()
        once = pos[pos == 1].index
        look = rows.set_index(["shard", "row"])["img_name"]
        pairs = pd.MultiIndex.from_arrays([rec["shard"].astype(np.int64), rec["row"].astype(np.int64)])
        ok = pairs.isin(once)
        names = np.array([look[p] if o else "" for p, o in zip(pairs, ok, strict=True)], dtype=object)
        buckets = [str(b) for b in rec["buckets"]]
        out = {
            "names": names,
            "unmatched": int((~ok).sum()),
            "patient_id": rec["patient_id"],
            "bucket": np.array([buckets[c] for c in rec["bucket_code"]], dtype=object),
        }
        self._cache["index_names"] = out
        return out

    # ---- trees
    def has_tree(self, key) -> bool:
        p = self.trees.get(key)
        return bool(p) and os.path.isdir(p)

    def table(self, key) -> pd.DataFrame:
        return self._get(
            ("table", key),
            "export_table.csv",
            key,
            lambda: pd.read_csv(
                os.path.join(self.trees[key], "export_table.csv"), dtype=str, keep_default_na=False
            ),
        )

    def label_names(self, key, split) -> tuple[list[str], int]:
        def read():
            z = np.load(os.path.join(self.trees[key], split, "labels.npz"), allow_pickle=False)
            names = [str(n) for n in z["names"]]
            return names, int(z["y"].shape[0])

        return self._get(("labels", key, split), "labels.npz", key, read)

    def png_count(self, key, split) -> int:
        return self._get(
            ("png", key, split),
            "split directory",
            key,
            lambda: sum(1 for f in os.listdir(os.path.join(self.trees[key], split)) if f.endswith(".png")),
        )


def _resolve(names, csv_map) -> tuple[list, int]:
    """Canonical patients of `names` (suffix .png included); count of names that do not resolve to
    exactly one CSV row with an integer patient. Unresolved names get None."""
    pats = [csv_map.get(n) for n in names]
    return pats, sum(p is None for p in pats)


def _tree_names14(inp, split) -> list[str]:
    t = inp.table("chestxray14")
    return [n + ".png" for n in t.loc[t["split"] == split, "name"]]


def _unresolved(n) -> str:
    return f"{n} names do not resolve to exactly one CSV row"


# ------------------------------------------------------------------------------ checks
def _check0(inp: _Inputs):
    """The index names of checks 1, 2, 2b, 1b, 4 and 5 come from <manifest-dir>/rows.csv, so a manifest
    directory that is not the one the index was built from would make them compare the wrong images."""
    recorded, now = inp.manifest_binding()
    one_side = set(recorded) ^ set(now)
    differ = sum(recorded[f] != now[f] for f in now if f in recorded)
    return [
        (
            C0,
            not one_side and not differ,
            f"{len(now)} manifest files, {differ} differ from the sha256 the index recorded, "
            f"{len(one_side)} named on one side only",
        )
    ]


def _check1_2(inp: _Inputs):
    """(a) rows alignment trains on (bucket retained) vs tree val + test; (b) ALL index rows vs tree test.
    The validation bucket is the source of the tree val, so it is deliberately not compared with it."""
    csv_map = inp.csv_map()[0]
    idx = inp.index()
    ret = np.array([b == "retained" for b in idx["bucket"]], dtype=bool)
    ipat, bad_i = _resolve(idx["names"], csv_map)
    ipat = np.array(ipat, dtype=object)
    val_n, test_n = _tree_names14(inp, "val"), _tree_names14(inp, "test")
    vpat, bad_v = _resolve(val_n, csv_map)
    tpat, bad_t = _resolve(test_n, csv_map)
    bad = bad_i + bad_v + bad_t
    inames = np.array(idx["names"], dtype=object)
    nm_all = {n for n in inames if n}
    nm_ret = {n for n in inames[ret] if n}
    out = []
    if idx["unmatched"] or bad:
        out.append(
            (
                C1,
                False,
                f"cannot compare: {idx['unmatched']} index rows not in rows.csv exactly once, "
                f"{bad_i} index names and {bad_v + bad_t} tree names do not resolve to exactly one CSV row",
            )
        )
    else:
        a = set(ipat[ret]) & (set(vpat) | set(tpat))
        b = set(ipat) & set(tpat)
        out.append(
            (
                C1,
                not a and not b and bool(ret.any()) and bool(vpat) and bool(tpat),
                f"retained patients {len(set(ipat[ret]))}, tree val+test patients {len(set(vpat) | set(tpat))}, "
                f"retained vs val+test shared {len(a)}; all index patients {len(set(ipat))}, "
                f"tree test patients {len(set(tpat))}, all vs test shared {len(b)}",
            )
        )
    sa = nm_ret & (set(val_n) | set(test_n))
    sb = nm_all & set(test_n)
    out.append(
        (
            C2,
            not sa
            and not sb
            and not idx["unmatched"]
            and not bad
            and bool(nm_ret)
            and bool(val_n)
            and bool(test_n),
            f"retained names {len(nm_ret)}, tree val+test names {len(set(val_n) | set(test_n))}, "
            f"retained vs val+test shared {len(sa)}; all index names {len(nm_all)}, "
            f"tree test names {len(set(test_n))}, all vs test shared {len(sb)}; "
            f"{idx['unmatched']} index rows without a name, {_unresolved(bad)}",
        )
    )
    return out


def _check2b(inp: _Inputs):
    csv_map = inp.csv_map()[0]
    idx = inp.index()
    pats, bad = _resolve(idx["names"], csv_map)
    pats = np.array(pats, dtype=object)
    ret = np.array([b == "retained" for b in idx["bucket"]], dtype=bool)
    val = np.array([b == "validation" for b in idx["bucket"]], dtype=bool)
    shared = set(pats[ret]) & set(pats[val])
    return [
        (
            C2B,
            not shared and not bad and not idx["unmatched"] and bool(ret.any()) and bool(val.any()),
            f"retained patients {len(set(pats[ret]))}, validation patients {len(set(pats[val]))}, "
            f"{len(shared)} shared, {_unresolved(bad)}",
        )
    ]


def _check1b(inp: _Inputs):
    csv_map = inp.csv_map()[0]
    idx = inp.index()
    pats, bad = _resolve(idx["names"], csv_map)
    mism = sum(p is not None and str(int(a)) != p for p, a in zip(pats, idx["patient_id"], strict=True))
    return [
        (
            C1B,
            bad == 0 and mism == 0 and not idx["unmatched"],
            f"{len(pats)} index rows, {bad} unresolved, {mism} disagree",
        )
    ]


def _check1c(inp: _Inputs):
    csv_map = inp.csv_map()[0]
    t = inp.table("chestxray14")
    pats, bad = _resolve([n + ".png" for n in t["name"]], csv_map)
    col = [_canon(p) for p in t["patient"]]
    mism = sum(p is not None and c != p for p, c in zip(pats, col, strict=True))
    return [(C1C, bad == 0 and mism == 0, f"{len(pats)} tree rows, {bad} unresolved, {mism} disagree")]


def _check3(inp: _Inputs):
    csv_map = inp.csv_map()[0]
    tv, te = inp.list_names(TRAIN_VAL_LIST), inp.list_names(TEST_LIST)
    ptv, b1 = _resolve(tv, csv_map)
    pte, b2 = _resolve(te, csv_map)
    if b1 or b2:
        return [(C3, False, f"cannot compare: {_unresolved(b1 + b2)}")]
    shared = set(ptv) & set(pte)
    return [
        (
            C3,
            not shared,
            f"{len(set(ptv))} train/val patients, {len(set(pte))} test patients, {len(shared)} shared",
        )
    ]


def _check4(inp: _Inputs):
    idx = inp.index()
    sets = {
        b: {n for n, bb in zip(idx["names"], idx["bucket"], strict=True) if bb == b and n}
        for b in ("retained", "validation")
    }
    tr, va, te = (set(_tree_names14(inp, s)) for s in SPLITS)
    want_te = set(inp.list_names(TEST_LIST))
    a, b = len(tr - sets["retained"]), len(va - sets["validation"])
    c, d = len(te - want_te), len(want_te - te)
    ok = a == b == c == d == 0 and not idx["unmatched"]
    return [
        (
            C4,
            ok,
            f"{a} train names outside retained, {b} val names outside validation, "
            f"{c} test names not in the official list, {d} official test names missing, "
            f"{idx['unmatched']} index rows without a name",
        )
    ]


def _check5(inp: _Inputs):
    idx = inp.index()
    got = {
        "index_retained": int((idx["bucket"] == "retained").sum()),
        "index_validation": int((idx["bucket"] == "validation").sum()),
        "train_val_list": len(inp.list_names(TRAIN_VAL_LIST)),
        "test_list": len(inp.list_names(TEST_LIST)),
        "csv_rows": inp.csv_map()[1],
    }
    bad = [k for k in PINNED if got[k] != PINNED[k]]
    detail = f"index {got['index_retained']} + {got['index_validation']}, lists {got['train_val_list']} / {got['test_list']}, CSV {got['csv_rows']}"
    if bad:
        detail += f"; {len(bad)} differ from the pinned counts ({', '.join(bad)})"
    return [(C5, not bad, detail)]


def _check6(inp: _Inputs):
    t = inp.table("chexpert")
    fl = t["frontal_lateral"] if "frontal_lateral" in t else pd.Series([""] * len(t))
    non_frontal = int((fl != "Frontal").sum())
    blank = int((t["patient"].str.strip() == "").sum())
    per = {s: set(t.loc[t["split"] == s, "patient"]) for s in SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    shared = {f"{a}/{b}": len(per[a] & per[b]) for a, b in pairs}
    ok = not blank and not non_frontal and not any(shared.values()) and len(t) > 0
    return [
        (
            C6,
            ok,
            f"{len(t)} rows, {blank} without a patient, {non_frontal} not frontal, shared patients "
            + ", ".join(f"{k} {v}" for k, v in shared.items()),
        )
    ]


def _check7(inp: _Inputs):
    t = inp.table("vindr_cxr_kaggle")
    per = {s: list(t.loc[t["split"] == s, "name"]) for s in SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    shared = {f"{a}/{b}": len(set(per[a]) & set(per[b])) for a, b in pairs}
    ok = not any(shared.values()) and len(t) > 0
    return [(C7, ok, f"{len(t)} rows, shared image ids " + ", ".join(f"{k} {v}" for k, v in shared.items()))]


def _check8(inp: _Inputs, key):
    sizes = F.OFFICIAL_CXR[key]["sizes"]
    t = inp.table(key)
    probs = []
    parts = []
    for split, want in sizes.items():
        tn = list(t.loc[t["split"] == split, "name"])
        ln, ny = inp.label_names(key, split)
        npng = inp.png_count(key, split)
        parts.append(f"{split} {len(tn)}/{len(ln)}/{npng} (want {want})")
        if not (len(tn) == len(ln) == ny == npng == want):
            probs.append(split)
        elif set(tn) != set(ln) or len(set(tn)) != len(tn):
            probs.append(split)
    extra = sorted(set(t["split"]) - set(sizes))
    if extra:
        probs.append(f"{len(extra)} unexpected splits")
    detail = "table/labels/png " + "; ".join(parts)
    if probs:
        detail += f"; {len(probs)} problems"
    return [(C8(key), not probs, detail)]


def _run_one(results, names, fn, inp):
    """Run one check function; a failure to read or compute becomes one FAIL line per check name."""
    try:
        results.extend(fn(inp))
    except _Unreadable as e:
        results.extend((n, False, str(e)) for n in names)
    except Exception as e:  # noqa: BLE001 - counts only: the text is never shown
        results.extend((n, False, f"check raised {type(e).__name__}") for n in names)


def _run(index_npz, manifest_dir, raw_dir, trees):
    inp = _Inputs(index_npz, manifest_dir, raw_dir, trees)
    keys = sorted(F.OFFICIAL_CXR)
    present = {k: inp.has_tree(k) for k in keys}
    skipped = [k for k in keys if not present[k]]
    r: list = []
    _run_one(r, (C0,), _check0, inp)
    if present["chestxray14"]:
        _run_one(r, (C1, C2), _check1_2, inp)
    _run_one(r, (C2B,), _check2b, inp)
    if present["chestxray14"]:
        _run_one(r, (C1C,), _check1c, inp)
        _run_one(r, (C4,), _check4, inp)
    _run_one(r, (C1B,), _check1b, inp)
    _run_one(r, (C3,), _check3, inp)
    _run_one(r, (C5,), _check5, inp)
    if present["chexpert"]:
        _run_one(r, (C6,), _check6, inp)
    if present["vindr_cxr_kaggle"]:
        _run_one(r, (C7,), _check7, inp)
    for k in keys:
        if present[k]:
            _run_one(r, (C8(k),), lambda i, k=k: _check8(i, k), inp)
    return r, skipped


def check(
    index_npz: str, manifest_dir: str, raw_dir: str, trees: dict[str, str]
) -> list[tuple[str, bool, str]]:
    """(check name, passed, one-line aggregate detail) for every check whose inputs exist."""
    return _run(index_npz, manifest_dir, raw_dir, trees)[0]


def verdict(results, skipped) -> str:
    if any(not ok for _, ok, _ in results):
        return "FAIL"
    return "INCOMPLETE" if skipped else "PASS"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True, help="alignment index npz")
    p.add_argument("--manifest-dir", required=True, help="the manifest directory the index was built from")
    p.add_argument("--raw-dir", required=True, help="shipped ChestX-ray14 CSV and list files")
    p.add_argument(
        "--tree", action="append", default=[], metavar="KEY=PATH", help="probe tree root, repeatable"
    )
    p.add_argument("--out", required=True, help="result JSON; must not exist")
    args = p.parse_args(argv)
    if os.path.exists(args.out) or os.path.exists(args.out + ".partial"):
        raise SystemExit(
            f"{args.out} or {args.out}.partial exists; pick a new --out (nothing is overwritten)"
        )
    trees: dict[str, str] = {}
    for spec in args.tree:
        key, sep, path = spec.partition("=")
        if not sep or key not in F.OFFICIAL_CXR or not path or key in trees:
            raise SystemExit(f"--tree wants KEY=PATH once per key, KEY one of {sorted(F.OFFICIAL_CXR)}")
        trees[key] = path
    results, skipped = _run(args.index, args.manifest_dir, args.raw_dir, trees)
    lines = [f"{'PASS' if ok else 'FAIL'} {name}: {detail}" for name, ok, detail in results]
    lines += [f"SKIPPED {k}" for k in skipped]
    lines += [f"NOTE {n}" for n in NOTES]
    v = verdict(results, skipped)
    final = f"CXR_LEAKAGE_GATE {v}"
    print("\n".join([*lines, final]))
    rec = {
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in results],
        "skipped": skipped,
        "notes": list(NOTES),
        "verdict": final,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".partial", "w") as f:
        json.dump(rec, f, indent=1)
    os.rename(args.out + ".partial", args.out)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
