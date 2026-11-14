"""Measure (do not change) which of the retained Git-10M rows that eval.seg_overlap found near a
segmentation task were actually DRAWN by the training runs (option B, 2026-11-12).

eval.seg_overlap counts rows that are in the retained manifest and within the buffer of a
sample. A run of 20,000 steps x 8 rows reads 160,000 of ~8.4M training rows, so most of those
rows were never read. This tool answers, per run: how many of them did the optimizer see.

How a run's rows are known, strongest evidence first (the `verification` field of each run):
  logged       stats.npz holds `step/micro_row` (AlignStats.log_micro: the index position of
               every row of every microbatch). The reconstruction below must equal it exactly.
  fingerprint  no row log (runs older than the log). The run has the same loader configuration
               as a logged run of the same seed AND its step-0 flow loss is bit-equal to that
               run's, so the first optimizer step read the same rows with the same t and noise.
               The rows after step 0 are the reconstruction, not a log.
  unverified   reconstruction only. Quote with that label.

Reconstruction = the training code itself, on a stand-in dataset that returns its own position:
  main     DataLoader(sampler=datasets.git10m.PermutationSampler(n, seed), drop_last=True),
           steps x accumulation batches (Git10MWrapper.train_loader)
  warm-up  DataLoader(shuffle=True, generator=manual_seed(seed + 1), drop_last=True),
           proj_warmup_steps batches (train_diffusion.py, projector only, LoRA frozen). These
           rows are in no log, so the warm-up count is always a reconstruction.
A position in the train view maps to a row of the index npz through the holdout rule
(datasets.git10m.holdout_mask), exactly as Git10MDataset(which="train") does.

Measurement batches (meas_src) make no update and are not counted. `expected_uniform` is the
mean of a uniform draw of the same size; it is an expectation, not a measurement.

Gates (rule 13), each printed PASS / FAIL:
  matched rows   the match list holds exactly the retained-row count of the seg_overlap report
  partition      matched rows = not in index + holdout + train view
  train size     the train view has the `train_size` every run's meta.json records
  loader config  every run trained from this index, unresumed, uncapped, one pass at most
  row log        every logged run equals the reconstruction (no unfilled slot)
  control        planted rows: drawn ones are counted, holdout and undrawn ones are not

    python -m eval.seg_overlap_drawn --index <index.npz> --matches <prefix>.retained_matches.csv \
        --overlap-json <prefix>.json --dataset m_pv4ger_seg --runs-root <models/git10m> --out <prefix>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.git10m import ROW_ID_STRIDE, PermutationSampler, holdout_mask  # noqa: E402

MATCH_HEADER = ["shard", "row", "dataset", "split", "sample_name", "distance_m", "img_name", "status"]
ROW_HEADER = ["seed", "phase", "step", "slot", "index_pos", "shard", "row", "img_name",
              "eval_samples_near", "min_distance_m", "splits"]  # fmt: skip
FINGERPRINT_STEPS = 100


class _Positions(Dataset):
    """Stand-in for the train view: item i is i, so a loader over it yields the sampler's order."""

    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return int(i)


def _take(loader: DataLoader, n_batches: int) -> np.ndarray:
    """First n_batches of ONE pass, as an (n_batches, batch) array of train-view positions."""
    if n_batches > len(loader):
        raise ValueError(
            f"{n_batches} batches > one pass of {len(loader)}: a second pass is not reconstructed here"
        )
    out = np.full((n_batches, loader.batch_size), -1, dtype=np.int64)
    for k, b in enumerate(loader):
        if k == n_batches:
            break
        out[k] = np.asarray(b, dtype=np.int64)
    if (out < 0).any():
        raise RuntimeError("loader ended before the requested batches")
    return out


def main_draw(n_train: int, seed: int, batch: int, n_batches: int, num_workers: int = 0) -> np.ndarray:
    """Train-view positions of the main loop, in consumption order (Git10MWrapper.train_loader)."""
    loader = DataLoader(
        _Positions(n_train),
        batch_size=batch,
        sampler=PermutationSampler(n_train, seed, epoch=0, pos=0),
        num_workers=num_workers,
        drop_last=True,
    )
    return _take(loader, n_batches)


def warm_draw(n_train: int, seed: int, batch: int, n_batches: int, num_workers: int = 0) -> np.ndarray:
    """Train-view positions of the projector warm-up (train_diffusion.py: its own seeded loader)."""
    if n_batches == 0:
        return np.empty((0, batch), dtype=np.int64)
    loader = DataLoader(
        _Positions(n_train),
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed + 1),
        drop_last=True,
    )
    return _take(loader, n_batches)


def load_index(path: str, holdout_every: int) -> dict:
    z = np.load(path)
    if not ("one_row_per_image" in z.files and bool(z["one_row_per_image"])):
        raise ValueError(f"{path}: built before the one-row-per-image rule")
    shard, row = z["shard"].astype(np.int64), z["row"].astype(np.int64)
    ids = shard * ROW_ID_STRIDE + row
    order = np.argsort(ids, kind="stable")
    if (np.diff(ids[order]) == 0).any():
        raise ValueError(f"{path}: a (shard, row) appears twice")
    hold = holdout_mask(shard, row, holdout_every)
    return {
        "shard": shard,
        "row": row,
        "ids": ids,
        "order": order,
        "hold": hold,
        "train_pos": np.flatnonzero(~hold),
        "buckets": [str(b) for b in z["buckets"]],
        "bucket_code": z["bucket_code"].astype(np.int64),
    }


def read_matches(path: str, dataset: str) -> list[dict]:
    """Rows of a seg_overlap match list for one dataset; the header and the status are validated."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != MATCH_HEADER:
            raise ValueError(f"{path}: header {header} != {MATCH_HEADER}")
        rows = [dict(zip(MATCH_HEADER, x, strict=True)) for x in r]
    out = [x for x in rows if x["dataset"] == dataset]
    if not out:
        raise ValueError(
            f"{path}: no row for dataset {dataset!r} (has {sorted({x['dataset'] for x in rows})})"
        )
    bad = sorted({x["status"] for x in out} - {"retained"})
    if bad:
        raise ValueError(f"{path}: status {bad} in a retained-match list")
    for x in out:
        x["shard"], x["row"], x["distance_m"] = int(x["shard"]), int(x["row"]), float(x["distance_m"])
        if not np.isfinite(x["distance_m"]) or x["distance_m"] < 0:
            raise ValueError(f"{path}: distance_m {x['distance_m']} for {x['shard']},{x['row']}")
    return out


def locate(index: dict, ids: np.ndarray) -> np.ndarray:
    """Index position of each row id, -1 where the index does not hold it."""
    s = index["ids"][index["order"]]
    k = np.searchsorted(s, ids)
    k[k == len(s)] = 0
    return np.where(s[k] == ids, index["order"][k], -1)


def count_drawn(match_pos: np.ndarray, drawn_pos: np.ndarray) -> np.ndarray:
    """Boolean per matched row: its index position is among the drawn index positions."""
    return np.isin(match_pos, drawn_pos) & (match_pos >= 0)


def near_samples(pairs: list[dict], ids: set[int]) -> dict:
    """Evaluation samples within the buffer of (and overlapped by) any row of `ids`, per split."""
    out: dict = {}
    for x in pairs:
        if x["shard"] * ROW_ID_STRIDE + x["row"] not in ids:
            continue
        s = out.setdefault(x["split"], {"near": set(), "over": set()})
        s["near"].add(x["sample_name"])
        if x["distance_m"] == 0.0:
            s["over"].add(x["sample_name"])
    return {
        k: {"within_buffer": len(v["near"]), "overlapped": len(v["over"])} for k, v in sorted(out.items())
    }


def run_facts(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "meta.json")) as f:
        m = json.load(f)
    c = m["cfg"]
    accum = int(c["gradient_accumulation_steps"]) if c["use_gradient_accumulation"] else 1
    facts = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "seed": int(c["seed"]),
        "batch": int(c["batch_size"]),
        "accum": accum,
        "steps": int(c["max_train_steps"]),
        "warmup_steps": int(c["align"]["proj_warmup_steps"]),
        "index": c["git10m"]["index"],
        "holdout_every": int(c["git10m"]["holdout_every"]),
        "max_samples": c["max_samples"],
        "resumed_from": m["resumed_from"],
        "resume": c["align"]["resume"],
        "train_size": int(m["train_size"]),
        "samples_seen": int(m["samples_seen"]),
        "micro_row": None,
        "flow_loss": None,
    }
    stats = os.path.join(run_dir, "stats.npz")
    if os.path.isfile(stats):
        z = np.load(stats)
        if "step/micro_row" in z.files:
            facts["micro_row"] = z["step/micro_row"].astype(np.int64)
        if "step/flow_loss" in z.files:
            facts["flow_loss"] = z["step/flow_loss"][:FINGERPRINT_STEPS].astype(np.float64)
    return facts


def config_problems(r: dict, index_path: str, holdout_every: int, n_train: int) -> list[str]:
    """Why this run's rows are NOT the draw reconstructed here (empty = none found)."""
    p = []
    if os.path.realpath(r["index"]) != os.path.realpath(index_path):
        p.append(f"index {r['index']}")
    if r["holdout_every"] != holdout_every:
        p.append(f"holdout_every {r['holdout_every']}")
    if r["max_samples"] is not None:
        p.append(f"max_samples {r['max_samples']}")
    if r["resumed_from"] or r["resume"]:
        p.append(f"resumed ({r['resumed_from'] or r['resume']})")
    if r["train_size"] != n_train:
        p.append(f"train_size {r['train_size']} != train view {n_train}")
    if r["samples_seen"] != r["steps"] * r["accum"] * r["batch"]:
        p.append(f"samples_seen {r['samples_seen']} != {r['steps']} x {r['accum']} x {r['batch']}")
    if r["steps"] * r["accum"] > n_train // r["batch"]:
        p.append("more than one pass")
    return p


def _key(r: dict) -> tuple:
    return (r["seed"], r["batch"], r["accum"], r["steps"], r["warmup_steps"])


def control(index: dict, main_pos: np.ndarray) -> dict:
    """Planted rows through count_drawn: 3 drawn, 3 holdout, 3 in the train view but not drawn,
    3 absent from the index. Exactly the 3 drawn ones may count."""
    drawn = main_pos[:3]
    hold = np.flatnonzero(index["hold"])[:3]
    undrawn = np.setdiff1d(index["train_pos"], main_pos)[:3]
    absent = np.full(3, -1, dtype=np.int64)
    if min(len(drawn), len(hold), len(undrawn)) < 3:
        return {"pass": False, "detail": "index too small to plant 3 + 3 + 3 rows"}
    planted = np.concatenate([drawn, hold, undrawn, absent])
    got = count_drawn(planted, main_pos)
    want = np.array([True] * 3 + [False] * 9)
    return {"pass": bool((got == want).all()), "detail": f"planted 12 rows, counted {int(got.sum())} (expected 3 drawn)"}  # fmt: skip


def measure(index_path: str, holdout_every: int, pairs: list[dict], run_dirs: list[str],
            expect_matched: int | None) -> tuple[dict, list[list]]:  # fmt: skip
    index = load_index(index_path, holdout_every)
    train_pos, n_train = index["train_pos"], int(len(index["train_pos"]))
    gates: list[dict] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    by_id: dict[int, dict] = {}
    for x in pairs:
        by_id.setdefault(x["shard"] * ROW_ID_STRIDE + x["row"], x)
    m_ids = np.array(sorted(by_id), dtype=np.int64)
    m_pos = locate(index, m_ids)
    in_index = m_pos >= 0
    hold = np.zeros(len(m_ids), dtype=bool)
    hold[in_index] = index["hold"][m_pos[in_index]]
    train_view = in_index & ~hold
    bucket = {
        b: int((index["bucket_code"][m_pos[in_index]] == i).sum()) for i, b in enumerate(index["buckets"])
    }
    matched = {
        "total": int(len(m_ids)),
        "not_in_index": int((~in_index).sum()),
        "in_index": int(in_index.sum()),
        "holdout": int(hold.sum()),
        "train_view": int(train_view.sum()),
        "in_index_by_bucket": bucket,
    }
    if expect_matched is not None:
        gate("matched rows", matched["total"] == expect_matched,
             f"{matched['total']} distinct rows in the match list, seg_overlap report says {expect_matched}")  # fmt: skip
    gate("partition", matched["total"] == matched["not_in_index"] + matched["holdout"] + matched["train_view"]
         and sum(bucket.values()) == matched["in_index"],
         f"{matched['total']} = {matched['not_in_index']} not in index + {matched['holdout']} holdout + {matched['train_view']} train view")  # fmt: skip

    runs = [run_facts(d) for d in run_dirs]
    if not runs:
        raise ValueError("no run directory with a meta.json")
    sizes = sorted({r["train_size"] for r in runs})
    gate("train size", sizes == [n_train], f"train view {n_train:,} rows, meta.json train_size {sizes}")
    bad_cfg = {r["run"]: p for r in runs if (p := config_problems(r, index_path, holdout_every, n_train))}
    gate("loader config", not bad_cfg, f"{len(runs) - len(bad_cfg)}/{len(runs)} runs match"
         + ("" if not bad_cfg else f"; mismatches {json.dumps(bad_cfg)}"))  # fmt: skip

    draws, rows_out, log_bad, ctrl = {}, [], {}, None
    for key in sorted({_key(r) for r in runs if r["run"] not in bad_cfg}):
        seed, batch, accum, steps, warm_steps = key
        main = main_draw(n_train, seed, batch, steps * accum)  # (steps * accum, batch) train-view positions
        warm = warm_draw(n_train, seed, batch, warm_steps)
        main_pos, warm_pos = train_pos[main.ravel()], train_pos[warm.ravel()]
        if ctrl is None:
            ctrl = control(index, main_pos)
        group = [r for r in runs if _key(r) == key and r["run"] not in bad_cfg]
        ref = None
        for r in group:
            if r["micro_row"] is None:
                continue
            want = main_pos.reshape(steps, accum * batch)
            if r["micro_row"].shape != want.shape:
                log_bad[r["run"]] = f"shape {r['micro_row'].shape} != {want.shape}"
            elif (r["micro_row"] < 0).any():
                log_bad[r["run"]] = f"{int((r['micro_row'] < 0).sum())} unfilled slots"
            elif not np.array_equal(r["micro_row"], want):
                log_bad[r["run"]] = f"{int((r['micro_row'] != want).sum())} of {want.size} slots differ"
            else:
                r["verification"] = "logged"
                ref = ref or r
        for r in group:
            if r["micro_row"] is not None:
                r.setdefault("verification", "LOG MISMATCH")
                continue
            r["verification"] = "unverified"
            if ref is not None and r["flow_loss"] is not None and ref["flow_loss"] is not None:
                r["reference"] = ref["run"]
                r["step0_flow_loss"], r["reference_step0_flow_loss"] = (
                    float(r["flow_loss"][0]),
                    float(ref["flow_loss"][0]),
                )
                k = min(len(r["flow_loss"]), len(ref["flow_loss"]))
                r["flow_loss_median_abs_diff"] = float(
                    np.median(np.abs(r["flow_loss"][:k] - ref["flow_loss"][:k]))
                )
                if r["flow_loss"][0] == ref["flow_loss"][0]:
                    r["verification"] = "fingerprint"

        d = {
            "seed": seed,
            "batch": batch,
            "accumulation": accum,
            "steps": steps,
            "runs": [r["run"] for r in group],
        }
        for phase, pos in (
            ("main", main_pos),
            ("warmup", warm_pos),
            ("main_or_warmup", np.concatenate([main_pos, warm_pos])),
        ):
            hit = count_drawn(m_pos, pos)
            d[phase] = {
                "samples": int(len(pos)),
                "unique_rows": int(len(np.unique(pos))),
                "matched_rows_drawn": int(hit.sum()),
                "expected_uniform": float(matched["train_view"] * len(pos) / n_train),
                "eval_samples_near_a_drawn_row": near_samples(pairs, set(m_ids[hit].tolist())),
            }
        d["warmup"]["source"] = "reconstruction (the warm-up rows are in no log)"
        draws[str(seed) if str(seed) not in draws else f"{seed}:{batch}x{accum}x{steps}"] = d
        pos_of = dict(zip(m_pos[in_index].tolist(), m_ids[in_index].tolist(), strict=True))
        for phase, pos, per_step in (("main", main_pos, accum * batch), ("warmup", warm_pos, batch)):
            for k in np.flatnonzero(np.isin(pos, m_pos[in_index])):
                x = by_id[pos_of[int(pos[k])]]
                near = [y for y in pairs if (y["shard"], y["row"]) == (x["shard"], x["row"])]
                rows_out.append([seed, phase, int(k // per_step), int(k % per_step), int(pos[k]), x["shard"], x["row"],
                                 x["img_name"], len({y["sample_name"] for y in near}),
                                 f"{min(y['distance_m'] for y in near):.6f}", "+".join(sorted({y["split"] for y in near}))])  # fmt: skip

    logged = [r for r in runs if r.get("verification") == "logged"]
    gate("row log", bool(logged) and not log_bad,
         f"{len(logged)} logged runs equal the reconstruction slot for slot"
         + ("" if not log_bad else f"; MISMATCH {json.dumps(log_bad)}"))  # fmt: skip
    ctrl = ctrl or {"pass": False, "detail": "no draw was reconstructed"}
    gate("control", ctrl["pass"], ctrl["detail"])

    keep = ("run", "seed", "verification", "reference", "step0_flow_loss", "reference_step0_flow_loss",
            "flow_loss_median_abs_diff")  # fmt: skip
    rep = {
        "index": index_path,
        "index_rows": int(len(index["ids"])),
        "holdout_every": holdout_every,
        "train_view_rows": n_train,
        "matched_rows": matched,
        "draws": draws,
        "runs": [
            {k: r[k] for k in keep if k in r}
            | ({"config_mismatch": bad_cfg[r["run"]]} if r["run"] in bad_cfg else {})
            for r in runs
        ],  # fmt: skip
        "verification_counts": {
            v: sum(r.get("verification") == v for r in runs)
            for v in ("logged", "fingerprint", "unverified", "LOG MISMATCH")
        },  # fmt: skip
        "gates": gates,
        "pass": all(g["pass"] for g in gates),
    }
    return rep, rows_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="the training index npz the runs trained from; read only")
    ap.add_argument("--holdout-every", type=int, required=True)
    ap.add_argument("--matches", required=True, help="<prefix>.retained_matches.csv of eval.seg_overlap")
    ap.add_argument(
        "--overlap-json", required=True, help="<prefix>.json of eval.seg_overlap (row-count gate)"
    )
    ap.add_argument("--dataset", required=True, help="task in the match list, e.g. m_pv4ger_seg")
    ap.add_argument(
        "--runs-root", required=True, help="dir of run dirs (each with meta.json, stats.npz); read only"
    )
    ap.add_argument("--expect-runs", type=int, required=True, help="number of run dirs that must be found")
    ap.add_argument("--out", required=True, help="prefix; writes <out>.json and <out>.drawn_rows.csv")
    a = ap.parse_args(argv)
    for ext in (".json", ".drawn_rows.csv"):
        if os.path.exists(a.out + ext):
            raise SystemExit(f"{a.out + ext} exists: refusing to overwrite")
    with open(a.overlap_json) as f:
        ov = json.load(f)
    expect = int(ov["tasks"][a.dataset]["overlap"]["of_which_retained"])
    pairs = read_matches(a.matches, a.dataset)
    run_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.runs_root, "*", "meta.json")))
    if len(run_dirs) != a.expect_runs:
        raise SystemExit(f"{len(run_dirs)} run dirs under {a.runs_root}, expected {a.expect_runs}")
    rep, rows = measure(a.index, a.holdout_every, pairs, run_dirs, expect)
    rep.update(dataset=a.dataset, buffer_m=ov["buffer_m"], matches=a.matches, runs_root=a.runs_root)
    print(f"GATE seg-overlap-drawn {'PASS' if rep['pass'] else 'FAIL'}: "
          f"{sum(g['pass'] for g in rep['gates'])}/{len(rep['gates'])} checks", flush=True)  # fmt: skip
    if not rep["pass"]:
        print("GATE FAIL: do NOT quote any count above; diagnose the failed check first", flush=True)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    with open(a.out + ".drawn_rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ROW_HEADER)
        w.writerows(rows)
    print(json.dumps({"matched_rows": rep["matched_rows"], "draws": rep["draws"],
                      "verification_counts": rep["verification_counts"]}, indent=1), flush=True)  # fmt: skip
    for r in rep["runs"]:
        print("RUN", json.dumps(r), flush=True)
    print(f"SEG_OVERLAP_DRAWN_OK -> {a.out}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
