"""Timestep x block sweep on an OFFICIAL-split dataset: select (t, k, C) on val, report
test ONCE, and plot the val grid.

    # per block (one ISAAC array task each; writes results/eval/sweep/<run>.npz)
    python -m eval.sweep block --dataset m_eurosat --k 33 --arm flux-oneshot-ens8 \\
        --features 'models/m_eurosat_sweep_ens8_k33/*/multistep_{split}_feats_oneshot_g1.0.npz' \\
        --expect extraction_mode=ONESHOT ensemble_size=8
    # after every block landed
    python -m eval.sweep select --dataset m_eurosat --arm flux-oneshot-ens8 \\
        --blocks results/eval/sweep/sweep-block_m_eurosat_flux-oneshot-ens8-k*.npz

BLOCK scores every (candidate x C) cell of the official protocol at one block k
(candidates = the 7 single timesteps + the 7-t concat; eval.protocols.official_all_cells).
It stores val accuracies in the clear and the TEST predictions SEALED, so the feature
caches can be deleted afterwards (the ISAAC job does, once this file is verified).

SELECT picks the cell with the highest val metric (top-1; micro-F1 for multi-label tasks)
across all blocks (first max wins, in block / candidate / C order, as run_official does),
then unseals exactly ONE test vector -- the selected cell's -- and writes it in eval.probe's
result format, so eval.compare can pair it with any other official-protocol arm (rule 1:
nothing about test influenced the choice). The plot shows val only; test appears as the
one reported number.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import re
import os
import sys
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # noqa: E402

from eval import fusion as FU  # noqa: E402
from eval import features as F  # noqa: E402
from eval import probe as PR  # noqa: E402
from eval import protocols as P  # noqa: E402
from eval import wb  # noqa: E402

OUT_DIR = "results/eval/sweep"


def _sealed_key(cand: str, C: float) -> str:
    return f"sealed__{cand}__C{C}"


# ------------------------------------------------------------------------------ block
# --head tsweight blocks have no C: (lr, wd, epoch) is selected INSIDE the head on val. They
# record this one-element sentinel grid so every C-keyed structure (val rows, sealed keys,
# load_blocks identity, done-status) keeps working, and verify_block checks it like any grid.
TSWEIGHT_GRID = (0.0,)

# BLAS threads of every single-label lbfgs fit of a block: a protocol CONSTANT, not the job's
# allocation. lbfgs on the wide FLUX designs is only reproducible for a fixed thread count
# (OpenBLAS sums in a thread-dependent order; 1e-3 val moves and flipped test predictions
# between counts, eval/matrix.py blas_threads), and every reported block ran with
# --cpus-per-task=8 = OMP_NUM_THREADS (probe_multiblock.sbatch / probe_cpu.sbatch), so 8
# reproduces them bit for bit whatever the allocation (oversubscription when it is smaller:
# slower, same numbers). Multi-hot labels: each one-vs-rest label fit runs at
# protocols.OVR_FIT_THREADS inside _CountingLR regardless. The count is stamped in every block
# (`compute`, block_compute) and run_select prints a NOTE when a sweep mixes counts (rule 8;
# review 2026-10-16).
SWEEP_FIT_THREADS = 8


def block_compute(head: str, multilabel: bool, sel: dict | None = None) -> dict:
    """What a block's fits ran on, stamped as `compute` (rule 8: the operating point is
    measured, not assumed). lr head: the BLAS threads of its fits (the constants above);
    tsweight head: the torch device and thread count fusion.fit_select ran on."""
    d = {"cpus": len(os.sched_getaffinity(0)), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    if head == "tsweight":
        d.update({"device": sel["device"], "torch_threads": sel["torch_threads"]})
    else:
        d.update(
            {
                "blas_threads": P.OVR_FIT_THREADS if multilabel else SWEEP_FIT_THREADS,
                "ovr_n_jobs": P.ovr_n_jobs() if multilabel else 1,
            }
        )
    return d


def compute_of(d) -> dict:
    """A block's `compute` stamp ({} for a block written before 2026-10-16)."""
    return json.loads(str(d["compute"])) if "compute" in d.files else {}


def head_of(d) -> str:
    """'lr' (logistic regression x C, every block before 2026-10-11) or 'tsweight'."""
    return str(d["head"]) if "head" in d.files else "lr"


def fusion_grid_of(d) -> str | None:
    """The tsweight block's --fusion-grid as recorded in its config (None: lr block)."""
    if head_of(d) != "tsweight":
        return None
    return json.loads(str(d["config"]))["protocol_constants"]["fusion"].get("grid", "satdifuser")


def run_block(args) -> str:
    head = getattr(args, "head", "lr")
    args.fusion_grid = getattr(args, "fusion_grid", "satdifuser")
    if head != "tsweight" and args.fusion_grid != "satdifuser":
        raise SystemExit("--fusion-grid applies to --head tsweight only")
    if head == "tsweight" and args.c_grid:
        raise SystemExit("--head tsweight has no C grid (lr/wd/epoch are selected on val inside the head)")
    # parsed ONCE: the recorded grid is the fitted grid
    c_grid = TSWEIGHT_GRID if head == "tsweight" else PR.parse_c_grid(args.c_grid)
    pins = {**F.flux_pins(args.dataset, F.parse_expect(args.expect)), "k": args.k}
    if args.dataset not in F.OFFICIAL:
        raise SystemExit(f"{args.dataset}: no OFFICIAL entry (count the shipped partition first, rule 16)")
    files = {
        s: sorted(glob.glob(args.features.format(split=s))) for s in F.official_spec(args.dataset)["sizes"]
    }
    # keys of the per-split file map are split names, identical in smoke and real runs
    if not all(files.values()):
        raise SystemExit(f"no feature files for some split: { {s: len(v) for s, v in files.items()} }")
    config = {
        "protocol": "official-sweep",
        "protocol_constants": (
            {"head": "tsweight", "fusion": FU.fusion_constants(args.fusion_grid)}
            if head == "tsweight"
            else {**PR.PROTOCOL_CONSTANTS["official"], "C_grid": c_grid}
        ),
        "k": args.k,
        "pins": pins,
        "features": {s: [wb.file_identity(f) for f in v] for s, v in files.items()},
        "smoke_sizes": args.smoke_sizes,
    }
    sizes = PR.parse_smoke_sizes(args.smoke_sizes)
    exp = "sweep-block" + ("-smoke" if sizes else "")
    _, name = wb.init(exp, args.dataset, f"{args.arm}-k{args.k}", config, job_type="probe")
    print(f"== {name}")

    # Same loader + cross-split checks as eval.probe --protocol official (eps seeds, meta).
    ns = SimpleNamespace(dataset=args.dataset, features=args.features, pins=pins, view="all", sizes=sizes)
    cands, ytr, yva, yte, test_paths, inputs, extra = PR.official_flux(ns)
    for p in inputs:
        wb.use_features(p)
    if head == "tsweight":
        Xtr3, Xva3, Xte3 = extra.pop("X3")
        # test labels are NOT passed: the head predicts test without scoring it (sealed below)
        pred, sel, table = FU.tsweight_predict(
            Xtr3,
            ytr,
            Xva3,
            yva,
            Xte3,
            FU.n_outputs(args.dataset, ytr),
            extra.pop("slots"),
            grid=args.fusion_grid,
        )
        rows = [{"candidate": "tsweight", "C": TSWEIGHT_GRID[0], "val_acc": sel["val_acc"], "conv_warnings": 0,
                 "lr": sel["lr"], "wd": sel["wd"], "epoch": sel["epoch"], "slots": sel["slots"],
                 "slot_weights": sel["slot_weights"]}]  # fmt: skip
        preds = {("tsweight", TSWEIGHT_GRID[0]): pred}
        cands = {"tsweight": None}
        extra["tsweight_table"] = table
    else:
        from threadpoolctl import threadpool_limits

        extra.pop("X3", None), extra.pop("slots", None)
        sel = None
        with threadpool_limits(limits=SWEEP_FIT_THREADS, user_api="blas"):  # see SWEEP_FIT_THREADS
            rows, preds = P.official_all_cells(cands, ytr, yva, yte, c_grid=c_grid)
    for r in rows:
        warn = "" if r["conv_warnings"] == 0 else f"  [NOT CONVERGED x{r['conv_warnings']}]"
        print(f"  k={args.k} val {r['candidate']:>9} C={r['C']:<5} score={r['val_acc']:.4f}{warn}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    # Written under a temporary name and renamed only after verify_block passes: a file under the
    # final name therefore IS a verified block, which done_status relies on. (A ".npz" suffix
    # keeps np.savez from appending one.)
    tmp = os.path.join(args.out_dir, f".{name}.{os.getpid()}.tmp.npz")
    with P.sigterm_as_exit():  # Slurm --time/scancel: run `finally`, never leave sealed vectors
        try:
            np.savez(
                tmp,
                k=np.array(args.k),
                dataset=np.array(args.dataset),
                arm=np.array(args.arm),
                candidates=np.array(list(cands)),
                C_grid=np.array(c_grid),
                val_table=np.array(json.dumps(rows)),
                timesteps=np.array(extra["timesteps"]),
                head=np.array(head),
                **(
                    {"tsweight_table": np.array(json.dumps(extra["tsweight_table"]))}
                    if head == "tsweight"
                    else {}
                ),
                test_paths=np.array(test_paths),
                test_labels=yte,
                pins=np.array(json.dumps(pins)),
                run_name=np.array(name),
                # full config incl. every input file's sha1 and the C grid: lets a re-queued job tell a
                # result of THIS sweep from a stale one with the same filename prefix (done_status)
                config=np.array(json.dumps(config, default=str)),
                compute=np.array(json.dumps(block_compute(head, np.asarray(ytr).ndim == 2, sel))),
                smoke=np.array(bool(sizes)),
                **{_sealed_key(c, Cv): pr for (c, Cv), pr in preds.items()},
            )
            verify_block(tmp, expect_c_grid=c_grid, expect_head=head)
            os.replace(tmp, out)
        finally:  # any failure (verify, quota, I/O, SIGTERM): never leave sealed vectors in a hidden temp file
            if os.path.exists(tmp):
                os.remove(tmp)
    print(f"BLOCK OK: {len(rows)} cells (val in clear, test sealed) -> {out}")
    if wandb.run is not None:
        wb.log_table("val_grid", [{"k": args.k, **r} for r in rows])
        wb.log_results(out)
        wandb.finish()
    return out


def _real_pattern(pattern: str) -> str:
    """A glob pattern with its fixed (wildcard-free) directory prefix resolved through symlinks
    (models/ is a symlink to scratch), so it can be fnmatch'ed against _abs()'d paths."""
    cut = min([i for i, ch in enumerate(pattern) if ch in "*?["] or [len(pattern)])
    head = os.path.dirname(pattern[:cut])
    base = os.path.realpath(
        head if os.path.isabs(head) else os.path.join(wb.REPO_ROOT, head) if head else wb.REPO_ROOT
    )
    return os.path.join(base, pattern[len(head) :].lstrip(os.sep))


def _abs(path: str) -> str:
    """A recorded input path (wb.rel: repo-relative when under the repo) as a real absolute path."""
    return os.path.realpath(path if os.path.isabs(path) else os.path.join(wb.REPO_ROOT, path))


def done_status(
    result: str,
    features: str,
    dataset: str,
    c_grid: tuple,
    cache_dir: str,
    head: str = "lr",
    fusion_grid: str = "satdifuser",
) -> tuple[int, str]:
    """Is an existing block result DONE for the caches under cache_dir? One decision, from the
    input paths + sha1s the result records (eval.sweep done-status; both probe scripts branch on
    it). A result file exists only after verify_block passed (run_block renames atomically).
      0 done   -- every recorded input lies under cache_dir, and for each split the files there
                  NOW are exactly the recorded ones with the recorded sha1s, or there are none
                  (deleted after verification / purged: the result is valid for what it recorded)
      3 stale  -- the C grid differs, or a split now holds different files or changed content
                  (e.g. re-extracted into a new run dir after a config-hash change)
      4 legacy, counted done -- no recorded config (pre-2026-10-08) and no _meta.json under
                  cache_dir is newer than the result file (mtime only: a copy without -p looks new)
      5 not this extraction -- recorded inputs outside cache_dir, or a legacy result older than
                  the extraction now under cache_dir
    A check that cannot run raises (exit 1): never a verdict."""
    d = np.load(result, allow_pickle=True)
    if "config" not in d.files:
        r_mtime = os.path.getmtime(result)
        newer = [
            m for m in glob.glob(os.path.join(cache_dir, "*", "*_meta.json")) if os.path.getmtime(m) > r_mtime
        ]
        if newer:
            return 5, f"legacy result is older than extraction(s) under {cache_dir}: {newer[:2]}"
        return 4, (
            "no recorded config (pre-2026-10-08), no newer extraction by mtime -- unverifiable, "
            "counted done (a result copied without -p would also pass this)"
        )
    if head_of(d) != head:
        return 5, f"block head {head_of(d)!r} != requested {head!r}"
    if head == "tsweight":
        c_grid = TSWEIGHT_GRID
        if fusion_grid_of(d) != fusion_grid:
            return 3, f"fusion grid {fusion_grid_of(d)!r} != requested {fusion_grid!r}"
    recorded = json.loads(str(d["config"])).get("features", {})
    root = os.path.realpath(cache_dir)
    splits = list(F.official_spec(dataset)["sizes"])
    for s in splits:  # provenance first: a result of ANOTHER sweep is "not this extraction" (5)
        rec = recorded.get(s)
        if not rec:
            raise SystemExit(f"{result}: config records no input files for split {s} -- cannot check")
        pat = _real_pattern(features.format(split=s))
        for p in (_abs(x["path"]) for x in rec):
            if os.path.commonpath([p, root]) != root:
                return 5, f"recorded {s} input {p} is not under {cache_dir}"
            if not fnmatch.fnmatch(p, pat):  # e.g. another BLOCKS list's cache in the same dir
                return 5, f"recorded {s} input {p} does not match the requested features {features}"
    got_grid = tuple(float(c) for c in d["C_grid"])
    if got_grid != tuple(c_grid):
        return 3, f"C grid {got_grid} != requested {tuple(c_grid)}"
    for s in splits:
        rec = recorded[s]
        rec_paths = sorted(_abs(x["path"]) for x in rec)
        now = sorted(os.path.realpath(f) for f in glob.glob(features.format(split=s)))
        if not now:
            continue  # deleted after verification (or purged): valid for what it recorded
        if now != rec_paths:
            return 3, f"{s}: the files under {cache_dir} now are not the ones this result was computed on"
        if sorted(wb.file_identity(f)["sha1"] for f in now) != sorted(x["sha1"] for x in rec):
            return 3, f"{s}: input content changed since this result was computed"
    return 0, "verified result for these caches (inputs unchanged, or deleted after verification)"


# Set 2026-10-11 AFTER the six datasets' absolute margins were seen (lowest skill: m-bigearthnet
# 0.379 at 5/14 blocks, m-forestnet 0.453); a quarter of the headroom is a floor for a working
# instrument, not a tuned cut. Do not move it to pass a dataset.
CONTROL_MIN_SKILL = 0.25


def control_skill(best: float, chance: float) -> float:
    """Share of the above-chance headroom the best val cell reaches: (best - chance) / (1 - chance).
    An absolute margin is ill-posed under class imbalance: m-pv4ger's majority rate is 0.847, so
    a perfect probe could only beat it by 0.153 (the old ">= 0.10" rule failed its 0.940 val)."""
    return (best - chance) / (1.0 - chance) if chance < 1.0 else 0.0


def _val_labels_sorted(path: str) -> np.ndarray:
    raw = np.load(path)  # labels only: no feature read needed for the control
    return raw["labels"][np.argsort(raw["subset_indices"])]


def _train_rows_match(features: str, j: int):
    import zipfile

    hits = sorted(glob.glob(features.format(split="train")))
    if not hits:
        raise SystemExit(f"gate: no train cache for {features}")
    # Every train cache the glob matches: one file, or the shards load_flux_split merges (each
    # shard is its own ZIP, so each gets the memmap-vs-zipfile check). Labels carry the file when
    # there is more than one.
    for hit in hits:
        tag = f" ({os.path.basename(os.path.dirname(hit))})" if len(hits) > 1 else ""
        mm = F.npz_member_memmap(hit, "feats")
        with zipfile.ZipFile(hit) as zf, zf.open("feats.npy") as ef:
            if zf.getinfo("feats.npy").compress_type != zipfile.ZIP_STORED:
                raise SystemExit(f"gate: {hit} feats is compressed")
            version = np.lib.format.read_magic(ef)
            shape, fortran, dtype = np.lib.format._read_array_header(ef, version)
            start = ef._orig_compress_start + ef.tell()  # stored member: stream offset == file offset
        if tuple(shape) != mm.shape or fortran or dtype != mm.dtype:
            raise SystemExit(
                f"gate: train header {shape} {dtype} fortran={fortran} != memmap {mm.shape} {mm.dtype}"
            )
        rowbytes = int(np.prod(shape[1:])) * dtype.itemsize
        with open(hit, "rb") as f:
            for r in sorted({0, shape[0] // 2, shape[0] - 1}):
                f.seek(start + r * rowbytes)
                ref = np.frombuffer(f.read(rowbytes), dtype=dtype).reshape(shape[1:])
                yield f"{r}{tag}", bool(np.array_equal(np.asarray(mm[r][:, j, :]), ref[:, j, :]))


def gate(
    phase: str, dataset: str, k: int, features: str, out_dir: str, arm: str, expect: dict | None = None
) -> bool:
    """probe_multiblock's GATE (rule 13), executed PASS/FAIL. phase "slice" (before probing, in
    seconds): the loader's block-k slice of the val cache equals the raw cache's slice, so a wrong
    block index or memmap offset fails before hours of probing. phase "control" (after): the best
    val metric reaches CONTROL_MIN_SKILL of the headroom above the best CONSTANT predictor
    (protocols.trivial_baseline; control_skill). run_select applies the same control to every
    dataset before it opens the test vector."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")

    n_val = F.official_spec(dataset)["sizes"]["val"]
    # pins come from the caller (--expect, as for `block`), never a second hand-copied source
    pins = {**F.flux_pins(dataset, expect or {}), "k": k}
    if phase == "slice":
        X, _, *_, hits = F.load_flux_split(features, pins, "val", n_val)
        raw = np.load(hits[0])
        j = [int(b) for b in raw["block_idx"]].index(k)
        order = np.argsort(raw["subset_indices"])
        # reference via plain np.load -- INDEPENDENT of npz_member_memmap, so a header/offset bug in
        # the memmap path cannot shift both sides identically (rule 6); val is small enough
        ref = raw["feats"][:, :, j, :][order].astype(np.float64)
        check(f"k={k} loader slice == raw cache slice (val)", np.array_equal(X, ref))
        # train: the only ZIP64 (> 4 GB) member and what every fit reads. Rows 0 / mid / last of
        # the memmap vs the same bytes located by zipfile's OWN local-header parse, so a ZIP64
        # header/offset bug in npz_member_memmap cannot shift both sides alike (review 2026-10-11)
        for r, same in _train_rows_match(features, j):
            check(f"k={k} train row {r}: memmap == zipfile-located bytes", same)
    elif phase == "control":
        hits = sorted(glob.glob(features.format(split="val")))
        if len(hits) != 1:
            raise SystemExit(f"gate: expected one val cache for {features}, got {hits}")
        y = _val_labels_sorted(hits[0])
        res = glob.glob(f"{out_dir}/sweep-block_{dataset}_{arm}-k{k}_*.npz")
        check("one block result", len(res) == 1, res)
        if len(res) == 1:
            best = max(r["val_acc"] for r in json.loads(str(np.load(res[0], allow_pickle=True)["val_table"])))
            chance = P.trivial_baseline(y)
            skill = control_skill(best, chance)
            check(f"positive control: skill over best constant predictor >= {CONTROL_MIN_SKILL}",
                  skill >= CONTROL_MIN_SKILL, f"({best:.4f} vs {chance:.4f}: skill {skill:.3f})")  # fmt: skip
    else:
        raise SystemExit(f"unknown gate phase {phase!r}")
    print(f"MULTIBLOCK GATE ({phase}):", "PASS" if ok else "FAIL")
    return ok


def verify_block(path: str, expect_c_grid: tuple | None = None, expect_head: str = "lr") -> None:
    """The ISAAC job deletes the caches only after this passes. The expected cells come
    from the cache's own t-grid (each t + the all-t concat, x C_GRID), not from the stored
    candidate list, so a truncated file cannot validate itself. --head tsweight blocks hold
    exactly one cell (tsweight x TSWEIGHT_GRID); the head is checked against the CALLER's
    expectation, so a block of the other head cannot pass (rule 6)."""
    d = np.load(path, allow_pickle=True)
    if head_of(d) != expect_head:
        raise SystemExit(f"{path}: block head {head_of(d)!r} != expected {expect_head!r}")
    ts = [int(t) for t in d["timesteps"]]
    if expect_head == "tsweight":
        want_cands = ["tsweight"]
        expect_c_grid = TSWEIGHT_GRID
    else:
        want_cands = [f"t{t}" for t in ts] + [f"concat{len(ts)}t"]
    # The C grid is checked against an INDEPENDENT expectation (the caller's requested grid, else
    # the default official grid), not just the file's own record: a block run on the wrong grid
    # is internally consistent and would otherwise pass -- and sweep_block.sh deletes the caches
    # on a pass (rule 6).
    grid = tuple(float(c) for c in d["C_grid"])
    want_grid = tuple(expect_c_grid) if expect_c_grid is not None else tuple(P.C_GRID)
    if grid != want_grid:
        raise SystemExit(f"{path}: block C grid {grid} != expected {want_grid}")
    want_keys = {_sealed_key(c, Cv) for c in want_cands for Cv in grid}
    rows = json.loads(str(d["val_table"]))
    got_rows = {(r["candidate"], r["C"]) for r in rows}
    sealed = {f for f in d.files if f.startswith("sealed__")}
    if sealed != want_keys or got_rows != {(c, Cv) for c in want_cands for Cv in grid}:
        raise SystemExit(
            f"{path}: cells incomplete -- {len(rows)} val rows / {len(sealed)} sealed vectors, "
            f"expected {len(want_keys)} ({len(want_cands)} candidates x {len(grid)} C)"
        )
    if any(d[s].shape != d["test_labels"].shape for s in sealed):
        raise SystemExit(f"{path}: a sealed test vector has the wrong length")


# ----------------------------------------------------------------------------- select
class SealedTests:
    """Test predictions of every sweep cell; exactly ONE may be opened (rule 1)."""

    def __init__(self, blocks: dict):
        self._blocks = blocks
        self.opened: tuple | None = None

    def open(self, k: int, cand: str, C: float) -> np.ndarray:
        if self.opened is not None:
            raise RuntimeError(
                f"seal violation: test vector {self.opened} already opened; refusing {(k, cand, C)}"
            )
        self.opened = (k, cand, C)
        return self._blocks[k][_sealed_key(cand, C)]


def _input_records(d) -> dict | None:
    """{split: {cache path: sha1}} of the caches a block recorded (run_block's config.features),
    or None for a legacy block that recorded none (review 2026-10-11)."""
    if "config" not in d.files:
        return None
    feats = json.loads(str(d["config"])).get("features") or {}
    return {s: {x["path"]: x["sha1"] for x in v} for s, v in sorted(feats.items())}


def select_control(blocks: dict, best: dict) -> dict:
    """Positive control (rule 2/13) before the test vector is opened: the selected val cell must
    reach CONTROL_MIN_SKILL of the headroom above the best constant predictor on the val labels.
    The val labels come from the val cache(s) the blocks recorded, each checked against its sha1:
    one cache shared by every block (multi-block layout) or one per block (per-block layout,
    sweep_block.sh), all of which must hold the same labels."""
    if any("config" not in d.files for d in blocks.values()):
        raise SystemExit("select control: blocks do not all record one val cache -- cannot run the "
                         "positive control (pass --skip-control to override, and say so)")  # fmt: skip
    caches = {}
    for k, d in blocks.items():
        rec = json.loads(str(d["config"]))["features"]["val"]
        if len(rec) != 1:
            raise SystemExit(f"select control: block k={k} records {len(rec)} val caches (sharded val split); "
                             "the control reads one val cache per block -- merge the shards, or pass "
                             "--skip-control and say so")  # fmt: skip
        caches[_abs(rec[0]["path"])] = rec[0]["sha1"]
    labels = None
    for path, sha1 in sorted(caches.items()):
        if not os.path.exists(path) or wb.file_identity(path)["sha1"] != sha1:
            raise SystemExit(
                f"select control: val cache {path} missing or changed since the blocks were probed"
            )
        y = _val_labels_sorted(path)
        if labels is not None and not np.array_equal(labels, y):
            raise SystemExit(f"select control: val cache {path} holds different labels than the other "
                             "blocks' -- not one sweep")  # fmt: skip
        labels = y
    chance = P.trivial_baseline(labels)
    skill = control_skill(best["val_acc"], chance)
    ok = skill >= CONTROL_MIN_SKILL
    print(f"  {'PASS' if ok else 'FAIL'}  positive control: selected val {best['val_acc']:.4f} vs best constant "
          f"predictor {chance:.4f}: skill {skill:.3f} (>= {CONTROL_MIN_SKILL})")  # fmt: skip
    print("SELECT GATE (control):", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit("select control FAILED: diagnose the instrument; the test vector stays sealed")
    return {"chance": chance, "skill": skill, "min_skill": CONTROL_MIN_SKILL, "pass": True}


def load_blocks(paths: list[str], dataset: str):
    blocks, block_paths, ref = {}, {}, None
    inputs: dict[tuple[str, str], tuple[str, str]] = {}  # (split, cache path) -> (sha1, block file)
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        k = int(d["k"])
        if str(d["dataset"]) != dataset:
            raise SystemExit(f"{p}: dataset {d['dataset']} != {dataset}")
        if k in blocks:
            raise SystemExit(f"two result files for block k={k} -- pass exactly one per block")
        ident = (
            [F.path_key(str(x)) for x in d["test_paths"]],  # banked caches used an absolute path
            d["test_labels"].tolist(),
            list(d["candidates"]),
            json.loads(str(d["pins"])) | {"k": None},
            [float(c) for c in d["C_grid"]],  # blocks run with different C grids are not one sweep
            head_of(d),  # an lr block and a tsweight block are different protocols
            fusion_grid_of(d),  # tsweight blocks on different hyperparameter grids are not one sweep
        )
        if ref is None:
            ref = ident
        elif ident != ref:
            raise SystemExit(
                f"{p}: test set / candidates / pins / C grid differ from the other blocks -- not one sweep"
            )
        # Blocks probed on different extractions are not one sweep: a cache path that two blocks
        # both record must carry one sha1. The multi-block layout shares every cache, so this is
        # the whole input identity there; the per-block layout (sweep_block.sh: block k reads its
        # own k<K>/ caches) shares none and is held together by the pins / test set / labels
        # above and by the control's val-label check (review 2026-10-16: the sha1 LIST as
        # identity refused every per-block sweep).
        for split, recs in (_input_records(d) or {}).items():
            for path, sha1 in recs.items():
                prev = inputs.setdefault((split, path), (sha1, p))
                if prev[0] != sha1:
                    raise SystemExit(
                        f"{p}: {split} cache {path} has sha1 {sha1[:12]} but {prev[1]} recorded "
                        f"{prev[0][:12]} -- blocks probed on different extractions are not one sweep"
                    )
        blocks[k] = d
        block_paths[k] = p
    return dict(sorted(blocks.items())), dict(sorted(block_paths.items())), ref


def select(blocks: dict) -> tuple[dict, list[dict]]:
    grid, best = [], None
    for k, d in blocks.items():
        for r in json.loads(str(d["val_table"])):  # candidate order, then C order
            grid.append({"k": k, **r})
            if best is None or r["val_acc"] > best["val_acc"]:
                best = {"k": k, **r}
    return best, grid


def grid_C(grid: list[dict]) -> tuple[float, dict]:
    """ONE C for the whole display grid, chosen on val: the C with the best MEAN val metric
    over every (k, candidate) cell (first max wins, in C_GRID order). Comparing cells at each
    cell's own best C compares them at different regularization (rule 8) and gives every cell
    a best-of-3 bonus; a grid-wide C reads the block/timestep shape at matched C. Display only:
    the selection protocol (joint argmax over k x candidate x C) is unchanged."""
    means = {}
    for Cv in sorted({g["C"] for g in grid}):  # ascending, as every C grid is ordered
        vals = [g["val_acc"] for g in grid if g["C"] == Cv]
        means[Cv] = float(np.mean(vals))
    best_C = max(means, key=lambda c: means[c])
    return best_C, means


def _fmt_C(C: float) -> str:
    return f"{C:g}".replace("e-0", "e-")


def best_over_C(grid: list[dict], ks: list[int], cands: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per (k, candidate) cell: the BEST val metric over C and the C achieving it (first max in
    stored C order, the same tie rule as select). Val-only."""
    M = np.full((len(ks), len(cands)), np.nan)
    Cb = np.full((len(ks), len(cands)), np.nan)
    for g in grid:
        i, j = ks.index(g["k"]), cands.index(g["candidate"])
        if np.isnan(M[i, j]) or g["val_acc"] > M[i, j]:
            M[i, j], Cb[i, j] = g["val_acc"], g["C"]
    return M, Cb


def plot(
    grid: list[dict],
    best: dict,
    cands: list[str],
    title: str,
    out_png: str,
    metric: str = "top-1",
    test_score: float | None = None,
) -> None:
    """Val grid with every cell at ITS OWN best C (annotated), so the outlined selected cell is
    the grid maximum and shows exactly the selected val score; the one reported test number sits
    in the figure next to it. (Until 2026-10-11 cells showed one grid-wide C, so the selected
    cell could display a lower value than it was selected at.)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    ink, muted, surface = "#1f1e1c", "#6b6a66", "#fcfcfb"
    blue_ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", blue_ramp)
    ks = sorted({g["k"] for g in grid})
    M, Cb = best_over_C(grid, ks, cands)
    bi, bj = ks.index(best["k"]), cands.index(best["candidate"])
    # Executed consistency check: the outlined cell must show the selected val score, and no
    # cell may beat it (select is the joint argmax over k x candidate x C).
    if not (np.isclose(M[bi, bj], best["val_acc"]) and Cb[bi, bj] == best["C"]):
        raise RuntimeError(
            f"plot/select mismatch: cell ({best['k']}, {best['candidate']}) best-over-C "
            f"{M[bi, bj]} at C={Cb[bi, bj]} vs selected {best['val_acc']} at C={best['C']}"
        )
    if np.nanmax(M) > best["val_acc"] + 1e-12:
        raise RuntimeError(f"a cell's val ({np.nanmax(M)}) exceeds the selected val ({best['val_acc']})")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 0.62 * len(ks) + 2.6), facecolor=surface, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
    P100 = M * 100  # percent everywhere: cells, colorbar, curves
    im = ax1.imshow(P100, cmap=cmap, aspect="auto")
    lo, hi = np.nanmin(P100), np.nanmax(P100)
    for i in range(len(ks)):
        for j in range(len(cands)):
            v = P100[i, j]
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
            col = "white" if dark else ink
            ax1.text(j, i - 0.13, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color=col)
            sub = f"C={_fmt_C(Cb[i, j])}"
            if cands[j] == "tsweight":  # no C: the head's own val-selected lr / wd
                r = next(g for g in grid if g["k"] == ks[i] and g["candidate"] == "tsweight")
                sub = f"lr={_fmt_C(r['lr'])} wd={_fmt_C(r['wd'])}"
            ax1.text(j, i + 0.24, sub, ha="center", va="center", fontsize=5.2, color=col)
    # Two-tone ring: visible on the darkest (typically selected) cell and on light ones.
    ax1.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=ink, linewidth=4.5))
    ax1.add_patch(Rectangle((bj - 0.44, bi - 0.44), 0.88, 0.88, fill=False, edgecolor="white", linewidth=2.0))
    ax1.set_xticks(
        range(len(cands)), [c.replace("concat", "concat ") for c in cands], fontsize=8, color=muted
    )
    ax1.set_yticks(range(len(ks)), [f"k={k}" for k in ks], fontsize=8, color=muted)
    ax1.set_title(
        f"Val {metric} (%) per cell at its best C (shown); outlined = selected on val",
        fontsize=10,
        color=ink,
        loc="left",
    )
    for sp in ax1.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02, label=f"val {metric} (%)")
    cb.ax.yaxis.label.set_color(muted)
    cb.ax.yaxis.label.set_fontsize(8)
    cb.ax.tick_params(labelsize=7, colors=muted)
    cb.outline.set_visible(False)

    # Headline INSIDE the figure: the selected cell and its one test number.
    op = (f"lr={_fmt_C(best['lr'])}, wd={_fmt_C(best['wd'])}, epoch {best['epoch']}" if best["candidate"] == "tsweight"
          else f"C={_fmt_C(best['C'])}")  # fmt: skip
    head = f"Selected: k={best['k']}, {best['candidate']}, {op}  |  val {100 * best['val_acc']:.1f}"
    if test_score is not None:
        head += f"  |  TEST {metric} {100 * test_score:.1f}% (reported once)"
    ax2.text(
        0.0, 1.13, head, transform=ax2.transAxes, fontsize=10, color=ink, fontweight="bold", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaf2fd", edgecolor="#256abf", linewidth=1.2),
    )  # fmt: skip

    # Focus pattern: every block as thin gray context; the selected block and the banked
    # default k=28 highlighted and direct-labelled (8 ordinal hues fail the ΔL check).
    tcands = [c for c in cands if re.fullmatch(r"t\d+", c)]
    if not tcands:  # tsweight-only sweep: show the selected block's learned slot weights instead
        w, names = best.get("slot_weights") or [], best.get("slots") or []
        ax2.bar(range(len(w)), w, color="#256abf")
        ax2.axhline(1 / max(len(w), 1), color=muted, linestyle="--", linewidth=1)
        ax2.set_xticks(range(len(w)), names, fontsize=8, color=muted)
        ax2.set_ylabel("learned weight (softmax)", fontsize=9, color=muted)
        ax2.set_title(f"Learned timestep weights, selected block k={best['k']} (dashed: uniform)",
                      fontsize=10, color=ink, loc="left")  # fmt: skip
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, facecolor=surface)
        plt.close(fig)
        return
    ts = [int(c[1:]) for c in tcands]
    for i, k in enumerate(ks):
        y = [P100[i, cands.index(c)] for c in tcands]
        if k == best["k"]:
            ax2.plot(ts, y, color="#256abf", linewidth=2.2, marker="o", markersize=5, zorder=3)
            ax2.annotate(f"k={k} (selected)", (ts[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                         fontsize=8, color=ink, va="center")  # fmt: skip
        elif k == 28:
            ax2.plot(ts, y, color=ink, linewidth=1.6, linestyle="--", zorder=2)
            ax2.annotate("k=28 (banked)", (ts[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                         fontsize=8, color=muted, va="center")  # fmt: skip
        else:
            ax2.plot(ts, y, color="#c9c7c1", linewidth=1.0, zorder=1)
    ax2.set_xticks(ts, [str(t) for t in ts], fontsize=8, color=muted)
    ax2.tick_params(axis="y", labelsize=8, colors=muted)
    ax2.set_xlabel("timestep t", fontsize=9, color=muted)
    ax2.set_ylabel(f"val {metric} (%)", fontsize=9, color=muted)
    ax2.grid(axis="y", color="#e6e4df", linewidth=0.8)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color("#d6d4ce")
    ax2.set_title(
        f"Val {metric} vs t, each point at its best C (gray: other blocks)",
        fontsize=10,
        color=ink,
        loc="left",
    )
    ax2.set_xlim(ts[0] - 20, ts[-1] + 110)
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=surface)
    plt.close(fig)


def _title(dataset: str, arm: str, n_cells: int) -> str:
    """Setup only; the selected cell and the test number are drawn inside the figure."""
    return f"{dataset}: t x k probe sweep ({arm}), {n_cells} val cells, selection on val, one test evaluation"


def run_replot(args) -> str:
    """Redraw a select result's plot from what it already stores (val grid + the ONE test
    vector it opened). Opens nothing new, so it costs no test use. Overwrites <name>.png
    unless --out is given."""
    d = np.load(args.result, allow_pickle=True)
    grid = json.loads(str(d["val_grid"]))
    best = json.loads(str(d["info"]))["selected"]
    correct = d["correct__test"]
    metric = "micro-F1" if correct.ndim == 2 else "top-1"
    cands = list(dict.fromkeys(g["candidate"] for g in grid))
    png = getattr(args, "out", None) or args.result.replace(".npz", ".png")
    plot(
        grid, best, cands, _title(str(d["dataset"]), str(d["arm"]), len(grid)), png, metric, P.metric(correct)
    )
    print(f"replotted (no test use) -> {png}")
    return png


def run_select(args) -> str:
    paths = sorted({p for pat in args.blocks for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no block results match {args.blocks}")
    blocks, block_paths, (test_paths, test_labels, cands, _, c_grid, _head, _fgrid) = load_blocks(
        paths, args.dataset
    )
    smoke = any(bool(d["smoke"]) for d in blocks.values())
    # Rule 8: the fits' operating point (BLAS threads / torch device) per block, recorded in the
    # selection and flagged when the sweep mixes them (blocks before 2026-10-16 carry no stamp).
    compute = {k: compute_of(d) for k, d in blocks.items()}
    shares = sorted(
        {
            str(c.get("blas_threads", c.get("torch_threads", "unrecorded")))
            + (f"@{c['device']}" if "device" in c else "")
            for c in compute.values()
        }
    )
    if len(shares) > 1:
        print(
            f"NOTE: the blocks of this sweep ran their fits at different thread counts / devices "
            f"({', '.join(shares)}; `compute` in each block): wide-design fits move at the 1e-3 level "
            "with it (sweep.SWEEP_FIT_THREADS is the protocol constant since 2026-10-16)"
        )
    config = {
        "protocol": "official-sweep-select",
        "blocks": {k: wb.file_identity(p) for k, p in block_paths.items()},
        "C_grid": c_grid,
    }
    exp = "sweep-select" + ("-smoke" if smoke else "")
    _, name = wb.init(exp, args.dataset, args.arm, config, job_type="probe")
    print(f"== {name}  blocks {list(blocks)}")
    for p in paths:
        wb.use_results(p)

    best, grid = select(blocks)
    print(f"SELECTED on val: k={best['k']} {best['candidate']} C={best['C']} (val {best['val_acc']:.4f})")
    if args.skip_control:
        print("WARNING: --skip-control: positive control NOT run before opening the test vector")
        control = {"skipped": True}
    else:
        control = select_control(blocks, best)
    sealed = SealedTests(blocks)
    pred = sealed.open(best["k"], best["candidate"], best["C"])
    yte = np.asarray(test_labels)
    correct = P.per_image(yte, pred)
    score, line, extra_metrics = PR.report_test(args.dataset, yte, pred, correct)
    metric = "micro-F1" if yte.ndim == 2 else "top-1"
    print(f"TEST (reported once): {line}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    C_disp, C_means = grid_C(grid)
    info = {
        "selected": best,
        **extra_metrics,
        "n_cells": len(grid),
        "blocks": list(blocks),
        "display_C": C_disp,  # val-chosen grid-wide C (recorded; plots now show each cell at its best C)
        "display_C_mean_val": C_means,
        "control": control,
        "compute": compute,
    }
    np.savez(
        out,
        cells=np.array(["test"]),
        correct__test=correct,
        ev__test=np.arange(len(yte)),
        paths=np.array(test_paths),
        labels=yte,
        protocol=np.array("official"),
        arm=np.array(args.arm),
        dataset=np.array(args.dataset),
        run_name=np.array(name),
        config=np.array(json.dumps(config, default=str)),
        info=np.array(json.dumps(info, default=str)),
        val_grid=np.array(json.dumps(grid)),
        smoke=np.array(smoke),
    )
    png = out.replace(".npz", ".png")
    plot(grid, best, cands, _title(args.dataset, args.arm, len(grid)), png, metric, score)
    print(f"result -> {out}\nplot   -> {png}")
    if wandb.run is not None:
        wandb.run.summary.update({"acc/test": score, **extra_metrics, "selected": json.dumps(best)})
        wb.log_table("val_grid", grid)
        wandb.log({"sweep_plot": wandb.Image(png)})
        wb.log_results(out)
        wandb.finish()
    return out


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("block", help="score every (candidate x C) cell at one block k")
    b.add_argument("--dataset", required=True)
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--arm", required=True)
    b.add_argument("--features", required=True, help="cache pattern with {split}")
    b.add_argument("--expect", nargs="*", default=None, metavar="KEY=VALUE")
    b.add_argument(
        "--smoke-sizes", default=None, metavar="TRAIN,VAL,TEST", help="SMOKE ONLY: subset split sizes"
    )
    b.add_argument(
        "--c-grid", default=None, metavar="C1,C2,...", help="LR C values (default protocols.C_GRID)"
    )
    b.add_argument("--head", choices=["lr", "tsweight"], default="lr",
                   help="lr: each t + concat x C (default). tsweight: ONE learned timestep-weighting head "
                        "(eval/fusion.py), lr/wd/epoch selected on val -- use a separate --arm")  # fmt: skip
    b.add_argument("--fusion-grid", choices=["satdifuser", "reg"], default="satdifuser",
                   help="--head tsweight only: eval/fusion.py FUSION_GRIDS (recorded; own --arm per grid)")  # fmt: skip
    b.add_argument("--out-dir", default=OUT_DIR)
    s = sub.add_parser("select", help="select on val across blocks, report test once, plot")
    s.add_argument("--dataset", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--blocks", nargs="+", required=True, help="block result files or globs")
    s.add_argument("--out-dir", default=OUT_DIR)
    s.add_argument("--skip-control", action="store_true",
                   help="do NOT run the positive control before opening test (legacy blocks without a val cache)")  # fmt: skip
    ds = sub.add_parser("done-status", help="0 done / 3 stale / 4 legacy-done / 5 not this extraction")
    ds.add_argument("result")
    ds.add_argument("--dataset", required=True)
    ds.add_argument("--features", required=True, help="cache pattern with {split}")
    ds.add_argument("--cache-dir", required=True, help="the dir this sweep's caches live under")
    ds.add_argument("--c-grid", default=None, metavar="C1,C2,...")
    ds.add_argument("--head", choices=["lr", "tsweight"], default="lr")
    ds.add_argument("--fusion-grid", choices=["satdifuser", "reg"], default="satdifuser")
    g = sub.add_parser("gate", help="multi-block GATE: --phase slice (before probing) | control (after)")
    g.add_argument("--phase", required=True, choices=["slice", "control"])
    g.add_argument("--dataset", required=True)
    g.add_argument("--k", type=int, required=True)
    g.add_argument("--features", required=True, help="cache pattern with {split}")
    g.add_argument("--out-dir", default=OUT_DIR)
    g.add_argument("--arm", required=True)
    g.add_argument("--expect", nargs="*", default=None, metavar="KEY=VALUE")
    r = sub.add_parser("replot", help="redraw a select result's plot (each cell at its best C; no test use)")
    r.add_argument("result", help="a sweep-select_*.npz")
    r.add_argument("--out", default=None, help="output png (default: overwrite <result>.png)")
    args = p.parse_args(argv)
    if args.cmd == "done-status":
        grid = TSWEIGHT_GRID if args.head == "tsweight" else PR.parse_c_grid(args.c_grid)
        code, why = done_status(
            args.result, args.features, args.dataset, grid, args.cache_dir, args.head, args.fusion_grid
        )
        print(f"{args.result}: {why}")
        raise SystemExit(code)
    if args.cmd == "gate":
        ok = gate(
            args.phase,
            args.dataset,
            args.k,
            args.features,
            args.out_dir,
            args.arm,
            F.parse_expect(args.expect),
        )
        raise SystemExit(0 if ok else 1)
    return {"block": run_block, "select": run_select, "replot": run_replot}[args.cmd](args)


if __name__ == "__main__":
    main()
