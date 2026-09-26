"""W&B conventions for the evaluation pipeline (extract -> probe -> compare).

Every pipeline run lands in ONE project, named `{exp}_{dataset}_{arm}_{hash}` and grouped
by `exp`. The hash covers the run's protocol constants + input identities, so an unchanged
rerun reuses the name and any protocol change gets a new one (the probe-side analogue of
RunConfig.config_hash()).

Lineage is carried by artifacts, which works in offline mode (ISAAC compute nodes):
  - feature caches are REFERENCE artifacts (path + checksum, nothing uploaded). The
    extractor logs one; the probe constructs the identical reference and uses it, and
    identical digests resolve to the same artifact version -> extract -> probe edge.
  - per-image result vectors (small .npz) are uploaded; compare uses them the same way
    -> probe -> compare edge.
WANDB_MODE=disabled turns all of it into no-ops (the reproduction gates run that way).
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import wandb

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "sparse_representation_learning")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "DiT_remote_sensing")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(path: str) -> str:
    """Repo-relative path, so the same cache hashes identically on workstation and ISAAC."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, REPO_ROOT) if ap.startswith(REPO_ROOT + os.sep) else ap


_SHA_CACHE: dict = {}


def file_identity(path: str) -> dict:
    """Content identity: a re-extraction changes it, a copy (scp, cp without -p) does not."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return {"path": rel(path), "bytes": st.st_size, "sha1": _SHA_CACHE[key]}


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_name(exp: str, dataset: str, arm: str, config: dict) -> tuple[str, str]:
    h = config_hash({"exp": exp, "dataset": dataset, "arm": arm, **config})
    return f"{exp}_{dataset}_{arm}_{h}", h


def init(exp: str, dataset: str, arm: str, config: dict, job_type: str):
    name, h = run_name(exp, dataset, arm, config)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=name,
        group=exp,
        job_type=job_type,
        config={**config, "exp": exp, "dataset": dataset, "arm": arm, "hash": h},
    )
    return run, name


def _artifact_name(prefix: str, path: str) -> str:
    # wandb artifact names: alphanumerics, '-', '_', '.' only (max 128).
    parts = os.path.normpath(path).split(os.sep)[-2:]
    stem = ".".join(parts).removesuffix(".npz")
    return (prefix + "-" + re.sub(r"[^A-Za-z0-9_.-]", ".", stem))[:128]


def features_artifact(path: str) -> wandb.Artifact:
    art = wandb.Artifact(_artifact_name("feat", path), type="features", metadata=file_identity(path))
    art.add_reference("file://" + os.path.abspath(path), checksum=True)
    meta = path.replace(".npz", "_meta.json")
    if os.path.exists(meta):
        art.add_reference("file://" + os.path.abspath(meta), checksum=True)
    return art


def log_features(path: str) -> None:
    """Producer side (an extractor): record the cache it just wrote."""
    if wandb.run is not None:
        wandb.run.log_artifact(features_artifact(path))


def _use(art: wandb.Artifact) -> None:
    """Declare an input. Offline runs cannot (wandb raises), so they record name+digest in
    the summary and `python -m eval.wb link` attaches the edge after `wandb sync`."""
    if wandb.run is None:
        return
    if wandb.run.offline:
        pending = list(wandb.run.summary.get("pending_inputs", []))
        pending.append({"name": art.name, "type": art.type, "digest": art.manifest.digest()})
        wandb.run.summary["pending_inputs"] = pending
    else:
        wandb.run.use_artifact(art)


def use_features(path: str) -> None:
    """Consumer side (a probe): declare the cache it read."""
    _use(features_artifact(path))


def results_artifact(path: str) -> wandb.Artifact:
    art = wandb.Artifact(_artifact_name("res", path), type="probe-results")
    art.add_file(path)
    return art


def log_results(path: str) -> None:
    if wandb.run is not None:
        wandb.run.log_artifact(results_artifact(path))


def use_results(path: str) -> None:
    _use(results_artifact(path))


def log_table(key: str, rows: list[dict]) -> None:
    if wandb.run is None or not rows:
        return
    cols = list(rows[0])
    wandb.log({key: wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})


def link(project: str = f"{WANDB_ENTITY}/{WANDB_PROJECT}") -> None:
    """Attach the input edges that offline runs recorded (run after `wandb sync`).

    Resolves each pending input to the synced artifact version with the SAME digest --
    never `:latest`, which could be a re-extraction with different contents. Inputs with
    no producer artifact (caches extracted before extraction.py logged them) are skipped
    and recorded in the run's `inputs_unlinked`; they never block the other runs.
    """
    api = wandb.Api()
    for run in api.runs(project, filters={"summary_metrics.pending_inputs": {"$exists": True}}):
        pending = run.summary.get("pending_inputs") or []
        if not pending or run.summary.get("inputs_linked"):
            continue
        unlinked = []
        for inp in pending:
            try:
                coll = api.artifact_collection(inp["type"], f"{project}/{inp['name']}")
                match = [v for v in coll.artifacts() if v.digest == inp["digest"]]
                why = "no version with this digest"
            except Exception as e:  # collection absent: the producer never logged it
                match, why = [], type(e).__name__
            if match:
                run.use_artifact(match[0])
            else:
                unlinked.append({**inp, "reason": why})
        run.summary["inputs_linked"] = True
        run.summary["inputs_unlinked"] = unlinked
        run.summary.update()
        note = f" ({len(unlinked)} without a producer artifact)" if unlinked else ""
        print(f"{run.name}: linked {len(pending) - len(unlinked)}/{len(pending)} inputs{note}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] != ["link"]:
        raise SystemExit("usage: python -m eval.wb link [entity/project]")
    link(*sys.argv[2:3])
