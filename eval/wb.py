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

import contextlib
import hashlib
import json
import os
import re
import time

import wandb

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "sparse_representation_learning")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "DiT_remote_sensing")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(path: str) -> str:
    """Repo-relative path, so the same cache hashes identically on workstation and ISAAC."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, REPO_ROOT) if ap.startswith(REPO_ROOT + os.sep) else ap


_SHA_CACHE: dict = {}


def _sidecar(path: str) -> str:
    return path + ".sha1.json"


def file_identity(path: str) -> dict:
    """Content identity: a re-extraction changes it, a copy (scp, cp without -p) does not.
    The SHA1 is also cached in <path>.sha1.json keyed on (size, mtime_ns, ctime_ns): a multi-block cache is
    tens of GB and every block task of a sweep reads it, so hashing it once per FILE instead of
    once per PROCESS saves a full read per task. A rewritten file changes size/mtime and is
    re-hashed; an unwritable directory just skips the sidecar. ctime is in the key because cp -p /
    rsync -a preserve size and mtime but can never preserve ctime (review 2026-10-24): a different
    file copied over this one with its mtime kept is re-hashed instead of trusted."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def unchanged() -> bool:  # the file is still the one `st` describes
        now = os.stat(path)
        return (now.st_size, now.st_mtime_ns, now.st_ctime_ns) == (st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def from_sidecar():
        try:
            with open(_sidecar(path)) as f:
                rec = json.load(f)
            if (
                rec.get("bytes") == st.st_size
                and rec.get("mtime_ns") == st.st_mtime_ns
                and rec.get("ctime_ns") == st.st_ctime_ns
                and rec.get("sha1")
            ):
                _SHA_CACHE[key] = rec["sha1"]
        except (OSError, ValueError):
            pass

    if key not in _SHA_CACHE:
        from_sidecar()
    if key not in _SHA_CACHE:
        # Tasks that start together would each hash the same multi-GB file: serialize on a lock
        # file so the first one hashes and the rest read its sidecar. No flock support (some
        # Lustre mounts) -> hash without the lock; correctness never depends on it.
        lock = None
        retry = False
        try:
            import fcntl

            lock = open(_sidecar(path) + ".lock", "w")
            fcntl.flock(lock, fcntl.LOCK_EX)
            from_sidecar()  # another task may have finished while we waited
        except OSError:
            pass
        try:
            if key not in _SHA_CACHE:
                if not unchanged():  # rewritten while we waited on the lock: retry AFTER release
                    retry = True
            if key not in _SHA_CACHE and not retry:
                h = hashlib.sha1()
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 24), b""):
                        h.update(block)
                if not unchanged():  # rewritten while hashing: the digest matches neither version
                    raise OSError(f"{path} changed while being hashed -- retry once it is stable")
                _SHA_CACHE[key] = h.hexdigest()
                try:  # atomic replace: a reader never sees a half-written sidecar
                    tmp = f"{_sidecar(path)}.{os.getpid()}.tmp"
                    with open(tmp, "w") as f:
                        json.dump(
                            {
                                "bytes": st.st_size,
                                "mtime_ns": st.st_mtime_ns,
                                "ctime_ns": st.st_ctime_ns,
                                "sha1": _SHA_CACHE[key],
                            },
                            f,
                        )
                    os.replace(tmp, _sidecar(path))
                except OSError:
                    pass
        finally:
            if lock is not None:
                lock.close()  # releases the flock
        if retry:  # the lock is released before re-entering, so this cannot self-deadlock
            return file_identity(path)
    if not unchanged():  # rewritten after `st` (incl. while waiting): the sha1 above is for the OLD bytes
        _SHA_CACHE.pop(key, None)
        return file_identity(path)
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


def _md5_b64(path: str) -> str:
    """wandb's own reference digest (md5, base64), cached in <path>.md5.json keyed on
    (size, mtime_ns) like the SHA1 sidecar. wandb re-hashes a checksum=True reference on every
    add_reference, i.e. a full read of a 23-31 GB multi-block cache per probe task (review
    2026-10-11). Same digest as wandb computes, so producer/consumer artifacts still match."""
    from wandb.sdk.lib.hashutil import md5_file_b64

    st = os.stat(path)
    sc = path + ".md5.json"
    try:
        with open(sc) as f:
            rec = json.load(f)
        if rec.get("bytes") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns and rec.get("md5_b64"):
            return rec["md5_b64"]
    except (OSError, ValueError):
        pass
    digest = md5_file_b64(path)
    now = os.stat(path)
    if (now.st_size, now.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise OSError(f"{path} changed while being hashed -- retry once it is stable")
    try:  # atomic replace: a reader never sees a half-written sidecar
        tmp = f"{sc}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "md5_b64": digest}, f)
        os.replace(tmp, sc)
    except OSError:
        pass
    return digest


@contextlib.contextmanager
def _cached_reference_md5():
    """Route wandb's local-reference content hash through _md5_b64 for one add_reference.
    Private wandb hook: if it is absent (another wandb version) wandb hashes as before."""
    try:
        from wandb.sdk.artifacts.storage_handlers import local_file_handler as lfh
    except ImportError:
        lfh = None
    orig = getattr(lfh, "_md5_content", None)
    if orig is None:
        yield
        return
    lfh._md5_content = _md5_b64
    try:
        yield
    finally:
        lfh._md5_content = orig


def features_artifact(path: str) -> wandb.Artifact:
    art = wandb.Artifact(_artifact_name("feat", path), type="features", metadata=file_identity(path))
    with _cached_reference_md5():
        art.add_reference("file://" + os.path.abspath(path), checksum=True)
    meta = path.replace(".npz", "_meta.json")
    if os.path.exists(meta):
        art.add_reference("file://" + os.path.abspath(meta), checksum=True)
    return art


def log_features(path: str, attempts: int = 4, settle_s: float = 10.0) -> None:
    """Producer side (an extractor): record the cache it just wrote.
    A multi-GB file just closed on Lustre can still move its size/mtime/ctime for a few seconds
    while the OSTs settle, so hashing it right away can raise "changed while being hashed"
    (eurosat shard 2 of the GeoCore campaign, job 6533086, 2026-11-03, with the cache complete).
    That error is retried after a pause; any other error, or a file that keeps moving, raises."""
    if wandb.run is None:
        return
    for attempt in range(1, attempts + 1):
        st = os.stat(path)
        try:
            art = features_artifact(path)
            break
        except OSError as e:
            if "changed while being hashed" not in str(e) or attempt == attempts:
                raise
            now = os.stat(path)
            print(
                f"[wb] {path} moved while being hashed (attempt {attempt}/{attempts}): "
                f"size {st.st_size}->{now.st_size} mtime_ns {st.st_mtime_ns}->{now.st_mtime_ns} "
                f"ctime_ns {st.st_ctime_ns}->{now.st_ctime_ns}; retrying in {settle_s:g}s",
                flush=True,
            )
            time.sleep(settle_s)
    wandb.run.log_artifact(art)


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
