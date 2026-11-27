"""Disk-quota guard for the chest X-ray extraction jobs (stdlib only).

    python quota_guard.py --min-free-gb 900

Exit codes: 0 enough free space; 1 below the margin; 2 the quota could not be read or parsed
(failed or timed-out `lfs` call, unparseable output). Free space under the user's Lustre scratch quota is
compared with the margin. The quota is per user (CLAUDE.md memory: scratch + proj share it), so free space is the hard
block limit minus the kbytes used, as `lfs quota -u $USER <path>` reports them.
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys


def _kbytes(token: str) -> int:
    return int(token.rstrip("*"))  # an over-quota figure carries a trailing '*'


def free_gb(lfs_quota_text: str) -> float:
    """(limit_kbytes - used_kbytes) / 1e6 from `lfs quota -u USER PATH` output.

    A long filesystem name makes lfs wrap the numbers onto the next line, so the data row is
    read as whitespace tokens after the header: fs, kbytes, bquota, blimit, ...
    """
    lines = lfs_quota_text.splitlines()
    head = next((i for i, ln in enumerate(lines) if ln.split()[:2] == ["Filesystem", "kbytes"]), None)
    if head is None:
        raise ValueError("no 'Filesystem kbytes' header in lfs quota output")
    tok = " ".join(lines[head + 1 :]).split()
    if len(tok) < 4:
        raise ValueError("lfs quota output has no data row")
    used, soft, hard = _kbytes(tok[1]), _kbytes(tok[2]), _kbytes(tok[3])
    limit = hard or soft
    if not limit:
        raise ValueError("lfs quota reports no block limit")
    return (limit - used) / 1e6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-free-gb", type=float, required=True)
    ap.add_argument("--path", default="/lustre/isaac24/scratch")
    ap.add_argument("--user", default=getpass.getuser())
    a = ap.parse_args(argv)
    try:
        text = subprocess.run(
            ["lfs", "quota", "-u", a.user, a.path], check=True, capture_output=True, text=True, timeout=60
        ).stdout
        free = free_gb(text)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        print(f"quota_guard FAIL: cannot read quota ({type(e).__name__})")
        return 2
    ok = free >= a.min_free_gb
    print(f"quota_guard free_gb={free:.6f} min_free_gb={a.min_free_gb:.6f} {'OK' if ok else 'BELOW MARGIN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
