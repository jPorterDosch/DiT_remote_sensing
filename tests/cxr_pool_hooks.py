"""Hooks the VinDr-CXR pool tests install as `eval.export_cxr._VINDR_POOL_HOOK`.

They live here and not in test_export_cxr.py because a spawned pool worker unpickles its hook by
importing the module that defines it. Importing the test module costs each worker pandas, pyarrow,
torch and torchvision: 212 s cold on a compute node (job 6606235) against 13 s for eval.export_cxr
alone, which put the dying-worker test past its 240 s alarm in the first full-suite run (job
6605716). Keep this module to the standard library.
"""

from __future__ import annotations

import os
import warnings


def warn_secret(img: str) -> None:
    """Runs inside a pool worker: a library warning that quotes a secret."""
    warnings.warn("XSECRETX patient string")


def die_on(target: str, img: str) -> None:
    """Runs inside a pool worker (as functools.partial(die_on, <id>)): that scan kills its process."""
    if img == target:
        os._exit(1)
