import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Seed all GPUs
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic cuDNN behavior
        torch.backends.cudnn.deterministic = True
        # Avoid nondeterministic algorithms
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id) -> None:
    """
    Seed non-PyTorch RNGs inside each DataLoader worker.

    PyTorch assigns each worker a unique seed. This function reuses that seed
    for Python's `random` and NumPy so random augmentations/sampling performed
    inside `Dataset.__getitem__` do not share duplicated RNG state across workers.
    """
    worker_seed = torch.initial_seed()
    # Defensively seed NumPy RNG with modulo, since NumPy legacy API only accepts 32-bit seeds.
    # Our code SHOULD NOT be using the NumPy global seed, but older dependencies might.
    np.random.seed(worker_seed % 2**32)
    random.seed(worker_seed)


def to_jsonable(value: Any) -> Any:
    """
    Recursively convert dataclasses, dicts, lists, and tuples to JSON-serializable structures.
     - Dataclasses are converted to dicts using `asdict()`.
     - Dicts have their keys converted to strings and values processed recursively.
     - Lists and tuples have their elements processed recursively.
     - Other types are returned as-is (assuming they are already JSON-serializable).
     This function is useful for preparing complex nested data structures for JSON serialization,
     ensuring that all components are in a format that can be serialized by the `json` module.
     Note: This function does not handle all possible types (e.g., sets, custom objects without dataclass support),
     so additional handling may be needed for those cases.
     Args:
         value: The input value to convert to a JSON-serializable structure.
     Returns:
         A JSON-serializable version of the input value.
    """
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(value.items())}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return value


def env_value(name: str) -> str:
    """Shared 'unset, empty, and "0" all mean OFF' rule for the extraction-control env vars
    (FLUX_RANDOM_INIT / FIXED_COND_T / DEGRADE_TO). The rule was previously re-implemented
    inline at six call sites; gates that must agree byte-for-byte on when a flag is 'on'
    should share one implementation. Returns "" when off, else the raw value.
    """
    import os

    v = os.getenv(name, "")
    return "" if v in ("", "0") else v
