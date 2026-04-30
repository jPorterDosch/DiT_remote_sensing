import random

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
