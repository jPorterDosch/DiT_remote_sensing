from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from utils import seed_worker

# Use module constant for seed to ensure consistent data splits across runs.
_DATA_TRAIN_SEED = 42


def _strided_indices(n: int, k: int) -> list[int]:
    """k indices spread evenly over range(n) (all of them when k >= n)."""
    if k >= n:
        return list(range(n))
    step = n / k
    return [min(n - 1, int(i * step)) for i in range(k)]


def make_loaders(train_ds, test_ds, cfg) -> dict:
    """Build standard train/test DataLoaders shared by dataset wrappers.

    Train is shuffled; test is not. Both use cfg.batch_size / cfg.num_workers,
    pin memory, and seed their workers for reproducibility.

    cfg.max_samples, when set, caps BOTH splits at max_samples items for smoke tests
    (the field existed but was consumed nowhere before 2026-09-10). The cap STRIDES across
    the dataset rather than taking a prefix: sample order is class-directory-major, so a
    prefix of 32 lands entirely inside class 0 -- which is how the first smoke run produced
    a single-class test split and crashed the per-class F1 breakdown. Striding needs no
    label access (so it works for any dataset wrapper or Subset) and is deterministic, but
    it is only approximately balanced -- smoke accuracies remain meaningless by design.
    """
    if cfg.max_samples is not None:
        train_ds = Subset(train_ds, _strided_indices(len(train_ds), cfg.max_samples))
        test_ds = Subset(test_ds, _strided_indices(len(test_ds), cfg.max_samples))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return {"train": train_loader, "test": test_loader}
