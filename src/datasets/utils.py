from __future__ import annotations

from torch.utils.data import DataLoader

from utils import seed_worker

# Use module constant for seed to ensure consistent data splits across runs.
_DATA_TRAIN_SEED = 42


def make_loaders(train_ds, test_ds, cfg) -> dict:
    """Build standard train/test DataLoaders shared by dataset wrappers.

    Train is shuffled; test is not. Both use cfg.batch_size / cfg.num_workers,
    pin memory, and seed their workers for reproducibility.
    """
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
