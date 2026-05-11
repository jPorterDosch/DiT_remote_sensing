from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.eurosat_dataset import (
    EUROSAT_CLASSES,
    EuroSATDataset as _EuroSATDataset,
    get_eurosat_categories,
)
from registry import register_dataset
from utils import seed_worker

from .utils import _DATA_TRAIN_SEED


@register_dataset("eurosat")
class EuroSATDatasetWrapper:
    def __init__(self, cfg) -> None:
        self.category_list: list[str] = get_eurosat_categories()
        self.class_names: list[str] = list(EUROSAT_CLASSES)

    def get_data(self, cfg) -> dict:
        img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size
        train_ds = _EuroSATDataset(cfg.dataset.path, split="train", img_size=img_size)
        test_ds = _EuroSATDataset(cfg.dataset.path, split="test", img_size=img_size)

        label_fraction_size = int(len(train_ds) * cfg.label_fraction)
        if label_fraction_size < len(train_ds):
            generator = torch.Generator().manual_seed(_DATA_TRAIN_SEED)
            indices = torch.randperm(len(train_ds), generator=generator)[:label_fraction_size]
            train_ds = torch.utils.data.Subset(train_ds, indices=indices.tolist())

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
