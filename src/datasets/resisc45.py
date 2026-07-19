from __future__ import annotations

import torch

from data.resisc45_dataset import (
    RESISC45_CLASSES,
    RESISC45Dataset as _RESISC45Dataset,
    get_resisc45_categories,
)
from registry import register_dataset

from .utils import _DATA_TRAIN_SEED, make_loaders


@register_dataset("resisc45")
class RESISC45DatasetWrapper:
    def __init__(self, cfg) -> None:
        self.category_list: list[str] = get_resisc45_categories()
        self.class_names: list[str] = list(RESISC45_CLASSES)

    def get_data(self, cfg) -> dict:
        img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size
        train_ds = _RESISC45Dataset(cfg.dataset.path, split="train", img_size=img_size)
        test_ds = _RESISC45Dataset(cfg.dataset.path, split="test", img_size=img_size)

        label_fraction_size = int(len(train_ds) * cfg.label_fraction)
        if label_fraction_size < len(train_ds):
            generator = torch.Generator().manual_seed(_DATA_TRAIN_SEED)
            indices = torch.randperm(len(train_ds), generator=generator)[:label_fraction_size]
            train_ds = torch.utils.data.Subset(train_ds, indices=indices.tolist())

        return make_loaders(train_ds, test_ds, cfg)
