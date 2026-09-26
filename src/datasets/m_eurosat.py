"""GEO-Bench m-eurosat: the OFFICIAL benchmark partitions (default_partition.json =
16,200 train / 996 val / 996 test — NOT the "2,000" both the GEO-Bench and SatDiFuser
papers' prose claims; see CLAUDE.md rule 16 and RESEARCH_NOTES 6t context), served as an
RGB image tree exported by `python -m eval.export_geobench --task m-eurosat`:

    data/m_eurosat_rgb/{train,val,test}/<ClassName>/<id>.png

Images are the geobench 13-band samples' RGB (B04,B03,B02) scaled /4095 and clipped —
the SatDiFuser convention — saved as uint8, so this repo's standard (x/255 - 0.5)*2
input normalization reproduces their [-1,1] range. 64px native; img_size upsampling is
this pipeline's usual bicubic (theirs is bilinear to 256 — an instrument difference we
keep, since OUR features are the instrument).

Unlike eurosat/resisc45, get_data also returns a "val" loader: GEO-Bench publishes a
real validation split, which is the legitimate place for probe selection (rule 1).
"""

from __future__ import annotations

import os

from data.eurosat_dataset import EUROSAT_CLASSES, EuroSATDataset as _EuroSATDataset, get_eurosat_categories
from registry import register_dataset
from torch.utils.data import DataLoader
from utils import seed_worker


@register_dataset("m_eurosat")
class MEuroSATDatasetWrapper:
    def __init__(self, cfg) -> None:
        self.category_list: list[str] = get_eurosat_categories()
        self.class_names: list[str] = list(EUROSAT_CLASSES)

    def get_data(self, cfg) -> dict:
        img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size
        if cfg.max_samples is not None:
            # Extraction smokes should use --subset-size (stratified, keeps .samples
            # visible to the extraction task); a Subset-wrapped smoke cap would break
            # the task's .samples access and silently skip the size check below.
            raise ValueError("m_eurosat does not support max_samples; smoke with --subset-size instead")
        if cfg.label_fraction != 1.0:
            # GEO-Bench ships its own 0.01x-1.00x label-fraction partitions; reproducing
            # them is not this wrapper's job, and silently subsampling here would make a
            # run claim the official partition while using a private one.
            raise ValueError(
                "m_eurosat serves the official GEO-Bench partitions only; run with label_fraction=1.0"
            )
        loaders = {}
        for split in ("train", "val", "test"):
            root = os.path.join(cfg.dataset.path, split)
            if not os.path.isdir(root):
                raise FileNotFoundError(
                    f"{root} missing — run `python -m eval.export_geobench --task m-eurosat` first (see its docstring)"
                )
            ds = _EuroSATDataset(
                root, split=None, img_size=img_size
            )  # split dirs are physical; no 80/20 here
            loaders[split] = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=(split == "train"),
                num_workers=cfg.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
        # Expected official sizes; a partial export must fail here, not probe quietly.
        expect = {"train": 16200, "val": 996, "test": 996}
        for split, n in expect.items():
            got = len(loaders[split].dataset)
            if got != n:
                raise ValueError(f"m_eurosat {split} has {got} images, expected {n} (partial/failed export?)")
        return loaders
