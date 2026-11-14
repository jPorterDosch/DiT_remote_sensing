"""GEO-Bench segmentation tasks registered in eval/features.OFFICIAL_SEG, served from the trees
written by `python -m eval.export_geobench --task <task>` (<root>/<split>/images/*.png +
<root>/<split>/masks/*.png). Extraction (tasks/seg_tokens.py) reads the images only; the masks
are read by eval/seg_probe.py when the per-token head is trained. Images go through the same
transform as every classification arm (EuroSATDataset: bicubic to img_size, [-1, 1]); the
split listing is eval.features.list_official_seg_split, so the FLUX and DINO token caches hold
the identical ordered image list.
"""

from __future__ import annotations

import os

from data.eurosat_dataset import EuroSATDataset
from data.utils import round_up_to_multiple
from eval import features as F
from registry import register_dataset
from torch.utils.data import DataLoader
from utils import seed_worker


class OfficialSegSplit(EuroSATDataset):
    """EuroSATDataset.__getitem__ over the split's image files; label = 0 for every image (the
    masks live beside the images and are not read here); .masks keeps the paired mask paths."""

    def __init__(self, images, masks, img_size: int) -> None:
        self.img_size = round_up_to_multiple(img_size, 16)
        self.samples = [(f, 0, "") for f in images]
        self.masks = list(masks)


def _wrapper(name: str) -> type:
    class GeoBenchSegWrapper:
        def __init__(self, cfg) -> None:
            self.class_names: list[str] = list(F.official_seg_spec(name)["classes"])
            self.category_list: list[str] = self.class_names  # prompts are unused by extraction

        def get_data(self, cfg) -> dict:
            if cfg.max_samples is not None or cfg.subset_size is not None:
                raise ValueError(
                    f"{name}: the token caches hold the FULL official splits; smoke with --max-images (seg_tokens)"
                )
            if cfg.label_fraction != 1.0:
                raise ValueError(
                    f"{name} serves the official GEO-Bench partitions only; use label_fraction=1.0"
                )
            img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size
            root = F.official_seg_spec(name)["root"]
            if cfg.dataset.path is not None and os.path.normpath(cfg.dataset.path) != os.path.normpath(root):
                # eval/seg_probe.py compares the cache's image paths with the listing under the
                # registered root, so a cache listed from any other spelling is refused there
                raise ValueError(
                    f"{name}: --dataset.path {cfg.dataset.path!r} must be the registered root {root!r}"
                )
            loaders = {}
            for split in F.official_seg_spec(name)["sizes"]:  # list_official_seg_split checks the size
                ds = OfficialSegSplit(
                    *F.list_official_seg_split(name, split, root), img_size
                )  # root spelling = the listing's
                loaders[split] = DataLoader(
                    ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,  # extraction order = listing order, always
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                )
            return loaders

    GeoBenchSegWrapper.__name__ = f"GeoBenchSegWrapper_{name}"
    return GeoBenchSegWrapper


for _name in F.OFFICIAL_SEG:
    register_dataset(_name)(_wrapper(_name))
