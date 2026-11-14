"""GEO-Bench classification tasks registered in eval/features.OFFICIAL after m-eurosat
(m-forestnet, m-so2sat, m-brick-kiln, m-pv4ger, m-bigearthnet), served from the RGB trees
written by `python -m eval.export_geobench`. One generic wrapper: split sizes, class order,
root and label kind all come from the OFFICIAL entry, so a task is never described twice.

Images go through EuroSATDataset's transform (bicubic to img_size, [-1,1]) -- the same
instrument as m-eurosat. The split listing is eval.features.list_official_split, the one
the DINO arm uses, so both arms read the identical ordered image list. Multi-label labels
are multi-hot rows. m-eurosat keeps its own wrapper (src/datasets/m_eurosat.py): its banked
caches were extracted through it.
"""

from __future__ import annotations

from data.eurosat_dataset import EuroSATDataset
from data.utils import round_up_to_multiple
from eval import features as F
from registry import register_dataset
from torch.utils.data import DataLoader
from utils import seed_worker


class OfficialSplit(EuroSATDataset):
    """EuroSATDataset.__getitem__ over an explicit (files, labels) list."""

    def __init__(self, files, labels, img_size: int) -> None:
        self.img_size = round_up_to_multiple(img_size, 16)
        self.samples = [(f, y, "") for f, y in zip(files, labels, strict=True)]


def _wrapper(name: str) -> type:
    class GeoBenchWrapper:
        def __init__(self, cfg) -> None:
            self.class_names: list[str] = F.official_classes(name)
            self.category_list: list[str] = self.class_names  # prompts are unused by extraction

        def get_data(self, cfg) -> dict:
            # Same refusals as m_eurosat: smoke with --subset-size; GEO-Bench's own label-fraction
            # partitions are not reproduced, so a private subsample must not claim the official one.
            if cfg.max_samples is not None:
                raise ValueError(f"{name} does not support max_samples; smoke with --subset-size instead")
            if cfg.label_fraction != 1.0:
                raise ValueError(
                    f"{name} serves the official GEO-Bench partitions only; use label_fraction=1.0"
                )
            img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size
            loaders = {}
            for split in F.official_spec(name)["sizes"]:  # list_official_split checks the size
                ds = OfficialSplit(*F.list_official_split(name, split, cfg.dataset.path), img_size)
                loaders[split] = DataLoader(
                    ds,
                    batch_size=cfg.batch_size,
                    shuffle=(split == "train"),
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                )
            return loaders

    GeoBenchWrapper.__name__ = f"GeoBenchWrapper_{name}"
    return GeoBenchWrapper


for _name in F.OFFICIAL:
    if _name != "m_eurosat":
        register_dataset(_name)(_wrapper(_name))
