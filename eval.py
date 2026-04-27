# ruff: noqa: E402  — sys.path must be mutated before any local imports
from __future__ import annotations
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))  # src.data, src.models.*
sys.path.insert(0, os.path.join(_root, "src", "models"))  # flux.* internal imports

import torch
import tyro
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

import models  # noqa: F401  — resolves to src/models/, triggers @register_model decorators
import datasets  # noqa: F401  — triggers @register_dataset decorators
import tasks  # noqa: F401  — triggers @register_task decorators

from registry import MODELS, DATASETS, TASKS


@dataclass
class ModelConfig:
    name: str = "flux"
    ensemble_size: int = 8


@dataclass
class DatasetConfig:
    name: str = "eurosat"
    path: str = "/dataset/EuroSAT"


@dataclass
class EvalConfig:
    task: str = "classification"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    save_path: str = "features/"
    img_size: list[int] = field(default_factory=lambda: [224, 224])
    t: int = 260  ###调参[1,1000]
    k: int = 28  ###调参[0,57]
    cd: bool = False

    ## classification
    label_fractions: list[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 50.0, 100.0])
    clf_epochs: int = 50
    clf_lr: float = 1e-3
    clf_batch_size: int = 256
    seed: int = 42
    num_workers: int = 4
    overwrite_features: bool = False

    ## segmentation
    output_dir: str = "./davis_results_flux/"
    n_last_frames: int = 7
    size_mask_neighborhood: int = 12
    topk: int = 5
    temperature: float = 0.1


def main(cfg: EvalConfig) -> None:
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # this will let us know when we need to manually add a dataset, model, or task
    # "registering" new dataset is necessary but this silution still generalizes entrypoint
    assert cfg.dataset.name in DATASETS, f"Unknown dataset '{cfg.dataset.name}'. Registered: {list(DATASETS)}"
    assert cfg.model.name in MODELS, f"Unknown model '{cfg.model.name}'. Registered: {list(MODELS)}"
    assert cfg.task in TASKS, f"Unknown task '{cfg.task}'. Registered: {list(TASKS)}"

    dataset = DATASETS[cfg.dataset.name](cfg)
    model = MODELS[cfg.model.name](cfg, dataset.category_list)
    task = TASKS[cfg.task]()

    results_dir = os.path.join("results", cfg.dataset.name, cfg.model.name)
    os.makedirs(results_dir, exist_ok=True)

    task.run(cfg, model, dataset, results_dir)


if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
