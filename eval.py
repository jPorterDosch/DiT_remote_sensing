# ruff: noqa: E402  — sys.path must be mutated before any local imports
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))  # src.data, src.models.*
sys.path.insert(0, os.path.join(_root, "src", "models"))  # flux.* internal imports

import torch
import tyro

warnings.filterwarnings("ignore")

import datasets  # noqa: F401  — triggers @register_dataset decorators
import models  # noqa: F401  — resolves to src/models/, triggers @register_model decorators
import tasks  # noqa: F401  — triggers @register_task decorators
from registry import DATASETS, MODELS, TASKS

from src.utils import seed_all


@dataclass
class ModelConfig:
    name: str = "flux"
    ensemble_size: int = 8


@dataclass
class DatasetConfig:
    name: str = "eurosat"
    path: str = "/dataset/EuroSAT"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items())}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return value


@dataclass
class EvalConfig:
    task: str = "classification"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    save_dir: str | None = None  # if not specified, will be generated from config hash (see make_run_name())
    img_size: list[int] = field(default_factory=lambda: [224, 224])
    t: int = 260  ###调参[1,1000]
    k: int | list[int] = (
        28  # [0, 57], for now, we can currently extract from multiple blocks, but have no aggregation methods implemented yet. Future work could explore this direction (e.g. concatenation, attention-based fusion, etc.
    )
    cd: bool = False
    discard_channels: list[int] = field(default_factory=lambda: [154, 1446])

    ## correspondence (spair)
    captions_path: str = "spair_detailed_captions.json"

    ## classification
    label_fractions: list[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 50.0, 100.0])
    clf_epochs: int = 50
    clf_lr: float = 1e-3
    clf_batch_size: int = 256
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 4
    overwrite_features: bool = False
    max_samples: int | None = None  # cap images per split for smoke tests; None = no limit

    def make_run_name(self) -> str:
        payload = _to_jsonable(asdict(self))

        dataset_name = payload["dataset"]["name"]
        model_name = payload["model"]["name"]
        seed = payload["seed"]

        blacklist = {
            # --- Irrelevant for feature extraction and training
            "save_dir",
            "device",
            "num_workers",
            "overwrite_features",
        }

        for k in blacklist:
            payload.pop(k, None)

        # Remove fields already represented in run name
        payload["dataset"].pop("name", None)
        payload["model"].pop("name", None)
        payload.pop("seed", None)

        # Hash config to get deterministic identifier for run.
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(serialized.encode()).hexdigest()[:8]

        return f"{dataset_name}_{model_name}_{digest}+{seed}"

    def __post_init__(self) -> None:
        if self.save_dir is None:
            self.save_dir = self.make_run_name()


def main(cfg: EvalConfig) -> None:
    # Set global seed
    seed_all(cfg.seed)
    # Registering a new dataset is still necessary, but this solution keeps the entrypoint generic.
    if cfg.dataset.name not in DATASETS:
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Registered: {list(DATASETS)}")
    if cfg.model.name not in MODELS:
        raise ValueError(f"Unknown model '{cfg.model.name}'. Registered: {list(MODELS)}")
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task '{cfg.task}'. Registered: {list(TASKS)}")

    dataset = DATASETS[cfg.dataset.name](cfg)
    model = MODELS[cfg.model.name](cfg, dataset.category_list)
    task = TASKS[cfg.task]()

    task.run(cfg, model, dataset)


if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
