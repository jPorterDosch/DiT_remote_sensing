from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import torch


@runtime_checkable
class ModelProtocol(Protocol):
    def extract(
        self,
        img: torch.Tensor,
        timestep: int,
        block_idx: int,
        ensemble_size: int,
        **kwargs: Any,
    ) -> torch.Tensor: ...


@runtime_checkable
class DatasetProtocol(Protocol):
    category_list: list[str]

    def get_data(self, cfg: Any) -> Any: ...


@runtime_checkable
class TaskProtocol(Protocol):
    def run(self, 
            cfg: Any, 
            model: Any, 
            dataset: Any, 
            results_dir: str) -> dict: ...


MODELS:   dict[str, type] = {}
DATASETS: dict[str, type] = {}
TASKS:    dict[str, type] = {}


def register_model(name: str):
    def decorator(cls: type) -> type:
        MODELS[name] = cls
        return cls
    return decorator


def register_dataset(name: str):
    def decorator(cls: type) -> type:
        DATASETS[name] = cls
        return cls
    return decorator


def register_task(name: str):
    def decorator(cls: type) -> type:
        TASKS[name] = cls
        return cls
    return decorator
