# ruff: noqa: E402  — sys.path must be mutated before any local imports
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field

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
from config_types import ProbeType
from utils import seed_all, to_jsonable


@dataclass
class ModelConfig:
    name: str = "flux"
    ensemble_size: int = 8


@dataclass
class DatasetConfig:
    name: str = "eurosat"
    path: str = "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT"


@dataclass
class RunConfig:
    task: str = "classification"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Root to save extracted features and trained classifiers. Name derived from config will be appended to this path so that multiple runs can be organized under the same directory.
    save_dir: str = "./models"
    img_size: list[int] = field(default_factory=lambda: [224, 224])
    t: int = 340  # Timestep index in range [1,1000]
    k: int | list[int] = (
        29  # [0, 57], for now, we can currently extract from multiple blocks, but have no aggregation methods implemented yet. Future work could explore this direction (e.g. concatenation, attention-based fusion, etc.
    )
    cd: bool = False
    discard_channels: list[int] = field(default_factory=lambda: [154, 1446])

    ## correspondence (spair)
    captions_path: str = "spair_detailed_captions.json"

    # TODO: move this to nested dataclass -- cfg is quickly filling up with more hparams, would be good to group by function.
    ## classification
    label_fraction: float = 1.0
    clf_epochs: int = 50
    clf_lr: float = 1e-3
    clf_batch_size: int = 256
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 4
    overwrite_features: bool = False
    max_samples: int | None = None  # cap images per split for smoke tests; None = no limit
    probe_type: ProbeType = ProbeType.MLP

    # The following only apply for KAN classifier heads
    grid_size: int = 5
    polynomial_order: int = 3

    ## Diffusion/flow-matching training with LoRA
    mask_ratio: float = 0.75
    finetune_max_epochs: int = 10
    finetune_bs: int = 1
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    finetune_lr: float = 1e-3
    # TODO: test higher values of max_train_steps, setting default low so we can get it running.
    max_train_steps: int = 1000

    # Timesteps for train and val logging
    log_train_steps: int = 10
    log_val_steps: int = 50

    # Path to a saved LoRA checkpoint to load before eval/training (empty = base model)
    lora_checkpoint: str = ""

    # LoRA hyperparameters
    lora_wd: float = 0.0
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    wrap_output: bool = True  # whether to wrap the output projection in attention and/or MLP blocks with LoRA (in addition to the input projections, which are always wrapped). Future work could explore more flexible options for which projections to wrap.
    guidance_scale: float = 3.5

    # total_loss = flow_loss + mim_loss_weight * mim_loss. mim_loss_weight = the alpha
    mim_loss_weight: float = 1.0

    # Linear LR warmup to mitigate spikes early on
    warmup_steps: int = 100

    def make_run_name(self) -> str:
        payload = to_jsonable(asdict(self))

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
        if self.label_fraction <= 0 or self.label_fraction > 1:
            raise ValueError(f"label_fraction must be in the range (0, 1], got {self.label_fraction}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

        if len(self.img_size) != 2 or any(x <= 0 for x in self.img_size):
            raise ValueError(f"img_size must contain exactly two positive integers, got {self.img_size}")

        if self.t < 1 or self.t > 1000:
            raise ValueError(f"t must be in the range [1, 1000], got {self.t}")

        if isinstance(self.k, int):
            if self.k < 0 or self.k > 57:
                raise ValueError(f"k must be in the range [0, 57], got {self.k}")
        else:
            bad_k = [x for x in self.k if x < 0 or x > 57]
            if bad_k:
                raise ValueError(
                    f"all k values must be in the range [0, 57], got invalid values {bad_k}"
                )

        if any(ch < 0 for ch in self.discard_channels):
            raise ValueError(f"discard_channels must be non-negative, got {self.discard_channels}")

        if self.clf_epochs <= 0:
            raise ValueError(f"clf_epochs must be positive, got {self.clf_epochs}")

        if self.clf_lr <= 0:
            raise ValueError(f"clf_lr must be positive, got {self.clf_lr}")

        if self.clf_batch_size <= 0:
            raise ValueError(f"clf_batch_size must be positive, got {self.clf_batch_size}")

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive or None, got {self.max_samples}")

        if self.mask_ratio < 0 or self.mask_ratio >= 1:
            raise ValueError(f"mask_ratio must be in the range [0, 1), got {self.mask_ratio}")

        if self.mask_ratio == 0:
            warnings.warn(
                "mask ratio is set to 0, meaning no masking will be applied during training. If this is intentional, you can ignore this warning."
                " If you intended to apply masking, please set mask_ratio to a value in the range (0, 1)."
            )

        if self.finetune_max_epochs <= 0:
            raise ValueError(f"finetune_max_epochs must be positive, got {self.finetune_max_epochs}")

        if self.finetune_bs <= 0:
            raise ValueError(f"finetune_bs must be positive, got {self.finetune_bs}")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )

        if self.finetune_lr <= 0:
            raise ValueError(f"finetune_lr must be positive, got {self.finetune_lr}")

        if self.max_train_steps <= 0:
            raise ValueError(f"max_train_steps must be positive, got {self.max_train_steps}")

        if self.log_train_steps <= 0:
            raise ValueError(f"log_train_steps must be positive, got {self.log_train_steps}")

        if self.log_val_steps <= 0:
            raise ValueError(f"log_val_steps must be positive, got {self.log_val_steps}")

        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")

        if self.warmup_steps >= self.max_train_steps:
            raise ValueError(
                f"warmup_steps must be less than max_train_steps, got "
                f"{self.warmup_steps} >= {self.max_train_steps}"
            )

        if self.lora_checkpoint and not os.path.isfile(self.lora_checkpoint):
            raise ValueError(f"lora_checkpoint does not exist: {self.lora_checkpoint}")

        if self.lora_wd < 0:
            raise ValueError(f"lora_wd must be non-negative, got {self.lora_wd}")

        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")

        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")

        if self.lora_dropout < 0 or self.lora_dropout >= 1:
            raise ValueError(f"lora_dropout must be in the range [0, 1), got {self.lora_dropout}")

        if self.guidance_scale <= 0:
            raise ValueError(f"guidance_scale must be positive, got {self.guidance_scale}")

        if self.mim_loss_weight < 0:
            raise ValueError(f"mim_loss_weight must be non-negative, got {self.mim_loss_weight}")


def main(cfg: RunConfig) -> None:
    # Registering a new dataset is still necessary, but this solution keeps the entrypoint generic.
    if cfg.dataset.name not in DATASETS:
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Registered: {list(DATASETS)}")
    if cfg.model.name not in MODELS:
        raise ValueError(f"Unknown model '{cfg.model.name}'. Registered: {list(MODELS)}")
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task '{cfg.task}'. Registered: {list(TASKS)}")
    if not os.path.exists(cfg.dataset.path):
        raise ValueError(f"Dataset path '{cfg.dataset.path}' does not exist.")

    # Resolve save_dir for this run (after config is fully initialized and run name can be generated).
    cfg.save_dir = os.path.join(cfg.save_dir, cfg.make_run_name())

    # Check for save_dir existence, and error if it already exists to avoid accidental overwriting.
    if os.path.exists(cfg.save_dir):
        raise ValueError(f"Save directory '{cfg.save_dir}' already exists. Please change the config or remove the existing directory to avoid overwriting previous results.")
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    # Set global seed
    seed_all(cfg.seed)

    dataset = DATASETS[cfg.dataset.name](cfg)
    model = MODELS[cfg.model.name](cfg, dataset.category_list)
    task = TASKS[cfg.task]()

    task.run(cfg, model, dataset)


if __name__ == "__main__":
    cfg = tyro.cli(RunConfig)
    main(cfg)
