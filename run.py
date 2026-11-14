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
import wandb

warnings.filterwarnings("ignore")

import datasets  # noqa: F401  — triggers @register_dataset decorators
import models  # noqa: F401  — resolves to src/models/, triggers @register_model decorators
import tasks  # noqa: F401  — triggers @register_task decorators
from registry import DATASETS, MODELS, TASKS
from config_types import (
    ALIGN_ARMS,
    AlignConfig,
    Git10MConfig,
    NUM_BLOCKS,
    ExtractionMode,
    ProbeType,
    map_timesteps_to_grid,
    validate_inversion_block,
)
from utils import seed_all, to_jsonable
from tasks.extraction import env_provenance
from eval import features as F
from eval.wb import WANDB_ENTITY, WANDB_PROJECT


@dataclass
class ModelConfig:
    # "flux" (FLUX.1-dev, src/models/flux) or "geocore" (GeoCore-9B, src/models/geocore; needs
    # RunConfig.geocore_ckpt / geocore_vae).
    name: str = "flux"
    ensemble_size: int = 8


@dataclass
class DatasetConfig:
    name: str = "eurosat"
    # Hashed into config_hash: do NOT repoint (rule 9). The data now lives under
    # DiT_remote_sensing/datasets/; DeepLearning/datasets/{EuroSAT,SPair-71k} are symlinks to it.
    path: str = "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT"


@dataclass
class RunConfig:
    task: str = "classification"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    hash: str | None = None

    # Root to save extracted features and trained classifiers. Name derived from config will be appended to this path so that multiple runs can be organized under the same directory.
    save_dir: str = "./models"
    img_size: list[int] = field(default_factory=lambda: [224, 224])
    # Timestep index in range [1,1000]. A strictly increasing list of K timesteps enables
    # multi-timestep extraction (task="extract"): one-shot noising to each t independently,
    # same eps per image across all K forward passes. E.g. --t 100 180 260 340 420 500 580.
    # With extraction_mode=inversion these are the nominal timesteps features are cached at
    # along the chain; each must lie on the num_inversion_steps integration grid.
    t: int | list[int] = 260
    # How multi-timestep features are produced: "oneshot" = independent one-shot noising
    # per t (original behavior); "inversion" = a single chained RF-Solver reverse-ODE
    # trajectory per image, caching features at the requested t values.
    extraction_mode: ExtractionMode = ExtractionMode.ONESHOT
    # extraction_mode=inversion only: integration granularity of the reverse ODE — how
    # finely the chain from clean image toward noise is discretized. A SEPARATE knob from
    # the cached feature timesteps t, which are a subsample of this grid (cf. Diffusion
    # Hyperfeatures: 50 integration steps, 11 cached). Finer grids invert more accurately.
    num_inversion_steps: int = 50
    # Seed for the dedicated eps RNG stream in multi-timestep extraction. None = use cfg.seed.
    eps_seed: int | None = None
    # task="extract_tokens" only: subset of `t` at which PRE-POOL token features are also
    # cached, as (N, S, L, C). Kept deliberately short — a per-token cache is L times the
    # size of the pooled one (L ~= 196 at img_size 224), so this is 3 timesteps, not 7.
    # Every value must appear in `t`; the pooled (N, K, C) cache is written regardless.
    per_token_t: list[int] = field(default_factory=lambda: [260, 420, 580])
    # task="extract_tokens" + extraction_mode=inversion only: also cache the model's
    # VELOCITY at every cached timestep, as (N, K, T, d) packed-latent. The velocity is the
    # model's output (where the prior says the sample should go), not the state (where it
    # is) — for a deterministic flow the state determines the whole path, so state-derived
    # features are mutually redundant, whereas the velocity is not a function of the cached
    # state alone. Already computed at every non-terminal step, so this costs one extra
    # forward per image (the terminal step, which otherwise early-exits at block k).
    want_velocity: bool = False
    # task="extract" only: number of train images to extract (class-stratified with
    # subset_seed). None = the full train split.
    subset_size: int | None = None
    # task="extract" only: which split to extract features from. "test" exists for
    # official-split probe evaluation (fit on full train features, score on full test
    # features); every historical cache is train-split, so "train" stays the default and
    # the field is config_hash-blacklisted (rule 9) with a "+test" run-name suffix.
    extract_split: str = "train"
    # task="extract_seg_tokens" only: strided SMOKE cap on the split (stamped into the cache tag,
    # never quotable). None = the full split.
    max_images: int | None = None
    # task="extract" only: split the extraction into num_shards contiguous index ranges
    # and run only shard_index (for SLURM array jobs over the expensive inversion chain).
    # Blacklisted from config_hash (rule 9); shards are distinguished by a "+sNofM"
    # run-name suffix. Shard caches record their indices in subset_indices; the merge
    # step (eval/features.load_flux_split, via eval.probe --protocol official) verifies
    # disjoint, complete coverage.
    num_shards: int = 1
    shard_index: int = 0
    # Configurable, but should remain consistent across experiments.
    subset_seed: int = 42
    # Block index in [0, 56] (19 double + 38 single-stream blocks): reference block for
    # the timestep-isolation sweep — override per experiment (e.g. 29 is EuroSAT-optimal).
    # The extraction paths cache the adaLN mod triple, which only single-stream blocks
    # (19-56) expose; extraction_mode=inversion enforces a single k in that range. We can
    # currently extract from multiple blocks, but have no aggregation methods implemented
    # yet. Future work could explore this direction (e.g. concatenation, fusion, etc.)
    k: int | list[int] = 28
    cd: bool = False
    discard_channels: list[int] = field(default_factory=lambda: [154, 1446])
    # model.name == "geocore" only: the GeoCore-9B snapshot directory (model-*.safetensors +
    # model.safetensors.index.json, verified against pinned sha256s in models/geocore/adapter.py)
    # and the FLUX.2 VAE safetensors file. Both are hashed into config_hash when the model is
    # geocore (the user's 2026-11-02 decision); for every other model they must stay empty and are
    # dropped from the hash, so no FLUX run directory moves (rule 9; tests/test_geocore_config.py
    # pins a historical FLUX hash).
    geocore_ckpt: str = ""
    geocore_vae: str = ""

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
    # DO NOT change this default to match the finetune pins. mask_ratio / mim_loss_weight /
    # guidance_scale are NOT config_hash-blacklisted, so they hash for EVERY task -- changing
    # a default silently repoints the run directory of every extraction config that does not
    # pass the flag explicitly, and 7 probe scripts hardcode the old hashes (verified
    # 2026-09-10: eurosat_flux_0b91191d -> f60bb2fe). The pinned training values are passed
    # explicitly in experiments/train_diffusion.sh and ENFORCED in
    # tasks/train_diffusion.py (PINNED_GUIDANCE / PINNED_MIM_LOSS_WEIGHT), which is where a
    # pin belongs -- a default cannot be a pin without moving cache identity.
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
    # finetune-diffusion protocol flags (2026-09-13 realignment to the standard RS
    # evaluation protocol: probe trains on the FULL train split concurrently with the
    # backbone, evaluation on the OFFICIAL test split -- Scale-MAE/SatMAE style. The old
    # complement-split machinery existed only because probes were CV'd INSIDE the train
    # subset; with eval on the held-out test split it is unnecessary).
    #   freeze_backbone: train ONLY the probe head on frozen features (the "Frozen" arm;
    #     also the protocol-matched frozen counterpart the 6s review demanded).
    #   supervised_finetune: probe CE gradients flow into the LoRA (the "Finetune" arm).
    #     DEFAULT False: adaptation stays LABEL-FREE (probe head is gradient-detached),
    #     which is what the paper's story requires -- turning this on makes the arm
    #     supervised and must be reported as such.
    freeze_backbone: bool = False
    supervised_finetune: bool = False

    # LoRA hyperparameters
    lora_wd: float = 0.0
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    wrap_output: bool = True  # whether to wrap the output projection in attention and/or MLP blocks with LoRA (in addition to the input projections, which are always wrapped). Future work could explore more flexible options for which projections to wrap.
    # Feeds BOTH extraction (tasks/utils.py) and finetune-diffusion training. Every feature
    # cache in the project is g=1.0 and every extraction script passes --guidance-scale 1.0
    # explicitly; finetune-diffusion PINS it to 1.0 (train_diffusion.PINNED_GUIDANCE). The
    # default stays at FLUX's 3.5 for the cache-identity reason on mask_ratio above.
    guidance_scale: float = 3.5

    # total_loss = flow_loss + mim_loss_weight * mim_loss. mim_loss_weight = the alpha.
    # The MIM angle is DROPPED: finetune-diffusion pins this to 0.0
    # (train_diffusion.PINNED_MIM_LOSS_WEIGHT). Default unchanged for cache identity (above).
    mim_loss_weight: float = 1.0

    # Linear LR warmup to mitigate spikes early on
    warmup_steps: int = 100

    # W&B logging -- one project for the whole pipeline; defaults live in eval/wb.py
    wandb_entity: str = WANDB_ENTITY
    wandb_project: str = WANDB_PROJECT

    ## finetune-diffusion: semantic alignment to a frozen teacher (config_types.AlignConfig,
    ## tasks/align.py) and the Git-10M training corpus (config_types.Git10MConfig,
    ## datasets/git10m.py). Both nested blocks are config_hash-BLACKLISTED (rule 9: hashing new
    ## fields repoints every existing run dir); their identity is the +<arm>-<8 hex> run-name
    ## suffix (make_run_name) and is recorded in full in every checkpoint / meta.json.
    align: AlignConfig = field(default_factory=AlignConfig)
    git10m: Git10MConfig = field(default_factory=Git10MConfig)

    def align_hash(self) -> str:
        """8-hex identity of the alignment + corpus settings (the resume path and the checkpoint
        cadence excluded: a resume must produce the SAME run name)."""
        payload = to_jsonable({"align": self.align.identity(), "git10m": self.git10m.identity()})
        return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[
            :8
        ]

    def config_hash(self) -> str:
        payload = to_jsonable(asdict(self))

        blacklist = {
            # --- Irrelevant for feature extraction and training
            "save_dir",
            "device",
            "num_workers",
            "overwrite_features",
            "wandb_entity",
            "wandb_project",
            "hash",
            # Selects which timesteps get an ADDITIONAL per-token rider (task=
            # "extract_tokens" only). It cannot change the pooled features, so including
            # it would change the run name — and therefore the cache directory — of every
            # previously-extracted run, silently breaking the config->directory mapping
            # that provenance relies on. The rider's own timesteps are recorded in the
            # token cache FILENAME and its meta.json instead, so two riders that differ
            # only in per_token_t still cannot be confused for one another.
            "per_token_t",
            # finetune-only protocol flags; hashing new fields would repoint every existing
            # run dir (CLAUDE.md rule 9). Runs that set them are distinguished in
            # make_run_name by +frozen / +sup suffixes instead.
            "freeze_backbone",
            "supervised_finetune",
            # Same rule-9 reasoning: hashing this new field would repoint every existing
            # run dir. Test-split runs are distinguished by the +test run-name suffix.
            "extract_split",
            # Ditto: shards are distinguished by the +sNofM run-name suffix.
            "num_shards",
            "shard_index",
            # Ditto (rule 9): extract_seg_tokens' smoke cap is stamped into its cache tag instead.
            "max_images",
            # Same reasoning: requesting the velocity adds an ADDITIONAL artifact and does
            # not alter the pooled features (the terminal step's forward_velocity_feat
            # returns the same block-k features as the early-exiting forward_feat), so it
            # must not change the run name. The velocity lands in its own
            # multistep_train_vels_*.npz, so the two artifacts cannot be confused.
            "want_velocity",
            # Same rule-9 reasoning: the alignment / corpus blocks are new; their identity is the
            # +<arm>-<align_hash> suffix of make_run_name.
            "align",
            "git10m",
        }

        if self.model.name != "geocore":
            # Empty by construction (validated in __post_init__); dropping them keeps every
            # pre-2026-11-02 FLUX hash byte-identical.
            blacklist |= {"geocore_ckpt", "geocore_vae"}

        for k in blacklist:
            payload.pop(k, None)

        # Remove fields already represented in run name
        payload["dataset"].pop("name", None)
        payload["model"].pop("name", None)
        payload.pop("seed", None)

        # Hash config to get deterministic identifier for run.
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode()).hexdigest()[:8]

    def make_run_name(self) -> str:
        payload = to_jsonable(asdict(self))

        dataset_name = payload["dataset"]["name"]
        model_name = payload["model"]["name"]
        seed = payload["seed"]

        # FLUX_RANDOM_INIT is an ENV override, not a config field, so it cannot reach
        # config_hash() -- an untrained run would otherwise claim the identical run
        # directory as the real-weights run with the same settings. Tag the directory too,
        # so provenance is visible without opening the cache. Real runs are unaffected,
        # so no previously-extracted cache changes path.
        # Control-provenance suffix, derived from the SAME parts list the cache tag uses
        # (tasks.extraction.env_provenance) -- one source of truth for both formats.
        _, _, parts = env_provenance(self, honors=_task_honors(self.task, self.model.name))
        suffix = "".join("+" + p for p in parts)
        # Protocol-mode suffixes (fields are config_hash-blacklisted; the name is their
        # identity -- CLAUDE.md rule 9).
        if self.freeze_backbone:
            suffix += "+frozen"
        if self.supervised_finetune:
            suffix += "+sup"
        if self.extract_split != "train":
            suffix += f"+{self.extract_split}"
        if self.num_shards > 1:
            suffix += f"+s{self.shard_index}of{self.num_shards}"
        if self.align.arm:
            suffix += f"+{self.align.arm}-{self.align_hash()}"
        if suffix:
            return f"{dataset_name}_{model_name}_{self.config_hash()}+{seed}{suffix}"
        return f"{dataset_name}_{model_name}_{self.config_hash()}+{seed}"

    def __post_init__(self) -> None:
        if self.label_fraction <= 0 or self.label_fraction > 1:
            raise ValueError(f"label_fraction must be in the range (0, 1], got {self.label_fraction}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

        if len(self.img_size) != 2 or any(x <= 0 for x in self.img_size):
            raise ValueError(f"img_size must contain exactly two positive integers, got {self.img_size}")

        if isinstance(self.t, int):
            if self.t < 1 or self.t > 1000:
                raise ValueError(f"t must be in the range [1, 1000], got {self.t}")
        else:
            if len(self.t) == 0:
                raise ValueError("t must be a non-empty list of timesteps")
            bad_t = [x for x in self.t if x < 1 or x > 1000]
            if bad_t:
                raise ValueError(f"all t values must be in the range [1, 1000], got invalid values {bad_t}")
            # Increment/curvature analysis along the timestep axis assumes an ordered sweep.
            if any(a >= b for a, b in zip(self.t, self.t[1:])):
                raise ValueError(f"t list must be strictly increasing, got {self.t}")

        if self.model.name == "geocore":
            if not self.geocore_ckpt or not self.geocore_vae:
                raise ValueError(
                    "model.name='geocore' requires --geocore-ckpt <snapshot dir> and --geocore-vae <safetensors>"
                )
        elif self.geocore_ckpt or self.geocore_vae:
            raise ValueError(
                f"geocore_ckpt / geocore_vae are only valid with model.name='geocore', got model.name={self.model.name!r}"
            )

        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError(f"subset_size must be positive or None, got {self.subset_size}")

        if self.extract_split not in ("train", "test", "val"):
            raise ValueError(f"extract_split must be 'train', 'test' or 'val', got {self.extract_split!r}")
        if self.num_shards < 1 or not (0 <= self.shard_index < self.num_shards):
            raise ValueError(
                f"invalid shard spec: shard_index={self.shard_index}, num_shards={self.num_shards}"
            )
        if self.num_shards > 1 and self.task not in ("extract", "extract_seg_tokens"):
            raise ValueError(
                f"num_shards={self.num_shards} is only honored by task='extract' / 'extract_seg_tokens', got task={self.task!r}"
            )
        if self.extract_split != "train" and self.task not in ("extract", "extract_seg_tokens"):
            # A set-but-unhonored flag must raise, not silently mislabel (rule 10): only
            # tasks/extraction.py and tasks/seg_tokens.py read extract_split; every other task
            # would train/probe on its usual splits while the run dir claimed "+test".
            raise ValueError(
                f"extract_split={self.extract_split!r} is only honored by task='extract' / 'extract_seg_tokens', got task={self.task!r}"
            )
        if self.max_images is not None and (self.task != "extract_seg_tokens" or self.max_images < 1):
            raise ValueError(
                f"max_images={self.max_images} is only honored by task='extract_seg_tokens' (>= 1), got task={self.task!r}"
            )
        if self.task == "extract_seg_tokens" and self.dataset.name not in F.OFFICIAL_SEG:
            raise ValueError(
                f"task='extract_seg_tokens' is for the OFFICIAL_SEG datasets {sorted(F.OFFICIAL_SEG)}, got {self.dataset.name!r}"
            )
        if self.dataset.name in F.OFFICIAL_SEG and self.task != "extract_seg_tokens":
            raise ValueError(
                f"segmentation dataset {self.dataset.name!r} is served to task='extract_seg_tokens' only, got {self.task!r}"
            )
        if self.align.arm and self.task != "finetune-diffusion":
            # Rule 10 again: only train_diffusion reads align; the run name would carry +<arm>.
            raise ValueError(
                f"align.arm={self.align.arm!r} is only honored by task='finetune-diffusion', got task={self.task!r}"
            )
        if self.align.arm and self.align.arm not in ALIGN_ARMS:
            raise ValueError(
                f"align.arm must be one of {ALIGN_ARMS} (or empty = off), got {self.align.arm!r}"
            )
        if self.align.resume and not os.path.isfile(self.align.resume):
            raise ValueError(f"align.resume does not exist: {self.align.resume}")
        if self.align.stop_step:
            if self.align.arm not in ("naive", "proj"):
                raise ValueError(
                    f"align.stop_step needs arm naive or proj (fm has no semantic step to stop, sem would stop training), got {self.align.arm!r}"
                )
            if not (0 < self.align.stop_step < self.max_train_steps):
                raise ValueError(
                    f"align.stop_step={self.align.stop_step} must lie in (0, max_train_steps={self.max_train_steps})"
                )
        if self.align.snapshot_every < 0 or self.align.keep_ckpt_every < 0:
            raise ValueError("align.snapshot_every / keep_ckpt_every must be >= 0")
        if self.align.keep_ckpt_every and (
            not self.align.ckpt_every or self.align.keep_ckpt_every % self.align.ckpt_every
        ):
            raise ValueError(
                f"align.keep_ckpt_every={self.align.keep_ckpt_every} must be a multiple of align.ckpt_every={self.align.ckpt_every} (> 0)"
            )
        if self.align.ckpt_every < 0 or self.align.meas_every < 0 or self.align.proj_warmup_steps < 0:
            raise ValueError("align.ckpt_every / meas_every / proj_warmup_steps must be >= 0")
        if self.align.fid_n < 0 or self.align.fid_subsets < 1:
            raise ValueError("align.fid_n must be >= 0 and align.fid_subsets >= 1")
        if self.align.fid_n and self.align.no_gen:
            raise ValueError("align.fid_n > 0 needs generation: drop --align.no-gen")
        if self.align.fid_n and self.align.fid_n < 2 * self.align.gen_n:
            raise ValueError(
                f"align.fid_n={self.align.fid_n} too small: use >= 2 x gen_n ({self.align.gen_n})"
            )
        if self.align.feat not in ("ditf", "ln", "raw"):
            raise ValueError(f"align.feat must be 'ditf', 'ln' or 'raw', got {self.align.feat!r}")
        if self.align.target_shuffle not in ("", "batch"):
            raise ValueError(f"align.target_shuffle must be '' or 'batch', got {self.align.target_shuffle!r}")
        if self.align.target_shuffle and not self.align.arm:  # rule 10: set-but-unhonored raises
            raise ValueError("align.target_shuffle is only honored with align.arm set")
        if self.align.target_shuffle and self.batch_size < 2:
            raise ValueError(
                "align.target_shuffle='batch' needs --batch-size >= 2 (a 1-row batch rolls onto itself)"
            )
        if self.align.t_weights is not None:
            tw = list(self.align.t_weights)
            n_t = len(self.t) if isinstance(self.t, list) else 1
            if len(tw) != n_t or any(not (w > 0) for w in tw):
                raise ValueError(
                    f"align.t_weights must give one positive weight per t ({n_t} timesteps: {self.t}), got {tw}"
                )
            if not self.align.arm:  # rule 10: a set-but-unhonored flag raises (uniform sampling otherwise)
                raise ValueError(
                    "align.t_weights is only honored with align.arm set: set the arm or drop the weights"
                )
        if self.align.sem_t_weights is not None:
            sw = list(self.align.sem_t_weights)
            n_t = len(self.t) if isinstance(self.t, list) else 1
            if len(sw) != n_t or any(not (w >= 0) for w in sw) or not any(w > 0 for w in sw):
                raise ValueError(
                    f"align.sem_t_weights must give one weight >= 0 per t, not all 0 ({n_t} timesteps: {self.t}), got {sw}"
                )
            if self.align.arm not in (
                "naive",
                "proj",
                "sem",
            ):  # rule 10: the fm arm applies no semantic gradient
                raise ValueError("align.sem_t_weights needs align.arm in naive / proj / sem")
        if (
            self.align.weight < 0
            or self.align.proj_lr <= 0
            or self.align.gen_n <= 0
            or self.align.gen_steps <= 0
        ):
            raise ValueError("align.weight must be >= 0; align.proj_lr, gen_n, gen_steps must be positive")
        bad_mt = [x for x in self.align.meas_t if x < 1 or x > 1000]
        if bad_mt or any(a >= b for a, b in zip(self.align.meas_t, self.align.meas_t[1:])):
            raise ValueError(
                f"align.meas_t must be strictly increasing within [1, 1000], got {self.align.meas_t}"
            )
        if (self.git10m.index or self.git10m.val_index) and self.dataset.name != "git10m":
            raise ValueError(
                f"git10m.* settings are only honored by dataset.name='git10m', got {self.dataset.name!r}"
            )
        if self.dataset.name == "git10m":
            if self.task not in ("finetune-diffusion", "gen-eval"):
                raise ValueError(
                    "dataset.name='git10m' is a training corpus: task must be 'finetune-diffusion' "
                    "(or 'gen-eval', which only reads its val holdout as the FID reference)"
                )
            if self.task == "finetune-diffusion" and not self.align.arm:
                raise ValueError(
                    "dataset.name='git10m' has no labels: set align.arm (fm = flow matching only)"
                )
            if not self.git10m.index:
                raise ValueError("dataset.name='git10m' needs git10m.index")
            if self.git10m.holdout_every < 2:
                raise ValueError("git10m.holdout_every must be >= 2")
        if (self.freeze_backbone or self.supervised_finetune) and self.task != "finetune-diffusion":
            # Same rule-10 guard: only train_diffusion reads these, but make_run_name stamps
            # +frozen / +sup for every task, so e.g. an extraction would be mislabelled.
            flag = "freeze_backbone" if self.freeze_backbone else "supervised_finetune"
            raise ValueError(f"{flag} is only honored by task='finetune-diffusion', got task={self.task!r}")

        if self.num_inversion_steps < 1:
            raise ValueError(f"num_inversion_steps must be >= 1, got {self.num_inversion_steps}")

        if self.extraction_mode == ExtractionMode.INVERSION:
            if not isinstance(self.t, list):
                raise ValueError("extraction_mode=inversion requires a list of timesteps t")
            # Grid-alignment rule lives in one place (shared with Featurizer4Eval) so the
            # config check and the extractor can never disagree; raises on misalignment.
            map_timesteps_to_grid(self.t, self.num_inversion_steps)

        max_k = NUM_BLOCKS - 1
        if isinstance(self.k, int):
            if self.k < 0 or self.k > max_k:
                raise ValueError(f"k must be in the range [0, {max_k}], got {self.k}")
        else:
            bad_k = [x for x in self.k if x < 0 or x > max_k]
            if bad_k:
                raise ValueError(
                    f"all k values must be in the range [0, {max_k}], got invalid values {bad_k}"
                )

        if not isinstance(self.k, int) and self.task not in ("extract", "extract_seg_tokens"):
            # Only the one-shot multi-block extractors read a block list. Every other path takes
            # one block: FluxModel.extract would apply the first block's adaLN mod to all of them
            # and emit B rows per image (review 2026-10-11). Refuse before the model loads.
            raise ValueError(
                f"a block list k={self.k} is only honored by task='extract' / 'extract_seg_tokens', got task={self.task!r}"
            )

        if self.extraction_mode == ExtractionMode.INVERSION:
            # Reject at config time instead of crashing after the model loads; the block
            # range is defined once in config_types (mirrors Featurizer4Eval.invert_chain).
            validate_inversion_block(self.k)

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

        if self.mask_ratio == 0 and self.mim_loss_weight != 0:
            warnings.warn(
                "mask_ratio=0 with a nonzero mim_loss_weight: the MIM loss is averaged over "
                "masked positions, of which there are none, so the MIM term contributes "
                "nothing. Set mask_ratio in (0, 1) if you intended to train the MIM head.",
                stacklevel=2,
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

        if self.freeze_backbone and self.supervised_finetune:
            raise ValueError(
                "freeze_backbone and supervised_finetune are mutually exclusive: one trains "
                "nothing but the probe head, the other trains the backbone THROUGH the head."
            )

        if self.lora_wd < 0:
            raise ValueError(f"lora_wd must be non-negative, got {self.lora_wd}")

        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")

        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")

        if self.lora_dropout < 0 or self.lora_dropout >= 1:
            raise ValueError(f"lora_dropout must be in the range [0, 1), got {self.lora_dropout}")

        if self.model.name == "geocore":
            # No guidance embedding in GeoCore-9B: the pinned value is 0.0 and the cache tag reads g0.0
            # (models/geocore/adapter.PINNED_GUIDANCE); any other value would mislabel the cache.
            if self.guidance_scale != 0.0:
                raise ValueError(
                    f"model.name='geocore' requires guidance_scale=0.0, got {self.guidance_scale}"
                )
        elif self.guidance_scale <= 0:
            raise ValueError(f"guidance_scale must be positive, got {self.guidance_scale}")

        if self.mim_loss_weight < 0:
            raise ValueError(f"mim_loss_weight must be non-negative, got {self.mim_loss_weight}")


def _task_honors(task: str, model_name: str = "flux") -> tuple[str, ...]:
    """Which extraction-control env flags each task's code path actually implements, for
    env_provenance's set-but-not-honored guard.

    finetune-diffusion is SPLIT on FIXED_COND_T and that is why it is rejected there
    (CORRECTED 2026-09-10, finding 7 -- the earlier rationale, that the task never reads the
    flag, was wrong): the training loop calls the flux model directly and ignores it, but the
    post-training probe goes extract_features -> extract_raw -> Featurizer4Eval.forward,
    which DOES read it. One run name cannot honestly describe vanilla training plus
    fixed-conditioning probing, so the combination is refused rather than mislabelled; run
    the probe separately under task=extract if you want it.

    FLUX_RANDOM_INIT (model load) and DEGRADE_TO (dataset __getitem__, further guarded by
    the degrade-aware-dataset check) genuinely apply to every task.

    A non-FLUX backbone (geocore) honors none of them: its loader never reads FLUX_RANDOM_INIT,
    its one-shot path never reads FIXED_COND_T, and DEGRADE_TO is a dataset hook the GeoCore
    protocol does not use -- a set flag must raise, not stamp a control label on a vanilla cache."""
    if model_name != "flux":
        return ()
    if task == "finetune-diffusion":
        return ("FLUX_RANDOM_INIT", "DEGRADE_TO")
    if task == "gen-eval":
        # Generation only (tasks/gen_eval.py): denoise() never goes through Featurizer4Eval.forward,
        # the only reader of FIXED_COND_T, so a set flag would be stamped into the run name while
        # the samples are vanilla (rule 10; review 2026-10-17).
        return ("FLUX_RANDOM_INIT", "DEGRADE_TO")
    return ("FLUX_RANDOM_INIT", "FIXED_COND_T", "DEGRADE_TO")


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

    # Resolve hash and save_dir for this run (after config is fully initialized).
    cfg.hash = cfg.config_hash()
    run_name = cfg.make_run_name()
    cfg.save_dir = os.path.join(cfg.save_dir, run_name)

    # Check for save_dir existence, and error if it already exists to avoid accidental overwriting.
    # A resume (align.resume) continues an existing run directory and is the one exception.
    if cfg.align.resume and not os.path.isdir(cfg.save_dir):
        raise ValueError(f"align.resume given but the run directory '{cfg.save_dir}' does not exist")
    # align.split_eval is a read-only diagnostic of a FINISHED run: it needs that run's directory.
    if cfg.align.split_eval and not os.path.isdir(cfg.save_dir):
        raise ValueError(f"align.split_eval given but the run directory '{cfg.save_dir}' does not exist")
    if cfg.align.virtual_steps and not os.path.isdir(cfg.save_dir):
        raise ValueError(f"align.virtual_steps given but the run directory '{cfg.save_dir}' does not exist")
    if cfg.align.grad_geometry and not os.path.isdir(cfg.save_dir):
        raise ValueError(f"align.grad_geometry given but the run directory '{cfg.save_dir}' does not exist")
    if os.path.exists(cfg.save_dir) and not (
        cfg.align.resume or cfg.align.split_eval or cfg.align.virtual_steps or cfg.align.grad_geometry
    ):
        raise ValueError(
            f"Save directory '{cfg.save_dir}' already exists. Please change the config or remove the existing directory to avoid overwriting previous results."
        )

    os.makedirs(cfg.save_dir, exist_ok=True)
    # Set global seed
    seed_all(cfg.seed)

    # Env-control validation for EVERY task, not just extraction. make_run_name stamps
    # +randinit/+fixedcondN/+degN unconditionally, so without this a task whose code path
    # never reads the env var (e.g. DEGRADE_TO on a dataset without the hook, FIXED_COND_T
    # on the inversion path) would get a run directory labelled as a control while holding
    # clean outputs -- provenance lying in the opposite direction from the poisoning the
    # tags were built to prevent.
    env_provenance(cfg, honors=_task_honors(cfg.task, cfg.model.name))

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        # eval/ runs log to the same project grouped by stage (eval/wb.py); group
        # run.py runs by task to match.
        group=cfg.task,
        job_type=cfg.task,
        config=asdict(cfg),
    )

    dataset = DATASETS[cfg.dataset.name](cfg)
    model = MODELS[cfg.model.name](cfg, dataset.category_list)
    task = TASKS[cfg.task]()

    task.run(cfg, model, dataset)

    wandb.finish()


if __name__ == "__main__":
    cfg = tyro.cli(RunConfig)
    main(cfg)
