import math
from dataclasses import dataclass, field
from enum import StrEnum

# FLUX(.1-dev) DiT block layout: 19 double-stream blocks (indices 0-18) followed by
# 38 single-stream blocks (indices 19-56), 57 total. The inversion cache stores one
# single-stream block. ["Only single-stream blocks expose the adaLN mod triple" -- SUPERSEDED
# 2026-10-05: double blocks now return block input + img_mod1 too (DOUBLE_BLOCK_FEATURE below).] Single source of truth for
# the block-range checks in run.py and Featurizer4Eval.invert_chain (which additionally
# asserts the loaded model matches these counts).
NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38
NUM_BLOCKS = NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS  # 57 (valid k: 0..56)

# Feature definition of DOUBLE-stream blocks. Until 2026-10-05 DoubleStreamBlock.forward_feat
# returned a mid-block, post-attention, img_mod2-modulated activation and no mod; since then the
# block INPUT plus img_mod1 (the single blocks' definition). The config hash and filename did
# not change, so every extractor stamps it (double_block_stamp) and eval.features refuses an
# unstamped double-block cache. Single source of truth -- never hand-copy (rule 10).
DOUBLE_BLOCK_FEATURE = "block_input+img_mod1"


def double_block_stamp(k, num_double: int = NUM_DOUBLE_BLOCKS) -> dict:
    """Meta fields for a cache at block(s) k: the double-block definition stamp iff any k is a
    double-stream block, else {} (single-block-only caches keep their meta key-for-key).
    `num_double` is the model's double-block count: FLUX.1-dev's 19 by default; another backbone
    (GeoCore-9B: 8) passes its own, so the stamp follows the model that produced the cache."""
    ks = [k] if isinstance(k, int) else list(k)
    return {"double_block_feature": DOUBLE_BLOCK_FEATURE} if any(b < num_double for b in ks) else {}


def map_timesteps_to_grid(cache_timesteps, num_inversion_steps):
    """Map nominal timesteps ([1, 1000], the t/1000 convention) onto the uniform
    integration grid t_i = i / num_inversion_steps.

    Single source of truth for the on-grid rule: run.py config validation and
    Featurizer4Eval both call this so the two can never drift. Each requested timestep
    must land exactly on a grid point (a mismatch would silently extract features at a
    different t than the label claims). Returns {nominal_t: grid_index} in ascending
    order, or raises ValueError naming the smallest valid integer spacing.
    """
    # (t * n) % 1000 == 0  <=>  t is a multiple of 1000 / gcd(1000, n). This is always
    # an integer, unlike 1000/n, so the error message is actionable for any n.
    step = 1000 // math.gcd(1000, num_inversion_steps)
    mapping: dict[int, int] = {}
    for ct in cache_timesteps:
        exact = ct * num_inversion_steps / 1000.0
        gi = round(exact)
        if abs(exact - gi) > 1e-9:
            raise ValueError(
                f"timestep {ct} does not lie on the {num_inversion_steps}-step integration "
                f"grid: each t must be a multiple of {step}. Adjust t or num_inversion_steps."
            )
        mapping[ct] = gi
    return dict(sorted(mapping.items()))


def validate_inversion_block(k):
    """Validate the block index for inversion feature caching: exactly one single-stream
    block ([NUM_DOUBLE_BLOCKS, NUM_BLOCKS-1]). Returns the index, or raises ValueError.
    Shared by run.py config validation and Featurizer4Eval.invert_chain.
    """
    k_list = [k] if isinstance(k, int) else list(k)
    if len(k_list) != 1 or not (NUM_DOUBLE_BLOCKS <= k_list[0] <= NUM_BLOCKS - 1):
        raise ValueError(
            f"extraction_mode=inversion caches exactly one single-stream block: k must be a "
            f"single index in [{NUM_DOUBLE_BLOCKS}, {NUM_BLOCKS - 1}], got {k}. Double-block and "
            f"multi-block caching are implemented for the one-shot path only."
        )
    return k_list[0]


class ProbeType(StrEnum):
    """
    Probe type for downstream classification. KANs have learnable activation functions
    and splines instead of linear weights, which allows them to capture more complex relationships.

    Fourier KAN extends this by replacing the splines with Fourier series.

    KAN: https://arxiv.org/abs/2404.19756
    Fourier KAN/KAN for linear probing: https://arxiv.org/pdf/2408.08803
    """

    LINEAR = "LINEAR"
    MLP = "MLP"
    KAN = "KAN"
    FOURIER_KAN = "FOURIER_KAN"


class ExtractionMode(StrEnum):
    """How multi-timestep features are produced along the timestep axis.

    ONESHOT: independent one-shot noising to each timestep t (x_t = t*eps + (1-t)*x0),
             same eps per image across all t — the original extraction path.
    INVERSION: a single chained RF-Solver reverse-ODE trajectory per image from the
               clean image toward noise; each state depends on the previous one.
               Features are cached at the requested timesteps, which must lie on the
               num_inversion_steps integration grid (see run.py).
    """

    ONESHOT = "ONESHOT"
    INVERSION = "INVERSION"


# ---------------------------------------------------------------------------------------------
# finetune-diffusion: semantic alignment (tasks/align.py) and the Git-10M corpus (datasets/git10m)
# Both are nested RunConfig fields BLACKLISTED from config_hash (CLAUDE.md rule 9: hashing new
# fields would repoint every existing run directory); their identity goes into the run name as
# +<arm>-<8 hex> (RunConfig.make_run_name) and into every checkpoint / meta.json in full.
ALIGN_ARMS = ("fm", "naive", "proj", "sem")


@dataclass
class AlignConfig:
    """Flow matching + semantic alignment of block-k INPUT tokens to a frozen teacher.

    arm: "" = alignment OFF (the pre-2026-10 finetune-diffusion path, unchanged).
         "fm"    = g_FM only (the semantic gradient is measured, the projector trained, nothing
                   semantic reaches the LoRA);
         "naive" = g_FM + weight * g_SEM;
         "proj"  = g_FM + weight * P(g_SEM), P = one-sided per-block projection that removes the
                   component of g_SEM opposing g_FM (models/gradproj.py); g_FM itself is never
                   altered, and projector-only parameters get their ordinary semantic gradient.
         "sem"   = weight * g_SEM only (semantic-only adaptation, mentor P0 2026-10-23): the mirror
                   of fm -- g_FM is computed and measured every step, multiplied by zero. With a
                   single loss under AdamW the weight only rescales the gradient, which Adam
                   normalises away (clipping at 1.0 does not bind at these norms), so the default
                   0.5 is kept for identity only. The semantic loss at block k reaches blocks < k:
                   blocks >= k get no gradient at all and stay exactly pretrained.
    Every arm runs the same forwards/backwards per step on the same seeded batches, timesteps
    and noise, so "matched compute" holds in FLOPs and in data.
    """

    arm: str = ""
    # What the projector reads at block k. "ditf" (default, user decision 2026-10-14) = the DiTF
    # representation: the massive-activation channels (RunConfig.discard_channels) zeroed,
    # LayerNorm, the block's own adaLN (1 + scale) * x + shift (models/flux/adapter.py
    # FluxModel.extract, per token, no pooling) -- i.e. the block's actual modulated input.
    # "ln" = the massive channels zeroed + LayerNorm only, WITHOUT the block's scale/shift: the
    # ablation that isolates the massive-activation removal from the modulation.
    # "raw" = the block input tokens as REPA uses them (the massive channels included).
    # NOTE (review 2026-10-15): the OFFICIAL matrix probes (eval/probe.official_flux, eval/matrix.py)
    # read the RAW pooled block-input tokens (per-t slices + concat, StandardScaler + LR), not the
    # DiTF view (eval/features.ditf is the cv/budget path). The ditf choice is therefore about
    # WHAT gets aligned (modulated input without the massive channels), not probe parity.
    # ABLATION of interest: if the massive activations carry generation and the remaining
    # channels take the semantics without hurting it, raw vs ditf shows it (user note 2026-10-14).
    feat: str = "ditf"
    # Sampling weights over RunConfig.t for the training timesteps (microbatches + projector
    # warmup; validation / measurement grids are fixed). None = uniform over the grid (the
    # pre-2026-10-14 behaviour). Same length as t, positive; normalized here.
    t_weights: list[float] | None = None
    # Noise-restricted alignment (mentor direction 2026-10-25: gradient agreement is strongest at
    # high noise, discriminative features at low noise -- does alignment at one noise level move
    # the other?): a per-grid-t multiplier on the TRAINING alignment loss of each image, indexed
    # like t_weights over RunConfig.t. 0 = no semantic gradient from images drawn at that t; the
    # FM loss, the t sampling (t_weights), the validation / measurement passes (unweighted) and
    # the arm's combination rule are unchanged. The weights are divided by their expectation under
    # the t sampling (t_weights) so E[w] = 1: the semantic gradient mass per step matches the
    # unrestricted run in expectation and only the noise window differs (review 2026-10-26).
    # None = 1 everywhere (the pre-2026-10-25 identity; part of identity() when set). Same length
    # as t, >= 0, not all 0; naive / proj / sem only.
    sem_t_weights: list[float] | None = None
    # Frozen target encoder (models/align_targets.py): a DINO preset, family:arg or import:mod:Attr.
    target: str = "dinov3_vit7b16_sat"
    # Shuffled-teacher control (spec; built 2026-10-16). "" = the real pairing. "batch" = every
    # image is aligned to the target tokens of ANOTHER image of its microbatch (the targets rolled
    # by one along the batch): the same marginal distribution of targets, no link to the image.
    # Everything else (data order, timesteps, noise, LoRA, projector, arm) is identical to the
    # arm it controls, so any downstream gain the control shows is NOT image-linked semantics
    # (CLAUDE.md rule 2: a control that gains is a broken instrument). Rows of one microbatch are
    # independent draws of the seeded per-pass permutation (datasets.git10m), so the partner is
    # an unrelated scene. Needs batches of >= 2 rows: AlignState refuses a 1-row batch, which
    # would roll onto itself and silently un-shuffle.
    target_shuffle: str = ""
    # Fixed w in g_FM + w * g_SEM (REPA default 0.5). A per-step norm-matched weight was removed
    # on 2026-10-10: it optimizes no fixed objective and amplifies noise as |g_SEM| shrinks.
    weight: float = 0.5
    # Projector from the DiT hidden size to the target dim. "mlp" = REPA's 3-layer SiLU MLP.
    # TODO(spec s9): "linear" (a single nn.Linear) is the second required architecture; not built.
    proj_arch: str = "mlp"
    proj_hidden: int = 2048
    proj_lr: float = 1e-3
    # Projector-only microbatches before step 0 (LoRA frozen; identical in every arm) so the
    # semantic gradient at step 0 goes through a trained readout, not a random one.
    proj_warmup_steps: int = 100
    # Unit of the one-sided projection: per DiT block or the whole LoRA vector.
    proj_scope: str = "block"
    # Relative tolerances of the executed verifiers (fp64 dots): the projected gradient, and the
    # fp32 gradient the optimizer actually consumes.
    proj_tol: float = 1e-5
    applied_tol: float = 1e-4
    # Measurement passes: per-block cos(g_FM, g_SEM) at fixed timesteps on fixed batches
    # (posterior-mean latents, seeded eps) at init and final, plus every meas_every steps (0 = off).
    meas_t: list[int] = field(default_factory=lambda: [100, 260, 420, 580, 740, 900])
    meas_batches: int = 8
    meas_every: int = 0
    # Fixed-seed generation (step 0 = the pretrained model, and final): n images, sampler steps.
    gen_n: int = 4
    gen_steps: int = 28
    gen_seed: int = 0
    no_gen: bool = False
    # Distributional generation metric (spec s8, eval/gen_metrics.py): FID + KID on Inception-v3
    # pool features between fid_n fixed-seed samples (seeds gen_seed + 1000 + batch, batches of
    # gen_n, the grids' sampler settings) and the first fid_n strided rows of the run's validation
    # source (git10m: the holdout shared by every index, unflipped) -- the same reference images
    # in every arm, identity recorded. Computed at the FINAL checkpoint; fid_step0 also scores
    # the pretrained model at step 0 (identical in every arm: run it in ONE arm). 0 = off.
    fid_n: int = 0
    fid_step0: bool = False
    fid_subsets: int = 100
    # Resumable checkpoint (ckpt.pt in the run dir, atomic) every N optimizer steps; 0 = never.
    ckpt_every: int = 0
    # Path of a ckpt.pt to continue; the run name, config identity and step budget must match.
    resume: str = ""
    # Diagnostic, no training (mentor request 2026-10-22): path of a lora_*.pt of a FINISHED run of
    # this same configuration. Loads its LoRA + projector, runs the validation pass twice (LoRA on,
    # then every LoRALinear.scaling = 0 = the pretrained backbone under the trained projector) and
    # writes <run dir>/split_eval.json: how much of the teacher cosine the backbone carries versus
    # the projector head (tasks/train_diffusion._split_eval). Not part of identity(): same run dir.
    split_eval: str = ""
    # Alignment on/off branching (HASTE-style, user decision 2026-10-23): after stop_step optimizer
    # steps the semantic gradient no longer reaches the LoRA (w_eff = 0); everything else -- the
    # projector training, the measurements, the data / t / noise / VAE streams -- continues
    # unchanged, so the run is its arm up to stop_step and an fm arm after it. naive / proj only.
    # 0 = never (and then absent from identity(): existing run names do not move).
    stop_step: int = 0
    # Logging only, never part of identity() (no effect on training, own RNG streams):
    # snapshot_every   LoRA + projector (+ LoRA AdamW state) as checkpoints/lora_snap_step*.pt every
    #                  N steps (per-checkpoint split-eval / virtual steps / probes; 0 = off);
    # keep_ckpt_every  a copy of the resumable ckpt.pt as ckpt_step*.pt every N steps (exact branch
    #                  points; a multiple of ckpt_every; 0 = off);
    # feat_stats       per-block feature statistics at every measurement pass (token norm, the two
    #                  massive channels, participation ratio, drift from the pretrained model via
    #                  the LoRA-off forward) into stats.npz meas/<tag>/t<t>/feat_*.
    snapshot_every: int = 0
    keep_ckpt_every: int = 0
    feat_stats: bool = False
    # Diagnostic, no training (P1 "held-out virtual steps", plan 2026-10-23): comma-separated paths
    # or globs of lora_*.pt checkpoints of a FINISHED run of this same configuration. For each one:
    # the mean FM and semantic LoRA gradients on vs_grad_batches fixed TRAINING batches (the
    # measurement batches, posterior-mean latents, t in meas_t, the measure() noise stream), then
    # temporary normalised SGD steps along -g_FM, -g_SEM, -(g_FM + w g_SEM) and -(g_FM + w P(g_SEM))
    # of size eta x ||theta_LoRA|| (eta in VS_ETAS), each way (descent and ascent, which separates
    # the first-order change from the curvature), each scored by the validation pass (the 128
    # held-out images x the t grid), parameters restored bit for bit after every step. Writes
    # <run dir>/virtual_steps_<steps>[_pert].json, refusing to overwrite (tasks/train_diffusion._virtual_steps).
    # Not part of identity().
    virtual_steps: str = ""
    vs_grad_batches: int = 32
    # vs_per_t adds, per noise level in meas_t, the directions g_FM@t and g_SEM@t (the training
    # gradient at that t alone) to the stepped set: the t_grad x t_eval cross-effect of a step
    # taken at one noise level on the held-out FM loss and teacher cosine at every grid t (12 more
    # directions, ~x4 the validation passes). The per-t gradient statistics (within-objective
    # coherence, cross-t and cross-objective cosines, first-order predictions from the per-t
    # validation gradients) are always written. Not part of identity().
    vs_per_t: bool = False
    # Diagnostic, no training (gradient geometry, 2026-10-28): comma-separated paths or globs of
    # lora_*.pt checkpoints of a FINISHED run of this same configuration. For each one, the per-batch
    # LoRA gradients of the FM loss and of the semantic loss on vs_grad_batches fixed TRAINING batches
    # (the measurement batches, posterior-mean latents) at every t in meas_t with measure()'s noise
    # stream, and their Gram matrix per block (plus the pooled one): every cosine between any two of
    # the 2 x |meas_t| x vs_grad_batches gradient vectors, per block, from which cross-timestep /
    # cross-objective cosines of the per-t mean gradients and a PCA of the gradient cloud follow
    # (experiments/figs/grad_geometry.py). Writes <run dir>/grad_geometry_s<step>.npz per checkpoint,
    # refusing to overwrite (tasks/train_diffusion._grad_geometry). Not part of identity().
    grad_geometry: str = ""

    LOGGING_ONLY = (
        "resume",
        "ckpt_every",
        "split_eval",
        "snapshot_every",
        "keep_ckpt_every",
        "feat_stats",
        "virtual_steps",
        "vs_grad_batches",
        "vs_per_t",
        "grad_geometry",
    )

    def identity(self) -> dict:
        """The fields that define WHICH run this is (run name / resume check): everything but
        the resume path and the checkpoint cadence."""
        d = {k: v for k, v in self.__dict__.items() if k not in self.LOGGING_ONLY}
        if not self.stop_step:
            d.pop("stop_step")  # off = the pre-2026-10-23 identity
        if not self.target_shuffle:
            # Off = the pre-2026-10-16 identity, byte for byte: the run names of the three arms
            # and the resume-identity check of the runs in flight must not move.
            d.pop("target_shuffle")
        d["meas_t"] = list(self.meas_t)
        d["t_weights"] = None if self.t_weights is None else [float(w) for w in self.t_weights]
        if self.sem_t_weights is None:
            d.pop("sem_t_weights")  # off = the pre-2026-10-25 identity
        else:
            d["sem_t_weights"] = [float(w) for w in self.sem_t_weights]
        return d


@dataclass
class Git10MConfig:
    """Training rows for dataset.name == "git10m" (dataset.path = the converted arrow dir).
    index: one-row-per-image index npz built from the geographic-exclusion manifests
    (python -m datasets.git10m build-index); val_index: the index whose RETAINED holdout is the
    validation set for every run (the most restrictive one, index_retained_cities_unique.npz),
    so val losses are comparable between indexes; "" = this run's own holdout."""

    index: str = ""
    val_index: str = ""
    holdout_every: int = 10_000
    # Horizontal flip (p = 0.5, decided per row from (seed, index position)): the only augmentation.
    hflip: bool = True

    def identity(self) -> dict:
        return dict(self.__dict__)
