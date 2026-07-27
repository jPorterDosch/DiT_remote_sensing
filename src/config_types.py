import math
from enum import StrEnum

# FLUX(.1-dev) DiT block layout: 19 double-stream blocks (indices 0-18) followed by
# 38 single-stream blocks (indices 19-56), 57 total. Only single-stream blocks expose
# the adaLN mod triple the inversion feature cache stores. Single source of truth for
# the block-range checks in run.py and Featurizer4Eval.invert_chain (which additionally
# asserts the loaded model matches these counts).
NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38
NUM_BLOCKS = NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS  # 57 (valid k: 0..56)


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
            f"single index in [{NUM_DOUBLE_BLOCKS}, {NUM_BLOCKS - 1}], got {k}. Double blocks "
            f"(0-{NUM_DOUBLE_BLOCKS - 1}) do not expose the adaLN mod triple the cache stores, "
            f"and multi-block caching is not implemented."
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
