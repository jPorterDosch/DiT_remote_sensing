def round_up_to_multiple(val: int, m: int = 16) -> int:
    """Round val up to the nearest multiple of m (e.g. for VAE-compatible image sizes)."""
    return ((val + m - 1) // m) * m
