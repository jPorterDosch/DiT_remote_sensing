"""GeoCore-9B (Do et al., arXiv 2608.01896) as a frozen feature extractor for the GEO-Bench
harness. Vendored model code: flux2.py / vae_flux2.py (see VENDORED.md). Our code: null_text.py,
trunk.py, adapter.py."""

GEOCORE_COMMIT = "17406c03a9bd090ff1570809bf6bd7f4a35e2359"

from . import adapter  # noqa: E402, F401 -- triggers @register_model("geocore")
