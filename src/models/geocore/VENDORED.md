# Vendored GeoCore-9B model code

Source: https://github.com/KAIST-VICLab/GeoCore-9B at commit
`17406c03a9bd090ff1570809bf6bd7f4a35e2359` (2026-10-06), Apache License 2.0
(see LICENSE and NOTICE beside this file). Fetched 2026-11-02 from
`https://raw.githubusercontent.com/KAIST-VICLab/GeoCore-9B/<commit>/<path>`.

| file | upstream path | sha1 |
|---|---|---|
| flux2.py | models/flux2.py | 9ef41f7cb215366220cc8a78216c4e4ac7b63c24 |
| vae_flux2.py | models/vae_flux2.py | 25146aa69032186263f1f4d782db5483276b5495 |
| LICENSE | LICENSE | 8c625feb1daf943dc7264dfc883d0a4fabd95cad |
| NOTICE | NOTICE | 726cec048e83edbfaec59eaa7a6dfd8886c7fcd2 |

No line of these files was changed. Their imports are third-party only (torch, einops,
safetensors), so no import path needed rewriting. `tests/test_geocore_vendored.py` checks the
sha1s above against the files on disk.

Not vendored: `models/text_encoder.py`. It hard-codes the Hugging Face ids
`openai/clip-vit-large-patch14` and `google/t5-v1_1-xxl`; compute nodes have no network and
the weights live locally under `$STORE`. `null_text.py` in this directory reproduces its
`""`-prompt path (same tokeniser settings, same tensors) from the local directories.

Feature tapping (block inputs, early exit) is implemented in `trunk.py`, not by editing
`Flux2.forward`; the adapter in `adapter.py` wraps everything behind the registry contract.
