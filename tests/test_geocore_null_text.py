"""Empty-prompt tokenisation and embeddings for the GeoCore adapter (null_text.py).

pytest tests/test_geocore_null_text.py -v          # tokeniser-only test (CPU)
pytest --gpu tests/test_geocore_null_text.py -v    # also loads the encoders
"""

import os
import sys

import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from models.geocore import null_text as NT  # noqa: E402

needs_local = pytest.mark.skipif(not os.path.isdir(NT.T5_DIR_DEFAULT) or not os.path.isdir(NT.CLIP_DIR_DEFAULT),
                                 reason="local encoder snapshots absent")  # fmt: skip


@needs_local
def test_empty_prompt_token_ids():
    t5_ids, clip_ids = NT.null_token_ids()
    assert t5_ids == [1] + [0] * 255  # </s> then pad -- GeoCore's TextEncoder sees the same ids
    assert len(clip_ids) == 77 and clip_ids[0] == 49406 and clip_ids[1] == 49407  # BOS, EOS, then pad (= EOS)
    assert set(clip_ids[2:]) == {49407}


@needs_local
@pytest.mark.gpu
def test_null_embeddings_shapes():
    import torch

    ctx, ctx_ids, y = NT.null_text_embeddings(device=torch.device("cuda"))
    assert ctx.shape == (1, 256, 4096) and y.shape == (1, 768) and ctx_ids.shape == (1, 256, 3)
    assert ctx.dtype == torch.bfloat16 and y.dtype == torch.bfloat16
    assert torch.equal(ctx_ids[0, :, 0].long(), torch.arange(256, device=ctx.device))
    assert torch.equal(ctx_ids[0, :, 1:], torch.zeros_like(ctx_ids[0, :, 1:]))
    assert torch.isfinite(ctx.float()).all() and torch.isfinite(y.float()).all()
