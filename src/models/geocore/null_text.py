"""Empty-prompt conditioning for GeoCore-9B from the local T5-XXL and CLIP-L directories.

Reproduces GeoCore's `models/text_encoder.py::TextEncoder.__call__` for the prompt `""`
(the only prompt the frozen-feature protocol uses, spec §2): CLIP tokenised to 77 tokens,
T5 to 256, `padding="max_length"`, `truncation=True`, encoders fed `input_ids` only, the T5
sequence output `[0]` as `ctx` and CLIP's `pooler_output` as `y`. Position ids follow their
layout: `ctx_ids[:, :, 0] = arange(256)`, the other two columns zero.

Upstream loads the encoders by Hugging Face id; compute nodes have no network, so this
module takes the local snapshot directories instead (VENDORED.md).
"""

from __future__ import annotations

import gc

import torch

T5_DIR_DEFAULT = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/t5-v1_1-xxl"
CLIP_DIR_DEFAULT = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/clip-vit-large-patch14"
T5_MAX_LENGTH = 256
CLIP_MAX_LENGTH = 77
NULL_PROMPT = ""


def _tokenisers(t5_dir: str, clip_dir: str):
    from transformers import CLIPTokenizer, T5TokenizerFast

    return T5TokenizerFast.from_pretrained(t5_dir), CLIPTokenizer.from_pretrained(clip_dir)


def null_token_ids(
    t5_dir: str = T5_DIR_DEFAULT, clip_dir: str = CLIP_DIR_DEFAULT
) -> tuple[list[int], list[int]]:
    """Token ids the encoders see for the empty prompt (tokenisers only, no weights)."""
    tok_t5, tok_clip = _tokenisers(t5_dir, clip_dir)
    t5 = tok_t5(
        [NULL_PROMPT], padding="max_length", max_length=T5_MAX_LENGTH, truncation=True, return_tensors="pt"
    )
    clip = tok_clip(
        [NULL_PROMPT], padding="max_length", max_length=CLIP_MAX_LENGTH, truncation=True, return_tensors="pt"
    )
    return t5.input_ids[0].tolist(), clip.input_ids[0].tolist()


@torch.no_grad()
def null_text_embeddings(
    t5_dir: str = T5_DIR_DEFAULT,
    clip_dir: str = CLIP_DIR_DEFAULT,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (ctx (1, 256, 4096), ctx_ids (1, 256, 3), y (1, 768)) on `device` in `dtype`.
    Loads both encoders, runs the empty prompt once, then frees them."""
    from transformers import CLIPTextModel, T5EncoderModel

    tok_t5, tok_clip = _tokenisers(t5_dir, clip_dir)
    device = torch.device(device)

    enc_clip = CLIPTextModel.from_pretrained(clip_dir, torch_dtype=dtype).to(device).eval()
    clip_in = tok_clip(
        [NULL_PROMPT], padding="max_length", max_length=CLIP_MAX_LENGTH, truncation=True, return_tensors="pt"
    )
    y = enc_clip(clip_in.input_ids.to(device)).pooler_output.to(dtype)
    del enc_clip

    enc_t5 = T5EncoderModel.from_pretrained(t5_dir, torch_dtype=dtype).to(device).eval()
    t5_in = tok_t5(
        [NULL_PROMPT], padding="max_length", max_length=T5_MAX_LENGTH, truncation=True, return_tensors="pt"
    )
    ctx = enc_t5(t5_in.input_ids.to(device))[0].to(dtype)
    del enc_t5

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if ctx.shape != (1, T5_MAX_LENGTH, 4096) or y.shape != (1, 768):
        raise RuntimeError(f"unexpected encoder output shapes ctx={tuple(ctx.shape)} y={tuple(y.shape)}")
    ctx_ids = torch.zeros((1, T5_MAX_LENGTH, 3), device=device, dtype=dtype)
    ctx_ids[:, :, 0] = torch.arange(T5_MAX_LENGTH, device=device)
    return ctx, ctx_ids, y
