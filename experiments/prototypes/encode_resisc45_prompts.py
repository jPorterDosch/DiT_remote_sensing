"""One-off T5+CLIP encoding of the 45 RESISC45 class prompts (6w-B prerequisite).

Loads the local diffusers-layout text towers (ditf_models/FLUX.1-dev: tokenizer +
text_encoder = CLIP-L pooled 768, tokenizer_2 + text_encoder_2 = T5-XXL last-hidden
512x4096) and encodes "" plus "a photo of a {cat}" for each category.

INSTRUMENT CHECK (mandatory, printed PASS/FAIL): the empty-prompt encoding must match
src/models/flux/null_embeddings.pt within bf16 tolerance — proves this pipeline is the
same one that produced the null embeddings every extraction has used. A FAIL means the
class embeddings would live in a different embedding convention than the null arm they
are differenced against, and 6w-B must not run.

Output: models/prompts/resisc45_prompt_embeds.pt
  {"prompts": [str x46], "prompt_embeds": (46,512,4096) bf16, "vec": (46,768) bf16}
  (text_ids are zeros(512,3) by convention — same as the null file.)
"""

from __future__ import annotations

import os
import sys

import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))

from data.resisc45_dataset import get_resisc45_categories  # noqa: E402

FLUX_DIR = "ditf_models"  # diffusers root: text towers at top level (FLUX.1-dev subdir holds the raw DiT/AE)
OUT = "models/prompts/resisc45_prompt_embeds.pt"


def main() -> None:
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cats = get_resisc45_categories()
    assert len(cats) == 45, len(cats)
    prompts = [""] + [f"a photo of a {c}" for c in cats]

    clip_tok = CLIPTokenizer.from_pretrained(FLUX_DIR, subfolder="tokenizer")
    clip = CLIPTextModel.from_pretrained(FLUX_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    t5_tok = T5TokenizerFast.from_pretrained(FLUX_DIR, subfolder="tokenizer_2")
    t5 = T5EncoderModel.from_pretrained(FLUX_DIR, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(device)
    clip.eval()
    t5.eval()

    with torch.no_grad():
        tb = t5_tok(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
        prompt_embeds = t5(tb.input_ids).last_hidden_state  # (46, 512, 4096) bf16
        cb = clip_tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        vec = clip(cb.input_ids).pooler_output  # (46, 768) bf16

    # INSTRUMENT CHECK, content-position criterion (2026-09-18 diagnosis): with
    # attention_mask=None (the flux HFEmbedder convention), PADDING positions attend to
    # everything and their embeddings amplify bf16/kernel noise chaotically — the stored
    # null and a fresh encoding agree at the CONTENT position (max|d| 0.016 at pos 0, the
    # lone </s>) but diverge deep in the padding region (first bad pos 477). Token ids and
    # invocation are identical, so the convention matches; bit-level padding agreement is
    # unattainable across hardware. 6w-B therefore uses THIS file's own empty-prompt row
    # as its null arm (one convention for all 46 rows, encoded in one batch); its
    # velocities are internally consistent but not bit-comparable to historical caches.
    null = torch.load(os.path.join(_root, "src/models/flux/null_embeddings.pt"), weights_only=True)
    # empty prompt = a single </s> at position 0; positions >= 1 are already padding
    d_content = (prompt_embeds[0, 0].float().cpu() - null["prompt_embeds"][0, 0].float()).abs().max().item()
    d_vec = (vec[0].float().cpu() - null["vec"][0].float()).abs().max().item()
    d_pad = (prompt_embeds[0].float().cpu() - null["prompt_embeds"][0].float()).abs().max().item()
    print(f"content pos 0 max|d|={d_content:.5f}  clip max|d|={d_vec:.5f}  full-seq max|d|={d_pad:.5f} (padding noise)")
    ok = d_content < 0.07 and d_vec < 0.07  # bf16 tolerance on O(1)-scale activations
    print("INSTRUMENT CHECK (content positions):", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit("content-position encoding does not match null_embeddings.pt — convention mismatch, do NOT run 6w-B")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(
        {"prompts": prompts, "prompt_embeds": prompt_embeds.cpu(), "vec": vec.cpu()},
        OUT,
    )
    print(f"wrote {OUT}: prompt_embeds {tuple(prompt_embeds.shape)}, vec {tuple(vec.shape)}")


if __name__ == "__main__":
    main()
