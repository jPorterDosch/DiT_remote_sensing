"""Text-conditioned generation check (reviewer objection 2026-11-03): does the adapted model still
generate general text-conditioned images, or only satellite tiles?

Every training arm was trained AND scored at the null prompt, guidance 1.0, 224 px. This samples
the COCO val2017 captions of experiments/isaac/git10m/coco_ref.py with the pretrained FLUX.1-dev
or an arm's LoRA checkpoint at FLUX's own text-to-image operating point (guidance 3.5, 512 px,
28 steps, same fixed seed per prompt in every arm), saves the PNGs + manifest, and scores
  * FID / KID (eval.gen_metrics, torchvision Inception) against the PAIRED real COCO images
    (same n, same captions);
  * CLIP score: 100 x cosine(CLIP-L/14 image embedding, caption embedding), mean over prompts,
    for the samples and, as the ceiling, for the real images.
The adapter is loaded through the same registry path as every extraction / gen-eval job
(models.flux.adapter: config check against the checkpoint's training config, fenced RNG).
Prints PROMPT_GEN_OK <tag> fid=.. kid=.. clip=.. clip_real=..

--prompts FILE (instead of --ref): a fixed prompt list with NO paired real images, e.g.
experiments/isaac/git10m/rs_prompts.json (remote-sensing prompts from published prompt sets, see
rs_prompts.md). Same sampler, same per-prompt noise; FID / KID and the real-image CLIP ceiling do not
exist without a reference and are stamped null, never filled in. The CLIP score of the samples is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "models"))

CLIP_DIR = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/clip-vit-large-patch14"
ARROW = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/git10m/arrow"
# RunConfig refuses dataset.name='git10m' without an index (run.py __post_init__); the model build
# never reads it. Same file gen_eval.sbatch passes (REF_INDEX), so the stamped config matches.
INDEX = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/git10m/index_retained_cities_unique.npz"


def load_prompts(path: str, n: int | None = None) -> list[dict]:
    """The first n (default: all) entries of a prompt list: [{"caption": str, ...}, ...]. Refuses an empty
    or repeated caption (a repeat would put two rows of one prompt in the grid at different noise) and a
    list shorter than n."""
    prompts = json.load(open(path))
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"{path}: expected a non-empty JSON list of {{'caption': ...}} entries")
    for i, p in enumerate(prompts):
        if not isinstance(p, dict) or not isinstance(p.get("caption"), str) or not p["caption"].strip():
            raise SystemExit(f"{path}: entry {i} has no caption string")
    caps = [p["caption"] for p in prompts]
    dup = sorted({c for c in caps if caps.count(c) > 1})
    if dup:
        raise SystemExit(f"{path}: repeated captions {dup}")
    if n is not None:
        if not 0 < n <= len(prompts):
            raise SystemExit(f"{path}: {len(prompts)} prompts, --n {n} is out of range")
        prompts = prompts[:n]
    return prompts


def captions_sha1(captions: list[str]) -> str:
    """Identity of an ordered caption list (order matters: prompt index i fixes the noise draw)."""
    return hashlib.sha1(json.dumps(captions, ensure_ascii=False).encode()).hexdigest()


NOISE_RULE = (
    "torch.Generator(device).manual_seed(seed * 1_000_003 + prompt_index), randn (1, 16, H/8, W/8) bf16"
)


def build_model(ckpt: str, guidance: float, img_size: int, seed: int):
    """The registry's FLUX adapter with the arm's LoRA (or none), exactly as gen-eval builds it."""
    import tyro

    import run as R  # noqa: F401  (registers datasets / models / tasks)
    from registry import MODELS

    args = ["--task", "gen-eval", "--model.name", "flux", "--model.ensemble-size", "1",
            "--dataset.name", "git10m", "--dataset.path", ARROW, "--git10m.index", INDEX, "--git10m.no-hflip",
            "--k", "28", "--cd",
            "--guidance-scale", str(guidance), "--img-size", str(img_size), str(img_size), "--seed", str(seed)]  # fmt: skip
    if ckpt:
        args += ["--lora-checkpoint", ckpt]
    cfg = tyro.cli(R.RunConfig, args=args)
    model = MODELS["flux"](cfg, ["image"])
    return model._inner  # Featurizer4Eval: t5, clip, model (flux + LoRA), ae


T5_DIR = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/t5-v1_1-xxl"


def encode_prompts(prompts: list[str], device, batch: int = 16):
    """(txt (N, 512, 4096), vec (N, 768)) in bf16 on the CPU: T5-XXL (max_length 512, as FLUX.1-dev and
    save_null_embeds.py) and CLIP-L (77) loaded from the local directories, run once over every caption,
    then freed. The registry adapter keeps t5 / clip = None and serves the null prompt from
    src/models/flux/null_embeddings.pt, so prompted sampling has to bring its own encoders."""
    import torch
    from models.flux.modules.conditioner import HFEmbedder

    t0 = time.time()
    t5 = HFEmbedder(T5_DIR, max_length=512, torch_dtype=torch.bfloat16).to(device)
    clip = HFEmbedder(CLIP_DIR, max_length=77, torch_dtype=torch.bfloat16).to(device)
    print(f"text encoders loaded in {time.time() - t0:.0f} s", flush=True)
    txt, vec = [], []
    with torch.no_grad():
        for b0 in range(0, len(prompts), batch):
            ps = prompts[b0 : b0 + batch]
            txt.append(t5(ps).cpu())
            vec.append(clip(ps).cpu())
    del t5, clip
    torch.cuda.empty_cache()
    txt, vec = torch.cat(txt), torch.cat(vec)
    assert txt.shape == (len(prompts), 512, 4096) and vec.shape == (len(prompts), 768), (txt.shape, vec.shape)
    return txt, vec


def sample(feat, txt_all, vec_all, img_size: int, steps: int, guidance: float, batch: int, seed: int, device):
    """uint8 NHWC samples, one fixed noise draw per prompt index (seed + idx), plus the latent sha1s.
    txt_all / vec_all: encode_prompts output, one row per prompt."""
    import torch
    from models.flux.feat_flux import prepare
    from models.flux.sampling import denoise, get_schedule, unpack

    flux, ae = feat.model, feat.ae
    flux.eval()
    H = W = img_size
    out, shas = [], []
    t0 = time.time()
    n_total = len(txt_all)
    # No autograd: the LoRA params require grad, so without this the 4-step smoke kept every
    # block's activations and OOM'd a 44 GB A40 at batch 4 / 256 px (job 6540420).
    with torch.no_grad():
        for b0 in range(0, n_total, batch):
            n = min(batch, n_total - b0)
            zs = []
            for i in range(n):
                g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + b0 + i)
                zs.append(
                    torch.randn((1, 16, H // 8, W // 8), generator=g, device=device, dtype=torch.bfloat16)
                )
            z = torch.cat(zs)
            img, img_ids = prepare(z)
            img, img_ids = img.to(device), img_ids.to(device)
            txt = txt_all[b0 : b0 + n].to(device)
            y = vec_all[b0 : b0 + n].to(device)
            txt_ids = torch.zeros(n, txt.shape[1], 3, device=device)
            ts = get_schedule(steps, img.shape[1], shift=True)
            x = denoise(
                flux,
                img,
                img_ids,
                txt.to(img.dtype),
                txt_ids,
                y.to(img.dtype),
                timesteps=ts,
                guidance=guidance,
            )
            shas.append(hashlib.sha1(x.float().cpu().numpy().tobytes()).hexdigest()[:12])
            x = unpack(x.float(), H, W)
            pix = ae.decode(x.to(next(ae.parameters()).dtype)).float().clamp(-1, 1)
            out.append(((pix + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
            done = b0 + n
            print(f"  {done}/{n_total} sampled, {(time.time() - t0) / done:.1f} s/image", flush=True)
    return np.concatenate(out), shas


def clip_score(images_uint8: np.ndarray, captions: list[str], device, batch: int = 64) -> dict:
    """100 x cosine(CLIP-L/14 image, caption) per pair; mean, std and the per-pair list."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float32).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    scores = []
    with torch.no_grad():
        for b0 in range(0, len(captions), batch):
            ims = [Image.fromarray(a) for a in images_uint8[b0 : b0 + batch]]
            inp = proc(
                text=captions[b0 : b0 + batch],
                images=ims,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            # transformers 5.x: get_*_features return BaseModelOutputWithPooling whose pooler_output is the
            # projected embedding (modeling_clip.py get_image_features / get_text_features).
            ie = model.get_image_features(pixel_values=inp["pixel_values"]).pooler_output
            te = model.get_text_features(
                input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
            ).pooler_output
            assert ie.shape == te.shape == (len(ims), model.config.projection_dim), (ie.shape, te.shape)
            ie = torch.nn.functional.normalize(ie, dim=-1)
            te = torch.nn.functional.normalize(te, dim=-1)
            scores.append((100 * (ie * te).sum(-1)).cpu().numpy())
    s = np.concatenate(scores)
    return {"mean": float(s.mean()), "std": float(s.std(ddof=1)), "n": int(len(s)), "per_pair": s.round(3).tolist(),
            "definition": "100 x cosine(CLIP ViT-L/14 image embedding, caption embedding), openai/clip-vit-large-patch14"}  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="arm name for the manifest / OK line")
    ap.add_argument("--ckpt", default="", help="LoRA checkpoint; empty = pretrained FLUX.1-dev")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ref",
        help="coco_ref.py output dir (prompts.json + paired PNGs): FID / KID / CLIP vs the real images",
    )
    src.add_argument(
        "--prompts",
        help="prompt list JSON with no paired images (rs_prompts.json): samples + CLIP score only",
    )
    ap.add_argument(
        "--n", type=int, default=None, help="first n prompts; default 1000 with --ref, all with --prompts"
    )
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kid-subsets", type=int, default=100)
    a = ap.parse_args(argv)
    import torch

    from eval import gen_metrics as GM

    if os.path.exists(os.path.join(a.out, "manifest.json")):
        raise SystemExit(f"{a.out} already holds a finished run; never overwritten")
    if a.ckpt and not os.path.isfile(a.ckpt):
        raise SystemExit(f"--ckpt {a.ckpt} does not exist")
    real = ref_man = None
    if a.ref:
        a.n = 1000 if a.n is None else a.n
        prompts_all = json.load(open(os.path.join(a.ref, "prompts.json")))
        if len(prompts_all) < a.n:
            raise SystemExit(f"{a.ref}: {len(prompts_all)} prompts < --n {a.n}")
        prompts = prompts_all[: a.n]
        ref_man = json.load(open(os.path.join(a.ref, "manifest.json")))
        if ref_man["size"] != a.img_size:
            raise SystemExit(
                f"reference is {ref_man['size']} px, samples would be {a.img_size} px: build a matching reference"
            )
        real = GM._load_pngs(a.ref, a.n)
        if real.shape[0] != a.n:
            raise SystemExit(f"{a.ref}: {real.shape[0]} PNGs, need {a.n}")
    else:
        prompts = load_prompts(a.prompts, a.n)
        a.n = len(prompts)
    captions = [p["caption"] for p in prompts]
    print(
        f"{a.n} prompts, caption-list sha1 {captions_sha1(captions)[:12]}, source {a.ref or a.prompts}",
        flush=True,
    )
    device = torch.device("cuda")
    t0 = time.time()
    txt_all, vec_all = encode_prompts(captions, device)  # before FLUX loads: T5-XXL freed first
    feat = build_model(a.ckpt, a.guidance, a.img_size, a.seed)
    print(f"model built in {time.time() - t0:.0f} s ({'pretrained' if not a.ckpt else a.ckpt})", flush=True)
    fake, shas = sample(feat, txt_all, vec_all, a.img_size, a.steps, a.guidance, a.batch, a.seed, device)
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image

    for i in range(fake.shape[0]):
        Image.fromarray(fake[i]).save(os.path.join(a.out, f"{i:05d}.png"))
    Image.fromarray(np.concatenate(list(fake[:8]), axis=1)).save(os.path.join(a.out, "first8_grid.png"))
    # free the generator before the scorers
    del feat
    torch.cuda.empty_cache()
    # No paired real images (--prompts): there is no FID / KID and no real-image CLIP ceiling. Stamped
    # None, printed NA; nothing is substituted.
    metrics = (
        GM.compare(real, fake, GM.InceptionFeatures(device), subsets=a.kid_subsets)
        if real is not None
        else None
    )
    cs_fake = clip_score(fake, captions, device)
    cs_real = clip_score(real, captions, device) if real is not None else None
    man = {"tag": a.tag, "ckpt": a.ckpt, "ckpt_sha1": hashlib.sha1(open(a.ckpt, "rb").read()).hexdigest() if a.ckpt else None,
           "ref": a.ref, "ref_identity": ref_man["identity"] if ref_man else None,
           "prompts_file": os.path.abspath(a.prompts) if a.prompts else None, "captions_sha1": captions_sha1(captions),
           "n": a.n, "img_size": a.img_size, "steps": a.steps,
           "guidance": a.guidance, "batch": a.batch, "seed": a.seed, "noise": NOISE_RULE, "latent_sha1_per_batch": shas,
           "prompts": prompts, "fake_identity": GM.image_identity(fake), "gen_metrics": metrics,
           "clip_score": {k: v for k, v in cs_fake.items() if k != "per_pair"},
           "clip_score_real": {k: v for k, v in cs_real.items() if k != "per_pair"} if cs_real else None,
           "wall_s": time.time() - t0}  # fmt: skip
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    json.dump(
        {"fake": cs_fake["per_pair"], "real": cs_real["per_pair"] if cs_real else None},
        open(os.path.join(a.out, "clip_per_pair.json"), "w"),
    )
    scored = (
        f"fid={metrics['fid']:.2f} kid={metrics['kid_mean']:.4f}+-{metrics['kid_std']:.4f}"
        if metrics
        else "fid=NA kid=NA"
    )
    clip_real = f"{cs_real['mean']:.2f}" if cs_real else "NA"
    print(f"PROMPT_GEN_OK {a.tag} {scored} clip={cs_fake['mean']:.2f}+-{cs_fake['std']:.2f} "
          f"clip_real={clip_real} n={a.n} {a.img_size}px g={a.guidance} steps={a.steps}", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
