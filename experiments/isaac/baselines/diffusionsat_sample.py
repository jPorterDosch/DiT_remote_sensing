"""Sample DiffusionSat (Khanna et al., ICLR 2024) for the two-way FID table (OUTLINE §4.8).

Runs in the DiffusionSat env ($STORE/conda_envs/diffusionsat: its diffusers 0.17 fork, torch 2.2.2),
NOT in DiTF. Public 512 px checkpoint `finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64`,
checkpoint-150000 (Zenodo 13751498). The fine-tuned UNet REQUIRES the 7-value metadata vector
(lon, lat, gsd, cloud, year, month, day); its CFG null branch is prompt "" with metadata all-zero
(pipeline.prepare_metadata).

Metadata encoding = the TRAINING convention (data_util.fmow_numerical_metadata, then
metadata_normalize): lon + 180, lat + 90, cloud as a fraction, year - 1980. Until 2026-11-11 this
script passed raw lon / lat / calendar year to metadata_normalize, which applies no offsets; every
"geo" sample set made before that date (dsat_samples/geo_n5000) is conditioned on out-of-range
metadata and is SUPERSEDED. An executed gate (dsat_cond.known_answer_gate) now reproduces the one
normalized vector the authors ship and proves it rejects the old encoding, before any sampling.

Conditioning modes:
  null  matched to our protocol: prompt "", metadata zeros = the model's own null conditioning.
  fmow  DiffusionSat's native use on its own training domain. --rows is the fMoW-val reference
        rows.csv (i,label,path); every image gets ITS OWN caption and metadata, built by the
        authors' functions (fmow_tokenize_caption with nothing dropped: "a fmow satellite image of
        a <class> in <country>"; fmow_numerical_metadata) from the image's fMoW JSON (--fmow-cond,
        made by fmow_val_meta.py) and the authors' polygon csv (--fmow-meta). gsd = the JSON gsd
        (stored size taken as 512; the authors' shard sizes are not documented), see manifest.
  geo   Git-10M held-out rows: metadata from the row's tile centre (--coords) and the tile's
        ground extent (--gsd tile), date and cloud fixed by flags (Git-10M has neither), prompt
        --prompt. Rows without coordinates (non-Google sources) get the null metadata vector and
        are listed; samples224_located/ + ref224_located/ link the located subset for a matched
        score.

Samples: n fixed-seed images (torch.Generator seeded --seed + image index), 512 px, saved as
PNG under <out>/samples512/, then PIL-bicubic resized to --img-size (our reference decode) under
<out>/samples224/ for eval.gen_metrics. <out>/conditioning.csv holds every row's prompt and
normalized metadata; <out>/manifest.json records settings, counts and the uint8 identity of both
sets (eval.gen_metrics.image_identity). Refuses to overwrite. Prints DSAT_SAMPLE_OK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import dsat_cond as DC  # noqa: E402
from eval import gen_metrics as GM  # noqa: E402  (image_identity only; no torchvision model is built)

ENCODING = "training convention: [lon+180, lat+90, gsd, cloud fraction, year-1980, month, day] -> diffusionsat.metadata_normalize"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", required=True, help="dir holding the pipeline (model_index.json) and checkpoint-<step>/unet"
    )
    ap.add_argument("--step", type=int, default=150000)
    ap.add_argument("--mode", choices=["null", "geo", "fmow"], required=True)
    ap.add_argument(
        "--rows",
        required=True,
        help="reference rows.csv: geo-heldout (shard,row,img_name,datasets) or fMoW-val (i,label,path)",
    )
    ap.add_argument("--coords", default=None, help="csv shard,row,img_name,lon,lat (geo mode)")
    ap.add_argument("--fmow-cond", default=None, help="cond.csv from fmow_val_meta.py (fmow mode)")
    ap.add_argument("--fmow-meta", default=None, help="DiffusionSat datasets/fmow-val-meta.csv (fmow mode)")
    ap.add_argument("--prompt", default="a satellite image", help="geo mode prompt")
    ap.add_argument(
        "--guidance", type=float, default=None, help="default: 1.0 (null) / 7.5 (geo); required for fmow"
    )
    ap.add_argument(
        "--steps", type=int, default=20, help="20 = the notebook's setting; the paper's FID uses 100"
    )
    ap.add_argument(
        "--gsd",
        default="tile",
        help="geo mode: 'tile' (ground extent of the row's tile / 512) or a constant in m",
    )
    ap.add_argument("--cloud", type=float, default=0.0, help="geo mode cloud cover, fraction")
    ap.add_argument("--date", default="2020-06-15", help="geo mode acquisition date (Git-10M has none)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    if a.guidance is None and a.mode == "fmow":
        raise SystemExit("fmow mode needs an explicit --guidance")
    guidance = a.guidance if a.guidance is not None else (1.0 if a.mode == "null" else 7.5)

    from diffusionsat import (
        DiffusionSatPipeline,
        SatUNet,
        fmow_numerical_metadata,  # noqa: E402
        fmow_tokenize_caption,
        metadata_normalize,
    )

    with open(a.rows) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < a.n:
        raise SystemExit(f"{a.rows}: {len(rows)} rows < n={a.n}")
    rows = rows[: a.n]
    md = np.zeros((a.n, 7), dtype=np.float32)
    prompts = [""] * a.n
    located = [False] * a.n
    info: dict = {
        "normalize": "diffusionsat.metadata_normalize",
        "null_mode": "all zeros",
        "encoding": ENCODING,
    }
    gate = None
    if a.mode == "geo":
        if not a.coords:
            raise SystemExit("geo mode needs --coords")
        gate = DC.known_answer_gate(lambda m: metadata_normalize(m).tolist())
        year, month, day = (int(x) for x in a.date.split("-"))
        coords = {}
        with open(a.coords) as f:
            for r in csv.DictReader(f):
                coords[(int(r["shard"]), int(r["row"]))] = (r["img_name"], float(r["lon"]), float(r["lat"]))
        gsds = []
        for i, r in enumerate(rows):
            prompts[i] = a.prompt
            c = coords.get((int(r["shard"]), int(r["row"])))
            if c is None:
                continue  # md[i] stays all-zero: the model's "no metadata" vector
            name, lon, lat = c
            if name != r["img_name"]:
                raise SystemExit(f"row {i}: coords img_name {name!r} != rows img_name {r['img_name']!r}")
            gsd = DC.tile_gsd(name, lon, lat) if a.gsd == "tile" else float(a.gsd)
            md[i] = metadata_normalize(DC.raw_metadata(lon, lat, gsd, a.cloud, year, month, day)).numpy()
            located[i] = True
            gsds.append(gsd)
        if not gsds:
            raise SystemExit("geo mode: no row has coordinates")
        info.update(
            {
                "gsd": a.gsd,
                "gsd_min_median_max": [float(np.min(gsds)), float(np.median(gsds)), float(np.max(gsds))],
                "cloud": a.cloud,
                "date": a.date,
                "n_with_coords": int(sum(located)),
                "n_without_coords": int(a.n - sum(located)),
                "rows_without_coords": "null metadata vector (all zeros), same prompt",
            }
        )
        print(
            f"geo mode: {sum(located)}/{a.n} rows with coordinates; gsd min/median/max "
            f"{np.min(gsds):.2f}/{np.median(gsds):.2f}/{np.max(gsds):.2f} m",
            flush=True,
        )
    elif a.mode == "fmow":
        if not (a.fmow_cond and a.fmow_meta):
            raise SystemExit("fmow mode needs --fmow-cond and --fmow-meta")
        import pandas as pd

        meta_df = pd.read_csv(a.fmow_meta)  # as train_text_to_image.py reads it
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)  # stored size == target: gsd scale 1

        def example(cat, fname, gsd, cloud_pct, timestamp, country):
            return {
                "category.txt": cat,
                "input.npy": dummy,
                "metadata.json": {
                    "img_filename": fname,
                    "gsd": gsd,
                    "cloud_cover": cloud_pct,
                    "timestamp": timestamp,
                    "country_code": country,
                },
            }

        k = DC.KNOWN
        gate = DC.known_answer_gate(
            lambda m: metadata_normalize(m).tolist(),
            encode=lambda: fmow_numerical_metadata(
                example(
                    k["category"],
                    k["img_filename"],
                    k["json_gsd"],
                    k["cloud_cover_pct"],
                    k["timestamp"],
                    k["country_code"],
                ),
                meta_df,
                512,
                7,
            ).tolist(),
        )
        with open(a.fmow_cond) as f:
            cond = {int(r["i"]): r for r in csv.DictReader(f)}
        stamps = {
            (c, int(loc), int(g)): t
            for c, loc, g, t in zip(
                meta_df["category"], meta_df["location_id"], meta_df["image_id"], meta_df["timestamp"]
            )
        }
        gsds = []
        for i, r in enumerate(rows):
            c = cond.get(int(r["i"]))
            if c is None or c["path"] != r["path"]:
                raise SystemExit(f"row {i}: {a.fmow_cond} has no matching record for {r['path']}")
            cat, loc, img, fname = DC.parse_fmow_path(r["path"])
            if stamps.get((cat, loc, img)) != c["timestamp"]:
                raise SystemExit(
                    f"row {i} {fname}: JSON timestamp {c['timestamp']} != authors' csv {stamps.get((cat, loc, img))}"
                )
            ex = example(
                cat, fname, float(c["gsd"]), float(c["cloud_cover"]), c["timestamp"], c["country_code"]
            )
            md[i] = metadata_normalize(fmow_numerical_metadata(ex, meta_df, 512, 7)).numpy()
            prompts[i] = fmow_tokenize_caption(ex, None, drop_pct=-1.0, return_text=True)  # nothing dropped
            located[i] = True
            gsds.append(float(c["gsd"]))
        info.update(
            {
                "source": "per-image fMoW-rgb JSON (gsd, cloud_cover, country_code, timestamp) + authors' polygon csv (lon, lat)",
                "fmow_cond": a.fmow_cond,
                "fmow_meta": a.fmow_meta,
                "functions": "diffusionsat.fmow_numerical_metadata, diffusionsat.fmow_tokenize_caption(drop_pct=-1)",
                "gsd": "JSON gsd x 1 (stored min side taken as 512 = target; authors' shard sizes undocumented; "
                "their notebook example has 0.4114541 where the JSON says 0.5147183)",
                "gsd_min_median_max": [float(np.min(gsds)), float(np.median(gsds)), float(np.max(gsds))],
                "n_prompts_distinct": len(set(prompts)),
            }
        )
        print(
            f"fmow mode: {a.n} rows, {len(set(prompts))} distinct captions, e.g. {prompts[0]!r}; "
            f"md[0] = {[round(float(v), 2) for v in md[0]]}",
            flush=True,
        )
    if gate:
        print(gate, flush=True)
    if not np.isfinite(md).all() or md.min() < 0:
        raise SystemExit(f"normalized metadata out of range: min {md.min()}, finite {np.isfinite(md).all()}")
    if a.mode != "null" and md[:, [0, 1, 4, 5, 6]].max() > 1000.0 + 1e-3:
        raise SystemExit(
            f"normalized lon / lat / date above the trained range 0-1000: max {md[:, [0, 1, 4, 5, 6]].max()}"
        )

    dev = "cuda"
    unet = SatUNet.from_pretrained(
        os.path.join(a.ckpt, f"checkpoint-{a.step}"), subfolder="unet", torch_dtype=torch.float16
    )
    pipe = DiffusionSatPipeline.from_pretrained(
        a.ckpt, unet=unet, safety_checker=None, torch_dtype=torch.float16
    ).to(dev)
    pipe.set_progress_bar_config(disable=True)
    if not getattr(pipe.unet, "metadata_embedding", None):
        raise SystemExit("loaded UNet has no metadata embedding; wrong checkpoint")
    scheduler = type(pipe.scheduler).__name__

    # Metadata stays float32 up to the sinusoidal embedding. The pipeline casts it to the text
    # embedding dtype (float16 here); float16 tops out at 65,504, so a normalized gsd >= 65.5 m
    # x 1000 becomes inf, its embedding NaN and the sample flat. SatUNet.time_proj computes in
    # float32 and casts the EMBEDDING (values in [-1, 1]) to the weights' dtype, which is what
    # training does. Every run from 2026-11-12 on uses this path; manifest key "metadata_dtype".
    pipeline_prepare = pipe.prepare_metadata

    def prepare_metadata_fp32(batch_size, metadata, do_cfg, device, dtype):
        return pipeline_prepare(batch_size, metadata, do_cfg, device, torch.float32)

    pipe.prepare_metadata = prepare_metadata_fp32
    worst = md[int(np.argmax(md.max(axis=1)))].tolist()
    f16_lost = DC.cast_overflow_count(md, "float16")
    with torch.no_grad():
        got = pipe.prepare_metadata(1, worst, False, dev, torch.float16)
        emb = pipe.unet.time_proj(got.view(-1))
        old = pipeline_prepare(1, worst, False, dev, torch.float16)
        emb_old = pipe.unet.time_proj(old.view(-1))
    if got.dtype != torch.float32 or not torch.isfinite(got).all() or not torch.isfinite(emb).all():
        raise SystemExit(
            f"GATE metadata dtype FAIL: largest metadata row {worst} -> dtype {got.dtype}, "
            f"finite values {bool(torch.isfinite(got).all())}, finite embedding {bool(torch.isfinite(emb).all())}"
        )
    old_finite = bool(torch.isfinite(emb_old).all())
    if old_finite != (DC.cast_overflow_count(worst, "float16") == 0):
        raise SystemExit(
            f"GATE metadata dtype FAIL: the float16 control disagrees with the overflow count "
            f"(embedding finite {old_finite}, row {worst}); the gate cannot see the overflow it exists for"
        )
    md_gate = (
        f"GATE metadata dtype PASS: float32 to the embedding, largest row max {max(worst):.1f} finite; "
        f"{f16_lost} of {md.size} values overflow float16 (pipeline default cast: embedding finite = {old_finite})"
    )
    print(md_gate, flush=True)

    os.makedirs(os.path.join(a.out, "samples512"))
    os.makedirs(os.path.join(a.out, "samples224"))
    with open(os.path.join(a.out, "conditioning.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "i",
                "has_metadata",
                "prompt",
                "lon_n",
                "lat_n",
                "gsd_n",
                "cloud_n",
                "year_n",
                "month_n",
                "day_n",
            ]
        )
        for i in range(a.n):
            w.writerow([i, int(located[i]), prompts[i]] + [repr(float(v)) for v in md[i]])
    big = np.empty((a.n, 512, 512, 3), dtype=np.uint8)
    small = np.empty((a.n, a.img_size, a.img_size, 3), dtype=np.uint8)
    t0 = time.time()
    n_flat = 0
    for s in range(0, a.n, a.batch):
        idx = list(range(s, min(s + a.batch, a.n)))
        gens = [torch.Generator(device=dev).manual_seed(a.seed + i) for i in idx]
        out = pipe(
            [prompts[i] for i in idx],
            metadata=md[idx].tolist(),
            num_inference_steps=a.steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=gens,
            output_type="np",
        )
        imgs = (np.asarray(out.images) * 255.0).round().clip(0, 255).astype(np.uint8)
        if imgs.shape[1:] != (512, 512, 3):
            raise SystemExit(f"unexpected sample shape {imgs.shape}")
        flat = DC.flat_count(imgs)
        n_flat += flat
        if s == 0 and flat == len(idx):
            raise SystemExit(
                f"GATE sample sanity FAIL: all {flat} images of the first batch are flat "
                f"(pixel std < {DC.FLAT_STD}); NaN latents or a filter. Nothing usable was written."
            )
        if n_flat > 0.01 * a.n:  # same limit as the final gate; stop instead of sampling the rest
            raise SystemExit(
                f"GATE sample sanity FAIL: {n_flat} flat images after {idx[-1] + 1}/{a.n} "
                f"(pixel std < {DC.FLAT_STD}; limit 1% of n); do NOT score {a.out}"
            )
        for j, i in enumerate(idx):
            big[i] = imgs[j]
            pil = Image.fromarray(imgs[j])
            pil.save(os.path.join(a.out, "samples512", f"{i:05d}.png"))
            sm = pil.resize((a.img_size, a.img_size), Image.Resampling.BICUBIC)
            small[i] = np.asarray(sm)
            sm.save(os.path.join(a.out, "samples224", f"{i:05d}.png"))
        if s % (a.batch * 25) == 0:
            print(f"  {idx[-1] + 1}/{a.n} {time.time() - t0:.0f}s", flush=True)
    if a.mode == "geo":  # matched located subset: sample i against real image i, located rows only
        ref = os.path.dirname(os.path.abspath(a.rows))
        os.makedirs(os.path.join(a.out, "samples224_located"))
        os.makedirs(os.path.join(a.out, "ref224_located"))
        for i in range(a.n):
            if not located[i]:
                continue
            real = os.path.join(ref, f"{i:05d}.png")
            if not os.path.exists(real):
                raise SystemExit(f"reference image missing: {real}")
            os.symlink(
                os.path.join("..", "samples224", f"{i:05d}.png"),
                os.path.join(a.out, "samples224_located", f"{i:05d}.png"),
            )
            os.symlink(real, os.path.join(a.out, "ref224_located", f"{i:05d}.png"))
    if n_flat > 0.01 * a.n:
        raise SystemExit(
            f"GATE sample sanity FAIL: {n_flat}/{a.n} flat images (pixel std < {DC.FLAT_STD}); do NOT score {a.out}"
        )
    print(
        f"GATE sample sanity PASS: {n_flat}/{a.n} flat images (pixel std < {DC.FLAT_STD}; limit 1%)",
        flush=True,
    )
    man = {
        "model": "DiffusionSat finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64",
        "ckpt": a.ckpt,
        "step": a.step,
        "mode": a.mode,
        "prompt": {"null": "", "geo": a.prompt, "fmow": "per image, see conditioning.csv"}[a.mode],
        "guidance": guidance,
        "steps": a.steps,
        "scheduler": scheduler,
        "n": a.n,
        "seed": a.seed,
        "seed_rule": "torch.Generator(cuda).manual_seed(seed + image index)",
        "batch": a.batch,
        "metadata": info,
        "encoding_gate": gate,
        "n_flat": n_flat,
        "metadata_dtype": "float32 to the sinusoidal embedding (pipeline default would cast to the fp16 weights' dtype)",
        "metadata_dtype_gate": md_gate,
        "metadata_float16_overflow_values": f16_lost,
        "rows": a.rows,
        "coords": a.coords,
        "img_size": a.img_size,
        "resize": "PIL bicubic 512 -> img_size (same as the reference decode)",
        "identity512": GM.image_identity(big),
        "identity": GM.image_identity(small),
        "safety_checker": "disabled (safety_checker=None; satellite tiles trip the SD NSFW filter and would come back black)",
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(
        f"DSAT_SAMPLE_OK mode={a.mode} n={a.n} guidance={guidance} steps={a.steps} scheduler={scheduler} "
        f"({time.time() - t0:.0f}s) -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
