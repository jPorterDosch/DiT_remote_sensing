"""Paper figure: COCO val2017 reference row + one row per arm, first 8 prompts, production setting (512 px, 28 steps)."""

import json
import sys
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

S = "/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing"
ref = Path(f"{S}/git10m/fid_ref_coco_val2017_n1000_512")
arms = [("COCO reference", ref)] + [
    (t, Path(f"{S}/models/prompt_gen/{t}_n1000_512px_g3.5_s28"))
    for t in ["pretrained", "proj-s42", "naive-s42", "fm-s42"]
]
for t, d in arms:
    assert (d / "00007.png").exists(), d
prompts = [p["caption"] for p in json.load(open(ref / "prompts.json"))[:8]]
n, T, L, CAP = 8, 256, 120, 64
im = Image.new("RGB", (L + n * T, CAP + len(arms) * T), "white")
dr = ImageDraw.Draw(im)
try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
except Exception:
    font = ImageFont.load_default()
for j in range(n):
    dr.multiline_text(
        (L + j * T + 3, 3), "\n".join(textwrap.wrap(prompts[j], 40)[:4]), fill="black", font=font
    )
for i, (t, d) in enumerate(arms):
    dr.text((4, CAP + i * T + T // 2 - 6), t, fill="black", font=font)
    for j in range(n):
        im.paste(
            Image.open(d / f"{j:05d}.png").convert("RGB").resize((T, T), Image.LANCZOS),
            (L + j * T, CAP + i * T),
        )
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
im.save(out)
print(out, im.size)
