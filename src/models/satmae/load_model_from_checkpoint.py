# Adapted from repo, not final code, meant to demonstrate how to easily hook this module into main training entrypoint. Not meant to be used as is.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

# import wandb

import torch
from huggingface_hub import hf_hub_download
from models_vit_group_channels import vit_large_patch16
from timm.layers.weight_init import trunc_normal_

from utils import interpolate_pos_embed


def load_satmae_checkpoint(model, checkpoint):
    checkpoint_model = checkpoint.get("model", checkpoint)
    state_dict = model.state_dict()

    # remove incompatible keys
    for k in [
        "pos_embed",
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "head.weight",
        "head.bias",
    ]:
        if k in checkpoint_model and k in state_dict:
            if checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k}")
                del checkpoint_model[k]

    # interpolate positional embeddings
    interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # re-init head
    if hasattr(model, "head"):
        trunc_normal_(model.head.weight, std=2e-5)


if __name__ == "__main__":
    # Load the checkpoint
    model = vit_large_patch16(
        img_size=96,
        patch_size=8,
        in_chans=10,
        num_classes=62,
        global_pool=True,
    )

    ckpt_path = hf_hub_download(
        repo_id="mubashir04/checkpoint_ViT-L_finetune_fmow_sentinel",
        filename="checkpoint_ViT-L_finetune_fmow_sentinel.pth",
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    load_satmae_checkpoint(model, checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    x = torch.randn(1, 10, 96, 96, device=device)
    with torch.no_grad():
        y = model(x)
    print(y.shape)
