import torch

# Adjust this import if prepare_txt lives in a different file!
from flux.feat_flux import prepare_txt
from flux.util import load_clip, load_t5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading T5 and CLIP (bypassing FLUX model)...")
    # This only uses ~12GB of memory total!
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)

    print("Generating mathematical embeddings for an empty string...")
    prompt_embeds, text_ids, vec = prepare_txt(bs=1, t5=t5, clip=clip, prompt="")

    print("Saving tensors to disk...")
    # Move them to CPU before saving so they are standard files
    null_dict = {"prompt_embeds": prompt_embeds.cpu(), "text_ids": text_ids.cpu(), "vec": vec.cpu()}

    torch.save(null_dict, "null_embeddings.pt")
    print("Success! null_embeddings.pt is ready.")


if __name__ == "__main__":
    main()
