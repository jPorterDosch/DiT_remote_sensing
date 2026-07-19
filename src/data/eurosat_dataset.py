import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor
from PIL import Image

from data.utils import round_up_to_multiple


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Readable names for text prompts
EUROSAT_PROMPTS = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway",
    "Industrial": "industrial buildings",
    "Pasture": "pasture",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings",
    "River": "river",
    "SeaLake": "sea or lake",
}


# Returns images as [C, H, W] tensors normalized to [-1, 1],
# matching the input format used by Featurizer4Eval and the eval scripts.
class EuroSATDataset(Dataset):
    def __init__(self, root, split=None, img_size=224):
        """
        Args:
            root: Path to the EuroSAT dataset root directory (containing class folders).
            split: Optional. "train" or "test" for a fixed 80/20 split per class.
                   None returns all images.
            img_size: Target image size. Images are resized to (img_size, img_size).
                      Rounded up to a multiple of 16 for compatibility with the VAE.
        """
        self.root = root
        self.img_size = round_up_to_multiple(img_size, 16)
        self.samples = []

        for class_idx, class_name in enumerate(EUROSAT_CLASSES):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            filenames = sorted(
                [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]
            )

            if split is not None:
                pivot = int(len(filenames) * 0.8)
                if split == "train":
                    filenames = filenames[:pivot]
                elif split == "test":
                    filenames = filenames[pivot:]

            for fname in filenames:
                self.samples.append(
                    (
                        os.path.join(class_dir, fname),
                        class_idx,
                        class_name,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)

        # Normalize to [-1, 1]
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2

        return {
            "img": img_tensor,
            "label": class_idx,
            "class_name": class_name,
            "path": img_path,
        }


def get_eurosat_dataloader(root, split=None, img_size=224, batch_size=1, shuffle=False, num_workers=0):
    """
    Convenience function to create a EuroSAT DataLoader.

    Args:
        root: Path to EuroSAT dataset root.
        split: "train", "test", or None (all).
        img_size: Target image size (rounded to multiple of 16).
        batch_size: Batch size. Use 1 for feature extraction (matches eval script pattern).
        shuffle: Whether to shuffle.
        num_workers: Number of dataloader workers.

    Returns:
        DataLoader yielding dicts with keys: img, label, class_name, path.
    """
    dataset = EuroSATDataset(root, split=split, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# Returns the list of class names for use with Featurizer4Eval's cat_list.
def get_eurosat_categories():
    return list(EUROSAT_PROMPTS.values())


# Returns mapping from folder name to readable prompt text.
def get_eurosat_class_to_prompt():
    return EUROSAT_PROMPTS
