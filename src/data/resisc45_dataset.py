import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor
from PIL import Image

from data.utils import round_up_to_multiple


# NWPU-RESISC45: 45 scene classes, 700 images each (256x256 RGB).
RESISC45_CLASSES = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland",
]

# Readable names for text prompts
RESISC45_PROMPTS = {
    "airplane": "airplane",
    "airport": "airport",
    "baseball_diamond": "baseball diamond",
    "basketball_court": "basketball court",
    "beach": "beach",
    "bridge": "bridge",
    "chaparral": "chaparral",
    "church": "church",
    "circular_farmland": "circular farmland",
    "cloud": "cloud",
    "commercial_area": "commercial area",
    "dense_residential": "dense residential area",
    "desert": "desert",
    "forest": "forest",
    "freeway": "freeway",
    "golf_course": "golf course",
    "ground_track_field": "ground track field",
    "harbor": "harbor",
    "industrial_area": "industrial area",
    "intersection": "intersection",
    "island": "island",
    "lake": "lake",
    "meadow": "meadow",
    "medium_residential": "medium residential area",
    "mobile_home_park": "mobile home park",
    "mountain": "mountain",
    "overpass": "overpass",
    "palace": "palace",
    "parking_lot": "parking lot",
    "railway": "railway",
    "railway_station": "railway station",
    "rectangular_farmland": "rectangular farmland",
    "river": "river",
    "roundabout": "roundabout",
    "runway": "runway",
    "sea_ice": "sea ice",
    "ship": "ship",
    "snowberg": "snowberg",
    "sparse_residential": "sparse residential area",
    "stadium": "stadium",
    "storage_tank": "storage tank",
    "tennis_court": "tennis court",
    "terrace": "terrace",
    "thermal_power_station": "thermal power station",
    "wetland": "wetland",
}


# Returns images as [C, H, W] tensors normalized to [-1, 1],
# matching the input format used by Featurizer4Eval and the eval scripts.
class RESISC45Dataset(Dataset):
    def __init__(self, root, split=None, img_size=224):
        """
        Args:
            root: Path to the RESISC45 dataset root directory (containing class folders).
            split: Optional. "train" or "test" for a fixed 80/20 split per class.
                   None returns all images.
            img_size: Target image size. Images are resized to (img_size, img_size).
                      Rounded up to a multiple of 16 for compatibility with the VAE.
        """
        self.root = root
        self.img_size = round_up_to_multiple(img_size, 16)
        self.samples = []

        for class_idx, class_name in enumerate(RESISC45_CLASSES):
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


def get_resisc45_dataloader(root, split=None, img_size=224, batch_size=1, shuffle=False, num_workers=0):
    """
    Convenience function to create a RESISC45 DataLoader.

    Args:
        root: Path to RESISC45 dataset root.
        split: "train", "test", or None (all).
        img_size: Target image size (rounded to multiple of 16).
        batch_size: Batch size. Use 1 for feature extraction (matches eval script pattern).
        shuffle: Whether to shuffle.
        num_workers: Number of dataloader workers.

    Returns:
        DataLoader yielding dicts with keys: img, label, class_name, path.
    """
    dataset = RESISC45Dataset(root, split=split, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# Returns the list of class names for use with Featurizer4Eval's cat_list.
def get_resisc45_categories():
    return list(RESISC45_PROMPTS.values())


# Returns mapping from folder name to readable prompt text.
def get_resisc45_class_to_prompt():
    return RESISC45_PROMPTS
