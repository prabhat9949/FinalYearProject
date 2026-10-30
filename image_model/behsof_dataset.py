from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import hashlib


class BEHSOFDataset(Dataset):
    """
    BEHSOF ultrasound dataset.

    Class:
        0 -> Non-NAFLD
        1 -> NAFLD
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_samples(dataset_root, remove_duplicates=True):
    """
    Load BEHSOF image paths and labels.

    Duplicate images are detected using MD5 hashes.
    The original dataset files are NOT deleted.
    """

    dataset_root = Path(dataset_root)

    nafld_dir = dataset_root / "NAFLD"
    non_nafld_dir = dataset_root / "Non-NAFLD"

    samples = []

    # Non-NAFLD = class 0
    for image_path in sorted(non_nafld_dir.glob("*.jpg")):
        samples.append((str(image_path), 0))

    # NAFLD = class 1
    for image_path in sorted(nafld_dir.glob("*.jpg")):
        samples.append((str(image_path), 1))

    if not remove_duplicates:
        return samples

    unique_samples = []
    seen_hashes = set()

    for image_path, label in samples:

        with open(image_path, "rb") as f:
            image_hash = hashlib.md5(f.read()).hexdigest()

        if image_hash not in seen_hashes:
            seen_hashes.add(image_hash)
            unique_samples.append((image_path, label))

    print(f"Original images : {len(samples)}")
    print(f"Unique images   : {len(unique_samples)}")
    print(f"Duplicates removed: {len(samples) - len(unique_samples)}")

    return unique_samples