"""
ChestXrayDataset — PyTorch Dataset for NIH ChestX-ray14.

Key design decisions vs. the original implementation:
  - Accepts a pre-split DataFrame (produced by split.py) instead of a
    CSV path, so train/val/test transforms can differ independently.
  - Accepts a `transform` argument instead of baking in a fixed transform,
    which is required for augmentation in Step 4.
  - Builds a filename → full path lookup at init time across all image
    directories, which is necessary for the Kaggle NIH dataset (12 folders)
    and avoids slow per-item directory scans during training.
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import config


class ChestXrayDataset(Dataset):
    """
    Multi-label chest X-ray dataset for NIH ChestX-ray14.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pre-split metadata. Required columns:
            'Image Index'   — filename, e.g. '00000001_000.png'
            'Finding Labels'— pipe-separated disease names, e.g. 'Effusion|Mass'
    image_dirs : list[str]
        Directories to search for image files. All directories are indexed
        at init time; the first directory that contains a given filename wins.
    transform : callable, optional
        Torchvision transform pipeline applied to each PIL image.
        Pass a training transform (with augmentation) for the train split
        and an eval transform (without augmentation) for val/test.
    """

    def __init__(self, dataframe, image_dirs, transform=None):
        self.data      = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes   = config.CLASSES

        # ── Build filename → absolute path lookup ────────────────────────────
        # This is done once at init so __getitem__ never calls os.listdir.
        self.image_path_map: dict[str, str] = {}

        for image_dir in image_dirs:
            if not os.path.isdir(image_dir):
                continue
            for fname in os.listdir(image_dir):
                if fname not in self.image_path_map:   # first-match wins
                    self.image_path_map[fname] = os.path.join(image_dir, fname)

        print(
            f"[dataset] {len(self.data):>6} samples | "
            f"{len(self.image_path_map):>6} image files indexed"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row        = self.data.iloc[idx]
        image_name = row["Image Index"]

        image_path = self.image_path_map.get(image_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Image '{image_name}' was not found in any of the provided "
                f"image directories. Check config.IMAGE_DIRS."
            )

        # Load as RGB.
        # NIH CXRs are grayscale PNGs; .convert("RGB") triplicates the single
        # channel into three identical channels, which is what ImageNet-pretrained
        # backbones expect. This matches the CheXNet implementation.
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # ── Multi-label target vector ─────────────────────────────────────────
        # 'Finding Labels' is a pipe-separated string like 'Effusion|Mass|No Finding'.
        # 'No Finding' is not in CLASSES, so it naturally maps to an all-zero vector.
        labels_str = row["Finding Labels"]
        target     = torch.zeros(len(self.classes), dtype=torch.float32)

        for disease in labels_str.split("|"):
            disease = disease.strip()
            if disease in self.classes:
                target[self.classes.index(disease)] = 1.0

        return image, target