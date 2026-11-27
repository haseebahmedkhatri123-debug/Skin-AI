"""Dataset class and image transforms (torchvision v2)."""
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def load_rgb(image) -> Image.Image:
    """Open an image (path or PIL), apply EXIF rotation, and convert to RGB.

    Phone photos often store orientation in EXIF instead of rotating the
    pixels, and PNG uploads may be RGBA/palette images; both are normalized.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        if image.mode in ("RGBA", "LA", "P"):
            # Composite transparency onto white instead of black.
            image = image.convert("RGBA")
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.getchannel("A"))
            image = bg
        else:
            image = image.convert("RGB")
    return image


def train_transforms(img_size: int = config.IMG_SIZE) -> v2.Compose:
    """Heavy augmentation that mimics phone-camera photos of skin lesions."""
    return v2.Compose([
        v2.ToImage(),                                    # PIL -> uint8 tensor
        v2.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(30),
        v2.RandomPerspective(distortion_scale=0.2, p=0.3),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomApply([v2.JPEG(quality=(50, 95))], p=0.3),   # compression artifacts
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),  # focus blur
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transforms(img_size: int = config.IMG_SIZE) -> v2.Compose:
    """Deterministic preprocessing used for validation, test, and the app."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(int(round(img_size / 0.875)), antialias=True),  # e.g. 224 -> 256 short side
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class SkinDataset(Dataset):
    """Reads a split CSV with columns (image_path, label)."""

    def __init__(self, csv_path, transform, df=None):
        # pandas is imported here (training/evaluation only) so the deployed
        # app, which only needs the transforms, doesn't depend on it.
        import pandas as pd
        if df is None:
            df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
        self.targets = self.df["label"].map(self.class_to_idx).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image = load_rgb(self.df.at[i, "image_path"])
        return self.transform(image), int(self.targets[i])
