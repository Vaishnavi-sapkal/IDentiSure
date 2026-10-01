"""PyTorch data loading utilities for full SIDTD document images."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sidtd"
    / "templates"
    / "templates"
    / "Images"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = (224, 224)


class SIDTDFullDocumentDataset(Dataset):
    """Load full real/fake SIDTD documents from one leakage-safe JSONL split.

    Labels are binary: ``0`` for real documents and ``1`` for fake documents.
    The default ImageNet normalization matches common pretrained ResNet and
    EfficientNet backbones.
    """

    def __init__(self, split_path: str | Path, transform=None) -> None:
        self.split_path = Path(split_path)
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")
        self.transform = transform or transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.samples: list[tuple[Path, int]] = []
        with self.split_path.open("r", encoding="utf-8") as split_file:
            for line_number, line in enumerate(split_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_name, row_label = row.get("image"), row.get("label")
                if not isinstance(image_name, str) or row_label not in {"real", "fake"}:
                    raise ValueError(f"Invalid image/label at {self.split_path}:{line_number}")
                image_dir = "reals" if row_label == "real" else "fakes"
                self.samples.append((IMAGES_ROOT / image_dir / image_name, 0 if row_label == "real" else 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image for split sample not found: {image_path}")
        with Image.open(image_path) as image:
            # Converts grayscale, RGBA, CMYK, and palette images to the 3-channel
            # RGB inputs expected by ImageNet-pretrained model backbones.
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def make_dataloader(
    split: str,
    batch_size: int = 16,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for ``train``, ``val``, or ``test`` split JSONL."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    dataset = SIDTDFullDocumentDataset(PROJECT_ROOT / "docs" / "splits" / f"{split}.jsonl")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_split_dataloaders(batch_size: int = 16, num_workers: int = 0) -> dict[str, DataLoader]:
    """Return CPU-friendly train, validation, and test DataLoaders."""
    return {split: make_dataloader(split, batch_size=batch_size, num_workers=num_workers) for split in ("train", "val", "test")}
