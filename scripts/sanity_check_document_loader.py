"""Pull and visualize one full-document training batch before model training."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from document_dataset import IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT, make_dataloader


OUTPUT_PATH = PROJECT_ROOT / "docs" / "dataloader_sanity_batch.png"


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized tensors back to displayable RGB values."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def main() -> None:
    loader = make_dataloader("train", batch_size=16, shuffle=False)
    images, labels = next(iter(loader))
    expected_shape = (16, 3, 224, 224)
    if tuple(images.shape) != expected_shape:
        raise AssertionError(f"Expected batch shape {expected_shape}, got {tuple(images.shape)}")
    if labels.shape != (16,):
        raise AssertionError(f"Expected 16 labels, got {tuple(labels.shape)}")

    displayed = unnormalize(images[:3])
    grid = make_grid(displayed, nrow=3, padding=4)
    rgb = (grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUTPUT_PATH)

    print(f"CUDA available: {torch.cuda.is_available()} (batch size: 16)")
    print(f"Batch tensor shape: {tuple(images.shape)}")
    print(f"Displayed sample labels (0=real, 1=fake): {labels[:3].tolist()}")
    print(f"Saved un-normalized full-document preview: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
