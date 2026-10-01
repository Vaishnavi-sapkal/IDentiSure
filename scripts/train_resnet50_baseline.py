"""CPU-friendly one-epoch smoke test for full-document SIDTD classification.

The default invocation intentionally trains only one epoch. It fine-tunes the
last ResNet stage and a binary classification head while freezing earlier
ImageNet-pretrained layers; no checkpoint is written during the smoke test.
"""

import argparse
import time

import torch
from torch import nn, optim
from torchvision.models import ResNet50_Weights, resnet50

from document_dataset import PROJECT_ROOT, make_split_dataloaders


LEARNING_RATE = 1e-4
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch_cache"


def build_model() -> nn.Module:
    """Create an ImageNet-pretrained ResNet-50 with a one-logit binary head."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Preserve generic visual features in the stem and first three ResNet
    # stages; train only layer4 and the binary classifier initially.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run train mode when an optimizer is supplied; otherwise validate."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32).unsqueeze(1)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1, help="Smoke-test epochs; default is 1.")
    parser.add_argument("--batch-size", type=int, default=16, help="CPU-safe batch size.")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261001)
    # Keep model-weight downloads inside the project rather than relying on a
    # user-home cache that may be unavailable in managed environments.
    torch.hub.set_dir(str(TORCH_CACHE_DIR))
    loaders = make_split_dataloaders(batch_size=args.batch_size)
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Optimizer: Adam (lr={LEARNING_RATE}) | Loss: BCEWithLogitsLoss")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, loaders["val"], criterion, device)
        elapsed = time.perf_counter() - started
        print(
            f"Epoch {epoch}/{args.epochs} | train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2%} | val loss: {validation_loss:.4f} | "
            f"val accuracy: {validation_accuracy:.2%} | time: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
