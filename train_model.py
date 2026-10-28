from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset

from waste_model import WasteCNN

IMAGE_SIZE = 96
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class WasteImageDataset(Dataset):
    def __init__(self, folder: Path, classes: list[str]) -> None:
        self.folder = folder
        self.classes = classes
        self.class_indexes = {name: index for index, name in enumerate(classes)}
        self.samples = [
            (path, self.class_indexes[class_name])
            for class_name in classes
            for path in sorted((folder / class_name).glob("*"))
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if not self.samples:
            raise ValueError(f"No supported images found in {folder}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                pixels = torch.tensor(list(image.getdata()), dtype=torch.float32)
                tensor = pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).div(255)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read training image: {path}") from exc
        return tensor, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the waste material image classifier.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/waste_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    train_dir, validation_dir = args.data / "train", args.data / "val"
    classes = sorted(
        path.name for path in train_dir.iterdir()
        if path.is_dir()
    ) if train_dir.is_dir() else []
    if len(classes) < 2:
        parser.error(f"Create at least two class folders under {train_dir}")
    missing_validation = [name for name in classes if not (validation_dir / name).is_dir()]
    if missing_validation:
        parser.error(f"Missing validation class folders: {', '.join(missing_validation)}")

    train_data = WasteImageDataset(train_dir, classes)
    validation_data = WasteImageDataset(validation_dir, classes)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    model = WasteCNN(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == labels).sum())
                total += len(labels)
        accuracy = correct / total if total else 0.0
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / len(train_data):.4f} validation_accuracy={accuracy:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": classes, "state_dict": model.state_dict()}, args.output)
    print(f"Saved model checkpoint to {args.output}")


if __name__ == "__main__":
    main()
