from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError


class WasteClassifier:
    def __init__(self, model_path: Path) -> None:
        self.available = False
        self.message = "No trained model found. Follow README.md to train one."
        self._torch = None
        self._model = None
        self._classes: list[str] = []

        if not model_path.is_file():
            return

        try:
            import torch

            from waste_model import WasteCNN
        except ImportError as exc:
            self.message = f"Model exists, but PyTorch is unavailable: {exc}"
            return

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        classes = checkpoint.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"Model checkpoint has no valid classes list: {model_path}")
        model = WasteCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._torch = torch
        self._model = model
        self._classes = classes
        self.available = True
        self.message = "Image classifier is ready."

    def predict(self, image_stream: BinaryIO) -> tuple[str | None, float | None]:
        try:
            with Image.open(image_stream) as image:
                image.verify()
            image_stream.seek(0)
            with Image.open(image_stream) as image:
                image = image.convert("RGB")
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid image") from exc

        if not self.available or self._torch is None or self._model is None:
            return None, None

        image = image.resize((96, 96))
        pixels = self._torch.tensor(list(image.getdata()), dtype=self._torch.float32)
        tensor = pixels.reshape(96, 96, 3).permute(2, 0, 1).div(255).unsqueeze(0)
        with self._torch.inference_mode():
            probabilities = self._torch.softmax(self._model(tensor), dim=1)[0]
            confidence, index = probabilities.max(dim=0)
        return self._classes[int(index)], float(confidence)
