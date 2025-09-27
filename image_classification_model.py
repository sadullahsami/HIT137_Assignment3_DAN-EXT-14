from typing import Any, Dict, List
from io import BytesIO
from urllib.parse import urlparse
import os

import requests
from PIL import Image
from transformers import pipeline

from base_model import BaseModel, log_method_call, validate_input


def _is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False


class ImageClassificationModel(BaseModel):
    """Vision Transformer (ViT) based image classifier."""
    def __init__(self) -> None:
        self._model_name = "google/vit-base-patch16-224"
        self._pipeline = None  # lazily loaded

    @log_method_call
    def load_model(self) -> None:
        if self._pipeline is None:
            # downloads on first use; then cached by HF
            self._pipeline = pipeline(
                task="image-classification",
                model=self._model_name
            )

    @log_method_call
    @validate_input
    def process(self, image_input: Any) -> List[Dict[str, float]]:
        if self._pipeline is None:
            self.load_model()

        if isinstance(image_input, (bytes, bytearray)):
            img = Image.open(BytesIO(image_input)).convert("RGB")
        else:
            path = str(image_input).strip()
            if _is_url(path):
                resp = requests.get(path, timeout=20)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image file not found: {path}")
                img = Image.open(path).convert("RGB")

        # top-5 predictions
        result = self._pipeline(img, top_k=5)
        # ensure a List[Dict]
        if isinstance(result, dict):
            result = [result]
        return result

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self._model_name, "type": "image-classification"}
