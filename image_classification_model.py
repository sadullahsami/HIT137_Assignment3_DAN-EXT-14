from typing import Any, Dict, List
from io import BytesIO
import os

import requests
from PIL import Image
from transformers import pipeline

from base_model import BaseModel, log_method_call, validate_image_input
from utils import is_url

class ImageClassificationModel(BaseModel):
    """Vision Transformer (ViT) based image classifier."""
    def __init__(self) -> None:
        self._model_name = "google/vit-base-patch16-224"
        self._pipeline = None  # lazily loaded
        self._top_k = 5        # can be set by GUI before processing

    @log_method_call
    def load_model(self) -> None:
        if self._pipeline is None:
            self._pipeline = pipeline(task="image-classification", model=self._model_name)

    @log_method_call
    @validate_image_input
    def process(self, image_input: Any) -> List[Dict[str, float]]:
        if self._pipeline is None:
            self.load_model()

        # prepare PIL image
        if isinstance(image_input, (bytes, bytearray)):
            img = Image.open(BytesIO(image_input)).convert("RGB")
        else:
            s = str(image_input).strip()
            if is_url(s):
                resp = requests.get(s, timeout=20)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                if not os.path.exists(s):
                    raise FileNotFoundError(f"{s}")
                img = Image.open(s).convert("RGB")

        top_k = getattr(self, "_top_k", 5)
        result = self._pipeline(img, top_k=top_k)
        if isinstance(result, dict):
            result = [result]
        return result

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self._model_name,
            "type": "image-classification",
            "task": "Top-K object recognition on ImageNet-like labels",
            "provider": "Hugging Face / Google ViT",
            "notes": "Downloads on first use; cached afterwards.",
        }
