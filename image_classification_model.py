from typing import Any, Dict, List
from base_model import BaseModel, log_method_call, validate_input

class ImageClassificationModel(BaseModel):
    """Stub for an image-classification model (to be wired later)."""
    def __init__(self) -> None:
        self._model_name = "google/vit-base-patch16-224"
        self._pipeline = None  # real pipeline will be added later

    @log_method_call
    def load_model(self) -> None:
        # TODO: load transformers image-classification pipeline
        self._pipeline = "stub-loaded"

    @log_method_call
    @validate_input
    def process(self, image_input: Any) -> List[Dict[str, float]]:
        # TODO: run inference and return top-k predictions
        # Return a deterministic-looking placeholder for now
        return [{"label": "placeholder-class", "score": 0.99}]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self._model_name, "type": "image-classification"}
