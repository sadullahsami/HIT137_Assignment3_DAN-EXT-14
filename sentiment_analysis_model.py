from typing import Any, Dict, List
from base_model import BaseModel, log_method_call, validate_input

class SentimentAnalysisModel(BaseModel):
    """Stub for a sentiment-analysis model (to be wired later)."""
    def __init__(self) -> None:
        self._model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._pipeline = None  # real pipeline will be added later

    @log_method_call
    def load_model(self) -> None:
        # TODO: load transformers sentiment-analysis pipeline
        self._pipeline = "stub-loaded"

    @log_method_call
    @validate_input
    def process(self, text: Any) -> List[Dict[str, float]]:
        # TODO: run inference and return sentiment + confidence
        return [{"label": "NEUTRAL", "score": 0.95}]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self._model_name, "type": "sentiment-analysis"}
