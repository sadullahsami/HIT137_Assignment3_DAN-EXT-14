from typing import Any, Dict, List
from transformers import pipeline

from base_model import BaseModel, log_method_call, validate_text_input


class SentimentAnalysisModel(BaseModel):
    """RoBERTa-based sentiment classifier (CardiffNLP)."""
    def __init__(self) -> None:
        self._model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._pipeline = None  # lazily loaded

    @log_method_call
    def load_model(self) -> None:
        if self._pipeline is None:
            self._pipeline = pipeline(task="sentiment-analysis", model=self._model_name)

    @log_method_call
    @validate_text_input
    def process(self, text: Any) -> List[Dict[str, float]]:
        if self._pipeline is None:
            self.load_model()

        result = self._pipeline(text, truncation=True)
        if isinstance(result, dict):
            result = [result]
        return result

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self._model_name,
            "type": "sentiment-analysis",
            "task": "Classify text as Positive / Neutral / Negative",
            "provider": "Hugging Face / CardiffNLP RoBERTa",
            "notes": "Initial download can be slow depending on network.",
        }
