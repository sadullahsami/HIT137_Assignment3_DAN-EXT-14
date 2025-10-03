"""
Sentiment Analysis Model implementation using Hugging Face
"""

from base_model import BaseModel, ModelMixin, log_method_call, validate_input
from transformers import pipeline
import logging

class SentimentAnalysisModel(BaseModel, ModelMixin):
    """Sentiment analysis model - demonstrates multiple inheritance and polymorphism"""
    
    def __init__(self):
        super().__init__("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self._pipeline = None
    
    @log_method_call
    def load_model(self) -> None:
        """Load the sentiment analysis model - method overriding"""
        try:
            self._pipeline = pipeline("sentiment-analysis", model=self._model_name)
            self._is_loaded = True
            logging.info(f"Successfully loaded {self._model_name}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Fallback to default model if specific model fails
            try:
                self._pipeline = pipeline("sentiment-analysis")
                self._model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self._is_loaded = True
                logging.info(f"Loaded fallback model: {self._model_name}")
            except Exception as fallback_error:
                logging.error(f"Error loading fallback model: {fallback_error}")
                raise
    
    @validate_input
    @log_method_call
    def process(self, input_text: str) -> str:
        """Process text input for sentiment analysis - demonstrates polymorphism"""
        if not self._is_loaded:
            self.load_model()
        
        try:
            # Analyze sentiment
            result = self._pipeline(input_text)
            
            if isinstance(result, list) and len(result) > 0:
                sentiment_result = result[0]
                label = sentiment_result.get('label', 'Unknown')
                score = sentiment_result.get('score', 0.0)
                
                # Format the output using mixin method
                formatted_result = f"Sentiment: {label} (Confidence: {score:.2f})"
                
                # Use mixin method to validate output
                if self.validate_output(formatted_result):
                    return formatted_result
                else:
                    return "Error: Invalid sentiment analysis result"
            else:
                return "Error: No sentiment analysis result"
                
        except Exception as e:
            logging.error(f"Error processing sentiment: {e}")
            return f"Error processing sentiment: {str(e)}"
    
    @log_method_call
    def get_model_info(self) -> dict:
        """Override parent method to provide specific model info - method overriding"""
        base_info = super().get_model_info()
        base_info.update({
            "type": "Sentiment Analysis",
            "category": "Natural Language Processing",
            "description": "RoBERTa-based model fine-tuned for sentiment analysis. Classifies text as positive, negative, or neutral.",
            "input_type": "Text",
            "output_type": "Sentiment Classification with Confidence Score",
            "model_size": "Medium (~500MB)",
            "use_case": "Social media monitoring, customer feedback analysis, review classification",
            "labels": "POSITIVE, NEGATIVE, NEUTRAL"
        })
        return base_info
    
    def get_sentiment_distribution(self, text_list: list) -> dict:
        """Additional method specific to sentiment analysis - demonstrates encapsulation"""
        if not self._is_loaded:
            self.load_model()
        
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        
        for text in text_list:
            try:
                result = self._pipeline(text)
                if result and len(result) > 0:
                    label = result[0].get('label', 'NEUTRAL')
                    if label in sentiment_counts:
                        sentiment_counts[label] += 1
                    else:
                        # Map different label formats
                        if label.upper().startswith('POS'):
                            sentiment_counts["POSITIVE"] += 1
                        elif label.upper().startswith('NEG'):
                            sentiment_counts["NEGATIVE"] += 1
                        else:
                            sentiment_counts["NEUTRAL"] += 1
            except Exception as e:
                logging.error(f"Error processing text in batch: {e}")
                sentiment_counts["NEUTRAL"] += 1
        
        return sentiment_counts
