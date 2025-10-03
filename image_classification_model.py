"""
Image Classification Model implementation using Hugging Face
"""

from base_model import BaseModel, ModelMixin, log_method_call, validate_input
from transformers import pipeline
from PIL import Image
import requests
import logging
import io
import base64

class ImageClassificationModel(BaseModel, ModelMixin):
    """Image classification model - demonstrates multiple inheritance from BaseModel and ModelMixin"""
    
    def __init__(self):
        super().__init__("google/vit-base-patch16-224")  # Using Vision Transformer as it's free and effective
        self._pipeline = None
    
    @log_method_call
    def load_model(self) -> None:
        """Load the image classification model - method overriding"""
        try:
            self._pipeline = pipeline("image-classification", model=self._model_name)
            self._is_loaded = True
            logging.info(f"Successfully loaded {self._model_name}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Fallback to default model if specific model fails
            try:
                self._pipeline = pipeline("image-classification")
                self._model_name = "google/vit-base-patch16-224"
                self._is_loaded = True
                logging.info(f"Loaded fallback model: {self._model_name}")
            except Exception as fallback_error:
                logging.error(f"Error loading fallback model: {fallback_error}")
                raise
    
    @validate_input
    @log_method_call
    def process(self, input_data: str) -> str:
        """Process image input for classification - demonstrates polymorphism"""
        if not self._is_loaded:
            self.load_model()
        
        try:
            # Handle different input types: URL, file path, or demo mode
            if input_data.startswith(('http://', 'https://')):
                # Load image from URL
                try:
                    response = requests.get(input_data, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    return f"Error loading image from URL: {str(e)}"
            else:
                # Try to load as local file or create demo response
                try:
                    image = Image.open(input_data)
                except Exception:
                    # If file doesn't exist, provide demo classification
                    return self._create_demo_classification_response(input_data)
            
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Classify the image
            results = self._pipeline(image)
            
            # Format the results
            if results and len(results) > 0:
                formatted_results = "🖼️ Image Classification Results:\n\n"
                
                for i, result in enumerate(results[:5]):  # Show top 5 predictions
                    label = result['label']
                    score = result['score']
                    confidence_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    formatted_results += f"{i+1}. {label}\n"
                    formatted_results += f"   Confidence: {score:.3f} ({score*100:.1f}%)\n"
                    formatted_results += f"   [{confidence_bar}]\n\n"
                
                # Use mixin method to validate output
                if self.validate_output(formatted_results):
                    return formatted_results
                else:
                    return "Error: Invalid classification result"
            else:
                return "Error: No classification results obtained"
                
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return f"Error processing image: {str(e)}"
    
    def _create_demo_classification_response(self, input_text: str) -> str:
        """Create a demo response when actual image can't be loaded"""
        return f"""
🖼️ DEMO MODE - Image Classification

Input: "{input_text}"

[Demo Classification Results]

1. Golden Retriever (Dog)
   Confidence: 0.942 (94.2%)
   [████████████████████░]

2. Labrador Retriever
   Confidence: 0.038 (3.8%)
   [█░░░░░░░░░░░░░░░░░░░░]

3. Nova Scotia Duck Tolling Retriever  
   Confidence: 0.015 (1.5%)
   [░░░░░░░░░░░░░░░░░░░░░]

4. Brittany Spaniel
   Confidence: 0.003 (0.3%)
   [░░░░░░░░░░░░░░░░░░░░░]

5. Cocker Spaniel
   Confidence: 0.002 (0.2%)
   [░░░░░░░░░░░░░░░░░░░░░]

📝 Note: In full mode, this would classify an actual image.
💡 Try with image URLs like: https://example.com/image.jpg
📁 Or local image files: path/to/image.jpg
        """
    
    @log_method_call
    def get_model_info(self) -> dict:
        """Override parent method to provide specific model info - method overriding"""
        base_info = super().get_model_info()
        base_info.update({
            "type": "Image Classification",
            "category": "Computer Vision",
            "description": "Vision Transformer (ViT) model for image classification. Classifies images into 1000+ categories with confidence scores.",
            "input_type": "Images (URL, file path, or upload)",
            "output_type": "Classification labels with confidence scores",
            "model_size": "Medium (~350MB)",
            "use_case": "Object recognition, content categorization, automated tagging",
            "supported_formats": "JPEG, PNG, BMP, GIF (converted to RGB)",
            "categories": "1000+ ImageNet classes including animals, objects, vehicles, etc."
        })
        return base_info
    
    def get_sample_images(self) -> list:
        """Get sample image URLs for testing - demonstrates encapsulation"""
        return [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
            "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=400",
            "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
        ]
