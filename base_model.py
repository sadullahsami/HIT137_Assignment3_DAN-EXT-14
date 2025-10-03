"""
Base classes for AI models with OOP concepts implementation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import functools
import logging

# Decorator for logging method calls
def log_method_call(func):
    """Decorator to log method calls - demonstrates decorator usage"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logging.info(f"Calling {self.__class__.__name__}.{func.__name__}")
        result = func(self, *args, **kwargs)
        logging.info(f"Completed {self.__class__.__name__}.{func.__name__}")
        return result
    return wrapper

# Decorator for validation
def validate_input(func):
    """Decorator to validate input - demonstrates multiple decorators"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not args and not kwargs:
            raise ValueError("Input cannot be empty")
        return func(self, *args, **kwargs)
    return wrapper

class BaseModel(ABC):
    """Abstract base class for AI models - demonstrates encapsulation and abstraction"""
    
    def __init__(self, model_name: str):
        self._model_name = model_name  # Private attribute - encapsulation
        self._model = None
        self._is_loaded = False
    
    @property
    def model_name(self) -> str:
        """Getter for model name - demonstrates encapsulation"""
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded - demonstrates encapsulation"""
        return self._is_loaded
    
    @abstractmethod
    @log_method_call
    def load_model(self) -> None:
        """Abstract method to load the model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    @validate_input
    @log_method_call
    def process(self, input_data: Any) -> Any:
        """Abstract method to process input - demonstrates polymorphism"""
        pass
    
    @log_method_call
    def get_model_info(self) -> Dict[str, str]:
        """Get basic model information - can be overridden (method overriding)"""
        return {
            "name": self._model_name,
            "type": "Base Model",
            "status": "Loaded" if self._is_loaded else "Not Loaded"
        }

class ModelMixin:
    """Mixin class for additional functionality - demonstrates multiple inheritance"""
    
    def validate_output(self, output: Any) -> bool:
        """Validate model output"""
        return output is not None
    
    def format_output(self, output: Any) -> str:
        """Format output for display"""
        if isinstance(output, (list, tuple)):
            return f"Generated {len(output)} items"
        elif isinstance(output, str):
            return output[:100] + "..." if len(output) > 100 else output
        else:
            return str(output)
