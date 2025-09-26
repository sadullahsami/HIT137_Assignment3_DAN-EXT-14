from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict

def log_method_call(fn):
    """Minimal logger decorator (stub; can be enhanced later)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # print(f"[LOG] {fn.__name__} called")  # keep silent for now
        return fn(*args, **kwargs)
    return wrapper

def validate_input(fn):
    """Simple validation decorator (stub; per-model checks inside methods)."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # In later commits, enforce image/text expectations here.
        return fn(self, *args, **kwargs)
    return wrapper

class BaseModel(ABC):
    """Abstract interface for all models used by the GUI."""

    @abstractmethod
    def load_model(self) -> None:
        """Load any heavy resources. No-op for stubs."""
        ...

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Run inference on input data and return structured result."""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return static info about the model (name/type/etc.)."""
        ...
