from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict

# For friendly error messages
try:
    from PIL import UnidentifiedImageError  # type: ignore
except Exception:  # PIL may not be imported yet in some envs
    class UnidentifiedImageError(Exception):  # fallback
        pass

# requests is optional at import-time; guards below
try:
    import requests
    _REQ_EXC = (
        requests.exceptions.MissingSchema,
        requests.exceptions.InvalidURL,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )
except Exception:  # if requests isn't available yet
    _REQ_EXC = tuple()

class ModelInputError(Exception):
    """Raised when the user input fails validation (empty text, bad path, etc.)."""
    pass


def log_method_call(fn):
    """Minimal logger decorator (kept silent by default)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def validate_image_input(fn):
    """
    Ensure image input is either:
      - bytes/bytearray, or
      - a valid http(s) URL, or
      - an existing local file path
    """
    @wraps(fn)
    def wrapper(self, image_input: Any, *args, **kwargs):
        # bytes are accepted
        if isinstance(image_input, (bytes, bytearray)):
            return fn(self, image_input, *args, **kwargs)

        from utils import is_url
        import os

        s = str(image_input).strip()
        if not s:
            raise ModelInputError("Please provide an image URL or a local file path.")
        if is_url(s):
            return fn(self, s, *args, **kwargs)
        if os.path.exists(s):
            return fn(self, s, *args, **kwargs)

        # Not URL and not a file
        raise FileNotFoundError(s)
    return wrapper


def validate_text_input(fn):
    """Ensure text is non-empty (after trimming)."""
    @wraps(fn)
    def wrapper(self, text: Any, *args, **kwargs):
        s = str(text) if text is not None else ""
        if not s.strip():
            raise ModelInputError("Please enter some text to analyse.")
        return fn(self, s, *args, **kwargs)
    return wrapper


def friendly_error_message(e: Exception) -> str:
    """
    Map common exceptions to human-friendly messages shown in the GUI.
    """
    # Validation
    if isinstance(e, ModelInputError):
        return str(e)

    # File issues
    if isinstance(e, FileNotFoundError):
        return f"Image file not found: {e}"
    if isinstance(e, UnidentifiedImageError):
        return "The file/URL could not be opened as an image."

    # Requests / network
    if _REQ_EXC and isinstance(e, _REQ_EXC):
        try:
            import requests  # type: ignore
            if isinstance(e, requests.exceptions.MissingSchema) or isinstance(e, requests.exceptions.InvalidURL):
                return "That doesn't look like a valid URL. Please provide a full http/https link."
            if isinstance(e, requests.exceptions.ConnectionError):
                return "Couldn't reach the URL (connection error). Please check your network or the URL."
            if isinstance(e, requests.exceptions.Timeout):
                return "The request timed out while fetching the URL."
            if isinstance(e, requests.exceptions.HTTPError):
                status = getattr(getattr(e, 'response', None), 'status_code', '?')
                return f"HTTP error while fetching the URL (status {status})."
        except Exception:
            pass

    # Fallback
    return str(e)


class BaseModel(ABC):
    """Abstract interface for all models used by the GUI."""
    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def process(self, data: Any) -> Any:
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        ...
