from urllib.parse import urlparse

def is_url(s: str) -> bool:
    """Return True if s looks like an http(s) URL."""
    try:
        p = urlparse(s.strip())
        return p.scheme in ("http", "https")
    except Exception:
        return False
