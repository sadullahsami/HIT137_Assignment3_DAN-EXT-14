import json
import os
from urllib.parse import urlparse
from typing import Dict, Any

CONFIG_PATH = "app_config.json"
DEFAULT_CONFIG: Dict[str, Any] = {
    "image_top_k": 5,     # 1..10
    "text_max_len": 256,  # 32..2048
}

def is_url(s: str) -> bool:
    try:
        p = urlparse(s.strip())
        return p.scheme in ("http", "https")
    except Exception:
        return False

def clamp_int(val: int, lo: int, hi: int) -> int:
    try:
        v = int(val)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # fill any missing keys with defaults
        cfg = DEFAULT_CONFIG.copy()
        if isinstance(data, dict):
            cfg.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]) -> None:
    # only keep known keys
    out = {k: cfg.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
