"""
config.py
Configuration surface for AgentOS.

Values resolve in this order: environment variable, then config.json (if
present), then the default below. Secrets are read from the environment only —
they are never written to config.json and never committed. See .env.example for
the full list of variables.
"""
import json
import os
from pathlib import Path

_BASE = Path(__file__).parent
_CONFIG_PATH = _BASE / "config.json"

# key -> (env var, default). A default of "" means "must be supplied to use
# the feature that needs it"; nothing here carries a real value.
_SETTINGS = {
    # Local inference (Ollama)
    "ollama_host":         ("OLLAMA_HOST", "localhost"),
    "ollama_port":         ("OLLAMA_PORT", 11434),
    "ollama_model":        ("OLLAMA_MODEL", "llama3.1:8b"),
    "ollama_embed_model":  ("OLLAMA_EMBED_MODEL", "nomic-embed-text"),

    # Frontier inference (Anthropic)
    "anthropic_api_key":   ("ANTHROPIC_API_KEY", ""),
    "frontier_model":      ("FRONTIER_MODEL", "claude-haiku-4-5"),
    "frontier_vision_model": ("FRONTIER_VISION_MODEL", "claude-sonnet-4-6"),

    # Server
    "server_host":         ("SERVER_HOST", "127.0.0.1"),
    "server_port":         ("SERVER_PORT", 8000),
    "session_secret":      ("SESSION_SECRET", ""),
    "app_password_hash":   ("APP_PASSWORD_HASH", ""),
    "secure_cookies":      ("SECURE_COOKIES", True),

    # Runtime
    "max_concurrent_agents": ("MAX_CONCURRENT_AGENTS", 5),
    "output_path":         ("OUTPUT_PATH", str(_BASE / "output")),
}

_SECRET_KEYS = {"anthropic_api_key", "session_secret", "app_password_hash"}


def _coerce(default, raw: str):
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    return raw


def load() -> dict:
    file_cfg = {}
    if _CONFIG_PATH.exists():
        try:
            file_cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            file_cfg = {}

    cfg = {}
    for key, (env_var, default) in _SETTINGS.items():
        raw = os.environ.get(env_var)
        if raw not in (None, ""):
            cfg[key] = _coerce(default, raw)
        elif key in file_cfg and key not in _SECRET_KEYS:
            cfg[key] = file_cfg[key]
        else:
            cfg[key] = default
    return cfg


config = load()


def require(key: str):
    """Fetch a setting that must be configured, with a useful error."""
    value = config.get(key)
    if value in (None, ""):
        env_var = _SETTINGS.get(key, (key.upper(), None))[0]
        raise RuntimeError(
            f"{key} is not configured. Set the {env_var} environment variable "
            f"(see .env.example)."
        )
    return value
