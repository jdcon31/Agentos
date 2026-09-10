"""
tools/claude_api.py
Anthropic client — the frontier tier.

Mirrors the call shape of tools/ollama.py (complete / analyze_images) so the
router can pick a module rather than every call site branching on tier. The
retry policy lives here rather than at the call sites: transient API failures
are the common failure mode in a long content run, and a job that dies on a 529
halfway through costs more than the retries do.

EXCERPT: the full client also carries the image-to-HTML layout reproduction
path and its prompts. Not included here.
"""
import base64
import mimetypes
import time
from pathlib import Path

import anthropic

from config import config, require

_RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504, 529)
_RETRYABLE_MARKERS = ("OVERLOADED", "RATE_LIMIT", "RATE LIMIT", "UNAVAILABLE", "TIMEOUT")

MAX_ATTEMPTS = 5
BASE_DELAY_SECONDS = 2.0


def _client() -> anthropic.Anthropic:
    """Build a client, failing loudly if the key was never configured."""
    return anthropic.Anthropic(api_key=require("anthropic_api_key"))


def _default_model() -> str:
    return config.get("frontier_model")


def _default_vision_model() -> str:
    return config.get("frontier_vision_model")


def is_retryable(exc: Exception) -> bool:
    """Transient (retry) vs terminal (fail now). Pure — unit tested."""
    if getattr(exc, "status_code", None) in _RETRYABLE_STATUS:
        return True
    message = str(exc).upper()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def backoff_delay(attempt: int, base: float = BASE_DELAY_SECONDS) -> float:
    """Exponential backoff for attempt N (0-indexed). Pure — unit tested."""
    return base * (2 ** attempt)


def _create_with_retry(*, model: str, max_tokens: int, system: str | None,
                       messages: list, _sleep=time.sleep):
    client = _client()
    last_exc = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
            if system is not None:
                kwargs["system"] = system
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            delay = backoff_delay(attempt)
            snippet = (str(exc).splitlines() or [type(exc).__name__])[0][:180]
            print(f"[frontier:{model}] transient error "
                  f"(attempt {attempt + 1}/{MAX_ATTEMPTS}): {snippet} "
                  f"— retrying in {delay:.0f}s")
            _sleep(delay)
    raise last_exc


def complete(system: str, prompt: str, model: str | None = None,
             max_tokens: int = 4096) -> str:
    """Text completion. Same signature shape as tools.ollama.complete."""
    message = _create_with_retry(
        model=model or _default_model(),
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _encode_image(path: Path) -> dict:
    """Encode a local image as a base64 image block."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime = "image/jpeg"
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
    return {"type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data}}


def analyze_images(image_paths: list[Path], prompt: str,
                   model: str | None = None, max_tokens: int = 4096) -> str:
    """Vision call: one or more images plus a text prompt."""
    content = [_encode_image(Path(p)) for p in image_paths]
    content.append({"type": "text", "text": prompt})
    message = _create_with_retry(
        model=model or _default_vision_model(),
        max_tokens=max_tokens,
        system=None,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text.strip()
