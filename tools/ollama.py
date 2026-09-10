"""
tools/ollama.py
Thin client for the Ollama HTTP API. Uses streaming for all generation
calls so there's no read timeout and the terminal shows live progress.
"""
import json
import sys
import requests
from config import config

_TIMEOUT = (10, 600)  # (connect, read) — 10 min read ceiling


def _base_url() -> str:
    return f"http://{config['ollama_host']}:{config['ollama_port']}"


def _stream_generate(url: str, payload: dict, progress_cb=None) -> str:
    """POST to an Ollama streaming endpoint, return full text.
    progress_cb(tokens_so_far) is called every 20 tokens if provided."""
    payload = {**payload, "stream": True}
    parts = []
    token_count = 0
    notify_every = 20

    with requests.post(url, json=payload, stream=True, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)

            token = (
                data.get("response")
                or (data.get("message") or {}).get("content")
                or ""
            )
            parts.append(token)
            token_count += 1

            if token_count % notify_every == 0 and progress_cb:
                progress_cb(token_count)

            if data.get("done"):
                break

    return "".join(parts).strip()


def complete(prompt: str, system: str = "", model: str | None = None, progress_cb=None, options: dict | None = None) -> str:
    """Send a completion request and return the full response text."""
    url = f"{_base_url()}/api/generate"
    payload = {
        "model": model or config["ollama_model"],
        "prompt": prompt,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    return _stream_generate(url, payload, progress_cb=progress_cb)


def chat(messages: list[dict], model: str | None = None, progress_cb=None, options: dict | None = None) -> str:
    """
    Send a chat-format request using /api/chat (Ollama >=0.1.14).
    Falls back to /api/generate if the endpoint is unavailable.
    messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]
    options: Ollama model options dict (e.g. {"num_ctx": 8192, "num_predict": 1024})
    Returns the assistant reply text.
    """
    url = f"{_base_url()}/api/chat"
    payload = {
        "model": model or config["ollama_model"],
        "messages": messages,
    }
    if options:
        payload["options"] = options

    # Probe the endpoint first with a tiny non-streaming request
    try:
        probe = requests.post(
            url,
            json={**payload, "stream": False},
            timeout=(10, 5),
        )
        if probe.status_code == 404:
            return _chat_via_generate(messages, model)
        # If it responded (even with an error other than 404), use /api/chat streaming
    except requests.exceptions.ReadTimeout:
        pass  # timed out waiting for response — that's fine, proceed with streaming

    return _stream_generate(url, payload, progress_cb=progress_cb)


def _chat_via_generate(messages: list[dict], model: str | None = None, progress_cb=None) -> str:
    """Fallback: format chat messages into a single prompt for /api/generate."""
    system = ""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            system = content
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    prompt = "\n\n".join(parts)
    return complete(prompt, system=system, model=model, progress_cb=progress_cb)


def embed(text: str, model: str | None = None) -> list[float]:
    """Generate an embedding vector for the given text.
    Tries /api/embed (Ollama >=0.1.26) then falls back to /api/embeddings.
    Falls back to the main LLM model if the embed model isn't installed."""
    # Try configured embed model, fall back to main model
    models_to_try = list(dict.fromkeys([
        model or config["ollama_embed_model"],
        config["ollama_model"],
    ]))

    for m in models_to_try:
        # Try newer endpoint
        resp = requests.post(
            f"{_base_url()}/api/embed",
            json={"model": m, "input": text},
            timeout=(10, 60),
        )
        if resp.ok:
            return resp.json().get("embeddings", [[]])[0]

        # Try legacy endpoint
        resp2 = requests.post(
            f"{_base_url()}/api/embeddings",
            json={"model": m, "prompt": text},
            timeout=(10, 60),
        )
        if resp2.ok:
            return resp2.json()["embedding"]

    raise RuntimeError(
        f"Embedding failed with all models. "
        f"Run: ollama pull nomic-embed-text"
    )


def is_available() -> bool:
    """Check whether Ollama is running."""
    try:
        resp = requests.get(f"{_base_url()}/api/tags", timeout=5)
        return resp.ok
    except requests.exceptions.ConnectionError:
        return False
