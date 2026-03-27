"""Ollama API client for local LLM inference."""
import os
import time

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_HEALTH_TIMEOUT = int(os.getenv("OLLAMA_HEALTH_TIMEOUT", "5"))

REPAIR_SYSTEM_PROMPT = (
    "You are an expert programmer specializing in bug fixes. "
    "The user will provide buggy code. Return ONLY the corrected code — "
    "no explanations, no markdown fences, no commentary. "
    "Preserve the original indentation and style exactly."
)

REPAIR_PROMPT_TEMPLATE = """Fix the bug in the following code and return only the corrected code:

{code}"""


def is_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_HEALTH_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """Return list of model names available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_HEALTH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def model_exists(model_name: str) -> bool:
    """Check whether a model is available locally in Ollama."""
    models = list_models()
    # Accept exact match or prefix match (e.g. "llama3.2" matches "llama3.2:latest")
    for m in models:
        if m == model_name or m.startswith(model_name + ":") or m.startswith(model_name):
            return True
    return False


def generate(
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    max_retries: int = 2,
) -> str:
    """
    Send a generation request to Ollama and return the response text.
    Retries on timeout up to max_retries times.
    Raises requests.HTTPError on non-2xx responses.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload: dict = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(1)
        except requests.HTTPError:
            raise  # HTTP errors are not retried

    raise last_exc


def repair_code(model_name: str, buggy_code: str) -> tuple[str, str | None]:
    """
    Ask the model to repair buggy_code.
    Returns (patched_code, error_message).  error_message is None on success.
    """
    prompt = REPAIR_PROMPT_TEMPLATE.format(code=buggy_code)
    try:
        raw = generate(model_name, prompt, system_prompt=REPAIR_SYSTEM_PROMPT)
        cleaned = _strip_markdown_fences(raw)
        return cleaned, None
    except Exception as exc:
        return "", str(exc)


def repair_code_timed(
    model_name: str, buggy_code: str
) -> tuple[str, str | None, float, int]:
    """
    Like repair_code but also returns wall-clock duration and token counts.

    Returns (patched_code, error_message, duration_secs, tokens).
    Tokens are extracted from the Ollama response (eval_count + prompt_eval_count).
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    prompt = REPAIR_PROMPT_TEMPLATE.format(code=buggy_code)
    payload: dict = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "system": REPAIR_SYSTEM_PROMPT,
    }
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        cleaned = _strip_markdown_fences(raw)
        return cleaned, None, time.time() - t0, tokens
    except Exception as exc:
        return "", str(exc), time.time() - t0, 0


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ``` code fences if the model added them."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    # Drop opening fence (```python or ```)
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Drop closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
