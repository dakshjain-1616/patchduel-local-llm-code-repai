"""OpenRouter API client — cloud LLM inference via a single API key."""
from __future__ import annotations

import os
import time

import requests

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "60"))
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com/patchduel")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "PatchDuel")

# Curated list of current OpenRouter models (updated 2026-03)
CURATED_MODELS: list[str] = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "mistralai/mistral-small-2603",
    "minimax/minimax-m2.7",
    "xiaomi/mimo-v2-pro",
    "xiaomi/mimo-v2-omni",
    "z-ai/glm-5-turbo",
    "reka/reka-edge",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
    "x-ai/grok-4.20-beta",
    "x-ai/grok-4.20-multi-agent-beta",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

# Default model (best value/speed balance)
DEFAULT_OPENROUTER_MODEL = os.getenv(
    "DEFAULT_OPENROUTER_MODEL", "openai/gpt-5.4-mini"
)

REPAIR_SYSTEM_PROMPT = (
    "You are an expert programmer specializing in bug fixes. "
    "The user will provide buggy code. Return ONLY the corrected code — "
    "no explanations, no markdown fences, no commentary. "
    "Preserve the original indentation and style exactly."
)

REPAIR_PROMPT_TEMPLATE = """Fix the bug in the following code and return only the corrected code:

{code}"""


def is_configured() -> bool:
    """Return True if an OpenRouter API key is available."""
    return bool(OPENROUTER_API_KEY)


def list_models() -> list[str]:
    """Return the curated list of current OpenRouter models."""
    return list(CURATED_MODELS)


def generate(
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    max_retries: int = 2,
) -> tuple[str, int]:
    """
    Send a chat completion request to OpenRouter.

    Returns (response_text, total_tokens_used).
    Raises ValueError if the API key is not configured.
    Raises requests.HTTPError on non-2xx responses.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file to use OpenRouter models."
        )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_SITE_NAME,
        "Content-Type": "application/json",
    }
    payload = {"model": model_name, "messages": messages}

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=OPENROUTER_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return content, tokens
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2**attempt)
        except requests.HTTPError as exc:
            # Do not retry 4xx client errors
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2**attempt)

    raise last_exc


def repair_code(model_name: str, buggy_code: str) -> tuple[str, str | None, int]:
    """
    Ask the model to repair buggy_code via OpenRouter.

    Returns (patched_code, error_message, tokens_used).
    error_message is None on success.
    """
    from core.ollama_client import _strip_markdown_fences

    prompt = REPAIR_PROMPT_TEMPLATE.format(code=buggy_code)
    try:
        raw, tokens = generate(model_name, prompt, system_prompt=REPAIR_SYSTEM_PROMPT)
        cleaned = _strip_markdown_fences(raw)
        return cleaned, None, tokens
    except Exception as exc:
        return "", str(exc), 0


def repair_code_timed(
    model_name: str, buggy_code: str
) -> tuple[str, str | None, float, int]:
    """
    Like repair_code but also returns wall-clock duration in seconds.

    Returns (patched_code, error_message, duration_secs, tokens_used).
    """
    t0 = time.time()
    patch, err, tokens = repair_code(model_name, buggy_code)
    return patch, err, time.time() - t0, tokens
