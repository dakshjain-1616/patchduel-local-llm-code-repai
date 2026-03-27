"""Tests for the Ollama client (HTTP calls mocked with pytest-mock / unittest.mock)."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from patchduel_local_llm_.ollama_client import (
    _strip_markdown_fences,
    generate,
    is_ollama_running,
    list_models,
    model_exists,
    repair_code,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_response(status: int, body: dict) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


TAGS_RESPONSE = {
    "models": [
        {"name": "llama3.2:latest"},
        {"name": "mistral-nemo:latest"},
        {"name": "codellama:7b"},
    ]
}

GENERATE_RESPONSE = {"response": "def add(a, b):\n    return a + b"}


# ── is_ollama_running ─────────────────────────────────────────────────────────

class TestIsOllamaRunning:
    def test_returns_true_when_server_responds(self):
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)):
            assert is_ollama_running() is True

    def test_returns_false_on_connection_error(self):
        with patch("requests.get", side_effect=ConnectionRefusedError):
            assert is_ollama_running() is False

    def test_returns_false_on_non_200(self):
        with patch("requests.get", return_value=_make_response(500, {})):
            assert is_ollama_running() is False

    def test_returns_false_on_timeout(self):
        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            assert is_ollama_running() is False


# ── list_models ───────────────────────────────────────────────────────────────

class TestListModels:
    def test_returns_model_names(self):
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)):
            models = list_models()
        assert "llama3.2:latest" in models
        assert "mistral-nemo:latest" in models
        assert len(models) == 3

    def test_returns_empty_list_on_error(self):
        with patch("requests.get", side_effect=ConnectionRefusedError):
            models = list_models()
        assert models == []

    def test_returns_empty_list_on_http_error(self):
        with patch("requests.get", return_value=_make_response(404, {})):
            models = list_models()
        assert models == []


# ── model_exists ──────────────────────────────────────────────────────────────

class TestModelExists:
    """Test spec case 3: 'Select llama3.2 → Ollama API call verifies model exists locally'."""

    def test_exact_match_with_tag(self):
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)):
            assert model_exists("llama3.2:latest") is True

    def test_prefix_match_without_tag(self):
        """'llama3.2' should match 'llama3.2:latest'."""
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)):
            assert model_exists("llama3.2") is True

    def test_mistral_nemo_prefix_match(self):
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)):
            assert model_exists("mistral-nemo") is True

    def test_nonexistent_model_returns_false(self):
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)):
            assert model_exists("gpt-99") is False

    def test_returns_false_when_ollama_offline(self):
        with patch("requests.get", side_effect=ConnectionRefusedError):
            assert model_exists("llama3.2") is False

    def test_ollama_api_called_with_correct_endpoint(self):
        with patch("requests.get", return_value=_make_response(200, TAGS_RESPONSE)) as mock_get:
            model_exists("llama3.2")
        call_url = mock_get.call_args[0][0]
        assert "/api/tags" in call_url


# ── generate ──────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_response_text(self):
        with patch("requests.post", return_value=_make_response(200, GENERATE_RESPONSE)):
            result = generate("llama3.2", "fix this code")
        assert result == "def add(a, b):\n    return a + b"

    def test_raises_on_http_error(self):
        with patch("requests.post", return_value=_make_response(404, {})):
            with pytest.raises(requests.HTTPError):
                generate("llama3.2", "prompt")

    def test_system_prompt_included_in_payload(self):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return _make_response(200, GENERATE_RESPONSE)

        with patch("requests.post", side_effect=fake_post):
            generate("llama3.2", "fix", system_prompt="You are helpful.")
        assert captured["payload"]["system"] == "You are helpful."

    def test_stream_is_false(self):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return _make_response(200, GENERATE_RESPONSE)

        with patch("requests.post", side_effect=fake_post):
            generate("llama3.2", "fix")
        assert captured["payload"]["stream"] is False


# ── repair_code ───────────────────────────────────────────────────────────────

class TestRepairCode:
    def test_returns_patch_on_success(self):
        with patch("requests.post", return_value=_make_response(200, GENERATE_RESPONSE)):
            patch_text, err = repair_code("llama3.2", "buggy code")
        assert err is None
        assert "return a + b" in patch_text

    def test_returns_error_on_failure(self):
        with patch("requests.post", side_effect=ConnectionRefusedError("offline")):
            patch_text, err = repair_code("llama3.2", "buggy code")
        assert patch_text == ""
        assert err is not None

    def test_strips_code_fences(self):
        fenced = {"response": "```python\ndef add(a, b):\n    return a + b\n```"}
        with patch("requests.post", return_value=_make_response(200, fenced)):
            patch_text, err = repair_code("llama3.2", "buggy code")
        assert err is None
        assert patch_text.startswith("def add")
        assert "```" not in patch_text


# ── _strip_markdown_fences ────────────────────────────────────────────────────

class TestStripMarkdownFences:
    def test_no_fence_unchanged(self):
        code = "def foo(): pass"
        assert _strip_markdown_fences(code) == code

    def test_generic_fence_removed(self):
        fenced = "```\ndef foo(): pass\n```"
        assert _strip_markdown_fences(fenced) == "def foo(): pass"

    def test_python_fence_removed(self):
        fenced = "```python\ndef foo(): pass\n```"
        assert _strip_markdown_fences(fenced) == "def foo(): pass"

    def test_no_closing_fence(self):
        fenced = "```python\ndef foo(): pass"
        result = _strip_markdown_fences(fenced)
        assert "```" not in result
