"""Tests for the OpenRouter client (all HTTP calls mocked)."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from patchduel_local_llm_.openrouter_client import (
    generate,
    is_configured,
    list_models,
    repair_code,
    repair_code_timed,
    CURATED_MODELS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_response(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


_CHAT_RESPONSE = {
    "choices": [{"message": {"content": "def add(a, b):\n    return a + b"}}],
    "usage": {"total_tokens": 42},
}

_API_KEY = "sk-or-test-key"


# ── is_configured ─────────────────────────────────────────────────────────────

class TestIsConfigured:
    def test_returns_false_when_no_key(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", "")
        assert is_configured() is False

    def test_returns_true_when_key_set(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        assert is_configured() is True


# ── list_models ───────────────────────────────────────────────────────────────

class TestListModels:
    def test_returns_list_of_strings(self):
        models = list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    def test_contains_expected_model(self):
        assert "openai/gpt-5.4-mini" in list_models()

    def test_curated_models_not_empty(self):
        assert len(CURATED_MODELS) >= 5

    def test_returns_copy(self):
        # Modifying return value should not affect CURATED_MODELS
        models = list_models()
        models.clear()
        assert len(CURATED_MODELS) > 0


# ── generate ──────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", "")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            generate("openai/gpt-5.4-mini", "hello")

    def test_returns_content_and_tokens(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        with patch("requests.post", return_value=_make_response(200, _CHAT_RESPONSE)):
            content, tokens = generate("openai/gpt-5.4-mini", "fix this")
        assert "return a + b" in content
        assert tokens == 42

    def test_raises_on_4xx_without_retry(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        with patch("requests.post", return_value=_make_response(401, {})):
            with pytest.raises(requests.HTTPError):
                generate("openai/gpt-5.4-mini", "test")

    def test_system_prompt_included_in_payload(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        captured: dict = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _make_response(200, _CHAT_RESPONSE)

        with patch("requests.post", side_effect=fake_post):
            generate("openai/gpt-5.4-mini", "fix", system_prompt="Be concise.")

        messages = captured["payload"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    def test_auth_header_present(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        captured: dict = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers
            return _make_response(200, _CHAT_RESPONSE)

        with patch("requests.post", side_effect=fake_post):
            generate("openai/gpt-5.4-mini", "hello")

        assert f"Bearer {_API_KEY}" in captured["headers"]["Authorization"]

    def test_missing_usage_field_defaults_to_zero(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        body = {"choices": [{"message": {"content": "ok"}}]}  # no "usage" key
        with patch("requests.post", return_value=_make_response(200, body)):
            _, tokens = generate("openai/gpt-5.4-mini", "hello")
        assert tokens == 0


# ── repair_code ───────────────────────────────────────────────────────────────

class TestRepairCode:
    def test_returns_three_tuple(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        with patch("requests.post", return_value=_make_response(200, _CHAT_RESPONSE)):
            result = repair_code("openai/gpt-5.4-mini", "buggy code")
        assert len(result) == 3

    def test_success_returns_patch_and_no_error(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        with patch("requests.post", return_value=_make_response(200, _CHAT_RESPONSE)):
            patch_text, err, tokens = repair_code("openai/gpt-5.4-mini", "buggy code")
        assert err is None
        assert "return a + b" in patch_text
        assert tokens == 42

    def test_error_returns_empty_patch_and_message(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", "")
        patch_text, err, tokens = repair_code("openai/gpt-5.4-mini", "code")
        assert patch_text == ""
        assert err is not None
        assert tokens == 0

    def test_strips_markdown_fences(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        fenced = {
            "choices": [{"message": {"content": "```python\ndef f(): pass\n```"}}],
            "usage": {"total_tokens": 10},
        }
        with patch("requests.post", return_value=_make_response(200, fenced)):
            patch_text, err, _ = repair_code("openai/gpt-5.4-mini", "def f(): bug")
        assert "```" not in patch_text
        assert "def f(): pass" in patch_text


# ── repair_code_timed ─────────────────────────────────────────────────────────

class TestRepairCodeTimed:
    def test_returns_four_tuple(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        with patch("requests.post", return_value=_make_response(200, _CHAT_RESPONSE)):
            result = repair_code_timed("openai/gpt-5.4-mini", "code")
        assert len(result) == 4

    def test_duration_is_non_negative_float(self, monkeypatch):
        monkeypatch.setattr("patchduel_local_llm_.openrouter_client.OPENROUTER_API_KEY", _API_KEY)
        with patch("requests.post", return_value=_make_response(200, _CHAT_RESPONSE)):
            _, _, duration, _ = repair_code_timed("openai/gpt-5.4-mini", "code")
        assert isinstance(duration, float)
        assert duration >= 0
