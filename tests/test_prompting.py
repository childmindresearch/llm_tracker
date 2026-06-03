"""Tests for LLM response parsing, quote indexing, and the any-llm wrapper."""

from __future__ import annotations

import pytest

from llm_tracker.config import AnalyzerConfig
from llm_tracker.prompting import (
    PromptingError,
    call_llm_api,
    find_quote_index,
    parse_llm_response,
)


def test_find_quote_index_is_exact_only_by_default() -> None:
    text = "I don't know what to do anymore."
    quote = "I dont know what to do anymore."

    quote_index = find_quote_index(text, quote)

    assert quote_index is None


def test_find_quote_index_can_use_fuzzy_matching() -> None:
    text = "I don't know what to do anymore."
    quote = "I dont know what to do anymore."

    quote_index = find_quote_index(text, quote, fuzzy=True, threshold=0.8)

    assert quote_index == "0:31"


def test_find_quote_index_ignores_empty_quotes() -> None:
    quote_index = find_quote_index("Some source text.", "")

    assert quote_index is None


def test_parse_llm_response_respects_fuzzy_quote_matching_flag() -> None:
    response = (
        '{"instances": [{"construct": "stress", '
        '"quote": "I dont know what to do anymore.", "confidence": 2}]}'
    )
    text = "I don't know what to do anymore."

    exact = parse_llm_response(response, "doc_1", text)
    fuzzy = parse_llm_response(
        response,
        "doc_1",
        text,
        fuzzy_quote_matching=True,
        threshold=0.8,
    )

    assert exact.instances[0].quote_index is None
    assert fuzzy.instances[0].quote_index == "0:31"


# --- any-llm wrapper (call_llm_api) ---------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    """Minimal OpenAI-shaped response object, like any-llm returns."""

    def __init__(self) -> None:
        self.choices = [_FakeChoice('{"instances": []}')]
        self.model = "google/gemini-3-flash-preview"
        self.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.id = "resp_123"
        self.created = 1234567890

    def model_dump(self) -> dict:
        return {
            "model": self.model,
            "usage": self.usage,
            "id": self.id,
            "created": self.created,
        }


class _FakeClient:
    def __init__(self) -> None:
        self.received_kwargs: dict | None = None

    def completion(self, **kwargs: object) -> _FakeResponse:
        self.received_kwargs = kwargs
        return _FakeResponse()


def test_call_llm_api_maps_response_to_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "llm_tracker.prompting.AnyLLM.create",
        lambda provider, api_key=None: fake_client,
    )
    config = AnalyzerConfig(api_key="test-key", temperature=0)

    text, metadata = call_llm_api("hello", config)

    assert text == '{"instances": []}'
    assert metadata.model == "google/gemini-3-flash-preview"
    assert metadata.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert metadata.response_id == "resp_123"
    assert metadata.created == 1234567890
    assert metadata.latency_ms is not None
    assert metadata.raw_response is not None


def test_call_llm_api_forwards_response_format_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "llm_tracker.prompting.AnyLLM.create",
        lambda provider, api_key=None: fake_client,
    )
    config = AnalyzerConfig(api_key="test-key", temperature=0)

    call_llm_api("hello", config)

    assert fake_client.received_kwargs is not None
    assert fake_client.received_kwargs["response_format"] == {"type": "json_object"}
    assert fake_client.received_kwargs["temperature"] == 0


def test_call_llm_api_omits_temperature_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "llm_tracker.prompting.AnyLLM.create",
        lambda provider, api_key=None: fake_client,
    )
    config = AnalyzerConfig(
        api_key="test-key", temperature=None
    )  # temperature defaults to None

    call_llm_api("hello", config)

    assert fake_client.received_kwargs is not None
    assert "temperature" not in fake_client.received_kwargs


def test_call_llm_api_wraps_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(provider: str, api_key: str | None = None) -> object:
        raise RuntimeError("network down")

    monkeypatch.setattr("llm_tracker.prompting.AnyLLM.create", boom)
    config = AnalyzerConfig(api_key="test-key")

    with pytest.raises(PromptingError):
        call_llm_api("hello", config)
