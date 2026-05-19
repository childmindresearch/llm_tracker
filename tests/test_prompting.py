"""Tests for LLM response parsing and quote indexing."""

from __future__ import annotations

from llm_tracker.prompting import find_quote_index, parse_llm_response


def test_find_quote_index_is_exact_only_by_default() -> None:
    text = "I don't know what to do anymore."
    quote = "I dont know what to do anymore."

    assert find_quote_index(text, quote) is None


def test_find_quote_index_can_use_fuzzy_matching() -> None:
    text = "I don't know what to do anymore."
    quote = "I dont know what to do anymore."

    assert find_quote_index(text, quote, fuzzy=True, threshold=0.8) == "0:31"


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
