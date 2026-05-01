"""Tests for llm_tracker.prompting module."""

import json

import pytest

from llm_tracker.prompting import (
    ResponseParseError,
    construct_prompt,
    parse_llm_response,
)


def test_basic_prompt_construction() -> None:
    """Test basic prompt construction with placeholders."""
    template = "Text: {text}\nCodebook: {codebook}"
    text = "Sample interview text"
    codebook = {"constructs": [{"name": "Test", "definition": "A test"}]}

    prompt = construct_prompt(text, codebook, template)

    assert "Sample interview text" in prompt
    assert '"name": "Test"' in prompt


def test_multiline_text() -> None:
    """Test prompt construction with multiline text."""
    template = "Analyze: {text}\nUsing: {codebook}"
    text = "Line 1\nLine 2\nLine 3"
    codebook = {"constructs": []}

    prompt = construct_prompt(text, codebook, template)

    assert "Line 1\nLine 2\nLine 3" in prompt


def test_valid_json_response() -> None:
    """Test parsing a valid JSON response."""
    response = json.dumps(
        {
            "instances": [
                {
                    "construct": "Self-Efficacy",
                    "speaker_id": "P1",
                    "quote": "I can do it",
                    "confidence": 2,
                }
            ]
        }
    )

    result, was_repaired = parse_llm_response(response, "doc_001")

    assert result.document_id == "doc_001"
    assert len(result.instances) == 1
    assert result.instances[0].construct == "Self-Efficacy"
    assert result.instances[0].confidence == 2
    assert was_repaired is False


def test_json_with_markdown_wrapper() -> None:
    """Test parsing JSON wrapped in markdown code block."""
    response = """```json
{
"instances": [
    {
        "construct": "Resilience",
        "speaker_id": null,
        "quote": "I bounced back",
        "confidence": 1
    }
]
}
```"""

    result, was_repaired = parse_llm_response(response, "doc_002")

    assert len(result.instances) == 1
    assert result.instances[0].construct == "Resilience"
    assert result.instances[0].speaker_id is None
    assert was_repaired is True


def test_empty_instances() -> None:
    """Test parsing response with no instances."""
    response = '{"instances": []}'

    result, was_repaired = parse_llm_response(response, "empty_doc")

    assert len(result.instances) == 0
    assert was_repaired is False


def test_invalid_json_raises_error() -> None:
    """Test that invalid JSON raises error."""
    response = "This is not valid JSON at all"

    with pytest.raises(ResponseParseError):
        parse_llm_response(response, "bad_doc")


def test_json_with_extra_text() -> None:
    """Parse JSON embedded in extra text."""
    response = """
    Some preamble text.
    {"instances": [{"construct": "A", "quote": "q", "confidence": 2}]}
    Some trailing text.
    """

    result, was_repaired = parse_llm_response(response, "doc_extra")

    assert len(result.instances) == 1
    assert result.instances[0].construct == "A"
    assert was_repaired is True


def test_malformed_instance_skipped() -> None:
    """Test that malformed instances are skipped."""
    response = json.dumps(
        {
            "instances": [
                {"construct": "Valid", "quote": "valid quote", "confidence": 2},
                {
                    "construct": "Invalid",
                    "quote": "invalid",
                    "confidence": "not_a_number",
                },
            ]
        }
    )

    result, was_repaired = parse_llm_response(response, "doc_004")

    assert len(result.instances) == 1
    assert result.instances[0].construct == "Valid"
    assert was_repaired is False


def test_null_speaker_id() -> None:
    """Test handling of null speaker_id."""
    response = json.dumps(
        {
            "instances": [
                {
                    "construct": "Test",
                    "speaker_id": None,
                    "quote": "A quote",
                    "confidence": 1,
                }
            ]
        }
    )

    result, was_repaired = parse_llm_response(response, "doc_005")

    assert result.instances[0].speaker_id is None
    assert was_repaired is False


def test_confidence_boundary_values() -> None:
    """Test confidence scores at boundaries (0 and 2)."""
    response = json.dumps(
        {
            "instances": [
                {"construct": "A", "quote": "q1", "confidence": 0},
                {"construct": "B", "quote": "q2", "confidence": 2},
            ]
        }
    )

    result, was_repaired = parse_llm_response(response, "doc_006")

    assert result.instances[0].confidence == 0
    assert result.instances[1].confidence == 2
    assert was_repaired is False
