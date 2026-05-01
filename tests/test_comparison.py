"""Tests for llm_tracker.comparison module."""

import json

from llm_tracker.comparison import (
    LLMTrackerComparer,
    QuoteMatchDecision,
)


def test_exact_match_prefilter() -> None:
    """Exact quote matches should be detected without LLM calls."""
    human = {
        "document_id": "doc_1",
        "instances": [
            {"construct": "Self-Efficacy", "quote": "I can do it", "speaker_id": "P1"}
        ],
    }
    llm = {
        "document_id": "doc_1",
        "instances": [
            {"construct": "Self-Efficacy", "quote": "I can do it", "speaker_id": "P1"}
        ],
    }

    def _match_fn(_construct: str, _h: dict, _l: dict) -> QuoteMatchDecision:
        return QuoteMatchDecision(
            match=False, reason="should not be called", method="stub"
        )

    comparator = LLMTrackerComparer(
        api_key="test", match_model="test", match_fn=_match_fn
    )
    result = comparator._compare_instances(human["instances"], llm["instances"])

    assert len(result) == 1
    assert result[0]["construct"] == "Self-Efficacy"
    assert len(result[0]["matched"]) == 1
    assert result[0]["matched"][0]["match_method"] == "exact"
    assert result[0]["human_only"] == []
    assert result[0]["llm_only"] == []


def test_llm_matching_stub() -> None:
    """Non-exact matches should go through matcher stub."""
    human = {
        "document_id": "doc_2",
        "instances": [
            {
                "construct": "Growth Mindset",
                "quote": "I can improve",
                "speaker_id": "P1",
            }
        ],
    }
    llm = {
        "document_id": "doc_2",
        "instances": [
            {
                "construct": "Growth Mindset",
                "quote": "I can get better",
                "speaker_id": "P1",
            }
        ],
    }

    def _match_fn(_construct: str, _h: dict, _l: dict) -> QuoteMatchDecision:
        return QuoteMatchDecision(match=True, reason="similar meaning", method="stub")

    comparator = LLMTrackerComparer(
        api_key="test", match_model="test", match_fn=_match_fn
    )
    result = comparator._compare_instances(human["instances"], llm["instances"])

    assert len(result) == 1
    assert len(result[0]["matched"]) == 1
    assert result[0]["matched"][0]["match_method"] == "stub"
    assert result[0]["matched"][0]["match_reason"] == "similar meaning"
    assert result[0]["human_only"] == []
    assert result[0]["llm_only"] == []


def test_directory_comparison(tmp_path) -> None:
    """Directory comparison should handle union of document ids."""
    human_dir = tmp_path / "human"
    llm_dir = tmp_path / "llm"
    human_dir.mkdir()
    llm_dir.mkdir()

    human_doc = {
        "document_id": "doc_a",
        "instances": [{"construct": "A", "quote": "human only", "speaker_id": "P1"}],
    }
    llm_doc = {
        "document_id": "doc_b",
        "instances": [{"construct": "B", "quote": "llm only", "speaker_id": "P2"}],
    }

    with open(human_dir / "doc_a.json", "w", encoding="utf-8") as f:
        json.dump(human_doc, f)
    with open(llm_dir / "doc_b.json", "w", encoding="utf-8") as f:
        json.dump(llm_doc, f)

    def _match_fn(_construct: str, _h: dict, _l: dict) -> QuoteMatchDecision:
        return QuoteMatchDecision(match=False, reason="no match", method="stub")

    comparator = LLMTrackerComparer(
        api_key="test", match_model="test", match_fn=_match_fn
    )
    results = comparator.compare_directories(human_dir, llm_dir)

    assert set(results.keys()) == {"doc_a", "doc_b"}
    assert results["doc_a"]["document_id"] == "doc_a"
    assert results["doc_b"]["document_id"] == "doc_b"
