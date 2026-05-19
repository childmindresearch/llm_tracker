"""Focused tests for the DataFrame-first comparison workflow."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from llm_tracker.comparison import (
    COMPARISON_COLUMNS,
    LLMTrackerComparer,
    _compute_span_overlap,
    compute_pr_auc,
    compute_summary_tables,
)
from llm_tracker.models import AnalysisResult, ConstructInstance


def test_compute_span_overlap_uses_interval_overlap() -> None:
    assert _compute_span_overlap("0:10", "5:15") == 5 / 15
    assert _compute_span_overlap("0:10", "10:20") == 0.0
    assert _compute_span_overlap(None, "10:20") is None


def test_compare_results_returns_rows_for_one_sided_constructs() -> None:
    comparer = LLMTrackerComparer(config=SimpleNamespace(max_retries=0))
    human_results = {
        "doc_1": AnalysisResult(
            document_id="doc_1",
            instances=[
                ConstructInstance(
                    construct="stress",
                    quote="I feel overwhelmed",
                    quote_index="0:18",
                    confidence=None,
                )
            ],
        )
    }
    llm_results = {"doc_1": AnalysisResult(document_id="doc_1")}

    df = comparer.compare_results(human_results, llm_results)

    assert list(df.columns) == COMPARISON_COLUMNS
    assert df.to_dict("records") == [
        {
            "doc_id": "doc_1",
            "construct": "stress",
            "status": "human_only",
            "human_quote": "I feel overwhelmed",
            "llm_quote": None,
            "human_indices": "0:18",
            "llm_indices": None,
            "human_confidence": None,
            "llm_confidence": None,
            "paraphrase": None,
            "span_overlap": None,
            "match_confidence": None,
            "tp": 0,
            "fp": 0,
            "fn": 1,
        }
    ]


def test_compare_documents_wraps_single_json_pair(tmp_path) -> None:
    comparer = LLMTrackerComparer(config=SimpleNamespace(max_retries=0))
    human = AnalysisResult(
        document_id="human_doc_name",
        instances=[
            ConstructInstance(
                construct="stress",
                quote="I feel overwhelmed",
                quote_index="0:18",
            )
        ],
    )
    llm = AnalysisResult(document_id="llm_doc_name")
    human_path = tmp_path / "human.json"
    llm_path = tmp_path / "llm.json"
    human_path.write_text(human.model_dump_json(), encoding="utf-8")
    llm_path.write_text(llm.model_dump_json(), encoding="utf-8")

    df = comparer.compare_documents(human_path, llm_path)

    assert df.loc[0, "doc_id"] == "human_doc_name"
    assert df.loc[0, "status"] == "human_only"


def test_pr_auc_uses_llm_coding_confidence_not_matcher_confidence() -> None:
    df = pd.DataFrame(
        [
            {
                "construct": "stress",
                "status": "matched",
                "llm_confidence": 0,
                "match_confidence": 1.0,
            },
            {
                "construct": "stress",
                "status": "llm_only",
                "llm_confidence": 2,
                "match_confidence": None,
            },
        ]
    )

    # If matcher confidence were used, this would be 1.0 because the false
    # positive has no matcher score. Using coding confidence correctly ranks
    # the high-confidence false positive above the true positive.
    assert compute_pr_auc(df)["stress"] == 0.5


def test_compute_summary_tables_counts_and_pr_auc() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "doc_id": "doc_1",
                "construct": "stress",
                "status": "matched",
                "llm_confidence": 2,
                "tp": 1,
                "fp": 0,
                "fn": 0,
            },
            {
                "doc_id": "doc_1",
                "construct": "stress",
                "status": "llm_only",
                "llm_confidence": 1,
                "tp": 0,
                "fp": 1,
                "fn": 0,
            },
            {
                "doc_id": "doc_2",
                "construct": "stress",
                "status": "human_only",
                "llm_confidence": None,
                "tp": 0,
                "fp": 0,
                "fn": 1,
            },
        ]
    )

    per_doc, concatenated, weighted = compute_summary_tables(comparison_df)

    overall = concatenated[concatenated["construct"] == "Overall"].iloc[0]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 1)
    assert overall["precision"] == 0.5
    assert overall["sensitivity"] == 0.5
    assert overall["pr_auc"] == 1.0
    assert set(per_doc["doc_id"]) == {"doc_1", "doc_2"}
    assert "Overall" in set(weighted["construct"])
