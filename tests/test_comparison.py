"""Tests for comparison tables and metrics."""

from pathlib import Path

import pandas as pd
import pytest
from llm_tracker.comparison import (
    COMPARISON_COLUMNS,
    ComparisonError,
    LLMTrackerComparer,
    compute_pr_auc,
    compute_summary_tables,
)
from llm_tracker.models import AnalysisResult, ConstructInstance


def instance(
    construct: str,
    quote: str = "sample quote",
    quote_index: str | None = "0:12",
    confidence: int | None = None,
) -> ConstructInstance:
    return ConstructInstance(
        construct=construct,
        quote=quote,
        quote_index=quote_index,
        confidence=confidence,
    )


def result(document_id: str, *instances: ConstructInstance) -> AnalysisResult:
    return AnalysisResult(document_id=document_id, instances=list(instances))


class FakeComparer(LLMTrackerComparer):
    def __init__(self, matches: list[dict] | None = None) -> None:
        self.matches = matches or []

    def _match_construct(
        self,
        construct: str,
        human: list[ConstructInstance],
        llm: list[ConstructInstance],
    ) -> list[dict]:
        return self.matches


def test_compare_results_returns_empty_table_for_empty_inputs() -> None:
    df = FakeComparer().compare_results({}, {})

    assert df.empty
    assert list(df.columns) == COMPARISON_COLUMNS


def test_compare_results_marks_human_only_and_llm_only_rows() -> None:
    comparer = FakeComparer()
    human_results = {"doc_1": result("doc_1", instance("stress", quote="human quote"))}
    llm_results = {
        "doc_1": result(
            "doc_1",
            instance("burden", quote="llm quote", confidence=2),
        )
    }

    df = comparer.compare_results(human_results, llm_results)

    rows = df.sort_values("status").to_dict("records")
    assert len(rows) == 2
    assert rows[0]["status"] == "human_only"
    assert rows[0]["human_quote"] == "human quote"
    assert rows[0]["fn"] == 1
    assert rows[1]["status"] == "llm_only"
    assert rows[1]["llm_quote"] == "llm quote"
    assert rows[1]["fp"] == 1


def test_compare_results_marks_matched_rows() -> None:
    comparer = FakeComparer(
        [
            {
                "human_index": 0,
                "llm_index": 0,
                "paraphrase": False,
                "match_confidence": 0.9,
            }
        ]
    )
    human_results = {
        "doc_1": result(
            "doc_1",
            instance("stress", quote="human quote", quote_index="0:10"),
        )
    }
    llm_results = {
        "doc_1": result(
            "doc_1",
            instance(
                "stress",
                quote="llm quote",
                quote_index="5:15",
                confidence=2,
            ),
        )
    }

    row = comparer.compare_results(human_results, llm_results).iloc[0]

    assert row["status"] == "matched"
    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["span_overlap"] == pytest.approx(5 / 15)
    assert row["match_confidence"] == 0.9


def test_compare_documents_loads_json_files(tmp_path: Path) -> None:
    human_path = tmp_path / "human.json"
    llm_path = tmp_path / "llm.json"
    human_path.write_text(
        result("doc_1", instance("stress")).model_dump_json(),
        encoding="utf-8",
    )
    llm_path.write_text(result("doc_1").model_dump_json(), encoding="utf-8")

    df = FakeComparer().compare_documents(human_path, llm_path)

    assert df.loc[0, "doc_id"] == "doc_1"
    assert df.loc[0, "status"] == "human_only"


def test_compare_documents_raises_for_missing_json(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(result("doc_1").model_dump_json(), encoding="utf-8")

    with pytest.raises(ComparisonError, match="Result JSON not found"):
        FakeComparer().compare_documents(missing_path, llm_path)


def test_compare_documents_raises_for_invalid_json(tmp_path: Path) -> None:
    human_path = tmp_path / "human.json"
    llm_path = tmp_path / "llm.json"
    human_path.write_text("{not valid json", encoding="utf-8")
    llm_path.write_text(result("doc_1").model_dump_json(), encoding="utf-8")

    with pytest.raises(ComparisonError, match="Invalid JSON"):
        FakeComparer().compare_documents(human_path, llm_path)


def test_compute_pr_auc_uses_llm_coding_confidence() -> None:
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

    scores = compute_pr_auc(df)

    assert scores["stress"] == 0.5


def test_compute_pr_auc_returns_none_without_both_prediction_labels() -> None:
    df = pd.DataFrame(
        [
            {"construct": "stress", "status": "matched", "llm_confidence": 2},
            {"construct": "stress", "status": "human_only", "llm_confidence": None},
        ]
    )

    scores = compute_pr_auc(df)

    assert scores == {"Overall": None, "stress": None}


def test_compute_summary_tables_returns_core_metrics() -> None:
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

    per_doc, pooled, weighted = compute_summary_tables(comparison_df)
    overall = pooled[pooled["construct"] == "Overall"].iloc[0]

    assert set(per_doc["doc_id"]) == {"doc_1", "doc_2"}
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 1)
    assert overall["sensitivity"] == 0.5
    assert overall["precision"] == 0.5
    assert overall["f1"] == pytest.approx(0.5)
    assert overall["pabak"] == pytest.approx(-0.3333)
    assert overall["pr_auc"] == 1.0
    assert "Overall" in set(weighted["construct"])
