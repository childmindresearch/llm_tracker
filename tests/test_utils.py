"""Tests for utility table helpers."""

import pandas as pd
from llm_tracker.models import AnalysisResult, ConstructInstance
from llm_tracker.utils import format_coding_table


def instance(
    construct: str = "stress",
    quote: str = "I feel overwhelmed",
    speaker_id: str | None = None,
    quote_index: str | None = "0:18",
    confidence: int | None = 2,
) -> ConstructInstance:
    return ConstructInstance(
        construct=construct,
        speaker_id=speaker_id,
        quote=quote,
        quote_index=quote_index,
        confidence=confidence,
    )


def result(document_id: str, *instances: ConstructInstance) -> AnalysisResult:
    return AnalysisResult(document_id=document_id, instances=list(instances))


def test_format_coding_table_returns_empty_table_with_expected_columns() -> None:
    table = format_coding_table({})

    assert table.empty
    assert list(table.columns) == [
        "doc_id",
        "construct",
        "quote",
        "speaker_id",
        "quote_index",
        "confidence",
    ]


def test_format_coding_table_returns_one_row_per_instance() -> None:
    results = {
        "doc_1": result(
            "doc_1",
            instance("stress", speaker_id="A"),
            instance("burden", quote="This is too much", confidence=1),
        )
    }

    table = format_coding_table(results)

    assert table.loc[0, "doc_id"] == "doc_1"
    assert table.loc[0, "construct"] == "stress"
    assert table.loc[0, "quote"] == "I feel overwhelmed"
    assert table.loc[0, "speaker_id"] == "A"
    assert table.loc[0, "quote_index"] == "0:18"
    assert table.loc[0, "confidence"] == 2
    assert table.loc[1, "doc_id"] == "doc_1"
    assert table.loc[1, "construct"] == "burden"
    assert table.loc[1, "quote"] == "This is too much"
    assert pd.isna(table.loc[1, "speaker_id"])
    assert table.loc[1, "quote_index"] == "0:18"
    assert table.loc[1, "confidence"] == 1


def test_format_coding_table_truncates_long_quotes() -> None:
    results = {
        "doc_1": result(
            "doc_1",
            instance(quote="This quote is longer than needed"),
        )
    }

    table = format_coding_table(results, max_quote_length=13)

    assert table.loc[0, "quote"] == "This quote..."
