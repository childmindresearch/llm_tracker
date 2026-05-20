"""Utility functions for llm_tracker."""


import pandas as pd

from llm_tracker.models import AnalysisResult


def format_coding_table(
    results: dict[str, AnalysisResult],
    *,
    max_quote_length: int | None = None,
) -> pd.DataFrame:
    """Format analysis results into a row-level DataFrame.

    Each row represents a single construct instance found across all documents.
    Prints a description and column definitions on call.

    Args:
    ----
        results: Output of analyze_directory(), analyze_csv(), or
            load_human_coding(); maps document IDs to AnalysisResult objects.
        max_quote_length: Optional truncation length for the quote column,
            useful for display purposes.

    Returns:
    -------
        pandas DataFrame with one row per construct instance across all documents.

    """
    print(
        "Coding Table\n"
        "------------\n"
        "One row per construct instance across all documents. Each row represents\n"
        "a single quote identified by the LLM as an instance of a construct.\n"
        "\n"
        "Columns:\n"
        "  doc_id      : document identifier (filename or subreddit_author)\n"
        "  construct   : psychological construct the instance belongs to\n"
        "  quote       : exact quote extracted from the source text\n"
        "  speaker_id  : speaker identifier if available in the source text\n"
        "  quote_index : character-level start:end indices of the quote\n"
        "  confidence  : LLM confidence score (0=not mentioned/negated,\n"
        "                1=indirect, 2=clear)\n"
    )

    rows: list[dict] = []

    for document_id, result in results.items():
        for instance in result.instances:
            quote = instance.quote
            if max_quote_length is not None and len(quote) > max_quote_length:
                quote = quote[: max_quote_length - 3] + "..."

            rows.append(
                {
                    "doc_id": document_id,
                    "construct": instance.construct,
                    "quote": quote,
                    "speaker_id": instance.speaker_id,
                    "quote_index": instance.quote_index,
                    "confidence": instance.confidence,
                }
            )

    columns = [
        "doc_id",
        "construct",
        "quote",
        "speaker_id",
        "quote_index",
        "confidence",
    ]
    return pd.DataFrame(rows, columns=columns)
