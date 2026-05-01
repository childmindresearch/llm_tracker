"""Utility functions for llm_tracker."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_coding_table(
    results: dict,
    *,
    max_quote_length: int | None = None,
) -> pd.DataFrame:
    """Format the raw coding results dict into a row-level DataFrame.

    Each row represents a single construct instance found across all documents.
    Prints a description and column definitions on call.

    Args:
        results: Output of analyze_directory() or analyze_csv() — a dict
            mapping document IDs to their analysis result dicts.
        max_quote_length: Optional truncation length for the quote column,
            useful for display purposes.

    Returns:
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
        "  quote_index : character-level start:end indices of the quote in the source text\n"
        "  confidence  : LLM confidence score (0=not mentioned/negated, 1=indirect, 2=clear)\n"
    )

    rows: list[dict] = []

    for document_id, result in results.items():
        instances = result.get("instances", [])
        for instance in instances:
            quote = str(instance.get("quote", ""))
            if max_quote_length is not None:
                if len(quote) > max_quote_length:
                    quote = quote[:max_quote_length - 3] + "..."
            rows.append({
                "doc_id": document_id,
                "construct": instance.get("construct"),
                "quote": quote,
                "speaker_id": instance.get("speaker_id"),
                "quote_index": instance.get("quote_index"),
                "confidence": instance.get("confidence"),
            })

    columns = ["doc_id", "construct", "quote", "speaker_id", "quote_index", "confidence"]
    return pd.DataFrame(rows, columns=columns)
