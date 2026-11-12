"""Tests for local discovery result filtering."""

from copy import deepcopy

import pandas as pd
import pytest
from llm_tracker.discovery import (
    MERGED_COLUMNS,
    apply_merge_edits,
    filter_merged_dataframe,
)


@pytest.fixture
def merge_table():
    rows = [
        ("at boundary", 2, 0.3),
        ("strong fit", 4, 0.9),
        ("rare", 1, 0.9),
        ("weak fit", 3, 0.29),
        ("unscored", 3, None),
    ]
    table = pd.DataFrame(
        [
            {
                "merged_construct": "Group",
                "definition": "A group of related ideas.",
                "status": "merged",
                "construct_found": name,
                "document_count": count,
                "instance_count": count + 1,
                "document_frequency_percent": count * 10,
                "total_documents": 10,
                "prototypicality": score,
                "quotes": [f"Quote for {name}"],
            }
            for name, count, score in rows
        ],
        index=[8, 3, 6, 1, 4],
    )
    table["review_note"] = "Preserve extra columns"
    table.attrs["total_documents"] = 10
    return table


def test_default_filter_partitions_without_changing_original(merge_table):
    original = deepcopy(merge_table)

    kept, removed = filter_merged_dataframe(merge_table)

    pd.testing.assert_frame_equal(kept, original.iloc[:2])
    pd.testing.assert_frame_equal(removed, original.iloc[2:])
    pd.testing.assert_frame_equal(merge_table, original)
    assert merge_table.attrs == original.attrs
    assert kept.attrs["total_documents"] == 10
    assert len(kept.attrs["removed_constituents"]) == 3
    _, edited = apply_merge_edits(kept, {"Group": kept["construct_found"].tolist()})
    assert len(edited.attrs["removed_constituents"]) == 3


def test_custom_and_disabled_thresholds(merge_table):
    kept, removed = filter_merged_dataframe(
        merge_table, min_document_count=3, min_prototypicality=0.9
    )
    assert kept["construct_found"].tolist() == ["strong fit"]
    assert len(removed) == 4

    kept, removed = filter_merged_dataframe(
        merge_table, min_document_count=None, min_prototypicality=None
    )
    pd.testing.assert_frame_equal(kept, merge_table)
    assert removed.empty

    kept, _ = filter_merged_dataframe(
        merge_table, min_document_count=None, min_prototypicality=0.3
    )
    assert kept["construct_found"].tolist() == ["at boundary", "strong fit", "rare"]


@pytest.mark.parametrize("save_format", ["csv", "json", "both"])
def test_saves_both_complete_tables(merge_table, tmp_path, save_format):
    kept, removed = filter_merged_dataframe(
        merge_table, output_dir=tmp_path, save_format=save_format
    )
    suffixes = ["csv", "json"] if save_format == "both" else [save_format]
    for suffix in suffixes:
        (kept_file,) = tmp_path.glob(f"filtered_merge_table_*.{suffix}")
        (removed_file,) = tmp_path.glob(f"removed_merge_table_*.{suffix}")
        assert kept_file.stem.removeprefix("filtered_merge_table_") == (
            removed_file.stem.removeprefix("removed_merge_table_")
        )
        reader = pd.read_csv if suffix == "csv" else pd.read_json
        saved_kept = reader(kept_file)
        saved_removed = reader(removed_file)
        assert (
            saved_kept["construct_found"].tolist() == kept["construct_found"].tolist()
        )
        assert saved_removed["construct_found"].tolist() == (
            removed["construct_found"].tolist()
        )
        assert saved_removed["total_documents"].tolist() == [10, 10, 10]
        if suffix == "json":
            assert saved_kept["quotes"].tolist() == kept["quotes"].tolist()
            assert saved_removed["quotes"].tolist() == removed["quotes"].tolist()


def test_empty_and_all_removed_tables_keep_schema(merge_table, tmp_path):
    kept, removed = filter_merged_dataframe(
        merge_table, min_document_count=5, output_dir=tmp_path, save_format="both"
    )
    assert kept.empty
    pd.testing.assert_frame_equal(removed, merge_table)
    (kept_file,) = tmp_path.glob("filtered_merge_table_*.csv")
    assert pd.read_csv(kept_file).columns.tolist() == merge_table.columns.tolist()

    empty = pd.DataFrame(columns=MERGED_COLUMNS)
    kept, removed = filter_merged_dataframe(empty)
    assert kept.empty and removed.empty
    assert kept.columns.tolist() == MERGED_COLUMNS
    assert removed.columns.tolist() == MERGED_COLUMNS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_document_count": -1},
        {"min_document_count": 1.5},
        {"min_document_count": True},
        {"min_prototypicality": -0.1},
        {"min_prototypicality": 1.1},
        {"min_prototypicality": float("nan")},
        {"min_prototypicality": True},
    ],
)
def test_invalid_thresholds_raise(merge_table, kwargs):
    with pytest.raises(ValueError):
        filter_merged_dataframe(merge_table, **kwargs)


def test_missing_columns_raise(merge_table):
    with pytest.raises(ValueError, match="missing columns"):
        filter_merged_dataframe(merge_table.drop(columns="prototypicality"))
