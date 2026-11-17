"""Tests for file loading and saving helpers."""

import json
from pathlib import Path

import pandas as pd
import pytest
from llm_tracker.file_handlers import (
    FileLoadError,
    create_output_directory,
    get_document_files,
    import_codebook,
    load_codebook,
    load_document,
    load_error_records,
    load_human_coding,
    load_human_dataframe,
    save_analysis_result,
    save_error_record,
    save_metadata,
)
from llm_tracker.models import (
    AnalysisResult,
    APIMetadata,
    ConstructInstance,
    ErrorRecord,
)


def test_import_text_codebook_preserves_continued_groups(tmp_path: Path) -> None:
    path = tmp_path / "codebook.txt"
    path.write_text(
        "construct name: definition. Examples. (without bullet points)\n"
        "Calmness and contentment: Calmness: feeling at ease. Examples:\n"
        "Contentment: feeling satisfied. Examples: I feel calm; I am satisfied\n"
        "Connection: Feeling close to others. Examples: we talk; I feel heard"
        ". Examples:\n"
        "Worry: Anticipating difficulties. Examples:\n",
        encoding="utf-8",
    )

    converted = import_codebook(path)

    assert list(converted["codebook"]) == [
        "Calmness and contentment", "Connection", "Worry"
    ]
    assert converted["codebook"]["Calmness and contentment"] == {
        "definition": (
            "Calmness: feeling at ease. Contentment: feeling satisfied."
        ),
        "examples": ["I feel calm", "I am satisfied"],
        "counter_examples": [],
    }
    assert converted["codebook"]["Connection"]["examples"] == [
        "we talk", "I feel heard."
    ]
    assert converted["codebook"]["Worry"]["examples"] == []


@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t")])
def test_import_tabular_codebook_custom_columns(
    tmp_path: Path, suffix: str, separator: str
) -> None:
    path = tmp_path / f"codebook{suffix}"
    pd.DataFrame(
        [{"Name": "Worry, fear", "Meaning": "NA", "Quotes": "a thought; a feeling"}]
    ).to_csv(path, sep=separator, index=False)

    converted = import_codebook(
        path, construct_column="Name", definition_column="Meaning",
        examples_column="Quotes", metadata={"name": "Study"},
    )

    assert converted["metadata"] == {"name": "Study"}
    assert converted["codebook"]["Worry, fear"]["definition"] == "NA"
    assert converted["codebook"]["Worry, fear"]["examples"] == [
        "a thought", "a feeling"
    ]


def test_import_dataframe_preserves_list_examples_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "construct": ["Calmness", "Worry"],
            "definition": ["Feeling at ease.", "Anticipating difficulties."],
            "examples": [["A pause; then relief\nat last"], pd.NA],
            "counter_examples": [None, ["I feel calm"]],
        }
    )
    original = frame.copy(deep=True)

    converted = import_codebook(frame)

    assert converted["codebook"]["Calmness"]["examples"] == [
        "A pause; then relief\nat last"
    ]
    assert converted["codebook"]["Worry"]["examples"] == []
    assert converted["codebook"]["Worry"]["counter_examples"] == ["I feel calm"]
    pd.testing.assert_frame_equal(frame, original)
    json.dumps(converted)


@pytest.mark.parametrize(
    "rows,message",
    [
        ([], "no constructs"),
        ([{"construct": "Worry"}], "unique columns"),
        ([{"construct": " ", "definition": "A definition"}], "invalid construct"),
        ([{"construct": "Worry", "definition": None}], "invalid definition"),
        (
            [
                {"construct": "Worry", "definition": "First"},
                {"construct": " Worry ", "definition": "Second"},
            ],
            "Duplicate codebook constructs",
        ),
    ],
)
def test_import_codebook_rejects_invalid_rows(rows: list, message: str) -> None:
    frame = pd.DataFrame(rows, columns=["construct", "definition"]) if not rows else (
        pd.DataFrame(rows)
    )
    with pytest.raises(FileLoadError, match=message):
        import_codebook(frame)


def test_import_text_codebook_rejects_unstructured_lines(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text("A heading without a construct separator", encoding="utf-8")
    with pytest.raises(FileLoadError, match="line 1"):
        import_codebook(path)


def test_load_codebook_reads_json(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codebook.json"
    codebook_path.write_text('{"constructs": [{"name": "stress"}]}', encoding="utf-8")

    codebook = load_codebook(codebook_path)

    assert codebook == {"constructs": [{"name": "stress"}]}


def test_load_codebook_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileLoadError, match="Codebook file not found"):
        load_codebook(missing_path)


def test_load_codebook_raises_for_invalid_json(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codebook.json"
    codebook_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(FileLoadError, match="Invalid JSON"):
        load_codebook(codebook_path)


def test_load_document_reads_txt_file(tmp_path: Path) -> None:
    document_path = tmp_path / "interview.txt"
    document_path.write_text("plain text", encoding="utf-8")

    text, document_id = load_document(document_path)

    assert text == "plain text"
    assert document_id == "interview"


def test_load_document_extracts_text_from_csv_file(tmp_path: Path) -> None:
    document_path = tmp_path / "interview.csv"
    document_path.write_text(
        "speaker,text,id\nAlice,hello there,1\nBob,goodbye,2\n",
        encoding="utf-8",
    )

    text, document_id = load_document(document_path)

    assert text == "Alice: hello there\nBob: goodbye"
    assert document_id == "interview"


def test_load_document_raises_for_unsupported_file_type(tmp_path: Path) -> None:
    document_path = tmp_path / "interview.docx"
    document_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(FileLoadError, match="Unsupported file type"):
        load_document(document_path)


def test_get_document_files_returns_supported_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.csv").write_text("text\nhello\n", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("ignored", encoding="utf-8")

    files = get_document_files(tmp_path)

    assert [path.name for path in files] == ["a.csv", "b.txt"]


def test_get_document_files_raises_when_no_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("notes", encoding="utf-8")

    with pytest.raises(FileLoadError, match="No document files found"):
        get_document_files(tmp_path)


def test_output_helpers_write_and_load_json_files(tmp_path: Path) -> None:
    output_dir = create_output_directory("run", base_dir=tmp_path)
    result = AnalysisResult(
        document_id="doc_1",
        instances=[
            ConstructInstance(
                construct="stress",
                quote="I feel overwhelmed",
                quote_index="0:18",
                confidence=2,
            )
        ],
    )
    metadata = APIMetadata(model="test-model", num_retries=1)
    error = ErrorRecord(
        document_id="doc_2",
        document_path="doc_2.txt",
        error_message="failed",
        model_used="test-model",
    )

    result_path = save_analysis_result(result, output_dir)
    metadata_path = save_metadata(metadata, "doc_1", output_dir)
    error_path = save_error_record(error, output_dir)
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_errors = load_error_records(output_dir)

    assert output_dir.parent == tmp_path
    assert result_path.exists()
    assert metadata_path.exists()
    assert error_path.exists()
    assert saved_result["document_id"] == "doc_1"
    assert saved_errors[0].document_id == "doc_2"


def test_load_human_dataframe_converts_rows_to_analysis_results() -> None:
    df = pd.DataFrame(
        [
            {
                "Media Title": "doc_1",
                "Excerpt Copy": "I feel overwhelmed",
                "Excerpt Range": "0-18",
                "Codes Applied Combined": "stress, burden",
            }
        ]
    )

    results = load_human_dataframe(df)

    instances = results["doc_1"].instances
    assert [item.construct for item in instances] == ["stress", "burden"]
    assert [item.quote_index for item in instances] == ["0:18", "0:18"]
    assert all(item.confidence is None for item in instances)


def test_load_human_dataframe_raises_for_missing_columns() -> None:
    df = pd.DataFrame([{"Media Title": "doc_1"}])

    with pytest.raises(FileLoadError, match="missing required columns"):
        load_human_dataframe(df)


def test_load_human_coding_reads_csv_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "human.csv"
    csv_path.write_text(
        (
            "Media Title,Excerpt Copy,Excerpt Range,Codes Applied Combined\n"
            "doc_1,I feel overwhelmed,0-18,stress\n"
        ),
        encoding="utf-8",
    )

    results = load_human_coding(csv_path)

    assert results["doc_1"].instances[0].construct == "stress"


def test_load_human_coding_reads_tsv_file_with_custom_columns(tmp_path: Path) -> None:
    tsv_path = tmp_path / "human.tsv"
    tsv_path.write_text(
        "doc\tquote\tcodes\n" "doc_1\tI feel overwhelmed\tstress\n",
        encoding="utf-8",
    )

    results = load_human_coding(
        tsv_path,
        doc_id_col="doc",
        quote_col="quote",
        range_col=None,
        construct_col="codes",
    )

    assert results["doc_1"].instances[0].quote == "I feel overwhelmed"


def test_load_human_coding_raises_for_unsupported_file_type(tmp_path: Path) -> None:
    human_path = tmp_path / "human.txt"
    human_path.write_text("not tabular", encoding="utf-8")

    with pytest.raises(FileLoadError, match="Unsupported file extension"):
        load_human_coding(human_path)
