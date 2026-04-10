"""Tests for pychometrics.file_handlers module."""

import json
from pathlib import Path

import pytest
from pychometrics.file_handlers import (
    FileLoadError,
    create_output_directory,
    get_document_files,
    load_codebook,
    load_csv_document,
    load_dedoose_xlsx,
    load_document,
    load_txt_document,
    save_analysis_result,
    save_metadata,
    save_readme,
)
from pychometrics.models import AnalysisResult, APIMetadata, ConstructInstance


def test_valid_codebook(tmp_path) -> None:
    """Test loading a valid codebook."""
    codebook_data = {
        "constructs": [
            {
                "name": " -Efficacy",
                "definition": "Belief in capabilities",
                "examples": ["I can do it"],
            }
        ]
    }

    codebook_file = tmp_path / "codebook.json"
    with open(codebook_file, "w") as f:
        json.dump(codebook_data, f)

    result = load_codebook(codebook_file)

    assert "constructs" in result
    assert len(result["constructs"]) == 1


def test_nonexistent_codebook() -> None:
    """Test loading a nonexistent codebook raises error."""
    with pytest.raises(FileLoadError, match="not found"):
        load_codebook(Path("/nonexistent/codebook.json"))


def test_invalid_json(tmp_path):
    """Test loading invalid JSON raises error."""
    codebook_file = tmp_path / "bad.json"
    with open(codebook_file, "w") as f:
        f.write("not valid json {")

    with pytest.raises(FileLoadError, match="Invalid JSON"):
        load_codebook(codebook_file)


def test_non_json_file(tmp_path):
    """Test loading non-JSON file raises error."""
    codebook_file = tmp_path / "codebook.txt"
    codebook_file.write_text("some text")

    with pytest.raises(FileLoadError, match="must be a JSON"):
        load_codebook(codebook_file)


def test_simple_txt(tmp_path):
    """Test loading simple text file."""
    txt_file = tmp_path / "interview.txt"
    txt_file.write_text("Hello, this is an interview.")

    result = load_txt_document(txt_file)

    assert result == "Hello, this is an interview."


def test_multiline_txt(tmp_path):
    """Test loading multiline text file."""
    txt_file = tmp_path / "interview.txt"
    txt_file.write_text("Line 1\nLine 2\nLine 3")

    result = load_txt_document(txt_file)

    assert "Line 1" in result
    assert "Line 3" in result


def test_single_column_csv(tmp_path):
    """Test loading CSV with single text column."""
    csv_file = tmp_path / "interview.csv"
    csv_file.write_text("text\nFirst response\nSecond response")

    result = load_csv_document(csv_file)

    assert "First response" in result
    assert "Second response" in result


def test_speaker_and_text_columns(tmp_path):
    """Test loading CSV with speaker and text columns."""
    csv_file = tmp_path / "interview.csv"
    csv_file.write_text("speaker,text\nP1,Hello there\nP2,Hi back")

    result = load_csv_document(csv_file)

    assert "P1:" in result
    assert "Hello there" in result
    assert "P2:" in result


def test_multiple_text_columns(tmp_path):
    """Test loading CSV with multiple columns."""
    csv_file = tmp_path / "interview.csv"
    csv_file.write_text("id,content,notes\n1,Main text,Extra info")

    result = load_csv_document(csv_file)

    assert "Main text" in result


def test_txt_document(tmp_path):
    """Test loading TXT document."""
    txt_file = tmp_path / "interview_001.txt"
    txt_file.write_text("Interview content")

    text, doc_id = load_document(txt_file)

    assert text == "Interview content"
    assert doc_id == "interview_001"


def test_unsupported_format(tmp_path):
    """Test loading unsupported format raises error."""
    doc_file = tmp_path / "document.docx"
    doc_file.write_text("content")

    with pytest.raises(FileLoadError, match="Unsupported file type"):
        load_document(doc_file)


def test_finds_txt_and_csv(tmp_path):
    """Test finding both TXT and CSV files."""
    (tmp_path / "file1.txt").write_text("text")
    (tmp_path / "file2.csv").write_text("csv")
    (tmp_path / "file3.docx").write_text("docx")

    files = get_document_files(tmp_path)

    assert len(files) == 2
    names = [f.name for f in files]
    assert "file1.txt" in names
    assert "file2.csv" in names
    assert "file3.docx" not in names


def test_empty_directory(tmp_path):
    """Test empty directory raises error."""
    with pytest.raises(FileLoadError, match="No document files"):
        get_document_files(tmp_path)


def test_nonexistent_directory():
    """Test nonexistent directory raises error."""
    with pytest.raises(FileLoadError, match="not found"):
        get_document_files(Path("/nonexistent/dir"))


def test_creates_structure(tmp_path):
    """Test output directory structure is created."""
    output_dir = create_output_directory("test", tmp_path)

    assert output_dir.exists()
    assert (output_dir / "encodings").exists()
    assert (output_dir / "metadata").exists()


def test_includes_timestamp(tmp_path):
    """Test output directory name includes timestamp."""
    output_dir = create_output_directory("myanalysis", tmp_path)

    assert "myanalysis_" in output_dir.name


def test_default_name(tmp_path):
    """Test default name when none provided."""
    output_dir = create_output_directory(None, tmp_path)

    assert "pychometrics_output_" in output_dir.name


def test_saves_json(tmp_path):
    """Test analysis result is saved as JSON."""
    output_dir = create_output_directory("test", tmp_path)

    result = AnalysisResult(
        document_id="doc_001",
        instances=[ConstructInstance(construct="Test", quote="A quote", confidence=2)],
    )

    saved_path = save_analysis_result(result, output_dir)

    assert saved_path.exists()
    with open(saved_path) as f:
        data = json.load(f)

    assert data["document_id"] == "doc_001"
    assert len(data["instances"]) == 1


def test_saves_metadata_json(tmp_path):
    """Test metadata is saved as JSON."""
    output_dir = create_output_directory("test", tmp_path)

    metadata = APIMetadata(model="test-model", latency_ms=1234.5)

    saved_path = save_metadata(metadata, "doc_001", output_dir)

    assert saved_path.exists()
    assert "_meta.json" in saved_path.name


def test_creates_readme(tmp_path):
    """Test README is created with correct content."""
    output_dir = create_output_directory("test", tmp_path)

    readme_path = save_readme(
        output_dir=output_dir,
        model_name="test-model",
        codebook_name="codebook.json",
        input_dir_name="interviews",
        failed_documents=["failed.txt"],
        total_documents=5,
    )

    assert readme_path.exists()
    content = readme_path.read_text()

    assert "test-model" in content
    assert "codebook.json" in content
    assert "interviews" in content
    assert "failed.txt" in content
    assert "**Failed**: 1" in content


def test_readme_no_failures(tmp_path):
    """Test README when no documents failed."""
    output_dir = create_output_directory("test", tmp_path)

    readme_path = save_readme(
        output_dir=output_dir,
        model_name="model",
        codebook_name="codebook.json",
        input_dir_name="inputs",
        failed_documents=[],
        total_documents=3,
    )

    content = readme_path.read_text()
    assert "All documents processed successfully" in content


@pytest.fixture
def dedoose_codebook(tmp_path):
    """Minimal codebook with common test constructs."""
    data = {
        "constructs": [
            {"name": "Anxiety", "definition": "def"},
            {"name": "Depression", "definition": "def"},
        ]
    }
    path = tmp_path / "codebook.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def make_dedoose_xlsx(tmp_path):
    """Factory fixture — call it with rows to get an xlsx path back."""  # noqa: D401
    import pandas as pd

    def _make(rows, filename="dedoose.xlsx"):
        df = pd.DataFrame(
            rows,
            columns=[
                "Media Title",
                "Excerpt Range",
                "Excerpt Copy",
                "Codes Applied Combined",
            ],
        )
        path = tmp_path / filename
        df.to_excel(path, index=False)
        return path

    return _make


def test_dedoose_basic(dedoose_codebook, make_dedoose_xlsx):
    """Single row, single construct — happy path."""
    xlsx = make_dedoose_xlsx(
        [["interview_01", "100-200", "I felt very anxious", "Anxiety"]]
    )
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert len(results) == 1
    assert results[0].document_id == "interview_01"
    assert len(results[0].instances) == 1
    assert results[0].instances[0].construct == "Anxiety"
    assert results[0].instances[0].quote == "I felt very anxious"
    assert results[0].instances[0].quote_index == "100:200"


def test_dedoose_multiple_constructs_same_row(dedoose_codebook, make_dedoose_xlsx):
    """One excerpt tagged with multiple constructs splits into multiple instances."""
    xlsx = make_dedoose_xlsx(
        [
            [
                "interview_01",
                "0-50",
                "I feel anxious and depressed",
                "Anxiety, Depression",
            ]
        ]
    )
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert len(results[0].instances) == 2
    constructs = {i.construct for i in results[0].instances}
    assert constructs == {"Anxiety", "Depression"}


def test_dedoose_multiple_documents(dedoose_codebook, make_dedoose_xlsx):
    """Rows from different Media Titles produce separate AnalysisResult objects."""
    xlsx = make_dedoose_xlsx(
        [
            ["doc_a", "0-50", "quote one", "Anxiety"],
            ["doc_b", "0-50", "quote two", "Anxiety"],
        ]
    )
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    doc_ids = {r.document_id for r in results}
    assert doc_ids == {"doc_a", "doc_b"}


def test_dedoose_multiple_rows_same_document(dedoose_codebook, make_dedoose_xlsx):
    """Multiple excerpts from the same document are grouped into one AnalysisResult."""
    xlsx = make_dedoose_xlsx(
        [
            ["doc_a", "0-50", "first quote", "Anxiety"],
            ["doc_a", "100-150", "second quote", "Anxiety"],
        ]
    )
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert len(results) == 1
    assert len(results[0].instances) == 2


def test_dedoose_confidence_is_none(dedoose_codebook, make_dedoose_xlsx):
    """Human codings should have confidence=None."""
    xlsx = make_dedoose_xlsx([["doc_a", "0-50", "a quote", "Anxiety"]])
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert results[0].instances[0].confidence is None


def test_dedoose_speaker_id_is_none(dedoose_codebook, make_dedoose_xlsx):
    """Dedoose excerpts have no speaker — speaker_id should be None."""
    xlsx = make_dedoose_xlsx([["doc_a", "0-50", "a quote", "Anxiety"]])
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert results[0].instances[0].speaker_id is None


def test_dedoose_range_parsing(dedoose_codebook, make_dedoose_xlsx):
    """Excerpt range '858-1159' is converted to '858:1159'."""
    xlsx = make_dedoose_xlsx([["doc_a", "858-1159", "a quote", "Anxiety"]])
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert results[0].instances[0].quote_index == "858:1159"


def test_dedoose_malformed_range(dedoose_codebook, make_dedoose_xlsx):
    """Malformed excerpt range results in quote_index=None rather than crashing."""
    xlsx = make_dedoose_xlsx([["doc_a", "not-a-range", "a quote", "Anxiety"]])
    results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert results[0].instances[0].quote_index is None


def test_dedoose_unknown_construct_warns(dedoose_codebook, make_dedoose_xlsx):
    """Constructs not in the codebook emit a warning and are skipped."""
    xlsx = make_dedoose_xlsx([["doc_a", "0-50", "a quote", "Anxiety, UnknownCode"]])

    with pytest.warns(UserWarning, match="not found in codebook"):
        results = load_dedoose_xlsx(xlsx, dedoose_codebook)

    assert len(results[0].instances) == 1
    assert results[0].instances[0].construct == "Anxiety"


def test_dedoose_file_not_found(dedoose_codebook, tmp_path):
    """Missing xlsx raises FileLoadError."""
    with pytest.raises(FileLoadError, match="not found"):
        load_dedoose_xlsx(tmp_path / "nonexistent.xlsx", dedoose_codebook)


def test_dedoose_missing_columns(dedoose_codebook, tmp_path):
    """Xlsx missing required columns raises FileLoadError."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "Media Title": "doc",
                "Excerpt Range": "0-10",
                "Excerpt Copy": "quote",
                # 'Codes Applied Combined' intentionally missing
            }
        ]
    )
    path = tmp_path / "bad.xlsx"
    df.to_excel(path, index=False)

    with pytest.raises(FileLoadError, match="missing required columns"):
        load_dedoose_xlsx(path, dedoose_codebook)
