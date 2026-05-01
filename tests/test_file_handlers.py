"""Tests for llm_tracker.file_handlers module."""

import json
from pathlib import Path

import pytest
from llm_tracker.file_handlers import (
    FileLoadError,
    ValidationReport,
    create_output_directory,
    get_document_files,
    load_codebook,
    load_csv_document,
    load_document,
    load_human_coding,
    load_human_dataframe,
    load_txt_document,
    save_analysis_result,
    save_human_results,
    save_metadata,
    save_readme,
    validate_against_codebook,
)
from llm_tracker.models import AnalysisResult, APIMetadata, ConstructInstance


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

    assert "llm_tracker_output_" in output_dir.name


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
def dedoose_rows():
    """Factory fixture — returns a list of row dicts with Dedoose column names."""

    def _make(rows):
        return [
            {
                "Media Title": r[0],
                "Excerpt Range": r[1],
                "Excerpt Copy": r[2],
                "Codes Applied Combined": r[3],
            }
            for r in rows
        ]

    return _make


@pytest.fixture
def make_dedoose_xlsx(tmp_path, dedoose_rows):
    """Factory: call with rows to get back a path to a Dedoose-shaped xlsx."""  # noqa: D401
    import pandas as pd

    def _make(rows, filename="dedoose.xlsx"):
        df = pd.DataFrame(dedoose_rows(rows))
        path = tmp_path / filename
        df.to_excel(path, index=False)
        return path

    return _make


@pytest.fixture
def make_dedoose_csv(tmp_path, dedoose_rows):
    """Factory: call with rows to get back a path to a Dedoose-shaped csv."""  # noqa: D401
    import pandas as pd

    def _make(rows, filename="dedoose.csv"):
        df = pd.DataFrame(dedoose_rows(rows))
        path = tmp_path / filename
        df.to_csv(path, index=False)
        return path

    return _make


# ---------------------------------------------------------------------------
# load_human_coding  (file dispatcher: csv / tsv / xlsx / xls)
# ---------------------------------------------------------------------------


def test_human_coding_xlsx_basic(make_dedoose_xlsx):
    """Single row, single construct — xlsx happy path."""
    xlsx = make_dedoose_xlsx(
        [["interview_01", "100-200", "I felt very anxious", "Anxiety"]]
    )
    results = load_human_coding(xlsx)

    assert len(results) == 1
    assert results[0].document_id == "interview_01"
    assert len(results[0].instances) == 1
    assert results[0].instances[0].construct == "Anxiety"
    assert results[0].instances[0].quote == "I felt very anxious"
    assert results[0].instances[0].quote_index == "100:200"


def test_human_coding_csv_basic(make_dedoose_csv):
    """Same happy path, but via .csv."""
    csv = make_dedoose_csv(
        [["interview_01", "100-200", "I felt very anxious", "Anxiety"]]
    )
    results = load_human_coding(csv)

    assert len(results) == 1
    assert results[0].instances[0].construct == "Anxiety"
    assert results[0].instances[0].quote_index == "100:200"


def test_human_coding_multiple_constructs_same_row(make_dedoose_csv):
    """One excerpt with comma-joined codes splits into multiple instances."""
    csv = make_dedoose_csv(
        [
            [
                "interview_01",
                "0-50",
                "I feel anxious and depressed",
                "Anxiety, Depression",
            ]
        ]
    )
    results = load_human_coding(csv)

    constructs = {i.construct for i in results[0].instances}
    assert constructs == {"Anxiety", "Depression"}


def test_human_coding_multiple_documents(make_dedoose_csv):
    """Rows from different Media Titles produce separate AnalysisResult objects."""
    csv = make_dedoose_csv(
        [
            ["doc_a", "0-50", "quote one", "Anxiety"],
            ["doc_b", "0-50", "quote two", "Anxiety"],
        ]
    )
    results = load_human_coding(csv)

    doc_ids = {r.document_id for r in results}
    assert doc_ids == {"doc_a", "doc_b"}


def test_human_coding_multiple_rows_same_document(make_dedoose_csv):
    """Multiple excerpts from the same document are grouped into one AnalysisResult."""
    csv = make_dedoose_csv(
        [
            ["doc_a", "0-50", "first quote", "Anxiety"],
            ["doc_a", "100-150", "second quote", "Anxiety"],
        ]
    )
    results = load_human_coding(csv)

    assert len(results) == 1
    assert len(results[0].instances) == 2


def test_human_coding_confidence_is_none(make_dedoose_csv):
    """Human codings should have confidence=None."""
    csv = make_dedoose_csv([["doc_a", "0-50", "a quote", "Anxiety"]])
    results = load_human_coding(csv)
    assert results[0].instances[0].confidence is None


def test_human_coding_speaker_id_is_none(make_dedoose_csv):
    """Dedoose excerpts have no speaker — speaker_id should be None."""
    csv = make_dedoose_csv([["doc_a", "0-50", "a quote", "Anxiety"]])
    results = load_human_coding(csv)
    assert results[0].instances[0].speaker_id is None


def test_human_coding_malformed_range(make_dedoose_csv):
    """Malformed excerpt range results in quote_index=None rather than crashing."""
    csv = make_dedoose_csv([["doc_a", "not-a-range", "a quote", "Anxiety"]])
    results = load_human_coding(csv)
    assert results[0].instances[0].quote_index is None


def test_human_coding_unknown_construct_is_loaded_not_filtered(make_dedoose_csv):
    """Loader no longer validates against a codebook — all constructs pass through."""
    csv = make_dedoose_csv([["doc_a", "0-50", "a quote", "Anxiety, UnknownCode"]])
    results = load_human_coding(csv)
    constructs = {i.construct for i in results[0].instances}
    assert constructs == {"Anxiety", "UnknownCode"}


def test_human_coding_file_not_found(tmp_path):
    """Missing file raises FileLoadError."""
    with pytest.raises(FileLoadError, match="not found"):
        load_human_coding(tmp_path / "nonexistent.csv")


def test_human_coding_missing_columns(tmp_path):
    """File missing required columns raises FileLoadError."""
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
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(FileLoadError, match="missing required columns"):
        load_human_coding(path)


def test_human_coding_unsupported_extension(tmp_path):
    """Unknown extensions raise a clear FileLoadError."""
    path = tmp_path / "data.parquet"
    path.write_bytes(b"not really parquet")

    with pytest.raises(FileLoadError, match="Unsupported file extension"):
        load_human_coding(path)


def test_human_coding_tsv_uses_tab_separator(tmp_path, dedoose_rows):
    """A .tsv file is read with sep='\\t' by default."""
    import pandas as pd

    df = pd.DataFrame(dedoose_rows([["doc_a", "0-50", "a quote", "Anxiety"]]))
    path = tmp_path / "data.tsv"
    df.to_csv(path, sep="\t", index=False)

    results = load_human_coding(path)
    assert results[0].instances[0].construct == "Anxiety"


def test_human_coding_forwards_read_kwargs(tmp_path, dedoose_rows):
    """Extra kwargs are forwarded to the pandas reader (e.g. encoding)."""
    import pandas as pd

    df = pd.DataFrame(dedoose_rows([["doc_a", "0-50", "a quote", "Anxiety"]]))
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    # Smoke test that an explicit encoding kwarg is accepted and used.
    results = load_human_coding(path, encoding="utf-8")
    assert results[0].instances[0].construct == "Anxiety"


def test_human_coding_custom_columns_via_kwargs(tmp_path):
    """Non-Dedoose column names work via kwargs."""
    import pandas as pd

    df = pd.DataFrame(
        [{"doc": "a", "range": "0-20", "excerpt": "I felt scared", "codes": "Anxiety"}]
    )
    path = tmp_path / "custom.csv"
    df.to_csv(path, index=False)

    results = load_human_coding(
        path,
        doc_id_col="doc",
        quote_col="excerpt",
        range_col="range",
        construct_col="codes",
    )
    assert results[0].instances[0].construct == "Anxiety"
    assert results[0].instances[0].quote_index == "0:20"


def test_human_coding_cp1252_file_falls_back(tmp_path, dedoose_rows):
    """A cp1252-encoded CSV (e.g. Dedoose export with smart quotes) loads with a warning."""
    import pandas as pd

    df = pd.DataFrame(
        dedoose_rows([["doc_a", "0-50", "She said \u201chello\u201d", "Anxiety"]])
    )
    path = tmp_path / "cp1252.csv"
    df.to_csv(path, index=False, encoding="cp1252")

    with pytest.warns(UserWarning, match="fell back"):
        results = load_human_coding(path)

    assert results[0].instances[0].construct == "Anxiety"


def test_human_coding_utf8_file_no_warning(tmp_path, dedoose_rows):
    """A plain utf-8 file loads cleanly on the first try, no warning."""
    import pandas as pd
    import warnings

    df = pd.DataFrame(dedoose_rows([["doc_a", "0-50", "a quote", "Anxiety"]]))
    path = tmp_path / "utf8.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        results = load_human_coding(path)

    assert results[0].instances[0].construct == "Anxiety"


def test_human_coding_utf8_sig_bom_falls_back(tmp_path, dedoose_rows):
    """A utf-8-with-BOM file loads via the utf-8-sig fallback."""
    import pandas as pd

    df = pd.DataFrame(dedoose_rows([["doc_a", "0-50", "a quote", "Anxiety"]]))
    path = tmp_path / "bom.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # utf-8-sig is parseable as utf-8 (the BOM becomes part of the first
    # column name). If the first column header ends up with a BOM prefix
    # we'd see a "missing required columns" FileLoadError. Verify fallback
    # actually rescues us.
    results = load_human_coding(path)
    assert results[0].instances[0].construct == "Anxiety"


def test_human_coding_explicit_encoding_skips_fallback(tmp_path, dedoose_rows):
    """Passing encoding=... bypasses the fallback (broken file -> error)."""
    import pandas as pd

    df = pd.DataFrame(dedoose_rows([["doc_a", "0-50", "a quote", "Anxiety"]]))
    path = tmp_path / "cp1252.csv"
    df.to_csv(path, index=False, encoding="cp1252")

    # Write an actual byte that's invalid utf-8 to guarantee failure.
    data = path.read_bytes() + b"\x85"
    path.write_bytes(data)

    with pytest.raises(FileLoadError):
        load_human_coding(path, encoding="utf-8")


# ---------------------------------------------------------------------------
# load_human_coding save_dir + save_human_results
# ---------------------------------------------------------------------------


def test_save_dir_creates_analyzer_compatible_layout(
    tmp_path, make_dedoose_csv, monkeypatch
):
    """save_dir writes the same layout analyze_directory/analyze_csv produces."""
    import json

    csv = make_dedoose_csv(
        [
            ["doc_a", "0-50", "q1", "Anxiety"],
            ["doc_a", "60-90", "q2", "Depression"],
            ["doc_b", "0-50", "q3", "Anxiety"],
        ]
    )

    monkeypatch.chdir(tmp_path)
    results = load_human_coding(csv, save_dir="human_run")

    # Expect one timestamped directory under CWD (tmp_path).
    dirs = [
        p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("human_run_")
    ]
    assert len(dirs) == 1
    out_dir = dirs[0]

    # Standard layout mirrors the analyzer.
    assert (out_dir / "encodings").is_dir()
    assert (out_dir / "metadata").is_dir()
    assert (out_dir / "errors").is_dir()
    assert (out_dir / "README.md").exists()

    # One JSON per document, filenames match document_ids.
    json_files = sorted((out_dir / "encodings").glob("*.json"))
    assert [f.stem for f in json_files] == ["doc_a", "doc_b"]

    # Payload shape matches analyzer output.
    payload = json.loads((out_dir / "encodings" / "doc_a.json").read_text())
    assert payload["document_id"] == "doc_a"
    assert isinstance(payload["instances"], list)
    assert len(payload["instances"]) == 2
    inst = payload["instances"][0]
    assert "construct" in inst and "quote" in inst and "quote_index" in inst
    # Human codings have confidence=None.
    assert inst["confidence"] is None

    # The function still returns the list.
    assert len(results) == 2


def test_save_dir_none_is_a_noop(tmp_path, make_dedoose_csv, monkeypatch):
    """Without save_dir, nothing is written to disk."""
    csv = make_dedoose_csv([["doc_a", "0-50", "q", "Anxiety"]])
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    load_human_coding(csv)
    after = set(tmp_path.iterdir())
    assert before == after


def test_save_dir_readme_mentions_source(tmp_path, make_dedoose_csv, monkeypatch):
    """The auto-generated README records the source file and doc count."""
    csv = make_dedoose_csv([["doc_a", "0-50", "q", "Anxiety"]], filename="export.csv")
    monkeypatch.chdir(tmp_path)
    load_human_coding(csv, save_dir="human_run")

    out_dir = next(p for p in tmp_path.iterdir() if p.name.startswith("human_run_"))
    readme = (out_dir / "README.md").read_text()
    assert "export.csv" in readme
    assert "Documents**: 1" in readme


def test_save_human_results_standalone(tmp_path):
    """save_human_results can be called directly on in-memory AnalysisResult lists."""
    import json

    results = [
        AnalysisResult(
            document_id="doc_x",
            instances=[ConstructInstance(construct="Anxiety", quote="q")],
        )
    ]
    out = save_human_results(results, output_name="standalone", base_dir=tmp_path)

    # Created under tmp_path with a timestamped name
    assert out.parent == tmp_path
    assert out.name.startswith("standalone_")

    # Produces the same layout
    assert (out / "encodings" / "doc_x.json").exists()
    payload = json.loads((out / "encodings" / "doc_x.json").read_text())
    assert payload["document_id"] == "doc_x"


# ---------------------------------------------------------------------------
# load_human_dataframe  (generic loader)
# ---------------------------------------------------------------------------


def test_human_dataframe_custom_columns():
    """Generic loader works with arbitrary column names."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "doc": "interview_01",
                "range": "0-20",
                "excerpt": "I felt scared",
                "codes": "Anxiety",
            }
        ]
    )
    results = load_human_dataframe(
        df,
        doc_id_col="doc",
        quote_col="excerpt",
        range_col="range",
        construct_col="codes",
    )
    assert len(results) == 1
    assert results[0].document_id == "interview_01"
    assert results[0].instances[0].construct == "Anxiety"
    assert results[0].instances[0].quote_index == "0:20"


def test_human_dataframe_no_range_column():
    """range_col=None skips range parsing entirely; quote_index is None."""
    import pandas as pd

    df = pd.DataFrame([{"doc": "a", "excerpt": "quote", "codes": "Anxiety"}])
    results = load_human_dataframe(
        df,
        doc_id_col="doc",
        quote_col="excerpt",
        range_col=None,
        construct_col="codes",
    )
    assert results[0].instances[0].quote_index is None


def test_human_dataframe_colon_range_format():
    """range_format='colon' handles '858:1159' style ranges."""
    import pandas as pd

    df = pd.DataFrame(
        [{"doc": "a", "excerpt": "quote", "range": "100:250", "codes": "Anxiety"}]
    )
    results = load_human_dataframe(
        df,
        doc_id_col="doc",
        quote_col="excerpt",
        range_col="range",
        construct_col="codes",
        range_format="colon",
    )
    assert results[0].instances[0].quote_index == "100:250"


def test_human_dataframe_custom_separator():
    """construct_separator lets us handle constructs that contain commas."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "doc": "a",
                "excerpt": "quote",
                "range": "0-10",
                "codes": "trauma, abuse|Anxiety",
            }
        ]
    )
    results = load_human_dataframe(
        df,
        doc_id_col="doc",
        quote_col="excerpt",
        range_col="range",
        construct_col="codes",
        construct_separator="|",
    )
    constructs = {i.construct for i in results[0].instances}
    assert constructs == {"trauma, abuse", "Anxiety"}


def test_human_dataframe_rejects_non_dataframe():
    """Passing something that isn't a DataFrame raises FileLoadError."""
    with pytest.raises(FileLoadError, match="expected a pandas DataFrame"):
        load_human_dataframe("not a dataframe")  # type: ignore[arg-type]


def test_human_dataframe_invalid_range_format():
    """Unknown range_format raises a clear error."""
    import pandas as pd

    df = pd.DataFrame(
        [{"doc": "a", "excerpt": "q", "range": "0-10", "codes": "Anxiety"}]
    )
    with pytest.raises(ValueError, match="Unknown range_format"):
        load_human_dataframe(
            df,
            doc_id_col="doc",
            quote_col="excerpt",
            range_col="range",
            construct_col="codes",
            range_format="not_a_format",
        )


def test_human_dataframe_skips_rows_with_missing_required_fields():
    """Rows missing doc_id, quote, or construct are dropped; range may be NaN."""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        [
            {"doc": "a", "excerpt": "q1", "range": "0-10", "codes": "Anxiety"},
            {"doc": np.nan, "excerpt": "q2", "range": "0-10", "codes": "Anxiety"},
            {"doc": "b", "excerpt": "q3", "range": np.nan, "codes": "Anxiety"},
        ]
    )
    results = load_human_dataframe(
        df,
        doc_id_col="doc",
        quote_col="excerpt",
        range_col="range",
        construct_col="codes",
    )
    doc_ids = {r.document_id for r in results}
    # Row with missing doc_id is dropped; row with missing range is kept.
    assert doc_ids == {"a", "b"}


# ---------------------------------------------------------------------------
# validate_against_codebook
# ---------------------------------------------------------------------------


def _results_with(constructs_by_doc):
    """Helper: build AnalysisResult objects from {doc_id: [construct, ...]}."""
    out = []
    for doc_id, names in constructs_by_doc.items():
        instances = [ConstructInstance(construct=n, quote="q") for n in names]
        out.append(AnalysisResult(document_id=doc_id, instances=instances))
    return out


def test_validate_all_known(dedoose_codebook):
    """All constructs present in the codebook -> report.valid is True."""
    results = _results_with({"doc_a": ["Anxiety", "Depression"]})
    report = validate_against_codebook(results, dedoose_codebook)
    assert report.valid is True
    assert report.total_instances == 2
    assert report.total_unknown == 0
    assert report.unknown_constructs == {}


def test_validate_unknown_construct(dedoose_codebook):
    """Unknown constructs are recorded per-document without raising."""
    results = _results_with({"doc_a": ["Anxiety", "Mystery"], "doc_b": ["Depression"]})
    report = validate_against_codebook(results, dedoose_codebook)
    assert report.valid is False
    assert report.total_instances == 3
    assert report.total_unknown == 1
    assert report.unknown_constructs == {"doc_a": ["Mystery"]}


def test_validate_strict_raises(dedoose_codebook):
    """strict=True turns unknowns into a FileLoadError."""
    results = _results_with({"doc_a": ["Mystery"]})
    with pytest.raises(FileLoadError, match="validation failed"):
        validate_against_codebook(results, dedoose_codebook, strict=True)


def test_validate_accepts_loaded_codebook_dict(dedoose_codebook):
    """Pre-loaded codebook dicts are accepted, not just file paths."""
    codebook = load_codebook(dedoose_codebook)
    results = _results_with({"doc_a": ["Anxiety"]})
    report = validate_against_codebook(results, codebook)
    assert report.valid is True


def test_validate_accepts_single_result(dedoose_codebook):
    """A single AnalysisResult (not in a list) is accepted."""
    result = _results_with({"doc_a": ["Anxiety"]})[0]
    report = validate_against_codebook(result, dedoose_codebook)
    assert report.valid is True
    assert report.total_instances == 1


def test_validate_counts_duplicates(dedoose_codebook):
    """Duplicate unknown constructs within a document are all recorded."""
    results = _results_with({"doc_a": ["Mystery", "Mystery", "Anxiety"]})
    report = validate_against_codebook(results, dedoose_codebook)
    assert report.unknown_constructs == {"doc_a": ["Mystery", "Mystery"]}
    assert report.total_unknown == 2


def test_validate_report_str_is_readable(dedoose_codebook):
    """The report's __str__ produces a human-readable summary."""
    results = _results_with({"doc_a": ["Mystery"]})
    report = validate_against_codebook(results, dedoose_codebook)
    text = str(report)
    assert "FAIL" in text
    assert "Mystery" in text
    assert "doc_a" in text
