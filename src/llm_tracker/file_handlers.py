"""File handling utilities for llm_tracker."""

import csv
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from llm_tracker.models import (
    AnalysisResult,
    APIMetadata,
    ConstructInstance,
    ErrorRecord,
)

TEXT_COLUMN_PRIORITY = ["text", "content", "transcript", "response", "answer"]
SPEAKER_COLUMN_KEYWORDS = ["speaker", "participant", "interviewer", "name"]
NON_TEXT_COLUMN_KEYWORDS = ["speaker", "interviewer", "participant", "id", "name"]


class FileLoadError(Exception):
    """Exception raised when a file cannot be loaded."""


def load_codebook(codebook_path: Path | str) -> dict:
    """Load a codebook JSON file.

    Args:
    ----
        codebook_path: Path to the codebook JSON file.

    Returns:
    -------
        Parsed codebook dictionary.

    Raises:
    ------
        FileLoadError: If the file is missing, is not JSON, or cannot be read.

    """
    path = Path(codebook_path)

    if not path.exists():
        raise FileLoadError(f"Codebook file not found: {path}")

    if path.suffix.lower() != ".json":
        raise FileLoadError(f"Codebook must be a JSON file, got: {path.suffix}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise FileLoadError(f"Invalid JSON in codebook: {e}") from e
    except OSError as e:
        raise FileLoadError(f"Could not read codebook file: {e}") from e


def load_txt_document(file_path: Path) -> str:
    """Load a plain text document.

    Args:
    ----
        file_path: Path to the text file.

    Returns:
    -------
        Text content from the file.

    Raises:
    ------
        FileLoadError: If the file cannot be read.

    """
    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise FileLoadError(f"Could not read TXT file: {e}") from e


def _field_lookup(fieldnames: Sequence[str]) -> dict[str, str]:
    """Map lowercase field names to their original spelling."""
    return {field.lower(): field for field in fieldnames}


def _select_text_columns(
    fieldnames: Sequence[str],
    fields_by_lower_name: dict[str, str],
) -> list[str]:
    """Choose CSV columns that should contribute document text.

    Args:
    ----
        fieldnames: Original CSV column names.
        fields_by_lower_name: Mapping of lowercase column names to their
            original spelling.

    Returns:
    -------
        Preferred text columns when present, otherwise columns that do not look
        like speaker or identifier metadata.

    """
    preferred_columns = [
        fields_by_lower_name[name]
        for name in TEXT_COLUMN_PRIORITY
        if name in fields_by_lower_name
    ]
    if preferred_columns:
        return preferred_columns

    return [
        field
        for field in fieldnames
        if not any(keyword in field.lower() for keyword in NON_TEXT_COLUMN_KEYWORDS)
    ]


def _select_speaker_column(fields_by_lower_name: dict[str, str]) -> str | None:
    """Choose the CSV column containing speaker identifiers, if present.

    Args:
    ----
        fields_by_lower_name: Mapping of lowercase column names to their
            original spelling.

    Returns:
    -------
        Original column name for the speaker identifier, or None.

    """
    for keyword in SPEAKER_COLUMN_KEYWORDS:
        if keyword in fields_by_lower_name:
            return fields_by_lower_name[keyword]
    return None


def _csv_row_to_text(
    row: dict[str, str],
    text_columns: list[str],
    speaker_column: str | None,
) -> str | None:
    """Convert one CSV row into a text line."""
    line_parts = []

    if speaker_column and row.get(speaker_column):
        line_parts.append(f"{row[speaker_column]}:")

    for column in text_columns:
        value = row.get(column)
        if value:
            line_parts.append(str(value).strip())

    if not line_parts:
        return None
    return " ".join(line_parts)


def load_csv_document(file_path: Path) -> str:
    """Load and extract text from a CSV document.

    Args:
    ----
        file_path: Path to the CSV file.

    Returns:
    -------
        Extracted text content from the CSV.

    Raises:
    ------
        FileLoadError: If the file cannot be read or parsed.

    """
    try:
        with file_path.open("r", encoding="utf-8", newline="") as file:
            sample = file.read(8192)
            file.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(file, dialect=dialect)

            if not reader.fieldnames:
                raise FileLoadError("CSV file has no headers")

            fields_by_lower_name = _field_lookup(reader.fieldnames)
            text_columns = _select_text_columns(
                reader.fieldnames,
                fields_by_lower_name,
            )
            speaker_column = _select_speaker_column(fields_by_lower_name)
            lines = [
                line
                for row in reader
                if (line := _csv_row_to_text(row, text_columns, speaker_column))
            ]
            return "\n".join(lines)

    except OSError as e:
        raise FileLoadError(f"Could not read CSV file: {e}") from e
    except csv.Error as e:
        raise FileLoadError(f"Could not parse CSV file: {e}") from e


def load_document(file_path: Path | str) -> tuple[str, str]:
    """Load a supported document and return its text and identifier.

    Args:
    ----
        file_path: Path to a text or CSV document.

    Returns:
    -------
        Document text and document ID. The document ID is the file stem.

    Raises:
    ------
        FileLoadError: If the file is missing, unsupported, or cannot be read.

    """
    path = Path(file_path)

    if not path.exists():
        raise FileLoadError(f"Document file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = load_txt_document(path)
    elif suffix == ".csv":
        text = load_csv_document(path)
    else:
        raise FileLoadError(f"Unsupported file type: {suffix}. Supported: .txt, .csv")

    return text, path.stem


def get_document_files(input_dir: Path | str) -> list[Path]:
    """Get supported document files from a directory.

    Args:
    ----
        input_dir: Directory to search.

    Returns:
    -------
        Sorted list of supported document paths.

    Raises:
    ------
        FileLoadError: If the directory is missing, invalid, or empty.

    """
    path = Path(input_dir)

    if not path.exists():
        raise FileLoadError(f"Input directory not found: {path}")

    if not path.is_dir():
        raise FileLoadError(f"Input path is not a directory: {path}")

    supported_extensions = {".txt", ".csv"}
    document_files = [
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in supported_extensions
    ]

    if not document_files:
        raise FileLoadError(
            f"No document files found in {path}. "
            f"Supported formats: {supported_extensions}"
        )

    return sorted(document_files)


def create_output_directory(
    output_name: str | None = None,
    base_dir: Path | str | None = None,
) -> Path:
    """Create a timestamped analyzer output directory.

    Args:
    ----
        output_name: Optional base name for the output directory.
        base_dir: Optional parent directory. Defaults to the current directory.

    Returns:
    -------
        Created output directory path.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if output_name:
        directory_name = f"{output_name}_{timestamp}"
    else:
        directory_name = f"llm_tracker_output_{timestamp}"

    if base_dir:
        output_dir = Path(base_dir) / directory_name
    else:
        output_dir = Path.cwd() / directory_name

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory in ["encodings", "metadata", "errors"]:
        (output_dir / subdirectory).mkdir(exist_ok=True)

    return output_dir


def save_analysis_result(result: AnalysisResult, output_dir: Path) -> Path:
    """Save one analysis result as JSON.

    Args:
    ----
        result: Analysis result to save.
        output_dir: Analyzer output directory.

    Returns:
    -------
        Path to the saved JSON file.

    """
    file_path = output_dir / "encodings" / f"{result.document_id}.json"
    file_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return file_path


def save_metadata(metadata: APIMetadata, document_id: str, output_dir: Path) -> Path:
    """Save API metadata as JSON.

    Args:
    ----
        metadata: API metadata to save.
        document_id: Document identifier for the metadata file name.
        output_dir: Analyzer output directory.

    Returns:
    -------
        Path to the saved metadata file.

    """
    file_path = output_dir / "metadata" / f"{document_id}_meta.json"
    file_path.write_text(
        json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return file_path


def save_run_summary(
    output_dir: Path,
    model_name: str,
    codebook_name: str,
    input_dir_name: str,
    failed_documents: list[str],
    total_documents: int,
) -> Path:
    """Save a human readable summary for an analysis run.

    Args:
    ----
        output_dir: Analyzer output directory.
        model_name: Name of the model used.
        codebook_name: Name or path of the codebook used.
        input_dir_name: Name or path of the input data.
        failed_documents: Document names that failed.
        total_documents: Total number of documents processed.

    Returns:
    -------
        Path to the saved run summary file.

    """
    readme_path = output_dir / "README.md"
    successful_documents = total_documents - len(failed_documents)
    failed_section = "## Status\n\nAll documents processed successfully.\n"

    if failed_documents:
        failed_lines = "\n".join(f"- {name}" for name in failed_documents)
        failed_section = (
            "## Failed Documents\n\n"
            "The following documents failed to process after retry:\n\n"
            f"{failed_lines}\n"
        )

    content = f"""# LLM Tracker Analysis Results

## Analysis Information

- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model Used**: {model_name}
- **Codebook**: {codebook_name}
- **Input**: {input_dir_name}

## Processing Summary

- **Total Documents**: {total_documents}
- **Successful**: {successful_documents}
- **Failed**: {len(failed_documents)}

{failed_section}

## Output Structure

- `encodings/` - JSON files with construct instances for each document
- `metadata/` - API response metadata for each document
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def save_error_record(error: ErrorRecord, output_dir: Path) -> Path:
    """Save an error record as JSON.

    Args:
    ----
        error: Error record to save.
        output_dir: Analyzer output directory.

    Returns:
    -------
        Path to the saved error file.

    """
    errors_dir = output_dir / "errors"
    errors_dir.mkdir(exist_ok=True)

    file_path = errors_dir / f"{error.document_id}_error.json"
    file_path.write_text(
        json.dumps(error.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return file_path


def _parse_range(range_value: object, range_format: str) -> str | None:
    """Convert a character range value into start:end format."""
    if range_value is None:
        return None

    raw_value = str(range_value).strip()
    if not raw_value:
        return None

    separators = {"dash": "-", "colon": ":"}
    separator = separators.get(range_format)
    if separator is None:
        raise ValueError(
            f"Unknown range_format '{range_format}'. Use 'dash' or 'colon'."
        )

    parts = raw_value.split(separator)
    if len(parts) != 2:
        return None

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        return None

    return f"{start}:{end}"


def _split_constructs(codes_value: object, separator: str) -> list[str]:
    """Split a cell of construct names into cleaned construct names."""
    if codes_value is None:
        return []
    return [
        construct.strip()
        for construct in str(codes_value).split(separator)
        if construct.strip()
    ]


def load_human_dataframe(
    df: pd.DataFrame,
    *,
    doc_id_col: str = "Media Title",
    quote_col: str = "Excerpt Copy",
    range_col: str | None = "Excerpt Range",
    construct_col: str = "Codes Applied Combined",
    range_format: str = "dash",
    construct_separator: str = ",",
) -> dict[str, AnalysisResult]:
    """Convert a human coded DataFrame into AnalysisResult objects.

    Args:
    ----
        df: DataFrame where each row is one human coded excerpt.
        doc_id_col: Column containing the document identifier.
        quote_col: Column containing the coded excerpt text.
        range_col: Column containing the character range, or None.
        construct_col: Column containing construct names.
        range_format: Range format. Supported values are dash and colon.
        construct_separator: Separator for rows with multiple construct names.

    Returns:
    -------
        Results keyed by document ID.

    Raises:
    ------
        FileLoadError: If input is not a DataFrame or required columns are
            missing.

    """
    if not isinstance(df, pd.DataFrame):
        raise FileLoadError(
            "load_human_dataframe expected a pandas DataFrame as input."
        )

    required_columns = [doc_id_col, quote_col, construct_col]
    if range_col is not None:
        required_columns.append(range_col)

    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise FileLoadError(
            f"DataFrame is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    rows_with_required_values = df.dropna(subset=[doc_id_col, quote_col, construct_col])
    instances_by_doc: dict[str, list[ConstructInstance]] = {}

    for _, row in rows_with_required_values.iterrows():
        document_id = str(row[doc_id_col]).strip()
        quote = str(row[quote_col]).strip()
        quote_index = (
            _parse_range(row[range_col], range_format)
            if range_col is not None
            else None
        )
        constructs = _split_constructs(row[construct_col], construct_separator)

        if document_id not in instances_by_doc:
            instances_by_doc[document_id] = []

        for construct in constructs:
            instances_by_doc[document_id].append(
                ConstructInstance(
                    construct=construct,
                    speaker_id=None,
                    quote=quote,
                    quote_index=quote_index,
                    confidence=None,
                )
            )

    return {
        document_id: AnalysisResult(document_id=document_id, instances=instances)
        for document_id, instances in instances_by_doc.items()
    }


def load_human_coding(
    path: Path | str,
    *,
    doc_id_col: str = "Media Title",
    quote_col: str = "Excerpt Copy",
    range_col: str | None = "Excerpt Range",
    construct_col: str = "Codes Applied Combined",
    range_format: str = "dash",
    construct_separator: str = ",",
    **read_kwargs,
) -> dict[str, AnalysisResult]:
    """Load human coded data from a CSV, TSV, XLSX, or XLS file.

    Args:
    ----
        path: Path to the human coding file.
        doc_id_col: Column containing the document identifier.
        quote_col: Column containing the coded excerpt text.
        range_col: Column containing the character range, or None.
        construct_col: Column containing construct names.
        range_format: Range format. Supported values are dash and colon.
        construct_separator: Separator for rows with multiple construct names.
        **read_kwargs: Additional keyword arguments passed to pandas.

    Returns:
    -------
        Human coding results keyed by document ID.

    Raises:
    ------
        FileLoadError: If the file is missing, unsupported, or cannot be read.

    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileLoadError(f"Human coding file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".xlsx", ".xls"}:
        raise FileLoadError(
            f"Unsupported file extension '{suffix}'. "
            "Supported: ['.csv', '.tsv', '.xlsx', '.xls']"
        )

    try:
        if suffix == ".csv":
            df = pd.read_csv(input_path, **read_kwargs)
        elif suffix == ".tsv":
            df = pd.read_csv(input_path, sep="\t", **read_kwargs)
        else:
            df = pd.read_excel(input_path, **read_kwargs)
    except Exception as e:
        raise FileLoadError(f"Could not read {input_path}: {e}") from e

    return load_human_dataframe(
        df,
        doc_id_col=doc_id_col,
        quote_col=quote_col,
        range_col=range_col,
        construct_col=construct_col,
        range_format=range_format,
        construct_separator=construct_separator,
    )


def load_error_records(output_dir: Path | str) -> list[ErrorRecord]:
    """Load saved error records from an analyzer output directory.

    Args:
    ----
        output_dir: Analyzer output directory.

    Returns:
    -------
        Saved error records, sorted by file name.

    """
    errors_dir = Path(output_dir) / "errors"
    if not errors_dir.exists():
        return []

    errors = []
    for error_file in sorted(errors_dir.glob("*_error.json")):
        data = json.loads(error_file.read_text(encoding="utf-8"))
        errors.append(ErrorRecord(**data))

    return errors
