"""File handling utilities for pychometrics."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pychometrics.models import AnalysisResult, APIMetadata, Codebook


class FileLoadError(Exception):
    """Exception raised when a file cannot be loaded."""

    pass


def load_codebook(codebook_path: Path | str) -> dict:
    """Load and validate a codebook JSON file.

    Supports two formats:
    1. Dict format: {"construct_name": {"definition": "...", "examples": [...]}}
    2. List format: {"constructs": [{"name": "...", "definition": "..."}]}

    Args:
        codebook_path: Path to the codebook JSON file.

    Returns:
        Parsed codebook dictionary.

    Raises:
        FileLoadError: If the file cannot be loaded or is invalid.
    """
    path = Path(codebook_path)

    if not path.exists():
        raise FileLoadError(f"Codebook file not found: {path}")

    if path.suffix.lower() != ".json":
        raise FileLoadError(f"Codebook must be a JSON file, got: {path.suffix}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            codebook_data = json.load(f)
    except json.JSONDecodeError as e:
        raise FileLoadError(f"Invalid JSON in codebook: {e}") from e
    except IOError as e:
        raise FileLoadError(f"Could not read codebook file: {e}") from e

    return codebook_data


def load_txt_document(file_path: Path) -> str:
    """Load a plain text document.

    Args:
        file_path: Path to the TXT file.

    Returns:
        The text content of the file.

    Raises:
        FileLoadError: If the file cannot be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        raise FileLoadError(f"Could not read TXT file: {e}") from e


def load_csv_document(file_path: Path) -> str:
    """Load and extract text from a CSV document.

    This function handles various CSV formats:
    - Single column: Uses all values
    - Multiple columns: Concatenates all text values
    - Columns named 'text', 'content', 'transcript': Prioritized
    - Columns named 'speaker', 'interviewer', etc.: Used as speaker IDs

    Args:
        file_path: Path to the CSV file.

    Returns:
        Extracted text content from the CSV.

    Raises:
        FileLoadError: If the file cannot be read or parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(8192)
            f.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)

            if not reader.fieldnames:
                raise FileLoadError("CSV file has no headers")

            text_columns = []
            speaker_column = None
            fieldnames_lower = {fn.lower(): fn for fn in reader.fieldnames}

            text_priority = ["text", "content", "transcript", "response", "answer"]
            for col_name in text_priority:
                if col_name in fieldnames_lower:
                    text_columns.append(fieldnames_lower[col_name])

            if not text_columns:
                speaker_keywords = [
                    "speaker",
                    "interviewer",
                    "participant",
                    "id",
                    "name",
                ]
                text_columns = [
                    fn
                    for fn in reader.fieldnames
                    if not any(kw in fn.lower() for kw in speaker_keywords)
                ]

            speaker_keywords = ["speaker", "participant", "interviewer", "name"]
            for kw in speaker_keywords:
                if kw in fieldnames_lower:
                    speaker_column = fieldnames_lower[kw]
                    break

            lines = []
            for row in reader:
                line_parts = []

                if speaker_column and row.get(speaker_column):
                    line_parts.append(f"{row[speaker_column]}:")

                for col in text_columns:
                    if col in row and row[col]:
                        line_parts.append(str(row[col]).strip())

                if line_parts:
                    lines.append(" ".join(line_parts))

            return "\n".join(lines)

    except IOError as e:
        raise FileLoadError(f"Could not read CSV file: {e}") from e
    except csv.Error as e:
        raise FileLoadError(f"Could not parse CSV file: {e}") from e


def load_document(file_path: Path | str) -> tuple[str, str]:
    """Load a document file and extract its text content.

    Args:
        file_path: Path to the document (CSV or TXT).

    Returns:
        Tuple of (document_text, document_id).
        Document ID is derived from the filename without extension.

    Raises:
        FileLoadError: If the file type is unsupported or cannot be loaded.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileLoadError(f"Document file not found: {path}")

    suffix = path.suffix.lower()
    document_id = path.stem

    if suffix == ".txt":
        text = load_txt_document(path)
    elif suffix == ".csv":
        text = load_csv_document(path)
    else:
        raise FileLoadError(f"Unsupported file type: {suffix}. Supported: .txt, .csv")

    return text, document_id


def get_document_files(input_dir: Path | str) -> list[Path]:
    """Get all document files from a directory.

    Args:
        input_dir: Directory to search for documents.

    Returns:
        List of paths to document files (CSV and TXT).

    Raises:
        FileLoadError: If the directory doesn't exist or is empty.
    """
    path = Path(input_dir)

    if not path.exists():
        raise FileLoadError(f"Input directory not found: {path}")

    if not path.is_dir():
        raise FileLoadError(f"Input path is not a directory: {path}")

    supported_extensions = {".txt", ".csv"}
    files = [
        f
        for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files:
        raise FileLoadError(
            f"No document files found in {path}. "
            f"Supported formats: {supported_extensions}"
        )

    return sorted(files)


def create_output_directory(
    output_name: Optional[str] = None, base_dir: Optional[Path | str] = None
) -> Path:
    """Create the output directory structure.

    Creates a directory with timestamp in the name containing:
    - encodings/ subdirectory for JSON results
    - metadata/ subdirectory for API metadata

    Args:
        output_name: Optional name prefix for the output directory.
        base_dir: Optional base directory (defaults to current directory).

    Returns:
        Path to the created output directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if output_name:
        dir_name = f"{output_name}_{timestamp}"
    else:
        dir_name = f"pychometrics_output_{timestamp}"

    if base_dir:
        output_dir = Path(base_dir) / dir_name
    else:
        output_dir = Path.cwd() / dir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "encodings").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    return output_dir


def save_analysis_result(result: AnalysisResult, output_dir: Path) -> Path:
    """Save an analysis result to a JSON file.

    Args:
        result: The analysis result to save.
        output_dir: The output directory (containing encodings/).

    Returns:
        Path to the saved JSON file.
    """
    file_path = output_dir / "encodings" / f"{result.document_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return file_path


def save_metadata(metadata: APIMetadata, document_id: str, output_dir: Path) -> Path:
    """Save API metadata to a JSON file.

    Args:
        metadata: The API metadata to save.
        document_id: The document identifier.
        output_dir: The output directory (containing metadata/).

    Returns:
        Path to the saved metadata file.
    """
    file_path = output_dir / "metadata" / f"{document_id}_meta.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

    return file_path


def save_readme(
    output_dir: Path,
    model_name: str,
    codebook_name: str,
    input_dir_name: str,
    failed_documents: list[str],
    total_documents: int,
) -> Path:
    """Save the analysis README file.

    Args:
        output_dir: The output directory.
        model_name: Name of the model used.
        codebook_name: Name/path of the codebook used.
        input_dir_name: Name/path of the input directory.
        failed_documents: List of document names that failed.
        total_documents: Total number of documents processed.

    Returns:
        Path to the saved README file.
    """
    readme_path = output_dir / "README.md"

    content = f"""# Pychometrics Analysis Results

## Analysis Information

- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model Used**: {model_name}
- **Codebook**: {codebook_name}
- **Input Directory**: {input_dir_name}

## Processing Summary

- **Total Documents**: {total_documents}
- **Successful**: {total_documents - len(failed_documents)}
- **Failed**: {len(failed_documents)}

"""

    if failed_documents:
        content += """## Failed Documents

The following documents failed to process after retry:

"""
        for doc_name in failed_documents:
            content += f"- {doc_name}\n"
    else:
        content += """## Status

All documents processed successfully.
"""

    content += """
## Output Structure

- `encodings/` - JSON files with construct instances for each document
- `metadata/` - API response metadata for each document
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    return readme_path
