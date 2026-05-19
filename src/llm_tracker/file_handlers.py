"""File handling utilities for llm_tracker."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_tracker.models import (
    AnalysisResult,
    APIMetadata,
    ConstructInstance,
    ErrorRecord,
)


class FileLoadError(Exception):
    """Exception raised when a file cannot be loaded."""

    pass


def load_codebook(codebook_path: Path | str) -> dict:
    """Load and validate a codebook JSON file.

    Supports two formats:
    1. Dict format: {"construct_name": {"definition": "...", "examples": [...]}}
    2. List format: {"constructs": [{"name": "...", "definition": "..."}]}

    Args:
    ----
        codebook_path: Path to the codebook JSON file.

    Returns:
    -------
        Parsed codebook dictionary.

    Raises:
    ------
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
    ----
        file_path: Path to the TXT file.

    Returns:
    -------
        The text content of the file.

    Raises:
    ------
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
    ----
        file_path: Path to the document (CSV or TXT).

    Returns:
    -------
        Tuple of (document_text, document_id).
        Document ID is derived from the filename without extension.

    Raises:
    ------
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
    ----
        input_dir: Directory to search for documents.

    Returns:
    -------
        List of paths to document files (CSV and TXT).

    Raises:
    ------
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
    - errors/ subdirectory for failed documents

    Args:
    ----
        output_name: Optional name prefix for the output directory.
        base_dir: Optional base directory (defaults to current directory).

    Returns:
    -------
        Path to the created output directory.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if output_name:
        dir_name = f"{output_name}_{timestamp}"
    else:
        dir_name = f"llm_tracker_output_{timestamp}"

    if base_dir:
        output_dir = Path(base_dir) / dir_name
    else:
        output_dir = Path.cwd() / dir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "encodings").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "errors").mkdir(exist_ok=True)

    return output_dir


def save_analysis_result(result: AnalysisResult, output_dir: Path) -> Path:
    """Save an analysis result to a JSON file.

    Args:
    ----
        result: The analysis result to save.
        output_dir: The output directory (containing encodings/).

    Returns:
    -------
        Path to the saved JSON file.

    """
    file_path = output_dir / "encodings" / f"{result.document_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    return file_path


def save_metadata(metadata: APIMetadata, document_id: str, output_dir: Path) -> Path:
    """Save API metadata to a JSON file.

    Args:
    ----
        metadata: The API metadata to save.
        document_id: The document identifier.
        output_dir: The output directory (containing metadata/).

    Returns:
    -------
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
    ----
        output_dir: The output directory.
        model_name: Name of the model used.
        codebook_name: Name/path of the codebook used.
        input_dir_name: Name/path of the input directory.
        failed_documents: List of document names that failed.
        total_documents: Total number of documents processed.

    Returns:
    -------
        Path to the saved README file.

    """
    readme_path = output_dir / "README.md"

    content = f"""# LLM Tracker Analysis Results

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


def save_error_record(error: ErrorRecord, output_dir: Path) -> Path:
    """Save an error record to the errors directory."""
    from llm_tracker.models import ErrorRecord

    errors_dir = output_dir / "errors"
    errors_dir.mkdir(exist_ok=True)

    file_path = errors_dir / f"{error.document_id}_error.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(error.model_dump(), f, indent=2, ensure_ascii=False)

    return file_path


def _get_valid_constructs_from_codebook(codebook_data: dict) -> set[str]:
    """Extract valid construct names from a loaded codebook.

    Supports both codebook formats:
    - List format: {"constructs": [{"name": "...", ...}, ...]}
    - Dict format: {"construct_name": {"definition": "...", ...}, ...}
    """
    if "constructs" in codebook_data:
        return {c["name"].strip() for c in codebook_data["constructs"]}
    return {k.strip() for k in codebook_data.keys()}


def _parse_range(range_str: object, range_format: str) -> str | None:
    """Convert a character range string into the internal 'start:end' format.

    Args:
    ----
        range_str: The raw value from the range column.
        range_format: One of:
            - 'dash':  values like '858-1159' (e.g. Dedoose exports)
            - 'colon': values like '858:1159' (internal format; pass-through)

    Returns:
    -------
        String in 'start:end' format, or None if the value could not be parsed.
    """
    if range_str is None:
        return None

    raw = str(range_str).strip()
    if not raw:
        return None

    if range_format == "dash":
        separator = "-"
    elif range_format == "colon":
        separator = ":"
    else:
        raise ValueError(
            f"Unknown range_format '{range_format}'. Use 'dash' or 'colon'."
        )

    try:
        parts = raw.split(separator)
        if len(parts) == 2:
            return f"{int(parts[0])}:{int(parts[1])}"
    except (ValueError, AttributeError):
        pass
    return None


def _split_constructs(codes_str: object, separator: str) -> list[str]:
    """Split a cell of construct names on the given separator.

    Returns a list of stripped, non-empty construct names. No codebook
    validation is performed here — use ``validate_against_codebook`` on the
    resulting AnalysisResult objects if you want to check membership.
    """
    if codes_str is None:
        return []
    return [c.strip() for c in str(codes_str).split(separator) if c.strip()]


@dataclass
class ValidationReport:
    """Report from validating a set of AnalysisResult objects against a codebook.

    Attributes
    ----------
        valid: True if every construct on every instance is present in the codebook.
        unknown_constructs: Mapping from document_id -> list of construct names
            found on that document's instances that are NOT in the codebook.
            Duplicates within a document are preserved so frequency is visible.
        total_instances: Total number of ConstructInstance objects checked.
        total_unknown: Total count of instances whose construct name was unknown.
        known_constructs: The set of construct names extracted from the codebook,
            for reference.
    """

    valid: bool
    unknown_constructs: dict[str, list[str]]
    total_instances: int
    total_unknown: int
    known_constructs: set[str]

    def __str__(self) -> str:
        """Human-readable summary of the report."""
        lines = [
            "Codebook validation report",
            "--------------------------",
            f"Status          : {'PASS' if self.valid else 'FAIL'}",
            f"Total instances : {self.total_instances}",
            f"Unknown         : {self.total_unknown}",
            f"Codebook size   : {len(self.known_constructs)} constructs",
        ]
        if self.unknown_constructs:
            lines.append("")
            lines.append("Unknown constructs by document:")
            for doc_id, names in self.unknown_constructs.items():
                from collections import Counter

                counts = Counter(names)
                pretty = ", ".join(
                    f"{name!r} (x{n})" if n > 1 else repr(name)
                    for name, n in counts.most_common()
                )
                lines.append(f"  {doc_id}: {pretty}")
        return "\n".join(lines)


def load_human_dataframe(
    df: "pd.DataFrame",
    *,
    doc_id_col: str = "Media Title",
    quote_col: str = "Excerpt Copy",
    range_col: str | None = "Excerpt Range",
    construct_col: str = "Codes Applied Combined",
    range_format: str = "dash",
    construct_separator: str = ",",
) -> dict[str, AnalysisResult]:
    """Convert a human-coded DataFrame into AnalysisResult objects.

    This is a pure reshape operation — it does NOT validate construct names
    against any codebook. Run ``validate_against_codebook`` afterwards if
    you want to check codebook membership.

    The resulting AnalysisResult objects follow the same schema as LLM output,
    so they can be passed directly into the comparison pipeline.

    Default column names and formats match Dedoose exports so the common case
    is zero-configuration. Override any of them for other tools.

    Args:
    ----
        df: pandas DataFrame where each row is one human-coded excerpt.
        doc_id_col: Name of the column holding the document identifier.
            Default: "Media Title" (Dedoose).
        quote_col: Name of the column holding the excerpt text.
            Default: "Excerpt Copy" (Dedoose).
        range_col: Name of the column holding the character range, or None
            if the dataframe has no range column.
            Default: "Excerpt Range" (Dedoose).
        construct_col: Name of the column holding the construct name(s).
            Default: "Codes Applied Combined" (Dedoose).
        range_format: How to parse the range column. One of:
            - "dash":  values like "858-1159" (e.g. Dedoose)
            - "colon": values like "858:1159" (internal format)
            Ignored if range_col is None. Default: "dash".
        construct_separator: Delimiter used when a single row carries multiple
            constructs. Default: ",". Change this if your construct names
            contain commas.

    Returns:
    -------
        Dict mapping document_id to AnalysisResult. Each
        row with N constructs becomes N ConstructInstance entries. Rows
        missing any required field are skipped.

    Raises:
    ------
        FileLoadError: If pandas is not installed, the input is not a
            DataFrame, or required columns are missing.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise FileLoadError(
            "pandas is required to load human-coded dataframes. "
            "Install it with: pip install pandas"
        ) from e

    if not isinstance(df, pd.DataFrame):
        raise FileLoadError(
            "load_human_dataframe expected a pandas DataFrame as input."
        )

    required = [doc_id_col, quote_col, construct_col]
    if range_col is not None:
        required.append(range_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FileLoadError(
            f"DataFrame is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows where any required field (other than range) is missing.
    # We keep rows missing only the range — the instance gets quote_index=None.
    subset_for_dropna = [doc_id_col, quote_col, construct_col]
    cleaned = df.dropna(subset=subset_for_dropna)

    results_by_doc: dict[str, list[ConstructInstance]] = {}

    for _, row in cleaned.iterrows():
        doc_id = str(row[doc_id_col]).strip()
        quote = str(row[quote_col]).strip()

        if range_col is not None:
            quote_index = _parse_range(row[range_col], range_format)
        else:
            quote_index = None

        constructs = _split_constructs(row[construct_col], construct_separator)

        if doc_id not in results_by_doc:
            results_by_doc[doc_id] = []

        for construct in constructs:
            results_by_doc[doc_id].append(
                ConstructInstance(
                    construct=construct,
                    speaker_id=None,
                    quote=quote,
                    quote_index=quote_index,
                    confidence=None,  # not applicable for human codings
                )
            )

    return {
        doc_id: AnalysisResult(document_id=doc_id, instances=instances)
        for doc_id, instances in results_by_doc.items()
    }


_READERS: dict[str, str] = {
    ".csv": "read_csv",
    ".tsv": "read_csv",
    ".xlsx": "read_excel",
    ".xls": "read_excel",
}

# Tried in order when the user doesn't pass an explicit encoding.
# latin-1 is last because it can decode any byte sequence, guaranteeing
# we never crash on encoding — at worst we get some mojibake.
_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_with_encoding_fallback(reader, path: Path, *, is_text: bool, **read_kwargs):
    """Call a pandas reader, trying common encodings if none is specified.

    For binary readers (``read_excel``) encoding doesn't apply and this just
    calls the reader once. For text readers (``read_csv``), if the user did
    not pass an ``encoding`` kwarg, we try ``_ENCODING_FALLBACKS`` in order
    and emit a warning if we had to fall back past the first.
    """
    # Binary format (xlsx/xls) — encoding is not a thing.
    if not is_text:
        try:
            return reader(path, **read_kwargs)
        except Exception as e:
            raise FileLoadError(f"Could not read {path}: {e}") from e

    # User specified an encoding — respect it, don't second-guess.
    if "encoding" in read_kwargs:
        try:
            return reader(path, **read_kwargs)
        except Exception as e:
            raise FileLoadError(f"Could not read {path}: {e}") from e

    # No encoding specified — try fallbacks in order.
    last_error: Exception | None = None
    for i, enc in enumerate(_ENCODING_FALLBACKS):
        try:
            df = reader(path, encoding=enc, **read_kwargs)
            if i > 0:
                import warnings

                warnings.warn(
                    f"Could not read {path.name} as utf-8; "
                    f"fell back to {enc!r}. "
                    f"If the file should be utf-8, it may be corrupted; "
                    f"otherwise pass encoding={enc!r} explicitly to silence "
                    f"this warning.",
                    UserWarning,
                    stacklevel=3,
                )
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # Non-encoding read error — no point retrying with different encodings.
            raise FileLoadError(f"Could not read {path}: {e}") from e

    # latin-1 never raises UnicodeDecodeError, so in practice we never get
    # here — but be defensive.
    raise FileLoadError(
        f"Could not read {path} with any of the fallback encodings "
        f"{list(_ENCODING_FALLBACKS)}: {last_error}"
    ) from last_error


def save_human_results(
    results: dict[str, AnalysisResult],
    output_name: str,
    *,
    base_dir: Path | str | None = None,
    source_file: Path | str | None = None,
) -> Path:
    """Save human-coded AnalysisResult objects to disk.

    Writes one JSON file per document under ``<output_dir>/encodings/`` using
    the same layout that ``analyze_directory`` / ``analyze_csv`` produce, so
    the resulting directory can be passed directly to
    ``LLMTrackerComparer.compare_directories``.

    Args:
    ----
        results: Dict mapping document IDs to AnalysisResult objects.
        output_name: Name prefix for the output directory. A timestamp is
            appended automatically (matching the analyzer's behaviour).
        base_dir: Directory to create the output folder in. Defaults to CWD.
        source_file: Optional path to the original input file; recorded in
            the README if provided.

    Returns:
    -------
        Path to the created output directory.
    """
    output_path = create_output_directory(output_name=output_name, base_dir=base_dir)

    for result in results.values():
        save_analysis_result(result, output_path)

    # Minimal README so the directory is self-describing.
    readme_path = output_path / "README.md"
    n_instances = sum(len(r.instances) for r in results.values())
    source = source_file if source_file else "in-memory AnalysisResult objects"
    content = (
        f"# Human Codings\n\n"
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- **Source**: {source}\n"
        f"- **Documents**: {len(results)}\n"
        f"- **Total instances**: {n_instances}\n\n"
        f"## Structure\n\n"
        f"- `encodings/` - One JSON file per document, matching the schema "
        f"produced by `LLMTrackerAnalyzer.analyze_*`.\n"
    )
    readme_path.write_text(content, encoding="utf-8")

    return output_path


def load_human_coding(
    path: Path | str,
    *,
    doc_id_col: str = "Media Title",
    quote_col: str = "Excerpt Copy",
    range_col: str | None = "Excerpt Range",
    construct_col: str = "Codes Applied Combined",
    range_format: str = "dash",
    construct_separator: str = ",",
    save_dir: str | None = None,
    **read_kwargs,
) -> dict[str, AnalysisResult]:
    """Load human-coded data from a CSV or xlsx file.

    Dispatches on the file extension (.csv, .tsv, .xlsx, .xls), reads the file
    into a DataFrame with pandas, then passes it to ``load_human_dataframe``.
    Extra ``**read_kwargs`` are forwarded to the underlying pandas reader so
    you can override encoding, delimiters, sheet names, etc.

    Default column names match Dedoose exports, so most Dedoose users can
    call this with just ``load_human_coding("export.csv")``. Override the
    column name kwargs for other tools.

    No codebook validation is performed here — run ``validate_against_codebook``
    on the result if you want to check construct membership.

    Args:
    ----
        path: Path to the input file. Extension determines the reader:
            - .csv, .tsv  -> pandas.read_csv  (.tsv gets sep='\\t' by default)
            - .xlsx, .xls -> pandas.read_excel
        doc_id_col: See ``load_human_dataframe``.
        quote_col: See ``load_human_dataframe``.
        range_col: See ``load_human_dataframe``.
        construct_col: See ``load_human_dataframe``.
        range_format: See ``load_human_dataframe``.
        construct_separator: See ``load_human_dataframe``.
        save_dir: Optional name prefix for persisting the loaded codings to
            disk, matching the layout produced by ``analyze_directory`` /
            ``analyze_csv``. A timestamp is appended automatically. When set,
            the path to the created directory is printed so it can be passed
            to ``LLMTrackerComparer.compare_directories`` if desired.
        **read_kwargs: Forwarded to ``pd.read_csv`` or ``pd.read_excel``.
            Use this for things like ``encoding='latin-1'`` on weird CSVs or
            ``sheet_name='Sheet2'`` on multi-sheet xlsx files.

            For CSV/TSV reads, if no ``encoding`` is specified, the loader
            tries utf-8, utf-8-sig, cp1252, and latin-1 in order and emits
            a warning if it has to fall back past utf-8. Pass ``encoding=...``
            explicitly to skip the fallback.

    Returns:
    -------
        Dict mapping document IDs to AnalysisResult objects.

    Raises:
    ------
        FileLoadError: If the file doesn't exist, the extension is unsupported,
            the file can't be read, or required columns are missing.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise FileLoadError(
            "pandas is required to load human coding files. "
            "Install it with: pip install pandas"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileLoadError(f"Human coding file not found: {path}")

    suffix = path.suffix.lower()
    reader_name = _READERS.get(suffix)
    if reader_name is None:
        raise FileLoadError(
            f"Unsupported file extension '{suffix}'. Supported: {sorted(_READERS)}"
        )

    if reader_name == "read_excel":
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise FileLoadError(
                "openpyxl is required to read .xlsx files. "
                "Install it with: pip install openpyxl"
            ) from e

    # For .tsv default to tab separator unless the caller overrides it.
    if suffix == ".tsv" and "sep" not in read_kwargs:
        read_kwargs["sep"] = "\t"

    reader = getattr(pd, reader_name)
    df = _read_with_encoding_fallback(
        reader, path, is_text=(reader_name == "read_csv"), **read_kwargs
    )

    results = load_human_dataframe(
        df,
        doc_id_col=doc_id_col,
        quote_col=quote_col,
        range_col=range_col,
        construct_col=construct_col,
        range_format=range_format,
        construct_separator=construct_separator,
    )

    if save_dir is not None:
        output_path = save_human_results(
            results, output_name=save_dir, source_file=path
        )
        print(f"Human codings saved to: {output_path}")

    return results


def validate_against_codebook(
    results: "dict[str, AnalysisResult] | AnalysisResult",
    codebook: "dict | Path | str",
    *,
    strict: bool = False,
) -> ValidationReport:
    """Check that every construct on every instance appears in the codebook.

    This is decoupled from the loaders so you can load data once and validate
    against different codebooks, or skip validation entirely.

    Args:
    ----
        results: A single AnalysisResult or a dict mapping document IDs to
            AnalysisResult objects.
        codebook: Either an already-loaded codebook dict or a path to a
            codebook JSON file.
        strict: If True, raise FileLoadError when any unknown construct is
            found. If False (default), return a report and leave the
            decision to the caller.

    Returns:
    -------
        A ValidationReport describing any mismatches.

    Raises:
    ------
        FileLoadError: If ``strict=True`` and unknown constructs are found,
            or if the codebook path cannot be loaded.
    """
    if isinstance(results, AnalysisResult):
        result_items = [results]
    else:
        result_items = results.values()

    if isinstance(codebook, (str, Path)):
        codebook_data = load_codebook(codebook)
    else:
        codebook_data = codebook

    known = _get_valid_constructs_from_codebook(codebook_data)

    unknown_by_doc: dict[str, list[str]] = {}
    total_instances = 0
    total_unknown = 0

    for result in result_items:
        for instance in result.instances:
            total_instances += 1
            name = instance.construct.strip()
            if name not in known:
                unknown_by_doc.setdefault(result.document_id, []).append(name)
                total_unknown += 1

    report = ValidationReport(
        valid=total_unknown == 0,
        unknown_constructs=unknown_by_doc,
        total_instances=total_instances,
        total_unknown=total_unknown,
        known_constructs=known,
    )

    if strict and not report.valid:
        raise FileLoadError(
            f"Codebook validation failed: {total_unknown} instance(s) "
            f"across {len(unknown_by_doc)} document(s) have constructs "
            f"not in the codebook.\n\n{report}"
        )

    return report


def load_error_records(output_dir: Path) -> list[ErrorRecord]:
    """Load all error records from an output directory."""
    from llm_tracker.models import ErrorRecord

    errors_dir = output_dir / "errors"

    if not errors_dir.exists():
        return []

    errors = []
    for error_file in errors_dir.glob("*_error.json"):
        with open(error_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            errors.append(ErrorRecord(**data))

    return errors
