"""Main analyzer module for llm_tracker."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from llm_tracker import file_handlers
from llm_tracker.config import AnalyzerConfig
from llm_tracker.file_handlers import (
    FileLoadError,
    codebook_constructs,
    create_output_directory,
    get_document_files,
    load_codebook,
    load_document,
    load_error_records,
    save_analysis_result,
    save_error_record,
    save_metadata,
    save_run_summary,
)
from llm_tracker.models import AnalysisResult, APIMetadata, ErrorRecord
from llm_tracker.prompting import PromptingError, prompt_for_constructs


def _save_successful_result(
    result: AnalysisResult,
    metadata: APIMetadata,
    output_path: Path,
    results_by_doc: dict[str, AnalysisResult],
    metadata_by_doc: dict[str, dict],
) -> None:
    """Save a successful analysis result and update return dictionaries.

    Args:
    ----
        result: Parsed analysis result for one document.
        metadata: API metadata returned with the result.
        output_path: Analyzer output directory.
        results_by_doc: Results dictionary to update in place.
        metadata_by_doc: Metadata dictionary to update in place.

    Returns:
    -------
        None.

    """
    save_analysis_result(result, output_path)
    save_metadata(metadata, result.document_id, output_path)
    results_by_doc[result.document_id] = result
    metadata_by_doc[result.document_id] = metadata.model_dump()


def _metadata_from_error(error: Exception, config: AnalyzerConfig) -> APIMetadata:
    """Create metadata for a failed document.

    Args:
    ----
        error: Error raised while loading or prompting a document.
        config: Analyzer configuration used for the failed attempt.

    Returns:
    -------
        Metadata from the prompting error when available, otherwise a minimal
        metadata record describing the failure.

    """
    if isinstance(error, PromptingError) and error.metadata is not None:
        return error.metadata

    return APIMetadata(
        model=config.model_name,
        num_retries=config.max_retries,
        error_message=str(error),
        error_type=type(error).__name__,
        error_output=str(error),
    )


def _save_processing_error(
    document_id: str,
    document_path: Path | str,
    error: Exception,
    output_path: Path,
    config: AnalyzerConfig,
) -> ErrorRecord:
    """Save error details for a failed document.

    Args:
    ----
        document_id: Identifier for the failed document.
        document_path: Source path for the failed document.
        error: Error raised while loading or prompting the document.
        output_path: Analyzer output directory.
        config: Analyzer configuration used for the failed attempt.

    Returns:
    -------
        Saved ErrorRecord for the failed document.

    """
    error_record = ErrorRecord(
        document_id=document_id,
        document_path=str(document_path),
        error_message=str(error),
        model_used=config.model_name,
        timestamp=datetime.now().isoformat(),
    )
    save_error_record(error_record, output_path)
    save_metadata(_metadata_from_error(error, config), document_id, output_path)
    return error_record


class LLMTrackerAnalyzer:
    """Analyzer for identifying psychological constructs in text documents."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        custom_prompt: str | None = None,
        fuzzy_quote_matching: bool = False,
        quote_match_threshold: float = 0.85,
        config: AnalyzerConfig | None = None,
    ) -> None:
        """Create an analyzer.

        Args:
        ----
            api_key: OpenRouter API key or path to an env file containing it.
            model_name: OpenRouter model name for coding documents.
            custom_prompt: Optional prompt template for document coding.
            fuzzy_quote_matching: Whether to use fuzzy quote index recovery.
            quote_match_threshold: Minimum fuzzy match score for quote recovery.
            config: Optional AnalyzerConfig. When provided, other configuration
                arguments are ignored.

        """
        if config is not None:
            self.config = config
            return

        config_kwargs = {
            "api_key": api_key,
            "fuzzy_quote_matching": fuzzy_quote_matching,
            "quote_match_threshold": quote_match_threshold,
        }
        if model_name is not None:
            config_kwargs["model_name"] = model_name
        if custom_prompt is not None:
            config_kwargs["custom_prompt"] = custom_prompt

        self.config = AnalyzerConfig(**config_kwargs)

    def analyze_document(
        self, document_path: Path | str, codebook: dict
    ) -> tuple[AnalysisResult, APIMetadata]:
        """Analyze one document for codebook constructs.

        Args:
        ----
            document_path: Path to a supported text or CSV document.
            codebook: Loaded codebook dictionary.

        Returns:
        -------
            AnalysisResult and API metadata for the document.

        Raises:
        ------
            FileLoadError: If the document cannot be loaded.
            PromptingError: If the LLM request or response parsing fails.

        """
        path = Path(document_path)
        text, document_id = load_document(path)
        return prompt_for_constructs(
            text=text,
            codebook=codebook,
            document_id=document_id,
            config=self.config,
        )

    def analyze_directory(
        self,
        input_dir: Path | str,
        codebook_path: Path | str,
        output_dir: str | None = None,
    ) -> tuple[dict[str, AnalysisResult], dict[str, dict], list[ErrorRecord]]:
        """Analyze every supported document in a directory.

        Args:
        ----
            input_dir: Directory containing supported document files.
            codebook_path: Path to the codebook JSON file.
            output_dir: Optional base name for the analyzer output directory.

        Returns:
        -------
            A tuple containing results by document ID, metadata by document ID,
            and error records for failed documents.

        Raises:
        ------
            FileLoadError: If the codebook or input directory cannot be loaded.

        """
        input_path = Path(input_dir)
        codebook_file = Path(codebook_path)
        codebook = codebook_constructs(load_codebook(codebook_file))
        document_paths = get_document_files(input_path)
        output_path = create_output_directory(
            output_name=output_dir, base_dir=Path.cwd()
        )

        results_by_doc: dict[str, AnalysisResult] = {}
        metadata_by_doc: dict[str, dict] = {}
        errors: list[ErrorRecord] = []
        total_documents = len(document_paths)
        success_count = 0

        for position, document_path in enumerate(document_paths, start=1):
            print(f"Processing [{position}/{total_documents}]: {document_path.name}")

            try:
                result, metadata = self.analyze_document(document_path, codebook)
                _save_successful_result(
                    result,
                    metadata,
                    output_path,
                    results_by_doc,
                    metadata_by_doc,
                )
                success_count += 1
                print(f"  OK: Found {len(result.instances)} construct instances")

            except (FileLoadError, PromptingError) as error:
                print(f"  Failed: {error}")
                error_record = _save_processing_error(
                    document_id=document_path.stem,
                    document_path=document_path,
                    error=error,
                    output_path=output_path,
                    config=self.config,
                )
                errors.append(error_record)

            finally:
                print(
                    "  Progress: "
                    f"Successful {success_count}/{total_documents}; "
                    f"Errors {len(errors)}/{total_documents}"
                )

        save_run_summary(
            output_dir=output_path,
            model_name=self.config.model_name,
            codebook_name=codebook_file.name,
            input_dir_name=input_path.name,
            failed_documents=[error.document_id for error in errors],
            total_documents=total_documents,
        )

        print("\nAnalysis complete!")
        print(f"  Output directory: {output_path}")
        print(f"  Successful: {success_count}/{total_documents}")
        if errors:
            print(f"  Failed: {len(errors)} (see errors directory)")

        return results_by_doc, metadata_by_doc, errors

    def analyze_csv(
        self,
        csv_path: Path | str,
        codebook_path: Path | str,
        text_column: str,
        id_column: str | None = None,
        output_dir: str | None = None,
    ) -> tuple[dict[str, AnalysisResult], dict[str, dict], list[ErrorRecord]]:
        """Analyze each row in a CSV file as a separate document.

        Args:
        ----
            csv_path: Path to the input CSV file.
            codebook_path: Path to the codebook JSON file.
            text_column: Column containing the text to analyze.
            id_column: Optional column to use as the document ID. If omitted,
                the row index (0..N-1) is used. Duplicate IDs get a numeric
                suffix (e.g. "id", "id_2", "id_3").
            output_dir: Optional base name for the analyzer output directory.

        Returns:
        -------
            A tuple containing results by document ID, metadata by document ID,
            and error records for failed rows.

        Raises:
        ------
            ValueError: If the CSV is missing a required column.
            FileLoadError: If the codebook cannot be loaded.

        """
        csv_file = Path(csv_path)
        codebook_file = Path(codebook_path)
        codebook = codebook_constructs(load_codebook(codebook_file))
        output_path = create_output_directory(
            output_name=output_dir, base_dir=Path.cwd()
        )

        df = pd.read_csv(csv_file)
        required_columns = [text_column]
        if id_column is not None:
            required_columns.append(id_column)
        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        results_by_doc: dict[str, AnalysisResult] = {}
        metadata_by_doc: dict[str, dict] = {}
        errors: list[ErrorRecord] = []
        total_documents = len(df)
        success_count = 0
        document_id_counts: dict[str, int] = {}

        for position, (index, row) in enumerate(df.iterrows(), start=1):
            text = str(row[text_column])
            if id_column is not None:
                base_document_id = str(row[id_column]).strip()
            else:
                base_document_id = str(index)

            document_id_counts[base_document_id] = (
                document_id_counts.get(base_document_id, 0) + 1
            )
            duplicate_count = document_id_counts[base_document_id]
            if duplicate_count == 1:
                document_id = base_document_id
            else:
                document_id = f"{base_document_id}_{duplicate_count}"

            print(f"Processing [{position}/{total_documents}]: {document_id}")

            try:
                result, metadata = prompt_for_constructs(
                    text=text,
                    codebook=codebook,
                    document_id=document_id,
                    config=self.config,
                )
                _save_successful_result(
                    result,
                    metadata,
                    output_path,
                    results_by_doc,
                    metadata_by_doc,
                )
                success_count += 1
                print(f"  OK: Found {len(result.instances)} construct instances")

            except PromptingError as error:
                print(f"  Failed: {error}")
                error_record = _save_processing_error(
                    document_id=document_id,
                    document_path=csv_file,
                    error=error,
                    output_path=output_path,
                    config=self.config,
                )
                errors.append(error_record)

            finally:
                print(
                    "  Progress: "
                    f"Successful {success_count}/{total_documents}; "
                    f"Errors {len(errors)}/{total_documents}"
                )

        save_run_summary(
            output_dir=output_path,
            model_name=self.config.model_name,
            codebook_name=codebook_file.name,
            input_dir_name=csv_file.name,
            failed_documents=[error.document_id for error in errors],
            total_documents=total_documents,
        )

        print("\nAnalysis complete!")
        print(f"Output directory: {output_path}")
        print(f"Successful: {success_count}/{total_documents}")
        if errors:
            print(f"Failed: {len(errors)} (see errors directory)")

        return results_by_doc, metadata_by_doc, errors

    def retry_errors(
        self, output_dir: Path | str, codebook_path: Path | str
    ) -> tuple[dict[str, AnalysisResult], dict[str, dict], list[ErrorRecord]]:
        """Retry documents that failed in a previous analyzer run.

        Args:
        ----
            output_dir: Analyzer output directory containing saved errors.
            codebook_path: Path to the codebook JSON file.

        Returns:
        -------
            A tuple containing recovered results by document ID, recovered
            metadata by document ID, and records that still failed.

        Raises:
        ------
            FileLoadError: If the codebook cannot be loaded.

        """
        output_path = Path(output_dir)
        codebook = codebook_constructs(load_codebook(codebook_path))
        failed_records = load_error_records(output_path)

        if not failed_records:
            print("No errors found to retry.")
            return {}, {}, []

        print(f"Found {len(failed_records)} error(s) to retry.")

        results_by_doc: dict[str, AnalysisResult] = {}
        metadata_by_doc: dict[str, dict] = {}
        remaining_errors: list[ErrorRecord] = []

        for position, error_record in enumerate(failed_records, start=1):
            print(
                f"Retrying [{position}/{len(failed_records)}]: "
                f"{error_record.document_id}"
            )

            try:
                result, metadata = self.analyze_document(
                    error_record.document_path, codebook
                )
                _save_successful_result(
                    result,
                    metadata,
                    output_path,
                    results_by_doc,
                    metadata_by_doc,
                )

                old_error_file = (
                    output_path / "errors" / f"{error_record.document_id}_error.json"
                )
                if old_error_file.exists():
                    old_error_file.unlink()

                print(f"OK: Found {len(result.instances)} construct instances")

            except (FileLoadError, PromptingError) as error:
                print(f"Still failing: {error}")
                remaining_errors.append(error_record)

        print("\nRetry complete!")
        print(f"  Recovered: {len(results_by_doc)}")
        print(f"  Still failing: {len(remaining_errors)}")

        return results_by_doc, metadata_by_doc, remaining_errors
