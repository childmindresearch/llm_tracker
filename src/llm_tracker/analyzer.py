"""Main analyzer module for pychometrics."""

from pathlib import Path
from typing import Optional

from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_tracker.config import AnalyzerConfig
from llm_tracker.file_handlers import (
    FileLoadError,
    create_output_directory,
    get_document_files,
    load_codebook,
    load_document,
    load_error_records,
    save_analysis_result,
    save_error_record,
    save_metadata,
    save_readme,
)
from llm_tracker.models import AnalysisResult, APIMetadata, ErrorRecord
from llm_tracker.prompting import PromptingError, prompt_for_constructs


class PychometricsAnalyzer:
    """Analyzer for identifying psychological constructs in text documents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        config: Optional[AnalyzerConfig] = None,
    ):
        if config is not None:
            self.config = config
        else:
            kwargs = {"api_key": api_key}
            if model_name is not None:
                kwargs["model_name"] = model_name
            if custom_prompt is not None:
                kwargs["custom_prompt"] = custom_prompt
            if similarity_threshold is not None:
                kwargs["similarity_threshold"] = similarity_threshold
            self.config = AnalyzerConfig(**kwargs)

    def analyze_document(
        self, document_path: Path | str, codebook: dict
    ) -> tuple[AnalysisResult, APIMetadata]:
        """Analyze a single document for psychological constructs."""
        path = Path(document_path)
        text, document_id = load_document(path)

        result, metadata = prompt_for_constructs(
            text=text, codebook=codebook, document_id=document_id, config=self.config
        )

        return result, metadata

    def analyze_directory(
        self,
        input_dir: Path | str,
        codebook_path: Path | str,
        output_dir: Optional[str] = None,
    ) -> tuple[dict[str, dict], dict[str, dict], list[ErrorRecord]]:
        """Analyze all documents in a directory.

        Returns:
            Tuple of (results_dict, metadata_dict, errors_list).
        """
        input_path = Path(input_dir)
        codebook_path = Path(codebook_path)

        codebook = load_codebook(codebook_path)
        document_files = get_document_files(input_path)
        output_path = create_output_directory(
            output_name=output_dir, base_dir=Path.cwd()
        )

        results_dict: dict[str, dict] = {}
        metadata_dict: dict[str, dict] = {}
        errors: list[ErrorRecord] = []

        total_documents = len(document_files)
        success_count = 0
        error_count = 0

        for i, doc_path in enumerate(document_files, 1):
            document_name = doc_path.name
            print(f"Processing [{i}/{total_documents}]: {document_name}")

            try:
                result, metadata = self.analyze_document(doc_path, codebook)

                save_analysis_result(result, output_path)
                save_metadata(metadata, result.document_id, output_path)

                results_dict[result.document_id] = result.to_dict()
                metadata_dict[result.document_id] = metadata.model_dump()

                success_count += 1
                print(f"  ✓ Found {len(result.instances)} construct instances")

            except (FileLoadError, PromptingError) as e:
                print(f"  ✗ Failed: {e}")

                error = ErrorRecord(
                    document_id=doc_path.stem,
                    document_path=str(doc_path),
                    error_message=str(e),
                    model_used=self.config.model_name,
                    timestamp=datetime.now().isoformat(),
                )
                save_error_record(error, output_path)
                if isinstance(e, PromptingError) and e.metadata is not None:
                    save_metadata(e.metadata, doc_path.stem, output_path)
                else:
                    error_metadata = APIMetadata(
                        model=self.config.model_name,
                        num_retries=self.config.max_retries,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        error_output=str(e),
                    )
                    save_metadata(error_metadata, doc_path.stem, output_path)
                errors.append(error)
                error_count += 1
                continue
            finally:
                print(
                    "  Progress: "
                    f"Successful {success_count}/{total_documents}; "
                    f"Errors {error_count}/{total_documents}"
                )

        save_readme(
            output_dir=output_path,
            model_name=self.config.model_name,
            codebook_name=codebook_path.name,
            input_dir_name=input_path.name,
            failed_documents=[e.document_id for e in errors],
            total_documents=total_documents,
        )

        print(f"\nAnalysis complete!")
        print(f"  Output directory: {output_path}")
        print(f"  Successful: {len(results_dict)}/{total_documents}")

        if errors:
            print(f"  Failed: {len(errors)} (see errors/ directory)")

        return results_dict, metadata_dict, errors

    def analyze_csv(
        self,
        csv_path: "Path | str",
        codebook_path: "Path | str",
        text_column: str,
        subreddit_column: str = "subreddit",
        author_column: str = "author",
        output_dir: Optional[str] = None,
    ) -> "tuple[dict[str, dict], dict[str, dict], list[ErrorRecord]]":
        """Analyze all rows in a CSV file as individual documents.

        Each row is treated as a separate document. The document_id is
        constructed as {subreddit}_{author}. Duplicate subreddit/author
        combinations are disambiguated with a numeric suffix.

        Args:
            csv_path: Path to the input CSV file.
            codebook_path: Path to the codebook JSON file.
            text_column: Name of the column containing the text to analyze.
            subreddit_column: Name of the column containing the subreddit.
                Defaults to 'subreddit'.
            author_column: Name of the column containing the author.
                Defaults to 'author'.
            output_dir: Optional name for the output directory.
                A timestamp is appended automatically.

        Returns:
            Tuple of (results_dict, metadata_dict, errors_list).
        """
        import pandas as pd

        csv_path = Path(csv_path)
        codebook_path = Path(codebook_path)

        codebook = load_codebook(codebook_path)
        output_path = create_output_directory(
            output_name=output_dir, base_dir=Path.cwd()
        )

        df = pd.read_csv(csv_path)

        required = [text_column, subreddit_column, author_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        results_dict: dict[str, dict] = {}
        metadata_dict: dict[str, dict] = {}
        errors: list[ErrorRecord] = []

        total_documents = len(df)
        success_count = 0
        error_count = 0
        seen: dict[str, int] = {}

        for i, (_, row) in enumerate(df.iterrows(), 1):
            subreddit = str(row[subreddit_column]).strip()
            author = str(row[author_column]).strip()
            text = str(row[text_column])

            base_id = f"{subreddit}_{author}"
            if base_id in seen:
                seen[base_id] += 1
                document_id = f"{base_id}_{seen[base_id]}"
            else:
                seen[base_id] = 1
                document_id = base_id

            print(f"Processing [{i}/{total_documents}]: {document_id}")

            try:
                result, metadata = prompt_for_constructs(
                    text=text,
                    codebook=codebook,
                    document_id=document_id,
                    config=self.config,
                )

                save_analysis_result(result, output_path)
                save_metadata(metadata, result.document_id, output_path)

                results_dict[result.document_id] = result.to_dict()
                metadata_dict[result.document_id] = metadata.model_dump()

                success_count += 1
                print(f"  \u2713 Found {len(result.instances)} construct instances")

            except PromptingError as e:
                print(f"  \u2717 Failed: {e}")

                error = ErrorRecord(
                    document_id=document_id,
                    document_path=str(csv_path),
                    error_message=str(e),
                    model_used=self.config.model_name,
                    timestamp=datetime.now().isoformat(),
                )
                save_error_record(error, output_path)
                if e.metadata is not None:
                    save_metadata(e.metadata, document_id, output_path)
                else:
                    error_metadata = APIMetadata(
                        model=self.config.model_name,
                        num_retries=self.config.max_retries,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        error_output=str(e),
                    )
                    save_metadata(error_metadata, document_id, output_path)
                errors.append(error)
                error_count += 1
                continue
            finally:
                print(
                    "  Progress: "
                    f"Successful {success_count}/{total_documents}; "
                    f"Errors {error_count}/{total_documents}"
                )

        save_readme(
            output_dir=output_path,
            model_name=self.config.model_name,
            codebook_name=codebook_path.name,
            input_dir_name=csv_path.name,
            failed_documents=[e.document_id for e in errors],
            total_documents=total_documents,
        )

        print(f"\nAnalysis complete!")
        print(f"  Output directory: {output_path}")
        print(f"  Successful: {len(results_dict)}/{total_documents}")

        if errors:
            print(f"  Failed: {len(errors)} (see errors/ directory)")

        return results_dict, metadata_dict, errors

    def retry_errors(
        self, output_dir: Path | str, codebook_path: Path | str
    ) -> tuple[dict[str, dict], dict[str, dict], list[ErrorRecord]]:
        """Retry processing documents that previously failed.

        Args:
            output_dir: Path to the output directory containing errors/.
            codebook_path: Path to the codebook JSON file.

        Returns:
            Tuple of (new_results_dict, new_metadata_dict, remaining_errors).
        """
        output_path = Path(output_dir)
        codebook = load_codebook(codebook_path)

        errors = load_error_records(output_path)

        if not errors:
            print("No errors found to retry.")
            return {}, {}, []

        print(f"Found {len(errors)} error(s) to retry.")

        results_dict: dict[str, dict] = {}
        metadata_dict: dict[str, dict] = {}
        remaining_errors: list[ErrorRecord] = []

        for i, error in enumerate(errors, 1):
            print(f"Retrying [{i}/{len(errors)}]: {error.document_id}")

            try:
                result, metadata = self.analyze_document(error.document_path, codebook)

                save_analysis_result(result, output_path)
                save_metadata(metadata, result.document_id, output_path)
                results_dict[result.document_id] = result.to_dict()
                metadata_dict[result.document_id] = metadata.model_dump()

                old_error_file = (
                    output_path / "errors" / f"{error.document_id}_error.json"
                )
                if old_error_file.exists():
                    old_error_file.unlink()

                print(f"  ✓ Success! Found {len(result.instances)} instances")

            except (FileLoadError, PromptingError) as e:
                print(f"  ✗ Still failing: {e}")
                remaining_errors.append(error)

        print(f"\nRetry complete!")
        print(f"  Recovered: {len(results_dict)}")
        print(f"  Still failing: {len(remaining_errors)}")

        return results_dict, metadata_dict, remaining_errors
