"""Main analyzer module for pychometrics."""

from pathlib import Path
from typing import Optional

from datetime import datetime
from pathlib import Path
from typing import Optional

from pychometrics.config import AnalyzerConfig
from pychometrics.file_handlers import (
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
from pychometrics.models import AnalysisResult, APIMetadata, ErrorRecord
from pychometrics.prompting import PromptingError, prompt_for_constructs


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

        for i, doc_path in enumerate(document_files, 1):
            document_name = doc_path.name
            print(f"Processing [{i}/{total_documents}]: {document_name}")

            try:
                result, metadata = self.analyze_document(doc_path, codebook)

                save_analysis_result(result, output_path)
                save_metadata(metadata, result.document_id, output_path)

                results_dict[result.document_id] = result.to_dict()
                metadata_dict[result.document_id] = metadata.model_dump()

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
                errors.append(error)
                continue

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
