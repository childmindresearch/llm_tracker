"""Main analyzer module for pychometrics.

This module provides the PychometricsAnalyzer class, which orchestrates
the complete analysis workflow from loading documents to saving results.
"""

from pathlib import Path
from typing import Optional

from pychometrics.config import AnalyzerConfig
from pychometrics.file_handlers import (
    FileLoadError,
    create_output_directory,
    get_document_files,
    load_codebook,
    load_document,
    save_analysis_result,
    save_metadata,
    save_readme,
)
from pychometrics.models import AnalysisResult, APIMetadata
from pychometrics.prompting import PromptingError, prompt_for_constructs


class PychometricsAnalyzer:
    """Analyzer for identifying psychological constructs in text documents.

    This class provides methods for analyzing single documents or entire
    directories of documents using LLM-based construct identification.

    Attributes:
        config: Configuration for the analyzer including API settings.


    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        config: Optional[AnalyzerConfig] = None,
    ):
        """Initialize the analyzer.

        Args:
            api_key: OpenRouter API key. Can also be set via
                OPENROUTER_API_KEY environment variable.
            model_name: Model identifier for OpenRouter. Defaults to
                Claude 3.5 Sonnet.
            custom_prompt: Optional custom prompt template with {text}
                and {codebook} placeholders.
            config: Optional pre-configured AnalyzerConfig. If provided,
                other parameters are ignored.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        if config is not None:
            self.config = config
        else:
            kwargs = {"api_key": api_key}
            if model_name is not None:
                kwargs["model_name"] = model_name
            if custom_prompt is not None:
                kwargs["custom_prompt"] = custom_prompt
            self.config = AnalyzerConfig(**kwargs)

    def analyze_document(
        self, document_path: Path | str, codebook: dict
    ) -> tuple[AnalysisResult, APIMetadata]:
        """Analyze a single document for psychological constructs.

        Args:
            document_path: Path to the document file (CSV or TXT).
            codebook: Parsed codebook dictionary with construct definitions.

        Returns:
            Tuple of (AnalysisResult, APIMetadata).

        Raises:
            FileLoadError: If the document cannot be loaded.
            PromptingError: If the LLM analysis fails after retries.
        """
        text, document_id = load_document(document_path)

        result, metadata = prompt_for_constructs(
            text=text, codebook=codebook, document_id=document_id, config=self.config
        )

        return result, metadata

    def analyze_directory(
        self,
        input_dir: Path | str,
        codebook_path: Path | str,
        output_dir: Optional[str] = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Analyze all documents in a directory.

        This method processes all CSV and TXT files in the input directory,
        saves results to an output directory, and returns the results for
        programmatic use.

        Args:
            input_dir: Directory containing document files to analyze.
            codebook_path: Path to the codebook JSON file.
            output_dir: Optional name for the output directory. A timestamp
                will be appended automatically.

        Returns:
            Tuple of two dictionaries:
            - results_dict: Mapping of document_id to analysis result dict
            - metadata_dict: Mapping of document_id to API metadata dict

        Raises:
            FileLoadError: If the input directory or codebook cannot be loaded.
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
        failed_documents: list[str] = []

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
                failed_documents.append(document_name)
                continue

        save_readme(
            output_dir=output_path,
            model_name=self.config.model_name,
            codebook_name=codebook_path.name,
            input_dir_name=input_path.name,
            failed_documents=failed_documents,
            total_documents=total_documents,
        )

        print(f"\nAnalysis complete!")
        print(f"  Output directory: {output_path}")
        print(
            f"  Successful: {total_documents - len(failed_documents)}/{total_documents}"
        )

        if failed_documents:
            print(f"  Failed documents: {', '.join(failed_documents)}")

        return results_dict, metadata_dict
