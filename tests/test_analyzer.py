"""Tests for analyzer orchestration."""

from pathlib import Path

import llm_tracker.analyzer as analyzer_module
import pandas as pd
import pytest
from llm_tracker.analyzer import LLMTrackerAnalyzer
from llm_tracker.config import AnalyzerConfig
from llm_tracker.file_handlers import create_output_directory, save_error_record
from llm_tracker.models import (
    AnalysisResult,
    APIMetadata,
    ConstructInstance,
    ErrorRecord,
)
from llm_tracker.prompting import PromptingError


def config() -> AnalyzerConfig:
    return AnalyzerConfig(api_key="test-key", model_name="test-model")


def write_codebook(tmp_path: Path) -> Path:
    codebook_path = tmp_path / "codebook.json"
    codebook_path.write_text(
        '{"constructs": [{"name": "stress", "definition": "Stress"}]}',
        encoding="utf-8",
    )
    return codebook_path


def successful_result(document_id: str, text: str = "sample text") -> AnalysisResult:
    return AnalysisResult(
        document_id=document_id,
        instances=[
            ConstructInstance(
                construct="stress",
                quote=text,
                quote_index=None,
                confidence=2,
            )
        ],
    )


def successful_metadata(model: str = "test-model") -> APIMetadata:
    return APIMetadata(model=model)


def test_analyze_document_loads_text_and_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "doc_1.txt"
    document_path.write_text("I feel overwhelmed", encoding="utf-8")
    captured = {}

    def fake_prompt_for_constructs(text, codebook, document_id, config):
        captured["text"] = text
        captured["codebook"] = codebook
        captured["document_id"] = document_id
        return successful_result(document_id, text), successful_metadata()

    monkeypatch.setattr(
        analyzer_module,
        "prompt_for_constructs",
        fake_prompt_for_constructs,
    )
    analyzer = LLMTrackerAnalyzer(config=config())
    codebook = {"constructs": [{"name": "stress"}]}

    result, metadata = analyzer.analyze_document(document_path, codebook)

    assert captured == {
        "text": "I feel overwhelmed",
        "codebook": {"constructs": [{"name": "stress"}]},
        "document_id": "doc_1",
    }
    assert result.document_id == "doc_1"
    assert metadata.model == "test-model"


def test_analyze_directory_saves_successes_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "good.txt").write_text("good text", encoding="utf-8")
    (input_dir / "bad.txt").write_text("bad text", encoding="utf-8")
    codebook_path = write_codebook(tmp_path)

    def fake_prompt_for_constructs(text, codebook, document_id, config):
        if document_id == "bad":
            raise PromptingError("LLM failed")
        return successful_result(document_id, text), successful_metadata()

    monkeypatch.setattr(
        analyzer_module,
        "prompt_for_constructs",
        fake_prompt_for_constructs,
    )
    analyzer = LLMTrackerAnalyzer(config=config())

    results, metadata, errors = analyzer.analyze_directory(
        input_dir,
        codebook_path,
        output_dir="run",
    )

    output_dir = next(tmp_path.glob("run_*"))
    assert set(results) == {"good"}
    assert set(metadata) == {"good"}
    assert [error.document_id for error in errors] == ["bad"]
    assert (output_dir / "encodings" / "good.json").exists()
    assert (output_dir / "metadata" / "good_meta.json").exists()
    assert (output_dir / "errors" / "bad_error.json").exists()
    assert (output_dir / "README.md").exists()


def test_analyze_csv_uses_rows_as_documents_and_handles_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    codebook_path = write_codebook(tmp_path)
    csv_path = tmp_path / "posts.csv"
    pd.DataFrame(
        [
            {"subreddit": "ask", "author": "user", "body": "first post"},
            {"subreddit": "ask", "author": "user", "body": "second post"},
        ]
    ).to_csv(csv_path, index=False)

    def fake_prompt_for_constructs(text, codebook, document_id, config):
        return successful_result(document_id, text), successful_metadata()

    monkeypatch.setattr(
        analyzer_module,
        "prompt_for_constructs",
        fake_prompt_for_constructs,
    )
    analyzer = LLMTrackerAnalyzer(config=config())

    results, metadata, errors = analyzer.analyze_csv(
        csv_path,
        codebook_path,
        text_column="body",
        output_dir="csv_run",
    )

    assert set(results) == {"ask_user", "ask_user_2"}
    assert set(metadata) == {"ask_user", "ask_user_2"}
    assert errors == []


def test_analyze_csv_raises_for_missing_required_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    codebook_path = write_codebook(tmp_path)
    csv_path = tmp_path / "posts.csv"
    pd.DataFrame([{"subreddit": "ask", "author": "user"}]).to_csv(
        csv_path,
        index=False,
    )
    analyzer = LLMTrackerAnalyzer(config=config())

    with pytest.raises(ValueError, match="Missing columns"):
        analyzer.analyze_csv(
            csv_path,
            codebook_path,
            text_column="body",
        )


def test_retry_errors_recovers_successful_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codebook_path = write_codebook(tmp_path)
    output_dir = create_output_directory("run", base_dir=tmp_path)
    document_path = tmp_path / "failed.txt"
    document_path.write_text("retry text", encoding="utf-8")
    save_error_record(
        ErrorRecord(
            document_id="failed",
            document_path=str(document_path),
            error_message="LLM failed",
            model_used="test-model",
        ),
        output_dir,
    )

    def fake_prompt_for_constructs(text, codebook, document_id, config):
        return successful_result(document_id, text), successful_metadata()

    monkeypatch.setattr(
        analyzer_module,
        "prompt_for_constructs",
        fake_prompt_for_constructs,
    )
    analyzer = LLMTrackerAnalyzer(config=config())

    results, metadata, remaining_errors = analyzer.retry_errors(
        output_dir,
        codebook_path,
    )

    assert set(results) == {"failed"}
    assert set(metadata) == {"failed"}
    assert remaining_errors == []
    assert (output_dir / "encodings" / "failed.json").exists()
    assert not (output_dir / "errors" / "failed_error.json").exists()
