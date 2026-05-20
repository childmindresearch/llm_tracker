"""Tests for llm_tracker.models module."""

import pytest
from llm_tracker.models import (
    AnalysisResult,
    APIMetadata,
    Codebook,
    ConstructDefinition,
    ConstructInstance,
)
from pydantic import ValidationError


def test_valid_construct_with_examples() -> None:
    """Test creating a valid construct with examples."""
    construct = ConstructDefinition(
        name="Self-Efficacy",
        definition="Belief in one's capabilities",
        examples=["I can do it", "I believe in myself"],
    )

    assert construct.name == "Self-Efficacy"
    assert construct.definition == "Belief in one's capabilities"


def test_valid_construct_without_examples() -> None:
    """Test creating a valid construct without examples."""
    construct = ConstructDefinition(
        name="Resilience", definition="Ability to recover from setbacks"
    )

    assert construct.name == "Resilience"
    assert construct.examples is None


def test_valid_codebook() -> None:
    """Test creating a valid codebook."""
    codebook = Codebook(
        constructs=[ConstructDefinition(name="Test", definition="A test construct")]
    )

    assert len(codebook.constructs) == 1


def test_empty_constructs() -> None:
    """Test codebook with empty constructs list."""
    codebook = Codebook(constructs=[])
    assert len(codebook.constructs) == 0


def test_valid_instance() -> None:
    """Test creating a valid construct instance."""
    instance = ConstructInstance(
        construct="Self-Efficacy",
        speaker_id="P1",
        quote="I believe I can handle this",
        confidence=2,
    )

    assert instance.construct == "Self-Efficacy"
    assert instance.speaker_id == "P1"
    assert instance.quote == "I believe I can handle this"
    assert instance.confidence == 2


def test_instance_without_speaker() -> None:
    """Test instance without speaker_id."""
    instance = ConstructInstance(
        construct="Resilience", quote="I bounced back from failure", confidence=1
    )

    assert instance.speaker_id is None


def test_invalid_confidence_score() -> None:
    """Test that confidence outside 0-2 raises error."""
    with pytest.raises(ValidationError):
        ConstructInstance(construct="Test", quote="Test quote", confidence=5)


def test_negative_confidence_score() -> None:
    """Test that negative confidence raises error."""
    with pytest.raises(ValidationError):
        ConstructInstance(construct="Test", quote="Test quote", confidence=-1)


def test_valid_result() -> None:
    """Test creating a valid analysis result."""
    result = AnalysisResult(
        document_id="interview_001",
        instances=[
            ConstructInstance(construct="Test", quote="Test quote", confidence=2)
        ],
    )

    assert result.document_id == "interview_001"
    assert len(result.instances) == 1


def test_empty_instances() -> None:
    """Test result with no instances found."""
    result = AnalysisResult(document_id="empty_doc", instances=[])

    assert len(result.instances) == 0


def test_to_dict() -> None:
    """Test conversion to dictionary."""
    result = AnalysisResult(
        document_id="test_doc",
        instances=[
            ConstructInstance(
                construct="Test", speaker_id="P1", quote="A test quote", confidence=1
            )
        ],
    )

    d = result.model_dump()

    assert d["document_id"] == "test_doc"
    assert len(d["instances"]) == 1
    assert d["instances"][0]["construct"] == "Test"


def test_partial_metadata() -> None:
    """Test creating metadata with only some fields."""
    metadata = APIMetadata(model="test-model")

    assert metadata.model == "test-model"
    assert metadata.usage is None
    assert metadata.created is None
