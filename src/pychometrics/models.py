"""Data models for pychometrics package.

This module defines the Pydantic models used throughout the package for
type validation and serialization of analysis results, construct instances,
and codebook definitions.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ConstructDefinition(BaseModel):
    """Definition of a psychological construct in the codebook.

    Attributes:
        name: The name of the psychological construct.
        definition: A clear definition explaining the construct.
        examples: Optional list of example phrases or quotes.
    """

    name: str = Field(..., description="Name of the psychological construct")
    definition: str = Field(..., description="Definition of the construct")
    examples: Optional[list[str]] = Field(
        default=None, description="Optional example phrases demonstrating the construct"
    )


class Codebook(BaseModel):
    """A codebook containing psychological construct definitions.

    Attributes:
        constructs: List of construct definitions to identify in text.
    """

    constructs: list[ConstructDefinition] = Field(
        ..., description="List of psychological construct definitions"
    )


class ConstructInstance(BaseModel):
    """An instance of a psychological construct found in text.

    Attributes:
        construct: Name of the construct identified.
        speaker_id: Speaker identifier if available.
        quote: Exact quote from the text where construct appears.
        quote_index: Start and end indices of the quote in the original text.
        confidence: Ordinal confidence score (0-2).
    """

    construct: str = Field(..., description="Name of the identified construct")
    speaker_id: Optional[str] = Field(
        default=None, description="Speaker identifier if available"
    )
    quote: str = Field(..., description="Exact quote from the text")
    quote_index: Optional[str] = Field(
        default=None, description="Start:end indices of quote in original text"
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=2,
        description="Confidence score: 0=not mentioned/negated, 1=indirect, 2=clear",
    )


class AnalysisResult(BaseModel):
    """Result of analyzing a single document for psychological constructs.

    Attributes:
        document_id: Identifier for the document (from filename).
        instances: List of construct instances found in the document.
    """

    document_id: str = Field(..., description="Document identifier from filename")
    instances: list[ConstructInstance] = Field(
        default_factory=list, description="List of construct instances found"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the analysis result.
        """
        return self.model_dump()


class APIMetadata(BaseModel):
    """Metadata from the LLM API response.

    Attributes:
        model: The model used for the request.
        usage: Token usage information.
        created: Timestamp of the response.
        response_id: Unique identifier for the response.
        latency_ms: Response latency in milliseconds.
        raw_response: Complete raw response from the API.
    """

    model: Optional[str] = None
    usage: Optional[dict] = None
    created: Optional[int] = None
    response_id: Optional[str] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[dict] = None


class ErrorRecord(BaseModel):
    """Record of a failed document processing attempt.

    Attributes:
        document_id: Identifier for the document.
        document_path: Path to the original document.
        error_message: Description of the error.
        model_used: Model that was attempted.
        timestamp: When the error occurred.
    """

    document_id: str
    document_path: str
    error_message: str
    model_used: str = ""
    timestamp: Optional[str] = None
