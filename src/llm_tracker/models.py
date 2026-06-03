"""Data models for llm_tracker."""

from pydantic import BaseModel, Field


class ConstructDefinition(BaseModel):
    """Definition of a psychological construct in the codebook.

    Attributes:
    ----------
        name: The name of the psychological construct.
        definition: A clear definition explaining the construct.
        examples: Optional list of example phrases or quotes.

    """

    name: str = Field(..., description="Name of the psychological construct")
    definition: str = Field(..., description="Definition of the construct")
    examples: list[str] | None = Field(
        default=None, description="Optional example phrases demonstrating the construct"
    )


class Codebook(BaseModel):
    """A codebook containing psychological construct definitions.

    Attributes:
    ----------
        constructs: List of construct definitions to identify in text.

    """

    constructs: list[ConstructDefinition] = Field(
        ..., description="List of psychological construct definitions"
    )


class ConstructInstance(BaseModel):
    """An instance of a psychological construct found in text.

    Attributes:
    ----------
        construct: Name of the construct identified.
        speaker_id: Speaker identifier if available.
        quote: Exact quote from the text where construct appears.
        quote_index: Start and end indices of the quote in the original text.
        confidence: Ordinal confidence score (0-2).

    """

    construct: str = Field(..., description="Name of the identified construct")
    speaker_id: str | None = Field(
        default=None, description="Speaker identifier if available"
    )
    quote: str = Field(..., description="Exact quote from the text")
    quote_index: str | None = Field(
        default=None, description="Start:end indices of quote in original text"
    )
    confidence: int | None = Field(
        default=None,
        ge=0,
        le=2,
        description=(
            "Confidence score: 0=not mentioned/negated, 1=indirect, "
            "2=clear. None for human codings."
        ),
    )


class AnalysisResult(BaseModel):
    """Result of analyzing a single document for psychological constructs.

    Attributes:
    ----------
        document_id: Identifier for the document (from filename).
        instances: List of construct instances found in the document.

    """

    document_id: str = Field(..., description="Document identifier from filename")
    instances: list[ConstructInstance] = Field(
        default_factory=list, description="List of construct instances found"
    )


class APIMetadata(BaseModel):
    """Metadata from the LLM API response.

    Attributes:
    ----------
        model: The model used for the request.
        usage: Token usage information.
        created: Timestamp of the response.
        response_id: Unique identifier for the response.
        latency_ms: Response latency in milliseconds.
        raw_response: Complete raw response from the API.

    """

    model: str | None = None
    usage: dict | None = None
    created: int | None = None
    response_id: str | None = None
    latency_ms: float | None = None
    raw_response: dict | None = None
    num_retries: int = 0
    error_message: str | None = None
    error_type: str | None = None
    error_output: str | None = None


class ErrorRecord(BaseModel):
    """Record of a failed document processing attempt.

    Attributes:
    ----------
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
    timestamp: str | None = None
