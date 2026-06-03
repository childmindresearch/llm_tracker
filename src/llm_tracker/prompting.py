"""LLM prompting functionality for llm_tracker.

This module handles all interactions with LLM providers (via the any-llm
SDK), including constructing prompts, making completion requests, parsing
responses, and handling retries.
"""

import difflib
import json
import time

from any_llm import AnyLLM

from llm_tracker.config import AnalyzerConfig
from llm_tracker.models import AnalysisResult, APIMetadata, ConstructInstance


class PromptingError(Exception):
    """Exception raised when prompting fails after all retries."""

    def __init__(self, message: str, metadata: APIMetadata | None = None) -> None:
        """Create a prompting error.

        Args:
        ----
            message: Error message describing the failure.
            metadata: Optional API metadata captured before the failure.

        """
        super().__init__(message)
        self.metadata = metadata


def validate_llm_output(response_text: str) -> dict:
    """Validate LLM output into the expected JSON shape.

    Args:
    ----
        response_text: Raw response text returned by the LLM.

    Returns:
    -------
        Parsed response dictionary containing an instances list.

    Raises:
    ------
        PromptingError: If the response is empty, invalid JSON, or missing the
            expected instances list.

    """
    if response_text is None:
        raise PromptingError("Response text is empty.")

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise PromptingError(
            f"Invalid JSON in response (if using an uncommon model, it may not "
            f"support response_format): {e}"
        ) from e

    if (
        not isinstance(data, dict)
        or "instances" not in data
        or not isinstance(data["instances"], list)
    ):
        raise PromptingError('JSON must contain an "instances" list.')

    return data


def construct_prompt(text: str, codebook: dict, template: str) -> str:
    """Construct the full prompt by inserting text and codebook.

    Args:
    ----
        text: The document text to analyze.
        codebook: The codebook dictionary containing construct definitions.
        template: The prompt template with {text} and {codebook} placeholders.

    Returns:
    -------
        The complete prompt string ready for the API.

    """
    codebook_str = json.dumps(codebook, indent=2)
    return template.format(text=text, codebook=codebook_str)


def find_quote_index(
    text: str,
    quote: str,
    *,
    fuzzy: bool = False,
    threshold: float = 0.85,
) -> str | None:
    """Find the start:end index of a quote in the source text.

    Args:
    ----
        text: The original document text.
        quote: The quote to find.
        fuzzy: If True, fall back to fuzzy matching when exact matching fails.
        threshold: Minimum similarity ratio (0.0 to 1.0) to consider a match.

    Returns:
    -------
        String in format "start:end" or None if not found.

    """
    if not quote:
        return None

    start = text.find(quote)
    if start != -1:
        return f"{start}:{start + len(quote)}"

    if not fuzzy:
        return None

    quote_len = len(quote)
    best_ratio = 0.0
    best_start = -1

    for i in range(len(text) - quote_len + 1):
        window = text[i : i + quote_len]
        ratio = difflib.SequenceMatcher(None, quote.lower(), window.lower()).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio >= threshold:
        return f"{best_start}:{best_start + quote_len}"

    return None


def parse_llm_response(
    response_text: str,
    document_id: str,
    original_text: str = "",
    fuzzy_quote_matching: bool = False,
    threshold: float = 0.85,
) -> AnalysisResult:
    """Parse the LLM response into an AnalysisResult.

    Args:
    ----
        response_text: Raw text response from the LLM.
        document_id: The document identifier for the result.
        original_text: The original document text for finding quote indices.
        fuzzy_quote_matching: Whether to use fuzzy matching for quote indices.
        threshold: Minimum similarity ratio for fuzzy quote matching.

    Returns:
    -------
        Parsed AnalysisResult object.

    Raises:
    ------
        PromptingError: If the response is not valid JSON or does not contain
            the expected instances list.

    """
    data = validate_llm_output(response_text)

    instances = []
    raw_instances = data.get("instances", [])

    for item in raw_instances:
        try:
            quote = item.get("quote", "")
            quote_index = (
                find_quote_index(
                    original_text,
                    quote,
                    fuzzy=fuzzy_quote_matching,
                    threshold=threshold,
                )
                if original_text
                else None
            )

            instance = ConstructInstance(
                construct=item.get("construct", "Unknown"),
                speaker_id=item.get("speaker_id"),
                quote=quote,
                quote_index=quote_index,
                confidence=int(item.get("confidence", 1)),
            )
            instances.append(instance)
        except (ValueError, TypeError):
            continue

    return AnalysisResult(document_id=document_id, instances=instances)


def _to_dict(obj: object) -> dict | None:
    """Best-effort conversion of an any-llm response object to a plain dict.

    any-llm returns OpenAI-shaped response objects rather than raw dicts, so
    fields stored on APIMetadata (usage, raw_response) are coerced here. Returns
    None if the object cannot be represented as a dict.

    Args:
    ----
        obj: The response object or sub-object to convert.

    Returns:
    -------
        A dict representation of the object, or None.

    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                result = method()
            except Exception:  # noqa: BLE001 - best-effort serialization
                continue
            return result if isinstance(result, dict) else None
    try:
        return dict(obj)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def call_llm_api(prompt: str, config: AnalyzerConfig) -> tuple[str, APIMetadata]:
    """Make a chat completion request through any-llm.

    Routes the request to the configured provider (default "openrouter") using
    the resolved API key. The request and response shape are OpenAI-compatible,
    so response text is read from choices[0].message.content.

    Args:
    ----
        prompt: The complete prompt to send.
        config: Configuration including API key, provider, and model.

    Returns:
    -------
        Response text and API metadata.

    Raises:
    ------
        PromptingError: If the API request fails or the response shape is
            unexpected.

    """
    messages = [{"role": "user", "content": prompt}]

    request_kwargs: dict = {
        "model": config.model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if config.temperature is not None:
        request_kwargs["temperature"] = config.temperature

    start_time = time.time()

    try:
        client = AnyLLM.create(config.provider, api_key=config.api_key)
        response = client.completion(**request_kwargs)
    except Exception as e:  # noqa: BLE001 - normalize all provider errors
        raise PromptingError(f"API request failed: {e}") from e

    latency_ms = (time.time() - start_time) * 1000

    try:
        response_text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        raise PromptingError(f"Unexpected API response format: {e}") from e

    metadata = APIMetadata(
        model=getattr(response, "model", None),
        usage=_to_dict(getattr(response, "usage", None)),
        created=getattr(response, "created", None),
        response_id=getattr(response, "id", None),
        latency_ms=latency_ms,
        raw_response=_to_dict(response),
    )

    return response_text, metadata


def prompt_for_constructs(
    text: str,
    codebook: dict,
    document_id: str,
    config: AnalyzerConfig,
) -> tuple[AnalysisResult, APIMetadata]:
    """Prompt the LLM to identify constructs in text.

    Args:
    ----
        text: Document text to analyze.
        codebook: Codebook dictionary containing construct definitions.
        document_id: Identifier to attach to the parsed analysis result.
        config: Analyzer configuration for prompting, retries, and quote
            matching.

    Returns:
    -------
        Parsed analysis result and API metadata from the successful request.

    Raises:
    ------
        PromptingError: If every request or response parsing attempt fails.

    """
    prompt = construct_prompt(text, codebook, config.prompt_template)

    max_attempts = config.max_retries + 1
    last_metadata: APIMetadata | None = None

    for attempt in range(max_attempts):
        try:
            response_text, metadata = call_llm_api(prompt, config)
            last_metadata = metadata
            result = parse_llm_response(
                response_text,
                document_id,
                text,
                fuzzy_quote_matching=config.fuzzy_quote_matching,
                threshold=config.quote_match_threshold,
            )
            metadata.num_retries = attempt
            return result, metadata

        except PromptingError as e:
            if last_metadata is not None:
                last_metadata.num_retries = attempt

            if attempt < max_attempts - 1:
                time.sleep(1)
                continue

            if last_metadata is None:
                error_metadata = APIMetadata(
                    model=config.model_name,
                    num_retries=attempt,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    error_output=str(e),
                )
            else:
                last_metadata.error_message = str(e)
                last_metadata.error_type = type(e).__name__
                last_metadata.error_output = str(e)
                error_metadata = last_metadata

            raise PromptingError(
                f"Failed after {max_attempts} attempts for document "
                f"'{document_id}'. Last error: {e}",
                metadata=error_metadata,
            ) from e

    raise PromptingError(f"Unexpected failure for document '{document_id}'")
