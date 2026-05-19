"""LLM prompting functionality for llm_tracker.

This module handles all interactions with the OpenRouter API, including
constructing prompts, making API requests, parsing responses, and
handling retries.
"""

import difflib
import json
import time
from typing import Optional

import httpx

from llm_tracker.config import AnalyzerConfig
from llm_tracker.models import AnalysisResult, APIMetadata, ConstructInstance


class PromptingError(Exception):
    """Exception raised when prompting fails after all retries."""

    def __init__(self, message: str, metadata: Optional[APIMetadata] = None) -> None:
        super().__init__(message)
        self.metadata = metadata


class ResponseParseError(Exception):
    """Exception raised when response cannot be parsed as valid JSON."""

    pass


def validate_llm_output(response_text: str) -> dict:
    """Validate LLM output into the expected JSON shape."""
    if response_text is None:
        raise ResponseParseError("Response text is empty.")

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ResponseParseError(
            f"Invalid JSON in response (if using an uncommon model, it may not "
            f"support response_format): {e}"
        ) from e

    if (
        not isinstance(data, dict)
        or "instances" not in data
        or not isinstance(data["instances"], list)
    ):
        raise ResponseParseError('JSON must contain an "instances" list.')

    return data


def construct_prompt(text: str, codebook: dict, template: str) -> str:
    """Construct the full prompt by inserting text and codebook.

    Args:
        text: The document text to analyze.
        codebook: The codebook dictionary containing construct definitions.
        template: The prompt template with {text} and {codebook} placeholders.

    Returns:
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
) -> Optional[str]:
    """Find the start:end index of a quote in the source text.

    Args:
        text: The original document text.
        quote: The quote to find.
        fuzzy: If True, fall back to fuzzy matching when exact matching fails.
        threshold: Minimum similarity ratio (0.0 to 1.0) to consider a match.

    Returns:
        String in format "start:end" or None if not found.
    """

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
        response_text: Raw text response from the LLM.
        document_id: The document identifier for the result.
        original_text: The original document text for finding quote indices.
        fuzzy_quote_matching: Whether to use fuzzy matching for quote indices.
        threshold: Minimum similarity ratio for fuzzy quote matching.

    Returns:
        Parsed AnalysisResult object.
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


def call_llm_api(prompt: str, config: AnalyzerConfig) -> tuple[str, APIMetadata]:
    """Make a request to the OpenRouter API.

    Args:
        prompt: The complete prompt to send.
        config: Configuration including API key and model.

    Returns:
        Tuple of (response_text, api_metadata).

    Raises:
        PromptingError: If the API request fails.
    """
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/pychometrics",
        "X-Title": "LLM Tracker",
    }

    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    start_time = time.time()

    with httpx.Client(timeout=config.timeout) as client:
        response = client.post(config.base_url, headers=headers, json=payload)

    latency_ms = (time.time() - start_time) * 1000

    if response.status_code != 200:
        raise PromptingError(
            f"API request failed with status {response.status_code}: {response.text}"
        )

    response_data = response.json()

    try:
        response_text = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise PromptingError(f"Unexpected API response format: {e}") from e

    metadata = APIMetadata(
        model=response_data.get("model"),
        usage=response_data.get("usage"),
        created=response_data.get("created"),
        response_id=response_data.get("id"),
        latency_ms=latency_ms,
        raw_response=response_data,
    )

    return response_text, metadata


def prompt_for_constructs(
    text: str,
    codebook: dict,
    document_id: str,
    config: AnalyzerConfig,
) -> tuple[AnalysisResult, APIMetadata]:
    """Prompt the LLM to identify constructs in text."""
    prompt = construct_prompt(text, codebook, config.prompt_template)

    attempts = 0
    max_attempts = config.max_retries + 1
    last_metadata: Optional[APIMetadata] = None

    while attempts < max_attempts:
        attempts += 1

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
            metadata.num_retries = attempts - 1
            metadata.output_repaired = False
            return result, metadata

        except (PromptingError, ResponseParseError) as e:
            if last_metadata is not None:
                last_metadata.num_retries = attempts - 1
            if attempts < max_attempts:
                time.sleep(1)
                continue
            else:
                if last_metadata is None:
                    error_metadata = APIMetadata(
                        model=config.model_name,
                        num_retries=attempts - 1,
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
