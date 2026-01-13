"""LLM prompting functionality for pychometrics.

This module handles all interactions with the OpenRouter API, including
constructing prompts, making API requests, parsing responses, and
handling retries.
"""

import json
import time
from typing import Any, Optional

import httpx

from pychometrics.config import AnalyzerConfig
from pychometrics.models import APIMetadata, AnalysisResult, ConstructInstance


class PromptingError(Exception):
    """Exception raised when prompting fails after all retries."""

    pass


class ResponseParseError(Exception):
    """Exception raised when response cannot be parsed as valid JSON."""

    pass


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


def parse_llm_response(response_text: str, document_id: str) -> AnalysisResult:
    """Parse the LLM response into an AnalysisResult."""
    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    data = json.loads(cleaned)

    instances = []
    raw_instances = data.get("instances", [])

    for item in raw_instances:
        try:
            instance = ConstructInstance(
                construct=item.get("construct", "Unknown"),
                speaker_id=item.get("speaker_id"),
                quote=item.get("quote", ""),
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
        "X-Title": "Pychometrics",
    }

    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": prompt}],
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
    text: str, codebook: dict, document_id: str, config: AnalyzerConfig
) -> tuple[AnalysisResult, APIMetadata]:
    """Prompt the LLM to identify constructs in text.

    This function handles the complete prompting workflow including
    constructing the prompt, calling the API, parsing the response,
    and retrying on failure.

    Args:
        text: The document text to analyze.
        codebook: The codebook dictionary with construct definitions.
        document_id: Identifier for the document being analyzed.
        config: Configuration for the analyzer.

    Returns:
        Tuple of (AnalysisResult, APIMetadata).

    Raises:
        PromptingError: If prompting fails after all retry attempts.
    """
    prompt = construct_prompt(text, codebook, config.prompt_template)

    last_error: Optional[Exception] = None
    attempts = 0
    max_attempts = config.max_retries + 1

    while attempts < max_attempts:
        attempts += 1

        try:
            response_text, metadata = call_llm_api(prompt, config)
            result = parse_llm_response(response_text, document_id)
            return result, metadata

        except (PromptingError, ResponseParseError) as e:
            last_error = e
            if attempts < max_attempts:
                time.sleep(1)
                continue
            else:
                raise PromptingError(
                    f"Failed after {max_attempts} attempts for document "
                    f"'{document_id}'. Last error: {e}"
                ) from e

    raise PromptingError(f"Unexpected failure for document '{document_id}'")
