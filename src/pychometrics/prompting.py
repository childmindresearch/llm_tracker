"""LLM prompting functionality for pychometrics.

This module handles all interactions with the OpenRouter API, including
constructing prompts, making API requests, parsing responses, and
handling retries.
"""

import json
import time
from typing import Any, Optional
import difflib

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


def find_quote_index(text: str, quote: str, threshold: float = 0.85) -> Optional[str]:
    """Find the start:end index of a quote in the text using fuzzy matching.

    Args:
        text: The original document text.
        quote: The quote to find.
        threshold: Minimum similarity ratio (0.0 to 1.0) to consider a match.

    Returns:
        String in format "start:end" or None if not found.
    """

    start = text.find(quote)
    if start != -1:
        return f"{start}:{start + len(quote)}"

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
    threshold: float = 0.85,
) -> AnalysisResult:
    """Parse the LLM response into an AnalysisResult.

    Args:
        response_text: Raw text response from the LLM.
        document_id: The document identifier for the result.
        original_text: The original document text for finding quote indices.
        threshold: threshold for fuzzy matching quote indicies in text for cases where the llm does not perfectly copy the quote.

    Returns:
        Parsed AnalysisResult object.
    """
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
            quote = item.get("quote", "")
            quote_index = (
                find_quote_index(original_text, quote, threshold)
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
    text: str,
    codebook: dict,
    document_id: str,
    config: AnalyzerConfig,
    threshold: float = 0.85,
) -> tuple[AnalysisResult, APIMetadata]:
    """Prompt the LLM to identify constructs in text."""
    prompt = construct_prompt(text, codebook, config.prompt_template)

    attempts = 0
    max_attempts = config.max_retries + 1

    while attempts < max_attempts:
        attempts += 1

        try:
            response_text, metadata = call_llm_api(prompt, config)
            result = parse_llm_response(response_text, document_id, text, threshold)
            return result, metadata

        except (PromptingError, json.JSONDecodeError) as e:
            if attempts < max_attempts:
                time.sleep(1)
                continue
            else:
                raise PromptingError(
                    f"Failed after {max_attempts} attempts for document "
                    f"'{document_id}'. Last error: {e}"
                ) from e

    raise PromptingError(f"Unexpected failure for document '{document_id}'")
