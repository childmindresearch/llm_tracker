"""LLM prompting functionality for llm_tracker.

This module handles all interactions with LLM providers (via the any-llm
SDK), including constructing prompts, making completion requests, parsing
responses, and handling retries.
"""

import atexit
import difflib
import json
import math
import os
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import NoReturn

import httpx
from any_llm import AnyLLM

from llm_tracker.config import AnalyzerConfig
from llm_tracker.models import AnalysisResult, APIMetadata, ConstructInstance


class PromptingError(Exception):
    """Exception raised when prompting fails after all retries."""

    def __init__(
        self,
        message: str,
        metadata: APIMetadata | None = None,
        *,
        retryable: bool = True,
        retry_after: float | None = None,
    ) -> None:
        """Create a prompting error.

        Args:
        ----
            message: Error message describing the failure.
            metadata: Optional API metadata captured before the failure.
            retryable: Whether repeating the request may recover the failure.
            retry_after: Provider-requested minimum retry delay in seconds.

        """
        super().__init__(message)
        self.metadata = metadata
        self.retryable = retryable
        self.retry_after = retry_after


def wait_before_retry(config: AnalyzerConfig, attempt: int, error: Exception) -> None:
    """Wait with capped exponential backoff, honoring a longer Retry-After."""
    delay = min(config.retry_max_delay, config.retry_delay * 2 ** min(attempt, 30))
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        delay = max(delay, retry_after)
    if delay > 0:
        time.sleep(delay)


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
        result_dict: dict = dict(obj)  # type: ignore[call-overload]
        return result_dict
    except (TypeError, ValueError):
        return None


_CLIENT_CACHE: dict[tuple[str, str], AnyLLM] = {}
_OPENROUTER_CLIENTS: dict[tuple[str, str], httpx.Client] = {}


def _get_openrouter_client(config: AnalyzerConfig) -> httpx.Client:
    """Reuse a synchronous HTTP client without OpenAI response validation."""
    base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
    key = (base, config.api_key or "")
    client = _OPENROUTER_CLIENTS.get(key)
    if client is None:
        client = httpx.Client(
            base_url=base + "/",
            headers={"Authorization": f"Bearer {config.api_key}"},
        )
        _OPENROUTER_CLIENTS[key] = client
    return client


def _retry_after_seconds(value: str | None) -> float | None:
    """Read a Retry-After header expressed in seconds or as an HTTP date."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            seconds = (date - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _optional_int(value: object) -> int | None:
    """Normalize optional provider metadata without rejecting an answer."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (ValueError, TypeError, OverflowError):
        return None


def _call_openrouter(request: dict, config: AnalyzerConfig) -> tuple[str, APIMetadata]:
    """Check HTTP and body-level errors before reading completion content.

    OpenRouter can report failures in HTTP 200 responses, at the top level or
    inside choices. Never accept partial content from an error completion.
    """
    started = time.monotonic()
    try:
        response = _get_openrouter_client(config).post(
            "chat/completions", json=request, timeout=config.timeout
        )
    except httpx.RequestError as error:
        metadata = APIMetadata(
            model=config.model_name,
            provider="openrouter",
            latency_ms=(time.monotonic() - started) * 1000,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise PromptingError(
            f"OpenRouter request failed: {error}", metadata=metadata
        ) from error
    except (ImportError, ValueError) as error:
        raise PromptingError(
            f"OpenRouter client configuration failed: {error}", retryable=False
        ) from error

    metadata = APIMetadata(
        model=config.model_name,
        provider="openrouter",
        http_status=response.status_code,
        latency_ms=(time.monotonic() - started) * 1000,
    )
    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))

    def fail(message: str, retryable: bool = True) -> NoReturn:
        metadata.error_message = message
        metadata.error_type = "OpenRouterError"
        raise PromptingError(
            message, metadata, retryable=retryable, retry_after=retry_after
        )

    try:
        body = response.json()
    except ValueError:
        metadata.error_output = response.text[:4000]
        fail(
            f"OpenRouter returned non-JSON content (HTTP {response.status_code}).",
            response.status_code not in {400, 401, 402, 403, 404, 405, 413, 422},
        )
    if not isinstance(body, dict):
        fail(
            "OpenRouter returned a non-object JSON response "
            f"(HTTP {response.status_code})."
        )

    metadata.raw_response = body
    metadata.model = str(body.get("model") or config.model_name)
    metadata.response_id = str(body["id"]) if body.get("id") is not None else None
    metadata.created = _optional_int(body.get("created"))
    metadata.usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    choices = body.get("choices")
    choices = choices if isinstance(choices, list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    reason = first.get("finish_reason")
    metadata.finish_reason = str(reason) if reason is not None else None

    provider_error = body.get("error")
    failed_choice = False
    for choice in choices:
        if isinstance(choice, dict):
            if (
                choice.get("error") is not None
                or choice.get("finish_reason") == "error"
            ):
                failed_choice = True
                metadata.finish_reason = str(choice.get("finish_reason") or "error")
                if provider_error is None:
                    provider_error = choice.get("error")
    if not response.is_success or provider_error is not None or failed_choice:
        details = provider_error if isinstance(provider_error, dict) else {}
        code = _optional_int(details.get("code")) or response.status_code
        message = (
            details.get("message")
            or provider_error
            or "Completion failed; no provider explanation supplied."
        )
        extra = details.get("metadata")
        if isinstance(extra, dict) and extra.get("raw"):
            message = f"{message}; provider details: {str(extra['raw'])[:2000]}"
        metadata.error_output = json.dumps(provider_error, ensure_ascii=False)
        fail(
            f"OpenRouter error (HTTP {response.status_code}, code {code}, "
            f"generation {metadata.response_id or 'unknown'}): {message}",
            code not in {400, 401, 402, 403, 404, 405, 413, 422},
        )

    if reason in {"length", "content_filter"}:
        fail(
            f"OpenRouter completion ended with {reason!r}; "
            "no complete answer available.",
            False,
        )
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        fail("OpenRouter returned no usable completion text.")
    return content, metadata


def _get_client(config: AnalyzerConfig) -> AnyLLM:
    """Return a cached client for this provider and API key.

    any-llm creates a fresh event loop for every synchronous request, and each
    client owns an httpx connection pool that is closed asynchronously. Building
    a client per call therefore leaves an orphaned close task behind each time,
    which surfaces as "Task exception was never retrieved" / "Event loop is
    closed" tracebacks after a run. Reusing one client also avoids repeating the
    TCP and TLS handshake for every document.

    The cache is keyed on provider and API key, so analyzers configured with
    different credentials do not share a client. Because a cached client lives
    for the life of the process, mutating ``config.api_key`` in place after a
    request will not take effect; build a new AnalyzerConfig instead.

    Args:
    ----
        config: Configuration providing the provider id and API key.

    Returns:
    -------
        A client for the configured provider.

    """
    key = (config.provider, config.api_key or "")
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = AnyLLM.create(config.provider, api_key=config.api_key)
        _CLIENT_CACHE[key] = client
    return client


def reset_client_cache() -> None:
    """Discard all cached provider clients.

    Call this if a client ends up in a bad state -- for example after a
    connection error, or if the event loop it was created on has been torn
    down. The next request rebuilds the client it needs.
    """
    _CLIENT_CACHE.clear()
    for client in _OPENROUTER_CLIENTS.values():
        client.close()
    _OPENROUTER_CLIENTS.clear()


atexit.register(reset_client_cache)


def call_llm_api(prompt: str, config: AnalyzerConfig) -> tuple[str, APIMetadata]:
    """Make a completion request, preserving OpenRouter error responses.

    OpenRouter uses synchronous HTTP so its error envelopes are checked before
    parsing content. Other providers use any-llm. The return shape is shared.

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

    if config.provider == "openrouter":
        return _call_openrouter(request_kwargs, config)

    start_time = time.time()

    try:
        client = _get_client(config)
        response = client.completion(**request_kwargs, timeout=config.timeout)
    except Exception as e:  # noqa: BLE001 - normalize all provider errors
        # A failed request may have left the cached client's connection pool
        # unusable, so drop it rather than reusing it for the retry.
        _CLIENT_CACHE.pop((config.provider, config.api_key or ""), None)
        raise PromptingError(
            f"API request failed: {e}",
            retryable=getattr(e, "status_code", None)
            not in {400, 401, 402, 403, 404, 405, 413, 422},
        ) from e

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
        last_metadata = None
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
            if e.metadata is not None:
                last_metadata = e.metadata
            if last_metadata is not None:
                last_metadata.num_retries = attempt

            if attempt < max_attempts - 1 and e.retryable:
                wait_before_retry(config, attempt, e)
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
                f"Failed after {attempt + 1} attempts for document "
                f"'{document_id}'. Last error: {e}",
                metadata=error_metadata,
                retryable=e.retryable,
                retry_after=e.retry_after,
            ) from e

    raise PromptingError(f"Unexpected failure for document '{document_id}'")
