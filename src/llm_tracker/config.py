"""Configuration and constants for llm_tracker package."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _resolve_api_key(api_key_or_path: str) -> str:
    """Resolve an API key from a raw key or env file path.

    Args:
    ----
        api_key_or_path: Raw API key string or path to an env file containing
            OPENROUTER_API_KEY.

    Returns:
    -------
        Resolved API key string.

    Raises:
    ------
        ValueError: If the env file exists but does not contain
            OPENROUTER_API_KEY, or if the resolved key is empty.

    """
    possible_env_file = Path(api_key_or_path)
    is_env_file = possible_env_file.is_file() and (
        possible_env_file.name == ".env" or possible_env_file.suffix == ".env"
    )
    if not is_env_file:
        return api_key_or_path

    load_dotenv(dotenv_path=possible_env_file, override=True)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            f"OPENROUTER_API_KEY not found or empty in '{possible_env_file}'. "
            "Make sure your env file contains OPENROUTER_API_KEY."
        )
    return api_key


DEFAULT_PROMPT = """You are analyzing text to identify and extract instances
of psychological constructs.

Text to analyze:
{text}

Codebook of constructs:
{codebook}

Instructions:
1. Identify which constructs from the codebook appear in the text.
2. For each construct found, extract all instances where it appears.
3. For each instance, provide:
   - Speaker ID if available
   - The construct name
   - An exact quote from the text
   - An ordinal confidence score:
     0 = construct is not mentioned or is negated
     1 = indirect mention or not clear
     2 = clear and prototypical mention of the construct

You must respond with only a valid JSON object in exactly this format:
{{
    "instances": [
        {{
            "construct": "<construct name>",
            "speaker_id": "<speaker ID or null if not available>",
            "quote": "<exact quote from text>",
            "confidence": <0, 1, or 2>
        }}
    ]
}}

If no constructs are found, return: {{"instances": []}}

Important:
- Return only the JSON object, no other text.
- Ensure all quotes are properly escaped for JSON.
- Include all instances found for all constructs.
- Use null for missing speaker_id.
- Quotes must be copied exactly from the text, character for character.
- Do not change capitalization, punctuation, ellipses, or other characters.

Your response:"""


DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_PROVIDER = "openrouter"
MAX_RETRIES = 0
REQUEST_TIMEOUT = 120.0


@dataclass
class AnalyzerConfig:
    """Configuration for analyzer and matcher LLM requests.

    Attributes:
    ----------
        api_key: API key, path to an env file, or None to read from the
            OPENROUTER_API_KEY environment variable.
        model_name: Model identifier passed to the provider (e.g. an
            OpenRouter model id such as "google/gemini-3-flash-preview").
        provider: any-llm provider id used to route the request. Defaults to
            "openrouter"; change this (and supply the matching key) to call a
            provider natively.
        custom_prompt: Optional prompt template with text and codebook fields.
        max_retries: Number of retry attempts after the first failed request.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature. None lets the provider apply its own
            default; set 0 for (best-effort) deterministic output.
        fuzzy_quote_matching: Whether to use fuzzy quote index recovery.
        quote_match_threshold: Minimum fuzzy match score for quote recovery.

    """

    api_key: str | None = None
    model_name: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    custom_prompt: str | None = None
    max_retries: int = MAX_RETRIES
    timeout: float = REQUEST_TIMEOUT
    temperature: float | None = 0.0
    fuzzy_quote_matching: bool = False
    quote_match_threshold: float = 0.85

    def __post_init__(self) -> None:
        """Resolve the API key after dataclass initialization."""
        self.api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        """Load the API key from config, env file, or environment variable.

        Returns:
        -------
            Resolved API key string.

        Raises:
        ------
            ValueError: If no API key is provided or available from the
                environment.

        """
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key is required. Provide api_key or set OPENROUTER_API_KEY."
            )
        return _resolve_api_key(api_key)

    @property
    def prompt_template(self) -> str:
        """Return the custom prompt template or the default prompt.

        Returns:
        -------
            Prompt template used for document coding.

        """
        return self.custom_prompt or DEFAULT_PROMPT
