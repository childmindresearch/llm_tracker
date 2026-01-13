"""Configuration and constants for pychometrics package.

This module contains the default prompt template, configuration class,
and other constants used throughout the package.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_PROMPT = """You are analyzing text to identify and extract instances of psychological constructs.

Text to analyze:
{text}

Codebook of constructs:
{codebook}

Instructions:
1. Identify which constructs from the codebook appear in the text
2. For each construct found, extract ALL instances where it appears
3. For each instance, provide:
   - Speaker ID if available
   - The construct name
   - An exact quote from the text
   - Provide an ordinal score (0=construct is not mentioned or is negated, 1=indirect mention or not clear, 2=clear and prototypical mention of the construct) as to whether the interview clearly mentions the construct, according to its definition and examples.

You MUST respond with ONLY a valid JSON object in exactly this format:
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
- Return ONLY the JSON object, no other text
- Ensure all quotes are properly escaped for JSON
- Include ALL instances found for ALL constructs
- Use null (not "null" or "N/A") for missing speaker_id

Your response:"""


DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 1
REQUEST_TIMEOUT = 120.0


@dataclass
class AnalyzerConfig:
    """Configuration for the PychometricsAnalyzer.
    
    Attributes:
        api_key: OpenRouter API key. If not provided, reads from
            OPENROUTER_API_KEY environment variable.
        model_name: Model identifier for OpenRouter API.
        custom_prompt: Optional custom prompt template with {text} and
            {codebook} placeholders.
        max_retries: Maximum number of retry attempts for failed requests.
        timeout: Request timeout in seconds.
        base_url: Base URL for the API endpoint.
    """
    
    api_key: Optional[str] = None
    model_name: str = DEFAULT_MODEL
    custom_prompt: Optional[str] = None
    max_retries: int = MAX_RETRIES
    timeout: float = REQUEST_TIMEOUT
    base_url: str = OPENROUTER_BASE_URL
    
    def __post_init__(self) -> None:
        """Validate configuration and set defaults from environment."""
        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if self.api_key is None:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "set OPENROUTER_API_KEY environment variable."
            )
    
    @property
    def prompt_template(self) -> str:
        """Get the prompt template to use.
        
        Returns:
            The custom prompt if provided, otherwise the default prompt.
        """
        return self.custom_prompt if self.custom_prompt else DEFAULT_PROMPT
