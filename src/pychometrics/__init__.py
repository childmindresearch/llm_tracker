"""Pychometrics: Psychological construct identification using LLM analysis.

This package provides tools for analyzing interview transcripts and text documents
to identify instances of psychological constructs defined in a codebook.

Example:
    from pychometrics import PychometricsAnalyzer

    analyzer = PychometricsAnalyzer(
        api_key="your-api-key",
        model_name="anthropic/claude-3.5-sonnet"
    )
    
    results, metadata = analyzer.analyze_directory(
        input_dir="./interviews",
        codebook_path="./codebook.json"
    )
"""

from pychometrics.analyzer import PychometricsAnalyzer
from pychometrics.config import AnalyzerConfig
from pychometrics.models import ConstructInstance, AnalysisResult, Codebook

__version__ = "0.1.0"
__all__ = [
    "PychometricsAnalyzer",
    "AnalyzerConfig",
    "ConstructInstance",
    "AnalysisResult",
    "Codebook",
]
