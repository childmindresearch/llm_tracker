Feel free to use, but this is in beta version and is still being tested.

# LLM Tracker
LLM Tracker is a Python package for identifying psychological constructs in
text data (e.g., interviews, social media posts, chatbot interactions) and comparing LLM-coded results against human-coded results.

The package supports:

- Access to thousands of remote or local LLMs through Mozilla's any-llm API (OpenRouter, Azure, Ollama, etc.)
- Detect every instance of a construct (e.g., anxiety) and return the verbatim quote (e.g., "I'm worried about my cousin")
- Comparing human and LLM codings at the quote level (using LLMs to match human and LLM quotes)
- Computing inter-rater reliability metrics (Kappa, ICC, PABAK) and classification metrics (sensitivity, precision, F1, and PR AUC) and returning summary tables
- Automatically retrying submissions when LLM outputs are not parseable
- Saving analyzer outputs, metadata, and retryable error records
- Flexible loading of csv, txt, docx and preprocessing of dedoose human coding to match LLM coding dataframes. 
- Many new features coming soon: visualizations, automated prompt engineering, and more!

**Please cite this if you use this package:**

Low, D., Mair, P., Nock, M., & Ghosh, S. (2025). Text Psychometrics: Assessing Psychological Constructs in Text Using Natural Language Processing. PsyArxiv. https://osf.io/preprints/psyarxiv/9rdux_v4


## Installation

Install dependencies with Poetry:

```bash
poetry install
```

For tutorial extras such as corpus summaries:

```bash
poetry install --with tutorials
```

## API Key

LLM Tracker uses any-llm API for LLM calls, which can run the most common APIs (OpenRouter, Azure, Ollama). For instance, you can add an OpenRouter API key by adding a few dollars here https://openrouter.ai/. Each LLM call tends to cost a a fraction of a cent (see cost for specific models on OpenRouter). Provide an API key directly:

```python
api_key = "your-openrouter-key"
```

or set it in the environment:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

You can also pass the path to a `.env` file containing:

```text
OPENROUTER_API_KEY=your-openrouter-key
```

## Basic LLM Coding

```python
from llm_tracker import LLMTrackerAnalyzer

analyzer = LLMTrackerAnalyzer(
    api_key=api_key,
    model_name="google/gemini-3-flash-preview",
)

results_llm, metadata_llm, errors_llm = analyzer.analyze_csv(
    csv_path="sample_data.csv",
    codebook_path="codebook.json",
    text_column="post",
    subreddit_column="subreddit",
    author_column="author",
    output_dir="LLM_coding",
)
```

For a directory of supported document files:

```python
results_llm, metadata_llm, errors_llm = analyzer.analyze_directory(
    input_dir="documents",
    codebook_path="codebook.json",
    output_dir="LLM_coding",
)
```

Directory mode supports `.txt` and `.csv` files. Each file becomes one document.

## Human Coding Input

Human coding is loaded into memory and passed directly to the comparer:

```python
from llm_tracker.file_handlers import load_human_coding

human_results = load_human_coding(
    "human_coding.csv",
    doc_id_col="Media Title",
    quote_col="Excerpt Copy",
    range_col="Excerpt Range",
    construct_col="Codes Applied Combined",
)
```

The defaults are designed for Dedoose-style excerpt exports. For other sources,
pass the column names used by your file. The values in `doc_id_col` should match
the document IDs produced by the LLM run.

## Comparing Results

```python
from llm_tracker.comparison import (
    LLMTrackerComparer,
    compute_summary_tables,
    format_concatenated,
    format_weighted_summary,
)

comparer = LLMTrackerComparer(
    api_key=api_key,
    match_model="google/gemini-3-flash-preview",
)

comparison_table = comparer.compare_results(
    human_results=human_results,
    llm_results=results_llm,
    output_dir="comparison_run",
)

per_doc, pooled, weighted = compute_summary_tables(comparison_table)

format_concatenated(pooled)
format_weighted_summary(weighted)
```

The comparison table contains one row per matched, human-only, or LLM-only
construct instance. The human coding is treated as the reference set for
classification metrics.

## Quote Matching

Quote indices are recovered with exact matching by default. Fuzzy quote matching
is available but off by default:

```python
analyzer = LLMTrackerAnalyzer(
    api_key=api_key,
    fuzzy_quote_matching=True,
)
```

Use fuzzy matching when quotes may differ slightly from the source text due to
spacing, punctuation, or small transcription differences.

## Retry Failed Documents

Analyzer runs save error records for failed documents. You can retry them later:

```python
recovered_results, recovered_metadata, remaining_errors = analyzer.retry_errors(
    output_dir="LLM_coding_2026-05-20_120000",
    codebook_path="codebook.json",
)
```

## Tutorial

See [tutorial.ipynb](tutorial.ipynb) for a fuller walkthrough using the
sample data and codebook.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/childmindresearch/llm_tracker/blob/main/tutorial.ipynb)

## Testing

Run the test suite with:

```bash
poetry run pytest
```

The tests avoid real API calls and focus on package behavior, file handling,
comparison logic, configuration, and parsing.

## Data Privacy

This package sends text to an LLM API during analysis and matching. Do not send
identifiable or sensitive data unless it has been properly anonymized and your
API provider's data handling policy is appropriate for your use case.

