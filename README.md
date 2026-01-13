# Pychometrics

A Python package for identifying psychological constructs in text using Large Language Models (LLMs).

## Overview

Pychometrics analyzes interview transcripts and other text documents to identify instances of psychological constructs defined in a codebook. It uses LLM-based analysis via OpenRouter to extract quotes, assign confidence scores, and track speaker information.

## Features

- **Codebook-driven analysis**: Define constructs with names, definitions, and examples
- **Batch processing**: Analyze entire directories of documents
- **Flexible input**: Supports CSV and TXT interview files
- **Structured output**: JSON files with construct instances, quotes, and confidence scores
- **Metadata tracking**: Saves API response metadata for reproducibility
- **Custom prompts**: Override default prompts for specialized analysis
- **Error handling**: Automatic retry with failure logging

## Installation

```bash
# Using pip
pip install pychometrics

# Using Poetry (for development)
git clone https://github.com/your-org/pychometrics.git
cd pychometrics
poetry install
```

## Quick Start

### Python API

```python
from pychometrics import PychometricsAnalyzer

# Initialize the analyzer
analyzer = PychometricsAnalyzer(
    api_key="your-openrouter-api-key",
    model_name="anthropic/claude-3.5-sonnet"
)

# Run analysis
results, metadata = analyzer.analyze_directory(
    input_dir="./interviews",
    codebook_path="./codebook.json",
    output_dir="./results"  # Optional
)

# Results are returned as dict of dicts (one per document)
for doc_id, constructs in results.items():
    print(f"Document: {doc_id}")
    for instance in constructs.get("instances", []):
        print(f"  - {instance['construct']}: {instance['quote']}")
```

### Command Line Interface

```bash
# Basic usage
pychometrics analyze ./interviews ./codebook.json \
    --api-key YOUR_API_KEY \
    --model anthropic/claude-3.5-sonnet

# With custom output directory
pychometrics analyze ./interviews ./codebook.json \
    --api-key YOUR_API_KEY \
    --model anthropic/claude-3.5-sonnet \
    --output-dir my_analysis_results

# Using environment variable for API key
export OPENROUTER_API_KEY=your_key
pychometrics analyze ./interviews ./codebook.json \
    --model anthropic/claude-3.5-sonnet

# With custom prompt
pychometrics analyze ./interviews ./codebook.json \
    --api-key YOUR_API_KEY \
    --model anthropic/claude-3.5-sonnet \
    --prompt-file custom_prompt.txt
```

## Codebook Format

The codebook is a JSON file defining the psychological constructs to identify:

```json
{
    "constructs": [
        {
            "name": "Self-Efficacy",
            "definition": "An individual's belief in their capacity to execute behaviors necessary to produce specific performance attainments.",
            "examples": [
                "I believe I can handle difficult situations",
                "I'm confident in my ability to solve problems"
            ]
        },
        {
            "name": "Growth Mindset",
            "definition": "The belief that abilities and intelligence can be developed through dedication and hard work.",
            "examples": [
                "I can improve with practice",
                "Challenges help me grow"
            ]
        }
    ]
}
```

## Input Document Formats

### TXT Files

Plain text interview transcripts. Speaker identification can be indicated with prefixes:

```
Interviewer: How do you feel about learning new skills?
Participant: I believe that with enough practice, I can learn anything.
Interviewer: Can you give me an example?
Participant: Well, when I started my job, I didn't know Excel at all...
```

### CSV Files

CSV files should have columns for the interview content. Common formats include:
- Single column: Just the transcript text
- Multiple columns: Speaker, Text, Timestamp, etc.

The package will extract and concatenate text content appropriately.

## Output Structure

```
my_analysis_2024-01-15_143022/
├── README.md              # Analysis metadata
├── encodings/             # One JSON per document
│   ├── interview_001.json
│   ├── interview_002.json
│   └── ...
└── metadata/              # API response metadata
    ├── interview_001_meta.json
    ├── interview_002_meta.json
    └── ...
```

### Output JSON Format

Each document produces a JSON file with the following structure:

```json
{
    "document_id": "interview_001",
    "instances": [
        {
            "construct": "Self-Efficacy",
            "speaker_id": "Participant",
            "quote": "I believe that with enough practice, I can learn anything",
            "confidence": 2
        },
        {
            "construct": "Growth Mindset",
            "speaker_id": "Participant",
            "quote": "when I started my job, I didn't know Excel at all, but now I'm quite proficient",
            "confidence": 2
        }
    ]
}
```

### Confidence Scores

- **0**: Construct is not mentioned or is negated
- **1**: Indirect mention or unclear reference
- **2**: Clear and prototypical mention of the construct

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `PYCHOMETRICS_MODEL`: Default model to use

### Custom Prompts

Create a custom prompt file with placeholders:

```
{text}      - Will be replaced with the document text
{codebook}  - Will be replaced with the codebook content
```

## API Reference

### PychometricsAnalyzer

```python
class PychometricsAnalyzer:
    def __init__(
        self,
        api_key: str,
        model_name: str = "anthropic/claude-3.5-sonnet",
        custom_prompt: Optional[str] = None
    ):
        """Initialize the analyzer.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model identifier for OpenRouter
            custom_prompt: Optional custom prompt template
        """
    
    def analyze_document(
        self,
        document_path: Path,
        codebook: dict
    ) -> Tuple[dict, dict]:
        """Analyze a single document.
        
        Args:
            document_path: Path to the document file
            codebook: Parsed codebook dictionary
            
        Returns:
            Tuple of (analysis_result, api_metadata)
        """
    
    def analyze_directory(
        self,
        input_dir: str | Path,
        codebook_path: str | Path,
        output_dir: Optional[str | Path] = None
    ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
        """Analyze all documents in a directory.
        
        Args:
            input_dir: Directory containing documents
            codebook_path: Path to codebook JSON
            output_dir: Optional output directory name prefix
            
        Returns:
            Tuple of (results_dict, metadata_dict)
        """
```

## Error Handling

- Failed API calls are retried once automatically
- Documents that fail after retry are logged in the output README
- The analysis continues with remaining documents

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
