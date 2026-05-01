"""Pytest fixtures for llm_tracker tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_codebook() -> dict:
    """Sample codebook for testing."""
    return {
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


@pytest.fixture
def sample_interview_txt() -> str:
    """Sample interview text for testing."""
    return """Interviewer: How do you approach learning new skills?

Participant: Well, I really believe that with enough practice, I can learn just about anything. When I started this job, I didn't know anything about data analysis, but I worked hard at it and now I'm quite proficient.

Interviewer: Can you tell me about a challenging situation you faced?

Participant: Last year we had a major project crisis. At first it seemed overwhelming, but I told myself I could handle it. I broke it down into smaller tasks and tackled each one. In the end, we delivered on time.

Interviewer: How do you handle setbacks?

Participant: I see setbacks as learning opportunities. Every failure teaches me something new. I don't give up easily - I keep trying different approaches until something works."""


@pytest.fixture
def sample_interview_csv_content() -> str:
    """Sample CSV content for testing."""
    return """speaker,text
Interviewer,How do you approach learning new skills?
Participant,I believe practice makes perfect. I can improve at anything with dedication.
Interviewer,What about challenges?
Participant,Challenges help me grow. I'm confident I can handle most situations."""


@pytest.fixture
def temp_input_dir(tmp_path: Path, sample_interview_txt: str) -> Path:
    """Create a temporary input directory with sample files."""
    input_dir = tmp_path / "interviews"
    input_dir.mkdir()
    
    # Create sample TXT file
    (input_dir / "interview_001.txt").write_text(sample_interview_txt)
    
    # Create sample CSV file
    csv_content = """speaker,text
Interviewer,Tell me about yourself.
Participant,I'm a dedicated learner who believes in continuous improvement."""
    (input_dir / "interview_002.csv").write_text(csv_content)
    
    return input_dir


@pytest.fixture
def temp_codebook_file(tmp_path: Path, sample_codebook: dict) -> Path:
    """Create a temporary codebook file."""
    codebook_file = tmp_path / "codebook.json"
    with open(codebook_file, "w") as f:
        json.dump(sample_codebook, f, indent=2)
    return codebook_file


@pytest.fixture
def mock_llm_response() -> str:
    """Sample LLM response for testing."""
    return json.dumps({
        "instances": [
            {
                "construct": "Self-Efficacy",
                "speaker_id": "Participant",
                "quote": "I believe practice makes perfect",
                "confidence": 2
            },
            {
                "construct": "Growth Mindset",
                "speaker_id": "Participant",
                "quote": "I can improve at anything with dedication",
                "confidence": 2
            }
        ]
    })
