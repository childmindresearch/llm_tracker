"""Command line entry point for the optional local review workbench."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from llm_tracker.review.store import combine_exports, load_dataset


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared launch options used by the CLI and Streamlit app."""
    parser.add_argument(
        "input", type=Path, help="Classification or comparison CSV/XLSX"
    )
    parser.add_argument(
        "--codebook", type=Path, help="JSON codebook or YAML definitions"
    )
    parser.add_argument(
        "--construct", help="Construct for classification files without a column"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("review-data"))


def _write_demo(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "classifications.csv": (
            "participant_id,text,human_classification,machine_classification,"
            "human_reasoning,machine_reasoning,construct\n"
            "demo-001,I wondered why my friend was upset.,1,1,"
            "Considers a cause.,Considers a cause.,perspective_taking\n"
            "demo-002,I saw my friend leave the room.,0,1,"
            "Observation only.,Assumes an unstated cause.,perspective_taking\n"
            "demo-003,I thought about how this might affect my friend.,1,0,"
            "Considers an outcome.,Missed the outcome.,perspective_taking\n"
        ),
        "comparison_rows.csv": (
            "doc_id,construct,status,human_quote,llm_quote\n"
            "demo-001,perspective_taking,matched,"
            "I wondered why my friend was upset.,I wondered why my friend was upset.\n"
            "demo-002,perspective_taking,llm_only,,I saw my friend leave the room.\n"
            "demo-003,perspective_taking,human_only,"
            "I thought about how this might affect my friend.,\n"
        ),
        "codebook.json": json.dumps(
            {
                "perspective_taking": {
                    "definition": (
                        "Considering causes or consequences "
                        "of another person's suffering."
                    ),
                    "examples": ["I wondered why my friend was upset."],
                }
            },
            indent=2,
        )
        + "\n",
    }
    existing = [directory / name for name in files if (directory / name).exists()]
    if existing:
        raise ValueError(f"Demo file already exists: {existing[0]}")
    for name, contents in files.items():
        (directory / name).write_text(contents, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Launch local review, create synthetic demo inputs, or combine exports."""
    parser = argparse.ArgumentParser(
        prog="llm-tracker-review", description="Local manual review; no LLM calls."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create small synthetic example inputs")
    demo.add_argument("directory", type=Path, nargs="?", default=Path("review-demo"))
    run = commands.add_parser("run", help="Launch the review UI on 127.0.0.1")
    add_run_arguments(run)
    run.add_argument("--port", type=int, default=8501)
    run.add_argument("--no-browser", action="store_true")
    combine = commands.add_parser("combine", help="Combine reviewer CSV snapshots")
    combine.add_argument("exports", nargs="+", type=Path)
    combine.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            _write_demo(args.directory)
            print(f"Created synthetic examples in {args.directory.resolve()}")
            return 0
        if args.command == "combine":
            count = combine_exports(args.exports, args.output)
            print(f"Wrote {count} reviewer rows to {args.output.resolve()}")
            return 0
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535.")
        if importlib.util.find_spec("streamlit") is None:
            raise ImportError("streamlit")
        load_dataset(args.input, args.codebook, args.construct)
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).with_name("app.py")),
            "--server.address=127.0.0.1",
            f"--server.port={args.port}",
            f"--server.headless={str(args.no_browser).lower()}",
            "--browser.gatherUsageStats=false",
            "--",
            str(args.input.resolve()),
            "--data-dir",
            str(args.data_dir.resolve()),
        ]
        if args.codebook:
            command.extend(["--codebook", str(args.codebook.resolve())])
        if args.construct:
            command.extend(["--construct", args.construct])
        return subprocess.call(command)
    except ImportError:
        print(
            'Install the review extra: python -m pip install "llm-tracker[review]"',
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(f"Review error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
