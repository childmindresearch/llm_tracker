"""Validate review inputs and persist independent, resumable local sessions."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

CLASSIFICATION_COLUMNS = {
    "participant_id",
    "text",
    "human_classification",
    "machine_classification",
}
COMPARISON_COLUMNS = {"doc_id", "construct", "status", "human_quote", "llm_quote"}
BinaryLabel = Literal[0, 1]
Assessment = Literal["match", "no_match", "accept", "reject"]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def normalize_label(value: Any) -> BinaryLabel | None:
    """Parse a binary label; retain missing values and reject other values."""
    text = str(value).strip().lower() if value is not None else ""
    if text in {"", "nan", "<na>"}:
        return None
    if text in {"1", "1.0", "yes", "present", "true"}:
        return 1
    if text in {"0", "0.0", "no", "not present", "absent", "false"}:
        return 0
    raise ValueError(f"Expected a binary label (0/1, yes/no), got {value!r}.")


def normalize_reviewer(value: str) -> str:
    """Normalize a reviewer identifier and prevent it from becoming a path."""
    reviewer = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.@-]{0,63}", reviewer):
        raise ValueError(
            "Reviewer ID must be 1–64 characters: letters, numbers, _, ., @, or -."
        )
    return reviewer


def load_definitions(path: Path | None) -> dict[str, Any]:
    """Load flat/enveloped JSON codebooks or local YAML prompt definitions."""
    if path is None:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML codebook: {path}") from exc
        # Convert YAML dates and other scalar values to JSON-safe strings.
        data = json.loads(json.dumps(data, default=str))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Codebook must be .json, .yaml, or .yml.")
    if isinstance(data, dict) and "codebook" in data and "metadata" in data:
        data = data["codebook"]
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, dict) for key, value in data.items()
    ):
        raise ValueError("Codebook must map construct names to definition objects.")
    for name, entry in data.items():
        for field in ("definition", "prompt_text"):
            if field in entry and not isinstance(entry[field], str):
                raise ValueError(f"Codebook {name}: {field} must be text.")
        if "examples" in entry and not (
            isinstance(entry["examples"], list)
            and all(isinstance(example, str) for example in entry["examples"])
        ):
            raise ValueError(f"Codebook {name}: examples must be a list of text.")
    return dict(data)


@dataclass(frozen=True)
class ReviewDataset:
    """Validated rows and the content identity used to isolate review sessions."""

    source_name: str
    kind: Literal["classification", "comparison"]
    rows: list[dict[str, Any]]
    definitions: dict[str, Any]
    dataset_hash: str


def load_dataset(
    path: Path, codebook: Path | None = None, construct: str | None = None
) -> ReviewDataset:
    """Read a CSV/XLSX input, preserving string IDs, ordering, and duplicate rows.

    Classification inputs use ``participant_id``, ``text``, and human/machine
    binary classifications. Quote inputs use the native ``comparison_rows.csv``
    schema. A changed row, row order, construct override, or codebook creates a
    separate dataset identity, so old decisions cannot silently attach to it.
    """
    payload = path.read_bytes()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    elif path.suffix.lower() == ".xlsx":
        try:
            frame = pd.read_excel(
                io.BytesIO(payload), dtype=str, keep_default_na=False, engine="openpyxl"
            )
        except BadZipFile as exc:
            raise ValueError(f"Invalid XLSX workbook: {path}") from exc
    else:
        raise ValueError("Review input must be a .csv or .xlsx file.")
    if frame.empty:
        raise ValueError("Review input contains no rows.")
    kind: Literal["classification", "comparison"]
    if CLASSIFICATION_COLUMNS <= set(frame.columns):
        kind = "classification"
    elif COMPARISON_COLUMNS <= set(frame.columns):
        kind = "comparison"
    else:
        raise ValueError(
            "Unrecognized review input. Classification columns: "
            f"{', '.join(sorted(CLASSIFICATION_COLUMNS))}. Comparison columns: "
            f"{', '.join(sorted(COMPARISON_COLUMNS))}."
        )
    definitions = load_definitions(codebook)
    rows: list[dict[str, Any]] = []
    occurrences: Counter[str] = Counter()
    for index, (_, series) in enumerate(frame.iterrows()):
        source = {str(key): str(value) for key, value in series.items()}
        name = (source.get("construct") or construct or "").strip()
        if not name:
            raise ValueError("Provide a construct column or use --construct NAME.")
        if kind == "classification":
            if not source["text"].strip():
                raise ValueError(f"Row {index + 1}: text is empty.")
            for column in ("human_classification", "machine_classification"):
                try:
                    normalize_label(source[column])
                except ValueError as exc:
                    raise ValueError(f"Row {index + 1}, {column}: {exc}") from exc
        else:
            status = source["status"]
            if status not in {"matched", "human_only", "llm_only"}:
                raise ValueError(
                    f"Row {index + 1}: invalid comparison status {status!r}."
                )
            for column, expected in (
                ("human_quote", status != "llm_only"),
                ("llm_quote", status != "human_only"),
            ):
                if bool(source[column].strip()) != expected:
                    raise ValueError(
                        f"Row {index + 1}: {column} conflicts with {status}."
                    )
        signature = _digest({"construct": name, "source": source})
        occurrences[signature] += 1
        rows.append(
            {
                "row_id": _digest([signature, occurrences[signature]]),
                "row_number": len(rows) + 1,
                "construct": name,
                "source": source,
            }
        )
    identity = _digest({"kind": kind, "rows": rows, "definitions": definitions})
    return ReviewDataset(path.name, kind, rows, definitions, identity)


class ReviewDecision(BaseModel):
    """One draft or completed review, including its explicit completion state."""

    status: Literal["pending", "completed"] = "pending"
    assessment: Assessment | None = None
    adjudicated_label: BinaryLabel | None = None
    notes: str = ""
    needs_follow_up: bool = False
    completed_at: str | None = None
    last_saved_at: str | None = None


class ReviewSession(BaseModel):
    """Versioned decisions belonging to exactly one reviewer and dataset."""

    schema_version: Literal[1] = 1
    dataset_hash: str
    reviewer_id: str
    revision: int = 0
    created_at: str = Field(default_factory=_timestamp)
    updated_at: str = Field(default_factory=_timestamp)
    reviews: dict[str, ReviewDecision] = Field(default_factory=dict)


def set_review(
    dataset: ReviewDataset,
    session: ReviewSession,
    row_id: str,
    decision: ReviewDecision,
    *,
    complete: bool = False,
) -> ReviewSession:
    """Return a changed session after validating a draft or completion.

    Editing a completed decision as a draft reopens it. Unknown labels cannot
    count as agreement; a disagreement needs a final label and reviewer notes.
    Quote comparisons use accept/reject of the original matching outcome.
    """
    if session.dataset_hash != dataset.dataset_hash:
        raise ValueError("Session does not belong to this dataset.")
    row = next((row for row in dataset.rows if row["row_id"] == row_id), None)
    if row is None:
        raise ValueError("Unknown review row.")
    updated = decision.model_copy(deep=True)
    updated.notes = updated.notes.strip()
    allowed = (
        {"match", "no_match"}
        if dataset.kind == "classification"
        else {"accept", "reject"}
    )
    if updated.assessment is not None and updated.assessment not in allowed:
        raise ValueError("Review decision does not match this input type.")
    if dataset.kind == "comparison":
        updated.adjudicated_label = None
    elif updated.assessment == "match":
        human = normalize_label(row["source"]["human_classification"])
        machine = normalize_label(row["source"]["machine_classification"])
        if human is None or machine is None or human != machine:
            raise ValueError("Match requires two known, equal binary labels.")
        updated.adjudicated_label = human
    if complete:
        if updated.assessment is None:
            raise ValueError("Choose a review decision before completing this row.")
        if updated.assessment in {"no_match", "reject"} and not updated.notes:
            raise ValueError(
                "Add reviewer notes for a disagreement or rejected comparison."
            )
        if updated.assessment == "no_match" and updated.adjudicated_label is None:
            raise ValueError(
                "Choose a final Present/Not present label for a disagreement."
            )
    updated.status = "completed" if complete else "pending"
    updated.completed_at = _timestamp() if complete else None
    updated.last_saved_at = _timestamp()
    result = session.model_copy(deep=True)
    result.reviews[row_id] = updated
    return result


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_rows(dataset: ReviewDataset, session: ReviewSession) -> pd.DataFrame:
    """Flatten decisions with prefixed source columns, retaining pending rows."""
    records = []
    for row in dataset.rows:
        review = session.reviews.get(row["row_id"], ReviewDecision())
        records.append(
            {
                "review_schema_version": 1,
                "dataset_hash": dataset.dataset_hash,
                "reviewer_id": session.reviewer_id,
                "row_id": row["row_id"],
                "row_number": row["row_number"],
                "construct": row["construct"],
                "input_kind": dataset.kind,
                "source_name": dataset.source_name,
                **{f"source_{key}": value for key, value in row["source"].items()},
                **{
                    f"review_{key}": value for key, value in review.model_dump().items()
                },
            }
        )
    return pd.DataFrame(records)


def csv_text(frame: pd.DataFrame) -> str:
    """Create CSV text, escaping spreadsheet formulas in free text cells."""

    def escape(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(
            ("=", "+", "-", "@", "\t", "\r")
        ):
            return "'" + value
        return value

    return str(frame.apply(lambda column: column.map(escape)).to_csv(index=False))


class SessionStore:
    """Store sessions outside the package, with atomic saves and stale-tab checks."""

    def __init__(
        self, data_dir: Path, dataset: ReviewDataset, reviewer_id: str
    ) -> None:
        """Select local storage for one dataset and normalized reviewer ID."""
        self.dataset = dataset
        self.reviewer_id = normalize_reviewer(reviewer_id)
        self.path = data_dir / dataset.dataset_hash / f"{self.reviewer_id}.json"
        self.export_path = self.path.with_suffix(".csv")

    def load(self) -> ReviewSession:
        """Resume a session; refuse to overwrite a corrupt or incompatible file."""
        if not self.path.exists():
            return ReviewSession(
                dataset_hash=self.dataset.dataset_hash, reviewer_id=self.reviewer_id
            )
        try:
            session = ReviewSession.model_validate_json(self.path.read_text("utf-8"))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid session {self.path}; restore it from backup."
            ) from exc
        if (
            session.dataset_hash != self.dataset.dataset_hash
            or session.reviewer_id != self.reviewer_id
            or not set(session.reviews) <= {row["row_id"] for row in self.dataset.rows}
        ):
            raise ValueError(f"Session identity does not match input: {self.path}")
        return session

    def save(self, session: ReviewSession) -> ReviewSession:
        """Persist JSON then CSV, rejecting concurrent writes from stale sessions."""
        if (
            session.dataset_hash != self.dataset.dataset_hash
            or session.reviewer_id != self.reviewer_id
        ):
            raise ValueError("Session identity does not match this store.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError(
                "Another save is in progress. Retry after it finishes."
            ) from exc
        os.close(descriptor)
        try:
            current = self.load()
            if current.revision != session.revision:
                raise ValueError(
                    "Session changed in another tab. Reload before saving."
                )
            saved = session.model_copy(deep=True)
            saved.revision += 1
            saved.updated_at = _timestamp()
            _atomic_write(self.path, saved.model_dump_json(indent=2) + "\n")
            _atomic_write(self.export_path, csv_text(export_rows(self.dataset, saved)))
            return saved
        finally:
            lock_path.unlink(missing_ok=True)


def combine_exports(paths: list[Path], output: Path) -> int:
    """Combine CSV exports, retaining the newest snapshot per reviewer and row.

    Reviewers and dataset versions remain independent. This operation does not
    infer consensus or change comparison metrics. Existing outputs are refused.
    """
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    required = {
        "review_schema_version",
        "dataset_hash",
        "reviewer_id",
        "row_id",
        "review_status",
        "review_last_saved_at",
    }
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not required <= set(frame.columns) or set(
            frame["review_schema_version"]
        ) != {"1"}:
            raise ValueError(f"Not a version 1 review export: {path}")
        frames.append(frame)
    if not frames:
        raise ValueError("Provide at least one review export.")
    combined = pd.concat(frames, ignore_index=True).fillna("")
    key = ["dataset_hash", "reviewer_id", "row_id"]
    duplicates = combined[
        combined.duplicated([*key, "review_last_saved_at"], keep=False)
    ]
    if duplicates.drop_duplicates().duplicated([*key, "review_last_saved_at"]).any():
        raise ValueError(
            "Conflicting reviews have the same save time; resolve them first."
        )
    combined = combined.sort_values("review_last_saved_at").drop_duplicates(
        key, keep="last"
    )
    _atomic_write(output, csv_text(combined))
    return len(combined)
