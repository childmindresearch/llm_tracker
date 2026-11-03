"""Review input validation, persistence, exports, and command line behavior."""

import json
from pathlib import Path

import pandas as pd
import pytest
from llm_tracker.review.cli import main
from llm_tracker.review.store import (
    ReviewDecision,
    SessionStore,
    combine_exports,
    export_rows,
    load_dataset,
    normalize_label,
    normalize_reviewer,
    set_review,
)


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    assert main(["demo", str(tmp_path)]) == 0
    return tmp_path


def test_demo_refuses_to_overwrite_existing_files(inputs: Path) -> None:
    path = inputs / "classifications.csv"
    original = path.read_bytes()
    assert main(["demo", str(inputs)]) == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("value", [2, -1, 0.5, "maybe", "unknown"])
def test_non_binary_labels_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="binary label"):
        normalize_label(value)


@pytest.mark.parametrize(
    "value,expected", [("yes", 1), ("FALSE", 0), ("1.0", 1), (None, None), ("", None)]
)
def test_binary_labels_and_unknowns(value, expected) -> None:
    assert normalize_label(value) == expected


@pytest.mark.parametrize(
    "reviewer", ["", "../someone", "/tmp/x", "a/b", "a\\b", ".", "a" * 65]
)
def test_reviewer_cannot_escape_storage(reviewer: str) -> None:
    with pytest.raises(ValueError, match="Reviewer ID"):
        normalize_reviewer(reviewer)


def test_loader_preserves_ids_and_independent_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text(
        "participant_id,text,human_classification,machine_classification\n"
        "0001,NA,1,1\n0001,NA,1,1\nNA,Some text,,0\n",
        encoding="utf-8",
    )
    dataset = load_dataset(path, construct="test")
    assert dataset.rows[0]["source"]["participant_id"] == "0001"
    assert dataset.rows[0]["source"]["text"] == "NA"
    assert dataset.rows[2]["source"]["participant_id"] == "NA"
    assert len({row["row_id"] for row in dataset.rows}) == 3
    assert load_dataset(path, construct="test").dataset_hash == dataset.dataset_hash
    with pytest.raises(ValueError, match="construct column"):
        load_dataset(path)


def test_xlsx_and_csv_have_same_identity(inputs: Path) -> None:
    pytest.importorskip("openpyxl")
    csv = inputs / "classifications.csv"
    workbook = inputs / "classifications.xlsx"
    pd.read_csv(csv, dtype=str).to_excel(workbook, index=False)
    assert load_dataset(csv).dataset_hash == load_dataset(workbook).dataset_hash


def test_json_envelope_and_yaml_definitions(inputs: Path) -> None:
    pytest.importorskip("yaml")
    csv = inputs / "classifications.csv"
    codebook = inputs / "codebook.json"
    flat = load_dataset(csv, codebook)
    envelope = inputs / "envelope.json"
    envelope.write_text(json.dumps({"metadata": {}, "codebook": flat.definitions}))
    assert load_dataset(csv, envelope).dataset_hash == flat.dataset_hash
    yaml_path = inputs / "definitions.yaml"
    yaml_path.write_text(
        "perspective_taking:\n  prompt_text: Example prompt\n"
        "  date_selected: 2026-01-01\n"
    )
    assert (
        load_dataset(csv, yaml_path).definitions["perspective_taking"]["date_selected"]
        == "2026-01-01"
    )
    assert load_dataset(csv, yaml_path).dataset_hash != flat.dataset_hash


def test_native_comparison_output_is_supported(tmp_path: Path) -> None:
    from llm_tracker.comparison import LLMTrackerComparer
    from llm_tracker.models import AnalysisResult, ConstructInstance

    comparer = object.__new__(LLMTrackerComparer)
    human = {
        "doc-1": AnalysisResult(
            document_id="doc-1",
            instances=[ConstructInstance(construct="test", quote="A human quote")],
        )
    }
    frame = comparer.compare_results(human, {})
    path = tmp_path / "comparison_rows.csv"
    frame.to_csv(path, index=False)
    dataset = load_dataset(path)
    assert dataset.kind == "comparison"
    assert dataset.rows[0]["source"]["status"] == "human_only"
    store = SessionStore(tmp_path / "saved", dataset, "Reviewer")
    session = set_review(
        dataset,
        store.load(),
        dataset.rows[0]["row_id"],
        ReviewDecision(assessment="accept"),
        complete=True,
    )
    assert store.save(session).reviews[dataset.rows[0]["row_id"]].status == "completed"


def test_draft_completion_resume_and_reopen(inputs: Path) -> None:
    dataset = load_dataset(inputs / "classifications.csv")
    store = SessionStore(inputs / "saved", dataset, " Reviewer_A ")
    session = store.load()
    row_id = dataset.rows[0]["row_id"]
    draft = set_review(dataset, session, row_id, ReviewDecision(notes="Draft"))
    saved = store.save(draft)
    assert not session.reviews
    assert store.load() == saved
    assert saved.reviews[row_id].status == "pending"
    completed = set_review(
        dataset, saved, row_id, ReviewDecision(assessment="match"), complete=True
    )
    completed = store.save(completed)
    assert completed.reviews[row_id].adjudicated_label == 1
    assert completed.reviews[row_id].completed_at
    exported = pd.read_csv(store.export_path)
    assert len(exported) == 3
    assert exported.loc[0, "review_status"] == "completed"
    assert exported.loc[0, "review_adjudicated_label"] == 1
    reopened = set_review(dataset, completed, row_id, completed.reviews[row_id])
    assert reopened.reviews[row_id].status == "pending"
    assert reopened.reviews[row_id].completed_at is None
    assert SessionStore(inputs / "saved", dataset, "reviewer_a").path == store.path
    assert not SessionStore(inputs / "saved", dataset, "reviewer_b").load().reviews


def test_completion_validation_and_unknown_labels(inputs: Path) -> None:
    dataset = load_dataset(inputs / "classifications.csv")
    store = SessionStore(inputs / "saved", dataset, "r1")
    row_id = dataset.rows[1]["row_id"]
    with pytest.raises(ValueError, match="known, equal"):
        set_review(
            dataset,
            store.load(),
            row_id,
            ReviewDecision(assessment="match"),
            complete=True,
        )
    with pytest.raises(ValueError, match="reviewer notes"):
        set_review(
            dataset,
            store.load(),
            row_id,
            ReviewDecision(assessment="no_match", adjudicated_label=0),
            complete=True,
        )
    with pytest.raises(ValueError, match="final Present"):
        set_review(
            dataset,
            store.load(),
            row_id,
            ReviewDecision(assessment="no_match", notes="Explanation"),
            complete=True,
        )
    session = set_review(
        dataset,
        store.load(),
        row_id,
        ReviewDecision(
            assessment="no_match",
            notes="Observation only",
            adjudicated_label=0,
            needs_follow_up=True,
        ),
        complete=True,
    )
    assert session.reviews[row_id].adjudicated_label == 0
    assert export_rows(dataset, session).loc[1, "review_needs_follow_up"]
    dataset.rows[0]["source"]["human_classification"] = ""
    dataset.rows[0]["source"]["machine_classification"] = ""
    with pytest.raises(ValueError, match="known, equal"):
        set_review(
            dataset,
            store.load(),
            dataset.rows[0]["row_id"],
            ReviewDecision(assessment="match"),
            complete=True,
        )


def test_reject_comparison_requires_notes(inputs: Path) -> None:
    dataset = load_dataset(inputs / "comparison_rows.csv")
    store = SessionStore(inputs / "saved", dataset, "r")
    row_id = dataset.rows[1]["row_id"]
    with pytest.raises(ValueError, match="reviewer notes"):
        set_review(
            dataset,
            store.load(),
            row_id,
            ReviewDecision(assessment="reject"),
            complete=True,
        )
    session = set_review(
        dataset,
        store.load(),
        row_id,
        ReviewDecision(
            assessment="reject", notes="This observation is not perspective taking."
        ),
        complete=True,
    )
    assert session.reviews[row_id].assessment == "reject"
    assert session.reviews[row_id].adjudicated_label is None


def test_changed_inputs_and_codebooks_do_not_reuse_progress(inputs: Path) -> None:
    path, codebook = inputs / "classifications.csv", inputs / "codebook.json"
    dataset = load_dataset(path, codebook)
    store = SessionStore(inputs / "saved", dataset, "r")
    saved = store.save(
        set_review(
            dataset,
            store.load(),
            dataset.rows[0]["row_id"],
            ReviewDecision(notes="Keep me"),
        )
    )
    path.write_text(path.read_text().replace("friend", "colleague"))
    changed = load_dataset(path, codebook)
    assert changed.dataset_hash != dataset.dataset_hash
    assert not SessionStore(inputs / "saved", changed, "r").load().reviews
    codebook.write_text('{"perspective_taking": {"definition": "Different"}}')
    assert load_dataset(path, codebook).dataset_hash != changed.dataset_hash
    assert store.load() == saved


def test_stale_tab_corruption_and_lock_protection(inputs: Path) -> None:
    dataset = load_dataset(inputs / "classifications.csv")
    store = SessionStore(inputs / "saved", dataset, "r")
    stale = store.load()
    store.save(stale)
    original = store.path.read_bytes()
    with pytest.raises(ValueError, match="another tab"):
        store.save(stale)
    assert store.path.read_bytes() == original
    lock = store.path.with_suffix(".lock")
    lock.touch()
    with pytest.raises(ValueError, match="save is in progress"):
        store.save(store.load())
    lock.unlink()
    store.path.write_text("{broken json")
    with pytest.raises(ValueError, match="Invalid session"):
        store.save(stale)
    assert store.path.read_text() == "{broken json"


def test_combine_keeps_reviewers_and_latest_snapshots(inputs: Path) -> None:
    dataset = load_dataset(inputs / "classifications.csv")
    exports = []
    row_id = dataset.rows[0]["row_id"]
    for reviewer in ("alice", "bob"):
        store = SessionStore(inputs / "saved", dataset, reviewer)
        session = store.save(store.load())
        snapshot = inputs / f"{reviewer}-old.csv"
        snapshot.write_bytes(store.export_path.read_bytes())
        exports.append(snapshot)
        store.save(
            set_review(
                dataset, session, row_id, ReviewDecision(notes="=Formula-like text")
            )
        )
        exports.append(store.export_path)
    output = inputs / "combined.csv"
    assert combine_exports(exports, output) == 6
    frame = pd.read_csv(output, keep_default_na=False)
    assert set(frame["reviewer_id"]) == {"alice", "bob"}
    assert len(frame[frame["review_notes"] == "'=Formula-like text"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        combine_exports(exports, output)


def test_cli_missing_input_is_actionable(tmp_path: Path, capsys) -> None:
    pytest.importorskip("streamlit")
    assert main(["run", str(tmp_path / "missing.csv")]) == 2
    assert "Review error" in capsys.readouterr().err


def test_launch_uses_localhost_and_current_interpreter(
    inputs: Path, monkeypatch
) -> None:
    pytest.importorskip("streamlit")
    import sys

    commands = []

    def capture(command):
        commands.append(command)
        return 0

    monkeypatch.setattr("subprocess.call", capture)
    assert main(["run", str(inputs / "classifications.csv"), "--no-browser"]) == 0
    assert commands[0][0] == sys.executable
    assert "--server.address=127.0.0.1" in commands[0]
    assert "--browser.gatherUsageStats=false" in commands[0]


def test_cli_missing_extra_explains_install_command(
    inputs: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert main(["run", str(inputs / "classifications.csv")]) == 2
    assert 'pip install "llm-tracker[review]"' in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [
        '{"test": {"definition": 1}}',
        '{"test": {"examples": "example"}}',
        '{"test": "definition"}',
        "[]",
    ],
)
def test_invalid_codebook_is_reported(inputs: Path, content: str) -> None:
    codebook = inputs / "invalid.json"
    codebook.write_text(content)
    with pytest.raises(ValueError, match="Codebook"):
        load_dataset(inputs / "classifications.csv", codebook)


def test_failed_atomic_save_preserves_existing_session(
    inputs: Path, monkeypatch
) -> None:
    dataset = load_dataset(inputs / "classifications.csv")
    store = SessionStore(inputs / "saved", dataset, "reviewer")
    session = store.save(store.load())
    original = store.path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("Simulated disk error")

    monkeypatch.setattr("os.replace", fail_replace)
    changed = set_review(
        dataset, session, dataset.rows[0]["row_id"], ReviewDecision(notes="Draft")
    )
    with pytest.raises(OSError, match="disk error"):
        store.save(changed)
    assert store.path.read_bytes() == original
    assert not store.path.with_suffix(".lock").exists()


def test_combine_rejects_conflicting_copies(inputs: Path) -> None:
    dataset = load_dataset(inputs / "classifications.csv")
    store = SessionStore(inputs / "saved", dataset, "reviewer")
    store.save(store.load())
    frame = pd.read_csv(store.export_path, keep_default_na=False)
    frame.loc[0, "review_notes"] = "Different note with identical timestamp"
    conflicting = inputs / "conflicting.csv"
    frame.to_csv(conflicting, index=False)
    with pytest.raises(ValueError, match="Conflicting reviews"):
        combine_exports([store.export_path, conflicting], inputs / "combined.csv")


@pytest.mark.parametrize(
    "content",
    [
        "doc_id,construct,status,human_quote,llm_quote\n1,test,invalid,quote,quote\n",
        "doc_id,construct,status,human_quote,llm_quote\n1,test,matched,,quote\n",
        "doc_id,construct,status,human_quote,llm_quote\n1,test,human_only,quote,extra\n",
        "participant_id,text,human_classification,machine_classification\n1,text,2,0\n",
        "participant_id,text,human_classification,machine_classification\n1,,1,0\n",
        "participant_id,text,human_classification,machine_classification\n",
    ],
)
def test_invalid_inputs_are_reported(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_dataset(path, construct="test")
