"""Exercise review UI actions using Streamlit's browser-free test runner."""

from pathlib import Path

import pytest
from llm_tracker.review.cli import main
from llm_tracker.review.store import SessionStore

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).parents[1] / "src" / "llm_tracker" / "review" / "app.py"


@pytest.mark.parametrize(
    "filename,assessment",
    [("classifications.csv", "Match"), ("comparison_rows.csv", "Accept comparison")],
)
def test_review_save_resume_and_export(
    tmp_path: Path, monkeypatch, filename, assessment
) -> None:
    assert main(["demo", str(tmp_path)]) == 0
    monkeypatch.setattr(
        "sys.argv",
        [str(APP), str(tmp_path / filename), "--data-dir", str(tmp_path / "saved")],
    )
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    app.text_input[0].set_value("Reviewer_A")
    app.button[0].click().run()
    assert not app.exception
    app.radio[0].set_value(assessment)
    next(
        button for button in app.button if button.label == "Complete and next pending"
    ).click().run()
    assert not app.exception
    assert not app.error
    session = app.session_state["review_session"]
    assert sum(review.status == "completed" for review in session.reviews.values()) == 1
    store = app.session_state["store"]
    assert isinstance(store, SessionStore)
    assert store.export_path.exists()
    assert store.load() == session
    picker = next(box for box in app.selectbox if box.label == "Review row")
    assert "Row 2" in picker.format_func(picker.value)
    if filename == "classifications.csv":
        app.radio[0].set_value("Do not match")
        app.radio[1].set_value("Not present")
    else:
        app.radio[0].set_value("Reject comparison")
    next(
        button for button in app.button if button.label == "Complete and next pending"
    ).click().run()
    assert app.error
    app.text_area[0].set_value(
        "Observation only; no consideration of causes or consequences."
    )
    next(
        button for button in app.button if button.label == "Complete and next pending"
    ).click().run()
    assert not app.exception
    assert not app.error
    assert (
        sum(review.status == "completed" for review in store.load().reviews.values())
        == 2
    )
    next(
        button for button in app.button if button.label == "Reload saved reviews"
    ).click().run()
    assert not app.exception
    assert app.session_state["review_session"] == store.load()
