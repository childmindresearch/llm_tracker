"""Streamlit interface launched by ``llm-tracker-review run``."""

from __future__ import annotations

import argparse
from typing import Any, cast

import streamlit as st

from llm_tracker.review.cli import add_run_arguments
from llm_tracker.review.store import (
    Assessment,
    BinaryLabel,
    ReviewDataset,
    ReviewDecision,
    ReviewSession,
    SessionStore,
    csv_text,
    export_rows,
    load_dataset,
    normalize_label,
    set_review,
)


def _label(value: Any) -> str:
    label = normalize_label(value)
    return {1: "Present", 0: "Not present", None: "Unknown"}[label]


def _show_source(dataset: ReviewDataset, row: dict[str, Any]) -> None:
    source = row["source"]
    definition = dataset.definitions.get(row["construct"], {})
    with st.expander("Construct definition and source details", expanded=True):
        st.text(
            definition.get("definition")
            or definition.get("prompt_text")
            or "No definition supplied."
        )
        for example in definition.get("examples", []):
            st.text(f"Example: {example}")
        if source.get("prompt_text"):
            st.text("Prompt recorded in the input:")
            st.text(source["prompt_text"])
            if definition.get("prompt_text") and (
                source["prompt_text"].strip() != definition["prompt_text"].strip()
            ):
                st.warning("The workbook prompt differs from the supplied definition.")
        st.caption(f"Source: {dataset.source_name} · row {row['row_number']}")
    if dataset.kind == "classification":
        st.subheader("Source text")
        st.text(source["text"])
    else:
        st.info(f"Original comparison outcome: {source['status']}")
        if source.get("text"):
            st.text(source["text"])
    human, machine = st.columns(2)
    with human:
        st.subheader("Human coding")
        if dataset.kind == "classification":
            st.write(_label(source["human_classification"]))
            st.text(source.get("human_reasoning", ""))
        else:
            st.text(source["human_quote"] or "No human quote")
            st.caption(source.get("human_indices", ""))
    with machine:
        st.subheader("LLM coding")
        if dataset.kind == "classification":
            st.write(_label(source["machine_classification"]))
            st.text(source.get("machine_reasoning", ""))
        else:
            st.text(source["llm_quote"] or "No LLM quote")
            st.caption(source.get("llm_indices", ""))


def _review_form(
    dataset: ReviewDataset,
    store: SessionStore,
    session: ReviewSession,
    row: dict[str, Any],
) -> None:
    row_id = row["row_id"]
    review = session.reviews.get(row_id, ReviewDecision())
    choices: dict[str, Assessment | None] = (
        {"Not selected": None, "Match": "match", "Do not match": "no_match"}
        if dataset.kind == "classification"
        else {
            "Not selected": None,
            "Accept comparison": "accept",
            "Reject comparison": "reject",
        }
    )
    st.subheader("Your review")
    saved_at = review.last_saved_at or "not yet saved"
    st.caption(f"Status: {review.status} · Last saved: {saved_at}")
    # A revision-specific form resets widgets after saving or changing rows.
    with st.form(f"decision-{session.reviewer_id}-{row_id}-{session.revision}"):
        assessment = st.radio(
            "Do the human and LLM codings match?"
            if dataset.kind == "classification"
            else "Is the original comparison outcome correct?",
            list(choices),
            index=list(choices.values()).index(review.assessment),
        )
        final_label: BinaryLabel | None = None
        if dataset.kind == "classification":
            labels: dict[str, BinaryLabel | None] = {
                "Not selected": None,
                "Present": 1,
                "Not present": 0,
            }
            selected = st.radio(
                "Final label (required when codings do not match)",
                list(labels),
                index=list(labels.values()).index(review.adjudicated_label),
            )
            final_label = labels[selected]
            st.caption(
                "For a match, the final label is taken from the agreeing source labels."
            )
        notes = st.text_area("Reviewer notes", review.notes)
        follow_up = st.checkbox("Needs follow-up", review.needs_follow_up)
        draft = st.form_submit_button("Save draft / reopen")
        complete = st.form_submit_button("Complete and next pending", type="primary")
        st.caption("Changes are saved only when you press a save button.")
    if draft or complete:
        try:
            decision = ReviewDecision(
                assessment=choices[assessment],
                adjudicated_label=final_label,
                notes=notes,
                needs_follow_up=follow_up,
            )
            updated = set_review(dataset, session, row_id, decision, complete=complete)
            st.session_state.review_session = store.save(updated)
            if complete:
                visible = cast(list[dict[str, Any]], st.session_state.visible_rows)
                current = next(
                    i for i, item in enumerate(visible) if item["row_id"] == row_id
                )
                ordered = visible[current + 1 :] + visible[: current + 1]
                next_row = next(
                    (
                        item
                        for item in ordered
                        if (
                            updated.reviews.get(item["row_id"], ReviewDecision()).status
                            != "completed"
                        )
                    ),
                    row,
                )
                st.session_state.next_row_id = next_row["row_id"]
            st.session_state.save_notice = "Review saved."
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))


def main() -> None:
    """Render one local review session with explicit saving and CSV/JSON exports."""
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    args = parser.parse_args()
    st.set_page_config(page_title="LLM Tracker Review", layout="wide")
    st.title("LLM Tracker Review")
    st.caption(
        "Manual review on your computer. No LLM calls. "
        "Save before changing rows or leaving."
    )
    try:
        # Read once per browser session, avoiding repeated reads during review.
        if "dataset" not in st.session_state:
            st.session_state.dataset = load_dataset(
                args.input, args.codebook, args.construct
            )
        dataset = cast(ReviewDataset, st.session_state.dataset)
        with st.sidebar:
            st.header("Review session")
            with st.form("reviewer"):
                reviewer = st.text_input("Reviewer ID", placeholder="reviewer_01")
                connect = st.form_submit_button("Start / resume")
            if connect:
                store = SessionStore(args.data_dir, dataset, reviewer)
                resumed = store.load()
                st.session_state.store = store
                st.session_state.review_session = resumed
                st.session_state.pop("row_picker", None)
            if "review_session" not in st.session_state:
                st.info("Enter a reviewer ID to start or resume your saved reviews.")
                st.stop()
            store = cast(SessionStore, st.session_state.store)
            session = cast(ReviewSession, st.session_state.review_session)
            st.write(f"Reviewer: {session.reviewer_id}")
            if st.button("Reload saved reviews"):
                st.session_state.review_session = store.load()
                st.rerun()
            names = list(dict.fromkeys(row["construct"] for row in dataset.rows))
            selected_construct = st.selectbox("Construct", names)
            pending_only = st.checkbox("Pending only")
            flagged_only = st.checkbox("Needs follow-up only")
            done = sum(
                review.status == "completed" for review in session.reviews.values()
            )
            st.metric("Completed / total", f"{done} / {len(dataset.rows)}")
            st.progress(done / len(dataset.rows))
            st.caption(f"Saved locally in {args.data_dir.resolve()}")
            st.download_button(
                "Download review CSV",
                csv_text(export_rows(dataset, session)),
                file_name=f"reviews-{session.reviewer_id}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download session JSON",
                session.model_dump_json(indent=2),
                file_name=store.path.name,
                mime="application/json",
            )
        if notice := st.session_state.pop("save_notice", None):
            st.success(notice)
        rows = [
            row
            for row in dataset.rows
            if (
                row["construct"] == selected_construct
                and (
                    not pending_only
                    or session.reviews.get(row["row_id"], ReviewDecision()).status
                    == "pending"
                )
                and (
                    not flagged_only
                    or session.reviews.get(
                        row["row_id"], ReviewDecision()
                    ).needs_follow_up
                )
            )
        ]
        st.session_state.visible_rows = rows
        if not rows:
            st.info(
                "No rows match these filters. "
                "Change the filters to view completed reviews."
            )
            st.stop()
        ids = [row["row_id"] for row in rows]
        if st.session_state.get("next_row_id") in ids:
            st.session_state.row_picker = st.session_state.pop("next_row_id")
        else:
            st.session_state.pop("next_row_id", None)
        if st.session_state.get("row_picker") not in ids:
            st.session_state.row_picker = next(
                (
                    row["row_id"]
                    for row in rows
                    if session.reviews.get(row["row_id"], ReviewDecision()).status
                    == "pending"
                ),
                ids[0],
            )
        lookup = {row["row_id"]: row for row in rows}

        def describe(row_id: str) -> str:
            row = lookup[row_id]
            source = row["source"]
            status = session.reviews.get(row_id, ReviewDecision()).status
            document = source.get("participant_id", source.get("doc_id", ""))
            return f"Row {row['row_number']} · {document} · {status}"

        selected_row = st.selectbox(
            "Review row", ids, format_func=describe, key="row_picker"
        )
        row = lookup[selected_row]
        _show_source(dataset, row)
        _review_form(dataset, store, session, row)
    except (OSError, ValueError) as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
