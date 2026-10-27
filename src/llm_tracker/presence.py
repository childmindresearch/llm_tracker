"""Compare document-level construct presence without an LLM matcher."""

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from llm_tracker.models import AnalysisResult
from llm_tracker.outputs import RunOutput, source_run, export_quotes


def compare_presence(
    human_results: Mapping[str, AnalysisResult],
    llm_results: Mapping[str, AnalysisResult],
    *,
    constructs: Iterable[str],
    document_ids: Iterable[str] | None = None,
    output_dir: Path | str | None = None,
    master_output_dir: Path | str | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Compare whether each coder detected each construct in each document.

    Makes no API calls. Presence means at least one instance exists, regardless
    of confidence, quote text, or number of instances. Pass the full construct
    roster (usually codebook keys) so constructs neither coder detected are
    included. Unknown detected constructs raise rather than silently disappear.

    By default, compare the union of the input document IDs. ``document_ids``
    can supply the complete corpus roster or explicitly select a subset. Each
    selected document must have an AnalysisResult on BOTH sides. A successful
    result with no instances means absence; a missing result means incomplete
    coding and raises ValueError. Retry failed coding before comparing. Results
    alone cannot identify a failure mistakenly stored as a successful empty
    result; only supply completed, reviewed coding results.

    Returns one row per document and construct, with human_count, llm_count,
    human_present, llm_present, status (tp/fp/fn/tn), and binary tp/fp/fn/tn
    columns. Pass this grid directly to ``compute_binary_metrics``; do not pass
    it to the span-level summary or codebook optimization functions.

    master_output_dir (or output_dir) is the parent for a new timestamped
    comparison_short run. New coding results also carry their master directory
    automatically. The presence grid and both quote views save automatically;
    compute_binary_metrics(grid) saves the binary table in that same run.
    model_name supplies a fallback when plain loaded dictionaries lack model
    provenance. grid.attrs["output_path"] identifies the actual saved directory.
    Inputs are never modified.
    """

    def roster(values: Iterable[str], name: str) -> list[str]:
        if isinstance(values, str):
            raise TypeError(f"{name} must be an iterable of strings, not a string.")
        items = list(values)
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"{name} must contain nonempty strings.")
        return list(dict.fromkeys(items))

    names = roster(constructs, "constructs")
    if not names:
        raise ValueError("constructs must include the full, nonempty construct roster.")
    docs = roster(
        (
            sorted(set(human_results) | set(llm_results))
            if document_ids is None
            else document_ids
        ),
        "document_ids",
    )
    missing_human = [doc for doc in docs if doc not in human_results]
    missing_llm = [doc for doc in docs if doc not in llm_results]
    if missing_human or missing_llm:
        raise ValueError(
            "Incomplete coding results. Missing human document IDs: "
            f"{missing_human}; missing LLM document IDs: {missing_llm}. "
            "Retry failed coding or load the missing results. For wide human "
            "data, reload with include_empty=True. Only confirmed completed "
            "documents with no detections should have an empty AnalysisResult."
        )

    allowed = set(names)
    counts: dict[str, dict[str, Counter]] = {}
    for label, results in (("human", human_results), ("llm", llm_results)):
        counts[label] = {}
        for doc in docs:
            result = results[doc]
            if not isinstance(result, AnalysisResult):
                raise TypeError(
                    f"{label} result for {doc!r} must be an AnalysisResult."
                )
            if result.document_id != doc:
                raise ValueError(
                    f"{label} result document_id does not match key {doc!r}."
                )
            doc_counts = Counter(instance.construct for instance in result.instances)
            unknown = sorted(set(doc_counts) - allowed)
            if unknown:
                raise ValueError(
                    f"Unknown {label} constructs in document {doc!r}: {unknown}. "
                    "Check construct names and the supplied construct roster."
                )
            counts[label][doc] = doc_counts

    rows = []
    for doc in docs:
        for name in names:
            human_count = counts["human"][doc][name]
            llm_count = counts["llm"][doc][name]
            human_present, llm_present = int(human_count > 0), int(llm_count > 0)
            status = {(1, 1): "tp", (0, 1): "fp", (1, 0): "fn", (0, 0): "tn"}[
                human_present, llm_present
            ]
            rows.append(
                {
                    "doc_id": doc,
                    "construct": name,
                    "human_count": human_count,
                    "llm_count": llm_count,
                    "human_present": human_present,
                    "llm_present": llm_present,
                    "status": status,
                    **{
                        outcome: int(status == outcome)
                        for outcome in ("tp", "fp", "fn", "tn")
                    },
                }
            )
    grid = pd.DataFrame(
        rows,
        columns=[
            "doc_id",
            "construct",
            "human_count",
            "llm_count",
            "human_present",
            "llm_present",
            "status",
            "tp",
            "fp",
            "fn",
            "tn",
        ],
    )
    source = source_run(llm_results)
    human = source_run(human_results)
    root = (
        master_output_dir
        or output_dir
        or (source.record["master_output_dir"] if source else None)
    )
    if root is not None:
        run = RunOutput.create(
            root,
            "comparison_short",
            source.record["model"] if source else (model_name or "unknown"),
            source_encoding_run_id=source.record["run_id"] if source else "unavailable",
            source_human_run_id=human.record["run_id"] if human else "unavailable",
            input_documents=len(docs),
            successful_documents=len(docs),
            failed_documents=0,
            construct_count=len(names),
            comparison_rows=len(grid),
            matching_model="N/A",
        )
        run.table("presence_grid", grid)
        export_quotes(run, {doc: llm_results[doc] for doc in docs})
        run.finish()
    return grid
