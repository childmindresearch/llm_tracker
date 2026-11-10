"""Compare document-level construct presence without an LLM matcher."""

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from llm_tracker.models import AnalysisResult
from llm_tracker.confidence import prepare_llm_results, validate_rounding
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
    rounding_method: str = "floor",
    binarize_confidence: bool = True,
) -> pd.DataFrame:
    """Compare whether each coder detected each construct in each document.

    Makes no API calls. LLM presence means at least one positive-scored instance
    after conversion. rounding_method="floor" (default) maps 0/1 to 0 and 2 to 1;
    "ceiling" maps 0 to 0 and 1/2 to 1. binarize_confidence=False skips conversion
    for already-binary data (and retains other positive scores unchanged).
    Explicit zero instances mean absence, not detection; no document/construct
    rows are removed. Human detections/scores are unchanged. Quote text and
    repeated detections do not affect the binary decision. Pass the full construct
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
    rank_constructs_by_percentage(grid) ranks LLM document prevalence using the
    same converted flags and saves beside the comparison when output is enabled.

    master_output_dir (or output_dir) is the parent for a new timestamped
    comparison_short run. New coding results also carry their master directory
    automatically. The presence grid and both quote views save automatically;
    compute_binary_metrics(grid) saves the binary table in that same run.
    model_name supplies a fallback when plain loaded dictionaries lack model
    provenance. grid.attrs["output_path"] identifies the actual saved directory.
    llm_count counts positive LLM instances under the chosen policy. Comparison
    quote exports retain original confidence and add comparison_confidence;
    explicit zero records remain in these short-form exports. Inputs are never modified.
    """

    def roster(values: Iterable[str], name: str) -> list[str]:
        if isinstance(values, str):
            raise TypeError(f"{name} must be an iterable of strings, not a string.")
        items = list(values)
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"{name} must contain nonempty strings.")
        return list(dict.fromkeys(items))

    validate_rounding(rounding_method, binarize_confidence)
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

    selected_llm = {doc: llm_results[doc] for doc in docs}
    prepared_llm = prepare_llm_results(
        selected_llm,
        rounding_method=rounding_method,
        binarize_confidence=binarize_confidence,
        drop_zeros=False,
    )
    allowed = set(names)
    counts: dict[str, dict[str, Counter]] = {}
    for label, results in (("human", human_results), ("llm", prepared_llm)):
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
            if label == "llm":
                doc_counts = Counter(
                    instance.construct
                    for instance in result.instances
                    if instance.confidence > 0
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
    grid.attrs.update(
        rounding_method=rounding_method, binarize_confidence=binarize_confidence
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
            rounding_method=rounding_method,
            binarize_confidence=binarize_confidence,
        )
        run.table("presence_grid", grid)
        export_quotes(
            run,
            selected_llm,
            rounding_method=rounding_method,
            binarize_confidence=binarize_confidence,
            drop_zeros=False,
        )
        run.finish()
    return grid


def rank_constructs_by_percentage(
    presence_table: pd.DataFrame,
    *,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Rank constructs by the percentage of compared documents coded by the LLM.

    Args:
        presence_table: The COMPLETE document-by-construct grid returned by
            compare_presence(), not a quote table or compute_binary_metrics()
            summary. Include absence/true-negative rows and all selected docs.
            The required columns are doc_id, construct and binary llm_present.
            Duplicate pairs, missing cells and nonbinary flags raise ValueError.
            A valid reloaded CSV grid is accepted; see output_dir for saving.
        output_dir: Optional existing comparison run directory or a parent for
            a new comparison_short output run. By default uses the grid's
            attrs['output_path'], so no timestamped path needs to be typed when
            working with compare_presence() results. For a reloaded CSV supply
            its comparison run directory, since CSV does not retain attrs.
            If neither path is available, returns the table without saving.

    Returns:
        Three-column DataFrame: Rank, Construct, % (N). For example,
        '26.2% (28/107)' means the LLM detected the construct in 28 of the 107
        compared documents. The denominator includes every compared document,
        including those where neither coder found anything. Multiple quotes or
        detections of the same construct in ONE document still count only once.
        This is LLM prevalence, not human prevalence, quote share or agreement.

        The existing llm_present flags are used exactly as given: the chosen
        floor/ceiling policy has ALREADY been applied by compare_presence().
        This function never remaps scores or calls the LLM. Human flags do not
        affect ranking. Constructs with zero LLM detections are included.

        Sorts by unrounded percentage descending; tied counts are ordered by
        construct name (case-insensitive, then original spelling). Rank is a
        sequential row number, so tied percentages have consecutive ranks.
        There is no Overall row. The input is never modified. Counts and the
        document denominator are also available in the returned dataframe attrs.

        When saving, writes llm_construct_presence_ranked_<model>_<timestamp>.csv
        and registers its path/denominator in run metadata. Existing comparison
        runs reuse their model/timestamp and metadata row. A plain output parent
        creates a new tracked comparison_short run. Empty input returns/saves
        a header-only table with a zero-document denominator.
    """
    required = {"doc_id", "construct", "llm_present"}
    missing = required - set(presence_table)
    if missing:
        raise ValueError(f"Presence table is missing columns: {sorted(missing)}")
    grid = presence_table[["doc_id", "construct", "llm_present"]].copy()
    if grid[["doc_id", "construct"]].isna().any().any():
        raise ValueError("Document IDs and construct names cannot be missing.")
    if grid["doc_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Document IDs cannot be empty.")
    if any(not isinstance(name, str) or not name.strip() for name in grid["construct"]):
        raise ValueError("Construct names must be nonempty strings.")
    if grid.duplicated(["doc_id", "construct"]).any():
        raise ValueError(
            "Expected one row per document/construct pair, not repeated quotes."
        )
    flags = pd.to_numeric(grid["llm_present"], errors="coerce")
    if flags.isna().any() or not flags.isin([0, 1]).all():
        raise ValueError("llm_present must contain complete binary 0/1 flags.")
    grid["llm_present"] = flags.astype(int)
    total_documents = int(grid["doc_id"].nunique())
    if len(grid) != total_documents * grid["construct"].nunique():
        raise ValueError(
            "The presence grid is incomplete. Keep absence rows for every document/construct pair."
        )

    counts = grid.groupby("construct", sort=False)["llm_present"].sum()
    ranked = pd.DataFrame({"Construct": counts.index, "_count": counts.values})
    ranked["_name"] = ranked["Construct"].str.casefold()
    ranked = ranked.sort_values(
        ["_count", "_name", "Construct"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    document_counts = dict(zip(ranked["Construct"], map(int, ranked["_count"])))
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    ranked["% (N)"] = [
        f"{100 * count / total_documents:.1f}% ({int(count)}/{total_documents})"
        for count in ranked["_count"]
    ]
    ranked = ranked[["Rank", "Construct", "% (N)"]]
    ranked.attrs.update(
        total_documents=total_documents,
        document_counts=document_counts,
        percentage_definition="LLM-positive documents / all compared documents",
        rounding_method=presence_table.attrs.get("rounding_method", "unavailable"),
        binarize_confidence=presence_table.attrs.get(
            "binarize_confidence", "unavailable"
        ),
    )
    destination = (
        output_dir
        if output_dir is not None
        else presence_table.attrs.get("output_path")
    )
    if destination is not None:
        run = RunOutput.load(destination)
        created = run is None
        if created:
            source = RunOutput.load(presence_table.attrs.get("output_path"))
            run = RunOutput.create(
                destination,
                "comparison_short",
                source.record["model"] if source else "unknown",
                mode="llm_presence_ranking",
            )
        path = run.table("llm_construct_presence_ranked", ranked)
        run.update(
            ranking_total_documents=total_documents,
            ranking_construct_count=len(ranked),
            ranking_measure="LLM-positive documents / all compared documents",
            ranking_rounding_method=ranked.attrs["rounding_method"],
            ranking_binarize_confidence=ranked.attrs["binarize_confidence"],
        )
        if created:
            run.finish()
        ranked.attrs["csv_path"] = str(path)
    return ranked
