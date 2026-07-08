"""Compare human and LLM construct codings."""

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from llm_tracker.config import AnalyzerConfig
from llm_tracker.file_handlers import codebook_constructs, ensure_codebook_envelope
from llm_tracker.models import AnalysisResult, ConstructInstance
from llm_tracker.prompting import PromptingError, call_llm_api

if TYPE_CHECKING:
    from llm_tracker.analyzer import LLMTrackerAnalyzer

COMPARISON_COLUMNS = [
    "doc_id",
    "construct",
    "status",
    "human_quote",
    "llm_quote",
    "human_indices",
    "llm_indices",
    "human_confidence",
    "llm_confidence",
    "paraphrase",
    "span_overlap",
    "match_confidence",
    "tp",
    "fp",
    "fn",
]

METRICS = ["sensitivity", "precision", "f1", "pabak"]
# PR AUC is added separately because it ranks LLM predictions by coding confidence.

MATCH_PROMPT_TEMPLATE = """You are reconciling two sets of quotes for the SAME \
psychological construct, coded independently by a human and an LLM.

Construct: {construct}

Human-coded quotes (0-indexed):
{human_quotes}

LLM-coded quotes (0-indexed):
{llm_quotes}

Task:
Match each LLM quote to a human quote if they refer to the same passage or idea.
Each quote can only be matched once.
A match is valid even if the wording differs (paraphrase), as long as both quotes
refer to the same content.

Return ONLY valid JSON in this exact format:
{{
  "matches": [
    {{
      "human_index": <int>,
      "llm_index": <int>,
      "paraphrase": <true if wording differs meaningfully, false if nearly identical>,
      "match_confidence": <float between 0.0 and 1.0>
    }}
  ]
}}

If there are no matches, return {{"matches": []}}
"""


class ComparisonError(Exception):
    """Exception raised when comparison fails."""


def _load_result_json(path: Path | str) -> AnalysisResult:
    """Load one saved encoding JSON file as an AnalysisResult.

    Args:
    ----
        path: Path to an analyzer result JSON file.

    Returns:
    -------
        The parsed AnalysisResult.

    Raises:
    ------
        ComparisonError: If the file does not exist or is not valid JSON.

    """
    file_path = Path(path)
    if not file_path.exists():
        raise ComparisonError(f"Result JSON not found: {file_path}")
    try:
        return AnalysisResult(**json.loads(file_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        raise ComparisonError(f"Invalid JSON in {file_path}: {e}") from e


def _result_json_dir(path: Path | str) -> Path:
    """Resolve the folder containing saved encoding JSON files.

    Args:
    ----
        path: Either an analyzer run directory or its `encodings` subdir.

    Returns:
    -------
        Directory containing per-document result JSON files.

    Raises:
    ------
        ComparisonError: If the path is not a directory or contains no result
            JSON files.

    """
    directory = Path(path)
    if not directory.is_dir():
        raise ComparisonError(f"Result directory not found: {directory}")
    if any(directory.glob("*.json")):
        return directory
    encodings = directory / "encodings"
    if encodings.is_dir():
        return encodings
    raise ComparisonError(f"No JSON result files found in {directory}")


def _result_files(path: Path | str) -> dict[str, Path]:
    """Map document IDs to saved encoding JSON files.

    Args:
    ----
        path: Either an analyzer run directory or its `encodings` subdirectory.

    Returns:
    -------
        A dictionary mapping each document ID to its result JSON path.

    Raises:
    ------
        ComparisonError: If the path cannot be resolved to a directory
            containing result JSON files.

    """
    return {p.stem: p for p in _result_json_dir(path).glob("*.json")}


def _save_comparison_table(df: pd.DataFrame, output_dir: str) -> Path:
    """Save the row-level comparison table to a timestamped CSV folder.

    Args:
    ----
        df: Comparison DataFrame returned by compare_results().
        output_dir: Prefix for the timestamped output directory.

    Returns:
    -------
        Path to the created output directory.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path.cwd() / f"{output_dir}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison_rows.csv", index=False)
    return out_dir


def _group_by_construct(
    instances: list[ConstructInstance],
) -> dict[str, list[ConstructInstance]]:
    """Group one document's construct instances by construct name.

    The comparison step matches human and LLM quotes within the same construct,
    so this converts a flat list of instances into construct specific buckets.

    Args:
    ----
        instances: Construct instances from one document.

    Returns:
    -------
        A dictionary mapping each construct name to the instances coded for
        that construct.

    """
    grouped: dict[str, list[ConstructInstance]] = {}
    for item in instances:
        if item.construct not in grouped:
            grouped[item.construct] = []
        grouped[item.construct].append(item)
    return grouped


def _format_quotes(quotes: list[ConstructInstance]) -> str:
    """Format construct instances as a numbered quote list for the matcher.

    The matcher prompt refers to quotes by  index, so each quote is
    numbered and includes its character span when available.

    Args:
    ----
        quotes: Construct instances for one construct in one document.

    Returns:
    -------
        A numbered list of quotes and their source indices seperated by new lines.

    """
    lines = []
    for index, item in enumerate(quotes):
        quote_index = item.quote_index or "N/A"
        lines.append(f'{index}. "{item.quote}" (indices: {quote_index})')

    return "\n".join(lines)


def _parse_match_response(response_text: str) -> list[dict]:
    """Parse and normalize the matcher LLM's JSON response.

    Args:
    ----
        response_text: Raw text returned by the matcher LLM.

    Returns:
    -------
        A list of valid match dictionaries with integer quote indices,
        boolean paraphrase flags, and match confidence 0.0-1.0.

    Raises:
    ------
        ComparisonError: If the response is not valid JSON or does not contain
            a `matches` list.

    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ComparisonError(f"Invalid matcher JSON: {e}") from e

    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ComparisonError("Matcher response must contain a 'matches' list.")

    parsed_matches = []
    for match in matches:
        if not isinstance(match, dict):
            continue

        try:
            human_index = int(match["human_index"])
            llm_index = int(match["llm_index"])
            paraphrase = bool(match.get("paraphrase", False))
            confidence = float(match.get("match_confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            parsed_matches.append(
                {
                    "human_index": human_index,
                    "llm_index": llm_index,
                    "paraphrase": paraphrase,
                    "match_confidence": confidence,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return parsed_matches


def _parse_span(span: str | None) -> tuple[int, int] | None:
    """Parse a start:end character span.

    Args:
    ----
        span: Character span string e.g. "10:25".

    Returns:
    -------
        A (start, end) tuple, or None if the span is missing, malformed,
        or has an end index before its start index.

    """
    if not span:
        return None

    parts = span.split(":")
    if len(parts) != 2:
        return None

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        return None

    if end < start:
        return None

    return start, end


def _compute_span_overlap(human_idx: str | None, llm_idx: str | None) -> float | None:
    """Compute overlap between human and LLM character spans.

    Args:
    ----
        human_idx: Human coded quote span in start:end format.
        llm_idx: LLM coded quote span in start:end format.

    Returns:
    -------
        Jaccard overlap between the two spans, or None if either span is
        missing or malformed.

    """
    human_span = _parse_span(human_idx)
    llm_span = _parse_span(llm_idx)
    if human_span is None or llm_span is None:
        return None

    h_start, h_end = human_span
    l_start, l_end = llm_span
    human_length = h_end - h_start
    llm_length = l_end - l_start

    if human_length == 0 and llm_length == 0:
        return 1.0
    if human_length == 0 or llm_length == 0:
        return 0.0

    overlap_start = max(h_start, l_start)
    overlap_end = min(h_end, l_end)
    overlap_length = max(0, overlap_end - overlap_start)
    union_length = human_length + llm_length - overlap_length

    if union_length == 0:
        return 0.0

    return overlap_length / union_length


def _base_row(doc_id: str, construct: str, status: str) -> dict:
    """Create a default row for the comparison table.

    Args:
    ----
        doc_id: Document identifier shared by the human and LLM results.
        construct: Construct being compared.
        status: Match status for the row: matched, human_only, or
            llm_only.

    Returns:
    -------
        A comparison row with identifying fields filled in, optional fields set
        to None, and metric count fields initialized to zero.

    """
    return {
        "doc_id": doc_id,
        "construct": construct,
        "status": status,
        "human_quote": None,
        "llm_quote": None,
        "human_indices": None,
        "llm_indices": None,
        "human_confidence": None,
        "llm_confidence": None,
        "paraphrase": None,
        "span_overlap": None,
        "match_confidence": None,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


class LLMTrackerComparer:
    """Compare human-coded and LLM-coded results using an LLM matcher."""

    def __init__(
        self,
        api_key: str | None = None,
        match_model: str | None = None,
        config: AnalyzerConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif match_model is not None:
            self.config = AnalyzerConfig(api_key=api_key, model_name=match_model)
        else:
            self.config = AnalyzerConfig(api_key=api_key)

    def _match_construct(
        self,
        construct: str,
        human: list[ConstructInstance],
        llm: list[ConstructInstance],
    ) -> list[dict]:
        """Ask the matcher LLM to align quotes for one construct.

        Args:
        ----
            construct: Construct name shared by the quote lists.
            human: Human coded instances for this construct.
            llm: LLM coded instances for this construct.

        Returns:
        -------
            Parsed matcher results describing which human and LLM quote indices
            refer to the same passage or idea.

        Raises:
        ------
            ComparisonError: If the matcher fails after all retry attempts.

        """
        prompt = MATCH_PROMPT_TEMPLATE.format(
            construct=construct,
            human_quotes=_format_quotes(human),
            llm_quotes=_format_quotes(llm),
        )
        last_error = None
        total_attempts = self.config.max_retries + 1
        for _ in range(total_attempts):
            try:
                response_text, _metadata = call_llm_api(prompt, self.config)
                return _parse_match_response(response_text)
            except (PromptingError, ComparisonError) as e:
                last_error = e

        raise ComparisonError(f"Matcher failed for '{construct}': {last_error}")

    def _compare_construct(
        self,
        doc_id: str,
        construct: str,
        human: list[ConstructInstance],
        llm: list[ConstructInstance],
    ) -> list[dict]:
        """Compare human and LLM instances for one construct in one document.

        Args:
        ----
            doc_id: Document identifier for the compared instances.
            construct: Construct being compared.
            human: Human coded instances for this construct.
            llm: LLM coded instances for this construct.

        Returns:
        -------
            Row dictionaries for matched, human only, and LLM only instances.
            These rows feed the comparison DataFrame and metric counts.

        """

        def human_only_row(item: ConstructInstance) -> dict:
            row = _base_row(doc_id, construct, "human_only")
            row.update(
                human_quote=item.quote,
                human_indices=item.quote_index,
                human_confidence=item.confidence,
                fn=1,
            )
            return row

        def llm_only_row(item: ConstructInstance) -> dict:
            row = _base_row(doc_id, construct, "llm_only")
            row.update(
                llm_quote=item.quote,
                llm_indices=item.quote_index,
                llm_confidence=item.confidence,
                fp=1,
            )
            return row

        rows = []
        if not human:
            return [llm_only_row(item) for item in llm]

        if not llm:
            return [human_only_row(item) for item in human]

        used_human: set[int] = set()
        used_llm: set[int] = set()
        for match in self._match_construct(construct, human, llm):
            human_index = match["human_index"]
            llm_index = match["llm_index"]
            if (
                human_index in used_human
                or llm_index in used_llm
                or not 0 <= human_index < len(human)
                or not 0 <= llm_index < len(llm)
            ):
                continue

            human_item = human[human_index]
            llm_item = llm[llm_index]
            row = _base_row(doc_id, construct, "matched")
            row.update(
                human_quote=human_item.quote,
                llm_quote=llm_item.quote,
                human_indices=human_item.quote_index,
                llm_indices=llm_item.quote_index,
                human_confidence=human_item.confidence,
                llm_confidence=llm_item.confidence,
                paraphrase=match["paraphrase"],
                span_overlap=_compute_span_overlap(
                    human_item.quote_index, llm_item.quote_index
                ),
                match_confidence=match["match_confidence"],
                tp=1,
            )
            rows.append(row)
            used_human.add(human_index)
            used_llm.add(llm_index)

        for human_index, item in enumerate(human):
            if human_index not in used_human:
                rows.append(human_only_row(item))

        for llm_index, item in enumerate(llm):
            if llm_index not in used_llm:
                rows.append(llm_only_row(item))

        return rows

    def compare_results(
        self,
        human_results: dict[str, AnalysisResult],
        llm_results: dict[str, AnalysisResult],
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Compare human and LLM results across all documents.

        Args:
        ----
            human_results: Human coded results keyed by document ID.
            llm_results: LLM coded results keyed by document ID.
            output_dir: Optional base name for saving the row level comparison
                table to a timestamped CSV folder.

        Returns:
        -------
            Comparison DataFrame with one row per matched,
            human only, or LLM only construct instance.

        """
        rows = []

        document_ids = sorted(set(human_results) | set(llm_results))
        for doc_id in document_ids:
            human_result = human_results.get(doc_id)
            llm_result = llm_results.get(doc_id)
            human_instances = human_result.instances if human_result else []
            llm_instances = llm_result.instances if llm_result else []

            human_by_construct = _group_by_construct(human_instances)
            llm_by_construct = _group_by_construct(llm_instances)
            construct_names = sorted(set(human_by_construct) | set(llm_by_construct))

            for construct in construct_names:
                rows.extend(
                    self._compare_construct(
                        doc_id,
                        construct,
                        human_by_construct.get(construct, []),
                        llm_by_construct.get(construct, []),
                    )
                )

        df = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
        if not df.empty:
            df[["tp", "fp", "fn"]] = df[["tp", "fp", "fn"]].astype(int)
        if output_dir:
            _save_comparison_table(df, output_dir)
        return df

    def compare_documents(
        self,
        human_json: Path | str,
        llm_json: Path | str,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Compare one reference result JSON with one LLM result JSON.

        Args:
        ----
            human_json: Path to the reference result JSON file.
            llm_json: Path to the LLM result JSON file.
            output_dir: Optional base name for saving the comparison table.

        Returns:
        -------
            Comparison DataFrame with one row per matched, human only, or LLM
            only construct instance.

        Raises:
        ------
            ComparisonError: If either JSON file cannot be loaded or matching
            fails.

        """
        human_result = _load_result_json(human_json)
        llm_result = _load_result_json(llm_json)
        doc_id = human_result.document_id
        return self.compare_results(
            {doc_id: human_result},
            {doc_id: llm_result},
            output_dir=output_dir,
        )

    def compare_directories(
        self,
        human_dir: Path | str,
        llm_dir: Path | str,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Compare reference and LLM result directories.

        Args:
        ----
            human_dir: Path to a reference result directory or encodings folder.
            llm_dir: Path to an LLM result directory or encodings folder.
            output_dir: Optional base name for saving the comparison table.

        Returns:
        -------
            Comparison DataFrame with one row per matched, human only, or LLM
            only construct instance.

        Raises:
        ------
            ComparisonError: If either directory cannot be loaded or matching
            fails.

        """
        human_files = _result_files(human_dir)
        llm_files = _result_files(llm_dir)
        human_results: dict[str, AnalysisResult] = {}
        llm_results: dict[str, AnalysisResult] = {}

        for doc_id in sorted(set(human_files) | set(llm_files)):
            if doc_id in human_files:
                human_results[doc_id] = _load_result_json(human_files[doc_id])
            if doc_id in llm_files:
                llm_results[doc_id] = _load_result_json(llm_files[doc_id])

        return self.compare_results(
            human_results,
            llm_results,
            output_dir=output_dir,
        )


def _metrics(tp: int, fp: int, fn: int) -> dict:
    """Compute agreement metrics from tp, fp, and fn counts.

    Args:
        tp: Number of matched human and LLM instances.
        fp: Number of LLM only instances.
        fn: Number of human only instances.

    Returns:
        Dictionary containing the input counts, union count, sensitivity,
        precision, F1, and PABAK. Metrics with zero denominators are None.
    """
    union = tp + fp + fn
    observed_agreement = tp / union if union else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "union": union,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "pabak": None if observed_agreement is None else 2 * observed_agreement - 1,
    }


def _metrics_for_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Compute agreement metrics for each row of count totals.

    Args:
    ----
        counts: DataFrame with tp, fp, and fn columns.

    Returns:
    -------
        DataFrame containing counts and agreement metrics for each input row.

    """
    if counts.empty:
        return pd.DataFrame(columns=["tp", "fp", "fn", "union", *METRICS])
    return pd.DataFrame(
        [_metrics(row.tp, row.fp, row.fn) for row in counts.itertuples()]
    )


def compute_pr_auc(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute average precision from LLM coding confidence.

    Matched rows are correct LLM predictions. LLM only rows are incorrect LLM
    predictions. Human only rows are excluded because they are missed items, not
    LLM predictions. The ranking score is llm_confidence from the coding step,
    not matcher confidence.

    Args:
    ----
        df: Row level comparison DataFrame.

    Returns:
    -------
        Average precision scores for the full comparison table and for each
        construct. A value is None unless the group has at least one matched
        prediction and at least one LLM only prediction.

    """
    if df.empty:
        return {"Overall": None}

    predictions = df[df["status"].isin(["matched", "llm_only"])].copy()
    if predictions.empty:
        return {"Overall": None}

    predictions["label"] = (predictions["status"] == "matched").astype(int)
    predictions["score"] = pd.to_numeric(predictions["llm_confidence"], errors="coerce")
    predictions = predictions.dropna(subset=["score"])

    def score_group(group: pd.DataFrame) -> float | None:
        if group["label"].nunique() < 2:
            return None
        score = average_precision_score(group["label"], group["score"])
        return round(float(score), 4)

    results = {"Overall": score_group(predictions)}
    for construct, group in predictions.groupby("construct"):
        results[construct] = score_group(group)

    return results


def _doc_stats(
    construct: str, per_doc: pd.DataFrame, total_docs: int, constructs: list[str]
) -> dict:
    """Summarize how often a construct appears across documents.

    Args:
    ----
        construct: Construct name to summarize, or Overall for all constructs.
        per_doc: Per document metric table with union counts.
        total_docs: Total number of documents in the comparison.
        constructs: All construct names present in the comparison.

    Returns:
    -------
        Dictionary with the number of documents containing the construct and
        the fifth and ninety fifth percentiles of per document union counts.

    """
    if construct == "Overall":
        union_counts = per_doc["union"].tolist()
        possible_counts = total_docs * len(constructs)
        n_docs = per_doc["doc_id"].nunique()
    else:
        construct_rows = per_doc[per_doc["construct"] == construct]
        union_counts = construct_rows["union"].tolist()
        possible_counts = total_docs
        n_docs = len(construct_rows)

    missing_count = max(0, possible_counts - len(union_counts))
    union_counts = union_counts + [0] * missing_count
    union_counts = union_counts or [0]
    return {
        "n_docs": n_docs,
        "p5": round(float(np.percentile(union_counts, 5)), 2),
        "p95": round(float(np.percentile(union_counts, 95)), 2),
    }


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Compute the median value after applying row weights.

    Args:
    ----
        values: Metric values to summarize.
        weights: Weights for each value, usually the union count for that row.

    Returns:
    -------
        Weighted median of the input values.

    """
    value_weight_pairs = sorted(zip(values, weights), key=lambda item: item[0])
    halfway_weight = sum(weights) / 2
    cumulative_weight = 0.0

    for value, weight in value_weight_pairs:
        cumulative_weight += weight
        if cumulative_weight >= halfway_weight:
            return value

    return value_weight_pairs[-1][0]


def _weighted_summary(per_doc: pd.DataFrame) -> pd.DataFrame:
    """Build weighted metric summaries across documents.

    Args:
    ----
        per_doc: Per document metric table from compute_summary_tables.

    Returns:
    -------
        DataFrame with one row per construct plus Overall. Each metric is
        summarized with a weighted median, minimum, and maximum.

    """
    rows = []
    construct_names = list(per_doc["construct"].unique()) if not per_doc.empty else []

    for construct in [*construct_names, "Overall"]:
        construct_rows = (
            per_doc
            if construct == "Overall"
            else per_doc[per_doc["construct"] == construct]
        )
        row = {
            "construct": construct,
            "tp": int(construct_rows["tp"].sum()),
            "fp": int(construct_rows["fp"].sum()),
            "fn": int(construct_rows["fn"].sum()),
        }

        for metric in [*METRICS, "pr_auc"]:
            valid_rows = construct_rows[["union", metric]].dropna()
            valid_rows = valid_rows[valid_rows["union"] > 0]

            if valid_rows.empty:
                row[f"{metric}_median"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
            else:
                values = valid_rows[metric].tolist()
                weights = valid_rows["union"].tolist()
                row[f"{metric}_median"] = round(_weighted_median(values, weights), 4)
                row[f"{metric}_min"] = round(min(values), 4)
                row[f"{metric}_max"] = round(max(values), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute summary tables from row level comparison results.

    Args:
    ----
        df: Comparison DataFrame returned by compare_results.

    Returns:
    -------
        A tuple of three DataFrames:
        per_doc: Metrics grouped by document and construct.
        concatenated: Metrics pooled across documents by construct, plus overall.
        weighted: Weighted median, minimum, and maximum metrics by construct,
            plus overall.

    """
    if df.empty:
        empty_counts = pd.DataFrame(columns=["doc_id", "construct", "tp", "fp", "fn"])
        return empty_counts, pd.DataFrame(), pd.DataFrame()

    per_doc_counts = (
        df.groupby(["doc_id", "construct"])[["tp", "fp", "fn"]].sum().reset_index()
    )
    per_doc = pd.concat(
        [
            per_doc_counts[["doc_id", "construct"]].reset_index(drop=True),
            _metrics_for_counts(per_doc_counts),
        ],
        axis=1,
    )

    pooled_counts = df.groupby("construct")[["tp", "fp", "fn"]].sum().reset_index()
    overall_counts = pd.DataFrame(
        [
            {
                "construct": "Overall",
                "tp": pooled_counts["tp"].sum(),
                "fp": pooled_counts["fp"].sum(),
                "fn": pooled_counts["fn"].sum(),
            }
        ]
    )
    pooled_counts = pd.concat([pooled_counts, overall_counts], ignore_index=True)
    concatenated = pd.concat(
        [
            pooled_counts[["construct"]].reset_index(drop=True),
            _metrics_for_counts(pooled_counts),
        ],
        axis=1,
    )

    total_docs = df["doc_id"].nunique()
    construct_names = df["construct"].unique().tolist()
    concatenated_stats = [
        _doc_stats(row.construct, per_doc, total_docs, construct_names)
        for row in concatenated.itertuples()
    ]
    concatenated = pd.concat([concatenated, pd.DataFrame(concatenated_stats)], axis=1)

    pr_auc_by_construct = compute_pr_auc(df)
    concatenated["pr_auc"] = concatenated["construct"].map(pr_auc_by_construct)

    per_doc_pr_auc = []
    for row in per_doc.itertuples():
        rows_for_doc_construct = df[
            (df["doc_id"] == row.doc_id) & (df["construct"] == row.construct)
        ]
        scores = compute_pr_auc(rows_for_doc_construct)
        per_doc_pr_auc.append(scores.get(row.construct))
    per_doc["pr_auc"] = per_doc_pr_auc

    weighted = _weighted_summary(per_doc)
    weighted_stats = [
        _doc_stats(row.construct, per_doc, total_docs, construct_names)
        for row in weighted.itertuples()
    ]
    weighted = pd.concat([weighted, pd.DataFrame(weighted_stats)], axis=1)

    for table in [per_doc, concatenated]:
        for metric in [*METRICS, "pr_auc"]:
            if metric in table:
                table[metric] = table[metric].round(4)

    return per_doc, concatenated, weighted


def _format_range(row: pd.Series, metric: str) -> str:
    """Format one weighted summary metric for display.

    Args:
    ----
        row: Weighted summary row containing median, minimum, and maximum
            columns for the metric.
        metric: Base metric name, such as sensitivity, precision, f1, pabak,
            or pr_auc.

    Returns:
    -------
        Display string with the median followed by the minimum and maximum
        values. Returns a dash when the median is missing.

    """
    median_value = row.get(f"{metric}_median")
    if median_value is None or pd.isna(median_value):
        return "-"

    min_value = row[f"{metric}_min"]
    max_value = row[f"{metric}_max"]
    return f"{median_value:.2f} [{min_value:.2f}-{max_value:.2f}]"


def _format_doc_stats(row: pd.Series) -> str:
    """Format document count and percentile spread for display.

    Args:
    ----
        row: Summary row containing n_docs, p5, and p95 values.

    Returns:
    -------
        Display string with document count followed by the 5th and 95th
        percentile values.

    """
    document_count = int(row["n_docs"])
    p5 = row["p5"]
    p95 = row["p95"]
    return f"{document_count} [{p5:.2f}-{p95:.2f}]"


def format_concatenated(concatenated: pd.DataFrame) -> pd.DataFrame:
    """Format pooled construct metrics for notebook display.

    Args:
    ----
        concatenated: Pooled construct metrics from compute_summary_tables.

    Returns:
    -------
        Display DataFrame with rounded metrics and formatted document stats.

    """
    if concatenated.empty:
        return concatenated

    display_columns = ["construct", "tp", "fp", "fn"]
    display = concatenated[display_columns].copy()

    for metric in [*METRICS, "pr_auc"]:
        display[metric] = concatenated[metric].apply(
            lambda value: "-" if pd.isna(value) else f"{value:.2f}"
        )

    display["interviews_with_construct [p5-p95]"] = concatenated.apply(
        _format_doc_stats, axis=1
    )
    return display


def format_weighted_summary(weighted_summary: pd.DataFrame) -> pd.DataFrame:
    """Format weighted summary metrics for notebook display.

    Args:
    ----
        weighted_summary: Weighted summary table from compute_summary_tables.

    Returns:
    -------
        Display DataFrame with metric ranges and formatted document stats.

    """
    if weighted_summary.empty:
        return weighted_summary

    display_columns = ["construct", "tp", "fp", "fn"]
    display = weighted_summary[display_columns].copy()

    for metric in [*METRICS, "pr_auc"]:
        display[metric] = weighted_summary.apply(
            lambda row, metric=metric: _format_range(row, metric), axis=1
        )

    display["interviews_with_construct [p5-p95]"] = weighted_summary.apply(
        _format_doc_stats, axis=1
    )
    return display


def refine_codebook(
    comparison_df: pd.DataFrame,
    concatenated_summary: pd.DataFrame,
    codebook: dict,
    pabak_threshold: float = 0.8,
) -> dict:
    """Build a partial codebook (envelope) of the constructs needing work.

    The returned partial is itself enveloped: its ``codebook`` holds only the
    changed constructs, and its ``metadata`` marks it a partial and records the
    source codebook it was built from. No version is minted here -- versions are
    only assigned by ``merge_codebooks``.

    Args:
    ----
        comparison_df: Row-level comparison table from compare_results.
        concatenated_summary: Concatenated summary from compute_summary_tables,
            with ``construct`` and ``pabak`` columns.
        codebook: Codebook envelope (or flat mapping) to refine.
        pabak_threshold: Constructs with PABAK strictly below this are refined.

    Returns:
    -------
        A partial codebook envelope containing only the changed constructs.

    """
    source_meta = _codebook_metadata(codebook)
    constructs = codebook_constructs(codebook)

    summary = concatenated_summary[concatenated_summary["construct"] != "Overall"]
    underperforming = {
        str(row.construct)
        for row in summary.itertuples()
        if row.pabak is not None
        and not pd.isna(row.pabak)
        and float(row.pabak) < pabak_threshold
    }

    changed: dict = {}
    for construct in underperforming:
        if construct not in constructs:
            print(
                f"Skipping '{construct}': below PABAK threshold but not present "
                f"in the codebook."
            )
            continue

        entry = copy.deepcopy(constructs[construct])
        rows = comparison_df[comparison_df["construct"] == construct]

        fn_quotes = [
            str(q).strip()
            for q in rows.loc[rows["status"] == "human_only", "human_quote"]
            if isinstance(q, str) and q.strip()
        ]
        fp_quotes = [
            str(q).strip()
            for q in rows.loc[rows["status"] == "llm_only", "llm_quote"]
            if isinstance(q, str) and q.strip()
        ]

        added = False
        if fn_quotes:
            entry.setdefault("examples", [])
            added |= _extend_unique(entry["examples"], fn_quotes)
        if fp_quotes:
            entry.setdefault("counter_examples", [])
            added |= _extend_unique(entry["counter_examples"], fp_quotes)

        if added:
            changed[construct] = entry

    partial_meta = {
        "name": source_meta.get("name", ""),
        "partial": True,
        "citation": source_meta.get("citation", ""),
        "built_from": [
            {
                "name": source_meta.get("name", ""),
                "version": source_meta.get("version"),
            }
        ],
    }
    return {"metadata": partial_meta, "codebook": changed}


def optimize_codebook(
    comparison_df: pd.DataFrame,
    concatenated_summary: pd.DataFrame,
    codebook: dict,
    human_results: dict,
    analyzer: "LLMTrackerAnalyzer",
    csv_path: Path | str,
    analyze_kwargs: dict,
    base_name: str,
    output_dir: Path | str = ".",
    pabak_threshold: float = 0.8,
    rerun_optimized_codebook: int = 0,
) -> list[dict]:
    """Iteratively refine the poorly performing constructs in a codebook.

    Pass 1 uses the supplied comparison table and summary to produce a partial
    codebook of the constructs below ``pabak_threshold`` (saved as ``v001``).
    Each additional rerun re-codes the documents using ONLY the previous
    partial, compares against the human data filtered to those same constructs,
    recomputes metrics, and refines again -- focusing the loop ever more tightly
    on the constructs that are still struggling.

    Each iteration's partial is written to::

        {output_dir}/{base_name}_optimized_codebook_v{NNN}_{timestamp}.json

    Args:
    ----
        comparison_df: Pass-1 comparison table from compare_results.
        concatenated_summary: Pass-1 concatenated summary from
            compute_summary_tables.
        codebook: The starting (full) codebook envelope.
        human_results: Original human coding, keyed by document ID. Used,
            filtered to the flagged constructs, for re-comparison each rerun.
        analyzer: An LLMTrackerAnalyzer used to re-code each rerun. Its config
            also drives the matcher used for re-comparison.
        csv_path: The CSV of documents to re-code (the same corpus each pass).
        analyze_kwargs: Keyword arguments forwarded to analyzer.analyze_csv each
            rerun (e.g. {"text_column": "post"} plus whatever document-ID columns
            that corpus uses). The loop makes no assumptions about the schema; it
            simply replays the coding call you used originally.
        base_name: Prefix for saved file names.
        output_dir: Directory to write the versioned partials into.
        pabak_threshold: Constructs with PABAK strictly below this are refined.
        rerun_optimized_codebook: Number of additional reruns after pass 1.
            0 (default) produces only v001.

    Returns:
    -------
        The list of partial codebook envelopes produced (v001, v002, ...).

    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(partial: dict, version: int) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = f"{base_name}_optimized_codebook_v{version:03d}_{timestamp}.json"
        path = out_dir / name
        _write_codebook(partial, path)
        return path

    partials: list[dict] = []

    # --- Pass 1: use the supplied tables (no re-coding) ---
    partial = refine_codebook(
        comparison_df, concatenated_summary, codebook, pabak_threshold
    )
    if not partial["codebook"]:
        print("No constructs below the PABAK threshold; nothing to optimize.")
        return partials

    prev_path = _save(partial, 1)
    partials.append(partial)

    # --- Reruns: re-code with the previous partial only (Option B) ---
    comparer = LLMTrackerComparer(config=analyzer.config)

    for i in range(rerun_optimized_codebook):
        version = i + 2  # v002, v003, ...

        # Re-code using ONLY the previous partial codebook.
        llm_results, _meta, _errors = analyzer.analyze_csv(
            csv_path=csv_path,
            codebook_path=prev_path,
            **analyze_kwargs,
        )

        # Compare against human data filtered to the partial's constructs.
        flagged = set(partial["codebook"].keys())
        filtered_human = _filter_human_results(human_results, flagged)
        new_comparison = comparer.compare_results(filtered_human, llm_results)
        _per_doc, new_concat, _weighted = compute_summary_tables(new_comparison)

        # Refine again, accumulating onto the previous partial's entries.
        next_partial = refine_codebook(
            new_comparison, new_concat, partial, pabak_threshold
        )
        if not next_partial["codebook"]:
            print(
                f"No constructs below threshold after v{version - 1:03d}; "
                f"stopping early."
            )
            break

        prev_path = _save(next_partial, version)
        partials.append(next_partial)
        partial = next_partial

    return partials


def merge_codebooks(
    base: dict,
    partial: dict,
    version: int | None = None,
    output_path: Path | str | None = None,
) -> dict:
    """Merge a partial (changed-constructs) codebook into a base codebook.

    Each construct in the partial replaces the corresponding entry in the base.
    Constructs absent from the partial are left untouched. A new version is
    minted: ``version`` if given, otherwise the base version + 1. Lineage
    (``built_from``) records the base codebook's name and version.

    Args:
    ----
        base: The full codebook envelope to update.
        partial: A partial codebook envelope from refine_codebook / optimize.
        version: Optional explicit version for the merged codebook. If omitted,
            the base version is incremented by 1.
        output_path: Optional path to write the merged codebook envelope to.

    Returns:
    -------
        The merged codebook envelope (new dict).

    """
    base = ensure_codebook_envelope(copy.deepcopy(base))
    base_meta = base["metadata"]
    base_constructs = base["codebook"]

    if not partial.get("metadata", {}).get("partial"):
        print(
            "Warning: 'partial' argument is not flagged as a partial codebook; "
            "merging it anyway (its constructs will replace the base's)."
        )
    partial_constructs = codebook_constructs(partial)

    merged_constructs = copy.deepcopy(base_constructs)
    for construct, entry in partial_constructs.items():
        merged_constructs[construct] = copy.deepcopy(entry)

    base_version = base_meta.get("version", 1) or 1
    new_version = version if version is not None else base_version + 1

    merged_meta = {
        "name": base_meta.get("name", ""),
        "version": new_version,
        "citation": base_meta.get("citation", ""),
        "built_from": [
            {"name": base_meta.get("name", ""), "version": base_meta.get("version")}
        ],
    }
    merged = {"metadata": merged_meta, "codebook": merged_constructs}

    if output_path is not None:
        _write_codebook(merged, output_path)
    return merged


def _codebook_metadata(codebook: dict) -> dict:
    """Return the metadata block of a codebook envelope (defaults if flat)."""
    if isinstance(codebook, dict) and "metadata" in codebook:
        meta = codebook["metadata"]
        if isinstance(meta, dict):
            return meta
    return {"name": "", "version": 1, "citation": "", "built_from": []}


def _filter_human_results(human_results: dict, constructs: set) -> dict:
    """Filter each document's human instances to the given construct set.

    Documents that end up with no matching instances are kept with an empty
    instance list, so the document set still aligns with the LLM results during
    comparison.
    """
    filtered: dict = {}
    for doc_id, result in human_results.items():
        kept = [inst for inst in result.instances if inst.construct in constructs]
        filtered[doc_id] = AnalysisResult(document_id=doc_id, instances=kept)
    return filtered


def _extend_unique(target: list, new_items: list) -> bool:
    """Append items not already present. Returns True if anything was added."""
    seen = set(target)
    added = False
    for item in new_items:
        if item not in seen:
            target.append(item)
            seen.add(item)
            added = True
    return added


def _write_codebook(codebook: dict, output_path: Path | str) -> None:
    """Write a codebook dict to JSON."""
    Path(output_path).write_text(
        json.dumps(codebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
