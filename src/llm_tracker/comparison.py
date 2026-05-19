"""Compare human and LLM construct codings.

The comparison pipeline is intentionally DataFrame-first:

1. ``LLMTrackerComparer.compare_results`` aligns human and LLM quote instances.
2. It returns one row per matched, human-only, or LLM-only instance.
3. ``compute_summary_tables`` computes agreement metrics from those rows.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from llm_tracker.config import AnalyzerConfig
from llm_tracker.models import AnalysisResult, ConstructInstance
from llm_tracker.prompting import PromptingError, call_llm_api


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
        path: Path to a JSON file produced by the analyzer or by save_human_results()

    Returns:
        The parsed AnalysisResult.

    Raises:
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
        path: Either an analyzer run directory or its `encodings` subdir.

    Returns:
        Directory containing per-document result JSON files.

    Raises:
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
        path: Either an analyzer run directory or its `encodings` subdirectory.

    Returns:
        A dictionary mapping each document ID to its result JSON path.

    Raises:
        ComparisonError: If the path cannot be resolved to a directory
            containing result JSON files.
    """
    return {p.stem: p for p in _result_json_dir(path).glob("*.json")}


def _save_comparison_table(df: pd.DataFrame, output_dir: str) -> Path:
    """Save the row-level comparison table to a timestamped CSV folder.

    Args:
        df: Comparison DataFrame returned by compare_results().
        output_dir: Prefix for the timestamped output directory.

    Returns:
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
    grouped: dict[str, list[ConstructInstance]] = {}
    for item in instances:
        grouped.setdefault(item.construct, []).append(item)
    return grouped


def _format_quotes(quotes: list[ConstructInstance]) -> str:
    return "\n".join(
        f'{i}. "{item.quote}" '
        f"(indices: {item.quote_index if item.quote_index else 'N/A'})"
        for i, item in enumerate(quotes)
    )


def _parse_match_response(response_text: str) -> list[dict]:
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ComparisonError(f"Invalid matcher JSON: {e}") from e

    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ComparisonError("Matcher response must contain a 'matches' list.")

    parsed = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        try:
            confidence = float(match.get("match_confidence", 0.5))
            parsed.append(
                {
                    "human_index": int(match["human_index"]),
                    "llm_index": int(match["llm_index"]),
                    "paraphrase": bool(match.get("paraphrase", False)),
                    "match_confidence": max(0.0, min(1.0, confidence)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


def _parse_span(span: str | None) -> tuple[int, int] | None:
    if not span:
        return None
    try:
        start, end = map(int, span.split(":"))
    except (AttributeError, ValueError):
        return None
    return (start, end) if end >= start else None


def _compute_span_overlap(human_idx: str | None, llm_idx: str | None) -> float | None:
    """Compute Jaccard overlap between two ``start:end`` character spans."""
    human_span = _parse_span(human_idx)
    llm_span = _parse_span(llm_idx)
    if human_span is None or llm_span is None:
        return None

    h_start, h_end = human_span
    l_start, l_end = llm_span
    h_len = h_end - h_start
    l_len = l_end - l_start
    if h_len == 0 and l_len == 0:
        return 1.0
    if h_len == 0 or l_len == 0:
        return 0.0

    overlap = max(0, min(h_end, l_end) - max(h_start, l_start))
    union = h_len + l_len - overlap
    return overlap / union if union else 0.0


def _base_row(doc_id: str, construct: str, status: str) -> dict:
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
    ):
        prompt = MATCH_PROMPT_TEMPLATE.format(
            construct=construct,
            human_quotes=_format_quotes(human),
            llm_quotes=_format_quotes(llm),
        )
        last_error = None
        for _ in range(self.config.max_retries + 1):
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
        rows = []
        if not human:
            for item in llm:
                row = _base_row(doc_id, construct, "llm_only")
                row.update(
                    llm_quote=item.quote,
                    llm_indices=item.quote_index,
                    llm_confidence=item.confidence,
                    fp=1,
                )
                rows.append(row)
            return rows

        if not llm:
            for item in human:
                row = _base_row(doc_id, construct, "human_only")
                row.update(
                    human_quote=item.quote,
                    human_indices=item.quote_index,
                    human_confidence=item.confidence,
                    fn=1,
                )
                rows.append(row)
            return rows

        used_human: set[int] = set()
        used_llm: set[int] = set()
        for match in self._match_construct(construct, human, llm):
            h_idx = match["human_index"]
            l_idx = match["llm_index"]
            if (
                h_idx in used_human
                or l_idx in used_llm
                or not 0 <= h_idx < len(human)
                or not 0 <= l_idx < len(llm)
            ):
                continue

            human_item = human[h_idx]
            llm_item = llm[l_idx]
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
            used_human.add(h_idx)
            used_llm.add(l_idx)

        for i, item in enumerate(human):
            if i not in used_human:
                row = _base_row(doc_id, construct, "human_only")
                row.update(
                    human_quote=item.quote,
                    human_indices=item.quote_index,
                    human_confidence=item.confidence,
                    fn=1,
                )
                rows.append(row)

        for i, item in enumerate(llm):
            if i not in used_llm:
                row = _base_row(doc_id, construct, "llm_only")
                row.update(
                    llm_quote=item.quote,
                    llm_indices=item.quote_index,
                    llm_confidence=item.confidence,
                    fp=1,
                )
                rows.append(row)

        return rows

    def compare_results(
        self,
        human_results: dict[str, AnalysisResult],
        llm_results: dict[str, AnalysisResult],
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Compare loaded human and LLM results and return row-level outcomes."""
        rows = []

        for doc_id in sorted(set(human_results) | set(llm_results)):
            human_result = human_results.get(doc_id)
            llm_result = llm_results.get(doc_id)
            human_instances = human_result.instances if human_result else []
            llm_instances = llm_result.instances if llm_result else []
            human_by_construct = _group_by_construct(human_instances)
            llm_by_construct = _group_by_construct(llm_instances)
            constructs = sorted(set(human_by_construct) | set(llm_by_construct))

            for construct in constructs:
                rows.extend(
                    self._compare_construct(
                        doc_id,
                        construct,
                        human_by_construct.get(construct, []),
                        llm_by_construct.get(construct, []),
                    )
                )

        df = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
        if output_dir:
            _save_comparison_table(df, output_dir)
        return df

    def compare_documents(
        self,
        human_json: Path | str,
        llm_json: Path | str,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Compare one human result JSON with one LLM result JSON."""
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
        """Compare all JSON results in two analyzer output directories."""
        human_files = _result_files(human_dir)
        llm_files = _result_files(llm_dir)
        human_results: dict[str, AnalysisResult] = {}
        llm_results: dict[str, AnalysisResult] = {}

        for doc_id in sorted(set(human_files) | set(llm_files)):
            if doc_id in human_files:
                human_results[doc_id] = _load_result_json(human_files[doc_id])
            if doc_id in llm_files:
                llm_results[doc_id] = _load_result_json(llm_files[doc_id])

        return self.compare_results(human_results, llm_results, output_dir=output_dir)


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(tp: float, fp: float, fn: float) -> dict:
    tp, fp, fn = int(tp), int(fp), int(fn)
    union = tp + fp + fn
    observed_agreement = _divide(tp, union)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "union": union,
        "sensitivity": _divide(tp, tp + fn),
        "precision": _divide(tp, tp + fp),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "pabak": None if observed_agreement is None else 2 * observed_agreement - 1,
    }


def _metrics_for_counts(counts: pd.DataFrame) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame(columns=["tp", "fp", "fn", "union", *METRICS])
    return pd.DataFrame(
        [_metrics(row.tp, row.fp, row.fn) for row in counts.itertuples()]
    )


def compute_pr_auc(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute average precision from LLM coding confidence.

    Matched rows are correct LLM predictions. LLM-only rows are incorrect LLM
    predictions. Human-only rows are excluded because they are missed items, not
    LLM predictions. The ranking score is ``llm_confidence`` from the coding
    step, not matcher confidence.
    """
    if df.empty:
        return {"Overall": None}

    preds = df[df["status"].isin(["matched", "llm_only"])].copy()
    if preds.empty:
        return {"Overall": None}
    preds["label"] = (preds["status"] == "matched").astype(int)
    preds["score"] = pd.to_numeric(preds["llm_confidence"], errors="coerce")
    preds = preds.dropna(subset=["score"])

    results = {}
    groups = [("Overall", preds)] + [
        (construct, group) for construct, group in preds.groupby("construct")
    ]
    for name, group in groups:
        if len(set(group["label"])) < 2:
            results[name] = None
        else:
            results[name] = round(
                float(average_precision_score(group["label"], group["score"])), 4
            )
    return results


def _doc_stats(
    construct: str, per_doc: pd.DataFrame, total_docs: int, constructs: list[str]
) -> dict:
    if construct == "Overall":
        vals = per_doc["union"].tolist()
        possible = total_docs * len(constructs)
        n_docs = per_doc["doc_id"].nunique()
    else:
        rows = per_doc[per_doc["construct"] == construct]
        vals = rows["union"].tolist()
        possible = total_docs
        n_docs = len(rows)

    vals = vals + [0] * max(0, possible - len(vals))
    vals = vals or [0]
    return {
        "n_docs": n_docs,
        "p5": round(float(np.percentile(vals, 5)), 2),
        "p95": round(float(np.percentile(vals, 95)), 2),
    }


def _weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    midpoint = sum(weights) / 2
    total = 0.0
    for value, weight in pairs:
        total += weight
        if total >= midpoint:
            return value
    return pairs[-1][0]


def _weighted_summary(per_doc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    constructs = list(per_doc["construct"].unique()) if not per_doc.empty else []
    for construct in [*constructs, "Overall"]:
        group = (
            per_doc
            if construct == "Overall"
            else per_doc[per_doc["construct"] == construct]
        )
        row = {
            "construct": construct,
            "tp": int(group["tp"].sum()),
            "fp": int(group["fp"].sum()),
            "fn": int(group["fn"].sum()),
        }
        for metric in [*METRICS, "pr_auc"]:
            valid = group[["union", metric]].dropna()
            valid = valid[valid["union"] > 0]
            if valid.empty:
                row[f"{metric}_median"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
            else:
                values = valid[metric].tolist()
                weights = valid["union"].tolist()
                row[f"{metric}_median"] = round(_weighted_median(values, weights), 4)
                row[f"{metric}_min"] = round(min(values), 4)
                row[f"{metric}_max"] = round(max(values), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-document, pooled, and weighted comparison summaries."""
    if df.empty:
        empty_counts = pd.DataFrame(columns=["doc_id", "construct", "tp", "fp", "fn"])
        return empty_counts, pd.DataFrame(), pd.DataFrame()

    grouped = df.groupby(["doc_id", "construct"])[["tp", "fp", "fn"]]
    per_counts = grouped.sum().reset_index()
    per_doc = pd.concat(
        [
            per_counts[["doc_id", "construct"]].reset_index(drop=True),
            _metrics_for_counts(per_counts),
        ],
        axis=1,
    )

    concat_counts = df.groupby("construct")[["tp", "fp", "fn"]].sum().reset_index()
    overall = pd.DataFrame(
        [
            {
                "construct": "Overall",
                "tp": concat_counts["tp"].sum(),
                "fp": concat_counts["fp"].sum(),
                "fn": concat_counts["fn"].sum(),
            }
        ]
    )
    concat_counts = pd.concat([concat_counts, overall], ignore_index=True)
    concatenated = pd.concat(
        [
            concat_counts[["construct"]].reset_index(drop=True),
            _metrics_for_counts(concat_counts),
        ],
        axis=1,
    )

    total_docs = df["doc_id"].nunique()
    constructs = df["construct"].unique().tolist()
    stats = [
        _doc_stats(row.construct, per_doc, total_docs, constructs)
        for row in concatenated.itertuples()
    ]
    concatenated = pd.concat([concatenated, pd.DataFrame(stats)], axis=1)

    pr_auc = compute_pr_auc(df)
    concatenated["pr_auc"] = concatenated["construct"].map(pr_auc)
    per_doc["pr_auc"] = [
        compute_pr_auc(
            df[(df["doc_id"] == row.doc_id) & (df["construct"] == row.construct)]
        ).get(row.construct)
        for row in per_doc.itertuples()
    ]

    weighted = _weighted_summary(per_doc)
    weighted_stats = [
        _doc_stats(row.construct, per_doc, total_docs, constructs)
        for row in weighted.itertuples()
    ]
    weighted = pd.concat([weighted, pd.DataFrame(weighted_stats)], axis=1)

    for table in [per_doc, concatenated]:
        for metric in [*METRICS, "pr_auc"]:
            if metric in table:
                table[metric] = table[metric].round(4)

    return per_doc, concatenated, weighted


def format_per_interview(per_interview: pd.DataFrame) -> pd.DataFrame:
    """Return per-document metrics unchanged."""
    return per_interview


def _format_range(row: pd.Series, metric: str) -> str:
    median = row.get(f"{metric}_median")
    if median is None or pd.isna(median):
        return "-"
    return f"{median:.2f} [{row[f'{metric}_min']:.2f}-{row[f'{metric}_max']:.2f}]"


def _format_doc_stats(row: pd.Series) -> str:
    return f"{int(row['n_docs'])} [{row['p5']:.2f}-{row['p95']:.2f}]"


def format_concatenated(concatenated: pd.DataFrame) -> pd.DataFrame:
    """Format pooled construct metrics for notebook display."""
    if concatenated.empty:
        return concatenated
    display = concatenated[["construct", "tp", "fp", "fn"]].copy()
    for metric in [*METRICS, "pr_auc"]:
        display[metric] = concatenated[metric].apply(
            lambda value: "-" if pd.isna(value) else f"{value:.2f}"
        )
    display["interviews_with_construct [p5-p95]"] = concatenated.apply(
        _format_doc_stats, axis=1
    )
    return display


def format_weighted_summary(weighted_summary: pd.DataFrame) -> pd.DataFrame:
    """Format weighted summary metrics for notebook display."""
    if weighted_summary.empty:
        return weighted_summary
    display = weighted_summary[["construct", "tp", "fp", "fn"]].copy()
    for metric in [*METRICS, "pr_auc"]:
        display[metric] = weighted_summary.apply(
            lambda row, metric=metric: _format_range(row, metric), axis=1
        )
    display["interviews_with_construct [p5-p95]"] = weighted_summary.apply(
        _format_doc_stats, axis=1
    )
    return display
