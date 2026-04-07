"""Comparison utilities for aligning human and LLM-coded results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pychometrics.config import AnalyzerConfig
from pychometrics.prompting import PromptingError, call_llm_api


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

    pass


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

    return None


def _compute_span_overlap(human_idx: str | None, llm_idx: str | None) -> float | None:
    """Compute Jaccard overlap between two character-index spans ('start:end' format)."""
    if not human_idx or not llm_idx:
        return None
    try:
        h_start, h_end = map(int, human_idx.split(":"))
        l_start, l_end = map(int, llm_idx.split(":"))
    except (ValueError, AttributeError):
        return None

    human_chars = set(range(h_start, h_end))
    llm_chars = set(range(l_start, l_end))

    if not human_chars and not llm_chars:
        return 1.0
    if not human_chars or not llm_chars:
        return 0.0

    intersection = len(human_chars & llm_chars)
    union = len(human_chars | llm_chars)
    return intersection / union if union > 0 else 0.0


def _format_quotes_for_prompt(quotes: list[dict]) -> str:
    lines = []
    for i, q in enumerate(quotes):
        quote_text = q.get("quote", "")
        indices = q.get("quote_index", "N/A")
        lines.append(f'{i}. "{quote_text}" (indices: {indices})')
    return "\n".join(lines)


def _parse_construct_match_response(response_text: str) -> list[dict]:
    if response_text is None:
        raise ComparisonError("Empty response from LLM matcher.")

    cleaned = _strip_code_fences(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(cleaned) or _extract_first_json_object(
            response_text
        )
        if not extracted:
            raise ComparisonError("No valid JSON object found in matcher response.")
        data = json.loads(extracted)

    if not isinstance(data, dict) or "matches" not in data:
        raise ComparisonError("Matcher response must include a 'matches' field.")

    matches = data["matches"]
    if not isinstance(matches, list):
        raise ComparisonError("'matches' must be a list.")

    result = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        if "human_index" not in m or "llm_index" not in m:
            continue
        try:
            human_index = int(m["human_index"])
            llm_index = int(m["llm_index"])
        except (ValueError, TypeError):
            continue

        paraphrase = bool(m.get("paraphrase", False))
        try:
            confidence = float(m.get("match_confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        result.append(
            {
                "human_index": human_index,
                "llm_index": llm_index,
                "paraphrase": paraphrase,
                "match_confidence": confidence,
            }
        )

    return result


def _load_result_json(path: Path | str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise ComparisonError(f"Result JSON not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ComparisonError(f"Invalid JSON in {file_path}: {e}") from e


def _group_by_construct(instances: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for instance in instances:
        construct = str(instance.get("construct", "Unknown"))
        grouped.setdefault(construct, []).append(instance)
    return grouped


def _create_comparison_output_directory(
    output_name: str | None = None,
    base_dir: Path | str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dir_name = (
        f"{output_name}_{timestamp}" if output_name else f"comparison_{timestamp}"
    )
    output_dir = Path(base_dir) / dir_name if base_dir else Path.cwd() / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    return output_dir


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _format_single_comparison(
    result: dict,
    *,
    max_quote_length: int | None = None,
) -> "pd.DataFrame":
    """Build a row-level DataFrame from a single compare_documents() result.

    Private helper called by format_comparison_table. Not intended for direct use.
    """
    rows: list[dict] = []
    comparisons = result.get("comparisons", [])
    document_id = str(result.get("document_id", "unknown_document"))

    for block in comparisons:
        construct = str(block.get("construct", "Unknown"))

        for match in block.get("matched", []):
            human_quote = str(match.get("human_quote", ""))
            llm_quote = str(match.get("llm_quote", ""))
            if max_quote_length is not None:
                human_quote = _truncate(human_quote, max_quote_length)
                llm_quote = _truncate(llm_quote, max_quote_length)
            rows.append(
                {
                    "doc_id": document_id,
                    "construct": construct,
                    "status": "matched",
                    "human_quote": human_quote,
                    "llm_quote": llm_quote,
                    "human_indices": match.get("human_indices"),
                    "llm_indices": match.get("llm_indices"),
                    "paraphrase": match.get("paraphrase"),
                    "span_overlap": match.get("span_overlap"),
                    "match_confidence": match.get("match_confidence"),
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                }
            )

        for item in block.get("human_only", []):
            quote = str(item.get("quote", ""))
            if max_quote_length is not None:
                quote = _truncate(quote, max_quote_length)
            rows.append(
                {
                    "doc_id": document_id,
                    "construct": construct,
                    "status": "human_only",
                    "human_quote": quote,
                    "llm_quote": None,
                    "human_indices": item.get("indices"),
                    "llm_indices": None,
                    "paraphrase": None,
                    "span_overlap": None,
                    "match_confidence": None,
                    "tp": 0,
                    "fp": 0,
                    "fn": 1,
                }
            )

        for item in block.get("llm_only", []):
            quote = str(item.get("quote", ""))
            if max_quote_length is not None:
                quote = _truncate(quote, max_quote_length)
            rows.append(
                {
                    "doc_id": document_id,
                    "construct": construct,
                    "status": "llm_only",
                    "human_quote": None,
                    "llm_quote": quote,
                    "human_indices": None,
                    "llm_indices": item.get("indices"),
                    "paraphrase": None,
                    "span_overlap": None,
                    "match_confidence": None,
                    "tp": 0,
                    "fp": 1,
                    "fn": 0,
                }
            )

    columns = [
        "doc_id",
        "construct",
        "status",
        "human_quote",
        "llm_quote",
        "human_indices",
        "llm_indices",
        "paraphrase",
        "span_overlap",
        "match_confidence",
        "tp",
        "fp",
        "fn",
    ]

    return pd.DataFrame(rows, columns=columns)


def format_comparison_table(
    comparison_results: dict,
    *,
    max_quote_length: int | None = None,
) -> "pd.DataFrame":
    """Build a combined row-level DataFrame from all compare_directories() results.

    Args:
        comparison_results: Output of compare_directories() — a dict mapping
            document IDs to single-document comparison results.
        max_quote_length: Optional quote truncation for display convenience.

    Returns:
        pandas DataFrame with one row per matched/human_only/llm_only instance
        across all interviews. Prints a description and column definitions.
    """
    df = pd.concat(
        [
            _format_single_comparison(result, max_quote_length=max_quote_length)
            for result in comparison_results.values()
        ],
        ignore_index=True,
    )
    print(
        "Comparison Table\n"
        "----------------\n"
        "One row per coded instance across all interviews. Each row represents a quote\n"
        "identified by at least one coder, with its classification and match details.\n"
        "\n"
        "Columns:\n"
        "  doc_id           : interview identifier\n"
        "  construct        : psychological construct the instance belongs to\n"
        "  status           : matched (TP), human_only (FN), or llm_only (FP)\n"
        "  human_quote      : quote extracted by the human coder\n"
        "  llm_quote        : quote extracted by the LLM\n"
        "  human_indices    : character-level start:end indices of the human quote in the source text\n"
        "  llm_indices      : character-level start:end indices of the LLM quote in the source text\n"
        "  paraphrase       : True if the matched quotes differ meaningfully in wording\n"
        "  span_overlap     : Jaccard overlap between human and LLM character spans (matched rows only)\n"
        "  match_confidence : LLM matcher confidence that the two quotes refer to the same passage\n"
        "  tp/fp/fn         : binary indicators for this row's contribution to counts\n"
    )
    return df


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Return the weighted median of values, weighted by weights."""
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    cumulative = 0.0
    for val, w in pairs:
        cumulative += w
        if cumulative >= total / 2:
            return val
    return pairs[-1][0]


def _agreement_metrics_binary(rater_a: list[int], rater_b: list[int]) -> dict:
    """Compute Cohen's Kappa and PABAK between two binary raters.

    Args:
        rater_a: Binary ratings from rater A (0 or 1), one entry per item.
        rater_b: Binary ratings from rater B (0 or 1), same items, same order.

    Returns:
        Dict with cohens_kappa, prevalence_adjusted_kappa, observed_agreement,
        expected_agreement, and raw counts.

    Notes:
        TN = 0 by convention. This function is called with only the observed
        coded items (tp + fp + fn). Open-ended span coding tasks have no
        observable true negatives, so kappa will skew lower than tasks with
        a fixed item inventory. Do not compare these kappa values directly
        to benchmarks from fixed-item rating tasks.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater arrays must have the same length")

    n_items = len(rater_a)
    if n_items == 0:
        return {
            "cohens_kappa": None,
            "prevalence_adjusted_kappa": None,
            "observed_agreement": None,
            "expected_agreement": None,
        }

    both_positive = 0
    both_negative = 0
    a_positive_b_negative = 0
    a_negative_b_positive = 0

    for a, b in zip(rater_a, rater_b):
        if a == 1 and b == 1:
            both_positive += 1
        elif a == 0 and b == 0:
            both_negative += 1
        elif a == 1 and b == 0:
            a_positive_b_negative += 1
        elif a == 0 and b == 1:
            a_negative_b_positive += 1

    observed_agreement = (both_positive + both_negative) / n_items

    prob_a_positive = (both_positive + a_positive_b_negative) / n_items
    prob_b_positive = (both_positive + a_negative_b_positive) / n_items
    prob_a_negative = (both_negative + a_negative_b_positive) / n_items
    prob_b_negative = (both_negative + a_positive_b_negative) / n_items

    expected_agreement = (
        prob_a_positive * prob_b_positive + prob_a_negative * prob_b_negative
    )

    if expected_agreement == 1:
        cohens_kappa = 1.0
    else:
        cohens_kappa = (observed_agreement - expected_agreement) / (
            1 - expected_agreement
        )

    prevalence_adjusted_kappa = 2 * observed_agreement - 1

    return {
        "cohens_kappa": round(cohens_kappa, 4),
        "prevalence_adjusted_kappa": round(prevalence_adjusted_kappa, 4),
        "observed_agreement": round(observed_agreement, 4),
        "expected_agreement": round(expected_agreement, 4),
    }


def _metrics_from_counts(tp: float, fp: float, fn: float) -> dict:
    tp, fp, fn = int(tp), int(fp), int(fn)
    union = tp + fp + fn

    rater_a = [1] * tp + [1] * fn + [0] * fp
    rater_b = [1] * tp + [0] * fn + [1] * fp
    kappa_results = _agreement_metrics_binary(rater_a, rater_b)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "union": union,
        "sensitivity": _safe_divide(tp, tp + fn),
        "precision": _safe_divide(tp, tp + fp),
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "cohens_kappa": kappa_results["cohens_kappa"],
        "pabak": kappa_results["prevalence_adjusted_kappa"],
    }


def compute_pr_auc(df: "pd.DataFrame") -> dict[str, float | None]:
    """Compute PR AUC per construct and Overall from a comparison dataframe.

    LLM predictions are ranked by match_confidence. Matched rows are positive
    (label=1), llm_only rows are negative (label=0, score=0). human_only rows
    are excluded as they are not LLM predictions.

    Args:
        df: Concatenated output of format_comparison_table() across all documents.
            Must contain columns: construct, status, match_confidence.

    Returns:
        Dict mapping construct name (and 'Overall') to PR AUC float, or None
        if there are fewer than 2 unique labels for a construct.

    Notes:
        PR AUC is computed by pooling all LLM predictions across documents per
        construct. Per-document PR AUC is not computed as counts are too low
        to form a meaningful curve. score=0 is assigned to all llm_only rows
        (Option A convention - conservative, no schema changes required).
    """
    llm_preds = df[df["status"].isin(["matched", "llm_only"])].copy()
    llm_preds["_label"] = (llm_preds["status"] == "matched").astype(int)
    llm_preds["_score"] = llm_preds["match_confidence"].fillna(0.0)

    results: dict[str, float | None] = {}

    all_constructs = llm_preds["construct"].unique().tolist()
    groups = [("Overall", llm_preds)] + [
        (c, llm_preds[llm_preds["construct"] == c]) for c in all_constructs
    ]

    for name, group in groups:
        labels = group["_label"].tolist()
        scores = group["_score"].tolist()
        if len(set(labels)) < 2:
            results[name] = None
        else:
            results[name] = round(float(average_precision_score(labels, scores)), 4)

    return results


def compute_summary_tables(
    df: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """Compute per-interview, concatenated, and weighted summary tables.

    Args:
        df: Concatenated output of format_comparison_table() across all documents.
            Must contain columns: doc_id, construct, tp, fp, fn.

    Returns:
        Tuple of (per_interview, concatenated, weighted_summary) DataFrames.

        per_interview: one row per (doc_id, construct) with raw counts and metrics.
        concatenated: one row per construct (counts pooled across all docs) + Overall row.
            Includes n_docs (docs where construct appeared) and p5/p95 of instance counts.
        weighted_summary: one row per construct + Overall row, with weighted median
            and [min, max] of each metric across documents. Weight = union per document.
            Includes n_docs and p5/p95 of instance counts.
    """
    METRICS = ["sensitivity", "precision", "f1", "cohens_kappa", "pabak"]

    total_docs = df["doc_id"].nunique()
    all_constructs = df["construct"].unique().tolist()

    # --- Per interview ---
    grouped = (
        df.groupby(["doc_id", "construct"])[["tp", "fp", "fn"]].sum().reset_index()
    )
    metric_rows = [_metrics_from_counts(r.tp, r.fp, r.fn) for r in grouped.itertuples()]
    metric_df = pd.DataFrame(metric_rows)
    per_interview = pd.concat(
        [grouped[["doc_id", "construct"]].reset_index(drop=True), metric_df], axis=1
    )
    for metric in METRICS:
        per_interview[metric] = per_interview[metric].round(2)

    # --- Helper: n_docs and percentiles for a construct ---
    def _doc_stats(construct: str) -> dict:
        if construct == "Overall":
            union_vals = per_interview["union"].tolist()
            n_possible = total_docs * len(all_constructs)
            all_vals = union_vals + [0] * (n_possible - len(union_vals))
            n_docs = per_interview["doc_id"].nunique()
        else:
            rows = per_interview[per_interview["construct"] == construct]
            union_vals = rows["union"].tolist()
            all_vals = union_vals + [0] * (total_docs - len(union_vals))
            n_docs = len(rows)
        return {
            "n_docs": n_docs,
            "p5": round(float(np.percentile(all_vals, 5)), 2),
            "p95": round(float(np.percentile(all_vals, 95)), 2),
        }

    # --- Concatenated ---
    construct_totals = df.groupby("construct")[["tp", "fp", "fn"]].sum().reset_index()
    overall_row = pd.DataFrame(
        [
            {
                "construct": "Overall",
                "tp": construct_totals["tp"].sum(),
                "fp": construct_totals["fp"].sum(),
                "fn": construct_totals["fn"].sum(),
            }
        ]
    )
    concat_input = pd.concat([construct_totals, overall_row], ignore_index=True)
    concat_metrics = [
        _metrics_from_counts(r.tp, r.fp, r.fn) for r in concat_input.itertuples()
    ]
    concatenated = pd.concat(
        [
            concat_input[["construct"]].reset_index(drop=True),
            pd.DataFrame(concat_metrics),
        ],
        axis=1,
    )
    for metric in METRICS:
        concatenated[metric] = concatenated[metric].round(2)
    doc_stats = pd.DataFrame([_doc_stats(c) for c in concatenated["construct"]])
    concatenated = pd.concat([concatenated, doc_stats], axis=1)

    # --- Weighted summary (median [min, max]) ---
    weighted_rows = []
    constructs_with_overall = list(per_interview["construct"].unique()) + ["Overall"]

    for construct in constructs_with_overall:
        group = (
            per_interview
            if construct == "Overall"
            else per_interview[per_interview["construct"] == construct]
        )

        row: dict = {
            "construct": construct,
            "tp": int(group["tp"].sum()),
            "fp": int(group["fp"].sum()),
            "fn": int(group["fn"].sum()),
        }
        for metric in METRICS:
            valid = group[["union", metric]].dropna(subset=[metric])
            valid = valid[valid["union"] > 0]
            if valid.empty:
                row[f"{metric}_median"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
            else:
                vals = valid[metric].tolist()
                weights = valid["union"].tolist()
                row[f"{metric}_median"] = round(_weighted_median(vals, weights), 2)
                row[f"{metric}_min"] = round(min(vals), 2)
                row[f"{metric}_max"] = round(max(vals), 2)

        stats = _doc_stats(construct)
        row["n_docs"] = stats["n_docs"]
        row["p5"] = stats["p5"]
        row["p95"] = stats["p95"]
        weighted_rows.append(row)

    weighted_summary = pd.DataFrame(weighted_rows)

    # --- PR AUC ---
    pr_auc = compute_pr_auc(df)
    concatenated["pr_auc"] = concatenated["construct"].map(pr_auc)
    # PR AUC per (doc_id, construct)
    per_interview_pr_auc = []
    for _, row in per_interview[["doc_id", "construct"]].iterrows():
        doc_subset = df[
            (df["doc_id"] == row["doc_id"]) & (df["construct"] == row["construct"])
        ]
        result = compute_pr_auc(doc_subset)
        per_interview_pr_auc.append(result.get(row["construct"]))
    per_interview["pr_auc"] = per_interview_pr_auc

    # Aggregate pr_auc into weighted_summary: median [min, max] across per-doc values
    for construct in constructs_with_overall:
        group = (
            per_interview
            if construct == "Overall"
            else per_interview[per_interview["construct"] == construct]
        )
        valid = group["pr_auc"].dropna().tolist()
        idx = weighted_summary.index[weighted_summary["construct"] == construct][0]
        if len(valid) < 1:
            weighted_summary.at[idx, "pr_auc_median"] = None
            weighted_summary.at[idx, "pr_auc_min"] = None
            weighted_summary.at[idx, "pr_auc_max"] = None
        else:
            weights = group.loc[group["pr_auc"].notna(), "union"].tolist()
            weighted_summary.at[idx, "pr_auc_median"] = (
                round(_weighted_median(valid, weights), 4)
                if weights
                else round(float(sum(valid) / len(valid)), 4)
            )
            weighted_summary.at[idx, "pr_auc_min"] = round(min(valid), 4)
            weighted_summary.at[idx, "pr_auc_max"] = round(max(valid), 4)

    return per_interview, concatenated, weighted_summary


def format_per_interview(per_interview: "pd.DataFrame") -> "pd.DataFrame":
    """Format per_interview for display, printing a description and column definitions.

    Args:
        per_interview: Output of compute_summary_tables()[0].

    Returns:
        The per_interview DataFrame unchanged (formatting is handled by pandas display).
    """
    print(
        "Per-Interview Metrics\n"
        "---------------------\n"
        "One row per (interview, construct) combination. Constructs that did not appear\n"
        "in a given interview are absent — they do not appear as zero rows.\n"
        "\n"
        "Columns:\n"
        "  doc_id       : interview identifier\n"
        "  construct    : psychological construct\n"
        "  tp/fp/fn     : true positives, false positives, false negatives for this interview-construct\n"
        "  union        : total coded instances (TP + FP + FN)\n"
        "  sensitivity  : TP / (TP + FN)\n"
        "  precision    : TP / (TP + FP)\n"
        "  f1           : harmonic mean of sensitivity and precision\n"
        "  cohens_kappa : agreement beyond chance (TN=0 convention)\n"
        "  pabak        : prevalence-adjusted kappa\n"
        "  pr_auc       : area under precision-recall curve; NaN where insufficient label classes\n"
    )
    return per_interview


def format_weighted_summary(weighted_summary: "pd.DataFrame") -> "pd.DataFrame":
    """Format weighted_summary into display strings: 'median [min–max]'.

    Args:
        weighted_summary: Output of compute_summary_tables()[2].

    Returns:
        DataFrame with one display column per metric and interviews_with_construct [p5–p95] column.
    """
    print(
        "Weighted Summary\n"
        "----------------\n"
        "One row per construct. Metrics are computed per interview first, then summarized\n"
        "as weighted median [min\u2013max] across interviews, weighted by union size (TP+FP+FN).\n"
        "\n"
        "Columns:\n"
        "  tp/fp/fn                          : total true positives, false positives, false negatives across all interviews\n"
        "  sensitivity                       : TP / (TP + FN) — proportion of human-coded instances the LLM found\n"
        "  precision                         : TP / (TP + FP) — proportion of LLM-coded instances that were correct\n"
        "  f1                                : harmonic mean of sensitivity and precision\n"
        "  cohens_kappa                      : agreement beyond chance; computed over coded instances only (TN=0 convention)\n"
        "  pabak                             : prevalence-adjusted kappa; more stable when construct prevalence is low\n"
        "  pr_auc                            : area under precision-recall curve; ranks LLM predictions by match confidence\n"
        "  interviews_with_construct [p5-p95]: number of interviews containing the construct, with 5th-95th percentile of instance counts\n"
    )
    METRICS = ["sensitivity", "precision", "f1", "cohens_kappa", "pabak"]
    display = weighted_summary[["construct", "tp", "fp", "fn"]].copy()

    for metric in METRICS:
        med_col = f"{metric}_median"
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"

        def fmt_row(r, m=med_col, mn=min_col, mx=max_col):
            if pd.isna(r[m]):
                return "—"
            return f"{r[m]:.2f} [{r[mn]:.2f}–{r[mx]:.2f}]"

        display[metric] = weighted_summary.apply(fmt_row, axis=1)

    def fmt_n_docs(r):
        if pd.isna(r["p5"]):
            return str(int(r["n_docs"]))
        return f"{int(r['n_docs'])} [{r['p5']:.2f}–{r['p95']:.2f}]"

    def fmt_pr_auc(r):
        if r["pr_auc_median"] is None or (
            isinstance(r["pr_auc_median"], float) and pd.isna(r["pr_auc_median"])
        ):
            return "—"
        return f"{r['pr_auc_median']:.2f} [{r['pr_auc_min']:.2f}–{r['pr_auc_max']:.2f}]"

    display["pr_auc"] = weighted_summary.apply(fmt_pr_auc, axis=1)

    display["interviews_with_construct [p5–p95]"] = weighted_summary.apply(
        fmt_n_docs, axis=1
    )

    return display


def format_concatenated(concatenated: "pd.DataFrame") -> "pd.DataFrame":
    """Format concatenated into display strings, with interviews_with_construct [p5-p95] bracket column.

    Args:
        concatenated: Output of compute_summary_tables()[1].

    Returns:
        DataFrame with metric columns rounded to 2dp and interviews_with_construct [p5-p95] column.
        Separate p5 and p95 columns are dropped.
    """
    print(
        "Concatenated Metrics\n"
        "--------------------\n"
        "One row per construct. TP/FP/FN counts are pooled across all interviews before\n"
        "metrics are computed — treats the entire dataset as a single document.\n"
        "\n"
        "Columns:\n"
        "  tp/fp/fn                          : total counts pooled across all interviews\n"
        "  sensitivity                       : TP / (TP + FN) — proportion of human-coded instances the LLM found\n"
        "  precision                         : TP / (TP + FP) — proportion of LLM-coded instances that were correct\n"
        "  f1                                : harmonic mean of sensitivity and precision\n"
        "  cohens_kappa                      : agreement beyond chance; computed over coded instances only (TN=0 convention)\n"
        "  pabak                             : prevalence-adjusted kappa; more stable when construct prevalence is low\n"
        "  pr_auc                            : area under precision-recall curve; ranks LLM predictions by match confidence\n"
        "  interviews_with_construct [p5-p95]: number of interviews containing the construct, with 5th-95th percentile of instance counts\n"
    )
    METRICS = ["sensitivity", "precision", "f1", "cohens_kappa", "pabak"]
    display = concatenated[["construct", "tp", "fp", "fn"]].copy()

    for metric in METRICS:
        display[metric] = concatenated[metric].apply(
            lambda v: "—" if pd.isna(v) else f"{v:.2f}"
        )

    def fmt_n_docs(r):
        if pd.isna(r["p5"]):
            return str(int(r["n_docs"]))
        return f"{int(r['n_docs'])} [{r['p5']:.2f}-{r['p95']:.2f}]"

    display["pr_auc"] = concatenated["pr_auc"].apply(
        lambda v: "—"
        if v is None or (isinstance(v, float) and pd.isna(v))
        else f"{v:.2f}"
    )

    display["interviews_with_construct [p5-p95]"] = concatenated.apply(
        fmt_n_docs, axis=1
    )

    return display


class PychometricsComparator:
    """Compare human-coded and LLM-coded results using an LLM matcher."""

    def __init__(
        self,
        api_key: str | None = None,
        match_model: str | None = None,
        config: AnalyzerConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            if match_model is not None:
                self.config = AnalyzerConfig(api_key=api_key, model_name=match_model)
            else:
                self.config = AnalyzerConfig(api_key=api_key)

    def _llm_match_construct(
        self, construct: str, human_list: list[dict], llm_list: list[dict]
    ) -> list[dict]:
        """Make one LLM call for all quotes in a construct group and return match decisions."""
        prompt = MATCH_PROMPT_TEMPLATE.format(
            construct=construct,
            human_quotes=_format_quotes_for_prompt(human_list),
            llm_quotes=_format_quotes_for_prompt(llm_list),
        )

        attempts = 0
        max_attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                response_text, _ = call_llm_api(prompt, self.config)
                return _parse_construct_match_response(response_text)
            except (PromptingError, ComparisonError, json.JSONDecodeError) as e:
                last_error = e
                if attempts >= max_attempts:
                    break

        raise ComparisonError(
            f"Matcher failed after {max_attempts} attempts: {last_error}"
        )

    def _compare_instances(
        self, human_instances: list[dict], llm_instances: list[dict]
    ) -> list[dict]:
        comparisons: list[dict] = []
        human_by_construct = _group_by_construct(human_instances)
        llm_by_construct = _group_by_construct(llm_instances)
        all_constructs = sorted(set(human_by_construct) | set(llm_by_construct))

        for construct in all_constructs:
            human_list = list(human_by_construct.get(construct, []))
            llm_list = list(llm_by_construct.get(construct, []))

            matched: list[dict] = []
            human_only: list[dict] = []
            llm_only: list[dict] = []

            # If one side is empty, everything is unmatched
            if not human_list:
                for llm_item in llm_list:
                    llm_only.append(
                        {
                            "construct": construct,
                            "quote": llm_item.get("quote"),
                            "indices": llm_item.get("quote_index"),
                            "confidence": llm_item.get("confidence"),
                        }
                    )
                comparisons.append(
                    {
                        "construct": construct,
                        "matched": matched,
                        "human_only": human_only,
                        "llm_only": llm_only,
                    }
                )
                continue

            if not llm_list:
                for h in human_list:
                    human_only.append(
                        {
                            "construct": construct,
                            "quote": h.get("quote"),
                            "indices": h.get("quote_index"),
                            "confidence": h.get("confidence"),
                        }
                    )
                comparisons.append(
                    {
                        "construct": construct,
                        "matched": matched,
                        "human_only": human_only,
                        "llm_only": llm_only,
                    }
                )
                continue

            # One LLM call for all quotes in this construct
            try:
                match_decisions = self._llm_match_construct(
                    construct, human_list, llm_list
                )
            except ComparisonError:
                match_decisions = []

            used_human: set[int] = set()
            used_llm: set[int] = set()

            for decision in match_decisions:
                h_idx = decision["human_index"]
                l_idx = decision["llm_index"]

                if h_idx < 0 or h_idx >= len(human_list):
                    continue
                if l_idx < 0 or l_idx >= len(llm_list):
                    continue
                if h_idx in used_human or l_idx in used_llm:
                    continue

                h = human_list[h_idx]
                llm_item = llm_list[l_idx]

                matched.append(
                    {
                        "construct": construct,
                        "human_quote": h.get("quote"),
                        "llm_quote": llm_item.get("quote"),
                        "human_indices": h.get("quote_index"),
                        "llm_indices": llm_item.get("quote_index"),
                        "human_confidence": h.get("confidence"),
                        "llm_confidence": llm_item.get("confidence"),
                        "paraphrase": decision["paraphrase"],
                        "span_overlap": _compute_span_overlap(
                            h.get("quote_index"), llm_item.get("quote_index")
                        ),
                        "match_confidence": decision["match_confidence"],
                    }
                )
                used_human.add(h_idx)
                used_llm.add(l_idx)

            for i, h in enumerate(human_list):
                if i not in used_human:
                    human_only.append(
                        {
                            "construct": construct,
                            "quote": h.get("quote"),
                            "indices": h.get("quote_index"),
                            "confidence": h.get("confidence"),
                        }
                    )

            for i, llm_item in enumerate(llm_list):
                if i not in used_llm:
                    llm_only.append(
                        {
                            "construct": construct,
                            "quote": llm_item.get("quote"),
                            "indices": llm_item.get("quote_index"),
                            "confidence": llm_item.get("confidence"),
                        }
                    )

            comparisons.append(
                {
                    "construct": construct,
                    "matched": matched,
                    "human_only": human_only,
                    "llm_only": llm_only,
                }
            )

        return comparisons

    def compare_documents(
        self,
        human_json: Path | str,
        llm_json: Path | str,
        output_dir: str | None = None,
    ) -> dict:
        human_data = _load_result_json(human_json)
        llm_data = _load_result_json(llm_json)

        document_id = (
            human_data.get("document_id")
            or llm_data.get("document_id")
            or Path(human_json).stem
        )

        comparisons = self._compare_instances(
            human_data.get("instances", []), llm_data.get("instances", [])
        )

        result = {"document_id": document_id, "comparisons": comparisons}

        if output_dir:
            output_path = _create_comparison_output_directory(
                output_name=output_dir, base_dir=Path.cwd()
            )
            out_file = output_path / "comparisons" / f"{document_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    def compare_directories(
        self,
        human_dir: Path | str,
        llm_dir: Path | str,
        output_dir: str | None = None,
    ) -> dict[str, dict]:
        human_path = Path(human_dir)
        llm_path = Path(llm_dir)

        if not human_path.exists() or not human_path.is_dir():
            raise ComparisonError(f"Human directory not found: {human_path}")
        if not llm_path.exists() or not llm_path.is_dir():
            raise ComparisonError(f"LLM directory not found: {llm_path}")

        human_files = {p.stem: p for p in human_path.glob("*.json")}
        llm_files = {p.stem: p for p in llm_path.glob("*.json")}
        all_doc_ids = sorted(set(human_files) | set(llm_files))

        results: dict[str, dict] = {}

        output_path: Path | None = None
        if output_dir:
            output_path = _create_comparison_output_directory(
                output_name=output_dir, base_dir=Path.cwd()
            )

        for doc_id in all_doc_ids:
            human_file = human_files.get(doc_id)
            llm_file = llm_files.get(doc_id)

            human_data = _load_result_json(human_file) if human_file else {}
            llm_data = _load_result_json(llm_file) if llm_file else {}

            comparisons = self._compare_instances(
                human_data.get("instances", []), llm_data.get("instances", [])
            )

            result = {"document_id": doc_id, "comparisons": comparisons}
            results[doc_id] = result

            if output_path is not None:
                out_file = output_path / "comparisons" / f"{doc_id}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

        return results
