"""Comparison utilities for aligning human and LLM-coded results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import pandas as pd

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
            cleaned = cleaned[first_newline + 1:]
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
                    return text[start: idx + 1]

    return None


def _compute_span_overlap(
    human_idx: str | None, llm_idx: str | None
) -> float | None:
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
        extracted = (
            _extract_first_json_object(cleaned)
            or _extract_first_json_object(response_text)
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

        result.append({
            "human_index": human_index,
            "llm_index": llm_index,
            "paraphrase": paraphrase,
            "match_confidence": confidence,
        })

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


def format_comparison_table(
    result: dict,
    *,
    max_quote_length: int | None = None,
) -> "pd.DataFrame":
    """Build a row-level comparison dataframe from compare_documents() output.

    Args:
        result: A single-document comparison result from compare_documents().
        max_quote_length: Optional quote truncation for display convenience.

    Returns:
        pandas DataFrame with one row per matched/human_only/llm_only item.
        Columns: doc_id, construct, status, human_quote, llm_quote,
                 human_indices, llm_indices, paraphrase, span_overlap, match_confidence.
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
            rows.append({
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
            })

        for item in block.get("human_only", []):
            quote = str(item.get("quote", ""))
            if max_quote_length is not None:
                quote = _truncate(quote, max_quote_length)
            rows.append({
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
            })

        for item in block.get("llm_only", []):
            quote = str(item.get("quote", ""))
            if max_quote_length is not None:
                quote = _truncate(quote, max_quote_length)
            rows.append({
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
            })

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
    ]

    return pd.DataFrame(rows, columns=columns)


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
                    llm_only.append({
                        "construct": construct,
                        "quote": llm_item.get("quote"),
                        "indices": llm_item.get("quote_index"),
                        "confidence": llm_item.get("confidence"),
                    })
                comparisons.append({
                    "construct": construct,
                    "matched": matched,
                    "human_only": human_only,
                    "llm_only": llm_only,
                })
                continue

            if not llm_list:
                for h in human_list:
                    human_only.append({
                        "construct": construct,
                        "quote": h.get("quote"),
                        "indices": h.get("quote_index"),
                        "confidence": h.get("confidence"),
                    })
                comparisons.append({
                    "construct": construct,
                    "matched": matched,
                    "human_only": human_only,
                    "llm_only": llm_only,
                })
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

                matched.append({
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
                })
                used_human.add(h_idx)
                used_llm.add(l_idx)

            for i, h in enumerate(human_list):
                if i not in used_human:
                    human_only.append({
                        "construct": construct,
                        "quote": h.get("quote"),
                        "indices": h.get("quote_index"),
                        "confidence": h.get("confidence"),
                    })

            for i, llm_item in enumerate(llm_list):
                if i not in used_llm:
                    llm_only.append({
                        "construct": construct,
                        "quote": llm_item.get("quote"),
                        "indices": llm_item.get("quote_index"),
                        "confidence": llm_item.get("confidence"),
                    })

            comparisons.append({
                "construct": construct,
                "matched": matched,
                "human_only": human_only,
                "llm_only": llm_only,
            })

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
