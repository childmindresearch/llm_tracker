"""Comparison utilities for aligning human and LLM-coded results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Callable, Optional

from pychometrics.config import AnalyzerConfig
from pychometrics.prompting import PromptingError, call_llm_api


MATCH_PROMPT_TEMPLATE = """You are comparing two quotes for the SAME psychological construct.

Construct:
{construct}

Human-coded quote:
{human_quote}

LLM-coded quote:
{llm_quote}

Task:
Decide if these quotes refer to the SAME passage or substantially the SAME idea.
Return ONLY valid JSON in this format:
{{
  "match": true | false,
  "match_confidence": <float between 0.0 and 1.0>
}}
"""


class ComparisonError(Exception):
    """Exception raised when comparison fails."""

    pass


@dataclass
class QuoteMatchDecision:
    match: bool
    method: str
    match_confidence: Optional[float] = None


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
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


def _parse_match_response(response_text: str) -> QuoteMatchDecision:
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

    if not isinstance(data, dict) or "match" not in data:
        raise ComparisonError("Matcher response JSON must include a 'match' field.")

    match = bool(data.get("match"))
    raw_confidence = data.get("match_confidence", 1.0 if match else 0.0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 1.0 if match else 0.0

    confidence = max(0.0, min(1.0, confidence))

    return QuoteMatchDecision(match=match, method="llm", match_confidence=confidence)


def _build_match_prompt(construct: str, human_quote: str, llm_quote: str) -> str:
    return MATCH_PROMPT_TEMPLATE.format(
        construct=construct, human_quote=human_quote, llm_quote=llm_quote
    )


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
    output_name: Optional[str] = None, base_dir: Optional[Path | str] = None
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
    max_quote_length: Optional[int] = None,
) -> "pd.DataFrame":
    """Build a row-level comparison dataframe from compare_documents() output.

    Args:
        result: A single-document comparison result from compare_documents().
        max_quote_length: Optional quote truncation for display convenience.

    Returns:
        pandas DataFrame with one row per matched/human_only/llm_only item.
    """

    rows: list[dict] = []
    comparisons = result.get("comparisons", [])
    document_id = str(result.get("document_id", "unknown_document"))

    for block in comparisons:
        construct = str(block.get("construct", "Unknown"))

        for match in block.get("matched", []):
            row = {
                "document_id": document_id,
                "construct": construct,
                "status": "matched",
                "method": str(match.get("match_method", "")),
                "human_confidence": match.get("human_confidence"),
                "llm_confidence": match.get("llm_confidence"),
                "match_confidence": match.get("match_confidence"),
                "human_quote": str(match.get("human_quote", "")),
                "llm_quote": str(match.get("llm_quote", "")),
            }
            if max_quote_length is not None:
                row["human_quote"] = _truncate(row["human_quote"], max_quote_length)
                row["llm_quote"] = _truncate(row["llm_quote"], max_quote_length)
            rows.append(row)

        for item in block.get("human_only", []):
            row = {
                "document_id": document_id,
                "construct": construct,
                "status": "human_only",
                "method": "",
                "human_confidence": item.get("human_confidence"),
                "llm_confidence": item.get("llm_confidence"),
                "match_confidence": None,
                "human_quote": str(item.get("quote", "")),
                "llm_quote": "",
            }
            if max_quote_length is not None:
                row["human_quote"] = _truncate(row["human_quote"], max_quote_length)
            rows.append(row)

        for item in block.get("llm_only", []):
            row = {
                "document_id": document_id,
                "construct": construct,
                "status": "llm_only",
                "method": "",
                "human_confidence": item.get("human_confidence"),
                "llm_confidence": item.get("llm_confidence"),
                "match_confidence": None,
                "human_quote": "",
                "llm_quote": str(item.get("quote", "")),
            }
            if max_quote_length is not None:
                row["llm_quote"] = _truncate(row["llm_quote"], max_quote_length)
            rows.append(row)

    columns = [
        "document_id",
        "construct",
        "status",
        "method",
        "human_confidence",
        "llm_confidence",
        "match_confidence",
        "human_quote",
        "llm_quote",
    ]

    return pd.DataFrame(rows, columns=columns)


class PychometricsComparator:
    """Compare human-coded and LLM-coded results using an LLM matcher."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        match_model: Optional[str] = None,
        config: Optional[AnalyzerConfig] = None,
        match_fn: Optional[Callable[[str, dict, dict], QuoteMatchDecision]] = None,
    ) -> None:
        if match_fn is not None:
            self.match_fn = match_fn
        else:
            self.match_fn = None

        if config is not None:
            self.config = config
        else:
            if match_model is not None:
                self.config = AnalyzerConfig(api_key=api_key, model_name=match_model)
            else:
                self.config = AnalyzerConfig(api_key=api_key)

    def _llm_match(self, construct: str, human: dict, llm: dict) -> QuoteMatchDecision:
        prompt = _build_match_prompt(
            construct, human.get("quote", ""), llm.get("quote", "")
        )

        attempts = 0
        max_attempts = self.config.max_retries + 1
        last_error: Optional[Exception] = None

        while attempts < max_attempts:
            attempts += 1
            try:
                response_text, _metadata = call_llm_api(prompt, self.config)
                return _parse_match_response(response_text)
            except (PromptingError, ComparisonError, json.JSONDecodeError) as e:
                last_error = e
                if attempts >= max_attempts:
                    break

        raise ComparisonError(
            f"Matcher failed after {max_attempts} attempts: {last_error}"
        )

    def _match_quotes(
        self, construct: str, human: dict, llm: dict
    ) -> QuoteMatchDecision:
        if self.match_fn is not None:
            return self.match_fn(construct, human, llm)
        return self._llm_match(construct, human, llm)

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

            used_llm_indices: set[int] = set()

            for h in list(human_list):
                for idx, l in enumerate(llm_list):
                    if idx in used_llm_indices:
                        continue
                    if h.get("quote") == l.get("quote"):
                        matched.append(
                            {
                                "construct": construct,
                                "human_quote": h.get("quote"),
                                "llm_quote": l.get("quote"),
                                "human_confidence": h.get("confidence"),
                                "llm_confidence": l.get("confidence"),
                                "match": True,
                                "match_method": "exact",
                                "match_confidence": 1.0,
                            }
                        )
                        used_llm_indices.add(idx)
                        human_list.remove(h)
                        break

            for h in human_list:
                found = False
                for idx, l in enumerate(llm_list):
                    if idx in used_llm_indices:
                        continue
                    decision = self._match_quotes(construct, h, l)
                    if decision.match:
                        match_confidence = (
                            decision.match_confidence
                            if decision.match_confidence is not None
                            else 0.5
                        )
                        matched.append(
                            {
                                "construct": construct,
                                "human_quote": h.get("quote"),
                                "llm_quote": l.get("quote"),
                                "human_confidence": h.get("confidence"),
                                "llm_confidence": l.get("confidence"),
                                "match": True,
                                "match_method": decision.method,
                                "match_confidence": match_confidence,
                            }
                        )
                        used_llm_indices.add(idx)
                        found = True
                        break
                if not found:
                    human_only.append(
                        {
                            "construct": construct,
                            "quote": h.get("quote"),
                            "speaker_id": h.get("speaker_id"),
                            "human_confidence": h.get("confidence"),
                            "llm_confidence": None,
                        }
                    )

            for idx, l in enumerate(llm_list):
                if idx in used_llm_indices:
                    continue
                llm_only.append(
                    {
                        "construct": construct,
                        "quote": l.get("quote"),
                        "speaker_id": l.get("speaker_id"),
                        "human_confidence": None,
                        "llm_confidence": l.get("confidence"),
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
        output_dir: Optional[str] = None,
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
        output_dir: Optional[str] = None,
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

        output_path: Optional[Path] = None
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
