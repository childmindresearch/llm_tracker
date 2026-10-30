"""Discovery mode: find constructs in documents through a theoretical framework.

Discovery is the inverse of coding. Instead of locating instances of known
codebook constructs, the LLM reads each document through a chosen theoretical
framework and reports whichever constructs it judges to be present, with a
description and supporting quotes.

The pipeline has three steps:

1. ``discover`` reads every supported file under a path, runs each document
   through the framework, and pools every finding into one DataFrame.
2. ``merge_constructs`` consolidates the freeform construct names into canonical
   constructs, each with a synthesized definition and its constituents scored by
   prototypicality.
3. ``build_codebook`` turns the merge result into a codebook envelope the coding
   pipeline can consume, seeded with example quotes drawn from the discovery.

Names are normalized in code before the LLM sees them, so ``self_hate``,
``SELF HATE`` and ``self hate`` collapse into one constituent without spending a
model call on it. The LLM is left to do the semantic grouping only.
"""

import json
import random
import re
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from numbers import Integral
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

import pandas as pd

from llm_tracker.config import AnalyzerConfig
from llm_tracker.models import APIMetadata, ErrorRecord
from llm_tracker.prompting import PromptingError, call_llm_api

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx", ".xls"}
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}

DISCOVERY_COLUMNS = [
    "filename",
    "doc_id",
    "construct",
    "description",
    "quote",
]

MERGED_COLUMNS = [
    "merged_construct",
    "definition",
    "status",
    "construct_found",
    "document_count",
    "instance_count",
    "document_frequency_percent",
    "total_documents",
    "prototypicality",
    "quotes",
]

DiscoveryFramework = Literal[
    "Western psychology",
    "Evidence-based psychotherapy",
    "Cognitive behavioral therapy",
    "Psychodynamic psychotherapy",
    "Humanistic psychology",
    "Dialectical behavior therapy",
    "Acceptance and commitment therapy",
    "Buddhism",
    "Feminism",
    "Critical race theory",
    "Social constructivism",
    "Cultural relativism",
    "Phenomenology",
    "Rational choice theory",
]

DISCOVERY_PROMPT = """You are an expert qualitative researcher analyzing a \
document through the lens of a specific theoretical framework.

Framework: {framework}

Read the document below and identify the constructs from this framework that \
are present in the text. Use your own judgement about what counts as a \
construct of this framework; you are not working from a fixed list.

For each construct you find, provide:
- "construct": a short name for the construct
- "description": one sentence describing the construct as it appears here
- "quotes": a list of verbatim quotes from the document that evidence it

Only report constructs genuinely grounded in the text. If none are present, \
return an empty list.
Spell out construct names and descriptions in full. Do not use acronyms such as
CBT; write cognitive behavioral therapy instead.

Respond ONLY with a JSON object of the form:
{{"constructs": [{{"construct": "...", "description": "...", \
"quotes": ["..."]}}]}}

Document:
{document}
"""

MERGE_PROMPT = """You are an expert qualitative researcher consolidating \
constructs discovered across many documents under the framework: {framework}.

Below is a list of discovered constructs, each with a description. They were \
named freely by a model reading one document at a time, so the same idea often \
appears under different names. Your task is to group them into canonical \
constructs.

For each canonical construct provide:
- "name": the canonical name (synthesize the best name; it may be one of the \
grouped names)
- "definition": a one-sentence definition of the consolidated construct
- "constituents": the discovered constructs you grouped into it, each with a \
"prototypicality" score from 0 to 1 (1 = the first thing that comes to mind \
for this canonical construct; lower = related but less obvious)

Guidance:
- Group on meaning, using the descriptions as well as the names.
- A construct does NOT have to be merged with anything. If it stands on its \
own, return it as a canonical construct with an empty "constituents" list and \
keep its own name and description.
- Do not invent constructs that are not in the list below, and do not place \
the same construct in more than one group.
- Consider ignoring very low-frequency constructs that are not very \
prototypical of the merged construct, so that the merged construct does not \
become heterogeneous. Anything you leave out will be reported back separately.
{existing_note}
Respond ONLY with a JSON object of the form:
{{"merged": [{{"name": "...", "definition": "...", "constituents": \
[{{"name": "...", "prototypicality": 0.0}}]}}]}}

Discovered constructs:
{constructs}
"""

EXISTING_NOTE = """
An existing consolidation is provided below. Fold the newly discovered \
constructs into it: add them as constituents of existing canonical constructs \
where they fit, and create new canonical constructs only when nothing \
existing matches. Keep existing canonical names and definitions stable unless \
a change is clearly warranted.

Existing consolidation:
{existing}
"""


class DiscoveryError(Exception):
    """Raised when a discovery or merge response cannot be used."""


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """Reduce a construct name to a key for exact-match deduplication.

    Case, surrounding whitespace, and the choice of separator between words are
    incidental to what a construct name means, so ``self_hate``, ``SELF HATE``
    and ``  self-hate `` all reduce to ``self hate``. Doing this in code keeps
    the LLM's merging work semantic rather than clerical.

    Args:
    ----
        name: Construct name as the model wrote it.

    Returns:
    -------
        Normalized key. Empty string if the name has no usable characters.

    """
    key = str(name).strip().casefold()
    key = re.sub(r"[_\-/]+", " ", key)
    key = re.sub(r"[^\w\s]+", "", key)
    return re.sub(r"\s+", " ", key).strip()


def _display_name(variants: list[str]) -> str:
    """Choose the spelling to show for a group of equivalent names.

    Args:
    ----
        variants: Every original spelling seen for one normalized key.

    Returns:
    -------
        The most frequent spelling, breaking ties alphabetically so the choice
        is deterministic across runs.

    """
    counts = Counter(variants)
    top = max(counts.values())
    return sorted(name for name, n in counts.items() if n == top)[0]


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV, XLSX, or XLS file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _documents_from_file(
    path: Path,
    text_column: str | None,
    id_column: str | None,
) -> list[tuple[str, str]]:
    """Extract (document_id, text) pairs from one file.

    A ``.txt`` file is a single document whose ID is the filename stem. A
    tabular file is one document per row when ``text_column`` is given;
    without it the whole table is flattened into a single document, matching
    how the coding pipeline treats a bare CSV.

    Args:
    ----
        path: File to read.
        text_column: Column holding document text in tabular files.
        id_column: Column holding the document ID in tabular files.

    Returns:
    -------
        A list of (document_id, text) pairs.

    Raises:
    ------
        DiscoveryError: If the file cannot be read or a named column is absent.

    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        try:
            return [(path.stem, path.read_text(encoding="utf-8"))]
        except OSError as e:
            raise DiscoveryError(f"Could not read {path.name}: {e}") from e

    try:
        df = _read_tabular(path)
    except Exception as e:  # noqa: BLE001 - pandas raises many reader errors
        raise DiscoveryError(f"Could not read {path.name}: {e}") from e

    if text_column is None:
        text = "\n".join(
            " ".join(str(value) for value in row if pd.notna(value))
            for row in df.itertuples(index=False)
        )
        return [(path.stem, text)]

    if text_column not in df.columns:
        raise DiscoveryError(
            f"{path.name}: no column '{text_column}'. Found: {list(df.columns)}"
        )
    if id_column is not None and id_column not in df.columns:
        raise DiscoveryError(
            f"{path.name}: no column '{id_column}'. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_column])
    documents = []
    seen: dict[str, int] = {}
    for position, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        base = (
            str(row_dict[id_column]).strip() if id_column is not None else str(position)
        )
        seen[base] = seen.get(base, 0) + 1
        doc_id = base if seen[base] == 1 else f"{base}_{seen[base]}"
        documents.append((doc_id, str(row_dict[text_column])))
    return documents


def get_input_files(path: Path | str) -> list[Path]:
    """List the supported files at a path.

    Args:
    ----
        path: A single file, or a directory to search (non-recursively).

    Returns:
    -------
        Sorted list of supported file paths.

    Raises:
    ------
        DiscoveryError: If the path does not exist, or a directory holds no
            supported files.

    """
    resolved = Path(path)
    if not resolved.exists():
        raise DiscoveryError(f"Path not found: {resolved}")

    if resolved.is_file():
        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DiscoveryError(
                f"Unsupported file type '{resolved.suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [resolved]

    files = sorted(
        f
        for f in resolved.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise DiscoveryError(
            f"No supported files in {resolved}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return files


# ---------------------------------------------------------------------------
# Step 1: discovery
# ---------------------------------------------------------------------------


def _parse_discovery_response(response_text: str) -> list[dict]:
    """Parse and validate one document's discovery response."""
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise DiscoveryError(f"Response is not valid JSON: {e}") from e

    constructs = data.get("constructs")
    if not isinstance(constructs, list):
        raise DiscoveryError('Response missing a "constructs" list.')

    cleaned: list[dict] = []
    for entry in constructs:
        if not isinstance(entry, dict) or not str(entry.get("construct", "")).strip():
            raise DiscoveryError(f"Malformed construct entry: {entry!r}")
        quotes = entry.get("quotes", [])
        if not isinstance(quotes, list):
            raise DiscoveryError(f"Quotes must be a list: {entry!r}")
        cleaned.append(
            {
                "construct": str(entry["construct"]).strip(),
                "description": str(entry.get("description", "")).strip(),
                "quotes": [str(q).strip() for q in quotes if str(q).strip()],
            }
        )
    return cleaned


class LLMTrackerDiscoverer:
    """Discover constructs in documents through a theoretical framework."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        config: AnalyzerConfig | None = None,
        discovery_prompt: str | None = None,
    ) -> None:
        """Create a discoverer.

        Args:
        ----
            api_key: API key or path to an env file containing it.
            model_name: Model to use. Ignored if config is given.
            config: Optional AnalyzerConfig. When provided, api_key and
                model_name are ignored.
            discovery_prompt: Optional template overriding DISCOVERY_PROMPT.
                Must contain ``{framework}`` and ``{document}`` fields.

        """
        if config is not None:
            self.config = config
        else:
            kwargs: dict = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if model_name is not None:
                kwargs["model_name"] = model_name
            self.config = AnalyzerConfig(**kwargs)
        self.discovery_prompt = discovery_prompt or DISCOVERY_PROMPT

    def discover(
        self,
        path: Path | str,
        framework: DiscoveryFramework | str,
        text_column: str | None = None,
        id_column: str | None = None,
        output_dir: Path | str | None = None,
        save_format: str = "csv",
    ) -> tuple[pd.DataFrame, dict[str, APIMetadata], list[ErrorRecord]]:
        """Discover constructs in every document at a path.

        Accepts a single file or a directory of ``.txt`` / ``.csv`` / ``.xlsx``
        / ``.xls`` files. Tabular files are treated as one document per row when
        ``text_column`` is given; the same column names are applied to every
        tabular file found.

        Args:
        ----
            path: A file, or a directory of supported files.
            framework: Theoretical framework to discover through. One of the
                DiscoveryFramework defaults, or any framework name as a string.
            text_column: Column holding document text in tabular files. Without
                it, each tabular file is flattened into one document.
            id_column: Column holding the document ID in tabular files. Without
                it, the row index is used. Duplicate IDs get a numeric suffix.
            output_dir: Optional directory to write results into. Created if
                needed. Saves discoveries_<timestamp>.csv / .json, plus
                discovery_metadata, discovery_errors and discovery_context JSON
                with the same timestamp. Repeated saves create new files.
            save_format: ``"csv"``, ``"json"``, or ``"both"``.

        Returns:
        -------
            Tuple of (pooled discoveries DataFrame, per-document API metadata,
            error records). The DataFrame has one row per quote, with columns
            ``filename``, ``doc_id``, ``construct``, ``description``, ``quote``.
            df.attrs['total_documents'] records every successfully processed
            document, including empty discoveries; failed documents are excluded.
            This denominator is also saved in discovery_context_<timestamp>.json.
            Every output filename receives the same UTC timestamp for this save.

        Raises:
        ------
            DiscoveryError: If the path holds no supported files.

        """
        files = get_input_files(path)

        rows: list[dict] = []
        metadata: dict[str, APIMetadata] = {}
        errors: list[ErrorRecord] = []

        documents: list[tuple[Path, str, str]] = []
        for file_path in files:
            try:
                for doc_id, text in _documents_from_file(
                    file_path, text_column, id_column
                ):
                    documents.append((file_path, doc_id, text))
            except DiscoveryError as e:
                print(f"  Skipped {file_path.name}: {e}")
                errors.append(
                    ErrorRecord(
                        document_id=file_path.stem,
                        document_path=str(file_path),
                        error_message=str(e),
                        model_used=self.config.model_name,
                        timestamp=datetime.now().isoformat(),
                    )
                )

        total = len(documents)
        print(f"Discovering across {total} document(s) from {len(files)} file(s).\n")

        for position, (file_path, doc_id, text) in enumerate(documents, start=1):
            key = f"{file_path.name}::{doc_id}"
            print(f"Discovering [{position}/{total}]: {key}")
            try:
                found, meta = self._discover_document(text, framework)
                metadata[key] = meta
                quote_count = 0
                for entry in found:
                    # A construct with no quotes still belongs in the table, so
                    # it gets one row with an empty quote rather than vanishing.
                    for quote in entry["quotes"] or [""]:
                        rows.append(
                            {
                                "filename": file_path.name,
                                "doc_id": doc_id,
                                "construct": entry["construct"],
                                "description": entry["description"],
                                "quote": quote,
                            }
                        )
                        quote_count += 1
                print(f"  OK: {len(found)} construct(s), {quote_count} quote(s)")
            except (PromptingError, DiscoveryError) as e:
                print(f"  Failed: {e}")
                errors.append(
                    ErrorRecord(
                        document_id=doc_id,
                        document_path=str(file_path),
                        error_message=str(e),
                        model_used=self.config.model_name,
                        timestamp=datetime.now().isoformat(),
                    )
                )

        df = pd.DataFrame(rows, columns=DISCOVERY_COLUMNS)
        df.attrs["total_documents"] = len(metadata)
        df.attrs["successful_documents"] = list(metadata)
        print(
            f"\nDiscovery complete: {len(metadata)}/{total} document(s), "
            f"{df['construct'].nunique()} unique construct name(s), "
            f"{len(errors)} error(s)."
        )

        if output_dir is not None:
            _save_discoveries(df, metadata, errors, output_dir, save_format)
        return df, metadata, errors

    def _discover_document(
        self, text: str, framework: str
    ) -> tuple[list[dict], APIMetadata]:
        """Run discovery on one document, retrying on parse failure."""
        prompt = self.discovery_prompt.format(framework=framework, document=text)
        max_attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            response_text, meta = call_llm_api(prompt, self.config)
            meta.num_retries = attempt
            try:
                return _parse_discovery_response(response_text), meta
            except DiscoveryError as e:
                last_error = e
        raise DiscoveryError(
            f"Failed after {max_attempts} attempt(s). Last error: {last_error}"
        )


def _save_discoveries(
    df: pd.DataFrame,
    metadata: dict[str, APIMetadata],
    errors: list[ErrorRecord],
    output_dir: Path | str,
    save_format: str,
) -> Path:
    """Write the discovery table, metadata, and errors to a directory."""
    out_dir = _prepare_output_dir(output_dir)
    stamp = _timestamp()
    _write_table(df, out_dir / f"discoveries_{stamp}", save_format)
    _save_json(out_dir / f"discovery_context_{stamp}.json", df.attrs)
    (out_dir / f"discovery_metadata_{stamp}.json").write_text(
        json.dumps(
            {key: meta.model_dump() for key, meta in metadata.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / f"discovery_errors_{stamp}.json").write_text(
        json.dumps([e.model_dump() for e in errors], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved discovery output to {out_dir}")
    return out_dir


def _prepare_output_dir(output_dir: Path | str) -> Path:
    """Create an output directory if it does not already exist."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _timestamp() -> str:
    """UTC timestamp with microseconds; shared by files in one save operation."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _frequency(table: pd.DataFrame, total_documents: int | None) -> pd.DataFrame:
    """Attach percentages without inferring a denominator from positive rows."""
    result = table.copy(deep=True)
    if total_documents is not None:
        if (
            isinstance(total_documents, bool)
            or not isinstance(total_documents, Integral)
            or total_documents < 0
        ):
            raise ValueError("total_documents must be a nonnegative integer or None.")
        total_documents = int(total_documents)
        if not result.empty and result.document_count.max() > total_documents:
            raise ValueError(
                "total_documents is smaller than a construct's document_count."
            )
    result["total_documents"] = total_documents
    result["document_frequency_percent"] = (
        result["document_count"] / total_documents * 100
        if total_documents
        else float("nan")
    )
    result.attrs["total_documents"] = total_documents
    return result


def _write_table(df: pd.DataFrame, stem: Path, save_format: str) -> None:
    """Write a DataFrame as CSV, JSON, or both.

    Raises:
    ------
        ValueError: If save_format is not csv, json, or both.

    """
    if save_format not in {"csv", "json", "both"}:
        raise ValueError(
            f"save_format must be 'csv', 'json', or 'both', got {save_format!r}."
        )
    if save_format in {"csv", "both"}:
        df.to_csv(stem.with_suffix(".csv"), index=False)
    if save_format in {"json", "both"}:
        df.to_json(stem.with_suffix(".json"), orient="records", indent=2)


# ---------------------------------------------------------------------------
# Step 2: merging
# ---------------------------------------------------------------------------


def _validate_fuzzy_threshold(threshold: float | None) -> None:
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.8 <= threshold <= 1
    ):
        raise ValueError("fuzzy_threshold must be between 0.8 and 1, or None.")


def _match_name(key: str, candidates, threshold: float | None) -> str | None:
    """Resolve exact names first, then an unambiguous close spelling.

    Short names and names with different word counts are never fuzzy matched.
    A runner-up within 0.03 makes a fuzzy match ambiguous, so it is left alone.
    Similarity is spelling-based, not evidence of semantic equivalence.
    """
    if key in candidates:
        return key
    if threshold is None or len(key) < 6:
        return None
    scores = sorted(
        (
            (SequenceMatcher(None, key, candidate, autojunk=False).ratio(), candidate)
            for candidate in candidates
            if len(candidate) >= 6 and len(candidate.split()) == len(key.split())
        ),
        reverse=True,
    )
    if not scores or scores[0][0] < threshold:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.03:
        return None
    return scores[0][1]


def collapse_variants(
    df: pd.DataFrame, fuzzy_threshold: float | None = 0.92
) -> pd.DataFrame:
    """Collapse spelling variants of the same construct name.

    Groups the discovery table by ``normalize_name``, so names differing only in
    case, separators, or punctuation become one constituent. Each group keeps
    a lowercase normalized representative as its display name. Close spelling
    variants are combined only when the fuzzy match is unambiguous. Counts are
    recalculated over the combined source rows so documents are not double-counted.

    Args:
    ----
        df: Discovery DataFrame from ``discover``.
        fuzzy_threshold: Minimum spelling similarity for an unambiguous fuzzy
            match. Default 0.92; None disables fuzzy matching while retaining
            case, whitespace, and separator normalization.

    Returns:
    -------
        One row per distinct construct, with columns ``construct`` (display
        name, lowercase), ``description`` (the longest description seen, as the most
        informative), ``count`` (documents it was found in), ``instance_count``
        (nonempty quote occurrences before deduplication), ``quotes`` (every
        quote collected), ``variants`` (the original spellings collapsed), and
        ``norm_key`` (the normalized key they collapsed to).

    """
    _validate_fuzzy_threshold(fuzzy_threshold)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "construct",
                "description",
                "count",
                "instance_count",
                "quotes",
                "variants",
                "norm_key",
            ]
        )

    working = df.copy()
    working["norm_key"] = working["construct"].map(normalize_name)
    working = working[working["norm_key"] != ""]

    # Prefer the most frequently observed normalized spelling; alphabetic ties
    # make the choice independent of input row order. Compare with retained
    # representatives only, avoiding chains of progressively looser matches.
    frequencies = working["norm_key"].value_counts()
    ordered = sorted(frequencies.index, key=lambda key: (-frequencies[key], key))
    representatives: list[str] = []
    aliases: dict[str, str] = {}
    for key in ordered:
        match = _match_name(key, representatives, fuzzy_threshold)
        aliases[key] = match or key
        if match is None:
            representatives.append(key)
    working["norm_key"] = working["norm_key"].map(aliases)

    rows = []
    for key, group in working.groupby("norm_key", sort=True):
        variants = sorted(set(group["construct"]))
        descriptions = [d for d in group["description"] if str(d).strip()]
        quotes = [q for q in group["quote"] if pd.notna(q) and str(q).strip()]
        rows.append(
            {
                "construct": key,
                "description": max(descriptions, key=len) if descriptions else "",
                # Documents, not quotes: one document may evidence a construct
                # with several quotes, which is not the same as finding it twice.
                "count": group[["filename", "doc_id"]].drop_duplicates().shape[0],
                "instance_count": len(quotes),
                "quotes": list(dict.fromkeys(quotes)),
                "variants": variants,
                "norm_key": key,
            }
        )

    collapsed = pd.DataFrame(
        rows,
        columns=[
            "construct",
            "description",
            "count",
            "instance_count",
            "quotes",
            "variants",
            "norm_key",
        ],
    )
    collapsed = collapsed.sort_values("count", ascending=False).reset_index(drop=True)
    collapsed.attrs["name_matches"] = [
        {"original_name": name, "construct_found": aliases[normalize_name(name)]}
        for name in sorted(set(working["construct"]))
    ]
    return collapsed


def _estimate_prompt_size(collapsed: pd.DataFrame, framework: str) -> int:
    """Rough character count of the merge prompt for a set of constructs."""
    block = "\n".join(
        f'- "{row.construct}": {row.description}' for row in collapsed.itertuples()
    )
    return len(MERGE_PROMPT) + len(block) + len(framework)


def _format_construct_block(
    collapsed: pd.DataFrame,
    rng: random.Random,
) -> str:
    """Format constructs for the merge prompt in a randomized order.

    The order constructs appear in biases how a model groups them, so the list
    is shuffled with a seeded RNG: unbiased, but reproducible across runs.
    """
    lines = [
        (
            f'- "{row.construct}": {row.description}'
            if str(row.description).strip()
            else f'- "{row.construct}"'
        )
        for row in collapsed.itertuples()
    ]
    rng.shuffle(lines)
    return "\n".join(lines)


def _parse_merge_response(
    response_text: str,
    expected: dict[str, str],
    fuzzy_threshold: float | None = 0.92,
) -> tuple[dict, list[str], dict]:
    """Parse a merge response into canonical groups.

    Args:
    ----
        response_text: Raw text returned by the merge LLM.
        expected: Mapping of normalized key to display name for every construct
            sent to the model.

    Returns:
    -------
        Tuple of (groups, unplaced display names, diagnostics). Groups map a
        canonical name to its definition and constituent list.

    Raises:
    ------
        DiscoveryError: If the response is not valid JSON or has no ``merged``
            list. Bookkeeping problems are reported, not raised.

    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise DiscoveryError(f"Response is not valid JSON: {e}") from e

    raw_groups = data.get("merged")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise DiscoveryError('Response missing a "merged" list.')

    groups: dict = {}
    seen_keys: set[str] = set()
    unknown: list[str] = []
    duplicated: list[str] = []
    bad_scores: list[str] = []
    fuzzy_matches: list[dict] = []

    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        name = str(group.get("name", "")).strip()
        if not name:
            continue

        constituents = group.get("constituents")
        cleaned = []
        if isinstance(constituents, list):
            for constituent in constituents:
                if not isinstance(constituent, dict):
                    continue
                raw_name = str(constituent.get("name", "")).strip()
                key = normalize_name(raw_name)
                if not key:
                    continue
                resolved = _match_name(key, expected, fuzzy_threshold)
                if resolved is not None and resolved != key:
                    fuzzy_matches.append(
                        {
                            "original_name": raw_name,
                            "construct_found": expected[resolved],
                        }
                    )
                    key = resolved
                if key in seen_keys:
                    duplicated.append(raw_name)
                    continue
                if key not in expected:
                    unknown.append(raw_name)
                    continue
                score = constituent.get("prototypicality")
                if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                    bad_scores.append(raw_name)
                    score = 0.0
                seen_keys.add(key)
                cleaned.append({"key": key, "prototypicality": round(float(score), 3)})

        # A group whose own name is one of the discovered constructs and which
        # has no constituents is the model declaring it stands alone.
        own_key = _match_name(normalize_name(name), expected, fuzzy_threshold)
        if not cleaned and own_key in expected and own_key not in seen_keys:
            seen_keys.add(own_key)
            name = expected[own_key]
            groups[name] = {
                "definition": str(group.get("definition", "")).strip(),
                "constituents": [],
                "standalone_key": own_key,
            }
            continue

        if not cleaned:
            continue
        cleaned.sort(key=lambda c: c["prototypicality"], reverse=True)
        groups[name] = {
            "definition": str(group.get("definition", "")).strip(),
            "constituents": cleaned,
            "standalone_key": None,
        }

    unplaced = [expected[key] for key in expected if key not in seen_keys]
    diagnostics = {
        "unknown": sorted(set(unknown)),
        "duplicated": sorted(set(duplicated)),
        "bad_scores": sorted(set(bad_scores)),
        "fuzzy_matches": fuzzy_matches,
    }
    return groups, unplaced, diagnostics


def _merge_batch(
    collapsed: pd.DataFrame,
    config: AnalyzerConfig,
    framework: str,
    merge_prompt: str,
    existing: dict | None,
    rng: random.Random,
    fuzzy_threshold: float | None = 0.92,
) -> tuple[dict, list[str], dict]:
    """Run one merge call over a set of collapsed constructs."""
    expected = {row.norm_key: row.construct for row in collapsed.itertuples()}
    existing_note = (
        EXISTING_NOTE.format(existing=json.dumps(existing, indent=2))
        if existing
        else ""
    )
    prompt = merge_prompt.format(
        framework=framework,
        constructs=_format_construct_block(collapsed, rng),
        existing_note=existing_note,
    )

    max_attempts = config.max_retries + 1
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        response_text, _meta = call_llm_api(prompt, config)
        try:
            return _parse_merge_response(response_text, expected, fuzzy_threshold)
        except DiscoveryError as e:
            last_error = e
    raise DiscoveryError(
        f"Merge failed after {max_attempts} attempt(s). Last error: {last_error}"
    )


def merge_constructs(
    df: pd.DataFrame,
    config: AnalyzerConfig,
    framework: DiscoveryFramework | str,
    merge_prompt: str | None = None,
    existing: dict | None = None,
    batch: bool = False,
    batch_size: int = 150,
    max_prompt_chars: int = 60000,
    seed: int = 0,
    output_dir: Path | str | None = None,
    save_format: str = "csv",
    fuzzy_threshold: float | None = 0.92,
    total_documents: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Consolidate discovered constructs into canonical constructs.

    Spelling variants are collapsed in code first (see ``normalize_name``), then
    an LLM groups what remains on meaning, synthesizing a canonical name and
    definition per group and scoring each constituent's prototypicality.

    Every discovered construct ends up in the result with one of three statuses:

    - ``merged`` — grouped with others; has a constituent name and a
      prototypicality score.
    - ``standalone`` — the model judged it canonical on its own.
    - ``unplaced`` — the model omitted it from its response. It is rescued into
      its own canonical entry rather than being silently dropped, and is also
      returned in the diagnostics.

    Args:
    ----
        df: Discovery DataFrame from ``discover``.
        config: AnalyzerConfig providing the model and API access.
        framework: The framework the discoveries were made under.
        merge_prompt: Optional template overriding MERGE_PROMPT. Must contain
            ``{framework}``, ``{constructs}``, and ``{existing_note}`` fields.
        existing: Optional previous merge result to fold these discoveries into.
        batch: If True, split the constructs into batches of ``batch_size`` and
            merge each in turn, folding every batch into the result so far. Off
            by default because a single pass sees every construct at once and
            groups better.
        batch_size: Constructs per batch when ``batch`` is True.
        max_prompt_chars: Refuse to send a single prompt longer than this.
            Raises with instructions rather than sending a request likely to be
            truncated.
        seed: Seed for the prompt-order shuffle, so runs are reproducible.
        output_dir: Optional directory to save the merged table, detailed
            merged_constructs_dict.json, simple merged_constructs_groups.json
            mapping merged names to constituent lists, and
            constituent_name_matches.json. JSON dictionaries are saved even
            when save_format="csv".
        save_format: ``"csv"``, ``"json"``, or ``"both"``.
        fuzzy_threshold: Spelling similarity threshold, default 0.92. Set None
            to disable fuzzy matching. Constituent names are always normalized
            to lowercase. Name-match records are attached to table.attrs and
            saved separately when output_dir is provided.
        total_documents: Number of successfully processed documents, INCLUDING
            those with no discoveries, excluding failures. Defaults to the value
            in df.attrs from discover(). For a reloaded CSV supply this explicitly
            from its saved discovery_context JSON; never use positive-row counts.
            If unavailable, frequency is missing rather than guessed. Percent is
            100 * document_count / total_documents, per construct_found row.

    Returns:
    -------
        Tuple of (merged dict, merged DataFrame). The dict maps each canonical
        name to its definition and constituents; the DataFrame has one row per
        discovered construct (or standalone/unplaced construct), with columns
        ``merged_construct``, ``definition``, ``status``, ``construct_found``,
        ``document_count``, ``instance_count``, ``prototypicality``, and ``quotes``.
        Counts describe the construct_found row, not the whole merged group.
        Documents are distinct (filename, doc_id) pairs; instances are nonempty
        quote occurrences before deduplication. A finding without a quote counts
        as a document detection but contributes zero instances. For standalone
        and unplaced entries, construct_found repeats merged_construct.

    Raises:
    ------
        DiscoveryError: If the prompt would exceed ``max_prompt_chars`` and
            ``batch`` is False, or if the merge response is unusable.

    """
    merge_prompt = merge_prompt or MERGE_PROMPT
    rng = random.Random(seed)
    collapsed = collapse_variants(df, fuzzy_threshold=fuzzy_threshold)
    total_documents = (
        total_documents
        if total_documents is not None
        else df.attrs.get("total_documents")
    )
    _frequency(collapsed.rename(columns={"count": "document_count"}), total_documents)
    # Keep per-constituent document memberships for accurate group counts after edits.
    memberships = {}
    for row in collapsed.itertuples():
        source = df[df["construct"].isin(row.variants)]
        memberships[row.construct] = (
            source[["filename", "doc_id"]].drop_duplicates().astype(str).values.tolist()
        )
    if (
        total_documents is not None
        and len({tuple(doc) for docs in memberships.values() for doc in docs})
        > total_documents
    ):
        raise ValueError(
            "total_documents is smaller than the discovered document roster."
        )

    if collapsed.empty:
        table = _frequency(pd.DataFrame(columns=MERGED_COLUMNS), total_documents)
        table.attrs["name_matches"] = []
        if output_dir is not None:
            _save_merge_outputs({}, table, [], output_dir, save_format)
        return {}, table

    collapsed_count = len(collapsed)
    raw_count = df["construct"].nunique() if not df.empty else 0
    print(
        f"Collapsed {raw_count} discovered name(s) into {collapsed_count} "
        f"distinct construct(s)."
    )

    if not batch:
        size = _estimate_prompt_size(collapsed, str(framework))
        if size > max_prompt_chars:
            raise DiscoveryError(
                f"The merge prompt would be about {size:,} characters for "
                f"{collapsed_count} constructs, over the {max_prompt_chars:,} "
                f"limit. A prompt this long risks a truncated response, which "
                f"shows up as constructs silently going missing.\n\n"
                f"Options:\n"
                f"  - Pass batch=True to merge in batches of batch_size "
                f"(default {batch_size}), folding each batch into the last.\n"
                f"  - Raise max_prompt_chars if you know your model's context "
                f"window can take it.\n"
                f"  - Discover over fewer documents, or a narrower framework."
            )
        batches = [collapsed]
    else:
        batches = [
            collapsed.iloc[i : i + batch_size].reset_index(drop=True)
            for i in range(0, collapsed_count, batch_size)
        ]
        print(f"Merging in {len(batches)} batch(es) of up to {batch_size}.")

    lookup = {row.norm_key: row for row in collapsed.itertuples()}
    groups: dict = {}
    all_unplaced: list[str] = []
    carried = existing
    name_matches = list(collapsed.attrs.get("name_matches", []))

    for index, chunk in enumerate(batches, start=1):
        if len(batches) > 1:
            print(f"  Batch {index}/{len(batches)} ({len(chunk)} constructs)")
        chunk_groups, unplaced, diagnostics = _merge_batch(
            chunk,
            config,
            str(framework),
            merge_prompt,
            carried,
            rng,
            fuzzy_threshold,
        )
        groups.update(chunk_groups)
        all_unplaced.extend(unplaced)
        _report_diagnostics(diagnostics, unplaced)
        name_matches.extend(diagnostics.get("fuzzy_matches", []))
        carried = _groups_to_existing(groups, lookup)

    merged, table = _assemble_merged(groups, all_unplaced, lookup)
    table = _frequency(table, total_documents)
    table.attrs["name_matches"] = name_matches
    table.attrs["document_memberships"] = memberships
    merged = _table_to_merged(table)

    if output_dir is not None:
        _save_merge_outputs(merged, table, name_matches, output_dir, save_format)

    return merged, table


def _save_merge_outputs(
    merged: dict,
    table: pd.DataFrame,
    name_matches: list[dict],
    output_dir: Path | str,
    save_format: str,
) -> None:
    """Save the review table, detailed result, simple groups, and name matches."""
    out_dir = _prepare_output_dir(output_dir)
    stamp = _timestamp()
    _write_table(table, out_dir / f"merged_constructs_{stamp}", save_format)
    (out_dir / f"merged_constructs_dict_{stamp}.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    grouping = {
        name: list(dict.fromkeys(group["construct_found"]))
        for name, group in table.groupby("merged_construct", sort=False)
    }
    (out_dir / f"merged_constructs_groups_{stamp}.json").write_text(
        json.dumps(grouping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"constituent_name_matches_{stamp}.json").write_text(
        json.dumps(name_matches, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _save_json(out_dir / f"merge_context_{stamp}.json", table.attrs)
    print(f"Saved merge output to {out_dir}")


def _report_diagnostics(diagnostics: dict, unplaced: list[str]) -> None:
    """Print bookkeeping problems from one merge response."""
    for match in diagnostics.get("fuzzy_matches", []):
        print(
            f"  Fuzzy spelling match: {match['original_name']!r} -> "
            f"{match['construct_found']!r}"
        )
    if unplaced:
        print(
            f"  Note: {len(unplaced)} construct(s) were not placed by the "
            f"merge and are kept as their own canonical constructs "
            f"(status='unplaced'): {unplaced[:10]}"
            + ("..." if len(unplaced) > 10 else "")
        )
    if diagnostics["unknown"]:
        print(
            f"  Warning: {len(diagnostics['unknown'])} name(s) in the response "
            f"matched no discovered construct and were dropped: "
            f"{diagnostics['unknown'][:10]}"
        )
    if diagnostics["duplicated"]:
        print(
            f"  Warning: {len(diagnostics['duplicated'])} name(s) appeared in "
            f"more than one group; first placement kept: "
            f"{diagnostics['duplicated'][:10]}"
        )
    if diagnostics["bad_scores"]:
        print(
            f"  Warning: invalid prototypicality for "
            f"{diagnostics['bad_scores'][:10]}; set to 0.0."
        )


def _groups_to_existing(groups: dict, lookup: dict) -> dict:
    """Render groups so far in the shape the merge prompt expects as context."""
    return {
        name: {
            "definition": data["definition"],
            "constituents": [
                {
                    "name": lookup[c["key"]].construct,
                    "prototypicality": c["prototypicality"],
                }
                for c in data["constituents"]
                if c["key"] in lookup
            ],
        }
        for name, data in groups.items()
    }


def _assemble_merged(
    groups: dict,
    unplaced: list[str],
    lookup: dict,
) -> tuple[dict, pd.DataFrame]:
    """Build the merged dict and flat table from parsed groups."""
    by_display = {row.construct: key for key, row in lookup.items()}
    merged: dict = {}
    rows: list[dict] = []

    for name, data in groups.items():
        if data.get("standalone_key"):
            row = lookup[data["standalone_key"]]
            definition = data["definition"] or row.description
            merged[name] = {
                "definition": definition,
                "status": "standalone",
                "constituents": [],
                "count": int(row.count),
                "quotes": list(row.quotes),
            }
            rows.append(
                {
                    "merged_construct": name,
                    "definition": definition,
                    "status": "standalone",
                    "construct_found": name,
                    "document_count": int(row.count),
                    "instance_count": int(row.instance_count),
                    "prototypicality": None,
                    "quotes": list(row.quotes),
                }
            )
            continue

        constituents = []
        for constituent in data["constituents"]:
            row = lookup[constituent["key"]]
            constituents.append(
                {
                    "name": row.construct,
                    "prototypicality": constituent["prototypicality"],
                    "count": int(row.count),
                    "quotes": list(row.quotes),
                }
            )
            rows.append(
                {
                    "merged_construct": name,
                    "definition": data["definition"],
                    "status": "merged",
                    "construct_found": row.construct,
                    "document_count": int(row.count),
                    "instance_count": int(row.instance_count),
                    "prototypicality": constituent["prototypicality"],
                    "quotes": list(row.quotes),
                }
            )
        merged[name] = {
            "definition": data["definition"],
            "status": "merged",
            "constituents": constituents,
            "count": sum(c["count"] for c in constituents),
            "quotes": [q for c in constituents for q in c["quotes"]],
        }

    for display in unplaced:
        row = lookup[by_display[display]]
        merged[display] = {
            "definition": row.description,
            "status": "unplaced",
            "constituents": [],
            "count": int(row.count),
            "quotes": list(row.quotes),
        }
        rows.append(
            {
                "merged_construct": display,
                "definition": row.description,
                "status": "unplaced",
                "construct_found": display,
                "document_count": int(row.count),
                "instance_count": int(row.instance_count),
                "prototypicality": None,
                "quotes": list(row.quotes),
            }
        )

    table = pd.DataFrame(rows, columns=MERGED_COLUMNS)
    if not table.empty:
        table = table.sort_values(
            ["merged_construct", "prototypicality"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)
    return merged, table


def _table_total(table: pd.DataFrame, explicit: int | None = None) -> int | None:
    if explicit is not None:
        return explicit
    if table.attrs.get("total_documents") is not None:
        return table.attrs["total_documents"]
    if "total_documents" in table:
        values = table["total_documents"].dropna().unique()
        if len(values) > 1:
            raise ValueError("Conflicting total_documents values in the merge table.")
        if len(values) == 1 and float(values[0]).is_integer():
            return int(values[0])
    return None


def _table_to_merged(table: pd.DataFrame) -> dict:
    """Rebuild codebook-compatible details; never sum overlapping document counts.

    If document memberships were lost during CSV/JSON loading, a multi-member
    group's distinct-document count is unknown (None). Row-level counts remain
    valid. Restore attrs from merge_context JSON to retain membership provenance.
    """
    merged = {}
    memberships = table.attrs.get("document_memberships", {})
    for name, group in table.groupby("merged_construct", sort=False):
        rows = group.to_dict("records")
        quotes = list(dict.fromkeys(q for row in rows for q in row["quotes"]))
        members = group["construct_found"].tolist()
        if all(member in memberships for member in members):
            count = len(
                {tuple(doc) for member in members for doc in memberships[member]}
            )
        else:
            count = int(rows[0]["document_count"]) if len(rows) == 1 else None
        standalone = len(rows) == 1 and rows[0]["status"] in {"standalone", "unplaced"}
        merged[name] = {
            "definition": rows[0]["definition"],
            "status": rows[0]["status"] if standalone else "merged",
            "count": count,
            "instance_count": sum(int(row["instance_count"]) for row in rows),
            "quotes": quotes,
            "constituents": (
                []
                if standalone
                else [
                    {
                        "name": row["construct_found"],
                        "prototypicality": (
                            float(row["prototypicality"])
                            if pd.notna(row["prototypicality"])
                            else None
                        ),
                        "count": int(row["document_count"]),
                        "instance_count": int(row["instance_count"]),
                        "quotes": list(row["quotes"]),
                    }
                    for row in rows
                ]
            ),
        }
    return merged


def apply_merge_edits(
    table: pd.DataFrame,
    grouping: dict[str, list[str]],
    *,
    definitions: dict[str, str] | None = None,
    total_documents: int | None = None,
    output_dir: Path | str | None = None,
    save_format: str = "csv",
) -> tuple[dict, pd.DataFrame]:
    """Apply explicit grouping edits locally, without calling an LLM.

    Args:
        table: Raw merge table, not format_merged_constructs() output. Quotes
            must be lists. Restore table.attrs from saved merge_context JSON
            after reloading to retain edit history and document memberships.
        grouping: {merged_construct: [construct_found, ...]}. Names must refer
            to known constituents; each may appear only once. Omitted members
            are removed, empty groups disappear. Names are matched by normalized
            case/separators, not fuzzy matched. Output constituents are lowercase.
        definitions: Optional definitions keyed by the new group names. Otherwise
            preserve the definition of an existing named group or an unambiguous
            single source group. A wholly new mixed group gets an empty definition
            pending manual definition or reevaluate_merged_constructs().
        total_documents: Successful-document denominator, including zero-finding
            documents. Defaults to attrs or the table's total_documents column.
            Filtering/removal NEVER shrinks this denominator. Unknown stays missing.
        output_dir: If supplied, save timestamped table, detailed/simple JSON,
            name matches and context. No existing files are overwritten.
        save_format: csv, json, or both for the table; dictionary JSON always saved.

    Returns:
        (merged_dict, edited_table), without modifying the input. Counts/quotes
        are retained per constituent; frequencies are recalculated. Prototypicality
        is retained within its old group or a pure group rename, but set missing
        for moved members because the old score is no longer valid. Existing
        definitions may now be outdated. Reevaluation updates names/definitions,
        NOT prototypicality scores. Previous groups and removed members are
        retained in attrs and saved context for that later LLM call.
    """
    required = set(MERGED_COLUMNS) - {"document_frequency_percent", "total_documents"}
    if required - set(table):
        raise ValueError(
            f"Merge table missing columns: {sorted(required - set(table))}"
        )
    if not isinstance(grouping, dict):
        raise ValueError("grouping must be a dictionary of group names to lists.")
    source = table.copy(deep=True)
    keys = source["construct_found"].map(normalize_name)
    if keys.duplicated().any():
        raise ValueError(
            "Each construct_found must occur only once in the source table."
        )
    if not all(isinstance(q, list) for q in source["quotes"]):
        raise ValueError(
            "quotes must contain lists. Use the raw merge table or reload its JSON."
        )
    lookup = {
        normalize_name(row["construct_found"]): row for row in source.to_dict("records")
    }
    old_groups = {
        name: group for name, group in source.groupby("merged_construct", sort=False)
    }
    definitions = definitions or {}
    rows, seen, group_names = [], set(), set()
    for name, members in grouping.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(members, list)
        ):
            raise ValueError(
                "Each group needs a nonempty string name and a list of constituents."
            )
        name = name.strip()
        name_key = normalize_name(name)
        if not name_key or name_key in group_names:
            raise ValueError(f"Duplicate or unusable merged group name: {name!r}")
        group_names.add(name_key)
        member_keys = []
        for member in members:
            if not isinstance(member, str):
                raise ValueError("Constituent names must be strings.")
            key = normalize_name(member)
            if key not in lookup or key in seen:
                raise ValueError(f"Unknown or duplicate constituent: {member!r}")
            seen.add(key)
            member_keys.append(key)
        if not member_keys:
            continue
        origins = {lookup[key]["merged_construct"] for key in member_keys}
        origin = next(iter(origins)) if len(origins) == 1 else None
        origin_members = (
            set(old_groups[origin]["construct_found"].map(normalize_name))
            if origin is not None
            else set()
        )
        pure_rename = origin is not None and set(member_keys) == origin_members
        definition = definitions.get(name)
        if definition is None:
            definition = (
                old_groups[name].iloc[0]["definition"]
                if name in old_groups
                else (
                    old_groups[origin].iloc[0]["definition"]
                    if origin is not None
                    else ""
                )
            )
        if not isinstance(definition, str):
            raise ValueError("Definitions must be strings.")
        for key in member_keys:
            row = deepcopy(lookup[key])
            if row["merged_construct"] != name and not pure_rename:
                row["prototypicality"] = None
            row.update(
                merged_construct=name,
                definition=definition,
                construct_found=key,
                status="merged",
            )
            # A true standalone keeps its prior status when its name still agrees.
            if (
                len(member_keys) == 1
                and name == key
                and lookup[key]["status"] in {"standalone", "unplaced"}
            ):
                row["status"] = lookup[key]["status"]
            rows.append(row)
    result = pd.DataFrame(rows, columns=MERGED_COLUMNS)
    result.attrs = deepcopy(table.attrs)
    result = _frequency(result, _table_total(table, total_documents))
    result.attrs.setdefault("previous_merge", _table_to_merged(source))
    removed = json.loads(source.loc[~keys.isin(seen)].to_json(orient="records"))
    result.attrs["removed_constituents"] = (
        deepcopy(table.attrs.get("removed_constituents", [])) + removed
    )
    merged = _table_to_merged(result)
    if output_dir is not None:
        _save_merge_outputs(
            merged,
            result,
            result.attrs.get("name_matches", []),
            output_dir,
            save_format,
        )
    return merged, result


def filter_merged_constructs(
    table: pd.DataFrame,
    *,
    remove_constructs: list[str] | None = None,
    remove_merged_constructs: list[str] | None = None,
    min_document_count: int | None = None,
    min_instance_count: int | None = None,
    min_prototypicality: float | None = None,
    total_documents: int | None = None,
    output_dir: Path | str | None = None,
    save_format: str = "csv",
) -> tuple[dict, pd.DataFrame]:
    """Filter constituent rows locally and rebuild/save the merge result.

    Removal lists use case/separator normalization, not fuzzy matching; unknown
    names raise ValueError. Counts are inclusive minimums; all supplied criteria
    must pass. A missing prototypicality score is excluded if a minimum score is
    requested (including standalone/unplaced rows); otherwise it is retained.
    Counts and quotes are not sampled. Empty groups disappear. The successful
    document denominator NEVER decreases after filtering. Definitions and retained
    scores are preserved and may need reevaluation. History of removed rows and
    original groups is kept for reevaluate_merged_constructs(). All other behavior,
    outputs and returns follow apply_merge_edits(). No LLM calls are made.
    """
    for value in (min_document_count, min_instance_count):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError("Minimum counts must be nonnegative integers or None.")
    if min_prototypicality is not None and (
        isinstance(min_prototypicality, bool)
        or not isinstance(min_prototypicality, (int, float))
        or not 0 <= min_prototypicality <= 1
    ):
        raise ValueError("min_prototypicality must be between 0 and 1, or None.")
    remove = {normalize_name(name) for name in remove_constructs or []}
    remove_groups = {normalize_name(name) for name in remove_merged_constructs or []}
    if remove - set(table.construct_found.map(normalize_name)) or remove_groups - set(
        table.merged_construct.map(normalize_name)
    ):
        raise ValueError("A removal list contains unknown names.")
    keep = ~table.construct_found.map(normalize_name).isin(remove)
    keep &= ~table.merged_construct.map(normalize_name).isin(remove_groups)
    for column, cutoff in (
        ("document_count", min_document_count),
        ("instance_count", min_instance_count),
        ("prototypicality", min_prototypicality),
    ):
        if cutoff is not None:
            keep &= pd.to_numeric(table[column], errors="coerce").ge(cutoff)
    grouping = {
        name: group.construct_found.tolist()
        for name, group in table.loc[keep].groupby("merged_construct", sort=False)
    }
    return apply_merge_edits(
        table,
        grouping,
        total_documents=total_documents,
        output_dir=output_dir,
        save_format=save_format,
    )


REEVALUATE_PROMPT = """You are revising a qualitative codebook under {framework}.
Original merged constructs, definitions, and constituents:
{previous}

Removed constituents (do not restore them):
{removed}

Current groups after manual edits/filtering:
{current}

For EVERY current group, reconsider only its name and definition in light of the
remaining constituents and removed members. Keep suitable names and definitions.
Do not move, add, delete or restore any constituents, or combine/split groups.
Use full terms, not acronyms. Return exactly one entry per current group, with
unique nonempty new names, in this JSON object and no other text:
{{"reevaluated": [{{"old_name": "exact current group name", "name": "updated name",
"definition": "updated one-sentence definition"}}]}}
"""


def reevaluate_merged_constructs(
    table: pd.DataFrame,
    config: AnalyzerConfig,
    framework: DiscoveryFramework | str,
    *,
    previous_table: pd.DataFrame | None = None,
    reevaluation_prompt: str | None = None,
    max_prompt_chars: int = 60000,
    total_documents: int | None = None,
    output_dir: Path | str | None = None,
    save_format: str = "csv",
) -> tuple[dict, pd.DataFrame]:
    """Ask an LLM to revise names/definitions after editing or filtering.

    table is the raw current merge table. Previous grouping and removed members
    come from attrs saved by apply_merge_edits/filter_merged_constructs. Restore
    attrs from merge_context JSON after reloading, or supply previous_table to
    explicitly provide the pre-edit table. Without either, the current groups are
    used as context, with no presumed removed members. The complete original,
    removed and current data is placed in the prompt; no quotes are sampled.

    The optional prompt template uses {framework}, {previous}, {removed}, and
    {current}. It must request the REEVALUATE_PROMPT response schema. Names and
    definitions are the ONLY editable fields. Constituents, counts, quotes and
    the successful-document denominator are preserved. Missing prototypicality
    for moved members remains missing; this function does not rescore it.
    Invalid JSON, missing groups and duplicate new names trigger up to
    config.max_retries additional generation attempts. API failures follow the
    existing call_llm_api behavior. No incomplete response is applied; input
    data is never mutated. Empty input returns/saves an empty result without a
    call. If output_dir is supplied, all outputs have a shared new UTC timestamp.
    Returns (merged_dict, updated_table).

    max_prompt_chars bounds the full formatted prompt, including original/current
    groups and removed quotes. Oversized prompts raise before any API call; raise
    the bound only if the chosen model can accommodate the complete input.
    """
    # Validate/clone using a no-op local edit, then restore scores/status verbatim.
    grouping = {
        name: group.construct_found.tolist()
        for name, group in table.groupby("merged_construct", sort=False)
    }
    _, result = apply_merge_edits(table, grouping, total_documents=total_documents)
    original_rows = {
        normalize_name(row.construct_found): row for row in table.itertuples()
    }
    result["status"] = result.construct_found.map(
        lambda name: original_rows[normalize_name(name)].status
    )
    result["prototypicality"] = result.construct_found.map(
        lambda name: original_rows[normalize_name(name)].prototypicality
    )
    previous = (
        _table_to_merged(previous_table)
        if previous_table is not None
        else table.attrs.get("previous_merge", _table_to_merged(table))
    )
    removed = table.attrs.get("removed_constituents", [])
    if previous_table is not None:
        removed = json.loads(
            previous_table.loc[
                ~previous_table.construct_found.map(normalize_name).isin(
                    table.construct_found.map(normalize_name)
                )
            ].to_json(orient="records")
        )
    if not result.empty:
        prompt = (reevaluation_prompt or REEVALUATE_PROMPT).format(
            framework=framework,
            previous=json.dumps(previous, ensure_ascii=False),
            removed=json.dumps(removed, ensure_ascii=False),
            current=json.dumps(_table_to_merged(result), ensure_ascii=False),
        )
        if (
            isinstance(max_prompt_chars, bool)
            or not isinstance(max_prompt_chars, Integral)
            or max_prompt_chars <= 0
        ):
            raise ValueError("max_prompt_chars must be a positive integer.")
        if len(prompt) > max_prompt_chars:
            raise DiscoveryError(
                f"Reevaluation prompt is {len(prompt):,} characters, over max_prompt_chars={max_prompt_chars:,}. Raise the bound only if your model supports the full input."
            )
        last_error = None
        for attempt in range(config.max_retries + 1):
            response, _metadata = call_llm_api(prompt, config)
            try:
                data = json.loads(response)
                entries = data.get("reevaluated") if isinstance(data, dict) else None
                if not isinstance(entries, list):
                    raise ValueError("Response must contain a reevaluated list.")
                updates, names = {}, set()
                for entry in entries:
                    if not isinstance(entry, dict):
                        raise ValueError("Each reevaluated entry must be an object.")
                    old, name, definition = (
                        entry.get("old_name"),
                        entry.get("name"),
                        entry.get("definition"),
                    )
                    if not all(
                        isinstance(value, str) and value.strip()
                        for value in (old, name, definition)
                    ):
                        raise ValueError(
                            "Names and definitions must be nonempty strings."
                        )
                    name, definition = name.strip(), definition.strip()
                    if (
                        old in updates
                        or not normalize_name(name)
                        or normalize_name(name) in names
                    ):
                        raise ValueError("Duplicate old or new group name.")
                    updates[old] = (name, definition)
                    names.add(normalize_name(name))
                if set(updates) != set(grouping):
                    raise ValueError(
                        "Response must include every current group exactly once."
                    )
                break
            except (ValueError, TypeError) as error:
                last_error = error
        else:
            raise DiscoveryError(
                f"Reevaluation failed after {config.max_retries + 1} attempts: {last_error}"
            )
        result["definition"] = result.merged_construct.map(
            lambda name: updates[name][1]
        )
        result["merged_construct"] = result.merged_construct.map(
            lambda name: updates[name][0]
        )
        result.attrs["reevaluation"] = {
            old: {"name": new, "definition": definition}
            for old, (new, definition) in updates.items()
        }
    merged = _table_to_merged(result)
    if output_dir is not None:
        _save_merge_outputs(
            merged,
            result,
            result.attrs.get("name_matches", []),
            output_dir,
            save_format,
        )
    return merged, result


def format_merged_constructs(
    table: pd.DataFrame,
    n_quotes: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Format the merged table for review, with a sample of quotes per row.

    Args:
    ----
        table: Merged DataFrame from ``merge_constructs``.
        n_quotes: Maximum quotes to show per row.
        seed: Seed for quote sampling, so the display is reproducible.

    Returns:
    -------
        A display DataFrame whose ``quotes`` column holds up to ``n_quotes``
        quotes joined by ``";\\n"``. Adds document_frequency formatted as
        '25.0% (N=100)'. N is all successfully processed documents, including
        empty discoveries; unavailable percentages display as N/A. Raw numeric
        document_frequency_percent and total_documents columns remain unchanged.

    """
    if table.empty:
        return table

    rng = random.Random(seed)
    display = table.copy()
    display["document_frequency"] = [
        (
            f"{percent:.1f}% (N={int(total)})"
            if pd.notna(percent) and pd.notna(total)
            else "N/A"
        )
        for percent, total in zip(
            display["document_frequency_percent"], display["total_documents"]
        )
    ]
    display["quotes"] = [
        ";\n".join(
            quotes if len(quotes) <= n_quotes else rng.sample(list(quotes), n_quotes)
        )
        for quotes in display["quotes"]
    ]
    return display


# ---------------------------------------------------------------------------
# Step 3: codebook
# ---------------------------------------------------------------------------


def build_codebook(
    merged: dict,
    name: str = "",
    version: int = 1,
    n_examples: int = 5,
    seed: int = 0,
    output_path: Path | str | None = None,
) -> dict:
    """Turn a merge result into a codebook envelope.

    Each canonical construct becomes a codebook entry using the definition
    synthesized during merging, seeded with example quotes drawn from the
    discovery. Quotes are deduplicated and sampled uniformly from each merged
    construct's pooled quotes, not balanced across its constituents.

    Args:
    ----
        merged: Merge result from ``merge_constructs``.
        name: Optional name for the codebook metadata.
        version: Version number for the codebook metadata.
        n_examples: Maximum example quotes per construct. Fewer are used when
            fewer were discovered.
        seed: Seed for quote sampling, so builds are reproducible.
        output_path: Optional codebook JSON path. save_codebook adds a UTC
            timestamp to the filename and prints the actual saved path.

    Returns:
    -------
        A codebook envelope: ``{"metadata": {...}, "codebook": {...}}``.

    """
    rng = random.Random(seed)
    codebook: dict = {}

    for canonical, data in merged.items():
        quotes = list(dict.fromkeys(data.get("quotes", [])))
        examples = (
            quotes if len(quotes) <= n_examples else rng.sample(quotes, n_examples)
        )
        codebook[canonical] = {
            "definition": str(data.get("definition", "")).strip(),
            "examples": examples,
            "counter_examples": [],
        }

    envelope = {
        "metadata": {
            "name": name,
            "version": version,
            "citation": "",
            "built_from": [{"source": "discovery"}],
        },
        "codebook": codebook,
    }

    if output_path is not None:
        save_codebook(envelope, output_path)
    return envelope


def codebook_to_frame(codebook: dict, as_text: bool = False) -> pd.DataFrame:
    """Render a codebook envelope as an editable DataFrame.

    Use this to review or revise definitions and examples in a notebook, then
    pass the result to ``frame_to_codebook`` to convert it back.

    By default the ``examples`` and ``counter_examples`` columns hold real
    Python lists, so editing means assigning a list. That avoids a delimiter
    entirely -- quotes routinely contain semicolons and newlines, and any
    separator chosen to join them will eventually split one in half.

    Args:
    ----
        codebook: A codebook envelope.
        as_text: If True, join the list columns with ``";\\n"`` for a more
            readable display. Editing text cells is then delimiter-sensitive,
            so prefer the default when you intend to make changes.

    Returns:
    -------
        DataFrame with ``construct``, ``definition``, ``examples``, and
        ``counter_examples`` columns.

    """
    entries = codebook.get("codebook", codebook)
    rows = []
    for construct, entry in entries.items():
        examples = list(entry.get("examples", []))
        counter_examples = list(entry.get("counter_examples", []))
        rows.append(
            {
                "construct": construct,
                "definition": entry.get("definition", ""),
                "examples": ";\n".join(examples) if as_text else examples,
                "counter_examples": (
                    ";\n".join(counter_examples) if as_text else counter_examples
                ),
            }
        )
    return pd.DataFrame(
        rows, columns=["construct", "definition", "examples", "counter_examples"]
    )


def frame_to_codebook(
    df: pd.DataFrame,
    metadata: dict | None = None,
    output_path: Path | str | None = None,
) -> dict:
    """Convert an edited codebook DataFrame back into an envelope.

    Args:
    ----
        df: DataFrame from ``codebook_to_frame``, possibly edited. The
            ``examples`` and ``counter_examples`` cells may hold either lists or
            ``";"``-joined strings, so frames built with ``as_text=True`` still
            convert back.
        metadata: Optional metadata block. Defaults are used if omitted.
        output_path: Optional codebook JSON path. save_codebook adds a UTC
            timestamp to the filename and prints the actual saved path.

    Returns:
    -------
        A codebook envelope.

    Raises:
    ------
        ValueError: If a required column is missing.

    """
    required = {"construct", "definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

    def _as_list(value: object) -> list[str]:
        """Accept a list of quotes, or a ';'-joined string, or nothing."""
        if value is None or isinstance(value, float) and pd.isna(value):
            return []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in str(value).split(";") if part.strip()]

    codebook = {
        str(row.construct).strip(): {
            "definition": str(row.definition).strip(),
            "examples": _as_list(getattr(row, "examples", None)),
            "counter_examples": _as_list(getattr(row, "counter_examples", None)),
        }
        for row in df.itertuples()
        if str(row.construct).strip()
    }

    envelope = {
        "metadata": metadata
        or {"name": "", "version": 1, "citation": "", "built_from": []},
        "codebook": codebook,
    }
    if output_path is not None:
        save_codebook(envelope, output_path)
    return envelope


def save_codebook(codebook: dict, output_path: Path | str) -> Path:
    """Write a codebook envelope to JSON.

    Args:
    ----
        codebook: The codebook envelope to write.
        output_path: Requested destination. A UTC microsecond timestamp is added
            to its stem; e.g. codebook.json becomes codebook_<timestamp>.json.

    Returns:
    -------
        The actual timestamped path written to. Existing files are not replaced.

    """
    path = Path(output_path)
    path = path.with_name(f"{path.stem}_{_timestamp()}{path.suffix or '.json'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(codebook, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved codebook to {path}")
    return path
