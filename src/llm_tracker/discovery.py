"""Discovery mode: find constructs in documents through a theoretical framework.

Discovery is the inverse of coding. Instead of locating instances of known
codebook constructs, the LLM reads each document through a chosen theoretical
framework and reports which constructs it finds, with supporting quotes. A
separate merge step then consolidates the constructs discovered across
documents into canonical constructs, each with an LLM-synthesized name and
definition and its constituent constructs ranked by prototypicality.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from llm_tracker.config import AnalyzerConfig
from llm_tracker.file_handlers import (
    FileLoadError,
    get_document_files,
    load_document,
)
from llm_tracker.models import APIMetadata, ErrorRecord
from llm_tracker.prompting import PromptingError, call_llm_api

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

_DISCOVERY_PROMPT = """You are an expert qualitative researcher analyzing a \
document through the lens of a specific theoretical framework.

Framework: {framework}

Read the document below and identify the psychological or theoretical \
constructs from this framework that are present in the text. For each \
construct you find, provide:
- "construct": a short name for the construct
- "description": one sentence describing the construct as it appears here
- "quotes": a list of verbatim quotes from the document that evidence it

Only report constructs genuinely grounded in the text. If none are present, \
return an empty list.

Respond ONLY with a JSON object of the form:
{{"constructs": [{{"construct": "...", "description": "...", \
"quotes": ["..."]}}]}}

Document:
{document}
"""

_MERGE_PROMPT = """You are an expert qualitative researcher consolidating \
constructs discovered across many documents under the framework: {framework}.

Below is a list of discovered construct names (with brief descriptions). Merge \
constructs that are effectively the same into consolidated canonical \
constructs. For each canonical construct provide:
- "name": the canonical name (synthesize the best name; it may be one of the \
merged names)
- "definition": a one-sentence definition of the consolidated construct
- "constituents": every merged construct, each with a "prototypicality" score \
from 0 to 1 (1 = the first thing that comes to mind for this canonical \
construct; lower = related but less obvious)

Every input construct must appear in exactly one canonical construct's \
constituents. A canonical construct may have a single constituent if nothing \
else matches it.
{existing_note}
Respond ONLY with a JSON object of the form:
{{"merged": [{{"name": "...", "definition": "...", "constituents": \
[{{"name": "...", "prototypicality": 0.0}}]}}]}}

Discovered constructs:
{constructs}
"""

_EXISTING_NOTE = """
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


class LLMTrackerDiscoverer:
    """Discover constructs in documents through a theoretical framework."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        config: AnalyzerConfig | None = None,
    ) -> None:
        """Create a discoverer.

        Args:
        ----
            api_key: API key or path to an env file containing it.
            model_name: Model to use. Ignored if config is given.
            config: Optional AnalyzerConfig. When provided, other arguments are
                ignored.

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

    def discover(
        self,
        path: Path | str,
        framework: DiscoveryFramework | str,
        text_column: str | None = None,
        id_column: str | None = None,
        output_dir: str | None = None,
    ) -> tuple[dict[str, list[dict]], dict[str, APIMetadata], list[ErrorRecord]]:
        """Discover constructs in documents, auto-detecting the input type.

        Accepts either a single CSV file (one document per row) or a directory
        of ``.txt`` / ``.csv`` files (one document per file) -- the same inputs
        the coding pipeline accepts. The input type is detected from the path.

        Args:
        ----
            path: A CSV file, or a directory of supported document files.
            framework: Theoretical framework to discover through. One of the
                DiscoveryFramework defaults, or any framework name as a string.
            text_column: Column containing the document text. Required when
                ``path`` is a CSV; ignored for a directory.
            id_column: Optional CSV column to use as the document ID. If omitted,
                the row index (0..N-1) is used. Ignored for a directory (there
                the filename without extension is the document ID).
            output_dir: Optional base name for the output directory. When given,
                per-document discoveries are saved under
                ``<output_dir>_<timestamp>/discoveries/``.

        Returns:
        -------
            Tuple of (discoveries keyed by document ID, per-document API
            metadata, error records).

        Raises:
        ------
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is a CSV but no text_column is given.

        """
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")

        if resolved.is_dir():
            return self._discover_directory(resolved, framework, output_dir)

        if resolved.suffix.lower() == ".csv":
            if text_column is None:
                raise ValueError(
                    "text_column is required when discovering from a CSV file."
                )
            return self._discover_csv(
                resolved, framework, text_column, id_column, output_dir
            )

        raise ValueError(
            f"Unsupported input: {resolved}. Provide a CSV file or a directory "
            f"of .txt/.csv files."
        )

    def _discover_csv(
        self,
        csv_path: Path | str,
        framework: DiscoveryFramework | str,
        text_column: str,
        id_column: str | None = None,
        output_dir: str | None = None,
    ) -> tuple[dict[str, list[dict]], dict[str, APIMetadata], list[ErrorRecord]]:
        """Discover constructs in every document of a CSV.

        Args:
        ----
            csv_path: CSV with one document per row.
            framework: Theoretical framework to discover through. One of the
                DiscoveryFramework defaults, or any framework name as a string.
            text_column: Column containing the document text.
            id_column: Optional column to use as the document ID. If omitted,
                the row index (0..N-1) is used. Duplicate IDs get a numeric
                suffix.
            output_dir: Optional base name for the output directory. When
                given, per-document discoveries and metadata are saved under
                ``<output_dir>_<timestamp>/discoveries/``.

        Returns:
        -------
            Tuple of (discoveries keyed by document ID, per-document API
            metadata, error records). Each document's discoveries are a list of
            dicts with ``construct``, ``description``, and ``quotes``.

        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        df = pd.read_csv(csv_file)
        required = [text_column] + ([id_column] if id_column else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        out_path = self._make_output_dir(output_dir)

        discoveries: dict[str, list[dict]] = {}
        metadata: dict[str, APIMetadata] = {}
        errors: list[ErrorRecord] = []
        id_counts: dict[str, int] = {}
        total = len(df)

        for position, (index, row) in enumerate(df.iterrows(), start=1):
            text = str(row[text_column])
            base_id = str(row[id_column]).strip() if id_column else str(index)
            id_counts[base_id] = id_counts.get(base_id, 0) + 1
            doc_id = (
                base_id
                if id_counts[base_id] == 1
                else f"{base_id}_{id_counts[base_id]}"
            )

            print(f"Discovering [{position}/{total}]: {doc_id}")
            try:
                found, meta = self._discover_document(text, framework)
                self._record_success(
                    doc_id, found, meta, discoveries, metadata, out_path
                )
            except (PromptingError, DiscoveryError) as e:
                print(f"  Failed: {e}")
                errors.append(
                    ErrorRecord(
                        document_id=doc_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        timestamp=datetime.now().isoformat(),
                    )
                )

        print(
            f"\nDiscovery complete: {len(discoveries)}/{total} documents, "
            f"{len(errors)} error(s)."
        )
        return discoveries, metadata, errors

    def _discover_directory(
        self,
        input_dir: Path | str,
        framework: DiscoveryFramework | str,
        output_dir: str | None = None,
    ) -> tuple[dict[str, list[dict]], dict[str, APIMetadata], list[ErrorRecord]]:
        """Discover constructs in every supported document in a directory.

        Mirrors analyze_directory: each file becomes one document, and the
        filename without extension becomes the document ID.

        Args:
        ----
            input_dir: Directory containing supported document files.
            framework: Theoretical framework to discover through. One of the
                DiscoveryFramework defaults, or any framework name as a string.
            output_dir: Optional base name for the output directory. When
                given, per-document discoveries are saved under
                ``<output_dir>_<timestamp>/discoveries/``.

        Returns:
        -------
            Tuple of (discoveries keyed by document ID, per-document API
            metadata, error records).

        """
        document_paths = get_document_files(Path(input_dir))

        out_path = self._make_output_dir(output_dir)
        discoveries: dict[str, list[dict]] = {}
        metadata: dict[str, APIMetadata] = {}
        errors: list[ErrorRecord] = []
        total = len(document_paths)

        for position, document_path in enumerate(document_paths, start=1):
            print(f"Discovering [{position}/{total}]: {document_path.name}")
            try:
                text, doc_id = load_document(document_path)
                found, meta = self._discover_document(text, framework)
                self._record_success(
                    doc_id, found, meta, discoveries, metadata, out_path
                )
            except (FileLoadError, PromptingError, DiscoveryError) as e:
                print(f"  Failed: {e}")
                errors.append(
                    ErrorRecord(
                        document_id=document_path.stem,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        timestamp=datetime.now().isoformat(),
                    )
                )

        print(
            f"\nDiscovery complete: {len(discoveries)}/{total} documents, "
            f"{len(errors)} error(s)."
        )
        return discoveries, metadata, errors

    def _make_output_dir(self, output_dir: str | None) -> Path | None:
        """Create the timestamped discoveries directory when requested."""
        if output_dir is None:
            return None
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = Path(f"{output_dir}_{timestamp}") / "discoveries"
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path

    def _record_success(
        self,
        doc_id: str,
        found: list[dict],
        meta: APIMetadata,
        discoveries: dict[str, list[dict]],
        metadata: dict[str, APIMetadata],
        out_path: Path | None,
    ) -> None:
        """Store one document's discoveries in memory and on disk."""
        discoveries[doc_id] = found
        metadata[doc_id] = meta
        print(f"  OK: Found {len(found)} construct(s)")
        if out_path is not None:
            (out_path / f"{doc_id}.json").write_text(
                json.dumps(
                    {"document_id": doc_id, "constructs": found},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def _discover_document(
        self, text: str, framework: str
    ) -> tuple[list[dict], APIMetadata]:
        """Run discovery on one document, with parse retries."""
        prompt = _DISCOVERY_PROMPT.format(framework=framework, document=text)
        max_attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            response_text, meta = call_llm_api(prompt, self.config)
            meta.num_retries = attempt
            try:
                constructs = _parse_discovery_response(response_text)
                return constructs, meta
            except DiscoveryError as e:
                last_error = e
        raise DiscoveryError(
            f"Failed after {max_attempts} attempt(s). Last error: {last_error}"
        )


def merge_constructs(
    discoveries: dict[str, list[dict]],
    config: AnalyzerConfig,
    framework: DiscoveryFramework | str,
    existing: dict | None = None,
    output_path: Path | str | None = None,
) -> dict:
    """Consolidate discovered constructs into canonical constructs.

    An LLM proposes the grouping, canonical names, one-sentence definitions,
    and per-constituent prototypicality scores; the code validates that every
    input construct is accounted for exactly once and that scores are in
    [0, 1], retrying on structural failures. Pass a previous result as
    ``existing`` to fold a new batch into an established consolidation instead
    of merging from scratch.

    Args:
    ----
        discoveries: Output of discover_csv (document ID -> discovered
            construct dicts).
        config: AnalyzerConfig providing the model and API access.
        framework: The framework the discoveries were made under.
        existing: Optional previous merge result to fold these discoveries
            into. Canonical names are kept stable where possible.
        output_path: Optional path to save the merged result JSON to.

    Returns:
    -------
        Dict of canonical construct name -> {"definition": str,
        "constituents": [{"name": str, "prototypicality": float}, ...]},
        constituents sorted by prototypicality descending.

    """
    items: dict[str, str] = {}
    for found in discoveries.values():
        for entry in found:
            name = str(entry.get("construct", "")).strip()
            if name and name not in items:
                items[name] = str(entry.get("description", "")).strip()
    if existing:
        for canonical, data in existing.items():
            for constituent in data.get("constituents", []):
                items.pop(constituent["name"], None)

    if not items and not existing:
        return {}
    if not items:
        return existing or {}

    constructs_block = "\n".join(
        f'- "{name}": {desc}' if desc else f'- "{name}"' for name, desc in items.items()
    )
    existing_note = (
        _EXISTING_NOTE.format(existing=json.dumps(existing, indent=2))
        if existing
        else ""
    )
    prompt = _MERGE_PROMPT.format(
        framework=framework,
        constructs=constructs_block,
        existing_note=existing_note,
    )

    max_attempts = config.max_retries + 1
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        response_text, _meta = call_llm_api(prompt, config)
        try:
            merged = _parse_merge_response(response_text, set(items.keys()), existing)
            if output_path is not None:
                Path(output_path).write_text(
                    json.dumps(merged, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            return merged
        except DiscoveryError as e:
            last_error = e
    raise DiscoveryError(
        f"Merge failed after {max_attempts} attempt(s). Last error: {last_error}"
    )


def format_merged_constructs(merged: dict) -> pd.DataFrame:
    """Format a merge result as a review table.

    One row per constituent, grouped by canonical construct and sorted by
    prototypicality descending within each group.
    """
    rows = []
    for canonical, data in merged.items():
        for constituent in data.get("constituents", []):
            rows.append(
                {
                    "canonical_construct": canonical,
                    "definition": data.get("definition", ""),
                    "constituent": constituent["name"],
                    "prototypicality": constituent["prototypicality"],
                }
            )
    df = pd.DataFrame(
        rows,
        columns=["canonical_construct", "definition", "constituent", "prototypicality"],
    )
    if df.empty:
        return df
    return df.sort_values(
        ["canonical_construct", "prototypicality"], ascending=[True, False]
    ).reset_index(drop=True)


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
                "quotes": [str(q) for q in quotes],
            }
        )
    return cleaned


def _parse_merge_response(
    response_text: str,
    expected: set[str],
    existing: dict | None,
) -> dict:
    """Parse a merge response, reporting bookkeeping problems without failing.

    Raises DiscoveryError (triggering a retry) only when the response is
    structurally unusable: not valid JSON, or missing the "merged" list.
    Bookkeeping violations -- constituents the model dropped, invented,
    duplicated, or scored invalidly -- are printed as diagnostics and handled
    best-effort, so a flawed merge is still returned for inspection instead of
    discarded.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise DiscoveryError(f"Response is not valid JSON: {e}") from e

    groups = data.get("merged")
    if not isinstance(groups, list) or not groups:
        raise DiscoveryError('Response missing a "merged" list.')

    carried: set[str] = set()
    if existing:
        for entry in existing.values():
            carried |= {c["name"] for c in entry.get("constituents", [])}

    merged: dict = {}
    seen: set[str] = set()
    unknown: list[str] = []
    duplicated: list[str] = []
    bad_scores: list[str] = []

    for group in groups:
        name = str(group.get("name", "")).strip()
        if not name:
            print(f"Warning: skipping canonical construct with no name: {group!r}")
            continue
        constituents = group.get("constituents")
        if not isinstance(constituents, list) or not constituents:
            print(f"Warning: '{name}' has no constituents; skipping it.")
            continue

        cleaned = []
        for constituent in constituents:
            cname = str(constituent.get("name", "")).strip()
            score = constituent.get("prototypicality")
            if not cname:
                print(f"Warning: unnamed constituent in '{name}'; skipped.")
                continue
            if cname in seen:
                duplicated.append(cname)
                continue
            if cname not in expected and cname not in carried:
                unknown.append(cname)
                continue
            if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                bad_scores.append(cname)
                score = 0.0
            seen.add(cname)
            cleaned.append({"name": cname, "prototypicality": round(float(score), 3)})

        if not cleaned:
            continue
        cleaned.sort(key=lambda c: c["prototypicality"], reverse=True)
        merged[name] = {
            "definition": str(group.get("definition", "")).strip(),
            "constituents": cleaned,
        }

    missing = sorted(expected - seen)
    if missing:
        print(
            f"Warning: {len(missing)} constituent(s) were not placed by the "
            f"merge and are NOT in the result: {missing}"
        )
    if unknown:
        print(
            f"Warning: {len(unknown)} constituent(s) in the response matched "
            f"no discovered construct (name-mangling?) and were dropped: "
            f"{sorted(set(unknown))}"
        )
    if duplicated:
        print(
            f"Warning: {len(duplicated)} constituent(s) appeared more than "
            f"once; first placement kept: {sorted(set(duplicated))}"
        )
    if bad_scores:
        print(
            f"Warning: invalid prototypicality for {sorted(set(bad_scores))}; "
            f"set to 0.0."
        )
    return merged


def discovery_to_codebook(
    merged: dict,
    name: str = "",
    version: int = 1,
) -> dict:
    """Convert a merged discovery result into a codebook envelope.

    Turns consolidated constructs into a codebook in the same format the coding
    pipeline consumes, so a discovery run can seed a coding run. Each canonical
    construct becomes a codebook entry using the definition synthesized during
    merging; the ``examples`` list is left empty (discovery does not carry
    quotes into the merged result -- add examples later, by hand or via the
    optimizer). Metadata records that the codebook came from discovery.

    Args:
    ----
        merged: A merge result from merge_constructs (canonical construct name
            -> {"definition": str, "constituents": [...]}).
        name: Optional name for the codebook metadata.
        version: Version number for the codebook metadata. Defaults to 1.

    Returns:
    -------
        A codebook envelope: {"metadata": {...}, "codebook": {construct: {
        "definition": str, "examples": []}, ...}}.

    """
    codebook: dict = {}
    for canonical, data in merged.items():
        codebook[canonical] = {
            "definition": str(data.get("definition", "")).strip(),
            "examples": [],
        }

    metadata = {
        "name": name,
        "version": version,
        "citation": "",
        "built_from": [{"source": "discovery"}],
    }
    return {"metadata": metadata, "codebook": codebook}
