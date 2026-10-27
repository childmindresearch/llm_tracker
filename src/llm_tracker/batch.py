"""Persisted OpenRouter batch coding; submission never retries automatically."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from llm_tracker.outputs import (
    RunOutput,
    redact,
    RunResults,
    atomic_json,
    config_fields,
    snapshots,
    finish_encoding,
    usage_totals,
    document_stem,
)

import httpx
import pandas as pd

from llm_tracker.file_handlers import (
    codebook_constructs,
    get_document_files,
    load_codebook,
    load_document,
    save_analysis_result,
    save_error_record,
    save_metadata,
)
from llm_tracker.models import (
    AnalysisResult,
    APIMetadata,
    BatchJob,
    ConstructInstance,
    ErrorRecord,
)
from llm_tracker.prompting import (
    PromptingError,
    construct_prompt,
    parse_llm_response,
    validate_llm_output,
)

if TYPE_CHECKING:
    from llm_tracker.config import AnalyzerConfig

TERMINAL = {"completed", "failed", "expired", "cancelled"}
LOCAL_TERMINAL = {"rejected", "not_submitted"}
UNKNOWN = {"submitting", "submission_unknown"}
BASE_URL = "https://openrouter.ai/api/v1"


class BatchError(RuntimeError):
    """Batch operation failed; output_path identifies saved recovery information."""

    def __init__(self, message: str, output_path: Path | None = None) -> None:
        super().__init__(message)
        self.output_path = output_path


class BatchNotReadyError(BatchError):
    """At least one job is unfinished, unsubmitted, or awaiting reconciliation."""


def _write(path: Path, value: dict) -> None:
    """Atomically replace a local JSON record."""
    atomic_json(path, value)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference(run: Path, manifest: dict) -> BatchJob:
    return BatchJob(
        output_path=run,
        batch_ids=[j["batch_id"] for j in manifest["jobs"] if j.get("batch_id")],
    )


def _run(value: BatchJob | Path | str) -> tuple[Path, dict]:
    run = (value.output_path if isinstance(value, BatchJob) else Path(value)).resolve()
    manifest = _read(run / "batch" / "manifest.json")
    if manifest.get("version") != 1:
        raise BatchError("Unsupported batch manifest version.", run)
    return run, manifest


def _save(run: Path, manifest: dict) -> None:
    _write(run / "batch" / "manifest.json", manifest)
    tracked = RunOutput.load(run)
    if tracked:
        states = [job["status"] for job in manifest["jobs"]]
        status = (
            "submission_unknown"
            if any(v in UNKNOWN for v in states)
            else (
                "preparing"
                if "prepared" in states
                else (
                    "awaiting_collection"
                    if all(v in TERMINAL | LOCAL_TERMINAL for v in states)
                    else "submitted"
                )
            )
        )
        tracked.update(
            status=status,
            input_documents=len(manifest["documents"]),
            batch_size=manifest["batch_size"],
            batch_job_count=len(states),
            remote_batch_ids=[
                j["batch_id"] for j in manifest["jobs"] if j.get("batch_id")
            ],
            successful_documents="pending",
            failed_documents="pending",
            finished_at="pending",
            duration_seconds="pending",
            **usage_totals([], pending=True),
        )


def _size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("batch_size must be a positive integer.")


def _valid_id(value: str) -> None:
    # IDs are used verbatim as filenames in existing encodings/metadata/errors.
    if not value.strip() or value in {".", ".."} or any(c in value for c in "/\\\x00"):
        raise ValueError(f"Invalid document ID for a saved encoding: {value!r}.")


def _remote_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise BatchError("OpenRouter returned an invalid batch ID.")


class BatchAnalyzerMixin:
    """OpenRouter-only batch methods exposed on LLMTrackerAnalyzer.

    config.max_retries controls ordinary coding, not these methods. HTTP calls
    have config.timeout and perform one attempt; repeat failed reads explicitly.
    No background polling, full-price fallback, or automatic inference retry.
    Operate on a given run from only one process at a time.
    """

    config: "AnalyzerConfig"

    def _batch_http(self, method: str, path: str, payload: dict | None = None) -> dict:
        if self.config.provider != "openrouter":
            raise ValueError(
                "Batch coding currently supports provider='openrouter' only."
            )
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.request(
                method,
                f"{BASE_URL}/{path}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise BatchError("Unexpected OpenRouter batch response.")
        return data

    def submit_batch_csv(
        self,
        csv_path: Path | str,
        codebook_path: Path | str,
        text_column: str,
        id_column: str | None = None,
        output_dir: Path | str = "LLM_batch_coding",
        *,
        batch_size: int = 1000,
    ) -> BatchJob:
        """Freeze CSV inputs and submit independent document requests in batches.

        Returns immediately after submission, not after inference. output_dir
        is a legacy base path; config.master_output_dir takes precedence, else
        its parent is used. Runs use llm_encodings_<model>_<timestamp>.
        batch_size is a maximum
        documents per remote job, not a prompt size or concurrency setting.
        Blank/missing text or IDs and duplicate IDs raise before any paid calls.
        IDs are read as strings to preserve leading zeros; omitted id_column
        uses CSV row indices. Unlike analyze_csv, duplicate IDs are rejected.
        """
        _size(batch_size)
        frame = pd.read_csv(csv_path, dtype={id_column: str} if id_column else None)
        required = [text_column] + ([id_column] if id_column is not None else [])
        missing = [c for c in required if c not in frame]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        documents = []
        for index, row in frame.iterrows():
            raw_id = row[id_column] if id_column is not None else str(index)
            if pd.isna(raw_id) or pd.isna(row[text_column]):
                raise ValueError(f"Missing document ID or text at CSV row {index}.")
            documents.append(
                {
                    "document_id": str(raw_id).strip(),
                    "text": str(row[text_column]),
                    "source": str(Path(csv_path).resolve()),
                    "source_row": int(index),
                }
            )
        return self._new_batch_run(documents, codebook_path, output_dir, batch_size)

    def submit_batch_directory(
        self,
        input_dir: Path | str,
        codebook_path: Path | str,
        output_dir: Path | str = "LLM_batch_coding",
        *,
        batch_size: int = 1000,
    ) -> BatchJob:
        """Submit supported .txt/.csv files using existing directory text loaders.

        Each file is one request; IDs are file stems. All files are loaded and
        validated before submission. Duplicate stems raise instead of overwriting.
        """
        _size(batch_size)
        documents = []
        for path in get_document_files(input_dir):
            text, document_id = load_document(path)
            documents.append(
                {
                    "document_id": document_id,
                    "text": text,
                    "source": str(path.resolve()),
                }
            )
        return self._new_batch_run(documents, codebook_path, output_dir, batch_size)

    def _new_batch_run(
        self,
        documents: list[dict],
        codebook_path: Path | str,
        output_dir: Path | str,
        batch_size: int,
    ) -> BatchJob:
        if self.config.provider != "openrouter":
            raise ValueError(
                "Batch coding currently supports provider='openrouter' only."
            )
        if not documents:
            raise ValueError("No documents to submit.")
        codebook = codebook_constructs(load_codebook(codebook_path))
        seen = set()
        run_id = uuid4().hex
        for index, document in enumerate(documents):
            doc_id = document["document_id"]
            _valid_id(doc_id)
            # Case-fold protects filenames on macOS's usual case-insensitive FS.
            if doc_id.casefold() in seen:
                raise ValueError(f"Duplicate document ID: {doc_id!r}. Use unique IDs.")
            seen.add(doc_id.casefold())
            if not document["text"].strip():
                raise ValueError(f"Empty text for document {doc_id!r}.")
            document["custom_id"] = f"{run_id}_{index:08d}"
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": construct_prompt(
                            document["text"], codebook, self.config.prompt_template
                        ),
                    }
                ],
                "response_format": {"type": "json_object"},
            }
            if self.config.temperature is not None:
                body["temperature"] = self.config.temperature
            document["body"] = body
        root = self.config.master_output_dir or Path(output_dir).expanduser().parent
        tracked = RunOutput.create(
            root,
            "encoding",
            self.config.model_name,
            mode="batch",
            **config_fields(self.config),
        )
        snapshots(tracked, self.config, codebook_path)
        tracked.update(
            input_path=sorted({d["source"] for d in documents}),
            input_documents=len(documents),
        )
        run = tracked.path
        self.output_path = run
        for subdir in ("encodings", "metadata", "errors", "batch", "DF"):
            (run / subdir).mkdir(exist_ok=True)
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model_name,
            "codebook": codebook,
            "prompt_template": self.config.prompt_template,
            "fuzzy_quote_matching": self.config.fuzzy_quote_matching,
            "quote_match_threshold": self.config.quote_match_threshold,
            "batch_size": batch_size,
            "documents": documents,
            "jobs": [],
        }
        self._prepare_jobs(manifest, [d["custom_id"] for d in documents], batch_size)
        _save(run, manifest)
        (run / "README.md").write_text(
            "# OpenRouter batch coding\n\nUse this run directory with batch_status(), "
            "collect_batch(), and retry_batch_errors().\nThe batch folder contains frozen "
            "source text, prompts, and remote responses. No API key is saved.\n"
            "Existing encoding/metadata/error outputs become available on collection.\n"
            "Use only one process at a time for this run.\n",
            encoding="utf-8",
        )
        print(f"Batch run directory: {run}")
        return self.resume_batch_submission(run)

    def _prepare_jobs(self, manifest: dict, custom_ids: list[str], size: int) -> None:
        for start in range(0, len(custom_ids), size):
            manifest["jobs"].append(
                {
                    "local_id": uuid4().hex,
                    "status": "prepared",
                    "custom_ids": custom_ids[start : start + size],
                }
            )

    def resume_batch_submission(self, run_dir: BatchJob | Path | str) -> BatchJob:
        """Submit only prepared chunks after interruption; never repeat known jobs.

        A submitting/submission_unknown chunk must first be reconciled using
        resolve_batch_submission(). It may already be running remotely.
        Exceptions include output_path so the persisted run can be recovered.
        """
        run, manifest = _run(run_dir)
        if self.config.provider != "openrouter":
            raise ValueError(
                "Batch coding currently supports provider='openrouter' only."
            )
        if any(j["status"] in UNKNOWN for j in manifest["jobs"]):
            raise BatchError(
                "Submission outcome unknown. Check OpenRouter's batches and use "
                "resolve_batch_submission() before continuing; do not resubmit blindly.",
                run,
            )
        documents = {d["custom_id"]: d for d in manifest["documents"]}
        for job in manifest["jobs"]:
            if job["status"] != "prepared":
                continue
            job["status"] = "submitting"
            _save(run, manifest)
            # Top-level routing fields must precede requests in OpenRouter JSON.
            payload = {
                "endpoint": "/v1/chat/completions",
                "model": manifest["model"],
                "completion_window": "24h",
                "requests": [
                    {"custom_id": cid, "body": documents[cid]["body"]}
                    for cid in job["custom_ids"]
                ],
            }
            try:
                remote = self._batch_http("POST", "batches", payload)
                _remote_id(remote.get("id"))
                if remote.get("status") not in TERMINAL | {
                    "validating",
                    "in_progress",
                    "finalizing",
                    "cancelling",
                }:
                    raise BatchError("Unexpected OpenRouter submission status.", run)
                job.update(
                    batch_id=remote["id"], status=remote.get("status", "validating")
                )
                _save(run, manifest)
                _write(run / "batch" / f"{job['local_id']}_submission.json", remote)
            except Exception as error:
                definite = (
                    isinstance(error, httpx.HTTPStatusError)
                    and 400 <= error.response.status_code < 500
                    and error.response.status_code != 408
                )
                if not job.get("batch_id"):
                    job["status"] = "rejected" if definite else "submission_unknown"
                job["error"] = f"{type(error).__name__}: {error}"
                _save(run, manifest)
                raise BatchError(
                    f"Batch submission {job['status']}. Saved run: {run}. "
                    "Check batch_status(refresh=False) and the saved manifest before retrying.",
                    run,
                ) from error
            _save(run, manifest)
        return _reference(run, manifest)

    def resolve_batch_submission(
        self,
        run_dir: BatchJob | Path | str,
        local_id: str,
        *,
        batch_id: str | None = None,
        confirmed_not_submitted: bool = False,
    ) -> BatchJob:
        """Manually reconcile an uncertain submission after checking OpenRouter.

        Supply a verified remote batch_id, or confirmed_not_submitted=True only
        if you have established that it was NOT accepted. No automatic matching
        is attempted. Model/count checks cannot prove identity: verify the ID
        belongs to this submission. Collection also validates its custom IDs.
        """
        run, manifest = _run(run_dir)
        if (batch_id is None) == (not confirmed_not_submitted):
            raise ValueError(
                "Supply exactly one of batch_id or confirmed_not_submitted=True."
            )
        matches = [j for j in manifest["jobs"] if j["local_id"] == local_id]
        if not matches or matches[0]["status"] not in UNKNOWN:
            raise BatchError("No uncertain submission with that local_id.", run)
        job = matches[0]
        if batch_id is not None:
            _remote_id(batch_id)
            if any(j.get("batch_id") == batch_id for j in manifest["jobs"]):
                raise BatchError(
                    "That remote batch ID is already attached to this run.", run
                )
            remote = self._batch_http("GET", f"batches/{batch_id}")
            if remote.get("id") != batch_id:
                raise BatchError("Remote batch ID does not match the supplied ID.", run)
            if (
                remote.get("model") != manifest["model"]
                or remote.get("endpoint") != "/v1/chat/completions"
            ):
                raise BatchError(
                    "Remote model/endpoint does not match the saved submission.", run
                )
            total = remote.get("request_counts", {}).get("total")
            if total is not None and total != len(job["custom_ids"]):
                raise BatchError("Remote request count does not match.", run)
            job.update(batch_id=batch_id, status=remote["status"])
            if remote["status"] in TERMINAL:
                self._cache_response(run, job, remote)
        else:
            job["status"] = "not_submitted"
            job["error"] = "User confirmed this request was not accepted remotely."
        _save(run, manifest)
        return _reference(run, manifest)

    def _cache_response(self, run: Path, job: dict, remote: dict) -> None:
        filename = f"{job['local_id']}_response.json"
        _write(run / "batch" / filename, remote)
        job["response_file"] = filename

    def batch_status(
        self, run_dir: BatchJob | Path | str, *, refresh: bool = True
    ) -> pd.DataFrame:
        """Check once and return one status row per remote job/attempt.

        refresh=False is an offline view of the manifest. Terminal responses are
        cached locally, including raw results; repeated collection needs no API
        calls for cached jobs. Network errors can be retried by rerunning this
        read. Pending jobs are not failures and never become empty encodings.
        """
        run, manifest = _run(run_dir)
        rows = []
        for job in manifest["jobs"]:
            if refresh and job.get("batch_id") and not job.get("response_file"):
                try:
                    remote = self._batch_http("GET", f"batches/{job['batch_id']}")
                    if remote.get("id") != job["batch_id"] or not isinstance(
                        remote.get("status"), str
                    ):
                        raise BatchError("Unexpected batch status response.", run)
                    job["status"] = remote["status"]
                    job["request_counts"] = remote.get("request_counts", {})
                    if job["status"] in TERMINAL:
                        self._cache_response(run, job, remote)
                    _save(run, manifest)
                except Exception as error:
                    raise BatchError(
                        f"Status check failed for {job['batch_id']}; rerun the read. "
                        "No inference was resubmitted.",
                        run,
                    ) from error
            rows.append(
                {
                    "local_id": job["local_id"],
                    "batch_id": job.get("batch_id"),
                    "status": job["status"],
                    "requested": len(job["custom_ids"]),
                    "completed": job.get("request_counts", {}).get("completed"),
                    "failed": job.get("request_counts", {}).get("failed"),
                }
            )
        return pd.DataFrame(rows)

    def collect_batch(
        self, run_dir: BatchJob | Path | str, *, refresh: bool = True
    ) -> tuple[dict[str, AnalysisResult], dict[str, dict], list[ErrorRecord]]:
        """Collect all terminal chunks into ordinary coding outputs.

        Raises BatchNotReadyError if any chunk is pending/prepared/uncertain;
        never returns a misleading partial success set. A completed job may
        still contain failed document requests. These return ErrorRecords.
        Missing/duplicate/unknown response IDs and malformed instance data are
        errors. Successful instances=[] means genuine completed absence.
        Uses frozen input text/settings even if analyzer config changed later.
        """
        status = self.batch_status(run_dir, refresh=refresh)
        run, manifest = _run(run_dir)
        if not status["status"].isin(TERMINAL | LOCAL_TERMINAL).all():
            raise BatchNotReadyError(
                "Run not ready. Check batch_status(); collect after all jobs "
                "finish, or resume/reconcile unsubmitted chunks.",
                run,
            )
        entries = {}
        for job in manifest["jobs"]:
            remote = (
                _read(run / "batch" / job["response_file"])
                if job.get("response_file")
                else {}
            )
            expected = set(job["custom_ids"])
            if job["status"] == "completed":
                items = remote.get("results")
                if not isinstance(items, list):
                    raise BatchError("Completed batch has no results array.", run)
                mapped = {}
                for item in items:
                    if (
                        not isinstance(item, dict)
                        or item.get("custom_id") not in expected
                    ):
                        raise BatchError(
                            "Unknown or malformed result ID in returned batch.", run
                        )
                    cid = item["custom_id"]
                    if cid in mapped:
                        raise BatchError("Duplicate result ID in returned batch.", run)
                    mapped[cid] = item
                for cid in expected:
                    entries[cid] = (job, mapped.get(cid))
            else:
                for cid in expected:
                    entries[cid] = (
                        job,
                        {
                            "error": remote.get("error")
                            or job.get("error")
                            or job["status"]
                        },
                    )
        results, metadata, errors = {}, {}, []
        for document in manifest["documents"]:
            doc_id = document["document_id"]
            job, item = entries[document["custom_id"]]
            raw = item or {"error": "Missing result for submitted document."}
            meta = APIMetadata(
                model=manifest["model"],
                raw_response={"batch_id": job.get("batch_id"), "result": raw},
            )
            try:
                if not item or item.get("error") is not None:
                    raise ValueError(f"Batch request failed: {raw.get('error')}")
                response = item.get("response") or {}
                if response.get("status_code") != 200:
                    raise ValueError(
                        f"Batch request HTTP status: {response.get('status_code')}"
                    )
                body = response.get("body") or {}
                if body.get("error"):
                    raise ValueError(f"Provider error: {body['error']}")
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(
                        f"Generation did not finish normally: {choice.get('finish_reason')}"
                    )
                text = choice["message"]["content"]
                data = validate_llm_output(text)
                # Existing parser skips invalid individual instances. For batch
                # coding reject the document instead of inventing empty absence.
                for instance in data["instances"]:
                    if isinstance(instance, dict) and instance.get("confidence") == 0:
                        instance = {"quote": "", **instance}
                    ConstructInstance.model_validate(instance)
                    if not instance.get("construct", "").strip() or (
                        instance.get("confidence") != 0
                        and not instance.get("quote", "").strip()
                    ):
                        raise ValueError(
                            "Instance construct and quote must be nonempty."
                        )
                    if instance.get("confidence") not in (0, 1, 2):
                        raise ValueError("LLM instance confidence must be 0, 1, or 2.")
                result = parse_llm_response(
                    text,
                    doc_id,
                    document["text"],
                    fuzzy_quote_matching=manifest["fuzzy_quote_matching"],
                    threshold=manifest["quote_match_threshold"],
                )
                meta.model = body.get("model") or manifest["model"]
                meta.usage = (
                    body.get("usage") if isinstance(body.get("usage"), dict) else None
                )
                meta.response_id = body.get("id")
                meta.created = (
                    body.get("created")
                    if isinstance(body.get("created"), int)
                    else None
                )
                save_analysis_result(result, run)
                (run / "errors" / f"{document_stem(doc_id)}_error.json").unlink(
                    missing_ok=True
                )
                results[doc_id] = result
            except (
                PromptingError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                AttributeError,
            ) as error:
                meta.error_message = str(error)
                meta.error_type = type(error).__name__
                record = ErrorRecord(
                    document_id=doc_id,
                    document_path=document["source"],
                    error_message=str(error),
                    model_used=manifest["model"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                save_error_record(record, run)
                (run / "encodings" / f"{document_stem(doc_id)}.json").unlink(
                    missing_ok=True
                )
                errors.append(record)
            save_metadata(meta, doc_id, run)
            metadata[doc_id] = redact(meta.model_dump())
        _write(
            run / "batch" / "collection_summary.json",
            {
                "total_documents": len(manifest["documents"]),
                "successful": len(results),
                "failed_document_ids": [e.document_id for e in errors],
            },
        )
        tracked = RunOutput.load(run)
        if tracked:
            results = finish_encoding(tracked, results, metadata, errors)
            usages = []
            for job in manifest["jobs"]:
                if job.get("response_file"):
                    raw = _read(run / "batch" / job["response_file"])
                    usages.append({"usage": raw.get("usage")})
                elif not job.get("batch_id"):
                    usages.append(
                        {
                            "usage": dict(
                                prompt_tokens=0,
                                completion_tokens=0,
                                total_tokens=0,
                                cost=0,
                            )
                        }
                    )
            tracked.update(
                usage_scope="all_batch_jobs_including_retries", **usage_totals(usages)
            )
        return results, metadata, errors

    def retry_batch_errors(
        self, run_dir: BatchJob | Path | str, *, batch_size: int | None = None
    ) -> BatchJob:
        """Explicitly submit failed documents in new batches within the same run.

        First collects/checks that all existing jobs are terminal. Successful
        documents are excluded; frozen original prompts/model/settings are reused.
        No failures means no POST. Repeating while retries run raises not-ready.
        Return to batch_status/collect_batch afterwards. Do not use retry_errors
        on these batch errors; it does not understand the saved CSV row mapping.
        """
        _, _, errors = self.collect_batch(run_dir)
        run, manifest = _run(run_dir)
        size = manifest["batch_size"] if batch_size is None else batch_size
        _size(size)
        if not errors:
            return _reference(run, manifest)
        failed = {e.document_id for e in errors}
        ids = [
            d["custom_id"] for d in manifest["documents"] if d["document_id"] in failed
        ]
        self._prepare_jobs(manifest, ids, size)
        _save(run, manifest)
        return self.resume_batch_submission(run)
