"""Shared output directories, descriptive filenames, and a readable run register."""

import csv
import hashlib
import inspect
import json
import os
import platform
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

import pandas as pd

SECRET_KEYS = re.compile(
    r"api_?key|authorization|access_?token|refresh_?token|password|secret", re.I
)
BASE_COLUMNS = [
    "run_id",
    "run_type",
    "mode",
    "status",
    "model",
    "matching_model",
    "provider",
    "started_at",
    "updated_at",
    "finished_at",
    "duration_seconds",
    "output_directory",
    "source_encoding_run_id",
    "source_human_run_id",
    "source_comparison_run_id",
    "input_path",
    "codebook_path",
    "prompt_path",
    "settings_path",
    "input_documents",
    "successful_documents",
    "failed_documents",
    "construct_count",
    "comparison_rows",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "usage_scope",
    "temperature",
    "max_retries",
    "timeout",
    "fuzzy_quote_matching",
    "quote_match_threshold",
    "text_column",
    "id_column",
    "batch_size",
    "remote_batch_ids",
    "batch_job_count",
    "jaccard_threshold",
    "n_examples",
    "n_counterexamples",
    "summarize_examples",
    "requested_reruns",
    "completed_versions",
    "seed",
    "error_message",
]
TABLE_NAMES = {
    "presence_grid": "llm_human_presence_grid",
    "doc_binary": "metrics_llm_human_binary",
    "doc_ordinal": "metrics_llm_human_instance_counts",
    "span_per_doc": "metrics_llm_human_quote_matching_by_document",
    "span_concatenated": "metrics_llm_human_quote_matching_pooled",
    "span_weighted": "metrics_llm_human_quote_matching_weighted",
    "comparison_rows": "llm_human_quote_matches",
    "llm_quotes": "llm_quotes_by_doc_index_score",
    "construct_quotes": "llm_quotes_by_construct",
}
SECRETS = set()


def register_secret(value):
    if value:
        SECRETS.add(value)


def slug(value):
    """Use letters/numbers/underscores for generated name components."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "unknown"


def document_stem(value):
    """Keep raw IDs in data; sanitize filenames without introducing collisions."""
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_]+", text):
        return text
    return slug(text) + "_" + hashlib.sha256(text.encode()).hexdigest()[:12]


def redact(value, secrets=()):
    """Remove credential fields and redact key strings from nested saved values."""
    secrets = tuple(SECRETS) + tuple(secrets)
    if isinstance(value, dict):
        return {
            str(k): redact(v, secrets)
            for k, v in value.items()
            if not SECRET_KEYS.search(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for secret in secrets:
            if isinstance(secret, str) and secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"sk-or-[A-Za-z0-9_-]+", "[REDACTED]", value)
        value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
    return value


def atomic_json(path, data, secrets=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("_tmp_" + uuid4().hex)
    try:
        tmp.write_text(
            json.dumps(
                redact(data, secrets), ensure_ascii=False, indent=2, default=str
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def register_lock(root):
    """Serialize register updates; do not lose another process's run row."""
    lock = root / "_run_metadata_lock"
    deadline = time.monotonic() + 10
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Metadata register locked: {lock}. If a writer crashed, remove this lock after verifying it stopped."
                )
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


class RunResults(dict):
    """Ordinary result dictionary carrying provenance for later comparison runs."""

    def __init__(self, values=(), *, output_path=None):
        super().__init__(values)
        self.output_path = Path(output_path) if output_path else None


def source_run(results):
    path = getattr(results, "output_path", None)
    return RunOutput.load(path) if path else None


class RunOutput:
    """Persistent run identity shared by tables and the master metadata CSV."""

    def __init__(self, path, record):
        self.path, self.record = Path(path), record

    @classmethod
    def load(cls, path):
        if path is None:
            return None
        p = Path(path).expanduser().resolve()
        marker = p / "run_metadata.json"
        if not marker.exists():
            return None
        return cls(p, json.loads(marker.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, root, run_type, model="unknown", *, mode="N/A", **fields):
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")
        prefix = {
            "encoding": "llm_encodings",
            "comparison_short": "comparison_short",
            "comparison_long": "comparison_long",
            "optimization": "codebook_optimization",
            "text_descriptives": "text_descriptives",
        }[run_type]
        suffix = f"{slug(model)}_{stamp}"
        path = root if run_type == "text_descriptives" else root / f"{prefix}_{suffix}"
        if path != root:
            path.mkdir(exist_ok=False)
        record = dict.fromkeys(BASE_COLUMNS, "N/A")
        parent = ACTIVE_RUN.get()
        if parent:
            record["parent_run_id"] = parent.record["run_id"]
        record.update(
            run_id=uuid4().hex,
            run_type=run_type,
            mode=mode,
            model=model,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            master_output_dir=str(root),
            output_directory=str(path),
            suffix=suffix,
            timestamp=stamp,
            python_version=platform.python_version(),
            storage_schema_version=1,
        )
        run = cls(path, record)
        run.update(**fields)
        return run

    def update(self, **fields):
        # Reload to preserve fields added by another API step using the same run.
        if self.record["run_type"] != "text_descriptives":
            saved = self.load(self.path)
            if saved:
                self.record = saved.record
        self.record.update(redact(fields))
        self.record["updated_at"] = datetime.now(timezone.utc).isoformat()
        root = Path(self.record["master_output_dir"])
        with register_lock(root):
            target = root / "run_metadata.csv"
            rows, columns = [], list(BASE_COLUMNS)
            if target.exists():
                with target.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    columns = list(reader.fieldnames or BASE_COLUMNS)
                    rows = list(reader)
            # Internal routing fields are represented by output_directory/run_id.
            values = {
                k: v
                for k, v in redact(self.record).items()
                if k not in {"suffix", "timestamp", "master_output_dir"}
            }
            for key in values:
                if key not in columns:
                    columns.append(key)
            values = {
                k: (
                    "unavailable"
                    if v is None
                    else (
                        json.dumps(v, ensure_ascii=False, default=str)
                        if isinstance(v, (list, dict))
                        else v
                    )
                )
                for k, v in values.items()
            }
            rows = [r for r in rows if r.get("run_id") != self.record["run_id"]]
            rows.append(values)
            tmp = root / ("_metadata_" + uuid4().hex + ".csv")
            try:
                with tmp.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=columns, restval="N/A")
                    writer.writeheader()
                    writer.writerows(rows)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
        if self.record["run_type"] != "text_descriptives":
            atomic_json(self.path / "run_metadata.json", self.record)
        else:
            atomic_json(
                root / f"text_descriptives_metadata_{self.record['timestamp']}.json",
                self.record,
            )
        return self

    def finish(self, status="completed", **fields):
        now = datetime.now(timezone.utc)
        return self.update(
            status=status,
            finished_at=now.isoformat(),
            duration_seconds=round(
                (
                    now - datetime.fromisoformat(self.record["started_at"])
                ).total_seconds(),
                3,
            ),
            **fields,
        )

    def filename(self, stem, extension="csv"):
        return (
            f"{slug(TABLE_NAMES.get(stem, stem))}_{self.record['suffix']}.{extension}"
        )

    def table(self, stem, frame, *, folder=None):
        directory = self.path / folder if folder else self.path
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / self.filename(stem)
        safe_frame = frame.copy()
        for column in safe_frame.select_dtypes(include=["object", "string"]).columns:
            safe_frame[column] = safe_frame[column].map(redact)
        safe_frame.to_csv(target, index=False)
        frame.attrs["output_path"] = str(self.path)
        frame.attrs["run_id"] = self.record["run_id"]
        self.update(**{slug(stem) + "_path": str(target)})
        return target


def config_fields(config):
    return {
        name: getattr(config, name)
        for name in (
            "provider",
            "temperature",
            "max_retries",
            "timeout",
            "fuzzy_quote_matching",
            "quote_match_threshold",
        )
    }


def snapshots(run, config, codebook_path=None):
    settings = config_fields(config)
    atomic_json(run.path / "settings.json", settings)
    (run.path / "prompt.txt").write_text(
        redact(config.prompt_template, (config.api_key,)), encoding="utf-8"
    )
    fields = dict(
        settings_path=str(run.path / "settings.json"),
        prompt_path=str(run.path / "prompt.txt"),
    )
    if codebook_path:
        atomic_json(
            run.path / "codebook.json",
            json.loads(Path(codebook_path).read_text(encoding="utf-8")),
            (config.api_key,),
        )
        fields["codebook_path"] = str(Path(codebook_path).resolve())
    fields["prompt_sha256"] = hashlib.sha256(
        (run.path / "prompt.txt").read_bytes()
    ).hexdigest()
    if (run.path / "codebook.json").exists():
        fields["codebook_sha256"] = hashlib.sha256(
            (run.path / "codebook.json").read_bytes()
        ).hexdigest()
    run.update(**fields)


def usage_totals(metas, *, pending=False):
    if pending:
        return {
            k: "pending"
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")
        }
    usages = [m.get("usage") or {} for m in metas]

    def total(name):
        values = [u.get(name) for u in usages]
        # Never report partial usage/cost as the full run total.
        return (
            sum(values)
            if values and all(isinstance(v, (float, int)) for v in values)
            else "unavailable"
        )

    return dict(
        prompt_tokens=total("prompt_tokens"),
        completion_tokens=total("completion_tokens"),
        total_tokens=total("total_tokens"),
        cost_usd=total("cost"),
    )


def export_quotes(run, results, *, encoding=False):
    from llm_tracker.utils import format_coding_table

    quotes = format_coding_table(results)
    run.table("llm_quotes", quotes, folder="DF" if encoding else None)
    if not encoding:
        run.table(
            "construct_quotes",
            quotes[["construct", "quote"]]
            .sort_values("construct", kind="stable")
            .reset_index(drop=True),
        )


def finish_encoding(run, results, metadata, errors):
    totals = (
        {}
        if (run.path / "api_calls").exists()
        else usage_totals(list(metadata.values()))
    )
    run.finish(
        "completed_with_errors" if errors else "completed",
        successful_documents=len(results),
        failed_documents=len(errors),
        input_documents=len(results) + len(errors),
        construct_count=len(
            {i.construct for r in results.values() for i in r.instances}
        ),
        **totals,
    )
    export_quotes(run, results, encoding=True)
    return RunResults(results, output_path=run.path)


def encoding_run(func):
    """Wrap existing synchronous coding without changing its return tuple."""

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        bound = inspect.signature(func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        supplied = bound.arguments.get("output_dir")
        root = self.config.master_output_dir or (
            Path(supplied).parent if supplied else Path.cwd()
        )
        run = RunOutput.create(
            root,
            "encoding",
            self.config.model_name,
            mode="synchronous",
            **config_fields(self.config),
        )
        self.output_path = run.path
        token = ACTIVE_RUN.set(run)
        try:
            snapshots(run, self.config, bound.arguments.get("codebook_path"))
            run.update(
                input_path=str(
                    bound.arguments.get("csv_path", bound.arguments.get("input_dir"))
                ),
                text_column=bound.arguments.get("text_column", "N/A"),
                id_column=bound.arguments.get("id_column") or "row_index",
            )
            bound.arguments["output_dir"] = str(run.path)
            results, metadata, errors = func(*bound.args, **bound.kwargs)
            return finish_encoding(run, results, metadata, errors), metadata, errors
        except BaseException as e:
            run.finish(
                "interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
                error_message=redact(str(e), (self.config.api_key,)),
            )
            raise
        finally:
            ACTIVE_RUN.reset(token)

    return wrapped


def comparison_run(func):
    """Keep quote matching logic separate from storage/provenance."""

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        bound = inspect.signature(func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        supplied = bound.arguments.get("output_dir")
        root = self.config.master_output_dir or (Path(supplied) if supplied else None)
        if root is None:
            return func(*bound.args, **bound.kwargs)
        source = source_run(bound.arguments["llm_results"])
        human = source_run(bound.arguments["human_results"])
        model = source.record["model"] if source else self.config.model_name
        run = RunOutput.create(
            root,
            "comparison_long",
            model,
            matching_model=self.config.model_name,
            source_encoding_run_id=source.record["run_id"] if source else "unavailable",
            source_human_run_id=human.record["run_id"] if human else "unavailable",
            **config_fields(self.config),
        )
        snapshots(run, self.config)
        from llm_tracker.comparison import MATCH_PROMPT_TEMPLATE

        (run.path / "prompt.txt").write_text(
            redact(MATCH_PROMPT_TEMPLATE), encoding="utf-8"
        )
        run.update(
            mode="quote_matching",
            binarize_confidence=bound.arguments.get("binarize_confidence"),
        )
        bound.arguments["output_dir"] = None
        token = ACTIVE_RUN.set(run)
        try:
            frame = func(*bound.args, **bound.kwargs)
            run.table("comparison_rows", frame)
            self.output_path = run.path
            export_quotes(run, bound.arguments["llm_results"])
            count = len(
                set(bound.arguments["human_results"])
                | set(bound.arguments["llm_results"])
            )
            no_calls = (
                {}
                if (run.path / "api_calls").exists()
                else dict(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0
                )
            )
            run.finish(
                comparison_rows=len(frame),
                input_documents=count,
                successful_documents=count,
                failed_documents=0,
                **no_calls,
            )
            return frame
        except BaseException as e:
            run.finish("failed", error_message=redact(str(e), (self.config.api_key,)))
            raise
        finally:
            ACTIVE_RUN.reset(token)

    return wrapped


def table_run(output_dir, frame=None, *, kind="comparison_long", model="unknown"):
    """Reuse a run directory; otherwise create an explicitly requested run."""
    existing = RunOutput.load(output_dir)
    if existing:
        return existing
    if frame is not None and frame.attrs.get("output_path"):
        existing = RunOutput.load(frame.attrs["output_path"])
        if (
            existing
            and existing.record["run_type"].startswith("comparison")
            and Path(output_dir).expanduser().resolve()
            == Path(existing.record["master_output_dir"])
        ):
            return existing
    return RunOutput.create(output_dir, kind, model)


def save_tables(output_dir, tables):
    run = table_run(
        output_dir,
        next(iter(tables.values()), None),
        kind="comparison_short" if set(tables) == {"doc_binary"} else "comparison_long",
    )
    for stem, frame in tables.items():
        run.table(stem, frame)
    run.finish()
    return run.path


def optimization_run(func):
    """Track one optimization plus its linked encoding/comparison child runs."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        config = bound.arguments["analyzer"].config
        root = config.master_output_dir or bound.arguments["output_dir"]
        run = RunOutput.create(
            root, "optimization", config.model_name, **config_fields(config)
        )
        snapshots(run, config)
        fields = {
            k: bound.arguments[k]
            for k in (
                "jaccard_threshold",
                "n_examples",
                "n_counterexamples",
                "summarize_examples",
                "seed",
            )
        }
        fields.update(
            input_path=str(bound.arguments["path"]),
            requested_reruns=bound.arguments["rerun_optimized_codebook"],
            source_comparison_run_id=bound.arguments["comparison_df"].attrs.get(
                "run_id", "unavailable"
            ),
        )
        run.update(**fields)
        atomic_json(run.path / "source_codebook.json", bound.arguments["codebook"])
        run.update(
            codebook_path=str(run.path / "source_codebook.json"),
            analysis_options=bound.arguments["analyze_kwargs"],
        )
        bound.arguments["output_dir"] = run.path
        token = ACTIVE_RUN.set(run)
        try:
            partials = func(*bound.args, **bound.kwargs)
            no_calls = (
                {}
                if (run.path / "api_calls").exists()
                else dict(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0
                )
            )
            run.finish(completed_versions=len(partials), **no_calls)
            return partials
        except BaseException as e:
            run.finish("failed", error_message=redact(str(e)))
            raise
        finally:
            ACTIVE_RUN.reset(token)

    return wrapped


from contextvars import ContextVar

ACTIVE_RUN = ContextVar("llm_tracker_active_output_run", default=None)


def record_usage(func):
    """Capture every successful/failed API attempt for the active local run."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        run = ACTIVE_RUN.get()
        if run is None:
            return func(*args, **kwargs)
        calls = int(run.record.get("api_requests", 0)) + 1
        run.update(api_requests=calls)
        try:
            result = func(*args, **kwargs)
            metadata = result[1].model_dump()
            return result
        except Exception as error:
            metadata = {"usage": None, "error_message": redact(str(error))}
            raise
        finally:
            if "metadata" in locals():
                atomic_json(run.path / "api_calls" / f"call_{calls:06d}.json", metadata)
                current = usage_totals([metadata])
                totals = {}
                for name, value in current.items():
                    previous = 0 if calls == 1 else run.record.get(name, "unavailable")
                    totals[name] = (
                        previous + value
                        if isinstance(previous, (int, float))
                        and isinstance(value, (int, float))
                        else "unavailable"
                    )
                run.update(usage_scope="all_recorded_api_attempts", **totals)

    return wrapped


def descriptives_run(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        root = bound.arguments.get("output_dir")
        if root is None:
            return func(*bound.args, **bound.kwargs)
        fields = {
            k: v for k, v in bound.arguments.items() if k not in {"texts", "output_dir"}
        }
        run = RunOutput.create(
            root,
            "text_descriptives",
            "N/A",
            input_documents=len(bound.arguments["texts"]),
            **fields,
        )
        try:
            summary, raw = func(*bound.args, **bound.kwargs)
            run.table("text_descriptives_summary", summary)
            run.table("text_descriptives_by_document", raw)
            run.finish(successful_documents=len(raw), failed_documents=0)
            return summary, raw
        except BaseException as e:
            run.finish("failed", error_message=redact(str(e)))
            raise

    return wrapped


def retry_encoding_run(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        bound = inspect.signature(func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        run = RunOutput.load(bound.arguments["output_dir"])
        if not run:
            return func(*bound.args, **bound.kwargs)
        run.update(status="running", finished_at="pending")
        token = ACTIVE_RUN.set(run)
        try:
            recovered, metadata, errors = func(*bound.args, **bound.kwargs)
            from llm_tracker.models import AnalysisResult, ErrorRecord

            all_results = {}
            for p in (run.path / "encodings").glob("*.json"):
                result = AnalysisResult.model_validate_json(
                    p.read_text(encoding="utf-8")
                )
                all_results[result.document_id] = result
            all_metadata = {
                p.stem: json.loads(p.read_text(encoding="utf-8"))
                for p in (run.path / "metadata").glob("*.json")
            }
            all_errors = [
                ErrorRecord.model_validate_json(p.read_text(encoding="utf-8"))
                for p in (run.path / "errors").glob("*.json")
            ]
            finish_encoding(run, all_results, all_metadata, all_errors)
            return RunResults(recovered, output_path=run.path), metadata, errors
        except BaseException as e:
            run.finish("failed", error_message=redact(str(e)))
            raise
        finally:
            ACTIVE_RUN.reset(token)

    return wrapped
