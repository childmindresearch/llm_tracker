# Local review workbench

The optional review workbench lets a human inspect source text, human coding,
LLM coding, and construct definitions side by side. Reviewers can save drafts,
complete or reopen decisions, flag follow-up, resume work, and export reviews.
It runs on your computer and makes no LLM calls or API requests. No API key is
needed. Streamlit usage telemetry is disabled by the launcher.

## Install and try the synthetic demo

Use Python 3.11 or 3.12. The package's existing NumPy 1.x dependency has no
Python 3.13 wheels. From a checkout of this repository:

```bash
python -m pip install ".[review]"
llm-tracker-review demo review-demo
llm-tracker-review run review-demo/classifications.csv --codebook review-demo/codebook.json
```

On Windows, use `py -3.12 -m pip` if `python` does not select your supported
Python installation. If the `llm-tracker-review` command is not on your PATH,
use `python -m llm_tracker.review` with the same arguments.

After this feature is merged, installation directly from GitHub also works
without a checkout:

```bash
python -m pip install "llm-tracker[review] @ git+https://github.com/childmindresearch/llm_tracker.git"
```

For Poetry development, use `poetry install --extras review`, then prefix
commands with `poetry run`. The base package keeps the UI dependencies optional.

The demo creates three synthetic classification rows, three quote-comparison
rows, and a small codebook. It refuses to overwrite existing demo files.

## Review your own data

### Binary classification workbooks

```bash
llm-tracker-review run classifications_perspective_taking.xlsx --codebook definitions.yaml --construct perspective_taking --data-dir review-data
```

CSV and XLSX are supported; XLSX reads the first worksheet. Required columns:

| Column | Meaning |
| --- | --- |
| `participant_id` | Source document or participant identifier |
| `text` | Text being classified |
| `human_classification` | Human binary label |
| `machine_classification` | LLM binary label |

Optional columns include `human_reasoning`, `machine_reasoning`, `recall_type`,
`prompt_id`, `prompt_text`, and `construct`. Additional columns are retained in
CSV exports. A `construct` column can contain several constructs; when missing
or blank, `--construct` supplies the name. String identifiers such as `0001`
and the literal text `NA` are preserved.

Labels accept 0/1, yes/no, true/false, Present/Not present, and absent. Empty
labels remain unknown. Other values, including 2 and fractional numbers, are
rejected. Unknown labels cannot count as agreement.

Codebooks may be flat JSON mappings (`{"name": {"definition": "..."}}`),
the package's `metadata`/`codebook` JSON envelope, or the local YAML format
(`name: {prompt_text: ...}`). A codebook is optional. Definitions and examples
are displayed alongside the coding; input `prompt_text` is also shown and
differences from supplied YAML prompts are flagged.

### Native quote comparisons

Pass `comparison_rows.csv`, saved by `LLMTrackerComparer.compare_results()`:

```bash
llm-tracker-review run comparison_run/comparison_rows.csv --codebook codebooks/my_codebook.json
```

Required columns are `doc_id`, `construct`, `status`, `human_quote`, and
`llm_quote`. Status must be `matched`, `human_only`, or `llm_only`; the quote
columns must agree with that status. Quotes and available span indices are
displayed. If the file also contains a `text` column, it is shown as context.
Full documents are not automatically recovered from quote files.

Reviewers accept or reject each original comparison outcome. Rejection needs
an explanation. This records an audit decision; it does not rematch quotes,
rewrite the comparison table, or recompute the package's metrics.

## Reviewer workflow

1. Enter an ID such as `reviewer_01` and choose **Start / resume**. IDs are
   case-insensitive and are identifiers, not authentication or access controls.
2. Select a construct and a row. The app opens the first pending row. Filters
   can limit the list to pending rows or rows flagged for follow-up.
3. Read the source, definition, and both codings.
4. For classifications, choose **Match** or **Do not match**. A match requires
   known, equal source labels and uses that label as the final adjudication.
   A disagreement needs a final Present/Not present label and reviewer notes.
5. For quote comparisons, choose **Accept comparison** or **Reject comparison**.
6. Use **Save draft / reopen** to save unfinished work or reopen a completed
   review. Use **Complete and next pending** to finish a row and move on.

Changes save only when you press a save button. Save before changing rows,
filters, reviewers, reloading, downloading, or closing the browser. Downloads
contain the last saved state. A completed decision edited as a draft becomes
pending again. Follow-up flags are independent of completion.

Run the same command and enter the same ID to resume. **Reload saved reviews**
refreshes a session after another tab changes it. Stale saves are rejected to
prevent overwriting newer decisions. Use a single tab for each reviewer/dataset.

Stop the server with Ctrl+C in its terminal. `--no-browser` suppresses automatic
browser opening, and `--port 8502` chooses another port if 8501 is in use. The
launcher binds only to `127.0.0.1`; use the displayed local URL.

## Saved files and exports

By default, files are stored under the working directory:

```text
review-data/
  <dataset SHA-256>/
    reviewer_01.json
    reviewer_01.csv
```

Use `--data-dir /path/to/folder` to choose a writable destination. Files are
never written into the installed package. JSON is the authoritative session;
CSV is regenerated on each save. JSON and CSV are each written with atomic
replacement. If a CSV write fails after JSON is saved, reload the saved session
before retrying. Invalid JSON sessions are reported and preserved, rather than
silently starting over. Restore a backup to the displayed path to recover.

After an interrupted process, a `.lock` file can remain beside a session.
Stop all review processes before removing that file and restarting. The app
refuses saves while the lock exists.

Dataset identity includes row content, row order, construct assignment, and
codebook definitions. Changing any of these starts separate progress. Renaming
a file or converting a CSV to an equivalent XLSX does not. Identical duplicate
rows receive separate row IDs. This version deliberately does not migrate
decisions between changed datasets or import older standalone workbench logs.

Keep the original input, codebook, and JSON session together when backing up
or moving to a different machine. Session JSON contains decisions and identity,
not the original source data. The CSV includes the source data, reviewer ID,
dataset hash, row ID, construct, review status, assessment, final label, notes,
follow-up flag, and timestamps. Original input columns use the `source_` prefix;
decision columns use `review_`. Pending rows are retained, so filter
`review_status == completed` before analyzing finished decisions. Free text
starting with spreadsheet formula characters is prefixed with an apostrophe
in CSV exports; source inputs and session notes are unchanged.

Use the sidebar download buttons to obtain the current CSV or JSON. To collect
CSV exports from several reviewers, use:

```bash
llm-tracker-review combine reviews-alice.csv reviews-bob.csv --output combined-reviews.csv
```

Each reviewer and dataset stays separate. Repeated snapshots retain the most
recent save for each dataset/reviewer/row; conflicting snapshots with the same
save time are rejected. The command refuses to overwrite an existing output.
Combining does not infer consensus, compute reliability, or alter LLM results.

The review UI has no cloud sharing or login. Input text and exported notes stay
in the local process and local files, and are sent to the browser on this
computer. Keep files in an appropriate location for your study. Do not commit
participant data; the default `review-data/` directory is ignored by Git.

## Development and validation

```bash
poetry install --extras review
poetry run pytest
poetry run ruff check
poetry run ruff format --check
poetry run mypy .
poetry run deptry src
```

Tests cover CSV/XLSX inputs, codebooks, invalid labels, duplicate rows, native
comparison output, completion rules, save/resume, dataset and reviewer
isolation, stale saves, corrupt session preservation, exports, and combining.
When the review extra is installed, Streamlit's `AppTest` also exercises the
classification and comparison workflows. CI runs these tests separately with
the review extra on Python 3.11 and 3.12.
