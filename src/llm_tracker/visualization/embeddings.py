"""Select representative example quotes per construct using sentence embeddings.

The main entry point is :func:`examples`, which turns the row-level coding table
produced by :func:`llm_tracker.utils.format_coding_table` into a construct-level
table ranked by prevalence, where each construct is illustrated by a handful of
real quotes.

With the default ``method="centroid"`` the quotes of a construct are encoded with
a sentence-transformers model, and the examples are taken at evenly spaced ranks
of their cosine distance to the set centroid: the first is the most typical quote,
the last the most peripheral one. With ``method="random"`` the examples are simply
sampled at random and no model is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
CAPTION_FILENAME = "example_caption_README.txt"

TABLE_COLUMNS = ["construct", "N", "%", "examples"]


class Encoder(Protocol):
    """Minimal interface expected from an embedding model."""

    def encode(self, texts: list[str]) -> Any:  # noqa: D102
        ...


def _load_encoder(model_name: str) -> Encoder:
    """Instantiate a sentence-transformers model, with a helpful error if absent."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "method='centroid' requires sentence-transformers. Install it with "
            "`pip install sentence-transformers`, or call examples() with "
            "method='random', or pass your own model via encoder=."
        ) from exc

    return cast(Encoder, SentenceTransformer(model_name))


def _unique_quotes(quotes: pd.Series) -> list[str]:
    """Return the non-empty quotes of a construct, deduplicated, in order."""
    cleaned = (str(quote).strip() for quote in quotes.dropna())
    return list(dict.fromkeys(quote for quote in cleaned if quote))


def _spread_positions(n_available: int, n_wanted: int) -> list[int]:
    """Ranks to pick from a distance-sorted list, evenly spaced from first to last."""
    k = min(n_wanted, n_available)
    if k <= 0:
        return []
    if k == 1:
        return [0]
    spread = np.linspace(0, n_available - 1, k)
    return sorted({int(round(position)) for position in spread})


def _centroid_selection(vectors: np.ndarray, n: int) -> list[int]:
    """Indices of the quotes spanning closest to furthest from the set centroid."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1.0, norms)

    centroid = normalized.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    distances = 1.0 - normalized @ centroid
    order = np.argsort(distances, kind="stable")
    return [int(order[position]) for position in _spread_positions(len(order), n)]


def _random_selection(n_available: int, n: int, rng: np.random.Generator) -> list[int]:
    """Indices of ``n`` quotes sampled without replacement."""
    k = min(n, n_available)
    if k <= 0:
        return []
    return [int(i) for i in rng.choice(n_available, size=k, replace=False)]


def _format_examples(quotes: list[str]) -> str:
    """Join the selected quotes, one per line, each wrapped in double quotes."""
    return "\n".join(f'"{quote}"' for quote in quotes)


def _build_caption(method: str, n: int, model_name: str, random_state: int) -> str:
    """Describe how the examples in the table were chosen."""
    ranking = (
        "Constructs are ranked by the percentage of coded instances they account for."
    )

    if method == "random":
        return (
            f"Constructs with examples. The {n} examples per construct are quotes "
            f"sampled at random from the construct set "
            f"(random_state={random_state}). {ranking}"
        )

    if n == 3:
        return (
            "Constructs with examples. First example is the quote closest to the "
            f"centroid of the construct set encoded using {model_name}. Second "
            "example is close to the median. Third example is the furthest from "
            f"the centroid. {ranking}"
        )

    return (
        f"Constructs with examples. The {n} examples per construct are quotes taken "
        "at evenly spaced ranks of their cosine distance to the centroid of the "
        f"construct set encoded using {model_name}, from the closest to the "
        f"furthest. {ranking}"
    )


def examples(
    df_with_quotes: pd.DataFrame,
    *,
    n: int = 3,
    method: Literal["centroid", "random"] = "centroid",
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Encoder | None = None,
    construct_column: str = "construct",
    quote_column: str = "quote",
    output_dir: str | Path | None = ".",
    random_state: int = 42,
    decimals: int = 1,
) -> pd.DataFrame:
    """Build a construct-level table of prevalence and representative quotes.

    Args:
    ----
        df_with_quotes: Row-level coding table, one row per construct instance,
            as returned by :func:`llm_tracker.utils.format_coding_table`.
        n: Number of example quotes per construct.
        method: ``"centroid"`` selects quotes at evenly spaced distances from the
            construct centroid (closest, ..., furthest); ``"random"`` samples them.
        model_name: Sentence-transformers model used to encode the quotes, and the
            model named in the caption. Use a multilingual model such as
            ``"paraphrase-multilingual-MiniLM-L12-v2"`` for non-English corpora.
            Ignored for ``method="random"``.
        encoder: Pre-loaded embedding model exposing ``encode(list[str])``. When
            given, ``model_name`` is only used for the caption.
        construct_column: Column holding the construct name.
        quote_column: Column holding the quote text.
        output_dir: Directory where the caption is written as
            ``example_caption_README.txt``. Pass ``None`` to only print it.
        random_state: Seed for ``method="random"``.
        decimals: Decimal places for the percentage column. Raise it when small
            constructs would otherwise round down to ``0.0``.

    Returns:
    -------
        DataFrame with one row per construct and the columns ``construct``, ``N``
        (number of coded instances), ``%`` (share of all instances) and
        ``examples`` (the selected quotes, one per line), sorted by ``%``.

    Raises:
    ------
        ValueError: If ``method`` is not ``"centroid"`` or ``"random"``, or if
            ``n`` is not positive.
        KeyError: If the construct or quote column is missing.

    """
    if method not in ("centroid", "random"):
        raise ValueError(f"method must be 'centroid' or 'random', got {method!r}")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")

    if df_with_quotes.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    missing = [
        column
        for column in (construct_column, quote_column)
        if column not in df_with_quotes.columns
    ]
    if missing:
        raise KeyError(f"df_with_quotes is missing column(s): {', '.join(missing)}")

    table = df_with_quotes[[construct_column, quote_column]].dropna(
        subset=[construct_column]
    )
    counts = table[construct_column].value_counts()
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    quotes_by_construct = {
        str(construct): _unique_quotes(group[quote_column])
        for construct, group in table.groupby(construct_column, sort=False)
    }

    vectors_by_quote: dict[str, np.ndarray] = {}
    if method == "centroid":
        all_quotes = list(
            dict.fromkeys(q for quotes in quotes_by_construct.values() for q in quotes)
        )
        if all_quotes:
            model = encoder if encoder is not None else _load_encoder(model_name)
            print(f"Encoding {len(all_quotes)} unique quotes with {model_name}…")
            encoded = np.asarray(model.encode(all_quotes), dtype=float)
            vectors_by_quote = dict(zip(all_quotes, encoded, strict=True))

    rng = np.random.default_rng(random_state)

    rows = []
    for construct, count in counts.items():
        quotes = quotes_by_construct[str(construct)]
        if method == "centroid" and quotes:
            vectors = np.vstack([vectors_by_quote[quote] for quote in quotes])
            selected = _centroid_selection(vectors, n)
        else:
            selected = _random_selection(len(quotes), n, rng)

        rows.append(
            {
                "construct": construct,
                "N": int(count),
                "%": round(100 * int(count) / total, decimals),
                "examples": _format_examples([quotes[i] for i in selected]),
            }
        )

    summary = (
        pd.DataFrame(rows, columns=TABLE_COLUMNS)
        .sort_values(["%", "construct"], ascending=[False, True])
        .reset_index(drop=True)
    )

    caption = _build_caption(method, n, model_name, random_state)
    print("\n" + caption)
    if output_dir is not None:
        caption_path = Path(output_dir) / CAPTION_FILENAME
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(caption + "\n", encoding="utf-8")
        print(f"Caption written to {caption_path}")

    return summary
