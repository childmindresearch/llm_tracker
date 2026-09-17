"""Tests for construct example selection via embeddings."""

import math

import numpy as np
import pandas as pd
from llm_tracker.visualization import embeddings


class FakeEncoder:
    """Deterministic stand-in for a SentenceTransformer, keyed by quote text."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003, ANN201
        return np.array([self.vectors[text] for text in texts], dtype=float)


def unit(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


# Five quotes placed on the unit circle. The centroid of the set sits at
# ~14.75 degrees, so ordered by distance to it they are:
#   a9 (closest), a6, a3, a0, a60 (furthest)
# and the evenly spaced picks for n=3 are a9, a3, a60.
ANGLED_VECTORS = {
    "a0": unit(0),
    "a3": unit(3),
    "a6": unit(6),
    "a9": unit(9),
    "a60": unit(60),
}


def coding_table(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"doc_id": f"doc_{i}", "construct": c, "quote": q}
            for i, (c, q) in enumerate(pairs)
        ]
    )


def examples_of(table: pd.DataFrame, row: int = 0) -> str:
    """The examples cell of one row, narrowed to str for the type checker."""
    return str(table.loc[row, "examples"])


def test_empty_table_returns_expected_columns() -> None:
    table = embeddings.examples(coding_table([]), output_dir=None)

    assert table.empty
    assert list(table.columns) == ["construct", "N", "%", "examples"]


def test_counts_and_percentages_sorted_by_percentage() -> None:
    rows = [("loneliness", f"l{i}") for i in range(5)]
    rows += [("hopelessness", f"h{i}") for i in range(3)]
    encoder = FakeEncoder({q: unit(10 * i) for i, (_, q) in enumerate(rows)})

    table = embeddings.examples(coding_table(rows), encoder=encoder, output_dir=None)

    assert list(table["construct"]) == ["loneliness", "hopelessness"]
    assert list(table["N"]) == [5, 3]
    assert list(table["%"]) == [62.5, 37.5]


def test_decimals_controls_percentage_rounding() -> None:
    rows = [("loneliness", f"l{i}") for i in range(999)]
    rows += [("hopelessness", "h0")]
    encoder = FakeEncoder({q: unit(i % 90) for i, (_, q) in enumerate(rows)})

    default = embeddings.examples(coding_table(rows), encoder=encoder, output_dir=None)
    coarse = embeddings.examples(
        coding_table(rows), encoder=encoder, decimals=0, output_dir=None
    )

    assert default.loc[1, "%"] == 0.1
    assert coarse.loc[1, "%"] == 0.0


def test_centroid_picks_closest_median_and_furthest() -> None:
    rows = [("loneliness", q) for q in ANGLED_VECTORS]
    encoder = FakeEncoder(ANGLED_VECTORS)

    table = embeddings.examples(coding_table(rows), encoder=encoder, output_dir=None)

    assert examples_of(table) == '"a9"\n"a3"\n"a60"'


def test_fewer_quotes_than_requested_returns_all_without_repeats() -> None:
    rows = [("loneliness", "a0"), ("loneliness", "a60")]
    encoder = FakeEncoder(ANGLED_VECTORS)

    table = embeddings.examples(coding_table(rows), encoder=encoder, output_dir=None)

    assert table.loc[0, "N"] == 2
    assert examples_of(table).count("\n") == 1


def test_duplicate_quotes_are_deduplicated_before_selection() -> None:
    rows = [("loneliness", q) for q in ["a0", "a0", "a0", "a3", "a60"]]
    encoder = FakeEncoder(ANGLED_VECTORS)

    table = embeddings.examples(coding_table(rows), encoder=encoder, output_dir=None)

    quotes = examples_of(table).split("\n")
    assert table.loc[0, "N"] == 5
    assert len(set(quotes)) == 3


def test_random_method_is_reproducible_and_needs_no_encoder() -> None:
    rows = [("loneliness", f"l{i}") for i in range(10)]

    first = embeddings.examples(
        coding_table(rows), method="random", random_state=7, output_dir=None
    )
    second = embeddings.examples(
        coding_table(rows), method="random", random_state=7, output_dir=None
    )

    assert examples_of(first) == examples_of(second)
    assert len(examples_of(first).split("\n")) == 3


def test_n_controls_how_many_examples_are_returned() -> None:
    rows = [("loneliness", q) for q in ANGLED_VECTORS]
    encoder = FakeEncoder(ANGLED_VECTORS)

    table = embeddings.examples(
        coding_table(rows), n=5, encoder=encoder, output_dir=None
    )

    expected = ['"a9"', '"a6"', '"a3"', '"a0"', '"a60"']
    assert examples_of(table).split("\n") == expected


def test_caption_is_written_to_output_dir_and_names_the_model(tmp_path) -> None:  # noqa: ANN001
    rows = [("loneliness", q) for q in ANGLED_VECTORS]
    encoder = FakeEncoder(ANGLED_VECTORS)

    embeddings.examples(
        coding_table(rows),
        encoder=encoder,
        model_name="my-embedding-model",
        output_dir=tmp_path,
    )

    caption = (tmp_path / "example_caption_README.txt").read_text()
    assert "closest to the centroid" in caption
    assert "my-embedding-model" in caption
    assert "furthest from the centroid" in caption
