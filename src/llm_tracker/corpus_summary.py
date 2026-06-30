"""
corpus_summary.py
-----------------
Generate a descriptive summary table for a corpus of text documents.

Core metrics (always computed via TextDescriptives + custom logic):
  - N documents
  - doc_length (word count)
  - n_sentences per doc
  - sentence_length_mean (SD)
  - n_tokens
  - n_unique_tokens  → also reported as % of total tokens
  - n_characters
  - n_stop_words     → also reported as % of tokens
  - proportion_1st_person_pronouns
  - type_token_ratio (TTR)
  - lexical_density  (content words / total tokens)
  - pos_prop_NOUN, pos_prop_VERB, pos_prop_ADJ, pos_prop_ADV
  - dependency_distance_mean (syntactic complexity proxy)
  - alpha_ratio       (proportion of alphabetic tokens; quality signal)

Optional metrics (pass flags to summarize_corpus):
  - coherence_sbert   : mean (SD) cosine similarity between adjacent sentence
                        embeddings, using a sentence-transformers model
  - readability       : Flesch Reading Ease, Flesch-Kincaid Grade,
                        Gunning Fog, Coleman-Liau, ARI, SMOG
  - sentence_complexity : dependency_distance mean & std (more detailed),
                          mean parse tree depth
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import spacy
import textdescriptives as td


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_1ST_PERSON = re.compile(
    r"\b(i|me|my|myself|mine|we|us|our|ourselves|ours)\b", re.IGNORECASE
)


def _fmt_mean_sd(values: pd.Series, decimals: int = 2) -> str:
    """Return 'mean (SD)' string, or 'N/A' if all NaN."""
    v = values.dropna()
    if v.empty:
        return "N/A"
    return f"{v.mean():.{decimals}f} ({v.std():.{decimals}f})"


def _fmt_mean_sd_range(values: pd.Series, decimals: int = 2) -> str:
    """Return 'mean X (SD Y) [min A – max B]' string with inline labels."""
    v = values.dropna()
    if v.empty:
        return "N/A"
    return (
        f"mean {v.mean():.{decimals}f} (SD {v.std():.{decimals}f}) "
        f"[min {v.min():.{decimals}f} – max {v.max():.{decimals}f}]"
    )


def _fmt_pct_n(values: pd.Series, decimals: int = 1) -> str:
    """Return 'mean X% (SD Y%) [min A% – max B%]' for a proportion series (0-1).

    Inline labels make each statistic explicit. Useful for proportions derived
    per-document.
    """
    v = values.dropna()
    if v.empty:
        return "N/A"
    pct = v * 100
    return (
        f"mean {pct.mean():.{decimals}f}% (SD {pct.std():.{decimals}f}%) "
        f"[min {pct.min():.{decimals}f}% – max {pct.max():.{decimals}f}%]"
    )


# ---------------------------------------------------------------------------
# SBERT coherence
# ---------------------------------------------------------------------------


def _sbert_coherence_per_doc(
    doc_text: str,
    model,
    nlp_sentences,
) -> Optional[float]:
    """
    Sentence-level SBERT coherence for one document.
    Returns mean cosine similarity between consecutive sentence embeddings,
    or NaN if the document has fewer than 2 sentences.
    """
    spacy_doc = nlp_sentences(doc_text)
    sents = [s.text.strip() for s in spacy_doc.sents if s.text.strip()]
    if len(sents) < 2:
        return np.nan
    embeddings = model.encode(sents, show_progress_bar=False, normalize_embeddings=True)
    # dot product of L2-normalised vectors = cosine similarity
    sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Parse-tree depth (sentence complexity)
# ---------------------------------------------------------------------------


def _tree_depth(token) -> int:
    """Recursive depth of a dependency parse subtree."""
    if not list(token.children):
        return 0
    return 1 + max(_tree_depth(c) for c in token.children)


def _mean_tree_depth(spacy_doc) -> float:
    """Mean parse-tree depth across sentences."""
    depths = []
    for sent in spacy_doc.sents:
        root = [t for t in sent if t.head == t]
        if root:
            depths.append(_tree_depth(root[0]))
    return float(np.mean(depths)) if depths else np.nan


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def summarize_corpus(
    texts: list[str],
    *,
    spacy_model: str = "en_core_web_lg",
    # optional modules
    include_coherence_sbert: bool = False,
    include_readability: bool = False,
    include_sentence_complexity: bool = False,
    sbert_model_name: str = "all-MiniLM-L6-v2",
    decimals: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a summary table for a list of document strings.

    Parameters
    ----------
    texts : list of str
        The corpus documents.
    spacy_model : str
        spaCy model to use (should match language). 'en_core_web_lg' recommended
        for dependency distance; 'en_core_web_sm' works for most metrics.
    include_coherence_sbert : bool
        If True, compute sentence-level SBERT embedding coherence.
    include_readability : bool
        If True, include Flesch Reading Ease, FK Grade, Gunning Fog,
        Coleman-Liau, ARI, and SMOG.
    include_sentence_complexity : bool
        If True, include dependency distance stats and mean parse tree depth.
    sbert_model_name : str
        Sentence-transformers model for SBERT coherence.
    decimals : int
        Decimal places for formatting.

    Returns
    -------
    tuple of (summary, raw)
        summary : pd.DataFrame with columns ['Metric', 'Value']
        raw     : pd.DataFrame with per-document metrics
    """
    # ------------------------------------------------------------------ #
    # 1.  TextDescriptives extraction
    # ------------------------------------------------------------------ #
    metrics_to_extract = ["descriptive_stats", "pos_proportions"]
    if include_readability:
        metrics_to_extract.append("readability")
    if include_sentence_complexity:
        metrics_to_extract.append("dependency_distance")

    print(f"Extracting TextDescriptives metrics ({', '.join(metrics_to_extract)})…")
    df_td = td.extract_metrics(
        text=texts,
        spacy_model=spacy_model,
        metrics=metrics_to_extract,
    )
    df_td = df_td.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2.  Custom per-document features
    # ------------------------------------------------------------------ #

    # 1st-person pronoun proportion (per-doc: count / n_tokens)
    n_tokens_col = df_td["n_tokens"] if "n_tokens" in df_td.columns else None

    prop_1pp = []
    for i, text in enumerate(texts):
        matches = len(_1ST_PERSON.findall(text))
        nt = df_td.loc[i, "n_tokens"] if n_tokens_col is not None else None
        prop_1pp.append(matches / nt if (nt and nt > 0) else np.nan)
    df_td["prop_1st_person"] = prop_1pp

    # ------------------------------------------------------------------ #
    # 3.  SBERT coherence (optional)
    # ------------------------------------------------------------------ #
    if include_coherence_sbert:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SBERT coherence.\n"
                "Install with: pip install sentence-transformers"
            )
        print(f"Loading SBERT model '{sbert_model_name}'…")
        sbert = SentenceTransformer(sbert_model_name)
        # lightweight spaCy for sentence splitting only
        nlp_sent = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
        print("Computing SBERT coherence per document…")
        df_td["coherence_sbert"] = [
            _sbert_coherence_per_doc(t, sbert, nlp_sent) for t in texts
        ]

    # ------------------------------------------------------------------ #
    # 4.  Parse-tree depth (optional, uses full spaCy doc)
    # ------------------------------------------------------------------ #
    if include_sentence_complexity:
        print("Computing parse-tree depths…")
        nlp_dep = spacy.load(spacy_model)
        tree_depths = [_mean_tree_depth(nlp_dep(t)) for t in texts]
        df_td["mean_tree_depth"] = tree_depths

    # ------------------------------------------------------------------ #
    # 5.  Assemble summary rows
    # ------------------------------------------------------------------ #
    rows = []

    def add(metric_name, value_str):
        rows.append({"Metric": metric_name, "Value": value_str})

    # --- Sample size ---
    add("N (documents)", str(len(texts)))

    # --- Document length (word count / doc_length) ---
    if "doc_length" in df_td.columns:
        add(
            "Document length (words)", _fmt_mean_sd_range(df_td["doc_length"], decimals)
        )

    # --- Sentences ---
    if "n_sentences" in df_td.columns:
        add(
            "Sentences per document", _fmt_mean_sd_range(df_td["n_sentences"], decimals)
        )

    # --- Sentence length ---
    if "sentence_length_mean" in df_td.columns:
        add(
            "Sentence length (words)",
            _fmt_mean_sd_range(df_td["sentence_length_mean"], decimals),
        )

    # --- Tokens ---
    if "n_tokens" in df_td.columns:
        add("Tokens per document", _fmt_mean_sd_range(df_td["n_tokens"], decimals))

    # --- Unique tokens (count + proportion) ---
    if "n_unique_tokens" in df_td.columns and "n_tokens" in df_td.columns:
        add(
            "Unique tokens per document",
            _fmt_mean_sd_range(df_td["n_unique_tokens"], decimals),
        )
        ttr = df_td["n_unique_tokens"] / df_td["n_tokens"].replace(0, np.nan)
        add("Type-token ratio (TTR)", _fmt_mean_sd_range(ttr, decimals))

    # --- Characters ---
    if "n_characters" in df_td.columns:
        add(
            "Characters per document",
            _fmt_mean_sd_range(df_td["n_characters"], decimals),
        )

    # --- Stop words ---
    if "n_stop_words" in df_td.columns and "n_tokens" in df_td.columns:
        add(
            "Stop words per document (N)",
            _fmt_mean_sd_range(df_td["n_stop_words"], decimals),
        )
        prop_sw = df_td["n_stop_words"] / df_td["n_tokens"].replace(0, np.nan)
        add("Stop words (%)", _fmt_pct_n(prop_sw, decimals))

    # --- 1st-person pronouns ---
    add("1st-person pronoun proportion", _fmt_pct_n(df_td["prop_1st_person"], decimals))

    # --- Lexical density (non-stop / total) ---
    if "n_stop_words" in df_td.columns and "n_tokens" in df_td.columns:
        lex_density = (df_td["n_tokens"] - df_td["n_stop_words"]) / df_td[
            "n_tokens"
        ].replace(0, np.nan)
        add("Lexical density (content words %)", _fmt_pct_n(lex_density, decimals))

    # --- Alpha ratio ---
    if "alpha_ratio" in df_td.columns:
        add("Alpha-token ratio", _fmt_mean_sd_range(df_td["alpha_ratio"], decimals))

    # --- POS proportions ---
    for pos, label in [
        ("pos_prop_NOUN", "Proportion NOUN"),
        ("pos_prop_VERB", "Proportion VERB"),
        ("pos_prop_ADJ", "Proportion ADJ"),
        ("pos_prop_ADV", "Proportion ADV"),
        ("pos_prop_PRON", "Proportion PRON"),
    ]:
        if pos in df_td.columns:
            add(label, _fmt_pct_n(df_td[pos], decimals))

    # --- Readability (optional) ---
    if include_readability:
        for col, label in [
            ("flesch_reading_ease", "Flesch Reading Ease"),
            ("flesch_kincaid_grade", "Flesch–Kincaid Grade"),
            ("gunning_fog", "Gunning Fog Index"),
            ("coleman_liau_index", "Coleman–Liau Index"),
            ("automated_readability_index", "Automated Readability Index (ARI)"),
            ("smog", "SMOG"),
        ]:
            if col in df_td.columns:
                add(label, _fmt_mean_sd_range(df_td[col], decimals))

    # --- Sentence complexity (optional) ---
    if include_sentence_complexity:
        for col, label in [
            ("dependency_distance_mean", "Dependency distance (mean)"),
            ("dependency_distance_std", "Dependency distance (SD across sentences)"),
        ]:
            if col in df_td.columns:
                add(label, _fmt_mean_sd_range(df_td[col], decimals))
        if "mean_tree_depth" in df_td.columns:
            add(
                "Parse tree depth (mean)",
                _fmt_mean_sd_range(df_td["mean_tree_depth"], decimals),
            )

    # --- SBERT coherence (optional) ---
    if include_coherence_sbert and "coherence_sbert" in df_td.columns:
        add(
            "Semantic coherence — SBERT (cosine sim.)",
            _fmt_mean_sd_range(df_td["coherence_sbert"], decimals),
        )

    summary = pd.DataFrame(rows)
    return summary, df_td  # also return the raw per-doc df


# ---------------------------------------------------------------------------
# Convenience: print or export
# ---------------------------------------------------------------------------


def print_summary(summary: pd.DataFrame) -> None:
    """Pretty-print the summary table."""
    col_w = max(summary["Metric"].str.len().max() + 2, 45)
    print("\n" + "=" * (col_w + 42))
    print(f"{'Metric':<{col_w}}{'Value'}")
    print("=" * (col_w + 42))
    for _, row in summary.iterrows():
        print(f"{row['Metric']:<{col_w}}{row['Value']}")
    print("=" * (col_w + 42) + "\n")


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_texts = [
        (
            "I really enjoyed the workshop yesterday. We covered a lot of ground "
            "on causal inference, and I think the team learned a great deal. "
            "The facilitators were excellent, and the exercises were practical."
        ),
        (
            "The patient presented with elevated cortisol and disrupted sleep architecture. "
            "We recommended a structured CBT protocol. Follow-up is scheduled for six weeks. "
            "She expressed ambivalence about medication but agreed to try the intervention."
        ),
        (
            "It's hard to explain. Everything feels heavy. I don't know where to begin. "
            "Maybe things will get better. I keep telling myself that."
        ),
    ]

    summary, raw = summarize_corpus(
        sample_texts,
        spacy_model="en_core_web_sm",  # change to lg for better dep-distance
        include_coherence_sbert=True,
        include_readability=True,
        include_sentence_complexity=True,
        sbert_model_name="all-MiniLM-L6-v2",
    )

    print_summary(summary)

    # Optional: save to CSV
    # summary.to_csv("corpus_summary_table.csv", index=False)
    # raw.to_csv("corpus_per_doc.csv", index=False)
