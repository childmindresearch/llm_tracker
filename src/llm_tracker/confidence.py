"""Confidence conversion shared by quote matching and presence comparison."""

from collections.abc import Mapping

from llm_tracker.models import AnalysisResult


def validate_rounding(rounding_method: str, binarize_confidence: bool) -> None:
    """Accept a single threshold policy; validate even when conversion is off."""
    if rounding_method not in ("floor", "ceiling"):
        raise ValueError(
            "rounding_method must be 'floor' or 'ceiling'; 'both' is not supported."
        )
    if not isinstance(binarize_confidence, bool):
        raise TypeError("binarize_confidence must be True or False.")


def comparison_confidence(
    score: int | None, *, rounding_method: str, binarize_confidence: bool
) -> int:
    """Round an ordinal score, or retain an already scored input unchanged.

    With conversion off, 0 stays absent and positive values stay detections;
    original 2s are also retained for compatibility with unrounded comparisons.
    Missing LLM scores are ambiguous and raise rather than inventing a detection.
    Human instances never pass through this function.
    """
    validate_rounding(rounding_method, binarize_confidence)
    if isinstance(score, bool) or score not in (0, 1, 2):
        raise ValueError(f"LLM confidence must be 0, 1, or 2, got {score!r}.")
    if not binarize_confidence:
        return score
    return int(score >= (2 if rounding_method == "floor" else 1))


def prepare_llm_results(
    results: Mapping[str, AnalysisResult],
    *,
    rounding_method: str = "floor",
    binarize_confidence: bool = True,
    drop_zeros: bool = False,
):
    """Return independent copies; preserve document rosters and source provenance.

    drop_zeros=True is only for quote matching. Presence keeps explicit zero
    records but interprets them as absence. Originals are never changed, so
    repeated floor/ceiling calls cannot re-round or destroy encoding results.
    """
    from llm_tracker.outputs import RunResults

    validate_rounding(rounding_method, binarize_confidence)
    prepared = RunResults(output_path=getattr(results, "output_path", None))
    for doc_id, result in results.items():
        if not isinstance(result, AnalysisResult):
            raise TypeError(f"LLM result for {doc_id!r} must be an AnalysisResult.")
        instances = []
        for instance in result.instances:
            try:
                score = comparison_confidence(
                    instance.confidence,
                    rounding_method=rounding_method,
                    binarize_confidence=binarize_confidence,
                )
            except ValueError as error:
                raise ValueError(
                    f"Document {doc_id!r}, construct {instance.construct!r}: {error}"
                ) from error
            if drop_zeros and score == 0:
                continue
            copied = instance.model_copy(deep=True)
            copied.confidence = score
            instances.append(copied)
        copied_result = result.model_copy(deep=True)
        copied_result.instances = instances
        prepared[doc_id] = copied_result
    return prepared
