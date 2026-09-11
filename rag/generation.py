"""
Grounded generation for the Nykaa support agent.

Task 4 responsibilities:
1. Retrieve relevant knowledge-base chunks.
2. Calibrate the similarity threshold empirically.
3. Generate answers using retrieved evidence only.
4. Fall back when evidence is insufficient.
5. Demonstrate grounded answers for in-scope queries.
6. Demonstrate fallback for an out-of-scope query.

This module does not call an external LLM.
Generation is deterministic and uses retrieved KB sentences only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .retrieval import RetrievedChunk, Retriever


DEFAULT_SIMILARITY_THRESHOLD = 0.45
DEFAULT_TOP_K = 5

FALLBACK_MESSAGE = (
    "I don't know based on the available Nykaa support knowledge base."
)


@dataclass(frozen=True)
class GenerationResult:
    """Result returned by grounded generation."""

    query: str
    answer: str
    grounded: bool
    fallback: bool
    threshold: float
    top_similarity: float | None
    retrieved_chunks: tuple[RetrievedChunk, ...]


def _extract_relevant_sentences(
    query: str,
    chunks: Iterable[RetrievedChunk],
    max_sentences: int = 3,
) -> list[str]:
    """
    Select deterministic evidence sentences from retrieved chunks.

    The selection uses lexical overlap between query terms and sentences.
    No external LLM is used.
    """

    query_terms = {
        word.lower().strip(".,?!:;()[]{}\"'")
        for word in query.split()
        if len(word.strip(".,?!:;()[]{}\"'")) >= 3
    }

    candidates: list[tuple[int, float, str]] = []

    for chunk in chunks:
        sentences = _split_sentences(chunk.text)

        for sentence_index, sentence in enumerate(sentences):
            sentence_terms = {
                word.lower().strip(".,?!:;()[]{}\"'")
                for word in sentence.split()
                if len(word.strip(".,?!:;()[]{}\"'")) >= 3
            }

            overlap = query_terms.intersection(sentence_terms)

            if overlap:
                score = len(overlap) / max(len(query_terms), 1)

                candidates.append(
                    (
                        len(overlap),
                        score,
                        sentence.strip(),
                    )
                )

    # Stable deterministic ordering:
    # more matching terms first, then higher overlap ratio,
    # then alphabetical sentence order.
    candidates.sort(
        key=lambda item: (-item[0], -item[1], item[2].lower())
    )

    selected: list[str] = []

    for _, _, sentence in candidates:
        if sentence not in selected:
            selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    return selected


def _split_sentences(text: str) -> list[str]:
    """Deterministically split text into sentences."""

    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _mock_generate(
    query: str,
    chunks: Iterable[RetrievedChunk],
) -> str:
    """
    Deterministic grounded response.

    The answer is constructed only from retrieved KB evidence.
    """

    sentences = _extract_relevant_sentences(
        query=query,
        chunks=chunks,
        max_sentences=3,
    )

    if not sentences:
        return FALLBACK_MESSAGE

    return " ".join(sentences)


class GroundedGenerator:
    """
    Generate answers only when retrieval evidence clears the threshold.
    """

    def __init__(
        self,
        retriever: Retriever,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        mock_llm: Any | None = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0 and 1."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.mock_llm = mock_llm

    def generate(self, query: str) -> GenerationResult:
        """Retrieve evidence and generate a grounded answer."""

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        retrieved = self.retriever.retrieve(
            query,
            top_k=self.top_k,
        )

        top_similarity = (
            retrieved[0].similarity
            if retrieved
            else None
        )

        if (
            top_similarity is None
            or top_similarity < self.similarity_threshold
        ):
            return GenerationResult(
                query=query,
                answer=FALLBACK_MESSAGE,
                grounded=False,
                fallback=True,
                threshold=self.similarity_threshold,
                top_similarity=top_similarity,
                retrieved_chunks=tuple(retrieved),
            )

        if self.mock_llm is not None:
            raise NotImplementedError(
                "External/mock LLM integration is not enabled in "
                "Task 4 generation. Use deterministic grounded "
                "generation from retrieved evidence."
            )

        answer = _mock_generate(
            query=query,
            chunks=retrieved,
        )

        return GenerationResult(
            query=query,
            answer=answer,
            grounded=True,
            fallback=False,
            threshold=self.similarity_threshold,
            top_similarity=top_similarity,
            retrieved_chunks=tuple(retrieved),
        )


def calibrate_threshold(
    retriever: Retriever,
    in_scope_queries: Iterable[str],
    out_of_scope_queries: Iterable[str],
) -> dict[str, Any]:
    """
    Empirically calibrate a similarity threshold.

    Requirements:
    - At least 3 in-scope queries.
    - At least 2 out-of-scope queries.

    If the observed score ranges are separable, the proposed
    threshold is the midpoint between the lowest in-scope score
    and the highest out-of-scope score.

    If they overlap, the function reports the overlap and uses
    a conservative midpoint of the observed overall range.
    """

    in_scope = list(in_scope_queries)
    out_of_scope = list(out_of_scope_queries)

    if len(in_scope) < 3:
        raise ValueError(
            "Threshold calibration requires at least "
            "3 in-scope queries."
        )

    if len(out_of_scope) < 2:
        raise ValueError(
            "Threshold calibration requires at least "
            "2 out-of-scope queries."
        )

    in_scope_scores: list[float] = []
    out_of_scope_scores: list[float] = []

    for query in in_scope:
        score = retriever.top_similarity(query)

        if score is None:
            raise RuntimeError(
                f"No retrieval score was returned for "
                f"in-scope query: {query!r}"
            )

        in_scope_scores.append(score)

    for query in out_of_scope:
        score = retriever.top_similarity(query)

        if score is None:
            raise RuntimeError(
                f"No retrieval score was returned for "
                f"out-of-scope query: {query!r}"
            )

        out_of_scope_scores.append(score)

    min_in_scope = min(in_scope_scores)
    max_out_of_scope = max(out_of_scope_scores)

    separable = max_out_of_scope < min_in_scope

    if separable:
        proposed_threshold = (
            min_in_scope + max_out_of_scope
        ) / 2
    else:
        overall_min = min(
            min_in_scope,
            min(out_of_scope_scores),
        )
        overall_max = max(
            max_out_of_scope,
            max(in_scope_scores),
        )

        proposed_threshold = (
            overall_min + overall_max
        ) / 2

    return {
        "in_scope_queries": in_scope,
        "out_of_scope_queries": out_of_scope,
        "in_scope_scores": in_scope_scores,
        "out_of_scope_scores": out_of_scope_scores,
        "min_in_scope_score": min_in_scope,
        "max_out_of_scope_score": max_out_of_scope,
        "separable": separable,
        "proposed_threshold": proposed_threshold,
    }


def _print_calibration(
    strategy: str,
    calibration: dict[str, Any],
) -> None:
    """Print calibration results in a transcript-friendly format."""

    print(f"\nStrategy: {strategy}")
    print("-" * 70)

    print("In-scope calibration scores:")

    for query, score in zip(
        calibration["in_scope_queries"],
        calibration["in_scope_scores"],
    ):
        print(f"  {score:.4f}  {query}")

    print("\nOut-of-scope calibration scores:")

    for query, score in zip(
        calibration["out_of_scope_queries"],
        calibration["out_of_scope_scores"],
    ):
        print(f"  {score:.4f}  {query}")

    print(
        f"\nLowest in-scope score:       "
        f"{calibration['min_in_scope_score']:.4f}"
    )

    print(
        f"Highest out-of-scope score:  "
        f"{calibration['max_out_of_scope_score']:.4f}"
    )

    print(
        f"Scores separable:            "
        f"{calibration['separable']}"
    )

    print(
        f"Proposed threshold:          "
        f"{calibration['proposed_threshold']:.4f}"
    )


def _run_generation_demo(
    strategy: str,
    retriever: Retriever,
    threshold: float,
    queries: list[str],
    fallback_query: str,
) -> None:
    """Run the required grounded-generation demonstrations."""

    print(f"\n{'=' * 70}")
    print(f"Grounded generation demonstration: {strategy}")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 70)

    generator = GroundedGenerator(
        retriever=retriever,
        similarity_threshold=threshold,
        top_k=DEFAULT_TOP_K,
    )

    for number, query in enumerate(queries, start=1):
        result = generator.generate(query)

        print(f"\n[{number}] Query:")
        print(f"    {query}")

        print(
            f"    Top similarity: "
            f"{result.top_similarity:.4f}"
            if result.top_similarity is not None
            else "    Top similarity: None"
        )

        print(f"    Grounded: {result.grounded}")
        print(f"    Fallback: {result.fallback}")
        print(f"    Answer: {result.answer}")

    print("\n[Fallback test] Query:")
    print(f"    {fallback_query}")

    result = generator.generate(fallback_query)

    print(
        f"    Top similarity: "
        f"{result.top_similarity:.4f}"
        if result.top_similarity is not None
        else "    Top similarity: None"
    )

    print(f"    Grounded: {result.grounded}")
    print(f"    Fallback: {result.fallback}")
    print(f"    Answer: {result.answer}")


def main() -> None:
    """
    Execute Task 4.

    Uses the shared evaluation queries from evaluation/queries.py.
    """

    from evaluation.queries import (
    FALLBACK_QUERY,
    CALIBRATION_IN_SCOPE_QUERIES,
    CALIBRATION_OUT_OF_SCOPE_QUERIES,
    IN_SCOPE_QUERY_TEXTS,
    )

    print("Task 4: Grounded Generation and Threshold Calibration")
    print("=" * 70)

    strategies = (
        "fixed_size",
        "sentence",
    )

    for strategy in strategies:
        print(f"\n{'#' * 70}")
        print(f"CALIBRATION: {strategy}")
        print("#" * 70)

        retriever = Retriever(
            strategy=strategy,
            top_k=DEFAULT_TOP_K,
        )

        calibration = calibrate_threshold(
            retriever=retriever,
            in_scope_queries=CALIBRATION_IN_SCOPE_QUERIES,
            out_of_scope_queries=CALIBRATION_OUT_OF_SCOPE_QUERIES,
        )

        _print_calibration(
            strategy=strategy,
            calibration=calibration,
        )

        _run_generation_demo(
            strategy=strategy,
            retriever=retriever,
            threshold=calibration["proposed_threshold"],
            queries=IN_SCOPE_QUERY_TEXTS,
            fallback_query=FALLBACK_QUERY,
        )

    print(f"\n{'=' * 70}")
    print("Task 4 execution completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()