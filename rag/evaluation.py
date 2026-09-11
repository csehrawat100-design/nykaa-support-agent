"""
Document-level precision/recall evaluation for the Nykaa RAG core.

Task 5 responsibilities:
1. Evaluate the same five in-scope queries used in Task 4.
2. Evaluate both fixed-size and sentence-based Chroma collections.
3. Map retrieved chunks to parent document IDs.
4. Deduplicate parent documents before scoring.
5. Calculate document-level precision and recall.
6. Compare both chunking strategies.
7. Provide a numbers-based recommendation.

This module does not generate answers.
It evaluates retrieval quality only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
import json

from .retrieval import RetrievedChunk, Retriever


DEFAULT_TOP_K = 5
RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "evaluation"
    / "results.json"
)


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Evaluation result for one query and one chunking strategy."""

    query: str
    strategy: str
    expected_documents: tuple[str, ...]
    retrieved_documents: tuple[str, ...]
    relevant_retrieved_documents: tuple[str, ...]
    precision: float
    recall: float
    true_positives: int
    retrieved_count: int
    expected_count: int


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    """Aggregate evaluation for one retrieval strategy."""

    strategy: str
    results: tuple[QueryEvaluation, ...]
    mean_precision: float
    mean_recall: float


def _deduplicate_document_ids(
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, ...]:
    """
    Map retrieved chunks to parent documents.

    Duplicate chunks belonging to the same parent document are counted
    only once. First-retrieved document order is preserved.
    """

    seen: set[str] = set()
    document_ids: list[str] = []

    for chunk in chunks:
        document_id = chunk.document_id.strip()

        if document_id and document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)

    return tuple(document_ids)


def precision_recall(
    retrieved_documents: Sequence[str],
    expected_documents: Sequence[str],
) -> tuple[float, float, int]:
    """
    Calculate document-level precision and recall.

    Precision:
        relevant retrieved documents / retrieved documents

    Recall:
        relevant retrieved documents / expected relevant documents
    """

    retrieved = set(retrieved_documents)
    expected = set(expected_documents)

    true_positives = len(retrieved & expected)

    precision = (
        true_positives / len(retrieved)
        if retrieved
        else 0.0
    )

    recall = (
        true_positives / len(expected)
        if expected
        else 0.0
    )

    return precision, recall, true_positives


def evaluate_query(
    retriever: Retriever,
    query: str,
    expected_documents: Sequence[str],
    *,
    top_k: int | None = None,
) -> QueryEvaluation:
    """Evaluate one query against one chunking strategy."""

    query = query.strip()

    if not query:
        raise ValueError("query must not be empty")

    expected = tuple(
        dict.fromkeys(
            doc.strip()
            for doc in expected_documents
            if doc.strip()
        )
    )

    chunks = retriever.retrieve(
        query,
        top_k=top_k,
    )

    retrieved = _deduplicate_document_ids(chunks)

    precision, recall, true_positives = precision_recall(
        retrieved,
        expected,
    )

    expected_set = set(expected)

    relevant_retrieved_documents = tuple(
        document_id
        for document_id in retrieved
        if document_id in expected_set
    )

    return QueryEvaluation(
        query=query,
        strategy=retriever.strategy,
        expected_documents=expected,
        retrieved_documents=retrieved,
        relevant_retrieved_documents=relevant_retrieved_documents,
        precision=precision,
        recall=recall,
        true_positives=true_positives,
        retrieved_count=len(retrieved),
        expected_count=len(expected),
    )


def evaluate_strategy(
    retriever: Retriever,
    queries: Mapping[str, Sequence[str]],
    *,
    top_k: int | None = None,
) -> StrategyEvaluation:
    """Evaluate all supplied queries for one retrieval strategy."""

    if not queries:
        raise ValueError(
            "queries must contain at least one evaluation query"
        )

    results = tuple(
        evaluate_query(
            retriever,
            query,
            expected_documents,
            top_k=top_k,
        )
        for query, expected_documents in queries.items()
    )

    mean_precision = (
        sum(result.precision for result in results)
        / len(results)
    )

    mean_recall = (
        sum(result.recall for result in results)
        / len(results)
    )

    return StrategyEvaluation(
        strategy=retriever.strategy,
        results=results,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
    )


def compare_strategies(
    fixed_retriever: Retriever,
    sentence_retriever: Retriever,
    queries: Mapping[str, Sequence[str]],
    *,
    top_k: int | None = None,
) -> dict[str, StrategyEvaluation]:
    """Run the same evaluation queries against both collections."""

    if fixed_retriever.strategy != "fixed_size":
        raise ValueError(
            "fixed_retriever must use the 'fixed_size' strategy"
        )

    if sentence_retriever.strategy != "sentence":
        raise ValueError(
            "sentence_retriever must use the 'sentence' strategy"
        )

    return {
        "fixed_size": evaluate_strategy(
            fixed_retriever,
            queries,
            top_k=top_k,
        ),
        "sentence": evaluate_strategy(
            sentence_retriever,
            queries,
            top_k=top_k,
        ),
    }


def recommendation(
    comparison: Mapping[str, StrategyEvaluation],
) -> str:
    """Produce a numbers-based deployment recommendation."""

    fixed = comparison["fixed_size"]
    sentence = comparison["sentence"]

    if (
        sentence.mean_precision > fixed.mean_precision
        and sentence.mean_recall >= fixed.mean_recall
    ):
        winner = "sentence-based"

    elif (
        fixed.mean_precision > sentence.mean_precision
        and fixed.mean_recall >= sentence.mean_recall
    ):
        winner = "fixed-size-with-overlap"

    elif (
        sentence.mean_recall > fixed.mean_recall
        and sentence.mean_precision >= fixed.mean_precision
    ):
        winner = "sentence-based"

    elif (
        fixed.mean_recall > sentence.mean_recall
        and fixed.mean_precision >= sentence.mean_precision
    ):
        winner = "fixed-size-with-overlap"

    else:
        winner = "neither strategy clearly dominates"

    return (
        f"Deploy {winner}. "
        f"Fixed-size mean precision="
        f"{fixed.mean_precision:.3f}, "
        f"mean recall="
        f"{fixed.mean_recall:.3f}; "
        f"sentence-based mean precision="
        f"{sentence.mean_precision:.3f}, "
        f"mean recall="
        f"{sentence.mean_recall:.3f}."
    )


def results_as_dict(
    comparison: Mapping[str, StrategyEvaluation],
) -> dict[str, object]:
    """Convert evaluation objects into JSON-serializable dictionaries."""

    return {
        strategy: {
            "strategy": evaluation.strategy,
            "mean_precision": evaluation.mean_precision,
            "mean_recall": evaluation.mean_recall,
            "results": [
                asdict(result)
                for result in evaluation.results
            ],
        }
        for strategy, evaluation in comparison.items()
    }


def save_results(
    comparison: Mapping[str, StrategyEvaluation],
    path: Path = RESULTS_PATH,
) -> None:
    """Save Task 5 evaluation results to evaluation/results.json."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = results_as_dict(comparison)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def _print_strategy_results(
    evaluation: StrategyEvaluation,
) -> None:
    """Print detailed results for one retrieval strategy."""

    print(f"\nStrategy: {evaluation.strategy}")
    print("-" * 70)

    for number, result in enumerate(
        evaluation.results,
        start=1,
    ):
        print(f"\n[{number}] Query:")
        print(f"    {result.query}")

        print(
            "    Expected documents: "
            f"{result.expected_documents}"
        )

        print(
            "    Retrieved documents: "
            f"{result.retrieved_documents}"
        )

        print(
            "    Relevant retrieved documents: "
            f"{result.relevant_retrieved_documents}"
        )

        print(
            f"    True positives: "
            f"{result.true_positives}"
        )

        print(
            f"    Retrieved document count: "
            f"{result.retrieved_count}"
        )

        print(
            f"    Expected document count: "
            f"{result.expected_count}"
        )

        print(
            f"    Precision: "
            f"{result.precision:.3f}"
        )

        print(
            f"    Recall: "
            f"{result.recall:.3f}"
        )

    print("\nAggregate:")
    print(
        f"    Mean precision: "
        f"{evaluation.mean_precision:.3f}"
    )

    print(
        f"    Mean recall: "
        f"{evaluation.mean_recall:.3f}"
    )


def main() -> None:
    """
    Execute Task 5.

    The same five in-scope queries from evaluation/queries.py
    are evaluated against both retrieval strategies.
    """

    from evaluation.queries import IN_SCOPE_QUERIES

    print("Task 5: Document-Level Precision and Recall")
    print("=" * 70)

    # Convert the shared EvaluationQuery objects into the mapping
    # expected by compare_strategies().
    queries = {
        item.query: item.expected_documents
        for item in IN_SCOPE_QUERIES
    }

    if len(queries) != 5:
        raise RuntimeError(
            "Task 5 requires exactly the same five shared "
            "in-scope queries used for the evaluation."
        )

    print(
        f"Evaluation queries: {len(queries)}"
    )

    print(
        f"Top-k retrieval setting: {DEFAULT_TOP_K}"
    )

    fixed_retriever = Retriever(
        strategy="fixed_size",
        top_k=DEFAULT_TOP_K,
    )

    sentence_retriever = Retriever(
        strategy="sentence",
        top_k=DEFAULT_TOP_K,
    )

    comparison = compare_strategies(
        fixed_retriever=fixed_retriever,
        sentence_retriever=sentence_retriever,
        queries=queries,
        top_k=DEFAULT_TOP_K,
    )

    print("\n" + "=" * 70)
    print("DOCUMENT-LEVEL EVALUATION")
    print("=" * 70)

    _print_strategy_results(
        comparison["fixed_size"]
    )

    _print_strategy_results(
        comparison["sentence"]
    )

    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    fixed = comparison["fixed_size"]
    sentence = comparison["sentence"]

    print(
        f"\nFixed-size-with-overlap:"
        f"\n  Mean precision = {fixed.mean_precision:.3f}"
        f"\n  Mean recall    = {fixed.mean_recall:.3f}"
    )

    print(
        f"\nSentence-based:"
        f"\n  Mean precision = {sentence.mean_precision:.3f}"
        f"\n  Mean recall    = {sentence.mean_recall:.3f}"
    )

    print("\nRecommendation:")
    print(
        f"  {recommendation(comparison)}"
    )

    save_results(comparison)

    print(
        f"\nResults saved to:"
        f"\n  {RESULTS_PATH}"
    )

    print("\n" + "=" * 70)
    print("Task 5 execution completed.")
    print("=" * 70)


__all__ = [
    "QueryEvaluation",
    "StrategyEvaluation",
    "compare_strategies",
    "evaluate_query",
    "evaluate_strategy",
    "precision_recall",
    "recommendation",
    "results_as_dict",
    "save_results",
]


if __name__ == "__main__":
    main()