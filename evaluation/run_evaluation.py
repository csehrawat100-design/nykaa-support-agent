"""Run Task 13 against the existing CrewAI agent."""
from __future__ import annotations
import json
from pathlib import Path
from evaluation.judge import average_scores, evaluate

def run() -> dict:
    # Canonical source from Task 4/5. Do not duplicate the query set here.
    from evaluation.queries import FULL_EVALUATION_QUERIES
    if len(FULL_EVALUATION_QUERIES) != 15:
        raise AssertionError(
            f"Expected 15 canonical queries, found {len(FULL_EVALUATION_QUERIES)}"
        )

    from agents.crew import run_crew
    results = []
    for query in FULL_EVALUATION_QUERIES:
        result = run_crew(query.query)
        answer = getattr(result, "response", str(result))
        results.append(evaluate(query, answer))

    report = {
        "mode": "MOCK_LLM",
        "query_count": len(results),
        "results": results,
        "averages": average_scores(results),
    }
    path = Path("evaluation/results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

if __name__ == "__main__":
    report = run()
    for item in report["results"]:
        print(
            f'{item["query"]}\n'
            f'  Accuracy={item["accuracy"]} '
            f'Grounding={item["grounding"]} '
            f'Completeness={item["completeness"]} '
            f'Safety={item["safety"]}'
        )
    print("\nAVERAGES")
    for name, value in report["averages"].items():
        print(f"{name.title()}: {value}")
