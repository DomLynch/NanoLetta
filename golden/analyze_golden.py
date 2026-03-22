"""
Analyze golden evaluation corpus from Hindsight.

Scores:
1. Retain quality — fact extraction accuracy, entity resolution, dedup
2. Recall quality — retrieval relevance, ranking, false positives
3. Reflect quality — synthesis accuracy, grounding, hallucination

Usage:
    python3 golden/analyze_golden.py
"""

import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent


def load_corpus() -> dict:
    path = GOLDEN_DIR / "golden_corpus.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_golden_eval.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def analyze_retain(corpus: dict) -> dict:
    """Analyze retain quality."""
    retains = corpus.get("retain", [])
    if not retains:
        return {"error": "no retain data"}

    total = len(retains)
    successful = [r for r in retains if r.get("success")]
    failed = [r for r in retains if not r.get("success")]

    total_tokens = sum(r.get("token_usage", {}).get("total", 0) for r in successful)
    total_facts = sum(r.get("items_count", 0) for r in successful)
    total_time = sum(r.get("elapsed_seconds", 0) for r in retains)
    avg_time = total_time / total if total > 0 else 0
    avg_facts_per_input = total_facts / len(successful) if successful else 0
    avg_tokens_per_input = total_tokens / len(successful) if successful else 0

    # Group by category
    by_category = {}
    for r in retains:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "success": 0, "facts": 0, "tokens": 0, "time": 0}
        by_category[cat]["total"] += 1
        if r.get("success"):
            by_category[cat]["success"] += 1
            by_category[cat]["facts"] += r.get("items_count", 0)
            by_category[cat]["tokens"] += r.get("token_usage", {}).get("total", 0)
        by_category[cat]["time"] += r.get("elapsed_seconds", 0)

    return {
        "total_scenarios": total,
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [r["id"] for r in failed],
        "total_facts_extracted": total_facts,
        "total_tokens_used": total_tokens,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_scenario": round(avg_time, 1),
        "avg_facts_per_input": round(avg_facts_per_input, 1),
        "avg_tokens_per_input": round(avg_tokens_per_input, 0),
        "by_category": by_category,
    }


def analyze_recall(corpus: dict) -> dict:
    """Analyze recall quality."""
    recalls = corpus.get("recall", [])
    if not recalls:
        return {"error": "no recall data"}

    total = len(recalls)
    successful = [r for r in recalls if not r.get("error")]
    failed = [r for r in recalls if r.get("error")]

    total_time = sum(r.get("elapsed_seconds", 0) for r in recalls)
    avg_results = sum(r.get("result_count", 0) for r in successful) / len(successful) if successful else 0

    # Empty results (possible quality issue)
    empty = [r for r in successful if r.get("result_count", 0) == 0]

    # Check for duplicates in results
    dedup_issues = []
    for r in successful:
        texts = [res.get("text", "") for res in r.get("results", [])]
        unique_texts = set(texts)
        if len(texts) != len(unique_texts):
            dedup_issues.append({
                "id": r["id"],
                "query": r["query"],
                "total": len(texts),
                "unique": len(unique_texts),
                "duplicates": len(texts) - len(unique_texts),
            })

    return {
        "total_queries": total,
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [r["id"] for r in failed],
        "empty_results": [r["id"] for r in empty],
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_query": round(total_time / total if total else 0, 2),
        "avg_results_per_query": round(avg_results, 1),
        "dedup_issues": dedup_issues,
        "dedup_issue_count": len(dedup_issues),
    }


def analyze_reflect(corpus: dict) -> dict:
    """Analyze reflect quality."""
    reflects = corpus.get("reflect", [])
    if not reflects:
        return {"error": "no reflect data"}

    total = len(reflects)
    successful = [r for r in reflects if not r.get("error")]
    failed = [r for r in reflects if r.get("error")]

    total_time = sum(r.get("elapsed_seconds", 0) for r in reflects)
    avg_length = sum(len(r.get("text", "")) for r in successful) / len(successful) if successful else 0
    avg_based_on = sum(r.get("based_on_count", 0) for r in successful) / len(successful) if successful else 0

    # Check for empty reflections
    empty = [r for r in successful if not r.get("text")]

    return {
        "total_queries": total,
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [r["id"] for r in failed],
        "empty_reflections": [r["id"] for r in empty],
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_query": round(total_time / total if total else 0, 1),
        "avg_response_length_chars": round(avg_length, 0),
        "avg_facts_cited": round(avg_based_on, 1),
    }


def main():
    corpus = load_corpus()
    if not corpus:
        return

    print("=" * 60)
    print("GOLDEN CORPUS ANALYSIS")
    print("=" * 60)

    # Retain
    retain = analyze_retain(corpus)
    print("\n--- RETAIN QUALITY ---")
    print(f"  Success rate: {retain['successful']}/{retain['total_scenarios']}")
    print(f"  Total facts extracted: {retain['total_facts_extracted']}")
    print(f"  Avg facts per input: {retain['avg_facts_per_input']}")
    print(f"  Avg tokens per input: {retain['avg_tokens_per_input']}")
    print(f"  Avg time per scenario: {retain['avg_time_per_scenario']}s")
    print(f"  Total time: {retain['total_time_seconds']}s")
    if retain.get("failed_ids"):
        print(f"  FAILED: {retain['failed_ids']}")
    print(f"  By category:")
    for cat, stats in retain.get("by_category", {}).items():
        print(f"    {cat}: {stats['success']}/{stats['total']} OK, {stats['facts']} facts, {stats['tokens']} tokens")

    # Recall
    recall = analyze_recall(corpus)
    print("\n--- RECALL QUALITY ---")
    print(f"  Success rate: {recall['successful']}/{recall['total_queries']}")
    print(f"  Avg results per query: {recall['avg_results_per_query']}")
    print(f"  Avg time per query: {recall['avg_time_per_query']}s")
    print(f"  Empty results: {recall['empty_results'] or 'none'}")
    print(f"  Dedup issues: {recall['dedup_issue_count']}")
    if recall.get("dedup_issues"):
        for d in recall["dedup_issues"][:5]:
            print(f"    {d['id']}: {d['duplicates']} duplicates in {d['total']} results")

    # Reflect
    reflect = analyze_reflect(corpus)
    print("\n--- REFLECT QUALITY ---")
    print(f"  Success rate: {reflect['successful']}/{reflect['total_queries']}")
    print(f"  Avg response length: {reflect['avg_response_length_chars']} chars")
    print(f"  Avg facts cited: {reflect['avg_facts_cited']}")
    print(f"  Avg time per query: {reflect['avg_time_per_query']}s")
    if reflect.get("failed_ids"):
        print(f"  FAILED: {reflect['failed_ids']}")

    # Save analysis
    analysis = {
        "retain": retain,
        "recall": recall,
        "reflect": reflect,
    }
    out_path = GOLDEN_DIR / "golden_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
