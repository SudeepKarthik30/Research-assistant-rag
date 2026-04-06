"""
eval/run_eval.py — Retrieval evaluation for the hybrid RAG pipeline.

Usage:
    1. Index at least one paper via the app first.
    2. Edit eval/test_set.json with your own questions from that paper.
    3. Run:  python eval/run_eval.py

Metrics:
    - Retrieval P@3  : Was the correct page in the top-3 retrieved chunks?
    - Keyword Hit Rate: Did expected keywords appear in the final answer?

Results are printed to the console and saved to eval/results.json.
"""

import json
import os
import sys

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_pipeline import RAGPipeline


def run_eval(test_file: str = "eval/test_set.json", top_k: int = 3) -> None:
    if not os.path.exists(test_file):
        print(f"[Eval] Test file not found: {test_file}")
        print("[Eval] Create eval/test_set.json first. See eval/test_set.example.json.")
        return

    with open(test_file) as f:
        test_cases = json.load(f)

    if not test_cases:
        print("[Eval] No test cases found.")
        return

    print(f"[Eval] Loading RAG pipeline...")
    rag = RAGPipeline()

    if rag.doc_count == 0:
        print("[Eval] No papers indexed. Upload a paper via the app first.")
        return

    print(f"[Eval] Running {len(test_cases)} test cases (top_k={top_k})...\n")

    retrieval_hits = 0
    keyword_scores = []
    results        = []

    for i, case in enumerate(test_cases, start=1):
        question         = case["question"]
        expected_page    = case.get("relevant_page")
        expected_section = case.get("relevant_section", "").lower()
        keywords         = case.get("expected_keywords", [])

        # ── Retrieval check ──────────────────────────────────────────────────
        chunks = rag.retrieve(question, top_k=top_k)
        retrieved_pages    = [c["metadata"]["page"]            for c in chunks]
        retrieved_sections = [c["metadata"].get("section", "").lower() for c in chunks]

        page_hit    = (expected_page in retrieved_pages)    if expected_page    else None
        section_hit = (expected_section in retrieved_sections) if expected_section else None

        if page_hit:
            retrieval_hits += 1

        # ── Answer keyword check ─────────────────────────────────────────────
        kw_score = 0.0
        answer   = ""
        if keywords:
            result = rag.answer(question, top_k=top_k)
            answer = result["answer"].lower()
            matched  = sum(1 for kw in keywords if kw.lower() in answer)
            kw_score = matched / len(keywords)
            keyword_scores.append(kw_score)

        results.append({
            "question":      question,
            "page_hit":      page_hit,
            "section_hit":   section_hit,
            "keyword_score": round(kw_score, 2),
            "pages_retrieved": retrieved_pages,
        })

        status = "✓" if page_hit else ("~" if page_hit is None else "✗")
        print(f"  [{i:02d}] {status} | pages={retrieved_pages} | kw={kw_score:.0%} | {question[:60]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n             = len(test_cases)
    page_cases    = [c for c in test_cases if c.get("relevant_page")]
    retrieval_pct = retrieval_hits / len(page_cases) if page_cases else 0
    keyword_pct   = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0

    print(f"\n{'─'*50}")
    print(f"  Retrieval P@{top_k}  : {retrieval_hits}/{len(page_cases)} = {retrieval_pct:.0%}")
    print(f"  Keyword hit rate: {keyword_pct:.0%}")
    print(f"  Test cases run  : {n}")
    print(f"{'─'*50}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "summary": {
            f"retrieval_p_at_{top_k}": f"{retrieval_pct:.0%}",
            "keyword_hit_rate":        f"{keyword_pct:.0%}",
            "test_cases":              n,
        },
        "details": results,
    }
    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Eval] Results saved to eval/results.json")


if __name__ == "__main__":
    run_eval()
