import json
from rag.retrieval.retriever import retrieve_chunks
from rag.retrieval.rag_pipeline import run_rag


def evaluate_rag(test_cases_path: str):
    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        question = case["question"]

        retrieved = retrieve_chunks(
            query=question,
            top_k=5,
            metadata_filter={"document_type": "policy"}
        )

        rag_result = run_rag(question, retrieved)

        results.append({
            "question": question,
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "confidence": rag_result["confidence"]
        })

    return results
