from genai.prompts.rag_prompt import build_rag_prompt
from genai.llm.client import generate_answer
from genai.guardrails.answer_validator import validate_answer


def run_rag(question: str, retrieved_chunks: list) -> dict:
    if not retrieved_chunks:
        return {
            "answer": "Not found in the provided documents.",
            "confidence": 0.0,
            "sources": [],
            "status": "rejected"
        }

    prompt = build_rag_prompt(question, retrieved_chunks)
    answer = generate_answer(prompt)

    validation = validate_answer(answer, retrieved_chunks)

    if validation["status"] == "rejected":
        return {
            "answer": "Not found in the provided documents.",
            "confidence": validation["confidence"],
            "sources": [],
            "status": "rejected"
        }

    sources = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        sources.append({
            "source": meta.get("source_file"),
            "page": meta.get("page_number")
        })

    return {
        "answer": answer,
        "confidence": validation["confidence"],
        "sources": sources,
        "status": "accepted"
    }
