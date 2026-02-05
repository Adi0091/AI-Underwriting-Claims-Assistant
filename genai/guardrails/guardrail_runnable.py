from langchain_core.runnables import RunnableLambda
from genai.guardrails.answer_validator import validate_answer


def guardrail_fn(inputs: dict):
    """
    inputs = {
        "answer": str,
        "chunks": list
    }
    """
    answer = inputs["answer"]
    chunks = inputs["chunks"]

    validation = validate_answer(answer, chunks)

    if validation["status"] == "rejected":
        return {
            "answer": "Not found in the provided documents.",
            "confidence": validation["confidence"],
            "status": "rejected",
            "sources": []
        }

    sources = []
    for c in chunks:
        meta = c.get("metadata", {})
        sources.append({
            "source": meta.get("source_file"),
            "page": meta.get("page_number")
        })

    return {
        "answer": answer,
        "confidence": validation["confidence"],
        "status": "accepted",
        "sources": sources
    }


guardrail_runnable = RunnableLambda(guardrail_fn)
