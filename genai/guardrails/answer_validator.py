from typing import List, Dict


GENERIC_PHRASES = [
    "generally",
    "typically",
    "it depends",
    "in most cases",
    "usually"
]


def is_empty_answer(answer: str) -> bool:
    return not answer or not answer.strip()


def is_overly_generic(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in GENERIC_PHRASES)


def context_overlap(answer: str, retrieved_chunks: List[Dict]) -> bool:
    """
    Simple faithfulness check:
    Does the answer share keywords with retrieved context?
    """
    answer_tokens = set(answer.lower().split())

    for chunk in retrieved_chunks:
        text = (
            chunk.get("chunk_text")
            or chunk.get("metadata", {}).get("chunk_text", "")
        ).lower()

        context_tokens = set(text.split())
        if len(answer_tokens & context_tokens) > 3:
            return True

    return False


def validate_answer(
    answer: str,
    retrieved_chunks: List[Dict]
) -> Dict:
    """
    Validate LLM answer and assign confidence.
    """

    if is_empty_answer(answer):
        return {
            "status": "rejected",
            "reason": "empty_answer",
            "confidence": 0.0
        }

    confidence = 1.0

    if is_overly_generic(answer):
        confidence -= 0.4

    if not context_overlap(answer, retrieved_chunks):
        confidence -= 0.5

    # Length-based sanity check
    if len(answer) < 30:
        confidence -= 0.3
    elif len(answer) > 1000:
        confidence -= 0.2

    confidence = max(confidence, 0.0)

    if confidence < 0.5:
        return {
            "status": "rejected",
            "reason": "low_confidence",
            "confidence": confidence
        }

    return {
        "status": "accepted",
        "confidence": confidence
    }
