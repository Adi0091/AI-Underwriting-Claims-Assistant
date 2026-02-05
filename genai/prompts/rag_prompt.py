def build_rag_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Build a RAG prompt using retrieved chunks.
    """

    if not retrieved_chunks:
        return None

    context_blocks = []

    for chunk in retrieved_chunks:
        text = chunk.get("chunk_text") or ""
        metadata = chunk.get("metadata", {})

        source = metadata.get(r"C:\Users\Admin\Desktop\ai-underwriting-claims-assistant\data\chunks\policies\general_liability_policy_chunks.json", "unknown")
        page = metadata.get("page_number", "N/A")

        context_blocks.append(
            f"{text}\n(Source: {source}, Page: {page})"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
SYSTEM:
You are an AI assistant helping with liability insurance underwriting and claims triage.
You must answer strictly using the provided context.
If the answer is not present in the context, say:
"Not found in the provided documents."

CONTEXT:
<<<
{context_text}
>>>

QUESTION:
{question}

INSTRUCTIONS:
- Use only the context above
- Be concise and factual
- Cite sources with page numbers
"""

    return prompt.strip()
