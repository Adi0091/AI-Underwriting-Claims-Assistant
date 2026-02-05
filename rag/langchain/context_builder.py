from langchain_core.runnables import RunnableLambda


def build_context(inputs: dict):
    """
    inputs = {
        "query": str,
        "chunks": list
    }
    """
    if not inputs["chunks"]:
        return {
            "query": inputs["query"],
            "context": None,
            "chunks": []
        }

    context_blocks = []

    for chunk in inputs["chunks"]:
        meta = chunk.get("metadata", {})
        text = meta.get("chunk_text", "")
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "N/A")

        context_blocks.append(
            f"{text}\n(Source: {source}, Page: {page})"
        )

    return {
        "query": inputs["query"],
        "context": "\n\n".join(context_blocks),
        "chunks": inputs["chunks"]
    }


context_runnable = RunnableLambda(build_context)
