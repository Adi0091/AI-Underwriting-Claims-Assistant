from langchain_core.runnables import RunnableLambda
from rag.retrieval.retriever import retrieve_chunks


def retriever_fn(inputs: dict):
    """
    inputs = {
        "query": str
    }
    """
    query = inputs["query"]

    chunks = retrieve_chunks(
        query=query,
        top_k=5,
        metadata_filter={"document_type": "policy"}
    )

    return {
        "query": query,
        "chunks": chunks
    }


retriever_runnable = RunnableLambda(retriever_fn)
