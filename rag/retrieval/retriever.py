from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from rag.vectorstore.pinecone_client import get_index


# Load SAME embedding model used for documents
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def retrieve_chunks(query: str,top_k: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
    """
    Retrieve relevant chunks from Pinecone for a user query.
    """

    # 1. Generate query embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    # 2. Get Pinecone index
    index = get_index()

    # 3. Query Pinecone
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter
    )

    results = []

    for match in response["matches"]:
        results.append({
            "chunk_id": match["id"],
            "score": match["score"],
            "metadata": match["metadata"]
        })

    return results
