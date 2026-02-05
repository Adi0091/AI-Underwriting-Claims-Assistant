import json
from pathlib import Path
from pinecone import Pinecone
from rag.vectorstore.pinecone_client import pc, INDEX_NAME

BATCH_SIZE = 100


def index_embeddings(embeddings_path: Path):
    index = pc.Index(INDEX_NAME)

    with open(embeddings_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    vectors = []
    total_indexed = 0

    for record in records:
        vectors.append({
            "id": record["chunk_id"],
            "values": record["embedding"],
            "metadata": {
    "document_type": record["metadata"]["document_type"],
    "source_file": record["metadata"]["source_file"],
    "page_number": record["metadata"]["page_number"],
    "chunk_text": record["chunk_text"]
}
        })

        if len(vectors) >= BATCH_SIZE:
            index.upsert(vectors=vectors)
            total_indexed += len(vectors)
            vectors = []

    # Insert remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
        total_indexed += len(vectors)

    print(f"Total vectors indexed: {total_indexed}")
