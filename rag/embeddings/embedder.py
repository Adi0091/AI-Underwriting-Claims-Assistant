import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer


# Load the embedding model once (important for performance)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    input_path: Path,
    output_path: Path
) -> List[Dict]:
    """
    Generate embeddings for chunked text data.

    Args:
        input_path: Path to chunked JSON file (from data/chunks/)
        output_path: Path to save embeddings JSON (to data/embeddings/)

    Returns:
        List of embedded chunk records
    """

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedded_chunks = []
    skipped = 0

    for chunk in chunks:
        # Support both keys safely
        text = chunk.get("chunk_text") or chunk.get("text")

        # Skip empty or invalid chunks
        if not text or not text.strip():
            skipped += 1
            print(f"[SKIP] Empty text | chunk_id={chunk.get('chunk_id')}")
            continue

        try:
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
        except Exception as e:
            skipped += 1
            print(f"[ERROR] Embedding failed | chunk_id={chunk.get('chunk_id')} | {e}")
            continue

        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_text": text,
            "embedding": embedding.tolist(),  # JSON serializable
            "metadata": {
                "document_type": chunk.get("document_type"),
                "source_file": chunk.get("source_file"),
                "page_number": chunk.get("page_number")
            }
        })

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)

    print("Embedding generation completed")
    print(f"Total input chunks   : {len(chunks)}")
    print(f"Embeddings generated : {len(embedded_chunks)}")
    print(f"Chunks skipped       : {skipped}")
    print(f"Embedding dimension  : {len(embedded_chunks[0]['embedding']) if embedded_chunks else 'N/A'}")

    return embedded_chunks
