
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load embedding model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embeddings(input_path: Path, output_path: Path):
    """
    Reads chunked data, generates embeddings for each chunk,
    and saves the result with metadata.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedded_chunks = []
    skipped_chunks = 0

    for chunk in chunks:
        text = chunk.get("chunk_text") or chunk.get("text")

        # Skip empty or invalid chunks
        if not text or not text.strip():
            skipped_chunks += 1
            print(f"[SKIP] Empty text in chunk_id={chunk.get('chunk_id')}")
            continue

        try:
            embedding = model.encode(text, convert_to_numpy=True)
        except Exception as e:
            skipped_chunks += 1
            print(f"[ERROR] Embedding failed for chunk_id={chunk.get('chunk_id')}: {e}")
            continue

        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_text": text,
            "embedding": embedding.tolist(),  # convert to JSON-serializable list
            "metadata": {
                "document_type": chunk.get("document_type"),
                "source_file": chunk.get("source_file"),
                "page_number": chunk.get("page_number")
            }
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)

    print(f"Total chunks processed: {len(chunks)}")
    print(f"Total embeddings generated: {len(embedded_chunks)}")
    print(f"Total chunks skipped: {skipped_chunks}")

    return embedded_chunks
