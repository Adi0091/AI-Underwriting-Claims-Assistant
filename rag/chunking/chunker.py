import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(input_path: Path, output_path: Path, chunk_size: int = 800, chunk_overlap: int = 120):
    
    # Initialize recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = []
    chunk_id = 0

    for page in pages:
        splits = splitter.split_text(page["text"])

        for split in splits:
            chunk_id += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "text": split,
                "document_type": page["document_type"],
                "source_file": page["source_file"],
                "page_number": page.get("page_number")
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return chunks
