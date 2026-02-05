import re

def normalize_text(text: str) -> str:
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common page number patterns
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    return text.strip()

def clean_pages(pages):
    seen_texts = set()
    cleaned_pages = []

    for page in pages:
        cleaned_text = normalize_text(page["text"])

        # Safe deduplication (exact match only)
        if cleaned_text in seen_texts:
            continue

        seen_texts.add(cleaned_text)

        cleaned_pages.append({
            "document_type": page["document_type"],
            "source_file": page["source_file"],
            "page_number": page["page_number"],
            "text": cleaned_text
        })

    return cleaned_pages
