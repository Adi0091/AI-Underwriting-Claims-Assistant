from pathlib import Path
from pypdf import PdfReader

def extract_pdf_text(pdf_path: Path, document_type: str):
    reader = PdfReader(pdf_path)
    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue  # likely scanned PDF

        pages_data.append({
            "document_type": document_type,
            "source_file": pdf_path.name,
            "page_number": page_number,
            "text": text.strip()
        })

    return pages_data

def _extract_txt_text(txt_path: Path, document_type: str):
    pages_data = []

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return pages_data

    pages_data.append({
        "document_type": document_type,
        "source_file": txt_path.name,
        "page_number": 1,  # TXT has no pages
        "text": text
    })

    return pages_data