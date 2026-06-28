"""Document loaders for RAG.

Supports PDF, plain text (.txt / .md), and Word (.docx).
Text is split into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter so chunk boundaries fall on sentence/paragraph
edges rather than mid-word.
"""

import re
from pathlib import Path


CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 150    # overlap to preserve context across boundaries


def _split(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_and_chunk(file_path: str, filename: str) -> list[str]:
    """Load a file and return a list of text chunks ready for embedding."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        raw = _load_pdf(file_path)
    elif ext in (".txt", ".md"):
        raw = _load_txt(file_path)
    elif ext == ".docx":
        raw = _load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}")

    return _split(_clean(raw))
