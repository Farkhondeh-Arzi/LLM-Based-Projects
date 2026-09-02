from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str       # source file name
    chunk_id: int      # chunk sequence number in document
    page: int | None = None  # page number (if PDF file)


def load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_pdf_file(path: str) -> list[tuple[str, int]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():  # Remove empty pages
            pages.append((text, i + 1))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size should be bigger than overlap")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            last_space = text.rfind(" ", start, end)
            last_newline = text.rfind("\n", start, end)
            cut_point = max(last_space, last_newline)
            if cut_point > start:
                end = cut_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start <= 0 or end == text_len:
            if end == text_len:
                break

    return chunks


def process_document(path: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    source_name = Path(path).name
    all_chunks: list[Chunk] = []

    if path.lower().endswith(".pdf"):
        pages = load_pdf_file(path)
        chunk_id = 0
        for page_text, page_num in pages:
            for piece in chunk_text(page_text, chunk_size, overlap):
                all_chunks.append(Chunk(
                    text=piece,
                    source=source_name,
                    chunk_id=chunk_id,
                    page=page_num,
                ))
                chunk_id += 1
    else:
        text = load_text_file(path)
        for chunk_id, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append(Chunk(
                text=piece,
                source=source_name,
                chunk_id=chunk_id,
                page=None,
            ))

    return all_chunks
