"""
ingest.py — PDF ingestion, chunking, and embedding storage.
Run once per textbook. Saves a FAISS index + metadata to disk.
"""

import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    import pypdf as PyPDF2

from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400        # words per chunk (keeps chunks small → cheaper retrieval)
CHUNK_OVERLAP = 50      # word overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"   # tiny, fast, runs offline
INDEX_DIR = Path("index")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF."""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_pages(pages: list[dict], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    """Split pages into overlapping word-level chunks."""
    chunks = []
    buffer_words = []
    buffer_meta = []   # (page_num, word_index)

    for page in pages:
        words = page["text"].split()
        for w in words:
            buffer_words.append(w)
            buffer_meta.append(page["page"])

    i = 0
    chunk_id = 0
    while i < len(buffer_words):
        window = buffer_words[i: i + chunk_size]
        pages_covered = list(set(buffer_meta[i: i + chunk_size]))
        chunks.append({
            "id": chunk_id,
            "text": " ".join(window),
            "pages": sorted(pages_covered),
        })
        chunk_id += 1
        i += chunk_size - overlap   # slide with overlap

    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings, dtype="float32")


def ingest(pdf_path: str, book_name: str = None):
    """Full pipeline: PDF → chunks → embeddings → saved index."""
    import faiss

    pdf_path = Path(pdf_path)
    book_name = book_name or pdf_path.stem
    out_dir = INDEX_DIR / book_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Reading {pdf_path} ...")
    pages = extract_text_from_pdf(str(pdf_path))
    print(f"[ingest] Extracted {len(pages)} pages.")

    chunks = chunk_pages(pages)
    print(f"[ingest] Created {len(chunks)} chunks.")

    print(f"[ingest] Embedding with {EMBED_MODEL} (runs locally, no API cost) ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(chunks, model)

    # Build FAISS flat index (exact search, good for <100k chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product = cosine on normalized vecs
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Persist
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "book": book_name,
        "pdf": str(pdf_path),
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "embed_model": EMBED_MODEL,
        "checksum": hashlib.md5(pdf_path.read_bytes()).hexdigest(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ingest] Done. Index saved to {out_dir}/")
    return out_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/textbook.pdf> [book_name]")
        sys.exit(1)
    pdf = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else None
    ingest(pdf, name)
