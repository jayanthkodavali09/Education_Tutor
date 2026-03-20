"""
retriever.py — Context Pruning engine.

Given a student query, retrieves only the top-k relevant chunks
from the pre-built FAISS index. This is the core cost-saving mechanism:
instead of sending the full textbook to the LLM, we send ~3-5 chunks.
"""

import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import faiss

INDEX_DIR = Path("index")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Singleton model cache (avoid reloading on every query)
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(name: str = EMBED_MODEL) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


class Retriever:
    def __init__(self, book_name: str):
        book_dir = INDEX_DIR / book_name
        if not book_dir.exists():
            raise FileNotFoundError(
                f"No index found for '{book_name}'. Run ingest.py first."
            )
        self.index = faiss.read_index(str(book_dir / "index.faiss"))
        with open(book_dir / "chunks.pkl", "rb") as f:
            self.chunks: list[dict] = pickle.load(f)
        self.model = _get_model()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Context Pruning: embed the query, find top_k nearest chunks.
        Returns only those chunks — not the full document.
        """
        q_vec = self.model.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)

        # Sort by relevance score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def build_context(self, query: str, top_k: int = 5, max_words: int = 1200) -> tuple[str, list[int]]:
        """
        Returns a pruned context string and the page numbers it covers.
        max_words caps the context to control token spend.
        """
        chunks = self.retrieve(query, top_k=top_k)
        context_parts = []
        total_words = 0
        pages_used = []

        for chunk in chunks:
            words = chunk["text"].split()
            if total_words + len(words) > max_words:
                # Trim to fit budget
                words = words[: max_words - total_words]
            context_parts.append(" ".join(words))
            pages_used.extend(chunk["pages"])
            total_words += len(words)
            if total_words >= max_words:
                break

        context = "\n\n---\n\n".join(context_parts)
        return context, sorted(set(pages_used))


def list_books() -> list[str]:
    if not INDEX_DIR.exists():
        return []
    return [d.name for d in INDEX_DIR.iterdir() if d.is_dir()]
