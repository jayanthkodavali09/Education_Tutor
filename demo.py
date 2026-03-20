"""
demo.py — Runs a full end-to-end demo without a real PDF.

Creates a synthetic "textbook" in memory, builds the index,
then runs sample queries and prints cost savings.

Run: python demo.py
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path

# ── Synthetic textbook content ────────────────────────────────────────────────
FAKE_CHAPTERS = [
    ("Photosynthesis", """
Photosynthesis is the process by which green plants, algae, and some bacteria convert
light energy into chemical energy stored as glucose. It occurs mainly in the chloroplasts.
The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
There are two stages: the light-dependent reactions (in the thylakoid membranes) and
the Calvin cycle (in the stroma). Chlorophyll absorbs red and blue light most efficiently.
"""),
    ("Cell Division", """
Cell division is the process by which a parent cell divides into two or more daughter cells.
Mitosis produces two genetically identical diploid cells and is used for growth and repair.
Meiosis produces four genetically unique haploid cells used in sexual reproduction.
The cell cycle consists of interphase (G1, S, G2) and the mitotic phase (prophase,
metaphase, anaphase, telophase, and cytokinesis).
"""),
    ("Newton's Laws of Motion", """
Newton's First Law (Inertia): An object at rest stays at rest, and an object in motion
stays in motion unless acted upon by an external force.
Newton's Second Law: Force equals mass times acceleration (F = ma).
Newton's Third Law: For every action, there is an equal and opposite reaction.
These laws form the foundation of classical mechanics and explain everyday motion.
"""),
    ("The Water Cycle", """
The water cycle (hydrological cycle) describes the continuous movement of water on,
above, and below Earth's surface. Key processes: evaporation (water turns to vapor),
condensation (vapor forms clouds), precipitation (rain/snow falls), and collection
(water gathers in oceans, rivers, lakes). Transpiration from plants also contributes.
The sun drives the cycle by providing energy for evaporation.
"""),
    ("Fractions and Decimals", """
A fraction represents a part of a whole. It has a numerator (top) and denominator (bottom).
To add fractions, find a common denominator. To multiply, multiply numerators and denominators.
Decimals are another way to represent fractions with denominators of powers of 10.
To convert a fraction to a decimal, divide the numerator by the denominator.
Example: 3/4 = 0.75. Percentages are fractions out of 100.
"""),
    ("The Indian Constitution", """
The Constitution of India came into effect on January 26, 1950. It is the supreme law of India.
Dr. B.R. Ambedkar is known as the chief architect of the Indian Constitution.
It establishes India as a sovereign, socialist, secular, democratic republic.
The Preamble outlines the objectives: justice, liberty, equality, and fraternity.
Fundamental Rights are guaranteed in Part III (Articles 12-35).
"""),
    ("Chemical Bonding", """
Chemical bonds hold atoms together in compounds. Ionic bonds form when electrons are
transferred between atoms (e.g., NaCl). Covalent bonds form when electrons are shared
(e.g., H2O, CO2). Metallic bonds exist in metals where electrons flow freely.
Bond strength is measured in kJ/mol. Electronegativity differences determine bond type:
>1.7 = ionic, 0.4-1.7 = polar covalent, <0.4 = nonpolar covalent.
"""),
    ("Mughal Empire", """
The Mughal Empire was founded by Babur in 1526 after the First Battle of Panipat.
Akbar the Great (1556-1605) is considered the greatest Mughal emperor, known for
religious tolerance and administrative reforms. Shah Jahan built the Taj Mahal.
Aurangzeb was the last major Mughal emperor. The empire declined in the 18th century
due to weak successors, regional revolts, and European colonial expansion.
"""),
]


def build_demo_index():
    """Build a FAISS index from synthetic textbook content."""
    import faiss
    from sentence_transformers import SentenceTransformer

    INDEX_DIR = Path("index/demo_textbook")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Build chunks from fake chapters
    chunks = []
    chunk_id = 0
    for i, (title, text) in enumerate(FAKE_CHAPTERS):
        words = text.split()
        chunks.append({
            "id": chunk_id,
            "text": f"[Chapter: {title}] {text.strip()}",
            "pages": [i * 5 + 1, i * 5 + 2],   # fake page numbers
        })
        chunk_id += 1

    print("[demo] Embedding chunks locally (no API cost)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Compute full-doc token count for cost comparison
    from cost_tracker import count_tokens
    full_text = " ".join(c["text"] for c in chunks)
    full_tokens = count_tokens(full_text)

    meta = {
        "book": "demo_textbook",
        "num_chunks": len(chunks),
        "full_doc_tokens": full_tokens,
    }
    with open(INDEX_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[demo] Index built. {len(chunks)} chunks, {full_tokens:,} full-doc tokens.\n")
    return full_tokens


def run_demo_without_api():
    """Show context pruning + cost savings without calling OpenAI."""
    from retriever import Retriever
    from cost_tracker import CostTracker, count_tokens

    full_tokens = build_demo_index()
    retriever = Retriever("demo_textbook")
    tracker = CostTracker("gpt-3.5-turbo")

    questions = [
        "What is photosynthesis and where does it occur?",
        "Explain Newton's second law with the formula.",
        "Who built the Taj Mahal?",
        "How do you convert a fraction to a decimal?",
        "What are the stages of the cell cycle?",
    ]

    print("=" * 65)
    print("  CONTEXT PRUNING DEMO — Cost Savings vs Baseline RAG")
    print("=" * 65)

    for q in questions:
        context, pages = retriever.build_context(q, top_k=2)
        pruned_tokens = count_tokens(context) + count_tokens(q)
        fake_answer = "[Answer would appear here from LLM]"

        entry = tracker.record(
            query=q,
            pruned_context=context,
            full_doc_tokens=full_tokens,
            output=fake_answer,
        )

        print(f"\nQ: {q}")
        print(f"   Pages retrieved: {pages}")
        print(f"   Pruned input:    {entry['pruned_total_input']:>5} tokens  (${entry['pruned_cost_usd']:.6f})")
        print(f"   Baseline input:  {entry['baseline_total_input']:>5} tokens  (${entry['baseline_cost_usd']:.6f})")
        print(f"   Savings:         {entry['savings_pct']:.1f}%")

    summary = tracker.summary()
    print("\n" + "=" * 65)
    print(f"  SESSION SUMMARY ({summary['queries']} queries)")
    print(f"  Total pruned cost:   ${summary['total_pruned_cost_usd']:.6f}")
    print(f"  Total baseline cost: ${summary['total_baseline_cost_usd']:.6f}")
    print(f"  Total saved:         ${summary['total_savings_usd']:.6f}")
    print(f"  Average savings:     {summary['avg_savings_pct']}%")
    print("=" * 65)
    print("\nTo use with a real textbook PDF:")
    print("  python app.py ingest your_textbook.pdf")
    print("  python app.py chat   your_textbook")


if __name__ == "__main__":
    run_demo_without_api()
