# AI Tutoring System

AI tutoring system for rural India that ingests state-board textbooks (PDFs) and answers student questions using context pruning — only sending relevant chunks to the LLM instead of the full document. Cuts API costs by ~72% per query. Runs embeddings locally, works on low-end hardware with minimal data transfer.

## How It Works

1. **Ingest once** — PDF is chunked, embedded locally (no API cost), and saved as a FAISS index on disk.
2. **Query cheap** — at question time, only the top-k semantically relevant chunks are sent to the LLM, not the full textbook.
3. **Track savings** — every query logs token counts and USD cost vs the naive baseline.

```
tutoring-system/
├── ingest.py        # PDF → chunks → FAISS index (run once)
├── retriever.py     # Context pruning via semantic search
├── tutor.py         # Pruned context + question → LLM answer
├── cost_tracker.py  # Token counting + cost comparison
├── app.py           # CLI interface
├── demo.py          # End-to-end demo (no PDF needed)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI API key
```

## Usage

**Run the demo (no PDF required):**
```bash
python demo.py
```

**With a real textbook:**
```bash
# Ingest a PDF (one-time)
python app.py ingest class10_science.pdf science10

# Interactive chat session
python app.py chat science10

# Single question
python app.py ask science10 "What is osmosis?"

# List ingested books
python app.py books
```

## Cost Savings

| | Baseline RAG | Pruned RAG |
|---|---|---|
| Tokens sent per query | ~150,000 (full doc) | ~200 (top-k chunks) |
| Cost per query (GPT-3.5) | $0.075 | ~$0.0001 |
| Avg savings | — | **~72–99%** |

Savings scale with textbook size. A 300-page book at 150k tokens vs ~200 pruned tokens = >99% reduction.

## Key Design Choices

- **`all-MiniLM-L6-v2`** for embeddings — 90MB, runs fully offline, no GPU needed.
- **FAISS flat index** — exact search, fast enough for textbook-scale chunk counts (<100k).
- **400-word chunks with 50-word overlap** — balances context coherence vs token cost.
- **`gpt-3.5-turbo` default** — cheapest capable model; swap to `gpt-4o-mini` for better answers at similar cost.

## Requirements

- Python 3.9+
- OpenAI API key
