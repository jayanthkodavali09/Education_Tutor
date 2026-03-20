"""
tutor.py — Core tutoring logic.

Flow:
  1. Student asks a question.
  2. Retriever prunes the textbook to top-k relevant chunks.
  3. Pruned context + question → LLM → answer.
  4. CostTracker logs token savings vs baseline.
"""

import os
from openai import OpenAI
from retriever import Retriever
from cost_tracker import CostTracker, count_tokens

# How many chunks to retrieve (tune this: more = better answers, higher cost)
DEFAULT_TOP_K = 5
DEFAULT_MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = """You are a helpful, patient tutor for students in rural India.
You answer questions strictly based on the provided textbook excerpts.
- Use simple, clear language.
- If the answer is not in the excerpts, say so honestly.
- Reference page numbers when possible.
- Keep answers concise but complete."""


class Tutor:
    def __init__(self, book_name: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K):
        self.retriever = Retriever(book_name)
        self.model = model
        self.top_k = top_k
        self.cost_tracker = CostTracker(model)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Cache full-doc token count (computed once, used for cost comparison)
        all_text = " ".join(c["text"] for c in self.retriever.chunks)
        self.full_doc_tokens = count_tokens(all_text)
        print(f"[tutor] Book loaded. Full doc = {self.full_doc_tokens:,} tokens.")

    def ask(self, question: str) -> dict:
        """
        Ask a question. Returns answer + cost breakdown.
        """
        # ── Step 1: Context Pruning ───────────────────────────────────────────
        pruned_context, pages = self.retriever.build_context(question, top_k=self.top_k)

        # ── Step 2: Build prompt ──────────────────────────────────────────────
        user_message = (
            f"Textbook excerpts (pages {pages}):\n\n"
            f"{pruned_context}\n\n"
            f"---\n\nStudent question: {question}"
        )

        # ── Step 3: LLM call ──────────────────────────────────────────────────
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

        # ── Step 4: Track costs ───────────────────────────────────────────────
        cost_entry = self.cost_tracker.record(
            query=question,
            pruned_context=pruned_context,
            full_doc_tokens=self.full_doc_tokens,
            output=answer,
        )

        return {
            "answer": answer,
            "pages": pages,
            "cost": cost_entry,
        }

    def cost_summary(self) -> dict:
        return self.cost_tracker.summary()
