"""
app.py — CLI interface for the tutoring system.

Usage:
  python app.py ingest <pdf_path> [book_name]
  python app.py ask    <book_name> "<question>"
  python app.py chat   <book_name>              # interactive session
  python app.py books                           # list ingested books
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False


def print_answer(result: dict):
    answer = result["answer"]
    pages = result["pages"]
    cost = result["cost"]

    if USE_RICH:
        console.print(Panel(answer, title=f"Answer (pages {pages})", border_style="green"))
        table = Table(title="Cost Breakdown", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Pruned input tokens",   f"{cost['pruned_total_input']:,}")
        table.add_row("Baseline input tokens", f"{cost['baseline_total_input']:,}")
        table.add_row("Output tokens",         f"{cost['output_tokens']:,}")
        table.add_row("Pruned cost (USD)",      f"${cost['pruned_cost_usd']:.6f}")
        table.add_row("Baseline cost (USD)",    f"${cost['baseline_cost_usd']:.6f}")
        table.add_row("Savings",               f"${cost['savings_usd']:.6f}  ({cost['savings_pct']:.1f}%)")
        console.print(table)
    else:
        print(f"\nAnswer (pages {pages}):\n{answer}")
        print(f"\n--- Cost ---")
        print(f"  Pruned input:   {cost['pruned_total_input']:,} tokens  (${cost['pruned_cost_usd']:.6f})")
        print(f"  Baseline input: {cost['baseline_total_input']:,} tokens  (${cost['baseline_cost_usd']:.6f})")
        print(f"  Savings:        {cost['savings_pct']:.1f}%  (${cost['savings_usd']:.6f})")


def cmd_ingest(args):
    if not args:
        print("Usage: python app.py ingest <pdf_path> [book_name]")
        sys.exit(1)
    from ingest import ingest
    pdf = args[0]
    name = args[1] if len(args) > 1 else None
    ingest(pdf, name)


def cmd_ask(args):
    if len(args) < 2:
        print("Usage: python app.py ask <book_name> \"<question>\"")
        sys.exit(1)
    book, question = args[0], args[1]
    from tutor import Tutor
    tutor = Tutor(book)
    result = tutor.ask(question)
    print_answer(result)


def cmd_chat(args):
    if not args:
        print("Usage: python app.py chat <book_name>")
        sys.exit(1)
    book = args[0]
    from tutor import Tutor

    tutor = Tutor(book)
    print(f"\nTutor ready for '{book}'. Type 'quit' to exit, 'costs' to see summary.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "costs":
            summary = tutor.cost_summary()
            print(f"\nSession summary: {summary}\n")
            continue

        result = tutor.ask(question)
        print_answer(result)

    summary = tutor.cost_summary()
    if summary:
        print(f"\nSession ended. Total savings: {summary['avg_savings_pct']}% avg | "
              f"${summary['total_savings_usd']:.4f} saved vs baseline.")


def cmd_books(_):
    from retriever import list_books
    books = list_books()
    if not books:
        print("No books ingested yet. Run: python app.py ingest <pdf>")
    else:
        print("Ingested books:")
        for b in books:
            print(f"  - {b}")


COMMANDS = {
    "ingest": cmd_ingest,
    "ask":    cmd_ask,
    "chat":   cmd_chat,
    "books":  cmd_books,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Commands: ingest | ask | chat | books")
        print("  python app.py ingest textbook.pdf [name]")
        print("  python app.py ask    <book> \"question\"")
        print("  python app.py chat   <book>")
        print("  python app.py books")
        sys.exit(1)

    cmd = sys.argv[1]
    COMMANDS[cmd](sys.argv[2:])
