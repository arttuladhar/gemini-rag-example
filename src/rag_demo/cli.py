from __future__ import annotations

import argparse
from textwrap import dedent

from .config import get_settings
from .data_loader import load_local_documents
from .pipeline import run_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag-demo",
        description="Minimal LangChain RAG sample powered by ChromaDB + Google Gemini",
    )
    parser.add_argument(
        "-q",
        "--query",
        dest="query",
        help="Question to run against the local knowledge base",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        settings = get_settings()
        documents = load_local_documents(settings.data_dir)
    except Exception as exc:  # pragma: no cover - CLI experience
        raise SystemExit(str(exc))

    question = args.query or settings.sample_question

    print(
        dedent(
            f"""
            🔎 Running Gemini RAG demo
            - Documents: {len(documents)}
            - Persisted vectors: {settings.persist_directory}
            - Question: {question}
            """
        ).strip()
    )

    answer = run_query(question, documents, settings)
    print("\n💡 Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
