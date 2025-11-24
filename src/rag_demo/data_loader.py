from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

TEXT_EXTENSIONS = (".md", ".txt")


def load_local_documents(data_dir: Path) -> List[Document]:
    """Load markdown/text files from ``data_dir`` as LangChain Documents."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = [path for path in data_dir.rglob("*") if path.suffix.lower() in TEXT_EXTENSIONS]
    if not files:
        raise FileNotFoundError(
            f"No .md/.txt documents found in {data_dir}. Add sample content to continue."
        )

    documents: List[Document] = []
    for path in sorted(files):
        text = path.read_text(encoding="utf-8").strip()
        metadata = {"source": str(path.relative_to(data_dir))}
        documents.append(Document(page_content=text, metadata=metadata))

    return documents
