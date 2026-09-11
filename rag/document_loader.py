"""Knowledge-base document loading for the Nykaa Support Agent.

This module deliberately handles only document loading and document-level
representation. Chunking, embeddings, vector storage, retrieval, and
generation belong to separate RAG modules.

The loader keeps the source filename and a stable document_id so later
chunking/retrieval stages can always map a chunk back to its parent document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """A complete knowledge-base document before chunking."""

    source: str
    document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """Shared chunk representation used by downstream RAG components."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_strategy: str


def _document_id(path: Path) -> str:
    """Return the stable parent-document identifier for a KB file."""
    return path.stem


def load_documents(knowledge_base_dir: str | Path | None = None) -> list[Document]:
    """Load all UTF-8 ``.txt`` knowledge-base documents.

    Args:
        knowledge_base_dir: Directory containing the KB text files.
            Defaults to ``<project_root>/knowledge_base`` when this module
            is located at ``<project_root>/rag/document_loader.py``.

    Returns:
        Documents sorted by filename for deterministic downstream behavior.

    Raises:
        FileNotFoundError: If the knowledge-base directory does not exist.
        NotADirectoryError: If the supplied path is not a directory.
        ValueError: If a KB file is empty or two files produce the same
            document identifier.
    """
    if knowledge_base_dir is None:
        knowledge_base_dir = Path(__file__).resolve().parent.parent / "knowledge_base"
    else:
        knowledge_base_dir = Path(knowledge_base_dir)

    if not knowledge_base_dir.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory does not exist: {knowledge_base_dir}"
        )

    if not knowledge_base_dir.is_dir():
        raise NotADirectoryError(
            f"Knowledge-base path is not a directory: {knowledge_base_dir}"
        )

    documents: list[Document] = []
    seen_document_ids: set[str] = set()

    for path in sorted(knowledge_base_dir.glob("*.txt"), key=lambda p: p.name.lower()):
        document_id = _document_id(path)

        if document_id in seen_document_ids:
            raise ValueError(
                f"Duplicate document_id '{document_id}' found in: {knowledge_base_dir}"
            )

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Knowledge-base document is empty: {path}")

        documents.append(
            Document(
                source=path.name,
                document_id=document_id,
                text=text,
            )
        )
        seen_document_ids.add(document_id)

    if not documents:
        raise ValueError(f"No .txt knowledge-base documents found in: {knowledge_base_dir}")

    return documents


def load_document(path: str | Path) -> Document:
    """Load one KB document from a UTF-8 text file.

    This helper is useful for focused tests and for the future ``/add-document``
    API endpoint without changing the bulk-loader interface.
    """
    path = Path(path)

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt knowledge-base document: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Knowledge-base document does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Knowledge-base path is not a file: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Knowledge-base document is empty: {path}")

    return Document(
        source=path.name,
        document_id=_document_id(path),
        text=text,
    )


__all__ = [
    "Chunk",
    "Document",
    "load_document",
    "load_documents",
]
