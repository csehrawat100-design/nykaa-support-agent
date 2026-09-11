"""Build the Nykaa support knowledge-base ChromaDB indexes.

This script connects the RAG ingestion pipeline:

    knowledge_base/*.txt
        -> document_loader
        -> fixed-size and sentence chunking
        -> local SentenceTransformer embeddings
        -> separate ChromaDB collections

Run from the project root with:
    python -m rag.build_index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import chunk_documents
from .document_loader import load_documents
from .embeddings import SentenceTransformerEmbedder
from .vector_store import DEFAULT_PERSIST_DIRECTORY, build_vector_stores


def build_index(
    *,
    knowledge_base_dir: str | Path | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    chunk_size: int = 300,
    overlap: int = 50,
    reset: bool = False,
) -> dict[str, int]:
    """Build both required ChromaDB collections and return index statistics."""
    documents = load_documents(knowledge_base_dir)

    if not documents:
        raise RuntimeError("No knowledge-base documents were loaded.")

    fixed_chunks = chunk_documents(
        documents,
        strategy="fixed_size",
        chunk_size=chunk_size,
        overlap=overlap,
    )

    sentence_chunks = chunk_documents(
        documents,
        strategy="sentence",
    )

    embedder = SentenceTransformerEmbedder()

    fixed_embeddings = embedder.embed_chunks(fixed_chunks)
    sentence_embeddings = embedder.embed_chunks(sentence_chunks)

    stores = build_vector_stores(
        fixed_chunks=fixed_chunks,
        fixed_embeddings=fixed_embeddings,
        sentence_chunks=sentence_chunks,
        sentence_embeddings=sentence_embeddings,
        persist_directory=persist_directory,
        reset=reset,
    )

    return {
        "documents": len(documents),
        "fixed_size_chunks": len(fixed_chunks),
        "sentence_chunks": len(sentence_chunks),
        "fixed_size_collection": stores["fixed_size"].count(),
        "sentence_collection": stores["sentence"].count(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Build the Nykaa support knowledge-base ChromaDB indexes."
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=None,
        help="Path to the knowledge_base directory (defaults to the project directory).",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
        help="Directory where ChromaDB data is persisted.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Fixed-size chunk length in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between fixed-size chunks in characters.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the two ChromaDB collections before indexing.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the indexes and print verification statistics."""
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0.")
    if args.overlap < 0:
        raise ValueError("--overlap cannot be negative.")  
    if args.overlap >= args.chunk_size:
        raise ValueError("--overlap must be smaller than --chunk-size.")

    print("Building Nykaa support knowledge-base indexes...\n")

    summary = build_index(
        knowledge_base_dir=args.knowledge_base,
        persist_directory=args.persist_directory,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset=args.reset,
    )

    print(f"Documents loaded:        {summary['documents']}")
    print(f"Fixed-size chunks:       {summary['fixed_size_chunks']}")
    print(f"Sentence chunks:         {summary['sentence_chunks']}")
    print(f"fixed_size_collection:   {summary['fixed_size_collection']}")
    print(f"sentence_collection:     {summary['sentence_collection']}")
    print("\nIndexing completed successfully.")

if __name__ == "__main__":
    main()