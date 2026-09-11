"""Chunking strategies for the Nykaa Support Agent RAG pipeline.

Two strategies are intentionally implemented:
1. Fixed-size chunks with character overlap.
2. Sentence-based chunks.

Both return the same Chunk representation from document_loader.py so the
embedding, vector-store, retrieval, and generation layers can remain
independent of the selected chunking strategy.
"""

from __future__ import annotations

import re
from typing import Sequence

from .document_loader import Chunk, Document


def fixed_size_chunks(
    document: Document,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a document into fixed-size character chunks with overlap.

    Args:
        document: Parent document to split.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Number of characters shared between adjacent chunks.

    Returns:
        A deterministic list of chunks retaining parent-document metadata.

    Raises:
        ValueError: If chunk_size/overlap values are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_number = 1
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}_{chunk_number:03d}",
                    document_id=document.document_id,
                    source=document.source,
                    text=chunk_text,
                    chunk_strategy="fixed_size",
                )
            )
            chunk_number += 1

        start += step

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split policy text into sentences while preserving sentence content.

    The KB is short prose, so a lightweight deterministic sentence splitter
    is preferable to adding another NLP dependency at this layer.
    """
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    # Split after ., !, or ? followed by whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_chunks(document: Document) -> list[Chunk]:
    """Create one chunk per sentence.

    Returns:
        Sentence chunks with stable IDs and the same metadata interface as
        fixed-size chunks.
    """
    sentences = _split_sentences(document.text)

    return [
        Chunk(
            chunk_id=f"{document.document_id}_{index:03d}",
            document_id=document.document_id,
            source=document.source,
            text=sentence,
            chunk_strategy="sentence",
        )
        for index, sentence in enumerate(sentences, start=1)
    ]


def chunk_documents(
    documents: Sequence[Document],
    strategy: str,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk multiple documents using one of the supported strategies.

    Args:
        documents: Documents loaded by document_loader.
        strategy: Either ``"fixed_size"`` or ``"sentence"``.
        chunk_size: Used only for fixed-size chunking.
        overlap: Used only for fixed-size chunking.

    Returns:
        Chunks in deterministic document/input order.

    Raises:
        ValueError: If strategy is unsupported.
    """
    if strategy == "fixed_size":
        chunker = lambda document: fixed_size_chunks(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif strategy == "sentence":
        chunker = sentence_chunks
    else:
        raise ValueError(
            f"Unsupported chunking strategy '{strategy}'. "
            "Use 'fixed_size' or 'sentence'."
        )

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker(document))

    return chunks


__all__ = [
    "chunk_documents",
    "fixed_size_chunks",
    "sentence_chunks",
]
