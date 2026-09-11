"""Retrieval layer for the Nykaa Support Agent RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .embeddings import SentenceTransformerEmbedder
from .vector_store import DEFAULT_PERSIST_DIRECTORY, ChromaVectorStore


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved chunk with source metadata and similarity."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_strategy: str
    distance: float
    similarity: float


class Retriever:
    """Retrieve chunks from one of the two RAG collections."""

    def __init__(
        self,
        strategy: str = "fixed_size",
        *,
        embedder: SentenceTransformerEmbedder | None = None,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        top_k: int = 5,
    ) -> None:
        if strategy not in {"fixed_size", "sentence"}:
            raise ValueError("strategy must be 'fixed_size' or 'sentence'")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.strategy = strategy
        self.top_k = top_k
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.store = ChromaVectorStore(
            strategy,
            persist_directory=persist_directory,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Embed a query and return its nearest chunks."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k must be greater than 0")

        raw = self.store.query(self.embedder.embed_query(query), top_k=k)
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        similarities = raw.get("similarities", [[]])[0]

        results: list[RetrievedChunk] = []
        for document, metadata, distance, similarity in zip(
            documents, metadatas, distances, similarities
        ):
            metadata = metadata or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    document_id=str(metadata.get("document_id", "")),
                    source=str(metadata.get("source", "")),
                    text=str(document),
                    chunk_strategy=str(
                        metadata.get("chunk_strategy", self.strategy)
                    ),
                    distance=float(distance),
                    similarity=float(similarity),
                )
            )

        return results

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[str]:
        """Return deduplicated parent document IDs for evaluation."""
        seen: set[str] = set()
        document_ids: list[str] = []
        for chunk in self.retrieve(query, top_k=top_k):
            if chunk.document_id and chunk.document_id not in seen:
                seen.add(chunk.document_id)
                document_ids.append(chunk.document_id)
        return document_ids

    def top_similarity(self, query: str) -> float:
        """Return top-1 similarity for empirical threshold calibration."""
        results = self.retrieve(query, top_k=1)
        return results[0].similarity if results else 0.0

    def has_results(self) -> bool:
        """Return whether this collection contains indexed chunks."""
        return self.store.count() > 0


def retrieve_from_both(
    query: str,
    *,
    top_k: int = 5,
    embedder: SentenceTransformerEmbedder | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
) -> dict[str, list[RetrievedChunk]]:
    """Retrieve the same query from both chunking collections."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    shared_embedder = embedder or SentenceTransformerEmbedder()
    query_embedding = shared_embedder.embed_query(query)
    output: dict[str, list[RetrievedChunk]] = {}

    for strategy in ("fixed_size", "sentence"):
        store = ChromaVectorStore(
            strategy,
            persist_directory=persist_directory,
        )
        raw = store.query(query_embedding, top_k=top_k)
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        similarities = raw.get("similarities", [[]])[0]

        output[strategy] = [
            RetrievedChunk(
                chunk_id=str((metadata or {}).get("chunk_id", "")),
                document_id=str((metadata or {}).get("document_id", "")),
                source=str((metadata or {}).get("source", "")),
                text=str(document),
                chunk_strategy=str(
                    (metadata or {}).get("chunk_strategy", strategy)
                ),
                distance=float(distance),
                similarity=float(similarity),
            )
            for document, metadata, distance, similarity in zip(
                documents, metadatas, distances, similarities
            )
        ]

    return output


__all__ = [
    "RetrievedChunk", 
    "Retriever", 
    "retrieve_from_both"
]
