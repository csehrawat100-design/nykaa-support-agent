"""ChromaDB vector-store layer for the Nykaa Support Agent."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from .document_loader import Chunk

DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAMES = {"fixed_size": "fixed_size_collection", "sentence": "sentence_collection"}

class ChromaVectorStore:
    """Manage one persistent ChromaDB collection for one chunking strategy."""
    def __init__(self, strategy: str, persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY) -> None:
        if strategy not in COLLECTION_NAMES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Use 'fixed_size' or 'sentence'.")
        self.strategy = strategy
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    @property
    def collection_name(self) -> str:
        return COLLECTION_NAMES[self.strategy]

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError("chromadb is required for vector storage.") from exc
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"chunk_strategy": self.strategy},
            )
        return self._collection

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        """Upsert chunks, embeddings, and parent-document metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must contain the same number of items.")
        if not chunks:
            return 0

        ids, documents, metadatas, vectors = [], [], [], []
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_strategy != self.strategy:
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' has strategy '{chunk.chunk_strategy}', "
                    f"expected '{self.strategy}'."
                )
            if not chunk.text.strip():
                raise ValueError(f"Chunk '{chunk.chunk_id}' has empty text.")
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "source": chunk.source,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_strategy": chunk.chunk_strategy,
            })
            vectors.append([float(value) for value in embedding])

        self.collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
        return len(chunks)

    def query(self, query_embedding: Sequence[float], *, top_k: int = 5) -> dict[str, Any]:
        """Return nearest chunks plus cosine-similarity values."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        distances = result.get("distances", [[]])[0]
        result["similarities"] = [[1.0 / (1.0 + float(distance)) for distance in distances]]
        return result

    def count(self) -> int:
        return int(self.collection.count())

    def reset(self) -> None:
        """Explicitly delete this collection so it can be rebuilt."""
        self.client.delete_collection(self.collection_name)
        self._collection = None


def build_vector_stores(
    fixed_chunks: Sequence[Chunk],
    fixed_embeddings: Sequence[Sequence[float]],
    sentence_chunks: Sequence[Chunk],
    sentence_embeddings: Sequence[Sequence[float]],
    *,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    reset: bool = False,
) -> dict[str, ChromaVectorStore]:
    """Build the two required, separate ChromaDB collections."""
    stores = {
        "fixed_size": ChromaVectorStore("fixed_size", persist_directory),
        "sentence": ChromaVectorStore("sentence", persist_directory),
    }
    if reset:
        for store in stores.values():
            try:
                store.reset()
            except Exception:
                pass
    stores["fixed_size"].upsert(fixed_chunks, fixed_embeddings)
    stores["sentence"].upsert(sentence_chunks, sentence_embeddings)
    return stores

__all__ = [
    "ChromaVectorStore", 
    "COLLECTION_NAMES", 
    "DEFAULT_PERSIST_DIRECTORY", 
    "build_vector_stores"
]
 