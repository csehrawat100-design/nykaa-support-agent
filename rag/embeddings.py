"""Local SentenceTransformers embeddings for the Nykaa Support Agent."""

from __future__ import annotations

from collections.abc import Sequence

from .document_loader import Chunk

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Wrapper around a local SentenceTransformers embedding model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Load the model lazily on first embedding request."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings."
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the configured model."""
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for '{self.model_name}'."
            )
        return int(dimension)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Embed text values in input order."""
        if not texts:
            return []

        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Embedding input contains an empty text value.")

        embeddings = self.model.encode(
            cleaned,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Embed chunks while preserving their ordering."""
        return self.embed_texts(
            [chunk.text for chunk in chunks],
            normalize=normalize,
        )

    def embed_query(
        self,
        query: str,
        *,
        normalize: bool = True,
    ) -> list[float]:
        """Embed one user query with the same model as the KB."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        return self.embed_texts([query], normalize=normalize)[0]


__all__ = [
    "DEFAULT_MODEL_NAME",
    "SentenceTransformerEmbedder",
]
