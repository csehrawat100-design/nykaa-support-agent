"""In-memory response cache for the grounded-generation step.

The cache key is the normalized query text. A cache hit returns the previously
computed grounded response and therefore skips the expensive generation/tool
step represented by the supplied compute function.
"""
from dataclasses import dataclass
import re
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


def normalize_query(query: str) -> str:
    """Normalize equivalent query text into a stable cache key."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    text = query.strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int

    @property
    def calls_avoided(self) -> int:
        return self.hits


class ResponseCache(Generic[T]):
    """Small process-local in-memory cache keyed by normalized query text."""

    def __init__(self) -> None:
        self._cache: dict[str, T] = {}
        self._hits = 0
        self._misses = 0

    def get(self, query: str) -> T | None:
        key = normalize_query(query)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, query: str, response: T) -> T:
        self._cache[normalize_query(query)] = response
        return response

    def get_or_compute(self, query: str, compute: Callable[[], T]) -> tuple[T, bool]:
        """Return (response, cache_hit); compute only on a cache miss."""
        cached = self.get(query)
        if cached is not None:
            return cached, True
        response = compute()
        self.put(query, response)
        return response, False

    def stats(self) -> CacheStats:
        return CacheStats(self._hits, self._misses)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._cache)
