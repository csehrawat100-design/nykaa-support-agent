"""Cache wrapper around the existing grounded-generation step.

The underlying generator is injected so Task 16 does not duplicate or replace
the project's existing RAG generation implementation.
"""
from typing import Callable, Any
from .response_cache import ResponseCache


class CachedGroundedGenerator:
    def __init__(self, generator: Callable[[str], Any], cache: ResponseCache | None = None):
        self.generator = generator
        self.cache = cache or ResponseCache()

    def generate(self, query: str) -> tuple[Any, bool]:
        """Generate once per normalized query; return response and hit flag."""
        return self.cache.get_or_compute(query, lambda: self.generator(query))