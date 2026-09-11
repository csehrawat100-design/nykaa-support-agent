from caching.response_cache import ResponseCache, normalize_query
from caching.grounded_generation import CachedGroundedGenerator


def test_normalization_produces_same_key_for_case_and_whitespace():
    assert normalize_query("  What is the RETURN window? ") == "what is the return window?"
    assert normalize_query("WHAT   IS the return   window?") == "what is the return window?"


def test_repeated_query_is_a_real_cache_hit_and_avoids_call():
    calls = {"count": 0}

    def fake_grounded_generation(query):
        calls["count"] += 1
        return f"grounded answer for: {query}"

    cached = CachedGroundedGenerator(fake_grounded_generation)

    first, first_hit = cached.generate("What is the return window for beauty products?")
    assert first_hit is False
    assert calls["count"] == 1

    second, second_hit = cached.generate("  what IS the return window for beauty products?  ")
    assert second_hit is True
    assert second == first
    assert calls["count"] == 1
    assert cached.cache.stats().calls_avoided == 1


def test_different_normalized_queries_are_not_collapsed():
    calls = {"count": 0}

    def compute(query):
        calls["count"] += 1
        return query

    cache = ResponseCache()
    a, hit_a = cache.get_or_compute("return window", lambda: compute("return window"))
    b, hit_b = cache.get_or_compute("delivery time", lambda: compute("delivery time"))
    assert not hit_a and not hit_b
    assert a != b
    assert calls["count"] == 2


def test_cache_can_be_cleared():
    calls = {"count": 0}

    def compute(query):
        calls["count"] += 1
        return "answer"

    cache = ResponseCache()
    cache.get_or_compute("same query", lambda: compute("same query"))
    cache.get_or_compute("same query", lambda: compute("same query"))
    assert calls["count"] == 1
    cache.clear()
    _, hit = cache.get_or_compute("same query", lambda: compute("same query"))
    assert hit is False
    assert calls["count"] == 2
