"""
LRU Embedding Cache for Nexus CV.
Hash-based caching of text embeddings to avoid redundant model inference.
Thread-safe, max 500 entries, automatic eviction.
"""

import hashlib
from collections import OrderedDict
import threading


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors."""

    def __init__(self, max_size=500):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text, max_chars=2000):
        """SHA-256 hash of the first max_chars characters."""
        return hashlib.sha256(text[:max_chars].encode("utf-8")).hexdigest()

    def get(self, text):
        """Retrieve cached embedding. Returns numpy array or None."""
        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, text, embedding):
        """Store embedding in cache. Evicts oldest if full."""
        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = embedding
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = embedding

    def clear(self):
        """Flush all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self):
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }
