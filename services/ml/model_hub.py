"""
Model Hub — Central lazy-loading model registry for Nexus CV.

All heavy ML models (BGE embeddings, BART classifier, XGBoost scorer)
are loaded on first use, not at import time. Embedding results are
cached via EmbeddingCache for performance.

Usage:
    from services.ml.model_hub import get_embedder, embed_text, embed_batch
    from services.ml.model_hub import get_classifier, get_ats_model
"""

import os
import logging
import numpy as np

from services.ml.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Singleton state ──
_embedder = None
_classifier = None
_ats_model = None
_cache = EmbeddingCache(max_size=500)

# ── Preferred model names (with fallbacks) ──
_BGE_MODEL = "BAAI/bge-large-en-v1.5"
_BGE_FALLBACK = "all-MiniLM-L6-v2"
_BART_MODEL = "facebook/bart-large-mnli"


# ────────────── EMBEDDER ──────────────

def get_embedder():
    """
    Returns a SentenceTransformer model.
    Tries BGE-large first, falls back to MiniLM, falls back to None.
    Loaded on first call only.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
        try:
            logger.info("Loading embedding model: %s", _BGE_MODEL)
            _embedder = SentenceTransformer(_BGE_MODEL)
            logger.info("BGE-large loaded successfully")
        except Exception as e:
            logger.warning("BGE-large failed (%s), falling back to MiniLM", e)
            _embedder = SentenceTransformer(_BGE_FALLBACK)
            logger.info("MiniLM fallback loaded")
    except ImportError:
        logger.warning("sentence-transformers not installed, embedder unavailable")
        _embedder = None

    return _embedder


def embed_text(text, normalize=True):
    """
    Encode a single text string into an embedding vector.
    Uses cache: returns cached result if available, otherwise computes and caches.
    Returns numpy array or None if embedder unavailable.
    """
    cached = _cache.get(text)
    if cached is not None:
        return cached

    embedder = get_embedder()
    if embedder is None:
        return None

    try:
        embedding = embedder.encode(
            text[:2000],
            normalize_embeddings=normalize
        )
        _cache.put(text, embedding)
        return embedding
    except Exception as e:
        logger.error("embed_text failed: %s", e)
        return None


def embed_batch(texts, normalize=True):
    """
    Encode multiple texts. Uses cache for each individual text.
    Returns list of numpy arrays (or None entries for failures).
    """
    embedder = get_embedder()
    if embedder is None:
        return [None] * len(texts)

    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    # Check cache first
    for i, text in enumerate(texts):
        cached = _cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text[:2000])

    # Batch encode uncached texts
    if uncached_texts:
        try:
            embeddings = embedder.encode(
                uncached_texts,
                normalize_embeddings=normalize,
                batch_size=32,
                show_progress_bar=False
            )
            for idx, emb in zip(uncached_indices, embeddings):
                results[idx] = emb
                _cache.put(texts[idx], emb)
        except Exception as e:
            logger.error("embed_batch failed: %s", e)

    return results


# ────────────── CLASSIFIER ──────────────

def get_classifier():
    """
    Returns HuggingFace zero-shot classification pipeline (BART-MNLI).
    Loaded on first call only. Returns None if unavailable.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading zero-shot classifier: %s", _BART_MODEL)
        _classifier = hf_pipeline(
            "zero-shot-classification",
            model=_BART_MODEL,
            device=-1   # CPU
        )
        logger.info("BART classifier loaded successfully")
    except Exception as e:
        logger.warning("BART classifier unavailable: %s", e)
        _classifier = None

    return _classifier


# ────────────── ATS XGBOOST MODEL ──────────────

def get_ats_model():
    """
    Returns trained XGBoost model for ATS scoring.
    Loaded from model/ats_xgb.pkl on first call.
    Returns None if model file doesn't exist or loading fails.
    """
    global _ats_model
    if _ats_model is not None:
        return _ats_model

    model_path = os.path.join(BASE_DIR, "model", "ats_xgb.pkl")
    if not os.path.exists(model_path):
        logger.warning("XGBoost model not found at %s", model_path)
        return None

    try:
        import pickle
        with open(model_path, "rb") as f:
            _ats_model = pickle.load(f)
        logger.info("XGBoost ATS model loaded from %s", model_path)
    except Exception as e:
        logger.warning("Failed to load XGBoost model: %s", e)
        _ats_model = None

    return _ats_model


# ────────────── CACHE MANAGEMENT ──────────────

def get_cache_stats():
    """Return embedding cache statistics."""
    return _cache.stats


def clear_cache():
    """Flush embedding cache."""
    _cache.clear()
    logger.info("Embedding cache cleared")
