"""
RAG Store — FAISS-backed vector store for Nexus CV.
Stores resume samples, JD samples, and skill taxonomy for
context-aware generation and analysis.

Usage:
    from utils.rag_store import rag_store
    context = rag_store.query("Python developer with 3 years experience", top_k=3)
"""

import os
import json
import logging

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError:
    np = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import services.ml.model_hub as model_hub
except ImportError:
    model_hub = None


class RAGStore:
    """FAISS-backed retrieval store."""

    def __init__(self):
        self._index = None
        self._documents = []
        self._metadata = []
        self._built = False

    def build_index(self, data_dir=None):
        """
        Build FAISS index from JSON files in data_dir.
        Each JSON file should be a list of dicts with 'text' and 'category' keys.
        Falls back to skills taxonomy if RAG data files don't exist.
        """
        if faiss is None or model_hub is None or np is None:
            logger.warning("FAISS or model_hub not available, RAG disabled")
            return False

        if data_dir is None:
            data_dir = os.path.join(BASE_DIR, "data", "rag")

        documents = []
        metadata = []

        # Load RAG data files
        if os.path.isdir(data_dir):
            for filename in os.listdir(data_dir):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        items = json.load(f)
                    for item in items:
                        text = item.get("text", "")
                        if len(text) > 20:
                            documents.append(text)
                            metadata.append({
                                "source": filename,
                                "category": item.get("category", "general"),
                                "role": item.get("role", ""),
                                "type": item.get("type", "document")
                            })
                except Exception as e:
                    logger.warning("Failed to load %s: %s", filename, e)

        # Fallback: load skills taxonomy as documents
        taxonomy_path = os.path.join(BASE_DIR, "data", "skills_taxonomy.json")
        if os.path.exists(taxonomy_path):
            try:
                with open(taxonomy_path, "r", encoding="utf-8") as f:
                    taxonomy = json.load(f)
                for category, skills in taxonomy.items():
                    doc = f"{category}: {', '.join(skills)}"
                    documents.append(doc)
                    metadata.append({
                        "source": "skills_taxonomy",
                        "category": category,
                        "role": "",
                        "type": "taxonomy"
                    })
            except Exception as e:
                logger.warning("Failed to load taxonomy: %s", e)

        if not documents:
            logger.warning("No documents found for RAG index")
            return False

        # Encode all documents
        try:
            embeddings = model_hub.embed_batch(documents)
            valid_embeddings = []
            valid_docs = []
            valid_meta = []

            for emb, doc, meta in zip(embeddings, documents, metadata):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_docs.append(doc)
                    valid_meta.append(meta)

            if not valid_embeddings:
                logger.warning("No valid embeddings produced")
                return False

            # Build FAISS index
            matrix = np.vstack(valid_embeddings).astype("float32")
            dim = matrix.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(matrix)
            self._index.add(matrix)

            self._documents = valid_docs
            self._metadata = valid_meta
            self._built = True

            logger.info("RAG index built: %d documents, %d dimensions", len(valid_docs), dim)
            return True

        except Exception as e:
            logger.error("Failed to build RAG index: %s", e)
            return False

    def query(self, text, top_k=3):
        """
        Retrieve top-k most similar documents.
        Returns list of dicts with 'text', 'score', and metadata.
        """
        if not self._built or self._index is None:
            return []

        try:
            embedding = model_hub.embed_text(text)
            if embedding is None:
                return []

            query_vec = np.array([embedding], dtype="float32")
            faiss.normalize_L2(query_vec)
            scores, indices = self._index.search(query_vec, min(top_k, len(self._documents)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._documents) and score > 0.2:
                    results.append({
                        "text": self._documents[idx],
                        "score": round(float(score), 3),
                        **self._metadata[idx]
                    })

            return results

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            return []

    def get_context_for_role(self, role, resume_text=None, top_k=3):
        """
        Get role-specific context for generation prompts.
        Combines role query with resume text for relevance.
        """
        query_text = f"{role} resume professional experience skills"
        if resume_text:
            query_text += f" {resume_text[:500]}"

        results = self.query(query_text, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for r in results:
            context_parts.append(f"[{r.get('category', 'general')}] {r['text'][:300]}")

        return "\n---\n".join(context_parts)

    def index_size(self):
        """Return number of indexed documents."""
        return len(self._documents) if self._built else 0


# ── Module-level singleton ──
rag_store = RAGStore()

# Auto-build on import if data exists
_rag_dir = os.path.join(BASE_DIR, "data", "rag")
if os.path.isdir(_rag_dir):
    try:
        rag_store.build_index()
    except Exception:
        pass
